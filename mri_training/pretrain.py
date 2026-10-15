"""Self-supervised pretraining of the MobileNetV3 encoder on unlabelled Mayo cardiac MRI (inside MCP).

    python -m mri_training.pretrain --manifest /workspace/mri/manifest.parquet --out /workspace/mri/runs/ssl_v1 --device cuda
    python -m mri_training.train --manifest ... --init /workspace/mri/runs/ssl_v1 --out /workspace/mri/runs/v1

No labels and no outside data: the encoder learns cardiac MRI from the short-axis LGE and cine slices of
the TRAINING split only (studies before splits.tune_start, Phase 2 patients excluded by build_manifest), so
the tuning and test periods stay unseen. Method: SimCLR (Chen et al., 2020) on 2-D slices. Two random views
of each slice (crop + rescale, rotation, intensity gamma, contrast, blur, noise; no flips, which would swap
the heart's handedness) are pulled together and away from every other slice in the batch (NT-Xent loss).

Output (out/): encoder.pt (plain state_dict of the encoder only: stem, blocks, head_conv), pretrain_card.json
(runtime fingerprint, data period and counts, settings, loss history). train.py --init loads the encoder
strictly and starts the decoder and study heads fresh.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import affine_transform, gaussian_filter

from . import runtime
from .dataset import load_config
from .runtime.model import build_model
from .runtime.preprocess import MRIInputError, model_input, prepare

ENCODER_PREFIXES = ("stem.", "blocks.", "head_conv.")


def encoder_state(model) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k.startswith(ENCODER_PREFIXES)}


class SimCLR(nn.Module):
    def __init__(self, backbone, dim: int = 128):
        super().__init__()
        self.backbone = backbone
        last = backbone.head_conv[0].out_channels
        self.proj = nn.Sequential(nn.Linear(last, 512), nn.ReLU(inplace=True), nn.Linear(512, dim))

    def forward(self, x):
        z, _ = self.backbone.encode(x)
        return F.normalize(self.proj(F.adaptive_avg_pool2d(z, 1).flatten(1)), dim=1)


def nt_xent(a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
    z = torch.cat([a, b])
    sim = z @ z.t() / tau
    n = a.shape[0]
    sim.fill_diagonal_(-1e9)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def view(img: np.ndarray, rng: np.random.Generator, aug: dict) -> np.ndarray:
    """One random view of a prepared slice (192 x 192 in [0, 1])."""
    ang = np.deg2rad(rng.uniform(-1, 1) * aug.get("rotate_deg", 20))
    sc = rng.uniform(*aug.get("scale", (0.6, 1.0)))                  # crop = zoom in
    m = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * sc
    c = np.array(img.shape) / 2 - 0.5
    shift = rng.uniform(-1, 1, 2) * (1 - sc) * img.shape[0] / 2
    out = affine_transform(img, m, c - m @ c + shift, order=1, mode="constant")
    out = np.clip(out, 0, 1) ** rng.uniform(*aug.get("gamma", (0.7, 1.4)))
    out = (out - out.mean()) * rng.uniform(*aug.get("contrast", (0.8, 1.2))) + out.mean()
    if rng.random() < aug.get("blur_p", 0.3):
        out = gaussian_filter(out, rng.uniform(0.3, 1.2))
    out = out + rng.normal(0, aug.get("noise", 0.03), out.shape)
    return np.clip(out, 0, 1).astype(np.float32)


def collect_slices(m: pd.DataFrame, cfg: dict, loader, log: Callable[[str], None], cine_phases: int = 2,
                   seed: int = 0) -> Dict[str, object]:
    """Prepared slices (+ sequence kind) of every readable training study; cine: a few phases per study."""
    rng = np.random.default_rng(seed)
    imgs: List[np.ndarray] = []
    kinds: List[str] = []
    n_ok = n_bad = 0
    for _, r in m.iterrows():
        try:
            ex = loader(r, cfg)
            pl = prepare(ex.lge)
            imgs += list(pl.images)
            kinds += ["lge"] * len(pl.images)
            if ex.cine is not None:
                pc = prepare(ex.cine)
                for ph in rng.choice(pc.images.shape[0], min(cine_phases, pc.images.shape[0]), replace=False):
                    imgs += list(pc.images[ph])
                    kinds += ["cine"] * pc.images.shape[1]
            n_ok += 1
        except (MRIInputError, ValueError, OSError):
            n_bad += 1
    log(f"pretraining slices: {len(imgs)} from {n_ok} studies ({n_bad} unreadable)")
    return {"images": np.stack(imgs).astype(np.float16) if imgs else np.zeros((0, 192, 192), np.float16),
            "kinds": np.asarray(kinds), "studies": n_ok, "unreadable": n_bad}


def pretrain(m: pd.DataFrame, cfg: dict, out_dir: str, loader=None, device: Optional[str] = None,
             log: Callable[[str], None] = print, slices: Optional[Dict] = None) -> dict:
    from .train import load_example
    pc = cfg["pretrain"]
    sp = cfg["splits"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(sp["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mayo = m[(m.get("source", pd.Series("mayo", index=m.index)).fillna("mayo") == "mayo") & (m.split == "train")]
    if "study_date" in mayo and len(mayo):
        late = pd.to_datetime(mayo.study_date) >= pd.Timestamp(sp["tune_start"])
        if late.any():
            raise AssertionError(f"{int(late.sum())} pretraining studies are on/after tune_start: tuning/test period leak")
    data = slices or collect_slices(mayo, cfg, loader or load_example, log, int(pc.get("cine_phases", 2)), seed)
    X, K = data["images"], data["kinds"]
    if len(X) < 2:
        raise ValueError("not enough slices to pretrain")
    model = SimCLR(build_model({**cfg["arch"], "study_heads": cfg["study_heads"]}), int(pc.get("proj_dim", 128))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(pc["lr"]), weight_decay=float(pc["weight_decay"]))
    bs = min(int(pc["batch_size"]), len(X))
    steps_per_epoch = max(1, len(X) // bs)
    total = int(pc["epochs"]) * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(pc["lr"]), total_steps=max(1, total))
    hist = []
    for ep in range(1, int(pc["epochs"]) + 1):
        model.train()
        order = rng.permutation(len(X))
        tot = 0.0
        for s in range(steps_per_epoch):
            idx = order[s * bs:(s + 1) * bs]
            v1 = np.stack([view(X[i].astype(np.float32), rng, pc["augment"]) for i in idx])
            v2 = np.stack([view(X[i].astype(np.float32), rng, pc["augment"]) for i in idx])
            kinds = K[idx]
            x1 = np.concatenate([model_input(v1[kinds == k], k) for k in ("lge", "cine") if (kinds == k).any()])
            x2 = np.concatenate([model_input(v2[kinds == k], k) for k in ("lge", "cine") if (kinds == k).any()])
            loss = nt_xent(model(torch.from_numpy(x1).to(device)), model(torch.from_numpy(x2).to(device)), float(pc["temperature"]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched.last_epoch + 1 < total:
                sched.step()
            tot += float(loss.detach())
        hist.append({"epoch": ep, "loss": tot / steps_per_epoch})
        log(f"pretrain epoch {ep}: NT-Xent {tot / steps_per_epoch:.4f}")
    state = encoder_state(model.backbone)
    torch.save(state, out / "encoder.pt")
    card = {
        "kind": "mri_encoder_selfsupervised", "method": "SimCLR (NT-Xent) on 2-D short-axis LGE + cine slices",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "arch": cfg["arch"], "runtime_fingerprint": runtime.fingerprint(),
        "encoder_sha256": hashlib.sha256((out / "encoder.pt").read_bytes()).hexdigest(),
        "data": {"source": "Mayo Clinic Platform, training split only", "before": sp["tune_start"],
                 "studies": int(data.get("studies", 0)), "unreadable": int(data.get("unreadable", 0)),
                 "slices": int(len(X)), "lge_slices": int((K == "lge").sum()), "cine_slices": int((K == "cine").sum())},
        "settings": pc, "history": hist, "uses_outside_data": False,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": device},
    }
    (out / "pretrain_card.json").write_text(json.dumps(card, indent=2, default=float))
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    card = pretrain(pd.read_parquet(a.manifest), cfg, a.out, device=a.device)
    print(json.dumps({k: card[k] for k in ("encoder_sha256", "data")}, indent=2))


if __name__ == "__main__":
    main()
