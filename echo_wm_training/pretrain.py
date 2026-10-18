"""Self-supervised pretraining of the MobileNetV3 frame encoder on unlabelled Mayo echo cines (inside MCP).

    python -m echo_wm_training.pretrain --manifest /workspace/echo/manifest.parquet --clips /workspace/echo/clips.parquet \
           --out /workspace/echo/runs/ssl_v1 --cache-dir /workspace/echo/cache --device cuda
    python -m echo_wm_training.train ... --init /workspace/echo/runs/ssl_v1

No labels and no outside data: every 2-D cine (any view) of the TRAINING split studies (before splits.tune_start;
Phase 2 patients already excluded by build_manifest). Method: SimCLR (NT-Xent, the same loss as the MRI encoder).
The two views of a sample are two frames of the same cine, a few frames apart, each randomly rotated, zoomed and
intensity-changed, so the encoder learns cardiac anatomy that persists through the heartbeat.

Output (out/): encoder.pt (encoder only: stem, blocks, head_conv), pretrain_card.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from mri_training.pretrain import SimCLR, encoder_state, nt_xent, view

from . import runtime
from .dataset import load_config
from .io import read_clip
from .runtime.model import build_model
from .train import ClipCache


def collect_frames(clips: pd.DataFrame, cache: ClipCache, per_clip: int, rng: np.random.Generator,
                   log: Callable[[str], None]):
    pairs = []
    ok = bad = 0
    for uri in clips.clip_uri:
        fr = cache(uri)
        if fr is None or len(fr) < 8:
            bad += 1
            continue
        ok += 1
        for _ in range(per_clip):
            i = int(rng.integers(0, len(fr)))
            j = int(np.clip(i + rng.integers(-4, 5), 0, len(fr) - 1))
            pairs.append((fr[i].astype(np.float16), fr[j].astype(np.float16)))
    log(f"pretraining frame pairs: {len(pairs)} from {ok} clips ({bad} unreadable)")
    return pairs, ok, bad


def pretrain(m: pd.DataFrame, clips: pd.DataFrame, cfg: dict, out_dir: str, loader=read_clip, device: Optional[str] = None,
             log: Callable[[str], None] = print, cache_dir: Optional[str] = None) -> dict:
    pc, sp = cfg["pretrain"], cfg["splits"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(sp["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tr = m[m.split == "train"]
    if "study_date" in tr and len(tr):
        late = pd.to_datetime(tr.study_date) >= pd.Timestamp(sp["tune_start"])
        if late.any():
            raise AssertionError(f"{int(late.sum())} pretraining studies are on/after tune_start: tuning/test period leak")
    use = clips[clips.study_id.astype(str).isin(tr.study_id.astype(str))]
    pairs, ok, bad = collect_frames(use, ClipCache(loader, cache_dir), int(pc["frames_per_clip"]), rng, log)
    if len(pairs) < 2:
        raise ValueError("not enough frames to pretrain")
    model = SimCLR(build_model(cfg["arch"]), int(pc.get("proj_dim", 128))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(pc["lr"]), weight_decay=float(pc["weight_decay"]))
    bs = min(int(pc["batch_size"]), len(pairs))
    spe = max(1, len(pairs) // bs)
    total = int(pc["epochs"]) * spe
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(pc["lr"]), total_steps=max(1, total))
    hist = []
    for ep in range(1, int(pc["epochs"]) + 1):
        model.train()
        order = rng.permutation(len(pairs))
        tot = 0.0
        for s in range(spe):
            idx = order[s * bs:(s + 1) * bs]
            a = np.stack([view(pairs[i][0].astype(np.float32), rng, pc["augment"]) for i in idx])[:, None]
            b = np.stack([view(pairs[i][1].astype(np.float32), rng, pc["augment"]) for i in idx])[:, None]
            norm = (lambda x: (x - x.mean((2, 3), keepdims=True)) / (x.std((2, 3), keepdims=True) + 1e-6))  # noqa: E731
            loss = nt_xent(model(torch.from_numpy(norm(a)).to(device)), model(torch.from_numpy(norm(b)).to(device)),
                           float(pc["temperature"]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched.last_epoch + 1 < total:
                sched.step()
            tot += float(loss.detach())
        hist.append({"epoch": ep, "loss": tot / spe})
        log(f"pretrain epoch {ep}: NT-Xent {tot / spe:.4f}")
    torch.save(encoder_state(model.backbone), out / "encoder.pt")
    card = {
        "kind": "echo_encoder_selfsupervised", "method": "SimCLR (NT-Xent) on pairs of nearby frames of 2-D echo cines",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "arch": cfg["arch"],
        "runtime_fingerprint": runtime.fingerprint(),
        "encoder_sha256": hashlib.sha256((out / "encoder.pt").read_bytes()).hexdigest(),
        "data": {"source": "Mayo Clinic Platform, training split only", "before": sp["tune_start"], "studies": int(len(tr)),
                 "clips": ok, "unreadable_clips": bad, "frame_pairs": len(pairs)},
        "settings": pc, "history": hist, "uses_outside_data": False,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": device},
    }
    (out / "pretrain_card.json").write_text(json.dumps(card, indent=2, default=float))
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache-dir", default=None)
    a = ap.parse_args()
    rd = (lambda p: pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p))   # noqa: E731
    card = pretrain(rd(a.manifest), rd(a.clips), load_config(a.config), a.out, device=a.device, cache_dir=a.cache_dir)
    print(json.dumps({k: card[k] for k in ("encoder_sha256", "data")}, indent=2))


if __name__ == "__main__":
    main()
