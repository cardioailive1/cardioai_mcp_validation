"""Train, threshold and evaluate the MobileNetV3-1D ECG model; write the export bundle.

    python -m ecg_training.train --manifest ecg_manifest.parquet --out runs/ecg_v1 [--config ecg_training/config.yaml]

Bundle (out/): weights.pt (plain state_dict), thresholds.json, model_card.json, metrics_*.csv.
Only weights.pt + thresholds.json + model_card.json leave the workspace (MobileNetV3 export
approved); manifests and per-ECG predictions stay inside MCP.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import runtime
from .dataset import load_config, summary
from .io import read_any
from .runtime.model import build_model, n_params
from .runtime.preprocess import PREPROCESS_VERSION, ECGInputError, preprocess

Loader = Callable[[str], tuple]


class ECGDataset(Dataset):
    def __init__(self, m: pd.DataFrame, heads: List[str], loader: Loader = read_any, augment: Optional[dict] = None, seed: int = 0):
        self.m = m.reset_index(drop=True)
        self.heads, self.loader, self.aug = heads, loader, augment
        self.y = self.m[heads].to_numpy(dtype=np.float32)              # NaN = unlabelled for that head
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.m)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        a, r = self.aug, self.rng
        x = x * r.uniform(*a.get("amplitude", (1, 1)))
        sh = int(r.uniform(-1, 1) * a.get("shift_s", 0) * 500)
        if sh:
            x = np.roll(x, sh, axis=1)
        if a.get("noise_mv"):
            x = x + r.normal(0, a["noise_mv"], x.shape)
        if r.random() < a.get("lead_dropout_p", 0):
            x[r.integers(0, 12)] = 0.0
        return x.astype(np.float32)

    def __getitem__(self, i):
        sig, fs, leads, units = self.loader(self.m.at[i, "waveform_uri"])
        try:
            x = preprocess(sig, fs, leads, units)
            ok = 1.0
        except ECGInputError:
            x, ok = np.zeros((12, 5000), np.float32), 0.0               # unusable recording: masked out
        if self.aug is not None and ok:
            x = self._augment(x)
        y = self.y[i]
        mask = (~np.isnan(y)).astype(np.float32) * ok
        return torch.from_numpy(x), torch.from_numpy(np.nan_to_num(y)), torch.from_numpy(mask)


def masked_bce(logits, y, mask, pos_weight):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def predict(model, ds: ECGDataset, device, batch_size=256, num_workers=0) -> np.ndarray:
    model.eval()
    out = []
    for x, _, _ in DataLoader(ds, batch_size=batch_size, num_workers=num_workers):
        out.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, len(ds.heads)))


def _auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_threshold(y, p, target_sens: float) -> tuple[float, str]:
    """Highest threshold whose sensitivity >= target (tuning split only)."""
    y, p = np.asarray(y), np.asarray(p)
    pos = np.sort(p[y == 1])
    if len(pos) < 5:
        return 0.5, f"only {len(pos)} tuning positives: default 0.5 (re-fit when more data)"
    k = int(math.floor((1 - target_sens) * len(pos)))
    return float(pos[k]), f"sensitivity >= {target_sens} on {len(pos)} tuning positives"


def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def evaluate(m: pd.DataFrame, probs: np.ndarray, heads: List[str], thresholds: Dict[str, float],
             n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Per head: AUROC with patient-cluster bootstrap CI; sens/spec/PPV/NPV at the locked threshold (Wilson CI)."""
    rows, rng = [], np.random.default_rng(seed)
    pids = m.person_id.to_numpy()
    for j, h in enumerate(heads):
        y = m[h].to_numpy()
        lab = ~np.isnan(y)
        yy, pp, pid = y[lab], probs[lab, j], pids[lab]
        auc = _auc(yy, pp)
        boots = []
        if not math.isnan(auc) and n_boot:
            uniq = np.unique(pid)
            idx = {u: np.where(pid == u)[0] for u in uniq}
            for _ in range(n_boot):
                take = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))])
                a = _auc(yy[take], pp[take])
                if not math.isnan(a):
                    boots.append(a)
        pred = pp >= thresholds[h]
        tp, fn = int(((yy == 1) & pred).sum()), int(((yy == 1) & ~pred).sum())
        tn, fp = int(((yy == 0) & ~pred).sum()), int(((yy == 0) & pred).sum())
        r = {"head": h, "n": int(lab.sum()), "positives": int((yy == 1).sum()), "threshold": thresholds[h],
             "auroc": auc, "auroc_lo": float(np.percentile(boots, 2.5)) if boots else float("nan"),
             "auroc_hi": float(np.percentile(boots, 97.5)) if boots else float("nan")}
        for name, k, n in (("sensitivity", tp, tp + fn), ("specificity", tn, tn + fp), ("ppv", tp, tp + fp), ("npv", tn, tn + fn)):
            lo, hi = _wilson(k, n)
            r.update({name: k / n if n else float("nan"), f"{name}_lo": lo, f"{name}_hi": hi})
        rows.append(r)
    return pd.DataFrame(rows)


def manifest_hash(m: pd.DataFrame, heads: List[str]) -> str:
    cols = ["ecg_id", "person_id", "split"] + heads
    blob = m[cols].sort_values(["split", "ecg_id"]).to_csv(index=False).encode()
    return hashlib.sha256(blob).hexdigest()


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def train(m: pd.DataFrame, cfg: dict, out_dir: str, loader: Loader = read_any, device: Optional[str] = None,
          log: Callable[[str], None] = print) -> dict:
    heads, tc = cfg["heads"], cfg["train"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["splits"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_no_waveform = int(m.waveform_uri.isna().sum()) if "waveform_uri" in m else 0
    if n_no_waveform:                       # Base64 rows that extract_waveforms could not decode
        log(f"dropping {n_no_waveform} ECGs without a decoded waveform")
        m = m[m.waveform_uri.notna()]
    tr, tu = m[m.split == "train"], m[m.split == "tune"]
    ds_tr = ECGDataset(tr, heads, loader, augment=tc.get("augment"), seed=seed)
    ds_tu = ECGDataset(tu, heads, loader)
    nw = int(tc.get("num_workers", 0))
    dl = DataLoader(ds_tr, batch_size=int(tc["batch_size"]), shuffle=True, num_workers=nw, drop_last=len(ds_tr) > int(tc["batch_size"]))
    model = build_model({**cfg["arch"], "heads": heads}).to(device)
    prev = np.nan_to_num(np.nanmean(ds_tr.y, axis=0), nan=0.5).clip(1e-3, 1 - 1e-3)
    pos_weight = torch.tensor(np.clip((1 - prev) / prev, 1.0, 50.0), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(tc["lr"]), total_steps=max(1, int(tc["epochs"]) * max(1, len(dl))))
    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, int(tc["epochs"]) + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y, mask in dl:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            loss = masked_bce(model(x), y, mask, pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, nb = tot + loss.item(), nb + 1
        p_tu = predict(model, ds_tu, device, num_workers=nw)
        aucs = {h: _auc(tu[h].to_numpy()[~np.isnan(tu[h].to_numpy())], p_tu[~np.isnan(tu[h].to_numpy()), j]) for j, h in enumerate(heads)}
        score = float(np.nanmean(list(aucs.values()))) if any(not math.isnan(a) for a in aucs.values()) else -float(tot / max(nb, 1))
        history.append({"epoch": ep, "train_loss": tot / max(nb, 1), "tune_auroc": aucs, "score": score, "seconds": time.time() - t0})
        log(f"epoch {ep}: loss {tot / max(nb, 1):.4f}  tune AUROC {', '.join(f'{h} {a:.3f}' for h, a in aucs.items())}")
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(tc["patience"]):
                log(f"early stop at epoch {ep}")
                break
    model.load_state_dict(best_state, strict=True)
    torch.save(best_state, out / "weights.pt")                       # plain state_dict only

    p_tu = predict(model, ds_tu, device, num_workers=nw)
    thresholds, how = {}, {}
    for j, h in enumerate(heads):
        y = tu[h].to_numpy()
        lab = ~np.isnan(y)
        thresholds[h], how[h] = fit_threshold(y[lab], p_tu[lab, j], float(cfg["thresholds"][h]["target_sensitivity"]))
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    metrics = {"tune": evaluate(tu, p_tu, heads, thresholds, n_boot=0)}
    for split in ("test_time", "test_site"):
        part = m[m.split == split]
        if len(part):
            metrics[split] = evaluate(part, predict(model, ECGDataset(part, heads, loader), device, num_workers=nw), heads, thresholds)
    for k, v in metrics.items():
        v.to_csv(out / f"metrics_{k}.csv", index=False)

    card = {
        "model_name": cfg["model_name"], "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "architecture": "MobileNetV3-1D (Small layout)", "arch": cfg["arch"], "heads": heads,
        "parameters": n_params(model), "runtime_fingerprint": runtime.fingerprint(), "preprocess_version": PREPROCESS_VERSION,
        "input": {"leads": 12, "fs_hz": 500, "seconds": 10, "units": "mV"},
        "weights_sha256": hashlib.sha256((out / "weights.pt").read_bytes()).hexdigest(),
        "thresholds": thresholds, "threshold_rule": how, "label_definitions": cfg["labels"], "splits": cfg["splits"],
        "manifest_sha256": manifest_hash(m, heads), "data_summary": summary(m, heads).to_dict("records"),
        "ecgs_without_waveform": n_no_waveform,
        "training": {k: v for k, v in tc.items()}, "history": history, "best_tune_score": best,
        "metrics": {k: v.to_dict("records") for k, v in metrics.items()},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": device},
        "code_commit": _git_commit(),
        "intended_use": "Flag acute MI, STEMI and atrial fibrillation/flutter on a resting 10 s 12-lead ECG for clinician over-read.",
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2, default=float))
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    m = pd.read_parquet(a.manifest)
    card = train(m, cfg, a.out, device=a.device)
    print(json.dumps({k: card[k] for k in ("parameters", "thresholds", "runtime_fingerprint", "weights_sha256")}, indent=2))


if __name__ == "__main__":
    main()
