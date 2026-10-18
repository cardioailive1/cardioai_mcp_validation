"""Train, threshold and evaluate the MobileNetV3-WM echo wall-motion model; write the export bundle.

    python -m echo_wm_training.train --manifest /workspace/echo/manifest.parquet --clips /workspace/echo/clips_views.parquet \
           --init /workspace/echo/runs/ssl_v1 --out /workspace/echo/runs/wm_v1 --cache-dir /workspace/echo/cache --device cuda

Stage V  view head: clips with a human view label or a DICOM view tag (A4C / A2C / A3C / other).
Stage W  whole model, one study per step: one clip per available apical view (random clip, random 1.6 s window)
         -> 17 segment scores (cross-entropy on report-labelled segments shown by those views), LVEF (report or
         measurement), global hypokinesis, and the views again. Gradients accumulated over `accumulate_studies`.
Early stopping on the tuning split (mean AUROC of the territory / any-abnormality / global heads).
Thresholds: fitted on the tuning split (2020) for the target sensitivities in config.yaml, then locked.

Manifest (dataset.build_manifest): study_id, person_id, split, seg_1..seg_17 (0..3 / NaN), lvef, global_hypokinesis,
  wma_any, terr_lad, terr_rca, terr_lcx, wmsi.
Clip table (dataset.build_clip_table + views.assign_views): study_id, clip_uri, view (A4C / A2C / A3C / other / None),
  view_source (label / tag / model).

Bundle (out/): weights.pt (plain state_dict), thresholds.json, model_card.json, metrics_*.csv. Only weights.pt +
thresholds.json + model_card.json leave the workspace (MobileNetV3 export approved).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import affine_transform

from ecg_training.train import _auc, _git_commit, evaluate, fit_threshold

from . import runtime
from .dataset import load_config, summary
from .io import read_clip
from .labels import SEG_COLUMNS
from .runtime.model import build_model, n_params, view_index
from .runtime.preprocess import CLIP_FRAMES, PREPROCESS_VERSION, EchoInputError, Video, prepare, standardise
from .runtime.segments import N_SCORE, TERRITORIES, VIEW_CLASSES, VIEWS, summarise

Loader = Callable[[str], Video]
HEADS_DEFAULT = ["wma_any", "terr_lad", "terr_rca", "terr_lcx", "global_hypokinesis"]


# --------------------------------------------------------------------------- clip cache
class ClipCache:
    """clip_uri -> preprocessed frames (T, 112, 112) at 20 fps, kept in memory and optionally on disk (.npy)."""

    def __init__(self, loader: Loader = read_clip, cache_dir: Optional[str] = None):
        self.loader, self.mem, self.failed = loader, {}, {}
        self.dir = Path(cache_dir) if cache_dir else None
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, uri: str) -> Optional[np.ndarray]:
        if uri in self.mem:
            return self.mem[uri]
        if uri in self.failed:
            return None
        path = self.dir / (hashlib.sha256(uri.encode()).hexdigest()[:32] + ".npy") if self.dir else None
        if path is not None and path.exists():
            fr = np.load(path).astype(np.float32) / 255.0
        else:
            try:
                fr = prepare(self.loader(uri)).frames
            except (EchoInputError, Exception) as e:  # noqa: BLE001  (one unreadable clip must not stop training)
                self.failed[uri] = str(e)[:200]
                return None
            if path is not None:
                np.save(path, (fr * 255).round().astype(np.uint8))
        if self.dir is None or len(self.mem) < 2000:
            self.mem[uri] = fr
        return fr


def window(fr: np.ndarray, rng: Optional[np.random.Generator]) -> np.ndarray:
    last = len(fr) - CLIP_FRAMES
    s = 0 if rng is None or last <= 0 else int(rng.integers(0, last + 1))
    return fr[s:s + CLIP_FRAMES]


def augment(clip: np.ndarray, a: dict, rng: np.random.Generator) -> np.ndarray:
    """Same small rotation / scale for every frame, intensity gamma and noise. No flips (they swap walls)."""
    ang = math.radians(rng.uniform(-a["rotate_deg"], a["rotate_deg"]))
    sc = rng.uniform(*a["scale"])
    c, s = math.cos(ang) / sc, math.sin(ang) / sc
    mat = np.array([[c, -s], [s, c]])
    ctr = np.array(clip.shape[1:]) / 2
    off = ctr - mat @ ctr
    out = np.stack([affine_transform(f, mat, off, order=1, mode="constant") for f in clip])
    out = np.clip(out, 0, 1) ** rng.uniform(*a["gamma"])
    return out + rng.normal(0, a["noise"], out.shape)


# --------------------------------------------------------------------------- study inputs
def study_clips(clips: pd.DataFrame, study_id: str, rng: Optional[np.random.Generator], per_view: int = 1) -> List[pd.Series]:
    g = clips[(clips.study_id == study_id) & clips["view"].isin(VIEWS)]
    picked = []
    for v in VIEWS:
        gv = g[g["view"] == v]
        if len(gv):
            idx = rng.choice(len(gv), min(per_view, len(gv)), replace=False) if rng is not None else range(min(per_view, len(gv)))
            picked += [gv.iloc[int(i)] for i in idx]
    return picked


def batch_for(rows: Sequence[pd.Series], cache: ClipCache, rng: Optional[np.random.Generator], aug: Optional[dict]):
    xs, views = [], []
    for r in rows:
        fr = cache(r.clip_uri)
        if fr is None or len(fr) < CLIP_FRAMES:
            continue
        c = window(fr, rng)
        if aug is not None and rng is not None:
            c = augment(c, aug, rng)
        xs.append(standardise(c))
        views.append(r["view"])
    if not xs:
        return None, None
    return torch.from_numpy(np.stack(xs)), views


def seg_targets(row: pd.Series) -> torch.Tensor:
    y = row[SEG_COLUMNS].to_numpy(dtype=float)
    return torch.tensor(np.where(np.isnan(y), -1, y), dtype=torch.long)


# --------------------------------------------------------------------------- stages
def train_views(model, clips: pd.DataFrame, cache: ClipCache, cfg: dict, device: str, log, rng) -> List[dict]:
    tc = cfg["train"]
    lab = clips[clips.view_source.isin(["label", "tag", "sr", "cluster"]) & clips["view"].isin(VIEW_CLASSES)].reset_index(drop=True)
    if not len(lab) or not int(tc["view_epochs"]):
        return []
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]))
    hist, bs = [], 8
    for ep in range(int(tc["view_epochs"])):
        model.train()
        order, tot, n = rng.permutation(len(lab)), 0.0, 0
        for i in range(0, len(order), bs):
            rows = [lab.iloc[int(j)] for j in order[i:i + bs]]
            x, views = batch_for(rows, cache, rng, tc["augment"])
            if x is None:
                continue
            y = torch.tensor([VIEW_CLASSES.index(v) for v in views], device=device)
            loss = F.cross_entropy(model(x.to(device))["view"], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, n = tot + loss.item() * len(y), n + len(y)
        hist.append({"stage": "V", "epoch": ep + 1, "loss": tot / max(n, 1)})
        log(f"view epoch {ep + 1}: loss {tot / max(n, 1):.4f}")
    return hist


@torch.no_grad()
def predict_views(model, clips: pd.DataFrame, cache: ClipCache, device: str) -> pd.DataFrame:
    model.eval()
    rows = []
    for _, r in clips.iterrows():
        fr = cache(r.clip_uri)
        if fr is None or len(fr) < CLIP_FRAMES:
            continue
        p = model(torch.from_numpy(standardise(window(fr, None))[None]).to(device))["view"].softmax(-1)[0].cpu().numpy()
        rows.append({"clip_uri": r.clip_uri, "view": VIEW_CLASSES[int(p.argmax())], "view_p": float(p.max())})
    return pd.DataFrame(rows, columns=["clip_uri", "view", "view_p"])


def study_loss(model, out, views, row: pd.Series, cfg: dict, cw: torch.Tensor, device: str):
    lw = cfg["train"]["loss_weights"]
    st = model.study_outputs(out, view_index(views).to(device))
    y = seg_targets(row).to(device)
    use = st["seen"] & (y >= 0)
    parts = {}
    if use.any():
        parts["segments"] = F.cross_entropy(st["seg"][use], y[use], weight=cw)
    parts["view"] = F.cross_entropy(out["view"], torch.tensor([VIEW_CLASSES.index(v) for v in views], device=device))
    if not math.isnan(float(row.get("lvef", float("nan")))):
        parts["lvef"] = F.smooth_l1_loss(st["lvef"], torch.tensor(float(row["lvef"]), device=device))
    gh = float(row.get("global_hypokinesis", float("nan")))
    if not math.isnan(gh):
        parts["global_hypokinesis"] = F.binary_cross_entropy_with_logits(st["ghypo"], torch.tensor(gh, device=device))
    return sum(float(lw[k]) * v for k, v in parts.items()), {k: v.item() for k, v in parts.items()}


@torch.no_grad()
def study_predict(model, m: pd.DataFrame, idx: Sequence[int], clips: pd.DataFrame, cache: ClipCache, device: str,
                  all_clips: bool = True) -> pd.DataFrame:
    """Deterministic study-level outputs (every clip of each apical view, first window)."""
    model.eval()
    rows = []
    for i in idx:
        r = m.loc[i]
        picked = study_clips(clips, r.study_id, None, per_view=99 if all_clips else 1)
        x, views = batch_for(picked, cache, None, None)
        if x is None:
            rows.append({"index": i, "error": "no usable apical clip"})
            continue
        out = model(x.to(device))
        st = model.study_outputs(out, view_index(views).to(device))
        prob = st["seg"].softmax(-1).cpu().numpy()
        seen = st["seen"].cpu().numpy()
        s = summarise(prob, seen)
        p_abn = 1 - prob[seen, 0]
        rec = {"index": i, "error": None, "p_wma_any": float(p_abn.max()),
               "p_global_hypokinesis": float(torch.sigmoid(st["ghypo"])), "lvef_pct": float(st["lvef"]), "wmsi": s["wmsi"],
               "views": ",".join(sorted(set(views)))}
        for t in TERRITORIES:
            pa = s["territories"][t]["p_abnormal"]
            rec[f"p_terr_{t.lower()}"] = float("nan") if pa is None else pa
        for k in range(17):
            rec[f"pred_seg_{k + 1}"] = int(prob[k].argmax()) if seen[k] else -1
        rows.append(rec)
    return pd.DataFrame(rows).set_index("index")


def _tune_score(m: pd.DataFrame, res: pd.DataFrame, heads: List[str]) -> Dict[str, float]:
    ok = res[res.error.isna()]
    out = {}
    for h in heads:
        y, p = m.loc[ok.index, h].to_numpy(dtype=float), ok[f"p_{h}"].to_numpy(dtype=float)
        lab = ~np.isnan(y) & ~np.isnan(p)
        out[f"auroc_{h}"] = _auc(y[lab], p[lab])
    return out


def train_wm(model, m, clips, cache, cfg, device, log, rng, heads) -> List[dict]:
    tc = cfg["train"]
    tr = [int(i) for i in m.index[m.split == "train"]]
    tu = [int(i) for i in m.index[m.split == "tune"]]
    acc = max(1, int(tc["accumulate_studies"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]))
    steps = max(1, int(tc["epochs"]) * math.ceil(len(tr) / acc))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(tc["lr"]), total_steps=steps, pct_start=0.1)
    cw = torch.tensor(tc["class_weights"], dtype=torch.float32, device=device)[:N_SCORE]
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(int(tc["epochs"])):
        model.train()
        order = rng.permutation(tr)
        opt.zero_grad()
        tot, n, k = {}, 0, 0
        for j, i in enumerate(order):
            row = m.loc[int(i)]
            x, views = batch_for(study_clips(clips, row.study_id, rng, int(tc["clips_per_view"])), cache, rng, tc["augment"])
            if x is None:
                continue
            loss, parts = study_loss(model, model(x.to(device)), views, row, cfg, cw, device)
            (loss / acc).backward()
            k += 1
            for a, b in parts.items():
                tot[a] = tot.get(a, 0.0) + b
            n += 1
            if k % acc == 0 or j == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                opt.zero_grad()
                if sched.last_epoch + 1 < steps:
                    sched.step()
        s = _tune_score(m, study_predict(model, m, tu, clips, cache, device, all_clips=False), heads) if tu else {}
        vals = [v for v in s.values() if not math.isnan(v)]
        score = float(np.mean(vals)) if vals else -float(sum(tot.values()) / max(n, 1))
        hist.append({"stage": "W", "epoch": ep + 1, **{f"loss_{a}": b / max(n, 1) for a, b in tot.items()}, **s, "score": score})
        log(f"epoch {ep + 1}: " + ", ".join(f"{a} {b / max(n, 1):.3f}" for a, b in tot.items()) + f"; tune {score:.3f}")
        if score > best:
            best, bad = score, 0
            best_state = {a: b.detach().cpu().clone() for a, b in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(tc["patience"]):
                log("early stop")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


# --------------------------------------------------------------------------- metrics
def _kappa_weighted(a: np.ndarray, b: np.ndarray, k: int = N_SCORE) -> float:
    if len(a) == 0:
        return float("nan")
    o = np.zeros((k, k))
    for x, y in zip(a, b):
        o[int(x), int(y)] += 1
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], float) / (k - 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    return float(1 - (w * o).sum() / (w * e).sum()) if (w * e).sum() else float("nan")


def quantitative(m: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    ok = res[res.error.isna()]
    part = m.loc[ok.index]
    rows = []
    ys, ps = [], []
    for k in range(1, 18):
        y = part[f"seg_{k}"].to_numpy(dtype=float)
        p = ok[f"pred_seg_{k}"].to_numpy(dtype=float)
        lab = ~np.isnan(y) & (p >= 0)
        ys.append(y[lab])
        ps.append(p[lab])
    y, p = np.concatenate(ys), np.concatenate(ps)
    rows.append({"measure": "segment score (17 segments pooled)", "n": len(y),
                 "accuracy": float((y == p).mean()) if len(y) else float("nan"),
                 "normal_vs_abnormal_accuracy": float(((y > 0) == (p > 0)).mean()) if len(y) else float("nan"),
                 "weighted_kappa": _kappa_weighted(y, p)})
    for pc, rc, name in (("wmsi", "wmsi", "WMSI vs report"), ("lvef_pct", "lvef", "LVEF vs report / measurement (%)")):
        a, b = pd.to_numeric(ok[pc], errors="coerce").to_numpy(float), part[rc].to_numpy(float)
        lab = ~np.isnan(a) & ~np.isnan(b)
        if lab.sum():
            d = a[lab] - b[lab]
            rows.append({"measure": name, "n": int(lab.sum()), "mae": float(np.abs(d).mean()), "bias": float(d.mean()),
                         "pearson_r": float(np.corrcoef(a[lab], b[lab])[0, 1]) if lab.sum() > 2 else float("nan")})
    a, b = pd.to_numeric(ok["lvef_pct"], errors="coerce").to_numpy(float), part["lvef"].to_numpy(float)
    lab = ~np.isnan(a) & ~np.isnan(b)
    if lab.sum():
        cat = lambda v: np.where(v <= 40, 0, np.where(v < 50, 1, 2))   # noqa: E731  HFrEF / HFmrEF / normal
        ca, cb = cat(a[lab]), cat(b[lab])
        rows.append({"measure": "EF category (<=40 / 41-49 / >=50) agreement", "n": int(lab.sum()),
                     "accuracy": float((ca == cb).mean()), "weighted_kappa": _kappa_weighted(cb, ca, 3)})
    return pd.DataFrame(rows)


def manifest_hash(m: pd.DataFrame, cols: List[str]) -> str:
    keep = [c for c in ["study_id", "person_id", "split"] + cols if c in m]
    return hashlib.sha256(m[keep].sort_values(["split", "study_id"]).to_csv(index=False).encode()).hexdigest()


def load_init(model, init: str, log) -> dict:
    from mri_training.pretrain import ENCODER_PREFIXES
    p = Path(init)
    if p.is_dir():
        p = p / "encoder.pt" if (p / "encoder.pt").exists() else p / "weights.pt"
    state = torch.load(p, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise TypeError("--init must be a plain state_dict (encoder.pt from pretrain.py or weights.pt from train.py)")
    encoder_only = all(k.startswith(ENCODER_PREFIXES) for k in state)
    if encoder_only:
        want = {k for k in model.state_dict() if k.startswith(ENCODER_PREFIXES)}
        if set(state) != want:
            raise RuntimeError(f"encoder weights do not match this architecture ({len(set(state) ^ want)} keys differ)")
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=True)
    info = {"weights_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "path": str(p),
            "scope": "encoder" if encoder_only else "whole model"}
    card = p.parent / "pretrain_card.json"
    if card.exists():
        c = json.loads(card.read_text())
        info.update({k: c.get(k) for k in ("method", "data", "uses_outside_data", "runtime_fingerprint") if k in c})
    log(f"initialised from {p} ({info['weights_sha256'][:12]})")
    return info


# --------------------------------------------------------------------------- main entry
def train(m: pd.DataFrame, clips: pd.DataFrame, cfg: dict, out_dir: str, loader: Loader = read_clip,
          device: Optional[str] = None, log: Callable[[str], None] = print, cache_dir: Optional[str] = None,
          n_boot: int = 1000, init: Optional[str] = None) -> dict:
    heads = list(cfg["labels"].get("heads", HEADS_DEFAULT))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["splits"]["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not cfg.get("use_public_datasets", False) and "source" in m and (m.source.fillna("mayo") != "mayo").any():
        raise ValueError("config use_public_datasets is false (Mayo-only training): remove the public rows")
    m = m.reset_index(drop=True)
    m["study_id"] = m.study_id.astype(str)
    clips = clips.copy()
    clips["study_id"] = clips.study_id.astype(str)
    for col in ["view", "view_source"]:
        if col not in clips:
            clips[col] = None
    cache = ClipCache(loader, cache_dir)
    model = build_model(cfg["arch"])
    init_info = load_init(model, init, log) if init else None
    model = model.to(device)
    hist = train_views(model, clips, cache, cfg, device, log, rng)
    todo = clips[clips["view"].isna()]
    if len(todo):                                     # unlabelled, untagged clips: confident view-head predictions only
        pred = predict_views(model, todo, cache, device).set_index("clip_uri")
        fill = clips["view"].isna() & clips.clip_uri.map(pred.view_p).fillna(0).ge(float(cfg["views"]["min_view_prob"]))
        clips.loc[fill, "view"] = clips.loc[fill, "clip_uri"].map(pred["view"])
        clips.loc[fill, "view_source"] = "model"
    view_counts = clips.groupby(["view", "view_source"], dropna=False).size().reset_index(name="clips").to_dict("records")
    hist += train_wm(model, m, clips, cache, cfg, device, log, rng, heads)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, out / "weights.pt")

    tu = [int(i) for i in m.index[m.split == "tune"]]
    res_tu = study_predict(model, m, tu, clips, cache, device)
    thresholds, how = {}, {}
    ok = res_tu[res_tu.error.isna()] if len(res_tu) else res_tu
    for h in heads:
        y = m.loc[ok.index, h].to_numpy(dtype=float) if len(ok) else np.array([])
        p = ok[f"p_{h}"].to_numpy(dtype=float) if len(ok) else np.array([])
        lab = ~np.isnan(y) & ~np.isnan(p)
        thresholds[h], how[h] = fit_threshold(y[lab], p[lab], float(cfg["thresholds"][h]["target_sensitivity"]))
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    metrics: Dict[str, pd.DataFrame] = {}
    splits = {"tune": res_tu}
    for split in ("test_time", "test_site"):
        idx = [int(i) for i in m.index[m.split == split]]
        if idx:
            splits[split] = study_predict(model, m, idx, clips, cache, device)
    for split, res in splits.items():
        okr = res[res.error.isna()] if len(res) else res
        if not len(okr):
            continue
        part = m.loc[okr.index]
        P = np.column_stack([okr[f"p_{h}"].to_numpy(dtype=float) for h in heads])
        P = np.nan_to_num(P, nan=-1.0)
        mm = part.copy()
        for j, h in enumerate(heads):
            mm.loc[P[:, j] < 0, h] = np.nan
        metrics[f"{split}_heads"] = evaluate(mm, P, heads, thresholds, n_boot=0 if split == "tune" else n_boot)
        metrics[f"{split}_quantitative"] = quantitative(m, res)
    for k, v in metrics.items():
        v.to_csv(out / f"metrics_{k}.csv", index=False)

    card = {
        "model_name": cfg["model_name"], "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "architecture": "MobileNetV3-Small frame encoder (2-D) + temporal motion block + view-specific segment slots (17 AHA segments)",
        "arch": cfg["arch"], "parameters": n_params(model), "heads": heads,
        "runtime_fingerprint": runtime.fingerprint(), "preprocess_version": PREPROCESS_VERSION,
        "input": {"views": {"A4C": "apical 4-chamber cine", "A2C": "apical 2-chamber cine", "A3C": "apical 3-chamber / long-axis cine"},
                  "required": "at least one apical view; all three for all 17 segments", "fps": 20, "clip_s": 1.6, "size": 112,
                  "format": "multi-frame greyscale DICOM (AVI / MP4 accepted without calibration)"},
        "outputs": {"segments": "score 1-4 per AHA segment with class probabilities", "territories": "LAD / RCA / LCx P(abnormal)",
                    "global": "LVEF (%), global hypokinesis, WMSI"},
        "weights_sha256": hashlib.sha256((out / "weights.pt").read_bytes()).hexdigest(),
        "thresholds": thresholds, "threshold_rule": how, "red_flags": cfg["red_flags"],
        "label_definitions": {**cfg["labels"], "parser": "echo_wm_training.labels.parse_report"},
        "splits": cfg["splits"], "manifest_sha256": manifest_hash(m, SEG_COLUMNS + heads),
        "data_summary": summary(m, heads).to_dict("records"), "clip_views": view_counts,
        "unreadable_clips": len(cache.failed), "training": dict(cfg["train"]), "history": hist,
        "metrics": {k: v.to_dict("records") for k, v in metrics.items()},
        "initialised_from": init_info, "uses_outside_data": bool(init_info and init_info.get("uses_outside_data")),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": device},
        "code_commit": _git_commit(),
        "intended_use": ("Transthoracic echo, apical 4-, 2- and 3-chamber cines: score regional wall motion in the 17 AHA "
                         "segments, map abnormalities to the coronary territory they are consistent with (LAD / RCA / LCx), "
                         "and assess global LV systolic function (LVEF estimate, global hypokinesis, WMSI) for heart-failure "
                         "assessment. For clinician review alongside the volumetric EF; not a stand-alone diagnosis."),
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2, default=float))
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--clips", required=True, help="clip table with views (dataset.build_clip_table + views.assign_views)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--init", default=None, help="self-supervised encoder (pretrain.py output) or a weights.pt")
    a = ap.parse_args()
    cfg = load_config(a.config)
    rd = (lambda p: pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p))
    card = train(rd(a.manifest), rd(a.clips), cfg, a.out, device=a.device, cache_dir=a.cache_dir, init=a.init)
    print(json.dumps({k: card[k] for k in ("parameters", "thresholds", "runtime_fingerprint", "weights_sha256")}, indent=2))


if __name__ == "__main__":
    main()
