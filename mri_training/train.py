"""Train, threshold and evaluate the MobileNetV3-UNet cardiac MRI model; write the export bundle.

    python -m mri_training.train --manifest mri_manifest.parquet [--public public_sets.csv] --out runs/mri_v1

Hybrid training (config.yaml):
  Stage A  segmentation only: public annotated sets (EMIDEC, MyoPS, ACDC) + Mayo reader-contoured studies.
           Partial labels use a marginal loss: a class a dataset does not draw is counted under the class
           that contains it (MyoPS MVO -> scar, ACDC scar -> myocardium, EMIDEC RV -> background).
  Stage B  whole model: every Mayo study (report labels for the study heads, masked where the report is
           silent) + the contoured / public studies for segmentation.
One study per step (slices of a study = the batch; BatchNorm sees one heart), gradients accumulated
over `accumulate_studies`.

Manifest columns
  Mayo   : study_id, person_id, split, dicom_uri (zip of the study's DICOM), label_uri (.npz contours or
           empty), study heads (1/0/NaN), report_lvef, report_infarct_size, report_mvo
  public : source (emidec | myops | acdc), image_uri, label_uri (NIfTI); split train | tune (public_rows())

Bundle (out/): weights.pt (plain state_dict), thresholds.json, model_card.json, metrics_*.csv.
Only weights.pt + thresholds.json + model_card.json leave the workspace (MobileNetV3 export approved);
manifests, contours and per-study predictions stay inside MCP.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import affine_transform

from ecg_training.train import _auc, _git_commit, evaluate, fit_threshold
from . import runtime
from .dataset import load_config, summary
from .runtime.model import CLASSES, build_model, n_params
from .runtime.pipeline import analyse
from .runtime.preprocess import PREPROCESS_VERSION, MRIInputError, Stack, model_input, prepare, prepare_labels
from .runtime.quantify import LV, MVO, MYO, RV, SCAR, cine_function, lge_measures

N_CLASSES = len(CLASSES)


# --------------------------------------------------------------------------- examples
@dataclass
class Example:
    """One study as read from storage (labels, if any, in the stack's own slice order)."""
    lge: Optional[Stack] = None
    lge_labels: Optional[np.ndarray] = None             # (S, H, W) class ids
    lge_rule: Optional[dict] = None                     # {"annotated": [...], "merge": {...}}
    cine: Optional[Stack] = None
    cine_labels: Optional[np.ndarray] = None            # (K, S, H, W) for phases cine_phases
    cine_phases: Optional[Sequence[int]] = None
    cine_rule: Optional[dict] = None


Loader = Callable[[pd.Series, dict], Example]


def _local(uri: str) -> str:
    if not str(uri).startswith("gs://"):
        return str(uri)
    import tempfile
    from google.cloud import storage
    bucket, _, name = str(uri)[5:].partition("/")
    dst = Path(tempfile.mkdtemp(prefix="mri_")) / Path(name).name
    storage.Client().bucket(bucket).blob(name).download_to_filename(str(dst))
    return str(dst)


def load_example(row: pd.Series, cfg: dict) -> Example:
    """Default loader: Mayo DICOM zip (+ .npz contours) or a public NIfTI pair."""
    from .io import read_nifti_pair, read_study
    src = row.get("source", "mayo") if hasattr(row, "get") else "mayo"
    if isinstance(src, str) and src in cfg["public_datasets"]:
        p = cfg["public_datasets"][src]
        stack, lab = read_nifti_pair(_local(row["image_uri"]), _local(row["label_uri"]), p["label_map"], p["kind"])
        rule = {"annotated": p["annotated"], "merge": p.get("merge", {})}
        return Example(lge=stack, lge_labels=lab, lge_rule=rule) if p["kind"] == "lge" else \
            Example(cine=Stack(stack.images[None], stack.pixel_spacing, stack.positions, stack.orientation,
                               stack.slice_thickness, "cine", stack.notes),
                    cine_labels=lab[None], cine_phases=[0], cine_rule=rule)
    lge, cine = read_study(Path(_local(row["dicom_uri"])).read_bytes())
    ex = Example(lge=lge, cine=cine)
    lu = row.get("label_uri")
    if isinstance(lu, str) and lu:
        z = np.load(_local(lu))
        mc = cfg["mayo_contours"]
        if "lge" in z:
            ex.lge_labels, ex.lge_rule = z["lge"], mc["lge"]
        if "cine" in z and cine is not None:
            ex.cine_labels, ex.cine_phases, ex.cine_rule = z["cine"], [int(v) for v in z["cine_phases"]], mc["cine"]
    return ex


@dataclass
class Part:
    """Model-ready slices of one sequence of one study."""
    x: np.ndarray                                        # (S, 2, 192, 192)
    y: Optional[np.ndarray]                              # (S, 192, 192) or None
    groups: Optional[List[List[int]]]                    # annotated class -> model classes counted under it
    study: bool                                          # contributes the study-level loss (LGE only)


def class_groups(rule: dict) -> List[List[int]]:
    ann = sorted(int(a) for a in rule["annotated"])
    if 0 not in ann:
        raise ValueError("background (0) must be annotated")
    merge = {int(k): int(v) for k, v in (rule.get("merge") or {}).items()}
    groups = {a: [a] for a in ann}
    for c in range(N_CLASSES):
        if c not in groups:
            groups[merge.get(c, 0)].append(c)
    return [groups[a] for a in ann]


def to_parts(ex: Example) -> List[Part]:
    parts: List[Part] = []
    if ex.lge is not None:
        pl = prepare(ex.lge)
        y = prepare_labels(ex.lge_labels, ex.lge.pixel_spacing, pl.order) if ex.lge_labels is not None else None
        parts.append(Part(model_input(pl.images, "lge"), y, class_groups(ex.lge_rule) if y is not None else None, True))
    if ex.cine is not None and ex.cine_labels is not None:
        pc = prepare(ex.cine)
        for k, ph in enumerate(ex.cine_phases):
            y = prepare_labels(ex.cine_labels[k], ex.cine.pixel_spacing, pc.order)
            parts.append(Part(model_input(pc.images[ph], "cine"), y, class_groups(ex.cine_rule), False))
    return parts


def augment(p: Part, a: dict, rng: np.random.Generator) -> Part:
    """Same in-plane rotation/scale for every slice of the part, intensity gamma + noise. No flips."""
    ang = np.deg2rad(rng.uniform(-1, 1) * a.get("rotate_deg", 0))
    sc = rng.uniform(*a.get("scale", (1, 1)))
    g = rng.uniform(*a.get("gamma", (1, 1)))
    m = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) / sc
    c = np.array(p.x.shape[-2:]) / 2 - 0.5
    off = c - m @ c
    img = np.stack([np.clip(affine_transform(s, m, off, order=1, mode="constant"), 0, 1) ** g for s in p.x[:, 0]])
    if a.get("noise"):
        img = np.clip(img + rng.normal(0, a["noise"], img.shape), 0, 1)
    x = p.x.copy()
    x[:, 0] = img
    y = None if p.y is None else np.stack([affine_transform(s, m, off, order=0, mode="constant") for s in p.y]).astype(np.int64)
    return Part(x.astype(np.float32), y, p.groups, p.study)


# --------------------------------------------------------------------------- losses
def marginal_seg_loss(logits: torch.Tensor, y: torch.Tensor, groups: List[List[int]]) -> torch.Tensor:
    """Cross-entropy + soft Dice on the annotated classes; unannotated classes summed into their parent."""
    lp = F.log_softmax(logits, 1)
    glp = torch.stack([torch.logsumexp(lp[:, g], 1) for g in groups], 1)            # (S, A, H, W)
    ann = [g[0] for g in groups]
    lut = torch.full((N_CLASSES,), 0, dtype=torch.long, device=y.device)
    for i, a in enumerate(ann):
        lut[a] = i
    t = lut[y]
    # class-balanced CE (inverse square-root frequency within the study): the heart is ~5% of the field
    # of view and scar / MVO far less; unweighted CE first teaches "everything is background"
    cnt = torch.bincount(t.flatten(), minlength=len(ann)).float()
    w = torch.where(cnt > 0, (cnt.sum() / cnt.clamp_min(1)).sqrt(), torch.zeros_like(cnt))
    ce = F.nll_loss(glp, t, weight=(w / w[cnt > 0].min()).clamp_max(50.0))
    p = glp.exp()
    oh = F.one_hot(t, len(ann)).permute(0, 3, 1, 2).float()
    inter, den = (p * oh).sum((0, 2, 3)), (p + oh).sum((0, 2, 3))
    dice = 1 - (2 * inter + 1) / (den + 1)
    return ce + dice[1:].mean()


def study_loss(logit: torch.Tensor, y: np.ndarray, pos_weight: torch.Tensor) -> Optional[torch.Tensor]:
    mask = ~np.isnan(y)
    if not mask.any():
        return None
    t = torch.from_numpy(np.nan_to_num(y).astype(np.float32)).to(logit.device)
    mk = torch.from_numpy(mask.astype(np.float32)).to(logit.device)
    l = F.binary_cross_entropy_with_logits(logit, t, reduction="none", pos_weight=pos_weight)
    return (l * mk).sum() / mk.sum()


# --------------------------------------------------------------------------- metrics
def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float("nan") if s == 0 else float(2 * (a & b).sum() / s)


SEG_STRUCTURES = {"lv_pool": (LV,), "lv_myocardium": (MYO, SCAR, MVO), "scar": (SCAR, MVO), "mvo": (MVO,), "rv_pool": (RV,)}


def agreement(pred: Sequence[float], ref: Sequence[float]) -> Dict[str, float]:
    p, r = np.asarray(pred, float), np.asarray(ref, float)
    ok = ~(np.isnan(p) | np.isnan(r))
    p, r = p[ok], r[ok]
    if len(p) == 0:
        return {"n": 0}
    d = p - r
    return {"n": int(len(p)), "mae": float(np.abs(d).mean()), "bias": float(d.mean()),
            "loa_lo": float(d.mean() - 1.96 * d.std(ddof=1)) if len(d) > 1 else float("nan"),
            "loa_hi": float(d.mean() + 1.96 * d.std(ddof=1)) if len(d) > 1 else float("nan"),
            "pearson_r": float(np.corrcoef(p, r)[0, 1]) if len(p) > 2 and p.std() > 0 and r.std() > 0 else float("nan")}


# --------------------------------------------------------------------------- training
class PartCache:
    """Prepared parts per manifest row (memory, or .npz files under cache_dir inside the workspace)."""

    def __init__(self, m: pd.DataFrame, cfg: dict, loader: Loader, cache_dir: Optional[str] = None):
        self.m, self.cfg, self.loader = m, cfg, loader
        self.dir = Path(cache_dir) if cache_dir else None
        self.mem: Dict[int, Optional[List[Part]]] = {}
        self.failed: Dict[int, str] = {}

    def __call__(self, i: int) -> Optional[List[Part]]:
        if i in self.mem:
            return self.mem[i]
        f = self.dir / f"{i}.pt" if self.dir else None
        if f is not None and f.exists():
            return torch.load(f, weights_only=False)
        try:
            parts = to_parts(self.loader(self.m.loc[i], self.cfg))
        except (MRIInputError, ValueError, OSError) as e:
            self.failed[i] = str(e)
            parts = None
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            torch.save(parts, f)
        else:
            self.mem[i] = parts
        return parts


def _is_mayo(m: pd.DataFrame) -> pd.Series:
    return m["source"].fillna("mayo").eq("mayo") if "source" in m else pd.Series(True, index=m.index)


@torch.no_grad()
def tune_scores(model, m: pd.DataFrame, idx: Sequence[int], cache: PartCache, heads: List[str], device: str) -> Dict[str, object]:
    """Quick per-epoch check on the tuning split: Dice (contoured) and study AUROC (report labels)."""
    model.eval()
    dices = {k: [] for k in SEG_STRUCTURES}
    probs, ys = [], []
    for i in idx:
        parts = cache(i)
        if not parts:
            continue
        for p in parts:
            seg, st = model(torch.from_numpy(p.x).to(device))
            if p.y is not None:
                lab = seg.argmax(1).cpu().numpy()
                ann = {g[0] for g in p.groups}
                for k, cls in SEG_STRUCTURES.items():
                    if set(cls) <= ann:
                        d = dice(np.isin(lab, cls), np.isin(p.y, cls))
                        if not math.isnan(d):
                            dices[k].append(d)
            if p.study and _is_mayo(m).loc[i]:
                probs.append(torch.sigmoid(st).cpu().numpy())
                ys.append(m.loc[i, heads].to_numpy(dtype=float))
    out: Dict[str, object] = {f"dice_{k}": float(np.mean(v)) if v else float("nan") for k, v in dices.items()}
    if probs:
        P, Y = np.vstack(probs), np.vstack(ys)
        for j, h in enumerate(heads):
            lab = ~np.isnan(Y[:, j])
            out[f"auroc_{h}"] = _auc(Y[lab, j], P[lab, j])
    return out


def _score(s: Dict[str, object], stage: str) -> float:
    seg = [s.get("dice_lv_myocardium"), s.get("dice_scar")]
    vals = [v for v in seg if v is not None and not math.isnan(v)]
    if stage == "B":
        vals += [v for k, v in s.items() if k.startswith("auroc_") and not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def run_stage(stage: str, model, m: pd.DataFrame, train_idx: List[int], tune_idx: List[int], cache: PartCache,
              cfg: dict, epochs: int, study_weight: float, pos_weight: torch.Tensor, device: str,
              log: Callable[[str], None], rng: np.random.Generator) -> List[dict]:
    tc, heads = cfg["train"], cfg["study_heads"]
    acc = max(1, int(tc["accumulate_studies"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]))
    steps = max(1, epochs * math.ceil(len(train_idx) / acc))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(tc["lr"]), total_steps=steps)
    mayo = _is_mayo(m)
    best, best_state, bad, hist = -float("inf"), None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        tot, n, k = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for i in rng.permutation(train_idx):
            parts = cache(int(i))
            if not parts:
                continue
            loss, used = 0.0, False
            for p in parts:
                needs_study = p.study and study_weight > 0 and mayo.loc[i]
                if p.y is None and not needs_study:
                    continue
                if tc.get("augment"):
                    p = augment(p, tc["augment"], rng)
                seg, st = model(torch.from_numpy(p.x).to(device))
                if p.y is not None:
                    loss = loss + float(tc["seg_weight"]) * marginal_seg_loss(seg, torch.from_numpy(p.y).to(device), p.groups)
                    used = True
                if needs_study:
                    sl = study_loss(st, m.loc[i, heads].to_numpy(dtype=float), pos_weight)
                    if sl is not None:
                        loss, used = loss + study_weight * sl, True
            if not used:
                continue
            (loss / acc).backward()
            tot, n, k = tot + float(loss.detach()), n + 1, k + 1
            if k == acc:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step() if sched.last_epoch + 1 < steps else None
                opt.zero_grad(set_to_none=True)
                k = 0
        if k:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step() if sched.last_epoch + 1 < steps else None
            opt.zero_grad(set_to_none=True)
        s = tune_scores(model, m, tune_idx, cache, heads, device)
        score = _score(s, stage)
        if math.isnan(score):
            score = -tot / max(n, 1)
        hist.append({"stage": stage, "epoch": ep, "train_loss": tot / max(n, 1), "studies": n, "score": score, **s})
        log(f"stage {stage} epoch {ep}: loss {tot / max(n, 1):.4f} score {score:.3f} "
            + " ".join(f"{a} {b:.3f}" for a, b in s.items() if not math.isnan(b)))
        if score > best:
            best, bad = score, 0
            best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(tc["patience"]):
                log(f"stage {stage}: early stop at epoch {ep}")
                break
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return hist


# --------------------------------------------------------------------------- evaluation
def study_level(model, m: pd.DataFrame, idx: Sequence[int], cfg: dict, loader: Loader, device: str) -> pd.DataFrame:
    """Full pipeline (runtime.pipeline.analyse, as served) per study + reference values where known."""
    rows = []
    for i in idx:
        r = m.loc[i]
        try:
            ex = loader(r, cfg)
            a = analyse(model, ex.lge, ex.cine, device)
        except (MRIInputError, ValueError, OSError) as e:
            rows.append({"row": i, "error": str(e)})
            continue
        f = a["findings"]
        out = {"row": i, "error": None, **{f"p_{h}": a["probabilities"][h] for h in cfg["study_heads"]},
               **{k: f.get(k) for k in ("infarct_size_pct_lv", "mvo_pct_lv", "lv_mass_g", "lvef_pct", "lvedv_ml", "lvesv_ml",
                                        "scar_median_depth", "scar_endocardial_involvement")}}
        if ex.lge_labels is not None and set(ex.lge_rule["annotated"]) >= {0, 1, 2, 3}:
            pl = prepare(ex.lge)
            ref = prepare_labels(ex.lge_labels, ex.lge.pixel_spacing, pl.order)
            for k, cls in SEG_STRUCTURES.items():
                out[f"dice_{k}"] = dice(np.isin(a["lge_labels"], cls), np.isin(ref, cls))
            rf = lge_measures(ref, pl.slice_spacing_mm, pl.row_cos, pl.col_cos)
            out["ref_infarct_size_pct_lv"], out["ref_mvo_pct_lv"] = rf["infarct_size_pct_lv"], rf["mvo_pct_lv"]
        if ex.cine_labels is not None and ex.cine is not None:
            pc = prepare(ex.cine)
            ref = np.stack([prepare_labels(ex.cine_labels[k], ex.cine.pixel_spacing, pc.order) for k in range(len(ex.cine_phases))])
            out["ref_lvef_pct"] = cine_function(ref, pc.slice_spacing_mm)["lvef_pct"]
        rows.append(out)
    return pd.DataFrame(rows, columns=None if rows else ["row", "error"]).set_index("row")


def _num(s) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def split_metrics(m: pd.DataFrame, res: pd.DataFrame, cfg: dict, thresholds: Dict[str, float], n_boot: int) -> Dict[str, pd.DataFrame]:
    heads = cfg["study_heads"]
    ok = res[res.error.isna()] if "error" in res else res
    part = m.loc[ok.index]
    out: Dict[str, pd.DataFrame] = {}
    mayo = part[_is_mayo(part)]
    if len(mayo):
        out["study"] = evaluate(mayo, ok.loc[mayo.index, [f"p_{h}" for h in heads]].to_numpy(dtype=float), heads, thresholds, n_boot=n_boot)
    rows = []
    for k in SEG_STRUCTURES:
        c = f"dice_{k}"
        if c in ok:
            v = _num(ok[c])
            v = v[~np.isnan(v)]
            if len(v):
                rows.append({"measure": c, "n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
                             "p10": float(np.percentile(v, 10))})
    pairs = [("infarct_size_pct_lv", "ref_infarct_size_pct_lv", "infarct size vs reader contours (% LV)"),
             ("mvo_pct_lv", "ref_mvo_pct_lv", "MVO vs reader contours (% LV)"),
             ("lvef_pct", "ref_lvef_pct", "LVEF vs reader contours (%)")]
    for p, r, name in pairs:
        if r in ok:
            rows.append({"measure": name, **agreement(_num(ok[p]), _num(ok[r]))})
    if len(mayo):
        for p, r, name in (("lvef_pct", "report_lvef", "LVEF vs report (%)"),
                           ("infarct_size_pct_lv", "report_infarct_size", "infarct size vs report (% LV)")):
            if r in mayo:
                rows.append({"measure": name, **agreement(_num(ok.loc[mayo.index, p]), _num(mayo[r]))})
        if "report_mvo" in mayo:
            rule = cfg["red_flags"]["mvo_present"]
            y = _num(mayo["report_mvo"])
            pred = _num(ok.loc[mayo.index, "mvo_pct_lv"]) >= float(rule["min"])
            lab = ~np.isnan(y)
            tp, fn = int(((y == 1) & pred & lab).sum()), int(((y == 1) & ~pred & lab).sum())
            tn, fp = int(((y == 0) & ~pred & lab).sum()), int(((y == 0) & pred & lab).sum())
            rows.append({"measure": "MVO present vs report", "n": int(lab.sum()),
                         "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
                         "specificity": tn / (tn + fp) if tn + fp else float("nan")})
    if rows:
        out["quantitative"] = pd.DataFrame(rows)
    return out


def manifest_hash(m: pd.DataFrame, heads: List[str]) -> str:
    cols = [c for c in ["source", "study_id", "image_uri", "person_id", "split"] + heads if c in m]
    key = [c for c in ("split", "study_id", "image_uri") if c in m]
    blob = m[cols].sort_values(key).to_csv(index=False).encode()
    return hashlib.sha256(blob).hexdigest()


def public_rows(csv_path: str, cfg: dict, tune_fraction: float = 0.1, seed: Optional[int] = None) -> pd.DataFrame:
    """Public annotated studies (columns source, image_uri, label_uri) -> manifest rows (train / tune)."""
    p = pd.read_csv(csv_path)
    unknown = set(p.source) - set(cfg["public_datasets"])
    if unknown:
        raise ValueError(f"unknown public sources {unknown}; add them to config.yaml public_datasets")
    rng = np.random.default_rng(cfg["splits"]["seed"] if seed is None else seed)
    p["split"] = np.where(rng.random(len(p)) < tune_fraction, "tune", "train")
    return p


def load_init(model, init: str, log: Callable[[str], None]) -> dict:
    """Start from weights trained elsewhere (e.g. on the public sets OUTSIDE MCP): plain state_dict, strict."""
    p = Path(init)
    wpath = p / "weights.pt" if p.is_dir() else p
    state = torch.load(wpath, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise TypeError("--init must be a plain state_dict (weights.pt from mri_training.train)")
    model.load_state_dict(state, strict=True)
    info = {"weights_sha256": hashlib.sha256(wpath.read_bytes()).hexdigest(), "path": str(wpath)}
    card = wpath.parent / "model_card.json"
    if card.exists():                                   # provenance of the starting point (which public sets)
        c = json.loads(card.read_text())
        info.update({k: c.get(k) for k in ("model_name", "runtime_fingerprint", "public_studies", "public_datasets", "code_commit")})
    log(f"initialised from {wpath} ({info['weights_sha256'][:12]})")
    return info


def train(m: pd.DataFrame, cfg: dict, out_dir: str, loader: Loader = load_example, device: Optional[str] = None,
          log: Callable[[str], None] = print, cache_dir: Optional[str] = None, n_boot: int = 1000,
          init: Optional[str] = None) -> dict:
    heads, tc = cfg["study_heads"], cfg["train"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["splits"]["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = m.reset_index(drop=True)
    if "source" not in m:
        m["source"] = "mayo"
    m["source"] = m["source"].fillna("mayo")
    for h in heads:
        if h not in m:
            m[h] = np.nan
    mayo = _is_mayo(m)
    has_lab = m["label_uri"].notna() & m["label_uri"].astype(str).ne("") if "label_uri" in m else pd.Series(False, index=m.index)
    tr = [int(i) for i in m.index[m.split == "train"]]
    tu = [int(i) for i in m.index[m.split == "tune"]]
    cache = PartCache(m, cfg, loader, cache_dir)
    model = build_model({**cfg["arch"], "study_heads": heads})
    init_info = load_init(model, init, log) if init else None
    model = model.to(device)
    prev = np.nan_to_num(np.nanmean(m.loc[mayo & (m.split == "train"), heads].to_numpy(dtype=float), axis=0)
                         if (mayo & (m.split == "train")).any() else np.full(len(heads), np.nan), nan=0.5).clip(1e-3, 1 - 1e-3)
    pos_weight = torch.tensor(np.clip((1 - prev) / prev, 1.0, 50.0), dtype=torch.float32, device=device)

    hist = run_stage("A", model, m, [i for i in tr if has_lab[i]], [i for i in tu if has_lab[i]], cache, cfg,
                     int(tc["stage_a_epochs"]), 0.0, pos_weight, device, log, rng)
    hist += run_stage("B", model, m, tr, tu, cache, cfg, int(tc["stage_b_epochs"]), float(tc["study_weight"]),
                      pos_weight, device, log, rng)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, out / "weights.pt")                                 # plain state_dict only
    model.eval()

    tu_mayo = [i for i in tu if mayo[i]]
    res_tu = study_level(model, m, tu_mayo, cfg, loader, device)
    thresholds, how = {}, {}
    ok_tu = res_tu[res_tu.error.isna()] if len(res_tu) else res_tu
    for h in heads:
        y = m.loc[ok_tu.index, h].to_numpy(dtype=float) if len(ok_tu) else np.array([])
        p = ok_tu[f"p_{h}"].to_numpy(dtype=float) if len(ok_tu) else np.array([])
        lab = ~np.isnan(y)
        thresholds[h], how[h] = fit_threshold(y[lab], p[lab], float(cfg["thresholds"][h]["target_sensitivity"]))
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    metrics: Dict[str, pd.DataFrame] = {}
    splits = {"tune": res_tu}
    for split in ("test_time", "test_site"):
        idx = [int(i) for i in m.index[(m.split == split) & mayo]]
        if idx:
            splits[split] = study_level(model, m, idx, cfg, loader, device)
    for split, res in splits.items():
        if len(res):
            for k, v in split_metrics(m, res, cfg, thresholds, n_boot=0 if split == "tune" else n_boot).items():
                metrics[f"{split}_{k}"] = v
    for k, v in metrics.items():
        v.to_csv(out / f"metrics_{k}.csv", index=False)

    card = {
        "model_name": cfg["model_name"], "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "architecture": "MobileNetV3-Large encoder (2-D) + U-Net decoder + study heads", "arch": cfg["arch"],
        "classes": CLASSES, "study_heads": heads, "parameters": n_params(model),
        "runtime_fingerprint": runtime.fingerprint(), "preprocess_version": PREPROCESS_VERSION,
        "input": {"sequences": {"lge": "short-axis LGE (required)", "cine": "short-axis cine (optional: LV volumes, LVEF)"},
                  "pixel_mm": 1.25, "size": 192, "format": "zip of the study's DICOM files"},
        "weights_sha256": hashlib.sha256((out / "weights.pt").read_bytes()).hexdigest(),
        "thresholds": thresholds, "threshold_rule": how, "red_flags": cfg["red_flags"],
        "label_definitions": cfg["labels"], "public_datasets": cfg["public_datasets"], "mayo_contours": cfg["mayo_contours"],
        "splits": cfg["splits"], "manifest_sha256": manifest_hash(m, heads),
        "data_summary": summary(m[mayo], heads).to_dict("records") if mayo.any() else [],
        "public_studies": m[~mayo].groupby(["source", "split"]).size().reset_index(name="n").to_dict("records") if (~mayo).any() else [],
        "unreadable_studies": len(cache.failed),
        "training": dict(tc), "history": hist,
        "metrics": {k: v.to_dict("records") for k, v in metrics.items()},
        "initialised_from": init_info,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": device},
        "code_commit": _git_commit(),
        "intended_use": ("Cardiac MRI (LGE + cine short axis) after suspected or confirmed STEMI: confirm infarct, quantify "
                         "infarct size, transmurality per AHA segment, MVO and LVEF, and flag findings needing further "
                         "investigation (non-ischaemic LGE pattern / STEMI mimic, LV thrombus, LV aneurysm, low LVEF). "
                         "For clinician review; not a stand-alone diagnosis."),
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2, default=float))
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None,
                    help="Mayo manifest (build_manifest + label_uri for contoured studies); omit with --public to train "
                         "the public-only model that pre-draws outlines for the readers (annotation/README.md)")
    ap.add_argument("--public", default=None, help="CSV of public annotated studies: source,image_uri,label_uri")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--init", default=None,
                    help="start from weights.pt (or its bundle folder) trained elsewhere, e.g. on the public sets outside MCP")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if not (a.manifest or a.public):
        ap.error("give --manifest, --public, or both")
    parts = [pd.read_parquet(a.manifest).assign(source="mayo")] if a.manifest else []
    if a.public:
        parts.append(public_rows(a.public, cfg))
    m = pd.concat(parts, ignore_index=True)
    card = train(m, cfg, a.out, device=a.device, cache_dir=a.cache_dir, init=a.init)
    print(json.dumps({k: card[k] for k in ("parameters", "thresholds", "runtime_fingerprint", "weights_sha256")}, indent=2))


if __name__ == "__main__":
    main()
