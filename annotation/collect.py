"""Collect readers' finished label files, check them, and write training labels + inter-reader agreement.

    python -m annotation.collect mri  --root /workspace/annotation --out /workspace/mri/contours
    python -m annotation.collect echo --root /workspace/annotation_echo --out /workspace/echo/traces

Output
  <out>/<study_id>__<reader>.npz   MRI: "lge" (S, H, W) in read_study() slice order; "cine" (2, S, H, W) +
                                   "cine_phases" [ed, es]  (the format mri_training.train.load_example reads)
                                   Echo: "frames" [ed, es], "masks" (2, H, W)
  <out>/collected.csv               one row per study x reader: status, errors, warnings, file
  <out>/agreement.csv               double-read studies: Dice per structure, infarct size / EF differences
  <out>/label_uri.csv               study_id -> label file of the PRIMARY reader (reader_1), for the manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mri_training.runtime.quantify import LV, MVO, MYO, RV, SCAR

from .geometry import from_nifti_array

MRI_ALLOWED = {"lge": {0, LV, MYO, SCAR, MVO, RV}, "cine": {0, LV, MYO, RV}}
STRUCTURES = {"lv_pool": (LV,), "lv_myocardium": (MYO, SCAR, MVO), "scar": (SCAR, MVO), "mvo": (MVO,), "rv_pool": (RV,)}


def _load(path: Path) -> Tuple[np.ndarray, str]:
    import nibabel as nib
    a = np.asarray(nib.load(str(path)).get_fdata()).round().astype(np.int64)
    return a, hashlib.sha256(np.ascontiguousarray(a.astype(np.uint8)).tobytes()).hexdigest()


def _check_lge(lab: np.ndarray) -> Tuple[List[str], List[str]]:
    """lab (S, H, W). Errors stop the study from being used; warnings are for review."""
    err, warn = [], []
    vals = set(np.unique(lab).tolist())
    if vals - MRI_ALLOWED["lge"]:
        err.append(f"LGE has label values {sorted(vals - MRI_ALLOWED['lge'])}; allowed {sorted(MRI_ALLOWED['lge'])}")
    myo_slices = [i for i in range(lab.shape[0]) if np.isin(lab[i], (MYO, SCAR, MVO)).any()]
    if len(myo_slices) < 3:
        err.append(f"LV myocardium drawn on {len(myo_slices)} slices (need the whole LV, at least 3)")
    if not (lab == LV).any():
        err.append("no LV blood pool drawn")
    for i in range(lab.shape[0]):
        if (lab[i] == MVO).any() and not (lab[i] == SCAR).any():
            warn.append(f"slice {i}: MVO without surrounding scar")
        if np.isin(lab[i], (SCAR, MVO)).any() and not (lab[i] == MYO).any():
            warn.append(f"slice {i}: scar but no viable myocardium (fully transmural ring?)")
    if not (lab == RV).any():
        warn.append("no RV drawn (AHA segments need the RV to find the septum)")
    return err, warn


def _check_cine(lab: np.ndarray, name: str) -> Tuple[List[str], List[str]]:
    err, warn = [], []
    vals = set(np.unique(lab).tolist())
    if vals - MRI_ALLOWED["cine"]:
        err.append(f"{name} has label values {sorted(vals - MRI_ALLOWED['cine'])}; allowed {sorted(MRI_ALLOWED['cine'])} (no scar on cine)")
    if not (lab == LV).any():
        err.append(f"{name}: no LV blood pool drawn")
    return err, warn


def collect_mri_folder(folder: Path, allow_unchanged: bool = False) -> Tuple[Optional[Dict[str, np.ndarray]], List[str], List[str]]:
    meta = json.loads((folder / "meta.json").read_text())
    err: List[str] = []
    warn: List[str] = []
    lge_nii, sha = _load(folder / "lge_label.nii.gz")
    if sha == meta["lge"]["prelabel_sha256"] and not allow_unchanged:
        err.append("LGE label file is unchanged from the export: not reviewed yet")
    order = np.asarray(meta["lge"]["order"])
    if lge_nii.shape != (meta["lge"]["shape_shw"][2], meta["lge"]["shape_shw"][1], meta["lge"]["shape_shw"][0]):
        return None, [f"LGE label shape {lge_nii.shape} differs from the image"], warn
    lge = from_nifti_array(lge_nii, order)
    e, w = _check_lge(lge)
    err += e
    warn += w
    out: Dict[str, np.ndarray] = {"lge": lge.astype(np.uint8)}
    if "cine" in meta and (folder / "cine_ed_label.nii.gz").exists():
        corder = np.asarray(meta["cine"]["order"])
        phases, labs = [], []
        for name in ("ed", "es"):
            a, s = _load(folder / f"cine_{name}_label.nii.gz")
            if s == meta["cine"][f"{name}_prelabel_sha256"] and not allow_unchanged:
                err.append(f"cine {name.upper()} label file is unchanged from the export: not reviewed yet")
            c = from_nifti_array(a, corder)
            e, w = _check_cine(c, f"cine {name.upper()}")
            err += e
            warn += w
            labs.append(c)
            phases.append(int(meta["cine"]["phases"][name]))
        if (labs[1] == LV).sum() >= (labs[0] == LV).sum():
            warn.append("ES LV pool is not smaller than ED: check the phases or the drawing")
        out["cine"] = np.stack(labs).astype(np.uint8)
        out["cine_phases"] = np.asarray(phases)
    return (None if err else out), err, warn


def collect_echo_folder(folder: Path, allow_unchanged: bool = False):
    meta = json.loads((folder / "meta.json").read_text())
    a, sha = _load(folder / "cine_label.nii.gz")
    if sha == meta["prelabel_sha256"] and not allow_unchanged:
        return None, ["label file is unchanged from the export: not reviewed yet"], []
    lab = np.transpose(a, (2, 1, 0))                                         # (F, H, W)
    err, warn = [], []
    if set(np.unique(lab).tolist()) - {0, 1}:
        err.append("only label 1 (LV cavity) is allowed")
    drawn = [f for f in range(lab.shape[0]) if (lab[f] > 0).any()]
    if len(drawn) != 2:
        err.append(f"LV cavity drawn on {len(drawn)} frames; draw exactly two (end-diastole and end-systole)")
    if err:
        return None, err, warn
    areas = [int((lab[f] > 0).sum()) for f in drawn]
    ed, es = (drawn[0], drawn[1]) if areas[0] >= areas[1] else (drawn[1], drawn[0])
    return {"frames": np.asarray([ed, es]), "masks": np.stack([lab[ed] > 0, lab[es] > 0]).astype(np.uint8)}, err, warn


# --------------------------------------------------------------------------- agreement
def _dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float("nan") if s == 0 else float(2 * (a & b).sum() / s)


def _infarct_pct(lge: np.ndarray) -> float:
    myo = np.isin(lge, (MYO, SCAR, MVO)).sum()
    return float("nan") if myo == 0 else 100.0 * np.isin(lge, (SCAR, MVO)).sum() / myo


def _ef(cine: np.ndarray) -> float:
    ed, es = (cine[0] == LV).sum(), (cine[1] == LV).sum()                   # voxel size cancels
    return float("nan") if ed == 0 else 100.0 * (ed - es) / ed


def mri_agreement(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, float]:
    r = {f"dice_{k}": _dice(np.isin(a["lge"], c), np.isin(b["lge"], c)) for k, c in STRUCTURES.items()}
    r["infarct_pct_a"], r["infarct_pct_b"] = _infarct_pct(a["lge"]), _infarct_pct(b["lge"])
    r["infarct_pct_diff"] = r["infarct_pct_a"] - r["infarct_pct_b"]
    if "cine" in a and "cine" in b:
        r["lvef_a"], r["lvef_b"] = _ef(a["cine"]), _ef(b["cine"])
        r["lvef_diff"] = r["lvef_a"] - r["lvef_b"]
    return r


def echo_agreement(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, float]:
    fac = lambda m: 100.0 * (m[0].sum() - m[1].sum()) / max(m[0].sum(), 1)  # noqa: E731  fractional area change
    return {"dice_ed": _dice(a["masks"][0] > 0, b["masks"][0] > 0), "dice_es": _dice(a["masks"][1] > 0, b["masks"][1] > 0),
            "same_frames": bool(np.array_equal(a["frames"], b["frames"])), "fac_diff": fac(a["masks"]) - fac(b["masks"])}


def collect(task: str, root: str, out: str, allow_unchanged: bool = False) -> Dict[str, pd.DataFrame]:
    rootp, outp = Path(root), Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    fn = collect_mri_folder if task == "mri" else collect_echo_folder
    rows, done = [], {}
    for meta_path in sorted(rootp.glob("*/*/meta.json")):
        folder = meta_path.parent
        reader, study = folder.parent.name, folder.name
        meta = json.loads(meta_path.read_text())
        try:
            labels, err, warn = fn(folder, allow_unchanged)
        except Exception as e:  # noqa: BLE001
            labels, err, warn = None, [f"could not read: {e}"], []
        path = None
        if labels is not None:
            path = outp / f"{study}__{reader}.npz"
            np.savez_compressed(path, **labels)
            done[(study, reader)] = labels
        primary = (meta.get("readers") or [reader])[0] == reader
        rows.append({"study_id": study, "reader": reader, "primary": primary, "status": "ok" if labels is not None else "rejected",
                     "errors": "; ".join(err), "warnings": "; ".join(warn), "label_file": str(path) if path else None})
    table = pd.DataFrame(rows, columns=["study_id", "reader", "primary", "status", "errors", "warnings", "label_file"])
    agree = []
    for study, g in table[table.status == "ok"].groupby("study_id"):
        if len(g) >= 2:
            ra, rb = g.reader.iloc[0], g.reader.iloc[1]
            f = mri_agreement if task == "mri" else echo_agreement
            agree.append({"study_id": study, "reader_a": ra, "reader_b": rb, **f(done[(study, ra)], done[(study, rb)])})
    agreement = pd.DataFrame(agree)
    label_uri = table[(table.status == "ok") & table.primary][["study_id", "label_file"]].rename(columns={"label_file": "label_uri"})
    table.to_csv(outp / "collected.csv", index=False)
    agreement.to_csv(outp / "agreement.csv", index=False)
    label_uri.to_csv(outp / "label_uri.csv", index=False)
    return {"collected": table, "agreement": agreement, "label_uri": label_uri}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["mri", "echo"])
    ap.add_argument("--root", required=True, help="the --out folder given to annotation.export")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-unchanged", action="store_true", help="accept label files identical to the pre-drawn outlines")
    a = ap.parse_args()
    res = collect(a.task, a.root, a.out, a.allow_unchanged)
    t = res["collected"]
    print(t.status.value_counts().to_string())
    if len(res["agreement"]):
        print(res["agreement"].filter(like="dice").describe().loc[["count", "mean", "min"]].round(3).to_string())


if __name__ == "__main__":
    main()
