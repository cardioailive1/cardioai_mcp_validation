"""Clinical measurements from short-axis label maps (CLASSES order in model.py).

  LV mass (g)            : myocardium + scar + MVO volume x 1.05 g/mL
  infarct size (% LV)    : (scar + MVO) / LV myocardium
  MVO (% LV)             : MVO / LV myocardium
  transmurality          : per AHA segment 1-16 (17 = apical cap, not imaged in short axis),
                           scar area / myocardial area in the segment (area-based approximation)
  scar wall depth        : position of scar within the wall, 0 = endocardium .. 1 = epicardium
                           (ischaemic scar starts at the endocardium: low median depth and endocardial
                           involvement; myocarditis-type scar is mid-wall or epicardial)
  LVEF / EDV / ESV       : cine, Simpson's disc summation over slices, ED = max, ES = min LV pool volume

AHA orientation: septum = direction from the LV centre to the RV centre; which neighbour of the
septum is anterior comes from the DICOM ImageOrientationPatient (patient LPS: anterior = -Y).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import distance_transform_edt

from .preprocess import PIXEL_MM

BG, LV, MYO, SCAR, MVO, RV = range(6)
DENSITY = 1.05
BASAL = ["anterior", "anteroseptal", "inferoseptal", "inferior", "inferolateral", "anterolateral"]
APICAL = ["anterior", "septal", "inferior", "lateral"]


def _voxel_ml(slice_mm: float) -> float:
    return PIXEL_MM * PIXEL_MM * slice_mm / 1000.0


def lv_slices(lab: np.ndarray) -> np.ndarray:
    """Indices of slices with LV myocardium, ordered base -> apex (base = end with the larger LV pool)."""
    has = [i for i in range(lab.shape[0]) if np.isin(lab[i], (MYO, SCAR, MVO)).any()]
    if len(has) < 2:
        return np.asarray(has)
    a0, a1 = (lab[has[0]] == LV).sum(), (lab[has[-1]] == LV).sum()
    return np.asarray(has if a0 >= a1 else has[::-1])


def _centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.nonzero(mask)
    return None if len(xs) == 0 else np.array([xs.mean(), ys.mean()])


def _anterior(theta: float, row_cos: np.ndarray, col_cos: np.ndarray) -> float:
    """Anterior component (patient -Y) of the image direction at angle theta (x = columns, y = rows)."""
    v = np.cos(theta) * row_cos + np.sin(theta) * col_cos
    return -float(v[1])


def segment_map(lab: np.ndarray, row_cos: np.ndarray, col_cos: np.ndarray) -> Dict[str, object]:
    """Per-pixel AHA segment number (0 = none) for the LV myocardium of every slice."""
    order = lv_slices(lab)
    seg = np.zeros(lab.shape, np.int16)
    if len(order) == 0:
        return {"segments": seg, "order": order, "note": "no LV myocardium"}
    septal = []
    for i in order:
        lc, rc = _centroid(np.isin(lab[i], (LV, MYO, SCAR, MVO))), _centroid(lab[i] == RV)
        septal.append(None if lc is None or rc is None else np.arctan2(rc[1] - lc[1], rc[0] - lc[0]))
    known = [a for a in septal if a is not None]
    if not known:
        return {"segments": seg, "order": order, "note": "RV not found: AHA segments unavailable"}
    ref = float(np.angle(np.mean(np.exp(1j * np.array(known)))))
    septal = [ref if a is None else a for a in septal]
    n = len(order)
    # thirds base -> apex (0 basal, 1 mid, 2 apical); any remainder slices go to the base
    level = np.repeat([0, 1, 2], [n - 2 * (n // 3), n // 3, n // 3]) if n >= 3 else np.arange(n)
    for k, i in enumerate(order):
        ts = septal[k]
        # which side of the septum is anterior (image handedness from DICOM orientation)
        s = 1.0 if _anterior(ts + np.pi / 3, row_cos, col_cos) > _anterior(ts - np.pi / 3, row_cos, col_cos) else -1.0
        myo = np.isin(lab[i], (MYO, SCAR, MVO))
        c = _centroid(np.isin(lab[i], (LV, MYO, SCAR, MVO)))
        ys, xs = np.nonzero(myo)
        rel = (s * (np.arctan2(ys - c[1], xs - c[0]) - ts)) % (2 * np.pi)          # 0 = septal centre, + towards anterior
        if level[k] < 2:
            # sectors of 60 deg from the septal centre towards anterior: anteroseptal, anterior, anterolateral,
            # inferolateral, inferior, inferoseptal
            sector = (rel // (np.pi / 3)).astype(int) % 6
            names = ["anteroseptal", "anterior", "anterolateral", "inferolateral", "inferior", "inferoseptal"]
            num = np.array([BASAL.index(names[j]) + 1 for j in sector]) + 6 * level[k]
        else:
            sector = (((rel + np.pi / 4) % (2 * np.pi)) // (np.pi / 2)).astype(int) % 4   # septal centred on 0
            names = ["septal", "anterior", "lateral", "inferior"]
            num = np.array([APICAL.index(names[j]) + 13 for j in sector])
        seg[i, ys, xs] = num
    return {"segments": seg, "order": order, "note": ""}


def _bin(frac: float) -> str:
    if frac <= 0:
        return "none"
    for hi, name in ((0.25, "1-25%"), (0.50, "26-50%"), (0.75, "51-75%")):
        if frac <= hi:
            return name
    return ">75%"


def lge_measures(lab: np.ndarray, slice_mm: float, row_cos: np.ndarray, col_cos: np.ndarray) -> Dict[str, object]:
    myo = np.isin(lab, (MYO, SCAR, MVO))
    scar = np.isin(lab, (SCAR, MVO))
    n_myo, n_scar, n_mvo = int(myo.sum()), int(scar.sum()), int((lab == MVO).sum())
    vox = _voxel_ml(slice_mm)
    out: Dict[str, object] = {
        "lv_mass_g": round(n_myo * vox * DENSITY, 1),
        "infarct_size_pct_lv": round(100 * n_scar / n_myo, 1) if n_myo else None,
        "infarct_mass_g": round(n_scar * vox * DENSITY, 1),
        "mvo_pct_lv": round(100 * n_mvo / n_myo, 2) if n_myo else None,
    }
    if n_scar:
        depths = []
        for i in range(lab.shape[0]):
            if not scar[i].any():
                continue
            d_endo = distance_transform_edt(lab[i] != LV)                 # distance to the LV pool
            d_epi = distance_transform_edt(myo[i] | (lab[i] == LV))       # distance to outside the LV
            depths.append(d_endo[scar[i]] / np.maximum(d_endo[scar[i]] + d_epi[scar[i]], 1e-6))
        dep = np.concatenate(depths)
        out["scar_median_depth"] = round(float(np.median(dep)), 3)
        out["scar_endocardial_involvement"] = round(float((dep < 0.2).mean()), 3)
    else:
        out["scar_median_depth"] = None
        out["scar_endocardial_involvement"] = None
    sm = segment_map(lab, row_cos, col_cos)
    segs = {}
    for k in range(1, 17):
        in_seg = sm["segments"] == k
        m = int((in_seg & myo).sum())
        if m:
            f = float((in_seg & scar).sum()) / m
            segs[k] = {"transmurality": round(f, 3), "category": _bin(f)}
    out["segments"] = segs
    out["segments_note"] = sm["note"]
    out["n_lv_slices"] = int(len(sm["order"]))
    return out


def cine_function(lab_phases: np.ndarray, slice_mm: float) -> Dict[str, Optional[float]]:
    """lab_phases: (P, S, H, W). Simpson's discs: LV pool volume per phase; ED = max, ES = min."""
    vols = (lab_phases == LV).reshape(lab_phases.shape[0], -1).sum(1) * _voxel_ml(slice_mm)
    if vols.max() <= 0:
        return {"lvedv_ml": None, "lvesv_ml": None, "lvef_pct": None}
    edv, esv = float(vols.max()), float(vols.min())
    return {"lvedv_ml": round(edv, 1), "lvesv_ml": round(esv, 1), "lvef_pct": round(100 * (edv - esv) / edv, 1)}


def red_flags(findings: Dict[str, object], probs: Dict[str, float], rules: Dict[str, Dict]) -> List[Dict[str, object]]:
    """Findings that need further investigation, by locked, clinically agreed rules (model card / lock file)."""
    out = []
    for name, r in rules.items():
        if "prob" in r:
            v = probs.get(r["prob"])
            hit = v is not None and v >= r["min"]
        else:
            v = findings.get(r["value"])
            hit = v is not None and ((v >= r["min"]) if "min" in r else (v < r["max"]))
        if hit:
            out.append({"flag": name, "value": v, "rule": r.get("text", "")})
    return out
