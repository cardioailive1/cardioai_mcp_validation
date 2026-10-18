"""AHA 17-segment model, apical views, coronary territories, and the wall-motion summaries built on them.

Segments (Cerqueira et al., AHA 2002; ASE/EACVI chamber quantification, Lang et al. 2015):
  basal  1 anterior  2 anteroseptal  3 inferoseptal  4 inferior  5 inferolateral  6 anterolateral
  mid    7 anterior  8 anteroseptal  9 inferoseptal 10 inferior 11 inferolateral 12 anterolateral
  apical 13 anterior 14 septal 15 inferior 16 lateral          17 apex (apical cap)

Each apical view shows 7 segments, listed here from the basal segment on the LEFT of the standard display
(apex at the top), up the left wall, the apex, and down the right wall to its basal segment. The model has
one spatial "slot" per position. A segment seen in more than one view is averaged over the views that show it.

Territories follow the AHA standard assignment; individual coronary anatomy varies (dominance, wrap-around
LAD), so territory output is "consistent with" a territory, never a coronary diagnosis.

Wall-motion scores (ASE): 1 normal / hyperkinetic, 2 hypokinetic, 3 akinetic, 4 dyskinetic or aneurysmal.
The model predicts classes 0..3 = scores 1..4. WMSI = mean score over the assessed segments.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

SEGMENTS: Dict[int, str] = {
    1: "basal anterior", 2: "basal anteroseptal", 3: "basal inferoseptal", 4: "basal inferior",
    5: "basal inferolateral", 6: "basal anterolateral", 7: "mid anterior", 8: "mid anteroseptal",
    9: "mid inferoseptal", 10: "mid inferior", 11: "mid inferolateral", 12: "mid anterolateral",
    13: "apical anterior", 14: "apical septal", 15: "apical inferior", 16: "apical lateral", 17: "apex",
}
N_SEG = 17
SCORES = ["normal", "hypokinetic", "akinetic", "dyskinetic"]       # class k -> ASE score k + 1
N_SCORE = len(SCORES)

VIEWS = ["A4C", "A2C", "A3C"]
VIEW_CLASSES = VIEWS + ["other"]
# 7 slots per view: basal-left, mid-left, apical-left, apex, apical-right, mid-right, basal-right
VIEW_SEGMENTS: Dict[str, List[int]] = {
    "A4C": [3, 9, 14, 17, 16, 12, 6],     # inferoseptum (left) .. anterolateral (right)
    "A2C": [4, 10, 15, 17, 13, 7, 1],     # inferior (left) .. anterior (right)
    "A3C": [5, 11, 16, 17, 13, 8, 2],     # inferolateral (left) .. anteroseptum (right); CONFIRM apical pair with the clinical lead
}
N_SLOT = 7

TERRITORIES: Dict[str, List[int]] = {
    "LAD": [1, 2, 7, 8, 13, 14, 17],
    "RCA": [3, 4, 9, 10, 15],
    "LCx": [5, 6, 11, 12, 16],
}


def slot_index() -> np.ndarray:
    """(n_views, 7) -> segment index 0..16 (for gathering per-view slot logits into segments)."""
    return np.array([[s - 1 for s in VIEW_SEGMENTS[v]] for v in VIEWS], dtype=np.int64)


def assessed(views: Iterable[str]) -> np.ndarray:
    """Boolean (17,) of the segments visible in the given views."""
    a = np.zeros(N_SEG, bool)
    for v in views:
        for s in VIEW_SEGMENTS[v]:
            a[s - 1] = True
    return a


def summarise(seg_prob: np.ndarray, seen: np.ndarray) -> Dict[str, object]:
    """seg_prob (17, 4) class probabilities, seen (17,) bool -> per-segment scores, WMSI, territory summaries.

    Territory abnormal probability = the highest P(abnormal) among its assessed segments; a territory with no
    assessed segment is reported as not assessed (None)."""
    p_abn = 1.0 - seg_prob[:, 0]
    cls = seg_prob.argmax(1)
    segs: Dict[str, Dict[str, object]] = {}
    for i in range(N_SEG):
        if seen[i]:
            segs[str(i + 1)] = {"name": SEGMENTS[i + 1], "score": int(cls[i]) + 1, "label": SCORES[int(cls[i])],
                                "p_abnormal": round(float(p_abn[i]), 4),
                                "probabilities": {SCORES[k]: round(float(seg_prob[i, k]), 4) for k in range(N_SCORE)}}
    scores = np.array([cls[i] + 1 for i in range(N_SEG) if seen[i]], float)
    terr: Dict[str, Dict[str, object]] = {}
    for t, members in TERRITORIES.items():
        idx = [s - 1 for s in members if seen[s - 1]]
        if not idx:
            terr[t] = {"assessed_segments": 0, "p_abnormal": None, "abnormal_segments": []}
            continue
        terr[t] = {"assessed_segments": len(idx), "p_abnormal": round(float(p_abn[idx].max()), 4),
                   "abnormal_segments": [s + 1 for s in idx if cls[s] > 0],
                   "worst_score": int(cls[idx].max()) + 1}
    return {"segments": segs, "assessed_segments": int(seen.sum()),
            "wmsi": round(float(scores.mean()), 2) if len(scores) else None,
            "n_abnormal_segments": int(sum(cls[i] > 0 for i in range(N_SEG) if seen[i])),
            "territories": terr}


def territory_labels(seg_scores: Dict[int, Optional[int]]) -> Dict[str, float]:
    """Report segment scores (1..4 or None) -> territory reference labels 1 / 0 / NaN, wma_any (any abnormal segment, regional or global) and WMSI."""
    out: Dict[str, float] = {}
    for t, members in TERRITORIES.items():
        vals = [seg_scores.get(s) for s in members]
        known = [v for v in vals if v is not None]
        if any(v > 1 for v in known):
            out[f"terr_{t.lower()}"] = 1.0
        elif known and len(known) == len(vals):
            out[f"terr_{t.lower()}"] = 0.0
        else:
            out[f"terr_{t.lower()}"] = float("nan")
    allv = [seg_scores.get(s) for s in SEGMENTS]
    known = [v for v in allv if v is not None]
    out["wma_any"] = 1.0 if any(v > 1 for v in known) else (0.0 if len(known) == N_SEG else float("nan"))
    out["wmsi"] = float(np.mean(known)) if len(known) == N_SEG else float("nan")
    return out
