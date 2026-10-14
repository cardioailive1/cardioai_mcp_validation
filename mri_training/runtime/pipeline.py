"""One cardiac MRI study -> segmentation, measurements, study probabilities, flags.

Used unchanged by evaluation in train.py, the Phase 2 validation study and the serving container,
so the numbers clinicians see are produced by the code that was validated.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .preprocess import Stack, model_input, prepare
from .quantify import cine_function, lge_measures, red_flags


@torch.no_grad()
def segment(model, slices: np.ndarray, kind: str, device: str = "cpu"):
    """(S, 192, 192) prepared slices -> (labels (S, 192, 192) int64, study probabilities (n_heads,))."""
    seg, study = model(torch.from_numpy(model_input(slices, kind)).to(device))
    return seg.argmax(1).cpu().numpy().astype(np.int64), torch.sigmoid(study).cpu().numpy()


def analyse(model, lge: Stack, cine: Optional[Stack] = None, device: str = "cpu") -> Dict[str, object]:
    """LGE (required) + cine (optional) short-axis stacks -> findings and study-level probabilities."""
    model.eval()
    notes: List[str] = list(lge.notes)
    pl = prepare(lge)
    lab, p = segment(model, pl.images, "lge", device)
    findings: Dict[str, object] = lge_measures(lab, pl.slice_spacing_mm, pl.row_cos, pl.col_cos)
    cine_lab = None
    if cine is not None:
        pc = prepare(cine)
        notes += list(cine.notes)
        cine_lab = np.stack([segment(model, pc.images[ph], "cine", device)[0] for ph in range(pc.images.shape[0])])
        findings.update(cine_function(cine_lab, pc.slice_spacing_mm))
    else:
        findings.update({"lvedv_ml": None, "lvesv_ml": None, "lvef_pct": None})
        notes.append("no short-axis cine: LV volumes and LVEF not measured")
    if findings.get("segments_note"):
        notes.append(str(findings["segments_note"]))
    return {"findings": findings, "probabilities": {h: float(v) for h, v in zip(model.study_heads, p)},
            "lge_labels": lab, "cine_labels": cine_lab, "notes": notes}


def interpret(analysis: Dict[str, object], thresholds: Dict[str, float], rules: Dict[str, Dict]) -> Dict[str, object]:
    """Locked thresholds -> yes/no per study finding; locked clinical rules -> red flags for follow-up."""
    probs = analysis["probabilities"]
    return {"flags": {h: bool(probs[h] >= thresholds[h]) for h in thresholds},
            "red_flags": red_flags(analysis["findings"], probs, rules)}
