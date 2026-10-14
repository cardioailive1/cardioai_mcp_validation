"""Cardiac MRI short-axis preprocessing shared by training, the validation study and serving.

Part of the model's input contract (with model.py and quantify.py); locked by runtime.fingerprint().

Input  : one short-axis stack per sequence, as slices with their DICOM geometry
         (LGE: S slices; cine: P phases x S slices)
Steps  : 1. order slices along the stack normal (row x column direction cosines)
         2. resample in-plane to 1.25 mm pixels (bilinear for images, nearest for labels)
         3. centre crop / zero-pad to 192 x 192 (240 mm field of view)
         4. per-slice intensity scaling: 1st-99.9th percentile -> [0, 1] (MR intensities are arbitrary;
            works for magnitude and PSIR LGE alike). The upper bound is 99.9, not 99: bright scar, blood
            pool and fat are a few % of the field of view, and a 99th-percentile clip would flatten
            scar and blood pool to the same value (the contrast LGE exists to show).
Output : float32 (S, 192, 192) or (P, S, 192, 192) plus geometry needed for quantification
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import zoom

PREPROCESS_VERSION = "1.0.0"
PIXEL_MM = 1.25
SIZE = 192
LGE, CINE = 1.0, 0.0            # value of the sequence channel fed to the model


class MRIInputError(ValueError):
    """The study cannot be prepared the way the model was trained (client error)."""


@dataclass
class Stack:
    """One short-axis stack. images: (S, H, W) or (P, S, H, W) for cine."""
    images: np.ndarray
    pixel_spacing: Tuple[float, float]              # (row spacing, column spacing) in mm
    positions: Sequence[Sequence[float]]            # ImagePositionPatient per slice (mm)
    orientation: Sequence[float]                    # ImageOrientationPatient (6 direction cosines)
    slice_thickness: Optional[float] = None
    kind: str = "lge"                               # "lge" | "cine"
    notes: List[str] = field(default_factory=list)


@dataclass
class Prepared:
    images: np.ndarray                              # (S, 192, 192) or (P, S, 192, 192) float32 in [0, 1]
    slice_spacing_mm: float
    row_cos: np.ndarray                             # patient direction of increasing column index
    col_cos: np.ndarray                             # patient direction of increasing row index
    order: np.ndarray                               # original slice index for each output slice
    kind: str
    notes: List[str]


def _normal(orientation: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    o = np.asarray(orientation, float)
    if o.shape != (6,):
        raise MRIInputError("ImageOrientationPatient must have 6 values")
    r, c = o[:3] / np.linalg.norm(o[:3]), o[3:] / np.linalg.norm(o[3:])
    return r, c, np.cross(r, c)


def order_slices(stack: Stack) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    r, c, n = _normal(stack.orientation)
    pos = np.asarray(stack.positions, float)
    n_slices = stack.images.shape[-3]
    if pos.shape != (n_slices, 3):
        raise MRIInputError(f"need one ImagePositionPatient per slice ({n_slices}), got {pos.shape}")
    d = pos @ n
    order = np.argsort(d)
    gaps = np.diff(d[order])
    if n_slices > 1 and (np.any(gaps <= 0.1) or gaps.max() > 3 * np.median(gaps)):
        raise MRIInputError("slices are duplicated or unevenly spaced: not a single short-axis stack")
    spacing = float(np.median(gaps)) if n_slices > 1 else float(stack.slice_thickness or 8.0)
    return order, spacing, r, c


def _fit(img: np.ndarray, spacing: Tuple[float, float], order: int) -> np.ndarray:
    fy, fx = spacing[0] / PIXEL_MM, spacing[1] / PIXEL_MM
    out = zoom(img, (fy, fx), order=order, prefilter=False)
    h, w = out.shape
    canvas = np.zeros((SIZE, SIZE), out.dtype)
    sy, sx = max(0, (h - SIZE) // 2), max(0, (w - SIZE) // 2)
    crop = out[sy:sy + SIZE, sx:sx + SIZE]
    oy, ox = (SIZE - crop.shape[0]) // 2, (SIZE - crop.shape[1]) // 2
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return canvas


def _scale(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99.9])
    if hi - lo < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)


def prepare(stack: Stack) -> Prepared:
    imgs = np.asarray(stack.images, dtype=np.float64)
    if stack.kind not in ("lge", "cine"):
        raise MRIInputError(f"unknown sequence kind {stack.kind!r}")
    if (stack.kind == "lge" and imgs.ndim != 3) or (stack.kind == "cine" and imgs.ndim != 4):
        raise MRIInputError(f"{stack.kind} images must be {'(S,H,W)' if stack.kind == 'lge' else '(P,S,H,W)'}")
    if imgs.shape[-3] < 3:
        raise MRIInputError("a short-axis stack needs at least 3 slices")
    if not np.isfinite(imgs).all():
        raise MRIInputError("images contain NaN or infinite values")
    if min(stack.pixel_spacing) <= 0 or max(stack.pixel_spacing) > 5:
        raise MRIInputError(f"implausible pixel spacing {stack.pixel_spacing} mm")
    order, spacing, r, c = order_slices(stack)
    if stack.kind == "lge":
        out = np.stack([_scale(_fit(imgs[i], stack.pixel_spacing, 1)) for i in order])
    else:
        out = np.stack([np.stack([_scale(_fit(imgs[p, i], stack.pixel_spacing, 1)) for i in order]) for p in range(imgs.shape[0])])
    return Prepared(out, spacing, r, c, order, stack.kind, list(stack.notes))


def prepare_labels(labels: np.ndarray, pixel_spacing: Tuple[float, float], order: np.ndarray) -> np.ndarray:
    """Label maps (S, H, W) -> same grid as prepare() (nearest neighbour)."""
    return np.stack([_fit(labels[i].astype(np.int16), pixel_spacing, 0) for i in order]).astype(np.int64)


def model_input(prepared_slices: np.ndarray, kind: str) -> np.ndarray:
    """(S, 192, 192) -> (S, 2, 192, 192): image + constant sequence channel (1 = LGE, 0 = cine)."""
    seq = np.full_like(prepared_slices, LGE if kind == "lge" else CINE)
    return np.stack([prepared_slices, seq], axis=1).astype(np.float32)
