"""Image geometry for annotation files: DICOM stacks <-> NIfTI, model grid <-> native grid.

Readers draw on the NATIVE images (original pixel size and slice order along the stack normal), in
NIfTI files any annotation tool opens with the right orientation (3D Slicer, ITK-SNAP, MONAI Label,
OHIF via conversion). The model works on its own 1.25 mm, 192 x 192 grid; unprepare_labels() maps its
pre-drawn outlines back to the native grid (inverse of runtime.preprocess._fit).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import zoom

from mri_training.runtime.preprocess import PIXEL_MM, SIZE, Stack, order_slices

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])


def stack_affine(stack: Stack) -> Tuple[np.ndarray, np.ndarray]:
    """(NIfTI RAS affine for data[x=column, y=row, z=sorted slice], sorted slice order)."""
    order, spacing, r, c = order_slices(stack)
    pos = np.asarray(stack.positions, float)[order[0]]
    dr, dc = float(stack.pixel_spacing[0]), float(stack.pixel_spacing[1])     # row spacing, column spacing
    n = np.cross(r, c)
    lps = np.eye(4)
    lps[:3, 0] = r * dc                      # +x = next column
    lps[:3, 1] = c * dr                      # +y = next row
    lps[:3, 2] = n * spacing                 # +z = next slice along the normal
    lps[:3, 3] = pos
    ras = np.eye(4)
    ras[:3, :] = LPS_TO_RAS @ lps[:3, :]
    return ras, order


def to_nifti_array(slices: np.ndarray, order: np.ndarray) -> np.ndarray:
    """(S, H, W) in stack order -> (W, H, S) in sorted order (NIfTI x, y, z)."""
    return np.transpose(np.asarray(slices)[order], (2, 1, 0))


def from_nifti_array(vol: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Inverse of to_nifti_array: (W, H, S) sorted -> (S, H, W) in the stack's own order."""
    s = np.transpose(np.asarray(vol), (2, 1, 0))
    out = np.empty_like(s)
    out[order] = s
    return out


def unprepare_labels(labels: np.ndarray, native_hw: Tuple[int, int], pixel_spacing: Tuple[float, float]) -> np.ndarray:
    """Model-grid labels (S, 192, 192), slices in SORTED order -> native grid (S, H, W), nearest neighbour."""
    h0, w0 = native_hw
    fy, fx = pixel_spacing[0] / PIXEL_MM, pixel_spacing[1] / PIXEL_MM
    # shape _fit produced after zooming the native image
    probe = zoom(np.zeros((h0, w0), np.int16), (fy, fx), order=0, prefilter=False)
    h, w = probe.shape
    sy, sx = max(0, (h - SIZE) // 2), max(0, (w - SIZE) // 2)
    ch, cw = min(SIZE, h), min(SIZE, w)
    oy, ox = (SIZE - ch) // 2, (SIZE - cw) // 2
    out = []
    for lab in np.asarray(labels):
        big = np.zeros((h, w), np.int16)
        big[sy:sy + ch, sx:sx + cw] = lab[oy:oy + ch, ox:ox + cw]
        back = zoom(big, (h0 / h, w0 / w), order=0, prefilter=False)
        fixed = np.zeros((h0, w0), np.int16)                     # zoom can be off by one pixel at the edge
        fixed[:min(h0, back.shape[0]), :min(w0, back.shape[1])] = back[:h0, :w0]
        out.append(fixed)
    return np.stack(out).astype(np.int64)
