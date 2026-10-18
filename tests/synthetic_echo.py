"""Synthetic apical echo cines with known segmental wall motion (tests only).

The LV is an inverted U (apex up) inside an ultrasound sector. Each point of the endocardium moves towards the
cavity centre in systole by an amount set by its segment's score (1 normal, 2 hypokinetic, 3 akinetic,
4 dyskinetic: moves outward). Views differ by which AHA segments sit on the left and right walls
(echo_wm_training.runtime.segments.VIEW_SEGMENTS) and by a landmark: A4C has an RV cavity on the left, A3C an
aortic root on the right, A2C neither. Reports are generated in words and parsed with the real parser, so the
labels a test trains on are exactly what labels.parse_report produces.
"""
from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

import numpy as np

from echo_wm_training.labels import parse_report
from echo_wm_training.runtime.segments import VIEW_SEGMENTS

H = W = 96
AMP = {1: 1.0, 2: 0.4, 3: 0.0, 4: -0.35}

REPORTS = {
    "normal": "Normal left ventricular size. No regional wall motion abnormalities. LVEF {ef}%.",
    "lad": "The mid and apical anterior wall, anteroseptum and apex are akinetic. Ejection fraction {ef}%.",
    "rca": "Hypokinesis of the basal and mid inferior wall and basal inferoseptum. LVEF {ef}%.",
    "lcx": "Hypokinesis of the basal and mid inferolateral and anterolateral walls. LVEF {ef}%.",
    "global": "Severe global hypokinesis. LVEF {ef}%.",
}
EF = {"normal": 62, "lad": 38, "rca": 50, "lcx": 48, "global": 25}


def _fill(poly: np.ndarray, h: int = H, w: int = W) -> np.ndarray:
    """Even-odd polygon fill. poly (N, 2) as (row, col)."""
    yy, xx = np.mgrid[0:h, 0:w]
    inside = np.zeros((h, w), bool)
    r0, c0 = poly[:, 0], poly[:, 1]
    r1, c1 = np.roll(r0, -1), np.roll(c0, -1)
    for a, b, c, d in zip(r0, c0, r1, c1):
        cond = (a > yy) != (c > yy)
        with np.errstate(divide="ignore", invalid="ignore"):
            xcross = (d - b) * (yy - a) / (c - a + 1e-9) + b
        inside ^= cond & (xx < xcross)
    return inside


def _slot(u: np.ndarray) -> np.ndarray:
    edges = [0.12, 0.28, 0.42, 0.58, 0.72, 0.88]
    return np.searchsorted(edges, u)


def render(view: str, scores: Dict[int, int], n_frames: int = 48, fps: float = 25.0, rng: Optional[np.random.Generator] = None,
           hr_bpm: float = 70.0) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    cy, cx, a, b, wall = 78.0, 48.0, 50.0, 18.0, 6.0
    n = 120
    u = np.linspace(0, 1, n)
    phi = np.pi * (1 - u)
    slot_seg = VIEW_SEGMENTS.get(view, [17] * 7)
    amp = np.array([AMP[scores.get(slot_seg[s], 1)] for s in _slot(u)])
    amp = np.convolve(np.pad(amp, 6, mode="edge"), np.ones(13) / 13, mode="valid")
    yy, xx = np.mgrid[0:H, 0:W]
    sector = (yy > 4) & (np.abs(xx - W / 2) < (yy - 4) * 0.62)
    period = 60.0 / hr_bpm
    frames = np.zeros((n_frames, H, W), np.uint8)
    phase0 = rng.uniform(0, 1)
    centre = np.array([cy - a * 0.45, cx])
    for t in range(n_frames):
        c = 0.5 * (1 - np.cos(2 * np.pi * (t / fps / period + phase0)))       # 0 at ED, 1 at ES
        endo = np.stack([cy - a * np.sin(phi), cx + b * np.cos(phi)], 1)
        endo = endo + (centre - endo) * (0.28 * amp * c)[:, None]
        epi0 = np.stack([cy - (a + wall) * np.sin(phi), cx + (b + wall) * np.cos(phi)], 1)
        epi = epi0 + (centre - epi0) * (0.12 * amp * c)[:, None]
        cav = _fill(np.vstack([endo, [[cy + 2, cx + b], [cy + 2, cx - b]]]))
        outer = _fill(np.vstack([epi, [[cy + 2, cx + b + wall], [cy + 2, cx - b - wall]]]))
        img = rng.normal(45, 12, (H, W))
        img[outer] = rng.normal(170, 25, outer.sum())
        img[cav] = rng.normal(12, 5, cav.sum())
        if view == "A4C":
            rv = ((yy - (cy - 18)) / 30) ** 2 + ((xx - (cx - b - 20)) / 9) ** 2 < 1
            img[rv] = rng.normal(12, 5, rv.sum())
        elif view == "A3C":
            ao = ((yy - (cy + 8)) / 7) ** 2 + ((xx - (cx + b + 4)) / 7) ** 2 < 1
            img[ao] = rng.normal(12, 5, ao.sum())
        elif view == "other":
            ring = np.abs(np.hypot(yy - 50, xx - 48) - 18) < 6
            img = rng.normal(45, 12, (H, W))
            img[ring] = rng.normal(170, 25, ring.sum())
        img[~sector] = 0
        frames[t] = np.clip(img, 0, 255).astype(np.uint8)
    return frames


def study(kind: str, seed: int = 0, views=("A4C", "A2C", "A3C"), n_frames: int = 48, fps: float = 25.0):
    """-> (report text, parsed labels, {view: frames})."""
    rng = np.random.default_rng(seed)
    ef = EF[kind] + int(rng.integers(-3, 4))
    text = REPORTS[kind].format(ef=ef)
    lab = parse_report(text)
    scores = {s: int(lab[f"seg_{s}"]) + 1 for s in range(1, 18)}
    clips = {v: render(v, scores, n_frames, fps, rng, hr_bpm=float(rng.uniform(60, 90))) for v in views}
    return text, lab, clips


def dicom_bytes(frames: np.ndarray, fps: float = 25.0, view_tag: Optional[str] = None, px_cm: float = 0.15,
                colour: bool = False) -> bytes:
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    from synthetic_cmr import _save
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.3.1"          # Ultrasound Multi-frame Image Storage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Modality = "US"
    ds.SeriesDescription = view_tag or ""
    ds.NumberOfFrames = len(frames)
    ds.FrameTime = 1000.0 / fps
    ds.Rows, ds.Columns = frames.shape[1:]
    if colour:
        rgb = np.repeat(frames[..., None], 3, -1).copy()
        rgb[:, 30:50, 30:50, 0] = 250
        rgb[:, 30:50, 30:50, 2] = 10
        ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.PlanarConfiguration = 3, "RGB", 0
        data = rgb
    else:
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        data = frames
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    r = Dataset()
    r.RegionSpatialFormat, r.RegionDataType = 1, (2 if colour else 1)
    r.PhysicalUnitsXDirection, r.PhysicalUnitsYDirection = 3, 3
    r.PhysicalDeltaX, r.PhysicalDeltaY = px_cm, px_cm
    ds.SequenceOfUltrasoundRegions = Sequence([r])
    ds.PixelData = np.ascontiguousarray(data, dtype=np.uint8).tobytes()
    buf = io.BytesIO()
    _save(ds, buf)
    return buf.getvalue()


def dataset(n: int, seed: int = 0, kinds=("normal", "lad", "rca", "lcx", "global"), views=("A4C", "A2C", "A3C"),
            n_frames: int = 40) -> List[Tuple[str, dict, Dict[str, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    return [study(str(rng.choice(kinds)), seed=seed * 1000 + i, views=views, n_frames=n_frames) for i in range(n)]
