"""Echo clip preprocessing for the wall-motion model (identical in training, validation and serving).

  1. decode: multi-frame DICOM (preferred: frame rate and pixel spacing), or AVI / MP4 (needs opencv)
  2. refuse colour Doppler / colour overlays (the model is trained on greyscale B-mode)
  3. greyscale, crop to the moving ultrasound sector, pad square, resize to SIZE x SIZE
  4. resample in time to FPS frames per second (nearest frame)
  5. clips of CLIP_FRAMES frames (CLIP_FRAMES / FPS = 1.6 s, at least one heartbeat), up to MAX_CLIPS evenly
     spaced through the cine; each clip is standardised (zero mean, unit variance)

Apical views are displayed apex up (the usual convention). A cine stored apex-down is flipped with
`flip_vertical` (DICOM metadata can't tell; the view check in pipeline.py reports a likely flip).
"""
from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import zoom

PREPROCESS_VERSION = "1.0.0"
SIZE = 112
FPS = 20.0
CLIP_FRAMES = 32
MAX_CLIPS = 3
MIN_FRAMES = CLIP_FRAMES
COLOUR_MAX = 0.02            # share of strongly coloured pixels above which the cine is refused


class EchoInputError(ValueError):
    """The cine can't be analysed as trained (colour Doppler, too short, unreadable)."""


@dataclass
class Video:
    frames: np.ndarray                     # (T, H, W) uint8 greyscale, or (T, H, W, 3) before greying
    fps: Optional[float] = None
    px_cm: Optional[float] = None          # cm per pixel (DICOM SequenceOfUltrasoundRegions PhysicalDeltaX)
    view: Optional[str] = None             # stated or tagged view (A4C / A2C / A3C)
    notes: List[str] = field(default_factory=list)


def _is_dicom(data: bytes) -> bool:
    return len(data) > 132 and data[128:132] == b"DICM"


def decode_dicom(data: bytes) -> Video:
    import pydicom
    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    arr = ds.pixel_array
    notes: List[str] = []
    n = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if n <= 1:
        raise EchoInputError("single-frame DICOM: a cine (multi-frame) is needed")
    if arr.ndim == 3 and getattr(ds, "SamplesPerPixel", 1) == 3 and arr.shape[-1] != 3:
        arr = arr[None]
    if str(getattr(ds, "PhotometricInterpretation", "")).startswith("YBR"):
        try:
            from pydicom.pixel_data_handlers.util import convert_color_space
            arr = convert_color_space(arr, ds.PhotometricInterpretation, "RGB")
        except Exception:  # noqa: BLE001
            notes.append("YBR colour space not converted")
    fps = None
    for tag in ("RecommendedDisplayFrameRate", "CineRate"):
        v = getattr(ds, tag, None)
        if v:
            fps = float(v)
            break
    if fps is None and getattr(ds, "FrameTime", None):
        fps = 1000.0 / float(ds.FrameTime)
    px = None
    for r in getattr(ds, "SequenceOfUltrasoundRegions", []) or []:
        if int(getattr(r, "RegionSpatialFormat", 0) or 0) == 1 and getattr(r, "PhysicalDeltaX", None):
            if int(getattr(r, "PhysicalUnitsXDirection", 3) or 3) == 3:      # 3 = cm
                px = float(r.PhysicalDeltaX)
                break
    return Video(frames=np.asarray(arr), fps=fps, px_cm=px, view=view_from_tags(ds), notes=notes)


def view_from_tags(ds) -> Optional[str]:
    """View from DICOM text tags when the scanner wrote it (often absent on echo)."""
    import re
    texts = [str(getattr(ds, t, "") or "") for t in ("SeriesDescription", "ImageComments", "ProtocolName")]
    for item in getattr(ds, "ViewCodeSequence", []) or []:
        texts.append(str(getattr(item, "CodeMeaning", "") or ""))
    t = " ".join(texts).lower()
    for view, pat in (("A4C", r"(a4c|ap4|apical\s*4|4\s*-?ch|four.?chamber)"),
                      ("A2C", r"(a2c|ap2|apical\s*2|2\s*-?ch|two.?chamber)"),
                      ("A3C", r"(a3c|ap3|apical\s*3|3\s*-?ch|three.?chamber|aplax|apical long)")):
        if re.search(pat, t):
            return view
    return None


def decode_video(data: bytes) -> Video:
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise EchoInputError("AVI / MP4 needs opencv-python-headless; send DICOM instead") from e
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr[..., ::-1])
        cap.release()
    finally:
        os.unlink(path)
    if not frames:
        raise EchoInputError("could not decode the video")
    return Video(frames=np.stack(frames), fps=float(fps) if fps else None, notes=["no pixel spacing (not DICOM)"])


def decode(data: bytes) -> Video:
    return decode_dicom(data) if _is_dicom(data) else decode_video(data)


def colour_fraction(frames: np.ndarray) -> float:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        return 0.0
    f = frames[:: max(1, len(frames) // 8)].astype(np.int16)
    spread = f.max(-1) - f.min(-1)
    return float((spread > 40).mean())


def grey(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 4:
        frames = frames[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    return np.clip(frames, 0, 255).astype(np.uint8)


def sector_box(frames: np.ndarray) -> Tuple[int, int, int, int]:
    """Bounding box of the moving ultrasound sector (pixels that change over time); whole frame if unclear."""
    sd = frames[:: max(1, len(frames) // 32)].astype(np.float32).std(0)
    mask = sd > max(2.0, np.percentile(sd, 60))
    if mask.mean() < 0.02:
        return 0, frames.shape[1], 0, frames.shape[2]
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def to_square(frames: np.ndarray, size: int = SIZE) -> Tuple[np.ndarray, float]:
    """Crop to the sector, pad square (centred), resize -> (T, size, size) float32 in [0, 1]; returns the scale."""
    r0, r1, c0, c1 = sector_box(frames)
    f = frames[:, r0:r1, c0:c1]
    h, w = f.shape[1:]
    s = max(h, w)
    pad = np.zeros((f.shape[0], s, s), np.uint8)
    pad[:, (s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = f
    k = size / s
    out = zoom(pad.astype(np.float32), (1, k, k), order=1)[:, :size, :size]
    return out / 255.0, k


def resample(frames: np.ndarray, fps: Optional[float], notes: List[str]) -> np.ndarray:
    if not fps or fps <= 0:
        notes.append(f"frame rate unknown: assumed {FPS:.0f} fps")
        return frames
    n = int(np.floor(len(frames) * FPS / fps))
    idx = np.clip(np.round(np.arange(n) * fps / FPS).astype(int), 0, len(frames) - 1)
    return frames[idx]


def clip_starts(n: int) -> List[int]:
    if n < CLIP_FRAMES:
        return []
    last = n - CLIP_FRAMES
    k = min(MAX_CLIPS, 1 + last // (CLIP_FRAMES // 2))
    return sorted({int(round(x)) for x in np.linspace(0, last, k)})


def standardise(clip: np.ndarray) -> np.ndarray:
    return ((clip - clip.mean()) / (clip.std() + 1e-6)).astype(np.float32)


@dataclass
class Prepared:
    clips: np.ndarray            # (K, CLIP_FRAMES, SIZE, SIZE) float32, standardised
    frames: np.ndarray           # (T, SIZE, SIZE) float32 in [0, 1] at FPS
    px_cm: Optional[float]       # cm per pixel AFTER resizing (None when unknown)
    view: Optional[str]
    notes: List[str]


def prepare(v: Video, flip_vertical: bool = False) -> Prepared:
    notes = list(v.notes)
    if colour_fraction(v.frames) > COLOUR_MAX:
        raise EchoInputError("colour Doppler or a colour overlay detected: send the greyscale B-mode cine")
    g = grey(v.frames)
    sq, k = to_square(g)
    if flip_vertical:
        sq = sq[:, ::-1].copy()
        notes.append("flipped vertically (apex up)")
    fr = resample(sq, v.fps, notes)
    starts = clip_starts(len(fr))
    if not starts:
        raise EchoInputError(f"cine too short: {len(fr)} frames at {FPS:.0f} fps; need {CLIP_FRAMES} "
                             f"({CLIP_FRAMES / FPS:.1f} s, at least one heartbeat)")
    clips = np.stack([standardise(fr[s:s + CLIP_FRAMES]) for s in starts])
    return Prepared(clips=clips, frames=fr.astype(np.float32), px_cm=(v.px_cm / k) if v.px_cm else None,
                    view=v.view, notes=notes)
