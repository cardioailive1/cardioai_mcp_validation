"""Read delivered echo studies: which DICOM files are 2-D greyscale cines, and their views when tagged.

An echo study holds 50-150 instances: 2-D cines of many views, colour Doppler, spectral Doppler, M-mode and
stills. Only multi-frame 2-D tissue cines are kept (DICOM SequenceOfUltrasoundRegions: a 2-D tissue region and
no colour-flow region; without regions, a multi-frame greyscale image). The view comes from the tags when the
scanner wrote it, otherwise from the view labels / view model (views.py).

    clips = list_clips("/workspace/echo/dicom/<study_uid>")      # DataFrame: clip_uri, frames, fps, view_tag ...
    video = read_clip(clips.clip_uri[0])                         # runtime.preprocess.Video
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .runtime.preprocess import Video, decode_dicom, view_from_tags


def _header(path_or_bytes) -> Optional[object]:
    import pydicom
    try:
        src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else str(path_or_bytes)
        return pydicom.dcmread(src, stop_before_pixels=True, force=True)
    except Exception:  # noqa: BLE001
        return None


def clip_kind(ds) -> str:
    """'cine' (2-D greyscale tissue cine), or why not: still, colour, doppler, mmode, unknown."""
    n = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if n < 10:
        return "still"
    regions = getattr(ds, "SequenceOfUltrasoundRegions", None) or []
    if regions:
        fmts = {int(getattr(r, "RegionSpatialFormat", 0) or 0) for r in regions}
        types = {int(getattr(r, "RegionDataType", 0) or 0) for r in regions}
        if 2 in types:
            return "colour"
        if 1 not in fmts:
            return "doppler" if 3 in fmts or types & {3, 4} else "mmode" if 2 in fmts else "unknown"
        if fmts & {2, 3}:
            return "mmode" if 2 in fmts else "doppler"
        return "cine"
    if int(getattr(ds, "SamplesPerPixel", 1) or 1) == 3 and str(getattr(ds, "PhotometricInterpretation", "")).startswith(("RGB", "YBR")):
        return "cine"           # many echo cines are stored as RGB greyscale; colour is checked on the pixels later
    return "cine"


def _files(path: str) -> List[Path]:
    p = Path(path)
    return [f for f in sorted(p.rglob("*")) if f.is_file() and not f.name.startswith(".")]


def list_clips(path: str, study_id: Optional[str] = None) -> pd.DataFrame:
    """One row per DICOM file of a study folder: clip_uri, kind, frames, fps, view_tag, sop_uid."""
    rows: List[Dict[str, object]] = []
    for f in _files(path):
        ds = _header(f)
        if ds is None or not hasattr(ds, "SOPClassUID"):
            continue
        fps = None
        for tag in ("RecommendedDisplayFrameRate", "CineRate"):
            if getattr(ds, tag, None):
                fps = float(getattr(ds, tag))
                break
        if fps is None and getattr(ds, "FrameTime", None):
            fps = 1000.0 / float(ds.FrameTime)
        rows.append({"study_id": study_id or Path(path).name, "clip_uri": str(f), "kind": clip_kind(ds),
                     "frames": int(getattr(ds, "NumberOfFrames", 1) or 1), "fps": fps, "view_tag": view_from_tags(ds),
                     "sop_uid": str(getattr(ds, "SOPInstanceUID", ""))})
    return pd.DataFrame(rows, columns=["study_id", "clip_uri", "kind", "frames", "fps", "view_tag", "sop_uid"])


def read_clip(path: str) -> Video:
    return decode_dicom(Path(path).read_bytes())


def read_zip_clips(data: bytes) -> List[Video]:
    """A study zip (serving) -> its 2-D cines as Videos (other instances skipped)."""
    out: List[Video] = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in sorted(z.namelist()):
            if name.endswith("/"):
                continue
            raw = z.read(name)
            ds = _header(raw)
            if ds is None or clip_kind(ds) != "cine":
                continue
            out.append(decode_dicom(raw))
    return out
