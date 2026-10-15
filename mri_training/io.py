"""Read cardiac MRI studies: DICOM series selection (LGE and cine short axis) and public NIfTI sets.

A CMR study holds many series (localisers, long-axis cines, T1/T2 maps, LGE in several planes).
select_series() picks:
  * LGE short axis : single image per slice position, InversionTime set (or LGE/PSIR/delayed in the
                     description), >= 3 parallel slices, not an axial stack
                     (PSIR series storing magnitude + real images: the real/PSIR images are used)
  * cine short axis: several phases per slice position (CardiacNumberOfImages / TriggerTime),
                     description cine/SSFP/FIESTA/TrueFISP or many phases, >= 3 parallel slices
Short axis is recognised by the description (sa / sax / short) or, failing that, by excluding axial,
sagittal and coronal stacks; the choice and the reason are reported in Stack.notes.
"""
from __future__ import annotations

import io as _io
import re
import zipfile
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .runtime.preprocess import MRIInputError, Stack

LGE_RE = re.compile(r"(?i)(lge|psir|delayed|late|\bde\b|mde|viab|phase.?sensitive)")
CINE_RE = re.compile(r"(?i)(cine|ssfp|fiesta|true.?fisp|trufi|bffe|balanced)")
SAX_RE = re.compile(r"(?i)(\bsa\b|sax|short|_sa_|sa_)")


def _read_datasets(files: Iterable) -> List:
    import pydicom
    out = []
    for f in files:
        try:
            ds = pydicom.dcmread(_io.BytesIO(f) if isinstance(f, (bytes, bytearray)) else f)
            if hasattr(ds, "PixelData"):
                out.append(ds)
        except Exception:  # noqa: BLE001  (non-DICOM files in a study folder are skipped)
            continue
    return out


def read_zip(data: bytes) -> List:
    with zipfile.ZipFile(_io.BytesIO(data)) as z:
        return _read_datasets(z.read(n) for n in z.namelist() if not n.endswith("/"))


def _plane(ds) -> str:
    o = np.asarray(ds.ImageOrientationPatient, float)
    n = np.abs(np.cross(o[:3], o[3:]))
    if n.max() > 0.95:
        return ["sagittal", "coronal", "axial"][int(np.argmax(n))]
    return "oblique"


def _stack(series: List, kind: str, desc: str, reason: str) -> Stack:
    o = [float(v) for v in series[0].ImageOrientationPatient]
    n = np.cross(o[:3], o[3:])
    by_pos: Dict[float, List] = defaultdict(list)
    for ds in series:
        by_pos[round(float(np.dot(np.asarray(ds.ImagePositionPatient, float), n)), 1)].append(ds)
    keys = sorted(by_pos)
    sp = [float(v) for v in series[0].PixelSpacing]
    if kind == "lge":
        imgs, pos = [], []
        for k in keys:
            cands = by_pos[k]
            real = [d for d in cands if any(t in [str(x).upper() for x in getattr(d, "ImageType", [])] for t in ("PSIR", "REAL", "R"))]
            d = (real or cands)[0]
            imgs.append(d.pixel_array.astype(np.float64) * float(getattr(d, "RescaleSlope", 1) or 1) + float(getattr(d, "RescaleIntercept", 0) or 0))
            pos.append([float(v) for v in d.ImagePositionPatient])
        images = np.stack(imgs)
    else:
        per = [sorted(by_pos[k], key=lambda d: float(getattr(d, "TriggerTime", 0) or 0)) for k in keys]
        n_ph = min(len(p) for p in per)
        images = np.stack([[per[s][ph].pixel_array.astype(np.float64) for s in range(len(keys))] for ph in range(n_ph)])
        pos = [[float(v) for v in per[s][0].ImagePositionPatient] for s in range(len(keys))]
    return Stack(images, (sp[0], sp[1]), pos, o, float(getattr(series[0], "SliceThickness", 0) or 0) or None, kind,
                 [f"{kind}: series '{desc}' ({reason})"])


def select_series(datasets: List) -> Tuple[Optional[Stack], Optional[Stack]]:
    """(LGE short-axis stack, cine short-axis stack); either may be None."""
    groups: Dict[str, List] = defaultdict(list)
    for ds in datasets:
        groups[str(getattr(ds, "SeriesInstanceUID", "?"))].append(ds)
    lge, cine = [], []
    for uid, s in groups.items():
        desc = str(getattr(s[0], "SeriesDescription", ""))
        try:
            plane = _plane(s[0])
        except Exception:  # noqa: BLE001
            continue
        n_pos = len({round(float(np.dot(np.asarray(d.ImagePositionPatient, float), np.cross(*np.split(np.asarray(d.ImageOrientationPatient, float), 2)))), 1) for d in s})
        if n_pos < 3 or plane in ("axial", "sagittal", "coronal"):
            continue
        sax = bool(SAX_RE.search(desc))
        phases = len(s) / n_pos
        has_ti = any(getattr(d, "InversionTime", None) not in (None, "") for d in s)
        if phases >= 6 and (CINE_RE.search(desc) or phases >= 15):
            cine.append((sax, n_pos, uid, desc))
        elif phases <= 2.5 and (has_ti or LGE_RE.search(desc)):
            lge.append((sax, n_pos, uid, desc))
    pick = lambda c: sorted(c, key=lambda t: (t[0], t[1]), reverse=True)[0] if c else None  # noqa: E731
    lg, ci = pick(lge), pick(cine)
    return (_stack(groups[lg[2]], "lge", lg[3], "short axis by description" if lg[0] else "oblique stack, short axis assumed") if lg else None,
            _stack(groups[ci[2]], "cine", ci[3], "short axis by description" if ci[0] else "oblique stack, short axis assumed") if ci else None)


def read_folder(path: str) -> List:
    """Every file under a folder (how MCP delivers de-identified DICOM: one folder per study or series)."""
    from pathlib import Path
    return _read_datasets(str(p) for p in sorted(Path(path).rglob("*")) if p.is_file())


def read_study_path(path: str) -> Tuple[Stack, Optional[Stack]]:
    """A study given as a folder of DICOM files or as a zip."""
    import os
    if os.path.isdir(path):
        lge, cine = select_series(read_folder(path))
        if lge is None:
            raise MRIInputError("no short-axis LGE series found in the study")
        return lge, cine
    with open(path, "rb") as f:
        return read_study(f.read())


def read_study(data: bytes) -> Tuple[Stack, Optional[Stack]]:
    """Zip of a study's DICOM files -> (LGE stack, cine stack or None). LGE is required."""
    lge, cine = select_series(read_zip(data))
    if lge is None:
        raise MRIInputError("no short-axis LGE series found in the study")
    return lge, cine


# --------------------------------------------------------------------------- public annotated datasets
def read_nifti_pair(image_path: str, label_path: str, label_map: Dict[int, int], kind: str) -> Tuple[Stack, np.ndarray]:
    """Public sets (EMIDEC, MyoPS, ACDC) as NIfTI: volume (X, Y, Z) -> Stack + labels in CLASSES ids.

    label_map: dataset label value -> our class id; values not listed become background (0).
    NIfTI carries no DICOM orientation, so AHA segments are not computed for public data (not needed
    for segmentation training).
    """
    import nibabel as nib
    img, lab = nib.load(image_path), nib.load(label_path)
    vol, seg = np.asarray(img.get_fdata()), np.asarray(lab.get_fdata()).astype(int)
    zooms = img.header.get_zooms()[:3]
    vol, seg = np.moveaxis(vol, 2, 0), np.moveaxis(seg, 2, 0)            # (Z, X, Y)
    mapped = np.zeros_like(seg)
    for src, dst in label_map.items():
        mapped[seg == int(src)] = int(dst)
    pos = [[0.0, 0.0, float(z) * float(zooms[2])] for z in range(vol.shape[0])]
    stack = Stack(vol, (float(zooms[0]), float(zooms[1])), pos, [1, 0, 0, 0, 1, 0], float(zooms[2]), kind,
                  [f"public NIfTI {image_path}"])
    return stack, mapped
