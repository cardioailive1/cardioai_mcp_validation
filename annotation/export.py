"""Prepare studies for annotation inside the MCP workspace (Mayo images never leave it).

    python -m annotation.export mri  --worklist worklist.csv --out /workspace/annotation [--bundle runs/prelabel]
    python -m annotation.export echo --worklist echo_worklist.csv --out /workspace/annotation_echo

One folder per reader and study: <out>/<reader>/<study_id>/
  MRI : lge.nii.gz + lge_label.nii.gz, cine_ed / cine_es (.nii.gz + _label.nii.gz), meta.json
  Echo: cine.nii.gz (frames along z) + cine_label.nii.gz, meta.json
Images keep their native pixels and orientation, so 3D Slicer, ITK-SNAP, MONAI Label and OHIF (after
conversion) show them the way the scanner wrote them. With --bundle (weights.pt + model_card.json of a
model trained on the public sets), the label files start with the model's outlines for readers to
correct; otherwise they start empty.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io as _io
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from mri_training.runtime.preprocess import Stack, prepare
from mri_training.runtime.quantify import LV, MVO, MYO, SCAR

from .geometry import stack_affine, to_nifti_array, unprepare_labels


def _save(arr: np.ndarray, affine: np.ndarray, path: Path, dtype) -> str:
    import nibabel as nib
    img = nib.Nifti1Image(np.asarray(arr, dtype=dtype), affine)
    img.header.set_xyzt_units("mm")
    nib.save(img, str(path))
    return hashlib.sha256(np.ascontiguousarray(np.asarray(arr, dtype=dtype)).tobytes()).hexdigest()


def load_prelabel_model(bundle: Optional[str], device: str = "cpu"):
    """(model, provenance) from a bundle folder (weights.pt + model_card.json), or (None, None)."""
    if not bundle:
        return None, None
    from mri_training.runtime.model import load_model
    b = Path(bundle)
    card = json.loads((b / "model_card.json").read_text())
    model = load_model(str(b / "weights.pt"), {**card["arch"], "study_heads": card["study_heads"]}, device)
    return model, {"model_name": card.get("model_name"), "runtime_fingerprint": card.get("runtime_fingerprint"),
                   "weights_sha256": hashlib.sha256((b / "weights.pt").read_bytes()).hexdigest()}


def _model_labels(model, stack: Stack, kind: str, device: str) -> np.ndarray:
    """Model outlines on the native grid, slices in SORTED order: (S, H, W) or (P, S, H, W) for cine."""
    from mri_training.runtime.pipeline import segment
    p = prepare(stack)
    hw = stack.images.shape[-2:]
    if kind == "lge":
        return unprepare_labels(segment(model, p.images, "lge", device)[0], hw, stack.pixel_spacing)
    return np.stack([unprepare_labels(segment(model, p.images[ph], "cine", device)[0], hw, stack.pixel_spacing)
                     for ph in range(p.images.shape[0])])


def export_mri_study(study_id: str, lge: Stack, cine: Optional[Stack], folder: Path, model=None,
                     provenance: Optional[dict] = None, device: str = "cpu", readers=()) -> Dict:
    folder.mkdir(parents=True, exist_ok=True)
    aff, order = stack_affine(lge)
    meta: Dict = {"task": "mri_contours", "study_id": study_id, "readers": list(readers),
                  "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                  "prelabel": provenance, "lge": {"order": [int(i) for i in order], "shape_shw": list(lge.images.shape),
                                                  "series": lge.notes}}
    _save(to_nifti_array(lge.images, order), aff, folder / "lge.nii.gz", np.float32)
    if model is not None:
        lab = _model_labels(model, lge, "lge", device)                      # sorted order already
        lab_nii = np.transpose(lab, (2, 1, 0))
    else:
        lab_nii = np.zeros(to_nifti_array(lge.images, order).shape, np.uint8)
    meta["lge"]["prelabel_sha256"] = _save(lab_nii, aff, folder / "lge_label.nii.gz", np.uint8)
    if cine is not None:
        caff, corder = stack_affine(cine)
        P = cine.images.shape[0]
        if model is not None:
            cl = _model_labels(model, cine, "cine", device)                  # (P, S, H, W) sorted
            cl[np.isin(cl, (SCAR, MVO))] = MYO                               # no scar on cine
            vol = (cl == LV).reshape(P, -1).sum(1)
            ed, es, how = int(vol.argmax()), int(vol.argmin()), "model: largest / smallest LV pool"
        else:
            cl = None
            ed, es, how = 0, int(round(0.35 * P)) % P, "default guess: first phase / 35% of the cycle (reader to check)"
        meta["cine"] = {"order": [int(i) for i in corder], "shape_pshw": list(cine.images.shape), "phases": {"ed": ed, "es": es},
                        "phase_choice": how, "series": cine.notes}
        for name, ph in (("ed", ed), ("es", es)):
            _save(to_nifti_array(cine.images[ph], corder), caff, folder / f"cine_{name}.nii.gz", np.float32)
            lab = np.transpose(cl[ph], (2, 1, 0)) if cl is not None else np.zeros(to_nifti_array(cine.images[ph], corder).shape, np.uint8)
            meta["cine"][f"{name}_prelabel_sha256"] = _save(lab, caff, folder / f"cine_{name}_label.nii.gz", np.uint8)
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# --------------------------------------------------------------------------- echo
def read_echo_cine(data: bytes):
    """A4C ultrasound cine DICOM -> (frames (F, H, W) float32 grey, mm per pixel (row, col) or None, fps or None)."""
    import pydicom
    ds = pydicom.dcmread(_io.BytesIO(data))
    try:
        a = ds.pixel_array
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"cannot decode pixels ({e}); compressed ultrasound may need pylibjpeg in the workspace") from e
    a = np.asarray(a, np.float32)
    if a.ndim == 4:                       # (F, H, W, 3)
        a = a.mean(-1)
    elif a.ndim == 3 and a.shape[-1] == 3 and int(getattr(ds, "NumberOfFrames", 1) or 1) == 1:
        a = a.mean(-1)[None]
    elif a.ndim == 2:
        a = a[None]
    spacing = None
    regions = getattr(ds, "SequenceOfUltrasoundRegions", None)
    if regions:
        r = regions[0]
        if getattr(r, "PhysicalUnitsXDirection", None) == 3 and getattr(r, "PhysicalDeltaX", None):   # 3 = cm
            spacing = (abs(float(r.PhysicalDeltaY)) * 10, abs(float(r.PhysicalDeltaX)) * 10)
    fps = None
    if getattr(ds, "CineRate", None):
        fps = float(ds.CineRate)
    elif getattr(ds, "FrameTime", None):
        fps = 1000.0 / float(ds.FrameTime)
    return a, spacing, fps


def export_echo_study(study_id: str, data: bytes, folder: Path, readers=()) -> Dict:
    folder.mkdir(parents=True, exist_ok=True)
    frames, spacing, fps = read_echo_cine(data)
    sy, sx = spacing or (1.0, 1.0)
    aff = np.diag([sx, sy, 1.0, 1.0])
    vol = np.transpose(frames, (2, 1, 0))                                    # x, y, frame
    _save(vol, aff, folder / "cine.nii.gz", np.float32)
    sha = _save(np.zeros(vol.shape, np.uint8), aff, folder / "cine_label.nii.gz", np.uint8)
    meta = {"task": "echo_lv_traces", "study_id": study_id, "readers": list(readers), "n_frames": int(frames.shape[0]),
            "pixel_mm": list(spacing) if spacing else None, "fps": fps, "prelabel_sha256": sha,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# --------------------------------------------------------------------------- CLI
def _bytes(uri: str) -> bytes:
    from mri_training.train import _local
    return Path(_local(uri)).read_bytes()


def export_worklist(task: str, worklist: pd.DataFrame, out: str, bundle: Optional[str] = None, device: str = "cpu",
                    reader_bytes=_bytes) -> pd.DataFrame:
    """Every (study, reader) pair of the worklist -> <out>/<reader>/<study_id>/. Returns a status table."""
    from mri_training.io import read_study
    model, prov = load_prelabel_model(bundle, device) if task == "mri" else (None, None)
    rows = []
    for _, w in worklist.iterrows():
        readers = [r for r in (w.get("reader_1"), w.get("reader_2")) if isinstance(r, str) and r]
        try:
            data = reader_bytes(w["dicom_uri"])
            if task == "mri":
                lge, cine = read_study(data)
            for rd in readers:
                f = Path(out) / rd / str(w["study_id"])
                if task == "mri":
                    export_mri_study(str(w["study_id"]), lge, cine, f, model, prov, device, readers)
                else:
                    export_echo_study(str(w["study_id"]), data, f, readers)
            rows.append({"study_id": w["study_id"], "status": "exported", "error": None})
        except Exception as e:  # noqa: BLE001  (one bad study must not stop the batch)
            rows.append({"study_id": w["study_id"], "status": "failed", "error": str(e)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["mri", "echo"])
    ap.add_argument("--worklist", required=True, help="CSV from annotation.select")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bundle", default=None, help="MRI pre-drawing model folder (weights.pt + model_card.json)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    res = export_worklist(a.task, pd.read_csv(a.worklist), a.out, a.bundle, a.device)
    res.to_csv(Path(a.out) / "export_status.csv", index=False)
    print(res.status.value_counts().to_string())


if __name__ == "__main__":
    main()
