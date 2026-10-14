"""Synthetic short-axis cardiac MRI with known anatomy (no patient data).

Geometry is set in PATIENT space and drawn through a DICOM orientation, so tests can flip the
image and still know where anterior is. LV: circular cavity + 10 mm wall, shrinking to the apex;
RV: crescent on the septal side; infarct: subendocardial sector centred on the patient-anterior
direction with a chosen transmurality, MVO core inside it (basal + mid slices).
LGE contrast: nulled myocardium dark, scar bright, MVO dark core, blood pools grey.
Cine: P phases with the cavity radius scaled per phase (known EF).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mri_training.runtime.preprocess import Stack  # noqa: E402

BG, LV, MYO, SCAR, MVO, RV = range(6)


def _angle_of(direction_patient, row_cos, col_cos):
    """Image angle (x = columns, y = rows) of a patient-space direction projected on the image plane."""
    return float(np.arctan2(np.dot(direction_patient, col_cos), np.dot(direction_patient, row_cos)))


def labels_slice(h, w, px, centre, r_endo, wall_mm, septal_theta, anterior_theta, transmural=0.0, mvo=False,
                 scar_width_deg=70.0):
    yy, xx = np.mgrid[0:h, 0:w]
    dx, dy = (xx - centre[0]) * px, (yy - centre[1]) * px
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    lab = np.zeros((h, w), np.int64)
    r_epi = r_endo + wall_mm
    # RV: crescent beyond the septum
    rv_c = (centre[0] + np.cos(septal_theta) * (r_epi + 14) / px, centre[1] + np.sin(septal_theta) * (r_epi + 14) / px)
    rv = (np.hypot((xx - rv_c[0]) * px, (yy - rv_c[1]) * px) < 18) & (r > r_epi + 2)
    lab[rv] = RV
    lab[r < r_endo] = LV
    myo = (r >= r_endo) & (r < r_epi)
    lab[myo] = MYO
    if transmural > 0:
        dth = np.angle(np.exp(1j * (th - anterior_theta)))
        sector = np.abs(dth) <= np.deg2rad(scar_width_deg / 2)
        depth = (r - r_endo) / wall_mm
        scar = myo & sector & (depth < transmural)
        lab[scar] = SCAR
        if mvo:
            lab[scar & (depth < transmural * 0.35) & (np.abs(dth) <= np.deg2rad(scar_width_deg / 6))] = MVO
    return lab


INTENSITY_LGE = {BG: 0.30, LV: 0.55, MYO: 0.05, SCAR: 0.95, MVO: 0.08, RV: 0.50}
INTENSITY_CINE = {BG: 0.25, LV: 0.90, MYO: 0.20, SCAR: 0.20, MVO: 0.20, RV: 0.85}


def render(lab, table, rnd, noise=0.03):
    img = np.zeros(lab.shape)
    for k, v in table.items():
        img[lab == k] = v
    return img + rnd.normal(0, noise, img.shape)


def study(seed=0, n_slices=10, h=224, w=224, px=1.4, gap_mm=10.0, flip=False, transmural=0.5, mvo=True,
          phases=8, ef=0.55, infarct=True, oblique=(35.0, 25.0)):
    rnd = np.random.default_rng(seed)
    # short axis is a double-oblique plane: start from columns -> patient left (L), rows -> posterior
    # (P; anterior if flipped) and tilt it like a real SAX plane
    a, b = np.deg2rad(oblique)
    tilt = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]) @ \
        np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    row_cos = tilt @ np.array([1.0, 0.0, 0.0])
    col_cos = tilt @ np.array([0.0, -1.0 if flip else 1.0, 0.0])
    normal = np.cross(row_cos, col_cos)
    anterior = np.array([0.0, -1.0, 0.0])                 # LPS: anterior = -Y
    patient_septal = np.array([-1.0, 0.0, 0.0])          # septum towards patient right (simplified)
    ant_t = _angle_of(anterior, row_cos, col_cos)
    sep_t = _angle_of(patient_septal, row_cos, col_cos)
    centre = (w / 2 + rnd.uniform(-6, 6), h / 2 + rnd.uniform(-6, 6))
    labs, cine_labs = [], []
    radii = np.linspace(26, 10, n_slices)                 # base -> apex
    for i in range(n_slices):
        tm = transmural if (infarct and i < int(n_slices * 2 / 3)) else 0.0
        labs.append(labels_slice(h, w, px, centre, radii[i], 10.0, sep_t, ant_t, tm, mvo and tm > 0))
    lab = np.stack(labs)
    scale = np.sqrt(1 - ef) ** (1 / 1)                    # ES radius factor so that EF = 1 - scale^2
    for p in range(phases):
        f = 1 - (1 - scale) * (0.5 - 0.5 * np.cos(2 * np.pi * p / phases))
        cine_labs.append(np.stack([labels_slice(h, w, px, centre, radii[i] * f, 10.0 / np.sqrt(f), sep_t, ant_t, 0.0)
                                   for i in range(n_slices)]))
    cine_lab = np.stack(cine_labs)
    perm = rnd.permutation(n_slices)                      # scanners do not store slices in order
    positions = [list(np.array([0.0, 0.0, 0.0]) + normal * gap_mm * k) for k in range(n_slices)]
    lge = Stack(images=np.stack([render(lab[k], INTENSITY_LGE, rnd) for k in perm]), pixel_spacing=(px, px),
                positions=[positions[k] for k in perm], orientation=list(row_cos) + list(col_cos), slice_thickness=8.0, kind="lge")
    cine = Stack(images=np.stack([[render(cine_lab[p, k], INTENSITY_CINE, rnd) for k in perm] for p in range(phases)]),
                 pixel_spacing=(px, px), positions=[positions[k] for k in perm], orientation=list(row_cos) + list(col_cos),
                 slice_thickness=8.0, kind="cine")
    myo = np.isin(lab, (MYO, SCAR, MVO)).sum()
    truth = {"infarct_size_pct_lv": 100 * np.isin(lab, (SCAR, MVO)).sum() / myo, "mvo_pct_lv": 100 * (lab == MVO).sum() / myo,
             "ef_pct": 100 * ef, "labels": lab[perm], "cine_labels": cine_lab[:, perm], "perm": perm, "flip": flip}
    return lge, cine, truth


# --------------------------------------------------------------------------- DICOM study (zip)
def _ds(img, series_uid, desc, number, pos, orient, px, extra):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"           # MR Image Storage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.SeriesInstanceUID, ds.SeriesDescription, ds.InstanceNumber = "MR", series_uid, desc, number
    ds.ImagePositionPatient = [float(v) for v in pos]
    ds.ImageOrientationPatient = [float(v) for v in orient]
    ds.PixelSpacing = [px, px]
    ds.SliceThickness = 8.0
    a = np.clip(img * 1000, 0, 4095).astype(np.uint16)
    ds.Rows, ds.Columns = a.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelData = a.tobytes()
    for k, v in extra.items():
        setattr(ds, k, v)
    return ds


def _save(ds, buf):
    """pydicom 3 (Python >= 3.10) and pydicom 2 (CI on Python 3.9) write APIs."""
    import pydicom
    if int(pydicom.__version__.split(".")[0]) >= 3:
        ds.save_as(buf, enforce_file_format=True)
    else:
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(buf, write_like_original=False)


def dicom_zip_cine_only(cine) -> bytes:
    """A study without LGE (cine + localisers only)."""
    return dicom_zip(None, cine)


def dicom_zip(lge, cine=None, distractors=True) -> bytes:
    """Synthetic study as a zip of DICOM files: SAX LGE, SAX cine and (optionally) an axial
    localiser and a 4-chamber cine that series selection must ignore."""
    import io
    import zipfile
    from pydicom.uid import generate_uid
    files = []
    uid = generate_uid()
    for k in range(lge.images.shape[0] if lge is not None else 0):
        files.append(_ds(lge.images[k], uid, "SA LGE PSIR", k + 1, lge.positions[k], lge.orientation, lge.pixel_spacing[0],
                         {"InversionTime": 300.0, "ImageType": ["DERIVED", "PRIMARY", "PSIR"]}))
    if cine is not None:
        uid = generate_uid()
        P = cine.images.shape[0]
        for ph in range(P):
            for k in range(cine.images.shape[1]):
                files.append(_ds(cine.images[ph, k], uid, "cine_sax SSFP", ph * 100 + k + 1, cine.positions[k], cine.orientation,
                                 cine.pixel_spacing[0], {"TriggerTime": 40.0 * ph, "CardiacNumberOfImages": P}))
    if distractors:
        uid = generate_uid()
        for k in range(3):                                                 # axial localiser
            files.append(_ds(np.random.default_rng(k).random((64, 64)), uid, "localizer", k + 1, [0, 0, 10.0 * k],
                             [1, 0, 0, 0, 1, 0], 3.0, {}))
        uid = generate_uid()
        for ph in range(20):                                               # 4-chamber cine: one position
            files.append(_ds(np.random.default_rng(ph).random((64, 64)), uid, "cine 4ch", ph + 1, [0, 0, 0],
                             [0.7, 0.7, 0, 0, 0, -1], 1.5, {"TriggerTime": 40.0 * ph}))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, ds in enumerate(files):
            b = io.BytesIO()
            _save(ds, b)
            z.writestr(f"IMG{i:05d}.dcm", b.getvalue())
        z.writestr("DICOMDIR.txt", "not dicom")
    return buf.getvalue()
