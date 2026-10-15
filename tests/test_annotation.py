"""Annotation kit: native-grid export (NIfTI geometry, model pre-drawn outlines), reader files back to
training labels, QC refusals, double-read agreement, worklist selection, echo traces, workspace check."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
torch = pytest.importorskip("torch")
nib = pytest.importorskip("nibabel")
pytest.importorskip("pydicom")

import synthetic_cmr as S  # noqa: E402
from annotation.collect import collect  # noqa: E402
from annotation.export import export_echo_study, export_mri_study, export_worklist  # noqa: E402
from annotation.geometry import from_nifti_array, stack_affine, to_nifti_array, unprepare_labels  # noqa: E402
from annotation.select import load_tasks, select_mri  # noqa: E402
from mri_training.runtime.preprocess import prepare, prepare_labels  # noqa: E402
from mri_training.runtime.quantify import LV, MYO, SCAR  # noqa: E402

TASKS = load_tasks()


def _dice(a, b):
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


def _write(folder, name, arr_shw, order):
    img = nib.load(str(folder / f"{name}.nii.gz"))
    nib.save(nib.Nifti1Image(to_nifti_array(arr_shw, order).astype(np.uint8), img.affine), str(folder / f"{name}_label.nii.gz"))


def _draw_truth(folder, truth, meta):
    """A reader who draws exactly the true anatomy."""
    _write(folder, "lge", truth["labels"], np.asarray(meta["lge"]["order"]))
    for name in ("ed", "es"):
        lab = truth["cine_labels"][meta["cine"]["phases"][name]].copy()
        lab[np.isin(lab, (3, 4))] = MYO
        nii = nib.load(str(folder / f"cine_{name}.nii.gz"))
        nib.save(nib.Nifti1Image(to_nifti_array(lab, np.asarray(meta["cine"]["order"])).astype(np.uint8), nii.affine),
                 str(folder / f"cine_{name}_label.nii.gz"))


def test_geometry_round_trip_and_nifti_orientation():
    lge, _, truth = S.study(seed=5, n_slices=8)
    aff, order = stack_affine(lge)
    pos = np.asarray(lge.positions)[order]
    first = np.diag([-1, -1, 1]) @ pos[0]
    assert np.allclose(aff[:3, 3], first)                                           # voxel (0,0,0) = first sorted slice, RAS
    assert np.allclose(aff[:3, :3] @ [0, 0, 1], np.diag([-1, -1, 1]) @ (pos[1] - pos[0]))  # z steps along the stack
    assert np.array_equal(from_nifti_array(to_nifti_array(truth["labels"], order), order), truth["labels"])
    p = prepare(lge)
    back = unprepare_labels(prepare_labels(truth["labels"], lge.pixel_spacing, p.order), lge.images.shape[-2:], lge.pixel_spacing)
    ref = truth["labels"][p.order]
    for c in (LV, MYO, SCAR):
        assert _dice(back == c, ref == c) > 0.9


def test_mri_export_draw_collect_gives_training_labels(tmp_path):
    lge, cine, truth = S.study(seed=6, n_slices=7, phases=6)
    f = tmp_path / "root" / "dr_a" / "s1"
    meta = export_mri_study("s1", lge, cine, f, readers=["dr_a"])
    for n in ("lge", "lge_label", "cine_ed", "cine_ed_label", "cine_es", "cine_es_label"):
        assert (f / f"{n}.nii.gz").exists()
    assert meta["cine"]["phases"] == {"ed": 0, "es": 2} and "guess" in meta["cine"]["phase_choice"]
    res = collect("mri", str(tmp_path / "root"), str(tmp_path / "out"))
    assert res["collected"].status.tolist() == ["rejected"] and "not reviewed" in res["collected"].errors[0]   # untouched
    _draw_truth(f, truth, meta)
    res = collect("mri", str(tmp_path / "root"), str(tmp_path / "out"))
    assert res["collected"].status.tolist() == ["ok"], res["collected"].errors.tolist()
    z = np.load(res["label_uri"].label_uri[0])
    assert np.array_equal(z["lge"], truth["labels"])                                # stack order, as training reads it
    assert list(z["cine_phases"]) == [0, 2] and z["cine"].shape == (2,) + truth["cine_labels"].shape[1:]
    # scar drawn on cine is refused
    meta = json.loads((f / "meta.json").read_text())
    bad = truth["cine_labels"][0].copy()
    bad[bad == MYO] = SCAR
    _write(f, "cine_ed", bad, np.asarray(meta["cine"]["order"]))
    res = collect("mri", str(tmp_path / "root"), str(tmp_path / "out"))
    assert res["collected"].status[0] == "rejected" and "no scar on cine" in res["collected"].errors[0]


def test_model_prelabels_and_phase_choice(tmp_path):
    from annotation.export import load_prelabel_model
    from mri_training import runtime
    from mri_training.runtime.model import MobileNetV3UNet
    import hashlib
    b = tmp_path / "bundle"
    b.mkdir()
    m = MobileNetV3UNet(width=0.25)
    torch.save(m.state_dict(), b / "weights.pt")
    (b / "model_card.json").write_text(json.dumps({"model_name": "pre", "arch": {"width": 0.25}, "study_heads": m.study_heads,
                                                     "runtime_fingerprint": runtime.fingerprint()}))
    model, prov = load_prelabel_model(str(b))
    assert prov["weights_sha256"] == hashlib.sha256((b / "weights.pt").read_bytes()).hexdigest()
    lge, cine, _ = S.study(seed=7, n_slices=5, phases=4)
    meta = export_mri_study("s2", lge, cine, tmp_path / "x", model, prov)
    assert meta["prelabel"]["model_name"] == "pre" and meta["cine"]["phase_choice"].startswith("model")
    lab = nib.load(str(tmp_path / "x" / "lge_label.nii.gz"))
    assert lab.shape == nib.load(str(tmp_path / "x" / "lge.nii.gz")).shape
    assert set(np.unique(np.asarray(nib.load(str(tmp_path / "x" / "cine_ed_label.nii.gz")).get_fdata()))) <= {0, 1, 2, 5}


def test_double_reads_agreement_and_label_uri_uses_primary(tmp_path):
    lge, cine, truth = S.study(seed=8, n_slices=6, phases=4)
    for rd in ("dr_a", "dr_b"):
        f = tmp_path / "root" / rd / "s3"
        meta = export_mri_study("s3", lge, cine, f, readers=["dr_a", "dr_b"])
        t = copy.deepcopy(truth)
        if rd == "dr_b":                                                            # second reader: smaller scar
            t["labels"][t["labels"] == 4] = SCAR
            t["labels"][(t["labels"] == SCAR) & (np.indices(t["labels"].shape)[2] % 2 == 0)] = MYO
        _draw_truth(f, t, meta)
    res = collect("mri", str(tmp_path / "root"), str(tmp_path / "out"))
    a = res["agreement"].iloc[0]
    assert a.dice_lv_pool == pytest.approx(1.0) and a.dice_scar < 0.9 and a.infarct_pct_diff > 0 and a.lvef_diff == 0
    assert res["label_uri"].label_uri[0].endswith("s3__dr_a.npz")


def test_worklist_selection():
    rng = np.random.default_rng(0)
    n = 3000
    m = pd.DataFrame({"study_id": [f"s{i}" for i in range(n)], "person_id": rng.integers(0, 2500, n),
                      "split": rng.choice(["train", "tune", "test_time", "test_site"], n, p=[.6, .15, .15, .1]),
                      "recent_stemi": (rng.random(n) < .2).astype(int), "nonischemic_lge": (rng.random(n) < .1).astype(float),
                      "dicom_uri": [f"gs://b/{i}.zip" for i in range(n)]})
    w = select_mri(m, ["r1", "r2", "r3"], TASKS["mri_contours"])
    assert len(w) == 300 and w.person_id.is_unique and set(w.split) == {"train", "tune", "test_time"}
    assert abs((w.split == "train").mean() - 0.70) < 0.02 and abs((w.stratum == "recent_stemi").mean() - 0.5) < 0.03
    d = w[w.reader_2 != ""]
    assert len(d) == 30 and (d.reader_1 != d.reader_2).all()
    assert w.reader_1.value_counts().max() - w.reader_1.value_counts().min() <= 1
    assert select_mri(m, ["r1", "r2", "r3"], TASKS["mri_contours"]).equals(w)                 # deterministic
    with pytest.raises(ValueError, match="annotator seats"):                                    # MCP: 5 seats
        select_mri(m, [f"r{i}" for i in range(6)], TASKS["mri_contours"])


def _echo_dicom(frames):
    import io
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.3.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID, ds.Modality = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID, "US"
    a = (frames * 255).astype(np.uint8)
    ds.NumberOfFrames, ds.Rows, ds.Columns = a.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = \
        1, "MONOCHROME2", 8, 8, 7, 0
    ds.FrameTime = 20.0
    r = Dataset()
    r.PhysicalUnitsXDirection, r.PhysicalUnitsYDirection, r.PhysicalDeltaX, r.PhysicalDeltaY = 3, 3, 0.03, 0.03
    ds.SequenceOfUltrasoundRegions = [r]
    ds.PixelData = a.tobytes()
    b = io.BytesIO()
    S._save(ds, b)
    return b.getvalue()


def test_echo_traces(tmp_path):
    yy, xx = np.mgrid[0:80, 0:64]
    radius = 20 - 8 * np.sin(np.linspace(0, np.pi, 12))[:, None, None]
    frames = (np.hypot(yy - 40, xx - 32)[None] < radius).astype(float) * 0.8
    f = tmp_path / "root" / "dr_a" / "e1"
    meta = export_echo_study("e1", _echo_dicom(frames), f, ["dr_a"])
    assert meta["n_frames"] == 12 and meta["fps"] == 50 and meta["pixel_mm"] == [0.3, 0.3]
    lab = np.zeros((12, 80, 64), np.uint8)
    lab[6] = frames[6] > 0                     # smallest cavity (ES), drawn first in time order? no: frame 6 > frame 0
    lab[0] = frames[0] > 0
    img = nib.load(str(f / "cine.nii.gz"))
    nib.save(nib.Nifti1Image(np.transpose(lab, (2, 1, 0)), img.affine), str(f / "cine_label.nii.gz"))
    res = collect("echo", str(tmp_path / "root"), str(tmp_path / "out"))
    assert res["collected"].status[0] == "ok"
    z = np.load(res["label_uri"].label_uri[0])
    assert list(z["frames"]) == [0, 6]                                            # ED = larger cavity
    lab[9] = frames[9] > 0                                                        # three frames -> refused
    nib.save(nib.Nifti1Image(np.transpose(lab, (2, 1, 0)), img.affine), str(f / "cine_label.nii.gz"))
    assert collect("echo", str(tmp_path / "root"), str(tmp_path / "out"))["collected"].status[0] == "rejected"


def test_export_worklist_and_training_reads_the_labels(tmp_path):
    from mri_training.dataset import load_config
    from mri_training.train import load_example, to_parts
    lge, cine, truth = S.study(seed=9, n_slices=6, phases=8)
    z = S.dicom_zip(lge, cine)
    (tmp_path / "s9.zip").write_bytes(z)
    wl = pd.DataFrame([{"study_id": "s9", "dicom_uri": str(tmp_path / "s9.zip"), "reader_1": "dr_a", "reader_2": ""},
                       {"study_id": "bad", "dicom_uri": str(tmp_path / "missing.zip"), "reader_1": "dr_a", "reader_2": ""}])
    st = export_worklist("mri", wl, str(tmp_path / "root"))
    assert st.status.tolist() == ["exported", "failed"]
    f = tmp_path / "root" / "dr_a" / "s9"
    meta = json.loads((f / "meta.json").read_text())
    # the reader draws on what was exported (read_study order); reconstruct truth in that order
    from mri_training.io import read_study
    l2, c2 = read_study(z)
    p_syn, p_dcm = prepare(lge), prepare(l2)
    lab_sorted = truth["labels"][p_syn.order]
    reorder = np.empty_like(p_dcm.order)
    reorder[p_dcm.order] = np.arange(len(p_dcm.order))
    t = {"labels": lab_sorted[reorder], "cine_labels": truth["cine_labels"][:, p_syn.order][:, reorder]}
    _draw_truth(f, t, meta)
    res = collect("mri", str(tmp_path / "root"), str(tmp_path / "out"))
    assert res["collected"].status.tolist() == ["ok"], res["collected"].errors.tolist()
    cfg = load_config()
    row = pd.Series({"source": "mayo", "dicom_uri": str(tmp_path / "s9.zip"), "label_uri": res["label_uri"].label_uri[0]})
    parts = to_parts(load_example(row, cfg))
    ref = prepare_labels(truth["labels"], lge.pixel_spacing, p_syn.order)
    assert _dice(parts[0].y == SCAR, ref == SCAR) > 0.95 and len(parts) == 3       # LGE + cine ED + ES


def test_workspace_check_runs():
    from annotation.check_workspace import run
    r = run(fetch=False)
    assert set(r["options"]) >= {"A_slicer_desktop", "C_browser_ohif_monai_label", "D_cvat_browser"}
    assert r["local_web_server"]["ok"]


def test_mayo_stage_a_model_pre_draws_for_readers(tmp_path):
    """Workflow step 3 (Mayo-only): a segmentation-only model trained on the first reader outlines, starting
    from the self-supervised Mayo encoder, pre-draws the remaining studies."""
    from annotation.export import load_prelabel_model
    from mri_training.dataset import load_config
    from mri_training.pretrain import pretrain
    from mri_training.train import Example, train
    cfg = copy.deepcopy(load_config())
    cfg["arch"]["width"] = 0.25
    cfg["pretrain"].update(epochs=1, batch_size=8)
    cfg["train"].update(stage_a_epochs=1, stage_b_epochs=0, accumulate_studies=1, augment=None)   # = --stage-a-only

    def loader(r, _c):
        lge, _, t = S.study(seed=int(r.seed), n_slices=5)
        ex = Example(lge=lge)
        if r.label_uri:
            ex.lge_labels, ex.lge_rule = t["labels"], cfg["mayo_contours"]["lge"]
        return ex
    m = pd.DataFrame([dict(study_id=f"m{i}", person_id=i, split="train" if i < 3 else "tune", seed=i, study_date="2018-05-01",
                           infarct=1.0, nonischemic_lge=0.0, lv_thrombus=0.0, lv_aneurysm=0.0, label_uri="x" if i != 1 else "")
                      for i in range(4)])
    pretrain(m, cfg, str(tmp_path / "ssl"), loader=loader, device="cpu", log=lambda s: None)
    card = train(m, cfg, str(tmp_path / "pre"), loader=loader, device="cpu", log=lambda s: None, n_boot=0, init=str(tmp_path / "ssl"))
    assert card["initialised_from"]["scope"] == "encoder" and card["initialised_from"]["uses_outside_data"] is False
    model, prov = load_prelabel_model(str(tmp_path / "pre"))
    assert model is not None and len(prov["weights_sha256"]) == 64
