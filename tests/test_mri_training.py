"""Cardiac MRI agent (MobileNetV3-UNet): report labels, DICOM series selection, preprocessing,
quantification (infarct size, MVO, AHA transmurality, LVEF), partial-label loss, training, serving.

Needs torch + pydicom (requirements-mri.txt); skipped cleanly when they are not installed.
"""
import base64
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
torch = pytest.importorskip("torch")
pytest.importorskip("pydicom")

import synthetic_cmr as S  # noqa: E402
from mri_training import runtime  # noqa: E402
from mri_training.dataset import load_config, make_labels, report_labels  # noqa: E402
from mri_training.io import read_study  # noqa: E402
from mri_training.runtime.model import CLASSES, MobileNetV3UNet, load_model, n_params  # noqa: E402
from mri_training.runtime.pipeline import analyse, interpret  # noqa: E402
from mri_training.runtime.preprocess import MRIInputError, SIZE, Stack, prepare, prepare_labels  # noqa: E402
from mri_training.runtime.quantify import LV, MVO, MYO, SCAR, cine_function, lge_measures, red_flags  # noqa: E402
from mri_training.train import Example, class_groups, marginal_seg_loss, train  # noqa: E402

CFG = load_config()


def _prepared_truth(lge, truth):
    p = prepare(lge)
    return p, prepare_labels(truth["labels"], lge.pixel_spacing, p.order)


# --------------------------------------------------------------------------- report labels
REPORTS = {
    "ischaemic": ("Subendocardial late gadolinium enhancement in the LAD territory consistent with prior myocardial "
                  "infarction. Infarct size 14% of LV mass. Microvascular obstruction present. LVEF 42%. No LV thrombus."),
    "myocarditis": ("Patchy subepicardial late gadolinium enhancement of the inferolateral wall, consistent with "
                    "myocarditis. LVEF 58%. No microvascular obstruction."),
    "normal": "Normal LV size and function, LVEF 63.5%. No late gadolinium enhancement to suggest myocardial scar or infarction.",
    "thrombus": "Transmural enhancement of the apex with an apical thrombus and apical aneurysm. Ejection fraction is 30%.",
}


def test_report_labels():
    r = {k: report_labels(v, CFG) for k, v in REPORTS.items()}
    assert r["ischaemic"]["infarct"] == 1 and r["ischaemic"]["nonischemic_lge"] == 0
    assert r["ischaemic"]["report_infarct_size"] == 14 and r["ischaemic"]["report_lvef"] == 42
    assert r["ischaemic"]["report_mvo"] == 1 and r["ischaemic"]["lv_thrombus"] == 0
    assert r["myocarditis"]["nonischemic_lge"] == 1 and r["myocarditis"]["infarct"] == 0 and r["myocarditis"]["report_mvo"] == 0
    assert r["normal"]["infarct"] == 0 and r["normal"]["nonischemic_lge"] == 0 and r["normal"]["report_lvef"] == 63.5
    assert r["thrombus"]["lv_thrombus"] == 1 and r["thrombus"]["lv_aneurysm"] == 1 and r["thrombus"]["report_lvef"] == 30
    short = report_labels("limited", CFG)
    assert all(np.isnan(v) for v in short.values())
    m = make_labels(pd.DataFrame({"report_text": list(REPORTS.values())}), CFG)
    assert list(m.infarct) == [1, 0, 0, 1]


def test_label_sql_runs_on_omop():
    from synthetic_omop import build
    from cardioai_mcp.config import load_study
    from cardioai_mcp.engine import DuckDBEngine
    from mri_training.dataset import build_manifest
    from ecg_training.dataset import check_disjoint
    st = load_study()
    st["schemas"].update(cdm="main", vocab="main", modality="main", work="main")
    con = build(2500)
    con.execute("""CREATE TABLE cmr_study AS
      SELECT co.person_id, 'cmr' || CAST(ROW_NUMBER() OVER () AS VARCHAR) AS study_uid,
             CAST(co.condition_start_date + INTERVAL 5 DAY AS TIMESTAMP) AS study_datetime,
             'gs://b/' || CAST(co.person_id AS VARCHAR) || '.zip' AS dicom_path,
             CASE WHEN co.person_id % 3 = 0 THEN 'Subendocardial late gadolinium enhancement consistent with infarction. LVEF 41%.'
                  WHEN co.person_id % 3 = 1 THEN 'No late gadolinium enhancement. Normal LV size and function, LVEF 60%.'
                  ELSE 'Mid-wall late gadolinium enhancement of the septum. LVEF 48%.' END AS report_text
      FROM condition_occurrence co""")
    cfg = copy.deepcopy(CFG)
    cfg["splits"]["heldout_site_fraction"] = 0.2
    m = build_manifest(DuckDBEngine(con, st), st, cfg, st["modality"])
    assert len(m) > 1000 and set(m.split) >= {"train", "tune", "test_time", "test_site"}
    assert not any(check_disjoint(m).values())
    assert (m.infarct == 1).sum() > 100 and (m.nonischemic_lge == 1).sum() > 100 and m.recent_stemi.sum() > 0
    assert m.report_lvef.notna().all()


# --------------------------------------------------------------------------- DICOM series selection
def test_dicom_study_selects_short_axis_lge_and_cine():
    lge, cine, _ = S.study(seed=1, n_slices=8, phases=6)
    l2, c2 = read_study(S.dicom_zip(lge, cine))                           # + axial localiser, 4-chamber cine, non-DICOM file
    assert l2.kind == "lge" and l2.images.shape == (8, 224, 224) and "SA LGE" in l2.notes[0]
    assert c2.kind == "cine" and c2.images.shape == (6, 8, 224, 224)
    assert np.corrcoef(prepare(lge).images.ravel(), prepare(l2).images.ravel())[0, 1] > 0.999
    assert np.corrcoef(prepare(cine).images.ravel(), prepare(c2).images.ravel())[0, 1] > 0.999
    l3, c3 = read_study(S.dicom_zip(lge, None, distractors=False))
    assert c3 is None and l3.images.shape[0] == 8
    with pytest.raises(MRIInputError, match="LGE"):
        read_study(S.dicom_zip_cine_only(cine))


# --------------------------------------------------------------------------- preprocessing
def test_preprocessing_orders_slices_and_refuses_bad_stacks():
    lge, _, truth = S.study(seed=2, n_slices=8)
    p = prepare(lge)
    assert p.images.shape == (8, SIZE, SIZE) and p.images.dtype == np.float32
    assert abs(p.slice_spacing_mm - 10.0) < 1e-6
    d = np.asarray(lge.positions)[p.order] @ np.cross(p.row_cos, p.col_cos)
    assert np.all(np.diff(d) > 0)                                         # sorted along the stack normal
    y = prepare_labels(truth["labels"], lge.pixel_spacing, p.order)
    assert p.images[y == SCAR].mean() > p.images[y == LV].mean() + 0.15   # scar not clipped to blood-pool level
    bad = [Stack(lge.images[:2], lge.pixel_spacing, lge.positions[:2], lge.orientation),
           Stack(lge.images, lge.pixel_spacing, [lge.positions[0]] * 8, lge.orientation),
           Stack(lge.images * np.nan, lge.pixel_spacing, lge.positions, lge.orientation),
           Stack(lge.images, (0.0, 0.0), lge.positions, lge.orientation),
           Stack(lge.images[None], lge.pixel_spacing, lge.positions, lge.orientation, kind="lge")]
    for b in bad:
        with pytest.raises(MRIInputError):
            prepare(b)


# --------------------------------------------------------------------------- quantification
@pytest.mark.parametrize("flip", [False, True])
def test_quantification_matches_truth_and_aha_handedness(flip):
    lge, cine, truth = S.study(seed=3, n_slices=9, flip=flip, transmural=0.5, phases=8, ef=0.55)
    p, y = _prepared_truth(lge, truth)
    f = lge_measures(y, p.slice_spacing_mm, p.row_cos, p.col_cos)
    assert abs(f["infarct_size_pct_lv"] - truth["infarct_size_pct_lv"]) < 0.6
    assert abs(f["mvo_pct_lv"] - truth["mvo_pct_lv"]) < 0.2
    scarred = {k for k, v in f["segments"].items() if v["transmurality"] > 0.2}
    assert {1, 7} <= scarred and not scarred & {3, 4, 5, 9, 10, 11}           # anterior, never septal/inferior/lateral
    assert f["segments"][1]["category"] in ("26-50%", "51-75%")
    assert f["scar_median_depth"] < 0.35 and f["scar_endocardial_involvement"] > 0.2   # subendocardial (ischaemic)
    pc = prepare(cine)
    cl = np.stack([prepare_labels(truth["cine_labels"][ph], cine.pixel_spacing, pc.order) for ph in range(8)])
    assert abs(cine_function(cl, pc.slice_spacing_mm)["lvef_pct"] - 55) < 2.5


def test_midwall_scar_is_not_endocardial():
    lge, _, truth = S.study(seed=4, n_slices=6, infarct=False)
    p, y = _prepared_truth(lge, truth)
    from scipy.ndimage import distance_transform_edt
    myo = y == MYO
    for i in range(y.shape[0]):
        d_endo = distance_transform_edt(y[i] != LV)
        d_epi = distance_transform_edt(myo[i] | (y[i] == LV))
        depth = d_endo / np.maximum(d_endo + d_epi, 1e-6)
        yy, xx = np.nonzero(myo[i])
        lateral = xx > np.median(xx)
        sel = np.zeros_like(myo[i])
        sel[yy[lateral], xx[lateral]] = True
        y[i][sel & (depth > 0.35) & (depth < 0.65)] = SCAR
    f = lge_measures(y, p.slice_spacing_mm, p.row_cos, p.col_cos)
    assert 0.4 < f["scar_median_depth"] < 0.6 and f["scar_endocardial_involvement"] < 0.05


def test_red_flag_rules():
    rules = CFG["red_flags"]
    hits = {r["flag"] for r in red_flags({"infarct_size_pct_lv": 18.0, "mvo_pct_lv": 1.2, "lvef_pct": 31.0},
                                         {"lv_thrombus": 0.8, "lv_aneurysm": 0.1, "nonischemic_lge": 0.2}, rules)}
    assert hits == {"large_infarct", "mvo_present", "lvef_below_35", "lvef_below_50", "lv_thrombus"}
    none = red_flags({"infarct_size_pct_lv": 3.0, "mvo_pct_lv": 0.0, "lvef_pct": None},
                     {"lv_thrombus": 0.1, "lv_aneurysm": 0.1, "nonischemic_lge": 0.1}, rules)
    assert none == []                                                     # unmeasured LVEF never raises a flag


# --------------------------------------------------------------------------- model + loss
def test_model_contract_and_strict_loading(tmp_path):
    m = MobileNetV3UNet(width=1.0)
    assert 6e6 < n_params(m) < 8.5e6
    assert not any(isinstance(x, torch.nn.BatchNorm2d) for x in m.modules())   # GroupNorm: same in training and serving
    small = MobileNetV3UNet(width=0.25)
    x = torch.rand(5, 2, SIZE, SIZE)
    seg, st = small(x)
    assert seg.shape == (5, len(CLASSES), SIZE, SIZE) and st.shape == (4,)
    small.train()
    a = small(x)[0]
    small.eval()
    assert torch.allclose(a, small(x)[0], atol=1e-5)
    torch.save(small.state_dict(), tmp_path / "w.pt")
    assert n_params(load_model(str(tmp_path / "w.pt"), {"width": 0.25})) == n_params(small)
    torch.save({"state_dict": small.state_dict(), "epoch": 3}, tmp_path / "wrapped.pt")
    with pytest.raises((TypeError, Exception)):
        load_model(str(tmp_path / "wrapped.pt"), {"width": 0.25})
    with pytest.raises(RuntimeError):
        load_model(str(tmp_path / "w.pt"), {"width": 0.5})


def test_marginal_loss_counts_undrawn_classes_under_their_parent():
    myops = CFG["public_datasets"]["myops"]
    g = class_groups({"annotated": myops["annotated"], "merge": myops["merge"]})
    assert [0, 4] not in g and [3, 4] in g and [0] in g                    # MVO -> scar; nothing else merged into bg
    assert class_groups(CFG["public_datasets"]["emidec"]) == [[0, 5], [1], [2], [3], [4]]
    y = torch.full((1, 4, 4), SCAR)
    as_scar = torch.zeros(1, len(CLASSES), 4, 4)
    as_scar[:, SCAR] = 8
    as_mvo = torch.zeros(1, len(CLASSES), 4, 4)
    as_mvo[:, MVO] = 8
    as_myo = torch.zeros(1, len(CLASSES), 4, 4)
    as_myo[:, MYO] = 8
    l_scar, l_mvo, l_myo = (float(marginal_seg_loss(z, y, g)) for z in (as_scar, as_mvo, as_myo))
    assert abs(l_scar - l_mvo) < 1e-5 and l_myo > l_scar + 1                # MVO inside MyoPS scar is not penalised


def test_runtime_fingerprint_covers_all_runtime_files(tmp_path):
    import shutil
    shutil.copytree(ROOT / "mri_training" / "runtime", tmp_path / "rt")
    assert runtime.fingerprint(tmp_path / "rt") == runtime.fingerprint()
    for name in runtime.FILES:
        f = tmp_path / "rt" / name
        old = f.read_bytes()
        f.write_bytes(old + b"\n# changed\n")
        assert runtime.fingerprint(tmp_path / "rt") != runtime.fingerprint()
        f.write_bytes(old)
    assert set(runtime.FILES) == {"model.py", "preprocess.py", "quantify.py", "pipeline.py"}


# --------------------------------------------------------------------------- training -> bundle -> serving
def _loader(cfg):
    def load(r, _cfg):
        lge, cine, t = S.study(seed=int(r.seed), n_slices=6, infarct=bool(r.infarct), phases=4, transmural=0.6)
        ex = Example(lge=lge, cine=cine)
        if r.label_uri:
            ex.lge_labels, ex.lge_rule = t["labels"], cfg["mayo_contours"]["lge"]
            ex.cine_labels, ex.cine_phases, ex.cine_rule = t["cine_labels"][[0, 2]], [0, 2], cfg["mayo_contours"]["cine"]
        return ex
    return load


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("mri_bundle")
    cfg = copy.deepcopy(CFG)
    cfg["arch"]["width"] = 0.25
    cfg["train"].update(stage_a_epochs=18, stage_b_epochs=2, accumulate_studies=1, lr=0.003, patience=50, augment=None)
    rows = []
    for i in range(9):
        split = "train" if i < 6 else ("tune" if i < 8 else "test_time")
        rows.append(dict(study_id=f"s{i}", person_id=i, split=split, seed=i, infarct=float(i % 2 == 0), nonischemic_lge=0.0,
                         lv_thrombus=0.0, lv_aneurysm=0.0, label_uri="contours" if i != 4 else "",
                         report_lvef=55.0, report_infarct_size=np.nan, report_mvo=float(i % 2 == 0)))
    card = train(pd.DataFrame(rows), cfg, str(out), loader=_loader(cfg), device="cpu", log=lambda s: None, n_boot=20)
    return out, cfg, card


def test_training_learns_and_writes_a_lockable_bundle(trained):
    out, cfg, card = trained
    hist = pd.DataFrame(card["history"])
    a = hist[hist.stage == "A"]
    assert a.train_loss.iloc[-1] < a.train_loss.iloc[0] * 0.8
    assert hist.dice_lv_myocardium.max() > 0.6 and hist.dice_lv_pool.max() > 0.6
    for f in ("weights.pt", "thresholds.json", "model_card.json"):
        assert (out / f).exists()
    assert card["runtime_fingerprint"] == runtime.fingerprint()
    assert card["weights_sha256"] == hashlib.sha256((out / "weights.pt").read_bytes()).hexdigest()
    assert set(card["thresholds"]) == set(cfg["study_heads"]) and card["red_flags"] == cfg["red_flags"]
    q = pd.DataFrame(card["metrics"]["test_time_quantitative"]).set_index("measure")
    # a few CPU epochs on 5 studies: the test checks the evaluation wiring, not accuracy
    for k in ("dice_lv_myocardium", "dice_scar", "infarct size vs reader contours (% LV)", "LVEF vs report (%)"):
        assert q.loc[k, "n"] >= 1
    reloaded = load_model(str(out / "weights.pt"), {**cfg["arch"], "study_heads": cfg["study_heads"]})
    assert n_params(reloaded) == card["parameters"]


def test_serving_matches_validated_pipeline_and_refuses_mismatches(trained, tmp_path):
    import shutil
    from mri_training.serve.predictor import BundleError, MRIPredictor, create_app
    out, cfg, card = trained
    pred = MRIPredictor(out, device="cpu", allowed_prefixes=[str(tmp_path)])
    lge, cine, _ = S.study(seed=42, n_slices=6, phases=4)
    z = S.dicom_zip(lge, cine)
    (tmp_path / "study.zip").write_bytes(z)
    res = pred.predict([{"study_zip_b64": base64.b64encode(z).decode()},
                        {"study_uri": str(tmp_path / "study.zip")},
                        {"study_uri": "/etc/passwd"},                                  # outside the allowed prefix
                        {"study_zip_b64": base64.b64encode(b"not a zip").decode()},
                        {"study_zip_b64": base64.b64encode(S.dicom_zip_cine_only(cine)).decode()},   # no LGE
                        {}])
    model = load_model(str(out / "weights.pt"), {**cfg["arch"], "study_heads": cfg["study_heads"]})
    ref = analyse(model, *read_study(z))
    assert res[0]["probabilities"] == pytest.approx(ref["probabilities"], abs=1e-5)
    assert res[0]["findings"]["infarct_size_pct_lv"] == ref["findings"]["infarct_size_pct_lv"]
    assert res[0]["findings"]["lvef_pct"] == ref["findings"]["lvef_pct"]
    assert res[0]["flags"] == interpret(ref, card["thresholds"], cfg["red_flags"])["flags"]
    assert res[1]["probabilities"] == res[0]["probabilities"]
    assert res[0]["runtime_fingerprint"] == runtime.fingerprint() and res[0]["weights_sha256"] == card["weights_sha256"]
    assert all(r["error_type"] == "input" for r in res[2:]), res[2:]
    json.dumps(res)                                                                    # JSON-serialisable response
    client = create_app(pred).test_client()
    assert client.get("/health").get_json()["runtime_fingerprint"] == runtime.fingerprint()
    assert client.post("/predict", json={}).status_code == 400
    # tampered bundle -> refuses to start
    bad = tmp_path / "bad"
    shutil.copytree(out, bad)
    c = json.loads((bad / "model_card.json").read_text())
    for patch, match in (({"runtime_fingerprint": "0" * 64}, "runtime"), ({"weights_sha256": "0" * 64}, "weights")):
        (bad / "model_card.json").write_text(json.dumps({**c, **patch}))
        with pytest.raises(BundleError, match=match):
            MRIPredictor(bad, device="cpu")
    (bad / "model_card.json").write_text(json.dumps(c))
    (bad / "thresholds.json").write_text(json.dumps({"infarct": 0.5}))
    with pytest.raises(BundleError, match="thresholds"):
        MRIPredictor(bad, device="cpu")


if os.environ.get("CARDIOAI_APP_REPO"):  # cross-repo: the app must pin exactly this runtime
    def test_app_lock_pins_this_runtime():
        import yaml
        reg = yaml.safe_load((Path(os.environ["CARDIOAI_APP_REPO"]) / "app" / "validated_agents" / "registry.yaml").read_text())
        assert reg["agents"]["mri"]["output_adapter"]["runtime_fingerprint"] == runtime.fingerprint()
