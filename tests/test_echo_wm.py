"""Echo wall-motion model: report labels, AHA segments and territories, preprocessing, DICOM clips, model,
pipeline, training bundle, serving container, MCP request files, view labelling and pretraining."""
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
torch = pytest.importorskip("torch")
pytest.importorskip("pydicom")

import synthetic_echo as E  # noqa: E402
from echo_wm_training import runtime  # noqa: E402
from echo_wm_training.dataset import load_config, make_labels  # noqa: E402
from echo_wm_training.labels import label_columns, parse_report  # noqa: E402
from echo_wm_training.runtime.model import build_model, load_model, view_index  # noqa: E402
from echo_wm_training.runtime.pipeline import analyse, interpret  # noqa: E402
from echo_wm_training.runtime.preprocess import CLIP_FRAMES, EchoInputError, Video, decode_dicom, prepare  # noqa: E402
from echo_wm_training.runtime.segments import SEGMENTS, TERRITORIES, VIEW_SEGMENTS, VIEWS, assessed, summarise  # noqa: E402

CFG = load_config()


def _abn(r):
    return {int(k[4:]): int(v) + 1 for k, v in r.items() if k.startswith("seg_") and v == v and v > 0}


# --------------------------------------------------------------------------- report labels
def test_report_parser():
    r = parse_report("Normal LV size. No regional wall motion abnormalities. LVEF 60%.")
    assert r["wm_described"] and _abn(r) == {} and r["wma_any"] == 0 and r["wmsi"] == 1.0 and r["report_lvef"] == 60
    r = parse_report("Hypokinesis of the basal and mid inferior and inferolateral walls. LVEF 45%.")
    assert _abn(r) == {4: 2, 5: 2, 10: 2, 11: 2}
    assert (r["terr_lad"], r["terr_rca"], r["terr_lcx"]) == (0.0, 1.0, 1.0)
    r = parse_report("The mid and apical anterior wall, anteroseptum and apex are akinetic. Ejection fraction 35-40%.")
    assert _abn(r) == {7: 3, 8: 3, 13: 3, 14: 3, 17: 3} and r["report_lvef"] == 37.5 and r["terr_lad"] == 1.0
    r = parse_report("Severe global hypokinesis with akinesis of the inferior wall. EF 20%.")
    assert r["global_hypokinesis"] == 1.0 and len(_abn(r)) == 17 and _abn(r)[4] == 3 and _abn(r)[1] == 2
    r = parse_report("The inferior wall is hypokinetic and the apex is akinetic.")
    assert _abn(r)[17] == 3 and _abn(r)[4] == 2
    assert _abn(parse_report("Apical segments are akinetic.")) == {13: 3, 14: 3, 15: 3, 16: 3, 17: 3}
    assert _abn(parse_report("Large apical aneurysm."))[17] == 4
    assert _abn(parse_report("Hypokinesis in the LAD territory.")) == {s: 2 for s in TERRITORIES["LAD"]}
    hedged = parse_report("Possible hypokinesis of the basal inferior wall. LVEF 55%.")
    assert hedged["wm_uncertain"] and np.isnan(hedged["seg_4"]) and np.isnan(hedged["wma_any"])
    none = parse_report("Normal LVEF. Mild mitral regurgitation.")
    assert not none["wm_described"] and np.isnan(none["seg_1"]) and np.isnan(none["global_hypokinesis"])
    assert parse_report("Previous LVEF 30%. Current LVEF 55%.")["report_lvef"] == 55
    assert not _abn(parse_report("No akinesis. No regional wall motion abnormalities."))


# --------------------------------------------------------------------------- segments
def test_segments_views_and_territories():
    assert sorted(s for t in TERRITORIES.values() for s in t) == list(range(1, 18))
    assert assessed(VIEWS).all() and assessed(["A4C"]).sum() == 7 and assessed(["A4C", "A2C"]).sum() == 13
    assert all(VIEW_SEGMENTS[v][3] == 17 for v in VIEWS)                       # the apex is in the middle slot
    p = np.tile([0.9, 0.05, 0.03, 0.02], (17, 1))
    p[3] = [0.1, 0.2, 0.6, 0.1]                                                # basal inferior akinetic
    s = summarise(p, assessed(["A4C", "A2C"]))
    assert s["segments"]["4"]["score"] == 3 and "5" not in s["segments"]
    assert s["territories"]["RCA"]["abnormal_segments"] == [4] and s["territories"]["LCx"]["assessed_segments"] == 3
    assert s["n_abnormal_segments"] == 1 and s["wmsi"] == round((12 + 3) / 13, 2)


# --------------------------------------------------------------------------- preprocessing and DICOM
def test_preprocess_and_dicom():
    _, _, clips = E.study("rca", seed=1, views=("A4C",), n_frames=60)
    v = decode_dicom(E.dicom_bytes(clips["A4C"], fps=30.0, view_tag="Apical 4ch", px_cm=0.15))
    assert abs(v.fps - 30.0) < 1e-6 and v.px_cm == 0.15 and v.view == "A4C" and v.frames.shape == (60, 96, 96)
    p = prepare(v)
    assert p.clips.shape[1:] == (CLIP_FRAMES, 112, 112) and len(p.frames) == 40 and abs(p.clips[0].mean()) < 1e-4
    with pytest.raises(EchoInputError, match="colour"):
        prepare(decode_dicom(E.dicom_bytes(clips["A4C"], colour=True)))
    with pytest.raises(EchoInputError, match="too short"):
        prepare(Video(clips["A4C"][:20], fps=25.0))
    with pytest.raises(EchoInputError, match="single-frame"):
        decode_dicom(E.dicom_bytes(clips["A4C"][:1]))


def test_list_clips_and_zip(tmp_path):
    from echo_wm_training.io import list_clips, read_zip_clips
    _, _, clips = E.study("normal", seed=2, views=("A4C", "A2C"), n_frames=44)
    (tmp_path / "a.dcm").write_bytes(E.dicom_bytes(clips["A4C"], view_tag="A4C"))
    (tmp_path / "b.dcm").write_bytes(E.dicom_bytes(clips["A2C"]))
    (tmp_path / "c.dcm").write_bytes(E.dicom_bytes(clips["A2C"], colour=True))
    (tmp_path / "d.dcm").write_bytes(E.dicom_bytes(clips["A2C"][:3]))
    t = list_clips(str(tmp_path), "s1").set_index("clip_uri")
    kinds = {Path(k).name: v for k, v in t.kind.items()}
    assert kinds == {"a.dcm": "cine", "b.dcm": "cine", "c.dcm": "colour", "d.dcm": "still"}
    assert {Path(k).name: v for k, v in t.view_tag.items()}["a.dcm"] == "A4C"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in sorted(tmp_path.glob("*.dcm")):
            z.write(f, f.name)
    assert len(read_zip_clips(buf.getvalue())) == 2


# --------------------------------------------------------------------------- model, pipeline
def test_model_shapes_strict_load_and_fingerprint(tmp_path):
    m = build_model({"width": 0.5, "d": 32})
    with torch.no_grad():
        out = m(torch.randn(2, CLIP_FRAMES, 112, 112))
        st = m.study_outputs(out, view_index(["A2C", "other"]))
    assert out["slots"].shape == (2, 3, 7, 4) and out["view"].shape == (2, 4)
    assert int(st["seen"].sum()) == 7 and 0 < float(st["lvef"]) < 100
    torch.save(m.state_dict(), tmp_path / "w.pt")
    load_model(str(tmp_path / "w.pt"), {"width": 0.5, "d": 32})
    with pytest.raises(RuntimeError):
        load_model(str(tmp_path / "w.pt"), {"width": 1.0, "d": 32})
    fp = runtime.fingerprint()
    copy = tmp_path / "rt"
    shutil.copytree(Path(runtime.__file__).parent, copy)
    assert runtime.fingerprint(copy) == fp
    (copy / "segments.py").write_text((copy / "segments.py").read_text().replace("LAD", "LAD "))
    assert runtime.fingerprint(copy) != fp


def test_pipeline_views_and_rules():
    m = build_model({"width": 0.5, "d": 32})
    _, _, clips = E.study("lad", seed=3, views=("A4C", "A2C", "other"), n_frames=44)
    a = analyse(m, [(Video(clips["A4C"], fps=25.0), "A4C"), (Video(clips["A2C"], fps=25.0), "A2C"),
                    (Video(clips["other"], fps=25.0), None)])
    f = a["findings"]
    assert f["assessed_segments"] == 13 and set(f["segments"]) == {str(s) for s in VIEW_SEGMENTS["A4C"] + VIEW_SEGMENTS["A2C"]}
    assert any("A3C" in n for n in a["notes"]) and f["territories"]["LCx"]["assessed_segments"] == 3
    used = {c["clip"]: c["used_as"] for c in f["views"]}
    assert used[1] == "A4C" and used[2] == "A2C"
    th = {"wma_any": 0.0, "terr_lad": 0.0, "terr_rca": 1.1, "terr_lcx": 1.1, "global_hypokinesis": 1.1}
    v = interpret(a, th, CFG["red_flags"])
    assert v["flags"]["terr_lad"] and not v["flags"]["terr_rca"] and v["pattern"] == "regional"
    names = {r["flag"] for r in v["red_flags"]}
    assert "regional_wma" in names and "multi_territory" not in names
    with pytest.raises(EchoInputError, match="no apical 4-, 2- or 3-chamber"):
        analyse(m, [(Video(clips["other"], fps=25.0), None)])            # untrained view head is never confident
    with pytest.raises(EchoInputError, match="no usable apical cine"):
        analyse(m, [(Video(clips["A4C"][:10], fps=25.0), "A4C")])


# --------------------------------------------------------------------------- training bundle + serving
def _synthetic_manifest(n=8, n_frames=44):
    data = E.dataset(n, seed=5, n_frames=n_frames)
    rows, clips, store = [], [], {}
    for i, (text, lab, cl) in enumerate(data):
        split = "train" if i < n - 3 else ("tune" if i < n - 1 else "test_time")
        rows.append({"study_id": f"s{i}", "person_id": i, "split": split, "report_text": text, "study_date": "2019-06-01",
                     **{k: lab[k] for k in label_columns()}, "lvef": lab["report_lvef"]})
        for v, f in cl.items():
            uri = f"mem://s{i}/{v}"
            store[uri] = Video(f, fps=25.0)
            clips.append({"study_id": f"s{i}", "clip_uri": uri, "view": v if i % 2 == 0 else None,
                          "view_source": "label" if i % 2 == 0 else None})
    return pd.DataFrame(rows), pd.DataFrame(clips), store


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    from echo_wm_training.train import train
    m, c, store = _synthetic_manifest()
    cfg = json.loads(json.dumps(CFG))
    cfg["arch"] = {"width": 0.35, "d": 32, "dropout": 0.0}
    cfg["train"].update({"view_epochs": 1, "epochs": 1, "accumulate_studies": 2})
    cfg["views"]["min_view_prob"] = 0.0
    out = tmp_path_factory.mktemp("echo_wm_bundle")
    card = train(m, c, cfg, str(out), loader=lambda u: store[u], device="cpu", log=lambda s: None, n_boot=10)
    return out, card, store


def test_training_bundle(bundle):
    out, card, _ = bundle
    assert {p.name for p in out.iterdir()} >= {"weights.pt", "thresholds.json", "model_card.json"}
    assert card["runtime_fingerprint"] == runtime.fingerprint() and card["uses_outside_data"] is False
    assert set(json.loads((out / "thresholds.json").read_text())) == set(CFG["labels"]["heads"])
    assert card["red_flags"] == CFG["red_flags"] and card["label_definitions"]["parser"].endswith("parse_report")
    sources = {r["view_source"] for r in card["clip_views"]}
    assert "model" in sources and "label" in sources                         # unlabelled clips got model views
    assert any(k.endswith("_quantitative") for k in card["metrics"])


def test_serving_container(bundle, tmp_path):
    from echo_wm_training.serve.predictor import BundleError, EchoWMPredictor, create_app
    out, _, store = bundle
    pred = EchoWMPredictor(out, device="cpu", allowed_prefixes=[str(tmp_path)])
    f4, f2 = store["mem://s0/A4C"].frames, store["mem://s0/A2C"].frames
    (tmp_path / "a2c.dcm").write_bytes(E.dicom_bytes(f2))
    import base64
    inst = {"clips": [{"video_b64": base64.b64encode(E.dicom_bytes(f4)).decode(), "view": "A4C"},
                      {"uri": str(tmp_path / "a2c.dcm"), "view": "A2C"}]}
    app = create_app(pred).test_client()
    r = app.post("/predict", json={"instances": [inst, {"clips": [{"uri": "/etc/passwd"}]}, {"clips": [{"view": "PLAX", "uri": "x"}]}]})
    p = r.get_json()["predictions"]
    assert p[0]["runtime_fingerprint"] == runtime.fingerprint() and set(p[0]["flags"]) == set(CFG["labels"]["heads"])
    assert p[0]["findings"]["assessed_segments"] == 13 and len(p[0]["findings"]["segments"]) == 13
    assert p[1]["error_type"] == "input" and "allowed prefix" in p[1]["error"]
    assert p[2]["error_type"] == "input" and "view must be" in p[2]["error"]
    assert app.get("/health").get_json()["weights_sha256"] == pred.weights_sha256
    card = json.loads((out / "model_card.json").read_text())
    bad = tmp_path / "bad"
    shutil.copytree(out, bad)
    (bad / "model_card.json").write_text(json.dumps({**card, "runtime_fingerprint": "0" * 64}))
    with pytest.raises(BundleError, match="runtime"):
        EchoWMPredictor(bad, device="cpu")


# --------------------------------------------------------------------------- MCP files, splits, views, pretraining
def test_mcp_request_reference_and_splits():
    from echo_wm_training import mcp_echo as me
    studies = pd.DataFrame({"person_id": [1, 2, 3], "study_uid": ["u1", "u2", "u3"], "patient_dk": ["d1", "d2", "d3"],
                            "stmo": ["s", "s", "s"], "modality": ["US"] * 3, "study_datetime": ["2019-01-01"] * 3,
                            "n_deid_ok": [80, 5, 90],
                            "report_text": [E.REPORTS["lad"].format(ef=35), E.REPORTS["normal"].format(ef=60), "Normal LVEF."]})
    req, rep = me.build_request(studies)
    assert req.study_uid.tolist() == ["u1"] and rep == {"studies": 3, "not_deidentifiable": 1, "no_wall_motion_statement": 1, "requested": 1}
    d = me.deid_request(studies, req)
    assert list(d.columns) == ["STUDY_INSTANCE_UID", "PATIENT_DK", "MODALITY", "STMO_NAME_IN_ADL"] and d.MODALITY[0] == "US"
    ref = me.reference_table(studies)
    assert ref.ref_wall_motion.tolist()[:2] == ["abnormal", "normal"] and ref.terr_lad[0] == 1.0
    sql = me.echo_studies_sql({"echo": {"mcp": {"schema": "{modality}", "echo_test_table": "FACT_ECHO_TEST",
                                                "inventory_table": "FACT_RADIOLOGY_DICOM_INVENTORY", "inventory_link": "DICOM_ACCESSION_NUMBER",
                                                "deid_ok": ["Ready"], "echo_test_columns": {"link": "ACCESSION_NBR", "datetime": "ECHO_DTM",
                                                                                         "note": "ECHO_NOTE", "patient_dk": "PATIENT_DK"}}}})
    assert "{{modality}}.FACT_ECHO_TEST" in sql and "MODALITY = 'US'" in sql
    ev = pd.DataFrame({"person_id": [1, 1, 2, 3], "study_id": ["a", "b", "c", "d"], "site": [1, 1, 1, 1],
                       "study_date": ["2018-01-01", "2022-01-01", "2020-06-01", "2023-01-01"],
                       "report_text": [E.REPORTS["rca"].format(ef=50)] * 4})
    from ecg_training.dataset import assign_splits
    cfg = {**CFG, "splits": {**CFG["splits"], "heldout_site_fraction": 0.0}}
    ms = assign_splits(make_labels(ev, cfg), cfg, exclude_persons=[3], date_col="study_date")
    assert dict(zip(ms.study_id, ms.split)) == {"a": "train", "c": "tune", "d": "test_time"}
    assert ms.terr_rca.eq(1.0).all() and ms.lvef.eq(50).all()


def test_view_sheet_png_and_assignment(tmp_path):
    from echo_wm_training.views import assign_views, write_png
    write_png(tmp_path / "x.png", (np.arange(64 * 32) % 255).reshape(32, 64))
    assert (tmp_path / "x.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    c = pd.DataFrame({"clip_uri": ["a", "b", "c", "d"], "view_tag": [None, "A2C", None, None]})
    lab = pd.DataFrame({"clip_uri": ["a"], "view": ["A3C"]})
    pred = pd.DataFrame({"clip_uri": ["a", "c", "d"], "view": ["A4C", "A4C", "other"], "view_p": [0.99, 0.95, 0.5]})
    v = assign_views(c, lab, pred, min_prob=0.9).set_index("clip_uri")
    assert v.view.to_dict() == {"a": "A3C", "b": "A2C", "c": "A4C", "d": None}
    assert v.view_source.to_dict() == {"a": "label", "b": "tag", "c": "model", "d": None}


def test_pretrain_training_period_only(tmp_path):
    from echo_wm_training.pretrain import pretrain
    m, c, store = _synthetic_manifest(n=4, n_frames=44)
    cfg = json.loads(json.dumps(CFG))
    cfg["arch"] = {"width": 0.35, "d": 32}
    cfg["pretrain"].update({"epochs": 1, "batch_size": 8, "frames_per_clip": 2})
    card = pretrain(m, c, cfg, str(tmp_path / "ssl"), loader=lambda u: store[u], device="cpu", log=lambda s: None)
    assert (tmp_path / "ssl" / "encoder.pt").exists() and card["uses_outside_data"] is False
    from echo_wm_training.train import load_init
    info = load_init(build_model(cfg["arch"]), str(tmp_path / "ssl"), lambda s: None)
    assert info["scope"] == "encoder" and info["uses_outside_data"] is False
    late = m.assign(study_date="2020-03-01")
    with pytest.raises(AssertionError, match="tune_start"):
        pretrain(late, c, cfg, str(tmp_path / "ssl2"), loader=lambda u: store[u], device="cpu", log=lambda s: None)


def test_label_audit_supports_echo():
    pytest.importorskip("openpyxl")
    from annotation import label_audit as L
    from annotation.auto_audit import AutoAuditor
    from annotation.select import load_tasks
    cfg = load_tasks()["label_audit"]
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        k = str(rng.choice(list(E.REPORTS)))
        t = E.REPORTS[k].format(ef=E.EF[k])
        rows.append({"study_id": f"s{i}", "person_id": i, "report_text": t, **{c: parse_report(t)[c] for c in label_columns()}})
    key = L.sample(pd.DataFrame(rows), "echo", ["ds1", "ds2"], cfg, auto="triage")
    assert key.auto_settled.sum() > 0.5 * len(key) and (key.auditor_1 != "").sum() < len(key)
    assert {r[0] for r in L.rule_text("echo")} >= {"wma_any", "terr_lad", "terr_rca", "terr_lcx", "global_hypokinesis"}
    aa = AutoAuditor("echo", cfg)
    assert aa.read("Possible hypokinesis of the inferior wall.")["terr_rca"] == "U"
    assert aa.read("Normal wall motion. LVEF 60%.") == {"wma_any": 0.0, "global_hypokinesis": 0.0, "terr_lad": 0.0,
                                                         "terr_rca": 0.0, "terr_lcx": 0.0, "report_lvef": 60.0}
