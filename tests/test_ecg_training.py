"""ECG model (MobileNetV3-1D): preprocessing, readers, labels/splits, export safety, and training.

Needs torch (see requirements-ecg.txt); skipped cleanly when it is not installed.
"""
import base64
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

import synthetic_ecg as S  # noqa: E402
from ecg_training import runtime  # noqa: E402
from ecg_training.dataset import assign_splits, check_disjoint, load_config, make_labels  # noqa: E402
from ecg_training.io import read_muse_xml  # noqa: E402
from ecg_training.runtime.model import MobileNetV3ECG, build_model, load_model, n_params  # noqa: E402
from ecg_training.runtime.preprocess import ECGInputError, N_SAMPLES, preprocess  # noqa: E402
from ecg_training.train import fit_threshold, train  # noqa: E402


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


# --------------------------------------------------------------------------- preprocessing
def test_output_contract_and_sample_rate_invariance():
    x500 = preprocess(S.ecg("stemi", fs=500, seed=5), 500, S.LEADS)
    assert x500.shape == (12, N_SAMPLES) and x500.dtype == np.float32
    for fs in (250, 1000):
        assert corr(x500, preprocess(S.ecg("stemi", fs=fs, seed=5), fs, S.LEADS)) > 0.97


def test_eight_lead_storage_derives_limb_leads_exactly():
    sig = S.ecg("normal", seed=2)
    keep = [0, 1, 6, 7, 8, 9, 10, 11]                     # GE MUSE stores I, II, V1-V6
    x8 = preprocess(sig[keep], 500, [S.LEADS[i] for i in keep])
    assert np.allclose(x8, preprocess(sig, 500, S.LEADS), atol=1e-5)


def test_units_and_lead_order_and_refusals():
    sig = S.ecg("af", seed=3)
    ref = preprocess(sig, 500, S.LEADS)
    assert np.allclose(preprocess(sig * 1000, 500, S.LEADS, units="uV"), ref, atol=1e-5)
    perm = np.random.default_rng(0).permutation(12)
    assert np.allclose(preprocess(sig[perm], 500, [S.LEADS[i] for i in perm]), ref, atol=1e-5)
    assert np.allclose(preprocess(sig.T, 500, S.LEADS), ref, atol=1e-5)          # (N, 12) accepted
    for bad, kw in ((sig * 1000, {}), (sig[:, :4000], {}), (sig[:6], {"leads": S.LEADS[6:]})):
        with pytest.raises(ECGInputError):
            preprocess(bad, 500, kw.get("leads", S.LEADS))


def test_st_elevation_survives_filtering():
    """The 0.67 Hz zero-phase high-pass must not erase ST elevation (STEMI depends on it)."""
    st, norm = preprocess(S.ecg("stemi", seed=11), 500, S.LEADS), preprocess(S.ecg("normal", seed=11), 500, S.LEADS)
    assert st.std() > 0 and abs(st.mean() - norm.mean()) < 0.2       # no gross baseline drift added


def test_runtime_fingerprint_covers_preprocessing_and_model(tmp_path):
    for f in ("model.py", "preprocess.py"):
        (tmp_path / f).write_bytes((Path(runtime.__file__).parent / f).read_bytes())
    assert runtime.fingerprint(tmp_path) == runtime.fingerprint()
    (tmp_path / "preprocess.py").write_text((tmp_path / "preprocess.py").read_text().replace("HP_HZ, LP_HZ = 0.67", "HP_HZ, LP_HZ = 0.5"))
    assert runtime.fingerprint(tmp_path) != runtime.fingerprint()


# --------------------------------------------------------------------------- readers
def test_muse_xml_reader():
    sig = S.ecg("normal", seed=4)
    keep = [0, 1, 6, 7, 8, 9, 10, 11]
    leads = "".join(
        f"<LeadData><LeadID>{S.LEADS[i]}</LeadID><LeadAmplitudeUnitsPerBit>4.88</LeadAmplitudeUnitsPerBit>"
        f"<LeadAmplitudeUnits>MICROVOLTS</LeadAmplitudeUnits><WaveFormData>"
        f"{base64.b64encode(np.round(sig[i] * 1000 / 4.88).astype('<i2').tobytes()).decode()}</WaveFormData></LeadData>"
        for i in keep)
    xml = f"<RestingECG><Waveform><WaveformType>Median</WaveformType></Waveform><Waveform><WaveformType>Rhythm</WaveformType><SampleBase>500</SampleBase>{leads}</Waveform></RestingECG>"
    s, fs, names, units = read_muse_xml(xml.encode())
    assert fs == 500 and names == [S.LEADS[i] for i in keep] and units == "uV"
    assert corr(preprocess(s, fs, names, units), preprocess(sig, 500, S.LEADS)) > 0.999


# --------------------------------------------------------------------------- labels and splits
def evidence(n=400, seed=0):
    rnd = np.random.default_rng(seed)
    pid = rnd.integers(0, 150, n)
    return pd.DataFrame({
        "person_id": pid, "ecg_id": [f"e{i}" for i in range(n)], "waveform_uri": [f"u{i}" for i in range(n)],
        "ecg_date": pd.to_datetime("2016-01-01") + pd.to_timedelta(rnd.integers(0, 3000, n), "D"),
        "site": pid % 9, "confirmed": 1,
        "read_text": rnd.choice(["Sinus rhythm.", "Atrial fibrillation with RVR.", "No evidence of atrial fibrillation.",
                                 "History of atrial flutter. Sinus rhythm.", None], n),
        "mi_dx": rnd.integers(0, 2, n), "stemi_dx": rnd.integers(0, 2, n), "nstemi_dx": rnd.integers(0, 2, n),
        "trop_pos": rnd.integers(0, 2, n), "mi_any_near": rnd.integers(0, 2, n)})


def test_label_rules():
    cfg = load_config()
    d = make_labels(evidence(), cfg)
    af = dict(zip(d.read_text, d.af))
    assert af["Atrial fibrillation with RVR."] == 1 and af["Sinus rhythm."] == 0
    assert af["No evidence of atrial fibrillation."] == 0 and af["History of atrial flutter. Sinus rhythm."] == 0
    assert d.loc[d.read_text.isna(), "af"].isna().all()
    pos = (d.mi_dx == 1) & (d.trop_pos == 1)
    assert (d.loc[pos, "acute_mi"] == 1).all() and (d.loc[~pos & (d.mi_any_near == 0), "acute_mi"] == 0).all()
    assert d.loc[~pos & (d.mi_any_near == 1), "acute_mi"].isna().all()               # ambiguous -> masked
    assert (d.loc[d.stemi == 1, "acute_mi"] == 1).all()                              # STEMI implies acute MI


def test_splits_patient_disjoint_time_ordered_and_exclusions():
    cfg = load_config()
    cfg["splits"]["heldout_site_fraction"] = 0.25
    d = make_labels(evidence(1200, seed=1), cfg)
    excluded = set(d.person_id.unique()[:10])
    m = assign_splits(d, cfg, exclude_persons=excluded)
    assert all(v == 0 for v in check_disjoint(m).values())
    tune_start, test_start = pd.Timestamp(cfg["splits"]["tune_start"]), pd.Timestamp(cfg["splits"]["test_start"])
    assert (m[m.split == "train"].ecg_date < tune_start).all()
    assert m[m.split == "tune"].ecg_date.between(tune_start, test_start, inclusive="left").all()
    assert (m[m.split == "test_time"].ecg_date >= test_start).all()
    assert not set(m[m.split.isin(["train", "tune"])].person_id) & excluded
    assert m[m.split.isin(["train", "tune"])].groupby("person_id").size().max() <= cfg["splits"]["max_ecgs_per_patient"]


def test_label_sql_runs_on_omop():
    from synthetic_omop import build
    from cardioai_mcp.config import load_study
    from cardioai_mcp.engine import DuckDBEngine
    from ecg_training.dataset import build_manifest
    st = load_study()
    st["schemas"].update(cdm="main", vocab="main", modality="main", work="main")
    con = build(2500)
    con.execute("ALTER TABLE measurement ADD COLUMN range_high DOUBLE")
    con.execute("""CREATE TABLE ecg_index AS
        SELECT ROW_NUMBER() OVER () AS ecg_id, po.person_id, CAST(po.procedure_date AS TIMESTAMP) AS acquisition_datetime,
               'synthetic/' || CAST(po.person_id AS VARCHAR) || '.npz' AS waveform_path, 500 AS sample_rate, 12 AS lead_count,
               (SELECT MAX(n.note_text) FROM note n WHERE n.person_id = po.person_id AND n.note_date = po.procedure_date
                  AND n.note_title ILIKE '%ecg%') AS diagnosis_statement, 1 AS confirmed_flag
        FROM procedure_occurrence po
        WHERE po.procedure_source_concept_id IN (SELECT concept_id FROM concept WHERE concept_code IN ('93000','93005','93010'))
           OR po.procedure_concept_id IN (SELECT concept_id FROM concept WHERE concept_code IN ('93000','93005','93010'))""")
    eng = DuckDBEngine(con, st)
    from cardioai_mcp.concepts import resolve_concept_sets
    resolve_concept_sets(eng, st["concept_sets"], only={"troponin"})
    con.execute("""UPDATE measurement SET range_high = 14 WHERE measurement_concept_id IN (SELECT concept_id FROM cs_members)
                   OR measurement_source_concept_id IN (SELECT concept_id FROM cs_members)""")
    cfg = load_config()
    cfg["splits"]["heldout_site_fraction"] = 0.2
    m = build_manifest(eng, st, cfg, st["modality"])
    assert len(m) > 1000 and set(m.split) >= {"train", "test_site"}
    assert (m.af == 1).sum() > 50 and (m.acute_mi == 1).sum() > 20 and (m.stemi == 1).sum() > 20


# --------------------------------------------------------------------------- export safety
def test_strict_loading_refuses_wrapped_or_mismatched_checkpoints(tmp_path):
    m = MobileNetV3ECG(width=0.25)
    arch = {"width": 0.25}
    torch.save(m.state_dict(), tmp_path / "ok.pt")
    torch.save({"epoch": 3, "model": m.state_dict()}, tmp_path / "wrapped.pt")
    torch.save(MobileNetV3ECG(width=0.5).state_dict(), tmp_path / "other.pt")
    loaded = load_model(str(tmp_path / "ok.pt"), arch)
    x = torch.randn(2, 12, 5000)
    with torch.no_grad():
        assert torch.allclose(loaded(x), m.eval()(x), atol=1e-5)
    with pytest.raises(TypeError):
        load_model(str(tmp_path / "wrapped.pt"), arch)
    with pytest.raises(RuntimeError):
        load_model(str(tmp_path / "other.pt"), arch)
    assert n_params(MobileNetV3ECG()) < 1.2e6                  # small MobileNetV3 (export-approved family)


def test_threshold_rule_meets_target_on_tuning():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(200), np.zeros(800)]
    p = np.r_[rng.beta(5, 2, 200), rng.beta(2, 5, 800)]
    t, _ = fit_threshold(y, p, 0.96)
    assert (p[y == 1] >= t).mean() >= 0.96
    assert fit_threshold(np.r_[np.ones(3), np.zeros(10)], np.linspace(0, 1, 13), 0.9)[0] == 0.5   # too few positives


# --------------------------------------------------------------------------- training end to end
def test_training_learns_and_writes_a_lockable_bundle(tmp_path):
    cache, rows = {}, []
    for split, n, seed in (("train", 320, 1), ("tune", 120, 2), ("test_time", 120, 3)):
        for i, (sig, fs, lab, kind) in enumerate(S.dataset(n, seed=seed)):
            uri = f"{split}/{i}"
            cache[uri] = sig
            rows.append(dict(ecg_id=uri, person_id=seed * 10000 + i, waveform_uri=uri, split=split, ecg_date="2019-06-01",
                             acute_mi=float(lab[0]), stemi=float(lab[1]), af=float(lab[2])))
    m = pd.DataFrame(rows)
    cfg = load_config()
    cfg["arch"]["width"] = 0.25
    cfg["train"].update(epochs=6, batch_size=32, num_workers=0, patience=6)
    card = train(m, cfg, str(tmp_path), loader=lambda u: (cache[u], 500.0, S.LEADS, "mV"), device="cpu", log=lambda s: None)
    test = pd.DataFrame(card["metrics"]["test_time"]).set_index("head")
    assert test.loc["af", "auroc"] > 0.9 and test.loc["stemi", "auroc"] > 0.9 and test.loc["acute_mi", "auroc"] > 0.85
    for f in ("weights.pt", "thresholds.json", "model_card.json"):
        assert (tmp_path / f).exists()
    assert card["runtime_fingerprint"] == runtime.fingerprint()
    assert json.loads((tmp_path / "thresholds.json").read_text()) == card["thresholds"]
    reloaded = load_model(str(tmp_path / "weights.pt"), {**cfg["arch"], "heads": cfg["heads"]})
    assert n_params(reloaded) == card["parameters"]


# --------------------------------------------------------------------------- serving container
def _bundle(tmp_path, cfg):
    """A bundle in the format train() writes (weights.pt, thresholds.json, model_card.json)."""
    m = MobileNetV3ECG(width=0.25).eval()
    torch.save(m.state_dict(), tmp_path / "weights.pt")
    import hashlib
    (tmp_path / "thresholds.json").write_text(json.dumps({"acute_mi": 0.5, "stemi": 0.5, "af": 0.5}))
    (tmp_path / "model_card.json").write_text(json.dumps({
        "model_name": "cardioai-ecg-mobilenetv3", "heads": ["acute_mi", "stemi", "af"], "arch": {"width": 0.25},
        "runtime_fingerprint": runtime.fingerprint(),
        "weights_sha256": hashlib.sha256((tmp_path / "weights.pt").read_bytes()).hexdigest()}))
    return m


def test_serving_container_matches_training_and_refuses_mismatches(tmp_path):
    from ecg_training.serve.predictor import BundleError, ECGPredictor
    cfg = load_config()
    m = _bundle(tmp_path, cfg)
    pred = ECGPredictor(tmp_path, device="cpu")
    sig = S.ecg("af", fs=250, seed=9)
    out = pred.predict([{"signal": sig.tolist(), "fs": 250, "leads": S.LEADS, "units": "mV"},
                        {"signal": (sig * 1000).tolist(), "fs": 250, "leads": S.LEADS},          # uV sent as mV -> refused
                        {"fs": 500}])
    with torch.no_grad():
        expect = torch.sigmoid(m(torch.from_numpy(preprocess(sig, 250, S.LEADS))[None]))[0].numpy()
    assert np.allclose([out[0]["probabilities"][h] for h in ("acute_mi", "stemi", "af")], expect, atol=1e-5)
    assert out[0]["runtime_fingerprint"] == runtime.fingerprint() and len(out[0]["weights_sha256"]) == 64
    assert out[1]["error_type"] == "input" and out[2]["error_type"] == "input"
    # tampered weights or a different runtime -> the container refuses to start
    card = json.loads((tmp_path / "model_card.json").read_text())
    (tmp_path / "model_card.json").write_text(json.dumps({**card, "runtime_fingerprint": "0" * 64}))
    with pytest.raises(BundleError, match="runtime"):
        ECGPredictor(tmp_path, device="cpu")
    (tmp_path / "model_card.json").write_text(json.dumps({**card, "weights_sha256": "0" * 64}))
    with pytest.raises(BundleError, match="weights"):
        ECGPredictor(tmp_path, device="cpu")


def test_serving_http_contract(tmp_path):
    pytest.importorskip("flask")
    from ecg_training.serve.predictor import ECGPredictor, create_app
    _bundle(tmp_path, load_config())
    client = create_app(ECGPredictor(tmp_path, device="cpu")).test_client()
    assert client.get("/health").json["runtime_fingerprint"] == runtime.fingerprint()
    r = client.post("/predict", json={"instances": [{"signal": S.ecg("normal").tolist(), "fs": 500, "leads": S.LEADS}]})
    p = r.json["predictions"][0]
    assert r.status_code == 200 and set(p["probabilities"]) == {"acute_mi", "stemi", "af"} and set(p["flags"]) == set(p["probabilities"])
    assert client.post("/predict", json={}).status_code == 400
