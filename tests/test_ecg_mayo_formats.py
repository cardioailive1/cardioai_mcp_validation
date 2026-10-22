"""Mayo specifics confirmed by MCP (2026-10-14): ECG waveforms stored as Base64 strings in the waveform
table; troponin = Troponin T 5th generation in ng/L with sex-specific upper limits (M <= 15, F <= 10)."""
import base64
import gzip
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
pytest.importorskip("scipy")

import synthetic_ecg as S  # noqa: E402
from ecg_training.io import decode_waveform, read_any  # noqa: E402
from ecg_training.probe_waveforms import describe  # noqa: E402

RAW = {"dtype": "<i2", "units_per_bit": 4.88, "units": "uV", "layout": "auto"}


def _counts(sig_mv):
    return np.round(sig_mv * 1000 / 4.88).astype("<i2")


def _pre(rec):
    from ecg_training.runtime.preprocess import preprocess
    return preprocess(*rec)


def _corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


@pytest.fixture(scope="module")
def sig():
    return S.ecg("stemi", fs=500, seed=21)


def test_raw_int16_lead_major_and_interleaved_are_detected(sig):
    pytest.importorskip("torch")
    ref = _pre((sig, 500.0, S.LEADS, "mV"))
    c = _counts(sig)
    for layout, flat in (("lead_major", c.ravel()), ("interleaved", c.T.ravel())):
        b64 = base64.b64encode(flat.tobytes()).decode()
        rec = decode_waveform(b64, n_leads=12, fs=500)
        assert rec[2] == S.LEADS and rec[3] == "uV"
        assert _corr(_pre(rec), ref) > 0.999, layout
        d = describe(b64, 12, 500, RAW)
        assert d["container"] == "raw" and d["layout"] == layout and d["seconds"] == 10.0 and 0.3 < d["p99_abs_mv"] < 5


def test_wrapped_containers_and_per_lead_columns(sig):
    pytest.importorskip("torch")
    ref = _pre((sig, 500.0, S.LEADS, "mV"))
    raw = _counts(sig).ravel().tobytes()
    gz = base64.b64encode(gzip.compress(raw)).decode()
    assert _corr(_pre(decode_waveform(gz, n_leads=12, fs=500)), ref) > 0.999
    assert describe(gz, 12, 500, RAW)["container"] == "gzip+raw"
    buf = io.BytesIO()
    np.savez(buf, signal=sig, fs=500.0, leads=np.asarray(S.LEADS), units="mV")
    assert _corr(_pre(decode_waveform(base64.b64encode(buf.getvalue()).decode())), ref) > 0.999
    js = base64.b64encode(json.dumps({"fs": 500, "units": "mV", **{ld: sig[i].tolist() for i, ld in enumerate(S.LEADS)}}).encode()).decode()
    assert _corr(_pre(decode_waveform(js)), ref) > 0.999
    per_lead = {ld: base64.b64encode(_counts(sig)[i].tobytes()).decode() for i, ld in enumerate(S.LEADS)}
    assert _corr(_pre(decode_waveform(per_lead, fs=500)), ref) > 0.999
    keep = [0, 1, 6, 7, 8, 9, 10, 11]
    xml = ("<RestingECG><Waveform><WaveformType>Rhythm</WaveformType><SampleBase>500</SampleBase>" + "".join(
        f"<LeadData><LeadID>{S.LEADS[i]}</LeadID><LeadAmplitudeUnitsPerBit>4.88</LeadAmplitudeUnitsPerBit>"
        f"<LeadAmplitudeUnits>MICROVOLTS</LeadAmplitudeUnits><WaveFormData>{base64.b64encode(_counts(sig)[i].tobytes()).decode()}"
        "</WaveFormData></LeadData>" for i in keep) + "</Waveform></RestingECG>")
    uri = "data:application/xml;base64," + base64.b64encode(xml.encode()).decode()
    assert _corr(_pre(decode_waveform(uri)), ref) > 0.999                       # 8 MUSE leads -> 12


def test_refusals(sig):
    pytest.importorskip("torch")
    from ecg_training.runtime.preprocess import ECGInputError
    b64 = base64.b64encode(_counts(sig).ravel().tobytes()).decode()
    with pytest.raises(ValueError, match="lead count"):
        decode_waveform(b64)
    with pytest.raises(ValueError, match="Base64"):
        decode_waveform("not*base64!")
    with pytest.raises(ValueError, match="split into"):
        decode_waveform(base64.b64encode(_counts(sig).ravel()[:-1].tobytes()).decode(), n_leads=12, fs=500)
    with pytest.raises(ECGInputError):                                          # wrong scale -> implausible amplitude
        _pre(decode_waveform(b64, n_leads=12, fs=500, units_per_bit=1000.0))


def _omop_with_ecgs(waveform_sql):
    from synthetic_omop import build
    from cardioai_mcp.config import load_study
    from cardioai_mcp.engine import DuckDBEngine
    st = load_study()
    st["schemas"].update(cdm="main", vocab="main", modality="main", work="main")
    st["modality"]["ecg"]["waveform_storage"] = "base64"       # older layout: one Base64 string per ECG
    con = build(1500)
    con.execute("ALTER TABLE measurement ADD COLUMN range_high DOUBLE")
    con.execute(f"""CREATE TABLE ecg_index AS
        SELECT CAST(ROW_NUMBER() OVER () AS VARCHAR) AS ecg_id, po.person_id, CAST(po.person_id AS VARCHAR) AS clinic_number,
               CAST(po.procedure_date AS TIMESTAMP) AS acquisition_datetime, {waveform_sql} AS waveform_path,
               500 AS sample_rate, 12 AS lead_count, 'Sinus rhythm.' AS diagnosis_statement, 1 AS confirmed_flag
        FROM procedure_occurrence po
        WHERE po.procedure_source_concept_id IN (SELECT concept_id FROM concept WHERE concept_code IN ('93000','93005','93010'))""")
    return con, st, DuckDBEngine(con, st)


def test_troponin_uses_sex_specific_mayo_limits():
    from cardioai_mcp.concepts import resolve_concept_sets
    from ecg_training.dataset import evidence_sql, load_config
    con, st, eng = _omop_with_ecgs("'x'")
    resolve_concept_sets(eng, st["concept_sets"], only={"mi_acute", "mi_any", "stemi", "nstemi", "troponin"})
    trop = con.execute("SELECT concept_id FROM cs_members WHERE set_name = 'troponin' LIMIT 1").fetchone()[0]
    rows = con.execute("""SELECT e.ecg_id, e.person_id, CAST(e.acquisition_datetime AS DATE), p.gender_concept_id
                          FROM ecg_index e JOIN person p USING (person_id)""").fetchall()
    pick = {}
    for ecg_id, pid, d, g in rows:
        if g in (8507, 8532) and g not in pick:
            pick[g] = (ecg_id, pid, d)
    con.execute("DELETE FROM measurement WHERE person_id IN (?, ?)", [pick[8507][1], pick[8532][1]])
    for g, (ecg_id, pid, d) in pick.items():                                    # 12 ng/L, no range_high on the row
        con.execute("INSERT INTO measurement (measurement_id, person_id, measurement_concept_id, measurement_source_concept_id, "
                    "measurement_date, value_as_number, unit_source_value) VALUES (?, ?, ?, ?, ?, 12.0, 'ng/L')",
                    [990000 + g, pid, trop, trop, d])
    ev = eng.query(evidence_sql(eng, st, load_config(), st["modality"])).set_index("ecg_id")
    assert ev.loc[pick[8532][0], "trop_pos"] == 1                               # female: above 10 ng/L
    assert ev.loc[pick[8507][0], "trop_pos"] == 0                               # male: not above 15 ng/L
    assert ev.waveform_uri.isna().all()                                         # Base64 never pulled into the manifest


def test_extract_waveforms_from_base64_table(tmp_path):
    from ecg_training.extract_waveforms import extract_waveforms
    con, st, eng = _omop_with_ecgs("'pending'")
    ids = [r[0] for r in con.execute("SELECT ecg_id FROM ecg_index ORDER BY CAST(ecg_id AS INT) LIMIT 5").fetchall()]
    for k, i in enumerate(ids[:4]):
        b64 = base64.b64encode(_counts(S.ecg("af" if k % 2 else "normal", seed=k)).ravel().tobytes()).decode()
        con.execute("UPDATE ecg_index SET waveform_path = ? WHERE ecg_id = ?", [b64, i])
    con.execute("UPDATE ecg_index SET waveform_path = 'corrupt!!' WHERE ecg_id = ?", [ids[4]])
    m = pd.DataFrame({"ecg_id": ids + ["missing"], "person_id": range(6), "split": "train"})
    logs = []
    out = extract_waveforms(eng, m, st["modality"], str(tmp_path), batch_size=2, log=logs.append)
    ok = out[out.waveform_uri.notna()]
    assert len(ok) == 4 and out.set_index("ecg_id").loc["missing", "waveform_error"] == "not in waveform table"
    assert "Base64" in out.set_index("ecg_id").loc[ids[4], "waveform_error"]
    s, fs, leads, units = read_any(ok.waveform_uri.iloc[0])
    assert s.shape == (12, 5000) and fs == 500 and units == "uV"
    assert np.allclose(s / 1000, S.ecg("normal", seed=0), atol=0.01)
    again = extract_waveforms(eng, m, st["modality"], str(tmp_path), batch_size=2, log=logs.append)   # resumes: no re-decoding
    assert again.waveform_uri.notna().sum() == 4
