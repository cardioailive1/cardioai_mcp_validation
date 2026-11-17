"""ECG 4 (proposed 2026-11-17): the cardiologist approves or rejects the acute MI / STEMI training labels from a
per-ECG timeline in a browser page; 10 planted errors; pass at 95% approved per labelled group."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
pytest.importorskip("duckdb")

from annotation import ecg_mi_review as E  # noqa: E402
from ecg_training.dataset import load_config  # noqa: E402

LAB = load_config()


def _manifest(n=600, seed=0):
    rng = np.random.default_rng(seed)
    acute = rng.choice([1.0, 0.0, np.nan], n, p=[0.3, 0.5, 0.2])
    stemi = np.where(acute == 1, rng.choice([1.0, 0.0, np.nan], n, p=[0.4, 0.4, 0.2]), np.where(acute == 0, 0.0, np.nan))
    return pd.DataFrame({"ecg_id": [str(i) for i in range(n)], "person_id": np.arange(n), "acute_mi": acute, "stemi": stemi,
                         "ecg_date": pd.Timestamp("2022-03-01"), "read_text": "Sinus rhythm. ST elevation anterior leads."})


def test_timeline_marks_type2_window_and_tni_without_limit():
    ecg = pd.Series({"ecg_date": "2022-03-01", "read_text": "Acute anterior STEMI."})
    conds = pd.DataFrame({"condition_start_date": ["2022-03-02", "2022-03-01", "2022-02-01"], "vocabulary_id": "ICD10CM",
                          "concept_code": ["I21.09", "I21.A1", "I25.2"], "concept_name": ["STEMI anterior", "Type 2 MI", "Old MI"],
                          "is_mi_acute": [1, 1, 0], "is_stemi": [1, 0, 0], "is_nstemi": [0, 0, 0], "is_type2": [0, 1, 0]})
    trops = pd.DataFrame({"measurement_date": ["2022-03-01", "2022-03-01"], "value_as_number": [80.0, 30.0],
                          "unit_source_value": ["ng/L", "ng/L"], "range_high": [np.nan, np.nan], "is_tnt": [1, 0],
                          "gender_concept_id": [8507, 8507]})
    t = E.timeline(ecg, conds, trops, LAB["labels"])
    assert "TYPE 2: not counted" in t["text"] and "[acute MI, STEMI]" in t["text"] and "[old / other MI]" in t["text"]
    assert "(limit 15, Mayo TnT limit)  ABOVE" in t["text"] and "(no limit: not judged)" in t["text"]
    assert len(t["highlight"]) == 3                       # STEMI code, type 2 code (in window), TnT above


def test_sql_runs_on_omop_and_the_review_scores(tmp_path):
    from test_ecg_mayo_formats import _omop_with_ecgs
    from cardioai_mcp.concepts import resolve_concept_sets
    from annotation.label_audit import troponin_rows_sql
    con, st, eng = _omop_with_ecgs("'x'")
    resolve_concept_sets(eng, st["concept_sets"], only=E.CONCEPT_SETS)
    ids = [r[0] for r in con.execute("SELECT ecg_id FROM ecg_index LIMIT 200").fetchall()]
    conds = eng.query(eng.render(E.conditions_sql(st["modality"], ids), {}))
    trops = eng.query(eng.render(troponin_rows_sql(st["modality"], ids), {}))
    assert {"concept_code", "is_type2", "is_stemi"} <= set(conds.columns) and len(trops)

    m = _manifest()
    meta = E.build(m, conds.assign(ecg_id="0"), trops.assign(ecg_id="0"), LAB, str(tmp_path))
    assert meta["groups"] == {**E.GROUPS, "planted": 10}
    html = (tmp_path / "ecg_mi_review.html").read_text()
    data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1))
    key = pd.read_csv(tmp_path / "key.csv", dtype={"planted": str}).fillna({"planted": ""})
    assert len(data["rows"]) == len(key) == 160 and data["mono"]
    shown = {r["review_id"]: dict(r["values"]) for r in data["rows"]}

    def word(v):
        return E._word(v)
    dec = []                                              # a careful cardiologist: reject what is shown wrong
    for _, k in key.iterrows():
        ok = shown[k.review_id]["Acute MI label"] == word(k.acute_mi) and shown[k.review_id]["STEMI label"] == word(k.stemi)
        dec.append({"review_id": k.review_id, "decision": "approve" if ok else "reject", "note": "" if ok else "label flipped"})
    out = {"sheet_id": data["sheet_id"], "signoff": {"cardiologist": "Tamanna Nahar", "date": "2026-11-17", "decision": "approve"},
           "decisions": dec}
    (tmp_path / "review_decisions.json").write_text(json.dumps(out))
    r = E.score(str(tmp_path))
    assert r["passed"], r["results"].to_string()
    assert json.loads((tmp_path / "signoff.json").read_text())["protocol"] == "ECG 4"

    for x in dec:                                         # approving everything misses the planted errors
        x["decision"], x["note"] = "approve", ""
    (tmp_path / "review_decisions.json").write_text(json.dumps(out))
    res = E.score(str(tmp_path))["results"].set_index("check")
    assert res.loc["planted errors rejected", "result"].startswith("REVIEW AGAIN")
