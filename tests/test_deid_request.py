"""The de-identification request MCP needs before releasing imaging (scripts/deid_request.py): exactly
STUDY_INSTANCE_UID, PATIENT_DK, MODALITY, STMO_NAME_IN_ADL, one row per study; built after the Phase 2 cohort, whose
index studies are always requested; PATIENT_DK from the unmerged DIM_PATIENT row; adults only; nothing written if
the file fails its checks."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
pytest.importorskip("duckdb")

from cardioai_mcp import deid  # noqa: E402
from test_mcp_imaging import _db  # noqa: E402

_spec = importlib.util.spec_from_file_location("deid_request", ROOT / "scripts" / "deid_request.py")
dr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dr)
WINDOW = ("1900-01-01", "2100-01-01")


def _cohort(con, name="mri_analytic_locked"):
    # Phase 2 MRI cohort: person 11 (study 1.1) and person 15 (study 1.5 is not de-identifiable)
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM (VALUES (11, DATE '2022-03-01'), (15, DATE '2022-03-01')) "
                "v(person_id, index_date)")


def _births(con):
    con.execute("ALTER TABLE person ADD COLUMN year_of_birth INTEGER")
    con.execute("ALTER TABLE person ADD COLUMN month_of_birth INTEGER")
    con.execute("ALTER TABLE person ADD COLUMN day_of_birth INTEGER")
    con.execute("UPDATE person SET year_of_birth = 1960, month_of_birth = 5, day_of_birth = 2")


def _run(eng, st, tmp_path, **kw):
    return dr.run(eng, st, ["mri"], str(tmp_path), *WINDOW, log=lambda s: None, **kw)


def test_refuses_before_the_cohort_exists(tmp_path):
    con, st, eng = _db()
    _births(con)
    with pytest.raises(SystemExit, match="Build the cohort first"):
        _run(eng, st, tmp_path)
    assert not (tmp_path / "deid_request_all.csv").exists()


def test_writes_the_mri_request(tmp_path):
    con, st, eng = _db()
    _births(con)
    _cohort(con)
    # C2 (study 1.2, 2022-03-01) is 17 at the study: not requested
    con.execute("UPDATE person SET year_of_birth = 2004, month_of_birth = 6, day_of_birth = 1 WHERE person_source_value = 'C2'")
    m = _run(eng, st, tmp_path)["mri"]
    with open(tmp_path / "deid_request_all.csv", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["STUDY_INSTANCE_UID", "PATIENT_DK", "MODALITY", "STMO_NAME_IN_ADL"]
    assert rows[1:] == [["1.1", "1000", "MR", "STMO1"]]                 # merged DIM_PATIENT row 999 not used
    assert (tmp_path / "deid_request_mri.csv").read_text() == (tmp_path / "deid_request_all.csv").read_text()
    s = json.loads((tmp_path / "deid_summary.json").read_text())["mri"]
    assert s == {**s, "studies_under_min_age": 1, "phase2_index_studies": 1, "training_studies": 0, "selected_studies": 1,
                 "phase2_patients_without_a_requestable_study": 1, "cohort_table": "mri_analytic_locked"}
    assert m["problems"] == [] and pd.read_parquet(tmp_path / "deid_selection_mri.parquet").purpose.tolist() == ["phase2_index"]


def test_saved_build_is_used_when_not_locked_and_dry_run_writes_nothing(tmp_path):
    con, st, eng = _db()
    _births(con)
    _cohort(con, "mri_analytic")
    m = _run(eng, st, tmp_path, dry_run=True)["mri"]
    assert m["cohort_table"] == "mri_analytic" and m["selected_studies"] == 2 and not any(tmp_path.iterdir())


def test_patient_key_ambiguous_or_missing_is_left_out_and_counted(tmp_path):
    con, st, eng = _db()
    _births(con)
    _cohort(con)
    con.execute("UPDATE DIM_PATIENT SET PATIENT_MERGED_FLAG = 'N' WHERE PATIENT_DK = 999")   # C1 now has two
    con.execute("DELETE FROM DIM_PATIENT WHERE PATIENT_CLINIC_NUMBER = 'C2'")                 # C2 has none
    m = _run(eng, st, tmp_path, dry_run=True)["mri"]
    assert (m["studies_with_several_patient_dk"], m["studies_without_patient_dk"], m["selected_studies"]) == (1, 1, 0)


def test_unknown_or_redacted_age_is_not_requested():
    con, st, eng = _db()
    _births(con)
    con.execute("UPDATE person SET year_of_birth = 1800 WHERE person_source_value = 'C2'")    # redacted birth year
    s = pd.DataFrame({"study_uid": ["1.1", "1.2", "1.3"], "person_id": [None, 12, 13],
                      "study_datetime": pd.to_datetime(["2022-03-01"] * 3)})
    kept, c = deid.adults_only(eng, st["modality"]["mri"]["mcp"], s, 18)
    assert kept.study_uid.tolist() == ["1.3"] and c["studies_age_unknown"] == 2


def test_validate():
    good = pd.DataFrame({"STUDY_INSTANCE_UID": ["1.2.840.1", "1.2.840.2"], "PATIENT_DK": [1, 2],
                         "MODALITY": ["MR", "US"], "STMO_NAME_IN_ADL": ["S", "S"]})
    assert deid.validate(good) == []
    assert any("MODALITY" in p for p in deid.validate(good, "mri"))
    bad = good.copy()
    bad.loc[1, "PATIENT_DK"] = None
    assert any("empty" in p for p in deid.validate(bad))
    assert any("duplicate" in p for p in deid.validate(pd.concat([good, good])))
    swapped = good[["PATIENT_DK", "STUDY_INSTANCE_UID", "MODALITY", "STMO_NAME_IN_ADL"]]
    assert any("columns" in p for p in deid.validate(swapped))
    assert any("UID" in p for p in deid.validate(good.assign(STUDY_INSTANCE_UID=["abc", "1.2"])))
