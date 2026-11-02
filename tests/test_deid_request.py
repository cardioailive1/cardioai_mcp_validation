"""The de-identification request MCP needs before releasing imaging: exactly STUDY_INSTANCE_UID, PATIENT_DK,
MODALITY, STMO_NAME_IN_ADL, one row per selected study; PATIENT_DK from DIM_PATIENT (unmerged row); adults only;
nothing written if the file fails its checks."""
import csv
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

sys.path.insert(0, str(ROOT / "scripts"))
import build_deid_request as bdr  # noqa: E402


def test_script_writes_the_mri_request(tmp_path):
    con, st, eng = _db()
    # C2 (study 1.2, 2022-03-01) is 17 at the study: not requested
    con.execute("UPDATE person SET year_of_birth = 2004, month_of_birth = 6, day_of_birth = 1 "
                "WHERE person_source_value = 'C2'")
    summary = bdr.run(eng, st, ["mri"], str(tmp_path))
    with open(tmp_path / "deid_request.csv", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["STUDY_INSTANCE_UID", "PATIENT_DK", "MODALITY", "STMO_NAME_IN_ADL"]
    assert rows[1:] == [["1.1", "501", "MR", "STMO1"]]                 # merged DIM_PATIENT row 502 not used
    assert (tmp_path / "deid_request_mri.csv").read_text() == (tmp_path / "deid_request.csv").read_text()
    m = json.loads((tmp_path / "deid_request_summary.json").read_text())["mri"]
    assert m["found"] == 3 and m["studies_with_lge"] == 2 and m["studies_under_min_age"] == 1 and m["rows"] == 1
    assert m["patient_dk_from"] == "DIM_PATIENT" and summary["total_rows"] == 1


def test_ambiguous_patient_key_stops_the_request(tmp_path):
    con, st, eng = _db()
    con.execute("UPDATE DIM_PATIENT SET PATIENT_MERGED_FLAG = 'N' WHERE PATIENT_DK = 502")   # C1 now has two
    with pytest.raises(ValueError, match="more than one unmerged"):
        bdr.run(eng, st, ["mri"], str(tmp_path))
    assert not (tmp_path / "deid_request.csv").exists()


def test_missing_patient_key_is_dropped_and_counted(tmp_path):
    con, st, eng = _db()
    con.execute("DELETE FROM DIM_PATIENT WHERE PATIENT_CLINIC_NUMBER = 'C2'")
    summary = bdr.run(eng, st, ["mri"], str(tmp_path))
    assert summary["mri"]["studies_without_patient_dk"] == 1 and summary["mri"]["rows"] == 1


def test_unknown_age_is_not_requested():
    con, st, eng = _db()
    con.execute("DELETE FROM person WHERE person_source_value = 'C1'")
    s = pd.DataFrame({"study_uid": ["1.1", "1.2"], "person_id": [None, 12],
                      "study_datetime": pd.to_datetime(["2022-03-01", "2022-03-01"])})
    kept, c = deid.adults_only(eng, st["modality"]["mri"]["mcp"], s, 18)
    assert kept.study_uid.tolist() == ["1.2"] and c["studies_age_unknown"] == 1


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
