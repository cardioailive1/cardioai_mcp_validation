"""Synthetic regression tests for accession-aware Mayo Echo reference candidates."""
import pandas as pd
import pytest

from echo_wm_training.exam_results import (
    build_exam_reference_candidates, exam_reference_rows_sql,
    select_exam_references,
)
from cardioai_mcp.config import load_study


def _row(patient, accession, fpk, metric, value, name, date="2019-04-02"):
    return {
        "person_id": patient,
        "index_date": "2019-04-02",
        "accession_number": accession,
        "echo_test_fpk": fpk,
        "exam_date": date,
        "result_name": name,
        "result_label": name,
        "result_value": value,
        "metric": metric,
    }


def test_accession_groups_multiple_test_records_without_merging_exams():
    results = load_study()["modality"]["echo"]["mcp"]["results"]
    rows = pd.DataFrame([
        _row(1, "ACC1", "test1", "lvef", "55", "LVEF visual"),
        _row(1, "ACC1", "test2", "lvef", "50", "LVEF biplane"),
        _row(1, "ACC1", "test2", "lvedv", "120", "LV_vol_BP_ed_2D_by_MOD"),
        _row(1, "ACC1", "test1", "lvedv", "130", "LV_vol_4C_ed_2D_by_MOD"),
        _row(2, "ACC2", "test3", "lvef", "45", "LVEF visual"),
        _row(2, "ACC3", "test4", "lvef", "65", "LVEF visual"),
        _row(3, None, "test5", "lvedv", "160", "LV_vol_4C_ed_2D_by_MOD"),
    ])
    selected = select_exam_references(rows, results)
    assert len(selected) == 4
    first = selected[selected.person_id == 1].iloc[0]
    assert first.status == "unique_accession"
    assert first.lvef == 50
    assert first.lvedv == 120
    assert first.lvedv_method == "LV_vol_BP_ed_2D_by_MOD"
    assert set(selected[selected.person_id == 2].status) == {"multiple_or_missing_accessions"}
    assert selected[selected.person_id == 3].status.iloc[0] == "missing_accession"


def test_mixed_missing_accession_cannot_be_unique():
    results = load_study()["modality"]["echo"]["mcp"]["results"]
    rows = pd.DataFrame([
        _row(1, "ACC1", "test1", "lvef", "55", "LVEF visual"),
        _row(1, None, "test2", "lvef", "65", "LVEF visual"),
    ])
    selected = select_exam_references(rows, results)
    assert set(selected.status) == {"missing_accession", "multiple_or_missing_accessions"}


def test_multi_date_accession_is_flagged():
    results = load_study()["modality"]["echo"]["mcp"]["results"]
    rows = pd.DataFrame([
        _row(1, "ACC1", "test1", "lvedv", "110", "LV_vol_BP_ed_2D_by_MOD"),
        _row(1, "ACC1", "test2", "lvedv", "115", "LV_vol_BP_ed_2D_by_MOD", date="2019-04-03"),
    ])
    selected = select_exam_references(rows, results)
    assert selected.status.tolist() == ["multi_date_accession"]


def test_sql_is_cohort_scoped_and_uses_verified_accession():
    cfg = load_study()["modality"]
    sql = exam_reference_rows_sql(cfg, start="2012-01-01", end="2023-12-31")
    assert "{{work}}.echo_analytic" in sql
    assert "ACCESSION_NUMBER" in sql
    assert "ECHO_TEST_FPK" in sql
    assert "DATEDIFF" in sql
    with pytest.raises(ValueError, match="start"):
        exam_reference_rows_sql(cfg, start="2024-01-01", end="2012-01-01")
