"""Synthetic, PHI-free cohort-specific source availability tests."""
import pandas as pd
import pytest

from cardioai_mcp.config import load_study
from echo_wm_training.cohort_type_availability import (
    cohort_type_rows_sql,
    summarize_cohort_types,
)


def _cohort():
    return pd.DataFrame([
        {"person_id": i, "index_date": "2021-04-03"} for i in range(1, 6)
    ])


def _candidates():
    return pd.DataFrame([
        {"person_id": 1, "index_date": "2021-04-03",
         "candidate_status": "single_exam_candidate", "lvedv": 120.},
        {"person_id": 2, "index_date": "2021-04-03",
         "candidate_status": "unresolved_method_disagreement", "lvedv": 160.},
        {"person_id": 3, "index_date": "2021-04-03",
         "candidate_status": "single_exam_candidate", "lvedv": None},
        {"person_id": 4, "index_date": "2021-04-03",
         "candidate_status": "ambiguous_multiple_exams", "lvedv": None},
    ])


def _rows():
    records = [
        (1, "A", "Adult TTE", "ACC", True),
        (1, "B", "Adult TTE", "ORD", True),  # ACC/ORD count once as patient-day
        (2, "C", "Adult TTE", "ACC", True),
        (3, "D", "Adult TTE", "ACC", False),
        (4, "E", "Adult TTE", "ACC", True),
        (4, "F", "Adult TTE contrast", "ACC", False),  # multi-type patient
    ]
    return pd.DataFrame([
        dict(person_id=pid, index_date="2021-04-03",
             echo_test_fpk=fpk, test_type=typ, record_source=source,
             has_mapped_lvedv=lv)
        for pid, fpk, typ, source, lv in records
    ])


def test_sql_is_read_only_native_bigquery_and_index_day_only():
    s = cohort_type_rows_sql(load_study()["modality"], start="2021-04-01",
                             end="2021-04-30")
    assert "DATE(t.ECHO_PERFORMED_DTM) = a.index_date" in s
    assert "SAFE_CAST" in s and "FLOAT64" in s
    assert "MC CV LV END DIASTOLIC VOLUME" in s
    assert "echo_analytic" in s
    assert "GROUP BY 1" in s
    assert "RLIKE" not in s
    assert "CREATE TABLE" not in s.upper()
    assert "INSERT INTO" not in s.upper()


def test_counts_are_patient_days_not_duplicate_acc_ord_records():
    detail, overall, types = summarize_cohort_types(
        _cohort(), _rows(), _candidates(), small_cell=1
    )
    assert len(detail) == 5
    counts = dict(zip(overall.coverage_class, overall.patient_days))
    assert counts == {
        "provisional_candidate_lvedv": 2,
        "tte_no_mapped_lvedv": 1,
        "source_lvedv_not_candidate": 1,
        "no_conventional_tte": 1,
    }
    adult = types[types.test_type == "Adult TTE"].iloc[0]
    assert int(adult.patient_days) == 4
    assert int(adult.with_mapped_lvedv) == 3
    assert int(adult.source_test_records) == 5
    assert int(adult.with_provisional_candidate) == 2
    # Type categories intentionally overlap across a patient-day.
    assert int(types.patient_days.sum()) == 5


def test_suppress_small_type_breakdown():
    _, _, types = summarize_cohort_types(
        _cohort(), _rows(), _candidates(), small_cell=2
    )
    assert types.empty


def test_duplicate_cohort_key_rejected():
    duplicate = pd.concat([_cohort(), _cohort().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_cohort_types(duplicate, _rows(), _candidates())


def test_candidate_without_same_day_test_rejected():
    c = _candidates()
    c = pd.concat([c, pd.DataFrame([{
        "person_id": 5, "index_date": "2021-04-03",
        "candidate_status": "single_exam_candidate", "lvedv": 110.
    }])], ignore_index=True)
    with pytest.raises(ValueError, match="no matching same-day"):
        summarize_cohort_types(_cohort(), _rows(), c, small_cell=1)
