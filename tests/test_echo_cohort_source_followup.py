"""Synthetic tests for second-pass read-only echo source discovery."""
import pandas as pd
import pytest

from cardioai_mcp.config import load_study
from echo_wm_training.cohort_source_discovery import echo_note_titles_sql
from echo_wm_training.cohort_source_followup import (
    cohort_numeric_source_sql,
    cohort_la_diameter_unit_sql,
    cohort_omop_note_overlap_sql,
    suppress_aggregate_counts,
    screen_numeric_counts,
)


def _cfg():
    return load_study()["modality"]


def test_numeric_sql_restricted_to_saved_cohort_and_known_results():
    sql = cohort_numeric_source_sql(_cfg(), start="2020-01-01", end="2020-12-31")
    assert "{{work}}.echo_analytic" in sql
    assert "FACT_ECHO_TEST" in sql
    assert "FACT_ECHO_RESULTS" in sql
    assert "ECHO_RESULTS_VALUE_UNIT_OF_MEASURE" in sql
    for name in ("LV_vol_BP_ed_2D_by_MOD", "LV_vol_4C_es_2D_by_MOD",
                 "MC CV LV END SYSTOLIC VOLUME", "LA_dim_s_MM",
                 "LA_dim_es_2D", "Body_Surface_area"):
        assert name in sql
    assert "SAFE_CAST" in sql
    assert "COUNTIF(has_numeric)" in sql
    assert "WHEN metric = 'lvedv'" in sql
    assert "WHEN metric = 'lvesv'" in sql
    assert "RLIKE" not in sql


def test_la_units_are_metadata_with_count_gate():
    sql = cohort_la_diameter_unit_sql(
        _cfg(), start="2020-01-01", end="2020-12-31", small_cell=11
    )
    assert "ECHO_RESULTS_VALUE_UNIT_OF_MEASURE" in sql
    assert "HAVING cohort_patient_days_with_result >= 11" in sql
    assert "LA_dim_es_2D" in sql
    assert "COUNT(DISTINCT" in sql


def test_note_overlap_uses_known_titles_and_date_scope_only():
    sql = cohort_omop_note_overlap_sql(
        _cfg(), start="2020-01-01", end="2020-12-31"
    )
    assert "echo report text" in sql
    assert "echocardiology image exam" in sql
    assert "hosp echocardiogram" in sql
    assert "BETWEEN a.index_date AND DATE_ADD(a.index_date, INTERVAL 1 DAY)" in sql
    assert "days_any_report_text_candidate" in sql
    assert "days_omop_candidate_not_current" in sql
    assert "days_source_or_omop_wall_cue" in sql
    assert "JOIN {{cdm}}.note n" in sql
    assert "LEFT JOIN omop_note_day" in sql
    # No clinical text in final output: only counts.
    assert "SELECT COUNT(*) AS saved_cohort_days" in sql


def test_note_title_pattern_does_not_match_letter():
    sql = echo_note_titles_sql("`p`.ds")
    assert r"\btte\b" in sql
    assert "r'echo|tte|" not in sql


def test_suppress_pairwise_and_complementary_small_cells():
    data = pd.DataFrame([dict(
        saved_cohort_days=1500,
        days_current_report=282,
        days_source_echo_note=592,
        days_omop_echo_candidate_note=580,
        days_any_report_text_candidate=1000,
    )])
    out = suppress_aggregate_counts(data).set_index("metric")
    assert out.loc["days_current_report", "patient_days"] == "282"
    assert out.loc["days_source_echo_note", "patient_days"] == "592"
    # If an overlap could be deduced from difference <11, redact the row.
    data["days_omop_echo_candidate_note"] = 589
    redacted = suppress_aggregate_counts(data)
    assert (redacted.patient_days == "suppressed").all()


def test_numeric_counts_suppress_small_missing_or_internal_difference():
    data = pd.DataFrame([
        dict(metric="lvedv",
             patient_days_with_named_result=970,
             patient_days_with_numeric_result=969,
             patient_days_in_screening_range=969),
        dict(metric="lvesv",
             patient_days_with_named_result=950,
             patient_days_with_numeric_result=920,
             patient_days_in_screening_range=900),
    ])
    out = screen_numeric_counts(data, denominator=1500).set_index("metric")
    assert out.loc["lvedv", "patient_days_with_named_result"] == "suppressed"
    assert out.loc["lvesv", "patient_days_with_named_result"] == "950"
    assert out.loc["lvesv", "patient_days_in_screening_range"] == "900"


def test_numeric_counts_rejects_impossible_count():
    data = pd.DataFrame([dict(
        metric="x",
        patient_days_with_named_result=1501,
        patient_days_with_numeric_result=1000,
        patient_days_in_screening_range=1000,
    )])
    with pytest.raises(ValueError, match="exceeds cohort"):
        screen_numeric_counts(data, denominator=1500)
