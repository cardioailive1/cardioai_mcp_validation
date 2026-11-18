"""Synthetic tests for read-only cohort-scoped echo discovery."""
import pandas as pd
import pytest

from cardioai_mcp.config import load_study
from echo_wm_training.cohort_source_discovery import (
    cohort_result_types_sql,
    cohort_echo_note_coverage_sql,
    echo_schema_sql,
    echo_note_titles_sql,
    display_safe_coverage,
)


def _cfg():
    return load_study()["modality"]


def test_result_type_discovery_stays_scoped_to_saved_conventional_tte():
    sql = cohort_result_types_sql(_cfg(), start="2020-01-01",
                                  end="2020-02-01", small_cell=11)
    assert "{{work}}.echo_analytic" in sql
    assert "FACT_ECHO_TEST" in sql
    assert "DIM_ECHO_TEST" in sql
    assert "FACT_ECHO_RESULTS" in sql
    assert "DIM_ECHO_TEST_RESULT_TYPE" in sql
    assert "DATE(t.ECHO_PERFORMED_DTM) = a.index_date" in sql
    assert "NOT REGEXP_CONTAINS" in sql
    assert "stress|strain|outside|review|fetal" in sql
    assert "COUNT(DISTINCT TO_JSON_STRING(STRUCT(t.person_id, t.index_date)))" in sql
    assert ">= 11" in sql
    assert "LIMIT 80" in sql
    assert "RLIKE" not in sql
    for command in ("CREATE TABLE", "UPDATE ", "INSERT ", "DELETE ", "MERGE "):
        assert command not in sql.upper()


def test_note_coverage_uses_actual_echo_note_and_cohort_report():
    sql = cohort_echo_note_coverage_sql(
        _cfg(), start="2020-01-01", end="2020-02-01"
    )
    assert "ECHO_NOTE" in sql
    assert "current_report_present" in sql
    assert "LOGICAL_OR(has_echo_note)" in sql
    assert "LEFT JOIN per_day" in sql
    assert "COUNTIF" in sql
    # No detail records from the final select.
    assert "SELECT COUNT(*) AS saved_cohort_days" in sql


def test_schema_and_notes_are_metadata_and_aggregate():
    schema = echo_schema_sql("`p`.dataset")
    titles = echo_note_titles_sql("`p`.dataset", small_cell=11)
    assert "INFORMATION_SCHEMA.COLUMNS" in schema
    assert "FACT_ECHO_TEST" in schema
    assert "GROUP BY 1" in titles
    assert "HAVING COUNT(*) >= 11" in titles
    assert "note_text" not in titles


def test_coverage_suppression_and_non_disclosive_rates():
    source = pd.DataFrame([dict(
        saved_cohort_days=1500,
        days_with_conventional_tte=1477,
        conventional_test_records=2900,
        days_with_current_extracted_report=1400,
        days_with_source_echo_note=1499,  # complementary missing = 1
        days_with_wall_motion_cue=5,       # small positive
        days_with_quality_cue=0,
        days_with_indication_cue=1500,
    )])
    out = display_safe_coverage(source, small_cell=11).set_index("measure")
    assert out.loc["saved_cohort_days", "patient_days"] == "1500"
    assert out.loc["days_with_conventional_tte", "patient_days"] == "1477"
    assert out.loc["days_with_source_echo_note", "patient_days"] == "suppressed"
    assert pd.isna(out.loc["days_with_source_echo_note", "pct"])
    assert out.loc["days_with_wall_motion_cue", "patient_days"] == "suppressed"
    assert out.loc["days_with_quality_cue", "patient_days"] == "0"
    assert out.loc["days_with_indication_cue", "patient_days"] == "1500"
    assert "conventional_test_records" not in out.index


def test_coverage_rejects_invalid_bounds():
    source = pd.DataFrame([dict(
        saved_cohort_days=20, days_with_conventional_tte=21,
        days_with_current_extracted_report=3, days_with_source_echo_note=4,
        days_with_wall_motion_cue=2, days_with_quality_cue=0,
        days_with_indication_cue=10,
    )])
    with pytest.raises(ValueError, match="Invalid count"):
        display_safe_coverage(source)


def test_invalid_date_window_or_cell_rejected():
    with pytest.raises(ValueError, match="index_start"):
        cohort_result_types_sql(_cfg(), start="2021-01-01",
                                end="2020-01-01")
    with pytest.raises(ValueError, match="small_cell"):
        cohort_result_types_sql(_cfg(), start="2020-01-01",
                                end="2020-02-01", small_cell=0)
