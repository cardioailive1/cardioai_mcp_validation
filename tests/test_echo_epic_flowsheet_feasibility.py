"""Synthetic-only tests for read-only Epic encounter flowsheet audit."""
import pandas as pd
import pytest

from cardioai_mcp.config import load_study
from echo_wm_training.epic_flowsheet_feasibility import (
    epic_flowsheet_candidate_sql,
    suppress_epic_flowsheet_counts,
)


def _row():
    return pd.DataFrame([{
        "cohort_days": 100,
        "single_epic_encounter_days": 80,
        "height_numeric_days": 55,
        "weight_numeric_days": 60,
        "baseline_height_missing_days": 45,
        "baseline_weight_missing_days": 50,
        "height_missing_with_numeric_candidate": 25,
        "weight_missing_with_numeric_candidate": 30,
    }])


def test_native_sql_is_read_only_and_joins_exact_epic_patient_encounter():
    sql = epic_flowsheet_candidate_sql(
        load_study()["modality"], start="2012-01-01", end="2023-12-31"
    )
    assert "DATE('2019-01-01')" in sql
    assert "DATE('2023-12-31')" in sql
    assert "CAST(t.ROW_SOURCE_ID AS STRING)" in sql
    assert "= '47918'" in sql
    assert "FACT_ECHO_TEST" in sql
    assert "FACT_FLOWSHEETS" in sql
    assert "DIM_FLOWSHEET_ROW_NAME" in sql
    assert "CAST(f.PATIENT_DK AS STRING) = e.patient_dk" in sql
    assert "CAST(f.ENCOUNTER_NBR AS STRING) = e.encounter_number" in sql
    assert "DATE(f.FLOWSHEET_ASSESSMENT_DTM) = DATE(e.index_date)" in sql
    assert "COUNT(DISTINCT echo_test_fpk)" in sql
    assert "e.source_test_count = 1" in sql
    assert "SAFE_CAST(TRIM(CAST(f.FLOWSHEET_RESULT_TXT AS STRING)) AS FLOAT64)" in sql
    assert "height_missing_with_numeric_candidate" in sql
    assert "weight_missing_with_numeric_candidate" in sql
    assert "SELECT COUNT(*) AS cohort_days" in sql
    assert "SELECT * FROM" not in sql
    assert "UPDATE " not in sql
    assert "INSERT " not in sql
    assert "CREATE TABLE" not in sql


def test_coarser_numeric_candidate_counts_are_displayable_when_complements_safe():
    result = suppress_epic_flowsheet_counts(_row(), small_cell=11)
    assert result.shape == (8, 2)
    assert result.set_index("measure").patient_days.loc[
        "single_epic_encounter_days"
    ] == "80"
    assert result.set_index("measure").patient_days.loc[
        "height_missing_with_numeric_candidate"
    ] == "25"
    assert result.set_index("measure").patient_days.loc[
        "weight_missing_with_numeric_candidate"
    ] == "30"
    assert not result.patient_days.eq("suppressed").any()
    assert "person_id" not in result
    assert "ENCOUNTER_NBR" not in result


def test_rare_primary_or_complementary_cell_suppresses_all():
    result = suppress_epic_flowsheet_counts(
        _row().assign(height_missing_with_numeric_candidate=5),
        small_cell=11,
    )
    assert (result.patient_days == "suppressed").all()

    # A rare subset may be inferred from known denominators even if
    # its visible parent is large.
    result = suppress_epic_flowsheet_counts(
        _row().assign(single_epic_encounter_days=95),
        small_cell=11,
    )
    assert (result.patient_days == "suppressed").all()


def test_invalid_aggregate_and_missing_fields_fail_closed():
    raw = _row()
    with pytest.raises(ValueError, match="one cohort-scoped"):
        suppress_epic_flowsheet_counts(
            raw.drop(columns="height_numeric_days")
        )
    with pytest.raises(ValueError, match="one cohort-scoped"):
        suppress_epic_flowsheet_counts(
            pd.concat([raw, raw], ignore_index=True)
        )
    with pytest.raises(ValueError, match="Impossible"):
        suppress_epic_flowsheet_counts(
            raw.assign(height_missing_with_numeric_candidate=60),
        )
    with pytest.raises(ValueError, match="small_cell"):
        suppress_epic_flowsheet_counts(raw, small_cell=0)
    with pytest.raises(ValueError, match="No 2019"):
        epic_flowsheet_candidate_sql(
            load_study()["modality"],
            start="2012-01-01", end="2018-12-31",
        )
