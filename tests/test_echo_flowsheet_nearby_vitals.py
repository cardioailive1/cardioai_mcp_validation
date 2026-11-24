"""Synthetic contract tests: strict-label and date-bounded Mayo vital availability."""
import pandas as pd
import pytest

from cardioai_mcp.config import load_study
from echo_wm_training.flowsheet_nearby_vitals import (
    classify_direct_vital_label,
    nearby_direct_vital_sql,
    protect_nearby_direct_vital_counts,
)


@pytest.mark.parametrize("label,expected", [
    ("Actual Height", "height"),
    ("Admit Height (cm)", "height"),
    ("(EXT)Height (cm)", "height"),
    ("Stature", "height"),
    ("Actual Weight", "weight"),
    ("Admit Weight (kg)", "weight"),
    ("Clinical Weight (kg)", "weight"),
    ("Current Weight (in lbs)", "weight"),
    ("Daily Weight (kg)", "weight"),
    ("Weight", "weight"),
    ("Dosing Weight", None),
    ("Calculation Weight for Calories", None),
    ("Dry Diaper Weight (g)", None),
    ("During the past week, what did you do to lose weight?", None),
    ("Adjusted Ideal Body Weight", None),
    ("Double Leg Calf Raises Reps/Sets/Weight", None),
    ("Approximate Height (feet)", None),
    ("Admission Height/Length Z-Score", None),
    ("Current Admission Height/Weight/BMI", None),
    ("Birth Weight", None),
    ("CLCurrentRxSegmentHeightLeftEye", None),
    ("Desired Weight Goal", None),
    ("Body Mass Index", None),
    (None, None),
])
def test_whole_description_direct_measurement_filter(label, expected):
    assert classify_direct_vital_label(label) == expected


def test_native_bigquery_sql_is_read_only_strict_and_date_bounded():
    sql = nearby_direct_vital_sql(
        load_study()["modality"],
        start="2012-01-01", end="2023-12-31",
    )
    assert "DATE('2019-01-01')" in sql
    assert "DATE('2023-12-31')" in sql
    assert "{{work}}.echo_analytic" in sql
    assert "DIM_FLOWSHEET_ROW_NAME" in sql
    assert "FACT_FLOWSHEETS" in sql
    assert "CAST(f.PATIENT_DK AS STRING) = k.patient_dk" in sql
    assert "distinct_patient_dk_count = 1" in sql
    assert "a.missing_height AND r.kind = 'height'" in sql
    assert "a.missing_weight AND r.kind = 'weight'" in sql
    assert "BETWEEN -1095 AND 30" in sql
    assert "BETWEEN -90 AND 30" in sql
    assert "SAFE_CAST(TRIM(CAST(f.FLOWSHEET_RESULT_TXT AS STRING)) AS FLOAT64) > 0" in sql
    assert "r'^(?:actual height" in sql
    assert "r'^(?:actual weight" in sql
    # BigQuery HAVING must use the qualified source column; the
    # SELECT aggregate alias MIN(dr.kind) AS kind must not be reused
    # inside COUNT(DISTINCT ...) or COUNTIF(...).
    assert "MIN(dr.kind) AS kind" in sql
    assert "FROM direct_rows dr" in sql
    assert "HAVING COUNT(DISTINCT dr.kind) = 1" in sql
    assert "COUNTIF(dr.kind IS NULL) = 0" in sql
    assert "COUNT(DISTINCT kind)" not in sql
    assert "COUNTIF(kind IS NULL)" not in sql
    assert "WHERE kind IS NOT NULL" not in sql
    assert "SELECT COUNT(*) AS cohort_days" in sql
    assert "SELECT * FROM" not in sql
    assert "CREATE " not in sql
    assert "UPDATE " not in sql
    assert "INSERT " not in sql
    assert "ECHO_TEST_FPK" not in sql
    assert "ROW_SOURCE_ID" not in sql


def _row(**change):
    r = dict(
        cohort_days=600,
        baseline_height_missing=41,
        baseline_weight_missing=38,
        height_missing_with_candidate=15,
        weight_missing_with_candidate=13,
    )
    r.update(change)
    return pd.DataFrame([r])


def test_readable_safe_synthetic_counts_have_no_identifiers():
    out = protect_nearby_direct_vital_counts(_row())
    assert out.set_index("measure").patient_days.loc[
        "height_missing_with_candidate"
    ] == "15"
    assert out.set_index("measure").patient_days.loc[
        "weight_missing_with_candidate"
    ] == "13"
    assert set(out) == {"measure", "patient_days"}


def test_small_candidate_or_small_complement_suppresses_all():
    for raw in [
        _row(height_missing_with_candidate=1),
        _row(weight_missing_with_candidate=10),
        _row(height_missing_with_candidate=35),  # 41-35 = 6
        _row(weight_missing_with_candidate=30),  # 38-30 = 8
        _row(baseline_height_missing=9,
             height_missing_with_candidate=0),
    ]:
        out = protect_nearby_direct_vital_counts(raw)
        assert (out.patient_days == "suppressed").all()


def test_zero_recovery_and_impossible_aggregates():
    safe = protect_nearby_direct_vital_counts(
        _row(height_missing_with_candidate=0,
             weight_missing_with_candidate=0)
    )
    assert safe.patient_days.iloc[-1] == "0"
    with pytest.raises(ValueError, match="Inconsistent"):
        protect_nearby_direct_vital_counts(
            _row(height_missing_with_candidate=42)
        )
    with pytest.raises(ValueError, match="Expected one"):
        protect_nearby_direct_vital_counts(
            _row().drop(columns="baseline_height_missing")
        )
    with pytest.raises(ValueError, match="small_cell"):
        protect_nearby_direct_vital_counts(_row(), small_cell=0)
    with pytest.raises(ValueError, match="No Epic-era"):
        nearby_direct_vital_sql(
            load_study()["modality"],
            start="2012-01-01", end="2018-12-31",
        )
