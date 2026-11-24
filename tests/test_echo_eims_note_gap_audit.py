"""Synthetic/no-clinical-data tests for source-FPK EIMS report availability."""
import pandas as pd
import pytest

from cardioai_mcp.config import load_study
from echo_wm_training.eims_note_gap_audit import (
    eims_note_gap_sql, protect_eims_note_gap_counts,
)


def _counts(**changes):
    values = dict(
        cohort_days=1500, sole_eims_test_days=1404,
        sole_eims_with_note=700,
        baseline_missing_wall_motion=1308,
        baseline_missing_image_quality=1218,
        baseline_missing_clinical_indication=815,
        missing_wall_motion_with_note=650,
        missing_image_quality_with_note=600,
        missing_clinical_indication_with_note=400,
        missing_wall_motion_with_cue=400,
        missing_image_quality_with_cue=70,
        missing_clinical_indication_with_cue=0,
    )
    values.update(changes)
    return pd.DataFrame([values])


def test_native_sql_selects_only_eims_exam_keyed_note_coverage():
    sql = eims_note_gap_sql(
        load_study()["modality"],
        start="2012-01-01", end="2023-12-31",
    )
    assert "WITH cohort AS" in sql
    assert "source_tests AS" in sql
    assert "study_day AS" in sql
    assert "source_notes AS" in sql
    assert "FROM source_tests s" in sql
    assert "CAST(t.ECHO_TEST_FPK AS STRING) = s.echo_test_fpk" in sql
    assert "t.ROW_SOURCE_ID" in sql
    assert "'12315'" in sql
    assert "COUNT(DISTINCT IF(test_row_source_id = '12315'," in sql
    assert "missing_wall_motion_with_note" in sql
    assert "missing_image_quality_with_cue" in sql
    assert "missing_clinical_indication_with_note" in sql
    assert "e.sole_eims_test AND e.has_note" in sql
    assert "RLIKE" not in sql
    assert "SELECT COUNT(*) AS cohort_days" in sql
    assert "FACT_ECHO_TEST" in sql
    assert "FACT_ECHO_RESULTS" not in sql
    assert "INSERT " not in sql
    assert "UPDATE " not in sql
    assert "CREATE TABLE" not in sql
    assert "SELECT * FROM" not in sql


def test_privacy_gate_safely_displays_synthetic_aggregates():
    result = protect_eims_note_gap_counts(_counts())
    by_field = result.set_index("measure").patient_days
    assert by_field["sole_eims_test_days"] == "1404"
    assert by_field["missing_wall_motion_with_note"] == "650"
    assert by_field["missing_image_quality_with_cue"] == "70"
    assert by_field["missing_clinical_indication_with_cue"] == "0"
    assert "person_id" not in result.columns
    assert "echo_note" not in result.columns


@pytest.mark.parametrize("change", [
    dict(missing_wall_motion_with_cue=5),
    dict(missing_wall_motion_with_note=695),  # 700 - 695 = 5
    dict(sole_eims_with_note=1400),  # single EIMS with no note=4
    dict(baseline_missing_clinical_indication=1499,
         missing_clinical_indication_with_note=700,
         missing_clinical_indication_with_cue=0),
])
def test_primary_and_derived_small_cells_suppress_everything(change):
    out = protect_eims_note_gap_counts(_counts(**change))
    assert (out.patient_days == "suppressed").all()



def test_malformed_overlap_counts_fail_closed():
    with pytest.raises(ValueError, match="Impossible"):
        protect_eims_note_gap_counts(
            _counts(missing_wall_motion_with_cue=651)
        )
    with pytest.raises(ValueError, match="Expected one"):
        protect_eims_note_gap_counts(
            _counts().drop(columns=["sole_eims_with_note"])
        )
    with pytest.raises(ValueError, match="small_cell"):
        protect_eims_note_gap_counts(_counts(), small_cell=0)
