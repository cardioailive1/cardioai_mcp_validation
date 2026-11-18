"""Synthetic-only tests for preview barrier audit disclosure safeguards."""
import pandas as pd
import pytest

from echo_wm_training.cohort_preview_diagnostics import (
    summarize_preview_barriers,
    within_mcp_reason_counts,
)


def _sample():
    # Cohort of 150 synthetic cases, with 70 single-ID examinations.
    data = []
    for i in range(150):
        if i < 70:
            exam = "single_identifier_group"
            if i < 20:
                edv = esv = "unique_consistent_ml"
            elif i < 40:
                edv = esv = "no_named_measurement"
            else:
                edv = "unknown_or_non_absolute_unit"
                esv = "disagreeing_measurements_review"
        elif i < 110:
            exam = "provisional_pair_requires_review"
            edv = esv = "exam_not_selected"
        elif i < 145:
            exam = "multiple_exams"
            edv = esv = "exam_not_selected"
        else:
            exam = "no_conventional_tte"
            edv = esv = "exam_not_selected"
        data.append({
            "person_id": i, "exam_status": exam,
            "lv_end_diastolic_volume_status": edv,
            "lv_end_systolic_volume_status": esv,
        })
    return pd.DataFrame(data)


def test_synthetic_barrier_counts_are_pure_aggregates():
    sample = _sample()
    exam, volumes = summarize_preview_barriers(sample, small_cell=11)
    assert exam.patient_days.tolist() == ["70", "40", "40"]
    assert "person_id" not in exam and "person_id" not in volumes
    edv = volumes.query("metric == 'lv_end_diastolic_volume'")
    assert edv.selected_group_patient_days.tolist() == ["20", "20", "30"]
    assert len(volumes) == 6
    assert int(edv.selected_group_patient_days.astype(int).sum()) == 70
    internal = within_mcp_reason_counts(sample)
    assert "patient_days" in internal
    assert "metric" in internal
    assert set(internal.metric) == {
        "lv_end_diastolic_volume", "lv_end_systolic_volume"
    }


def test_suppress_all_exam_categories_when_one_category_is_small():
    s = _sample()
    s["exam_status"] = "single_identifier_group"
    s.loc[:4, "exam_status"] = "provisional_pair_requires_review"
    exam, _ = summarize_preview_barriers(s, small_cell=11)
    assert (exam.patient_days == "suppressed").all()


def test_suppress_all_reason_categories_for_small_positive_remainder():
    s = _sample()
    # Creates one selected group reason count <11, which must not be
    # reconstructable by subtracting two printed counts from 70.
    s.loc[s.person_id.between(19, 20),
          "lv_end_diastolic_volume_status"] = "unknown_or_non_absolute_unit"
    s.loc[s.person_id.between(21, 39),
          "lv_end_diastolic_volume_status"] = "unique_consistent_ml"
    # leaves no_named_measurement = 0 and unique = 39 -> no suppression;
    # force no-named = 5 instead.
    s.loc[s.person_id.between(35, 39),
          "lv_end_diastolic_volume_status"] = "no_named_measurement"
    _, volume = summarize_preview_barriers(s, small_cell=11)
    edv = volume.query("metric == 'lv_end_diastolic_volume'")
    assert (edv.selected_group_patient_days == "suppressed").all()


def test_rejects_unknown_status_and_missing_columns():
    s = _sample()
    s.loc[0, "exam_status"] = "unknown"
    with pytest.raises(ValueError, match="Unknown examination"):
        summarize_preview_barriers(s)
    with pytest.raises(ValueError, match="Missing preview fields"):
        summarize_preview_barriers(pd.DataFrame({"exam_status": ["x"]}))
