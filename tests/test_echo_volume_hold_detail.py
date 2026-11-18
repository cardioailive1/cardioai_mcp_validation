"""Synthetic unit/method disagreement isolation; no clinical data."""
import pandas as pd
import pytest

from echo_wm_training.cohort_extraction_preview import preview_echo_extraction
from echo_wm_training.volume_hold_detail import (
    _classify_held_metric, summarize_volume_hold_causes,
)


def _cases():
    day = "2020-01-02"
    tests = pd.DataFrame([
        dict(person_id=i, index_date=day, echo_test_fpk=f"t{i}",
             exam_datetime=day + " 10:00:00", accession_number=f"acc{i}",
             order_proc_id=None, test_description="Adult TTE")
        for i in range(1, 7)
    ])
    samples = [
        (1, "LV_vol_BP_ed_2D_by_MOD", "120", "mL"),
        (1, "LV_vol_4C_ed_2D_by_MOD", "120", ""),
        (2, "LV_vol_BP_ed_2D_by_MOD", "120", "mL"),
        (2, "LV_vol_4C_ed_2D_by_MOD", "125", "mL"),
        (3, "LV_vol_BP_ed_2D_by_MOD", "120", "mL"),
        (3, "LV_vol_BP_ed_2D_by_MOD", "122", "mL"),
        (4, "LV_vol_BP_ed_2D_by_MOD", "120", ""),
        (4, "LV_vol_4C_ed_2D_by_MOD", "125", "mL"),
        (5, "LV_vol_BP_ed_2D_by_MOD", "120", "mL"),
    ]
    volumes = pd.DataFrame([
        dict(person_id=pid, index_date=day, echo_test_fpk=f"t{pid}",
             result_name=name, raw_value=value, reported_unit=unit)
        for pid, name, value, unit in samples
    ])
    cohort = tests[["person_id", "index_date"]].copy()
    notes = pd.DataFrame(columns=[
        "person_id", "index_date", "echo_test_fpk", "echo_note"
    ])
    private = preview_echo_extraction(cohort, tests, volumes, notes)
    return private, tests, volumes


def test_exclusive_hold_breakdown_with_no_selection_change():
    private, tests, volumes = _cases()
    summary = summarize_volume_hold_causes(
        private, tests, volumes, small_cell=1
    )
    edv = summary[summary.metric.eq("lv_end_diastolic_volume")]
    got = edv.set_index("hold_reason")["patient_days"].to_dict()
    assert got == {
        "unit_metadata_only": "1",
        "different_methods_only": "1",
        "same_method_conflict": "1",
        "unit_and_value_disagreement": "1",
    }
    # This diagnostic does not fill or modify the preview values.
    assert private.loc[private.person_id.lt(5),
                       "lv_end_diastolic_volume"].isna().all()
    assert private.loc[private.person_id.eq(5),
                       "lv_end_diastolic_volume"].iloc[0] == 120
    assert len(summary) == 8


def test_held_breakdown_suppresses_small_categories():
    private, tests, volumes = _cases()
    summary = summarize_volume_hold_causes(
        private, tests, volumes, small_cell=11
    )
    edv = summary[summary.metric.eq("lv_end_diastolic_volume")]
    assert (edv.patient_days == "suppressed").all()
    assert "person_id" not in summary


def test_detects_mismatched_preview_status():
    private, tests, volumes = _cases()
    private.loc[private.person_id.eq(1),
                "lv_end_diastolic_volume_status"] = "disagreeing_measurements_review"
    with pytest.raises(ValueError, match="Preview and hold reason disagree"):
        summarize_volume_hold_causes(private, tests, volumes, small_cell=1)


def test_no_conflict_is_not_a_held_category():
    row = pd.DataFrame([{
        "result_name": "LV_vol_BP_ed_2D_by_MOD",
        "raw_value": "140", "reported_unit": "mL",
    }])
    with pytest.raises(ValueError, match="neither unit nor numeric conflict"):
        _classify_held_metric(row, "lv_end_diastolic_volume")
