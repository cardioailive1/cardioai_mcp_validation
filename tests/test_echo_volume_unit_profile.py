"""Synthetic-only result-name/unit profiling tests; no Mayo patient records."""
import pandas as pd
import pytest

from echo_wm_training.volume_unit_profile import classify_numeric_unit_profile


def _fixtures():
    date = "2020-01-02"
    preview = pd.DataFrame([
        dict(person_id=i, index_date=date,
             exam_status=("multiple_exams" if i == 6
                          else "single_identifier_group"))
        for i in range(1, 7)
    ])
    tests = pd.DataFrame([
        dict(person_id=i, index_date=date, echo_test_fpk=f"test-{i}")
        for i in range(1, 7)
    ])
    rows = [
        (1, "LV_vol_BP_ed_2D_by_MOD", "120", "mL"),
        (2, "LV_vol_BP_ed_2D_by_MOD", "121", None),
        (3, "LV_vol_BP_ed_2D_by_MOD", "122", "cc"),
        (4, "LV_vol_BP_ed_2D_by_MOD", "123", "mL"),
        (4, "LV_vol_BP_ed_2D_by_MOD", "123", ""),
        (5, "LV_vol_4C_ed_2D_by_MOD", "110", "mL"),
        (6, "LV_vol_BP_ed_2D_by_MOD", "125", "mL"),
        # Ignore out-of-range values rather than treating missing unit on
        # an invalid numeric reading as a barrier to a valid result.
        (1, "LV_vol_BP_ed_2D_by_MOD", "99999", None),
    ]
    values = pd.DataFrame([
        dict(person_id=i, index_date=date,
             echo_test_fpk=f"test-{i}", result_name=method,
             raw_value=value, reported_unit=unit)
        for i, method, value, unit in rows
    ])
    return preview, tests, values


def test_each_name_counts_unique_selected_patient_days():
    private, tests, values = _fixtures()
    out = classify_numeric_unit_profile(
        private, tests, values, small_cell=1
    )
    first = out.set_index("result_name").loc["LV_vol_BP_ed_2D_by_MOD"]
    assert first.patient_days_with_numeric_in_range == "4"
    assert first.explicit_ml_only == "1"
    assert first.missing_only == "1"
    assert first.non_ml_only == "1"
    assert first.mixed_unit_metadata == "1"
    second = out.set_index("result_name").loc["LV_vol_4C_ed_2D_by_MOD"]
    assert second.patient_days_with_numeric_in_range == "1"
    assert second.explicit_ml_only == "1"
    assert "person_id" not in out
    assert "raw_value" not in out
    assert "reported_unit" not in out
    assert len(out) == 10  # five recognized names per metric


def test_small_unit_category_redacts_whole_named_result():
    private, tests, values = _fixtures()
    out = classify_numeric_unit_profile(
        private, tests, values, small_cell=11
    )
    observed = out.set_index("result_name").loc["LV_vol_BP_ed_2D_by_MOD"]
    assert all(value == "suppressed" for value in [
        observed.patient_days_with_numeric_in_range,
        observed.explicit_ml_only, observed.missing_only,
        observed.non_ml_only, observed.mixed_unit_metadata,
    ])
    absent = out.set_index("result_name").loc["LV_vol_ed_3D"]
    assert absent.patient_days_with_numeric_in_range == "0"


def test_reject_missing_fields_and_duplicate_patient_days():
    private, tests, values = _fixtures()
    with pytest.raises(ValueError, match="Duplicate preview"):
        classify_numeric_unit_profile(
            pd.concat([private, private.iloc[[0]]]), tests, values
        )
    with pytest.raises(ValueError, match="Missing source result fields"):
        classify_numeric_unit_profile(
            private, tests, values.drop(columns="reported_unit")
        )
    with pytest.raises(ValueError, match="small_cell"):
        classify_numeric_unit_profile(private, tests, values, small_cell=0)
