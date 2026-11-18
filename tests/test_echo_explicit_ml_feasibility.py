"""Synthetic, no-clinical-data tests for read-only explicit-mL feasibility."""
import pandas as pd
import pytest

from echo_wm_training.explicit_ml_feasibility import (
    _method_feasibility, summarize_explicit_ml_feasibility,
)


def _fixtures():
    date = "2020-02-03"
    preview = pd.DataFrame([
        dict(person_id=i, index_date=date,
             exam_status=("multiple_exams" if i == 9
                          else "single_identifier_group"))
        for i in range(1, 10)
    ])
    tests = pd.DataFrame([
        dict(person_id=i, index_date=date, echo_test_fpk=f"t{i}")
        for i in range(1, 10)
    ])
    src = [
        (1, "LV_vol_BP_ed_2D_by_MOD", "120", "mL"),
        (1, "LV_vol_4C_ed_2D_by_MOD", "127", "mL"),
        (1, "MC CV LV END DIASTOLIC VOLUME", "128", ""),
        (2, "LV_vol_ed_3D", "123", "ml"),
        (2, "LV_vol_4C_ed_2D_by_MOD", "130", "mL"),
        (3, "LV_vol_4C_ed_2D_by_MOD", "120", "mL"),
        (3, "LV_vol_2C_ed_2D_by_MOD", "120", "mL"),
        (4, "LV_vol_4C_ed_2D_by_MOD", "120", "mL"),
        (4, "LV_vol_2C_ed_2D_by_MOD", "125", "mL"),
        (5, "MC CV LV END DIASTOLIC VOLUME", "140", None),
        (6, "LV_vol_BP_ed_2D_by_MOD", "120", "mL"),
        (6, "LV_vol_BP_ed_2D_by_MOD", "123", "mL"),
        (7, "LV_vol_BP_ed_2D_by_MOD", "110", "cc"),
        (7, "LV_vol_4C_ed_2D_by_MOD", "130", "mL"),
        (9, "LV_vol_BP_ed_2D_by_MOD", "122", "mL"),
    ]
    volumes = pd.DataFrame([
        dict(person_id=pid, index_date=date, echo_test_fpk=f"t{pid}",
             result_name=name, raw_value=value, reported_unit=unit)
        for pid, name, value, unit in src
    ])
    return preview, tests, volumes


def test_priority_scenario_never_infers_mayo_unit_or_compares_lower_tiers():
    _, _, volume = _fixtures()
    assert _method_feasibility(volume[volume.person_id.eq(1)],
                               "lv_end_diastolic_volume") == "biplane_candidate"
    assert _method_feasibility(volume[volume.person_id.eq(2)],
                               "lv_end_diastolic_volume") == "3d_candidate"
    assert _method_feasibility(volume[volume.person_id.eq(3)],
                               "lv_end_diastolic_volume") == "single_plane_candidate"
    assert _method_feasibility(volume[volume.person_id.eq(4)],
                               "lv_end_diastolic_volume") == "conflicting_preferred_tier"
    assert _method_feasibility(volume[volume.person_id.eq(5)],
                               "lv_end_diastolic_volume") == "no_explicit_ml"
    assert _method_feasibility(volume[volume.person_id.eq(6)],
                               "lv_end_diastolic_volume") == "conflicting_preferred_tier"
    assert _method_feasibility(volume[volume.person_id.eq(7)],
                               "lv_end_diastolic_volume") == "single_plane_candidate"


def test_selected_cohort_counts_and_excludes_unselected_exams():
    p, t, v = _fixtures()
    summary, tiers = summarize_explicit_ml_feasibility(
        p, t, v, small_cell=1
    )
    lv = summary.set_index("metric").loc["lv_end_diastolic_volume"]
    assert lv.selected_patient_days == "8"
    assert lv.with_explicit_ml == "6"
    assert lv.without_explicit_ml == "2"
    by_tier = tiers[tiers.metric.eq("lv_end_diastolic_volume")]
    assert by_tier.patient_days.tolist() == ["1", "1", "2", "2", "2"]
    esv = summary.set_index("metric").loc["lv_end_systolic_volume"]
    assert esv.with_explicit_ml == "0"
    assert esv.without_explicit_ml == "8"
    assert "person_id" not in summary and "raw_value" not in tiers
    assert len(tiers) == 10


def test_suppression_covers_small_complementary_and_tier_cells():
    p, t, v = _fixtures()
    summary, tiers = summarize_explicit_ml_feasibility(
        p, t, v, small_cell=4
    )
    lv = summary.set_index("metric").loc["lv_end_diastolic_volume"]
    assert lv.with_explicit_ml == "suppressed"
    assert lv.without_explicit_ml == "suppressed"
    assert (tiers.loc[tiers.metric.eq("lv_end_diastolic_volume"),
                      "patient_days"] == "suppressed").all()


def test_bad_inputs_fail_closed():
    p, t, v = _fixtures()
    with pytest.raises(ValueError, match="Duplicate preview"):
        summarize_explicit_ml_feasibility(
            pd.concat([p, p.iloc[[0]]]), t, v
        )
    with pytest.raises(ValueError, match="Volume rows missing"):
        summarize_explicit_ml_feasibility(
            p, t, v.drop(columns="reported_unit")
        )
    with pytest.raises(ValueError, match="small_cell"):
        summarize_explicit_ml_feasibility(p, t, v, small_cell=0)
