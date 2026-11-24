"""Synthetic-only EIMS-first clinical volume candidate extraction tests.

No actual Mayo patient IDs, clinical values, or report text in this suite.
"""
import pandas as pd
import pytest

from echo_wm_training.eims_first_extractor import (
    _candidate_volume, stage_eims_first_volumes,
)

EDV_BP = "LV_vol_BP_ed_2D_by_MOD"
EDV_4C = "LV_vol_4C_ed_2D_by_MOD"
EDV_2C = "LV_vol_2C_ed_2D_by_MOD"
EDV_3D = "LV_vol_ed_3D"
ESV_BP = "LV_vol_BP_es_2D_by_MOD"
MC_EDV = "MC CV LV END DIASTOLIC VOLUME"


def _record(pid, fpk, name, value, *, unit="mL",
            source="12315", result_source="12315"):
    return {
        "person_id": pid, "index_date": "2022-04-05",
        "echo_test_fpk": fpk,
        "test_row_source_id": source,
        "result_row_source_id": result_source,
        "result_name": name, "raw_value": str(value),
        "reported_unit": unit,
    }


def _fixtures():
    cohort = pd.DataFrame([
        dict(person_id=i, index_date="2022-04-05")
        for i in range(1, 8)
    ])
    tests = pd.DataFrame([
        dict(person_id=pid, index_date="2022-04-05",
             echo_test_fpk=fpk, test_row_source_id=src)
        for pid, fpk, src in [
            (1, "e1", "12315"),                  # unique EIMS-only
            (2, "e2", "12315"), (2, "p2", "47918"),  # other source review
            (3, "e3a", "12315"), (3, "e3b", "12315"),  # ambiguity
            (4, "p4", "47918"),                  # no EIMS
            # 5: no TTE
            (6, "e6", "12315"),                  # preferred method conflict
            (7, "e7", "12315"),                  # lower tier valid, top tier invalid
        ]
    ])
    values = pd.DataFrame([
        _record(1, "e1", EDV_BP, 120),
        _record(1, "e1", EDV_BP, 120),   # duplicate identical
        _record(1, "e1", EDV_4C, 130),  # differs from biplane, ignored
        _record(1, "e1", ESV_BP, 45),
        _record(2, "e2", EDV_4C, 125),
        _record(2, "e2", EDV_2C, 125),
        _record(2, "p2", MC_EDV, 121, unit=None,
                source="47918", result_source="47918"),
        _record(3, "e3a", EDV_BP, 100),
        _record(3, "e3b", EDV_BP, 105),
        _record(6, "e6", EDV_BP, 115),
        _record(6, "e6", EDV_BP, 118),
        _record(6, "e6", EDV_3D, 117),
        _record(7, "e7", EDV_BP, 116, unit=None),
        _record(7, "e7", EDV_3D, 119),
    ])
    return cohort, tests, values


def test_staging_requires_same_fpk_explicit_ml_and_exposes_review_status():
    c, tests, results = _fixtures()
    staged = stage_eims_first_volumes(c, tests, results)
    assert len(staged) == len(c)
    by_pid = staged.set_index("person_id")
    assert by_pid.loc[1, "lv_end_diastolic_volume"] == 120
    assert by_pid.loc[1, "lv_end_diastolic_volume_method_tier"] == "biplane"
    assert by_pid.loc[1, "lv_end_systolic_volume"] == 45
    assert by_pid.loc[1, "exam_status"] == "single_eims_only_candidate"
    assert not bool(by_pid.loc[1, "lv_end_diastolic_volume_requires_exam_review"])

    assert by_pid.loc[2, "lv_end_diastolic_volume"] == 125
    assert by_pid.loc[2, "lv_end_diastolic_volume_method_tier"] == "single_plane"
    assert by_pid.loc[2, "exam_status"] == "single_eims_other_source_review"
    assert bool(by_pid.loc[2, "lv_end_diastolic_volume_requires_exam_review"])

    assert by_pid.loc[3, "exam_status"] == "multiple_eims_tests_review"
    assert pd.isna(by_pid.loc[3, "lv_end_diastolic_volume"])
    assert by_pid.loc[4, "exam_status"] == "no_eims_tte"
    assert by_pid.loc[5, "exam_status"] == "no_conventional_tte"
    assert by_pid.loc[6, "lv_end_diastolic_volume_status"] == (
        "conflicting_preferred_tier"
    )
    assert pd.isna(by_pid.loc[6, "lv_end_diastolic_volume"])
    assert by_pid.loc[7, "lv_end_diastolic_volume"] == 119
    assert by_pid.loc[7, "lv_end_diastolic_volume_method_tier"] == "3d"
    assert (staged.extraction_state == "staged_not_approved").all()


def test_method_status_and_private_numeric_value_always_agree():
    c, tests, results = _fixtures()
    staged = stage_eims_first_volumes(c, tests, results)
    for metric in ("lv_end_diastolic_volume", "lv_end_systolic_volume"):
        eligible = staged[metric].notna()
        assert (staged.loc[eligible, metric + "_status"]
                .str.endswith("_candidate")).all()
        assert staged.loc[~eligible, metric + "_method_tier"].isna().all()

    # Selecting a clinically arbitrary lower method when the preferred
    # tier conflicts would hide a disagreement.
    sub = pd.DataFrame([
        {"result_name": EDV_BP, "raw_value": "120", "reported_unit": "mL"},
        {"result_name": EDV_BP, "raw_value": "124", "reported_unit": "mL"},
        {"result_name": EDV_3D, "raw_value": "122", "reported_unit": "mL"},
    ])
    value, status, method = _candidate_volume(
        sub, "lv_end_diastolic_volume"
    )
    assert value is None and method is None
    assert status == "conflicting_preferred_tier"


def test_source_and_exam_safety_fail_closed():
    c, tests, results = _fixtures()
    mismatch = results.copy()
    mismatch.loc[mismatch.echo_test_fpk.eq("e1"), "test_row_source_id"] = "47918"
    with pytest.raises(ValueError, match="Inconsistent source"):
        stage_eims_first_volumes(c, tests, mismatch)

    not_eims = results.copy()
    not_eims.loc[not_eims.echo_test_fpk.eq("e1"),
                 "result_row_source_id"] = "47918"
    staged = stage_eims_first_volumes(c, tests, not_eims)
    by_pid = staged.set_index("person_id")
    assert pd.isna(by_pid.loc[1, "lv_end_diastolic_volume"])
    assert pd.isna(by_pid.loc[1, "lv_end_systolic_volume"])

    double = pd.concat([
        tests, tests.loc[tests.echo_test_fpk.eq("e1")].assign(
            test_row_source_id="47918"
        ),
    ], ignore_index=True)
    with pytest.raises(ValueError, match="Conflicting source"):
        stage_eims_first_volumes(c, double, results)


def test_cohort_cardinality_missing_columns_and_duplicates():
    c, tests, results = _fixtures()
    duplicate = pd.concat([c, c.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        stage_eims_first_volumes(duplicate, tests, results)
    with pytest.raises(ValueError, match="Missing EIMS test FPK"):
        stage_eims_first_volumes(c, tests.drop(columns="test_row_source_id"),
                                 results)
    with pytest.raises(ValueError, match="Missing EIMS test/result"):
        stage_eims_first_volumes(c, tests,
                                 results.drop(columns="result_row_source_id"))
    # Identical duplicate test rows do not create a second exam.
    repeated = pd.concat([tests, tests.iloc[[0]]], ignore_index=True)
    actual = stage_eims_first_volumes(c, repeated, results)
    expected = stage_eims_first_volumes(c, tests, results)
    pd.testing.assert_frame_equal(actual, expected)


def test_no_eims_measurement_is_inferred_from_epic_or_unitless_data():
    c, tests, results = _fixtures()
    # Unitless EIMS MC CV name should never create a candidate.
    only = pd.DataFrame([
        _record(1, "e1", MC_EDV, 125, unit=None)
    ])
    staged = stage_eims_first_volumes(c, tests, only).set_index("person_id")
    assert pd.isna(staged.loc[1, "lv_end_diastolic_volume"])
    assert staged.loc[1, "lv_end_diastolic_volume_status"] == "no_explicit_ml"


def test_candidate_counts_reconcile_with_prior_eims_feasibility():
    from echo_wm_training.eims_volume_eligibility import (
        summarize_eims_volume_eligibility,
    )

    cohort, tests, source_rows = _fixtures()
    staged = stage_eims_first_volumes(cohort, tests, source_rows)
    _, counts = summarize_eims_volume_eligibility(
        cohort, tests, source_rows, small_cell=1, report_style="compact"
    )
    for metric in ("lv_end_diastolic_volume", "lv_end_systolic_volume"):
        from_staged = int(staged[metric].notna().sum())
        from_feasibility = counts.loc[
            counts.metric.eq(metric)
            & counts.availability_category.eq("explicit_ml_candidate"),
            "sole_eims_fpk_patient_days",
        ]
        assert from_staged == int(from_feasibility.iloc[0])
