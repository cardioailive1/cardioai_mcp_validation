"""Synthetic-only EIMS volume candidate eligibility and disclosure tests."""
import pandas as pd
import pytest

from cardioai_mcp.config import load_study
from echo_wm_training.eims_volume_eligibility import (
    eims_test_source_sql, summarize_eims_volume_eligibility
)

EDV_BP = "LV_vol_BP_ed_2D_by_MOD"
EDV_4C = "LV_vol_4C_ed_2D_by_MOD"
ESV_BP = "LV_vol_BP_es_2D_by_MOD"


def _fixtures():
    day = "2021-03-22"
    cohort = pd.DataFrame([
        {"person_id": pid, "index_date": day}
        for pid in range(1, 8)
    ])
    tests = [
        # 1: sole-source EIMS, EIMS biplane mL candidate.
        (1, "e1", "12315"),
        # 2: one EIMS test plus separate Epic TTE on index date.
        (2, "e2", "12315"), (2, "p2", "47918"),
        # 3: two distinct EIMS FPKs, no selection allowed.
        (3, "e3a", "12315"), (3, "e3b", "12315"),
        # 4: Epic TTE but no EIMS test.
        (4, "p4", "47918"),
        # 5: no conventional TTE.
        # 6: sole EIMS with no qualifying mL measurement.
        (6, "e6", "12315"),
        # 7: one EIMS source test with conflicting biplane results.
        (7, "e7", "12315"),
    ]
    t = pd.DataFrame([
        {"person_id": pid, "index_date": day,
         "echo_test_fpk": fpk, "test_row_source_id": sid}
        for pid, fpk, sid in tests
    ])
    data = [
        (1, "e1", EDV_BP, "120", "mL", "12315"),
        (1, "e1", ESV_BP, "55", "mL", "12315"),
        (2, "e2", EDV_4C, "133", "mL", "12315"),
        (2, "p2", EDV_BP, "118", "mL", "47918"),
        (3, "e3a", EDV_BP, "120", "mL", "12315"),
        (3, "e3b", EDV_BP, "121", "mL", "12315"),
        (6, "e6", EDV_BP, "120", None, "12315"),
        (7, "e7", EDV_BP, "120", "mL", "12315"),
        (7, "e7", EDV_BP, "123", "mL", "12315"),
    ]
    v = pd.DataFrame([
        {"person_id": pid, "index_date": day,
         "echo_test_fpk": fpk, "result_name": name,
         "raw_value": val, "reported_unit": unit,
         "test_row_source_id": source_id,
         "result_row_source_id": source_id,
         }
        for pid, fpk, name, val, unit, source_id in data
    ])
    return cohort, t, v


def test_source_specific_exam_topology_and_no_cross_exam_substitution():
    cohort, tests, volume = _fixtures()
    exam, metrics = summarize_eims_volume_eligibility(
        cohort, tests, volume, small_cell=1
    )
    status = exam.loc[exam.period.eq("2019-2023")].set_index(
        "source_test_category"
    ).patient_days.to_dict()
    assert status == {
        "no_conventional_tte": "1",
        "no_eims_tte": "1",
        "multiple_eims_source_tests": "1",
        "single_eims_test_only": "3",
        "single_eims_test_other_source_present": "1",
    }
    edv = metrics.loc[
        metrics.period.eq("2019-2023")
        & metrics.metric.eq("lv_end_diastolic_volume")
    ].set_index("availability_category").sole_eims_fpk_patient_days.to_dict()
    assert edv == {
        "explicit_ml_candidate": "2",
        "preferred_tier_conflict": "1",
        "no_explicit_ml": "1",
    }
    esv = metrics.loc[
        metrics.period.eq("2019-2023")
        & metrics.metric.eq("lv_end_systolic_volume")
    ].set_index("availability_category").sole_eims_fpk_patient_days.to_dict()
    assert esv == {
        "explicit_ml_candidate": "1",
        "preferred_tier_conflict": "0",
        "no_explicit_ml": "3",
    }
    assert "person_id" not in exam and "echo_test_fpk" not in metrics
    assert "raw_value" not in metrics
    assert (exam.loc[exam.period.eq("2012-2018"), "patient_days"] == "0").all()


def test_small_cell_in_method_suppresses_exam_topology_as_well():
    c, t, v = _fixtures()
    exam, method = summarize_eims_volume_eligibility(
        c, t, v, small_cell=11
    )
    assert (exam.loc[
        exam.period.eq("2019-2023"), "patient_days"
    ] == "suppressed").all()
    assert (method.loc[
        method.period.eq("2019-2023"), "sole_eims_fpk_patient_days"
    ] == "suppressed").all()


def test_duplicate_source_test_rows_dont_change_exam_classification():
    c, t, v = _fixtures()
    duplicate = pd.concat([t, t.iloc[[0]]], ignore_index=True)
    original = summarize_eims_volume_eligibility(c, t, v, small_cell=1)
    repeated = summarize_eims_volume_eligibility(c, duplicate, v, small_cell=1)
    pd.testing.assert_frame_equal(original[0], repeated[0])
    pd.testing.assert_frame_equal(original[1], repeated[1])


def test_inconsistent_source_identifiers_fail_closed():
    c, t, v = _fixtures()
    mismatch = v.copy()
    mismatch.loc[
        mismatch.echo_test_fpk.eq("e1"), "result_row_source_id"
    ] = "47918"
    # Never substitute an Epic measurement on an EIMS test.
    # Only matching both provenance fields produces eligible source rows.
    exam, outcomes = summarize_eims_volume_eligibility(
        c, t, mismatch, small_cell=1
    )
    current = outcomes.loc[
        outcomes.period.eq("2019-2023")
        & outcomes.metric.eq("lv_end_systolic_volume")
    ].set_index("availability_category")
    assert current.loc["explicit_ml_candidate",
                       "sole_eims_fpk_patient_days"] == "0"

    mismatch_test = v.copy()
    mismatch_test.loc[
        mismatch_test.echo_test_fpk.eq("e1"), "test_row_source_id"
    ] = "47918"
    with pytest.raises(ValueError, match="Inconsistent source"):
        summarize_eims_volume_eligibility(c, t, mismatch_test, small_cell=1)

    repeated_conflict = pd.concat([
        t, t.loc[t.echo_test_fpk.eq("e1")].assign(
            test_row_source_id="47918"
        )
    ], ignore_index=True)
    with pytest.raises(ValueError, match="Conflicting source"):
        summarize_eims_volume_eligibility(c, repeated_conflict, v, small_cell=1)




def test_compact_all_era_rollup_keeps_exam_and_method_rules():
    c, t, v = _fixtures()
    exam, methods = summarize_eims_volume_eligibility(
        c, t, v, small_cell=1, report_style="compact"
    )
    assert set(exam.columns) == {"source_test_category", "patient_days"}
    assert exam.set_index("source_test_category").patient_days.to_dict() == {
        "exactly_one_eims_test": "4",
        "other_or_ambiguous": "3",
    }
    edv = methods.loc[
        methods.metric.eq("lv_end_diastolic_volume")
    ].set_index("availability_category").sole_eims_fpk_patient_days
    assert edv.to_dict() == {
        "explicit_ml_candidate": "2",
        "no_candidate_or_preferred_tier_conflict": "2",
    }
    esv = methods.loc[
        methods.metric.eq("lv_end_systolic_volume")
    ].set_index("availability_category").sole_eims_fpk_patient_days
    assert esv.to_dict() == {
        "explicit_ml_candidate": "1",
        "no_candidate_or_preferred_tier_conflict": "3",
    }
    assert "period" not in exam
    assert "period" not in methods
    assert "person_id" not in exam
    assert "echo_test_fpk" not in methods


def test_compact_rollup_joint_complementary_suppression_and_invalid_style():
    c, t, v = _fixtures()
    exam, methods = summarize_eims_volume_eligibility(
        c, t, v, small_cell=3, report_style="compact"
    )
    # EDV has 2 qualifying method candidates, so hide BOTH entire
    # tables to protect complements and the known cohort denominator.
    assert (exam.patient_days == "suppressed").all()
    assert (methods.sole_eims_fpk_patient_days == "suppressed").all()
    with pytest.raises(ValueError, match="report_style"):
        summarize_eims_volume_eligibility(
            c, t, v, small_cell=1, report_style="unknown"
        )


def test_compact_rollup_reconciles_two_eras_without_exposing_period_cells():
    c, t, v = _fixtures()
    # Include a patient/index-day in the historical period; patient 8
    # has one EIMS test with explicit mL candidate for both volumes.
    day = "2018-04-18"
    c = pd.concat([c, pd.DataFrame([{
        "person_id": 8, "index_date": day
    }])], ignore_index=True)
    t = pd.concat([t, pd.DataFrame([{
        "person_id": 8, "index_date": day,
        "echo_test_fpk": "e8", "test_row_source_id": "12315",
    }])], ignore_index=True)
    v = pd.concat([v, pd.DataFrame([
        {
            "person_id": 8, "index_date": day, "echo_test_fpk": "e8",
            "result_name": method, "raw_value": value,
            "reported_unit": "mL", "test_row_source_id": "12315",
            "result_row_source_id": "12315",
        }
        for method, value in [(EDV_BP, "110"), (ESV_BP, "44")]
    ])], ignore_index=True)
    exam, methods = summarize_eims_volume_eligibility(
        c, t, v, small_cell=1, report_style="compact"
    )
    assert exam.set_index("source_test_category").patient_days.loc[
        "exactly_one_eims_test"
    ] == "5"
    edv = methods.loc[
        methods.metric.eq("lv_end_diastolic_volume")
        & methods.availability_category.eq("explicit_ml_candidate")
    ]
    assert edv.sole_eims_fpk_patient_days.iloc[0] == "3"


def test_must_use_real_eims_source_ids_and_read_only_sql():
    query = eims_test_source_sql(
        load_study()["modality"],
        start="2012-01-01", end="2023-12-31",
    )
    assert "ROW_SOURCE_ID" in query and "ECHO_TEST_FPK" in query
    assert "echo_analytic" in query
    assert "UPDATE " not in query and "INSERT " not in query


def test_duplicate_cohort_and_missing_fields_rejected():
    c, t, v = _fixtures()
    with pytest.raises(ValueError, match="unique"):
        summarize_eims_volume_eligibility(
            pd.concat([c, c.iloc[[0]]]), t, v
        )
    with pytest.raises(ValueError, match="Missing saved-cohort"):
        summarize_eims_volume_eligibility(
            c, t.drop(columns="test_row_source_id"), v
        )
    with pytest.raises(ValueError, match="Missing source volume"):
        summarize_eims_volume_eligibility(
            c, t, v.drop(columns="reported_unit")
        )
    with pytest.raises(ValueError, match="small_cell"):
        summarize_eims_volume_eligibility(c, t, v, small_cell=0)
