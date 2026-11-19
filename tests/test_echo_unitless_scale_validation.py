"""Synthetic only: same-ECHO_TEST_FPK unitless Mayo result scale review."""
import pandas as pd
import pytest

from echo_wm_training.unitless_scale_validation import (
    compare_unitless_with_labeled_ml,
)

MC_EDV = "MC CV LV END DIASTOLIC VOLUME"
BP_EDV = "LV_vol_BP_ed_2D_by_MOD"
FOUR_EDV = "LV_vol_4C_ed_2D_by_MOD"


def _add(tests, volumes, pid, day, fpk, mc=None, biplane=None, *, mc_unit=None, bp_unit="mL"):
    tests.append({
        "person_id": pid, "index_date": day, "echo_test_fpk": fpk
    })
    if mc is not None:
        volumes.append({
            "person_id": pid, "index_date": day, "echo_test_fpk": fpk,
            "result_name": MC_EDV, "raw_value": str(mc),
            "reported_unit": mc_unit,
        })
    if biplane is not None:
        volumes.append({
            "person_id": pid, "index_date": day, "echo_test_fpk": fpk,
            "result_name": BP_EDV, "raw_value": str(biplane),
            "reported_unit": bp_unit,
        })


def test_pairs_only_same_exact_test_and_separates_periods():
    t, v = [], []
    # Different periods, same exact source result-unit semantics.
    _add(t, v, 1, "2018-12-01", "t1", 120, 120)
    _add(t, v, 2, "2020-01-01", "t2", 130, 125)
    # A patient/day with different test FPKs is NOT a pair.
    _add(t, v, 3, "2020-01-01", "t3a", 140, None)
    _add(t, v, 3, "2020-01-01", "t3b", None, 140)
    # Explicit comparator has a non-mL unit; cannot establish scale.
    _add(t, v, 4, "2020-01-01", "t4", 130, 130, bp_unit="cc")
    # MC CV must truly lack metadata, even when source label is recognized.
    _add(t, v, 5, "2020-01-01", "t5", 120, 120, mc_unit="mL")
    # Different numeric values for same named result are ambiguous.
    _add(t, v, 6, "2020-01-01", "t6", 120, 120)
    v.append({
        "person_id": 6, "index_date": "2020-01-01",
        "echo_test_fpk": "t6", "result_name": MC_EDV,
        "raw_value": "130", "reported_unit": None,
    })
    # Two different source tests both contain pairs; do not publish a
    # method's ratio for that patient-day, though presence is still true.
    _add(t, v, 7, "2020-01-01", "t7a", 100, 100)
    _add(t, v, 7, "2020-01-01", "t7b", 105, 105)

    summary, ratios = compare_unitless_with_labeled_ml(
        pd.DataFrame(t), pd.DataFrame(v), small_cell=1
    )
    s = summary.set_index(["metric", "period"])
    old = s.loc[("lv_end_diastolic_volume", "2012-2018")]
    assert old.unambiguous_unitless_mc_cv_patient_days == "1"
    assert old.paired_same_test_fpk_patient_days == "1"
    current = s.loc[("lv_end_diastolic_volume", "2019-2023")]
    assert current.unambiguous_unitless_mc_cv_patient_days == "4"  # 2, 3, 4, 7
    assert current.paired_same_test_fpk_patient_days == "2"  # 2, 7
    assert current.no_paired_test_fpk_patient_days == "2"
    r = ratios[
        ratios.metric.eq("lv_end_diastolic_volume")
        & ratios.period.eq("2019-2023")
        & ratios.comparator.eq("biplane_2d")
    ].iloc[0]
    assert r.patient_days == "1"  # only t2, not the two t7 source records
    assert "person_id" not in summary and "echo_test_fpk" not in ratios


def test_evidence_of_scale_requires_large_enough_unambiguous_sample():
    t, v = [], []
    for pid in range(1, 51):
        # 25 matching at ratio 1, 25 at ratio 1.5, without
        # asserting either is an appropriate clinical result.
        unitless = 100 if pid <= 25 else 150
        _add(t, v, pid, "2021-07-01", f"f{pid}", unitless, 100)
    summary, ratios = compare_unitless_with_labeled_ml(
        pd.DataFrame(t), pd.DataFrame(v), small_cell=11
    )
    main = summary.set_index(["metric", "period"]).loc[
        ("lv_end_diastolic_volume", "2019-2023")
    ]
    assert main.unambiguous_unitless_mc_cv_patient_days == "50"
    assert main.paired_same_test_fpk_patient_days == "50"
    assert main.no_paired_test_fpk_patient_days == "0"
    r = ratios.loc[
        ratios.metric.eq("lv_end_diastolic_volume")
        & ratios.period.eq("2019-2023")
        & ratios.comparator.eq("biplane_2d")
    ].iloc[0]
    assert r.patient_days == "50"
    assert r.median_mc_over_ml == "1.250"
    assert r.pct_within_10pct == "50.0"
    assert r.pct_within_25pct == "50.0"
    # With no observations, the sample count is exactly zero, not
    # disclosure-suppressed. Percentages and quantiles are inapplicable.
    absent = ratios.loc[ratios.period.eq("2012-2018")]
    assert (absent.patient_days == "0").all()
    assert (absent.median_mc_over_ml == "not_applicable").all()


def test_complementary_disclosure_suppresses_full_comparison():
    t, v = [], []
    for pid in range(1, 31):
        mc = 100 if pid <= 28 else 150
        _add(t, v, pid, "2022-02-02", f"t{pid}", mc, 100)
    _, ratio = compare_unitless_with_labeled_ml(
        pd.DataFrame(t), pd.DataFrame(v), small_cell=11
    )
    r = ratio.loc[
        ratio.metric.eq("lv_end_diastolic_volume")
        & ratio.period.eq("2019-2023")
        & ratio.comparator.eq("biplane_2d")
    ].iloc[0]
    assert r.patient_days == "suppressed"
    assert r.median_mc_over_ml == "suppressed"
    assert r.pct_within_10pct == "suppressed"


def test_mismatched_columns_and_invalid_small_cell_fail_closed():
    t = pd.DataFrame([{
        "person_id": 1, "index_date": "2020-01-01",
        "echo_test_fpk": "t1",
    }])
    v = pd.DataFrame([{
        **t.iloc[0].to_dict(), "result_name": MC_EDV,
        "raw_value": "100", "reported_unit": None,
    }])
    with pytest.raises(ValueError, match="Missing echo test identifier"):
        compare_unitless_with_labeled_ml(
            t.assign(echo_test_fpk=None), v
        )
    with pytest.raises(ValueError, match="Volume rows missing"):
        compare_unitless_with_labeled_ml(t, v.drop(columns="reported_unit"))
    with pytest.raises(ValueError, match="small_cell"):
        compare_unitless_with_labeled_ml(t, v, small_cell=0)
