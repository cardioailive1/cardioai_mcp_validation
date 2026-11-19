"""Synthetic-only distinct-source test echo-volume topology checks."""
import pandas as pd
import pytest

from echo_wm_training.volume_record_topology import (
    summarize_volume_record_topology,
)

MC = "MC CV LV END DIASTOLIC VOLUME"
BP = "LV_vol_BP_ed_2D_by_MOD"


def _data():
    t, v = [], []

    def test(pid, day, key, time, acc=None, order=None):
        t.append({
            "person_id": pid, "index_date": day,
            "echo_test_fpk": key, "exam_datetime": day + " " + time,
            "accession_number": acc, "order_proc_id": order,
        })

    def volume(pid, day, fpk, result, value, unit):
        v.append({
            "person_id": pid, "index_date": day, "echo_test_fpk": fpk,
            "result_name": result, "raw_value": str(value),
            "reported_unit": unit,
        })

    day = "2021-06-01"
    # Same-day distinct FPKs; reciprocal 10s provisional ACC/ORD candidate.
    test(1, day, "mc1", "10:00:00", acc="A1")
    test(1, day, "ml1", "10:00:10", order="O1")
    volume(1, day, "mc1", MC, 120, None)
    volume(1, day, "ml1", BP, 123, "mL")
    # Separate source records 3 hours apart: same-day, not reciprocal.
    test(2, day, "mc2", "10:00:00", acc="A2")
    test(2, day, "ml2", "13:00:00", order="O2")
    volume(2, day, "mc2", MC, 110, "")
    volume(2, day, "ml2", BP, 113, "mL")
    # MC only.
    test(3, day, "mc3", "10:00:00", acc="A3")
    volume(3, day, "mc3", MC, 130, None)
    # Both families on the *same* exact source FPK: not another test.
    test(4, day, "same4", "10:00:00", acc="A4")
    volume(4, day, "same4", MC, 140, None)
    volume(4, day, "same4", BP, 141, "mL")
    # Two distinct test records have the same accession group (not ACC/ORD).
    test(5, day, "mc5", "10:00:00", acc="A5")
    test(5, day, "ml5", "10:00:20", acc="A5")
    volume(5, day, "mc5", MC, 150, None)
    volume(5, day, "ml5", BP, 149, "mL")
    # Nonempty MC unit is excluded from "unitless" audit.
    test(6, day, "mc6", "10:00:00", acc="A6")
    volume(6, day, "mc6", MC, 110, "mL")
    # Conflicting duplicates on one MC source record are excluded.
    test(7, day, "mc7", "10:00:00", acc="A7")
    volume(7, day, "mc7", MC, 110, None)
    volume(7, day, "mc7", MC, 115, None)
    # Older period counterpart, not added to 2019+.
    old = "2018-06-01"
    test(8, old, "mc8", "10:00:00", acc="A8")
    test(8, old, "ml8", "10:00:10", order="O8")
    volume(8, old, "mc8", MC, 100, None)
    volume(8, old, "ml8", BP, 101, "mL")
    return pd.DataFrame(t), pd.DataFrame(v)


def test_other_source_records_and_reciprocal_candidates_not_equivalent():
    tests, volumes = _data()
    cooccur, provisional = summarize_volume_record_topology(
        tests, volumes, small_cell=1
    )
    cur = cooccur.loc[
        cooccur.metric.eq("lv_end_diastolic_volume")
        & cooccur.period.eq("2019-2023")
    ].iloc[0]
    assert cur.unitless_mc_cv_patient_days == "5"
    assert cur.explicit_ml_other_test_patient_days == "3"
    assert cur.no_explicit_ml_other_test_patient_days == "2"
    p = provisional.loc[
        provisional.metric.eq("lv_end_diastolic_volume")
        & provisional.period.eq("2019-2023")
    ].iloc[0]
    assert p.provisional_acc_ord_pair_with_ml_days == "1"
    assert p.without_provisional_pair_with_ml_days == "4"
    old = provisional.loc[
        provisional.metric.eq("lv_end_diastolic_volume")
        & provisional.period.eq("2012-2018")
    ].iloc[0]
    assert old.provisional_acc_ord_pair_with_ml_days == "1"
    assert "person_id" not in cooccur and "echo_test_fpk" not in provisional


def test_primary_and_complement_suppression():
    tests, volumes = _data()
    cooccur, provisional = summarize_volume_record_topology(
        tests, volumes, small_cell=11
    )
    for frame in (cooccur, provisional):
        row = frame.loc[
            frame.metric.eq("lv_end_diastolic_volume")
            & frame.period.eq("2019-2023")
        ].iloc[0]
        assert all(value == "suppressed" for value in row.iloc[2:])
        zero_row = frame.loc[
            frame.metric.eq("lv_end_systolic_volume")
            & frame.period.eq("2019-2023")
        ].iloc[0]
        assert all(value == "0" for value in zero_row.iloc[2:])


def test_missing_identifier_and_measurement_fields_rejected():
    tests, volumes = _data()
    with pytest.raises(ValueError, match="Missing ECHO_TEST_FPK"):
        summarize_volume_record_topology(
            tests.assign(echo_test_fpk=None), volumes
        )
    with pytest.raises(ValueError, match="Missing volume measurement columns"):
        summarize_volume_record_topology(
            tests, volumes.drop(columns="reported_unit")
        )
    with pytest.raises(ValueError, match="small_cell"):
        summarize_volume_record_topology(
            tests, volumes, small_cell=0
        )
