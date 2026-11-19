"""Synthetic-only encounter/order/accession echo source identity signals."""
import pandas as pd
import pytest

from cardioai_mcp.config import load_study
from echo_wm_training.encounter_volume_linkage import (
    encounter_test_rows_sql,
    summarize_encounter_volume_linkage,
)

MC = "MC CV LV END DIASTOLIC VOLUME"
BP = "LV_vol_BP_ed_2D_by_MOD"


def _fixtures():
    tests, values = [], []
    day = "2021-06-15"

    def add(pid, fpk, name, unit, *,
            encounter=None, order=None, accession=None,
            result=120, date=day):
        tests.append(dict(
            person_id=pid, index_date=date, echo_test_fpk=fpk,
            encounter_number=encounter, order_proc_id=order,
            accession_number=accession,
        ))
        values.append(dict(
            person_id=pid, index_date=date, echo_test_fpk=fpk,
            result_name=name, raw_value=str(result),
            reported_unit=unit,
        ))
    # No separate mL test record.
    add(1, "m1", MC, None, encounter="e1")
    # Multiple separate mL tests, no arbitrary selection.
    add(2, "m2", MC, None, encounter="e2")
    add(2, "x2", BP, "mL", encounter="e2")
    add(2, "y2", BP, "mL", encounter="e2")
    # Unique match with shared order and encounter.
    add(3, "m3", MC, None, encounter="e3", order="o3")
    add(3, "x3", BP, "mL", encounter="e3", order="o3")
    # Unique match with shared encounter but distinct/nonmatching orders.
    add(4, "m4", MC, None, encounter="e4", order="o4a")
    add(4, "x4", BP, "mL", encounter="e4", order="o4b")
    # Contradictory encounters trump otherwise matching order numbers.
    add(5, "m5", MC, None, encounter="e5a", order="o5")
    add(5, "x5", BP, "mL", encounter="e5b", order="o5")
    # Unique other-source measurement without a shared linkage key.
    add(6, "m6", MC, None, encounter="e6a")
    add(6, "x6", BP, "mL", encounter=None)
    # Different time period remains separate.
    old = "2018-06-15"
    add(7, "m7", MC, None, encounter="e7", date=old)
    add(7, "x7", BP, "mL", encounter="e7", date=old)
    return pd.DataFrame(tests), pd.DataFrame(values)


def test_identity_signals_are_exclusive_and_not_assumed_clinical_joins():
    t, v = _fixtures()
    out = summarize_encounter_volume_linkage(t, v, small_cell=1)
    current = out.loc[
        out.metric.eq("lv_end_diastolic_volume")
        & out.period.eq("2019-2023")
    ].set_index("linkage_category").patient_days.to_dict()
    assert current == {
        "no_explicit_ml_other_test": "1",
        "multiple_possible_test_pairs": "1",
        "unique_shared_order_or_accession": "1",
        "unique_shared_encounter_only": "1",
        "unique_conflicting_encounters": "1",
        "unique_without_shared_identifiers": "1",
    }
    old = out.loc[
        out.metric.eq("lv_end_diastolic_volume")
        & out.period.eq("2012-2018")
        & out.linkage_category.eq("unique_shared_encounter_only")
    ]
    assert old.patient_days.iloc[0] == "1"
    assert "person_id" not in out and "echo_test_fpk" not in out


def test_small_positive_category_suppresses_complete_partition():
    t, v = _fixtures()
    out = summarize_encounter_volume_linkage(t, v, small_cell=11)
    edv = out.loc[
        out.metric.eq("lv_end_diastolic_volume")
        & out.period.eq("2019-2023")
    ]
    assert (edv.patient_days == "suppressed").all()
    zero_esv = out.loc[out.metric.eq("lv_end_systolic_volume")]
    assert (zero_esv.patient_days == "0").all()


def test_source_test_sql_uses_confirmed_mayo_encounter_column():
    query = encounter_test_rows_sql(
        load_study()["modality"], start="2019-01-01", end="2023-12-31"
    )
    assert "test.ENCOUNTER_NUMBER" in query
    assert "ECHO_TEST_FPK" in query
    assert "FACT_ECHO_TEST" in query
    assert "echo_analytic" in query
    assert "UPDATE " not in query and "INSERT " not in query


def test_missing_linkage_columns_and_conflicting_source_ids_rejected():
    t, v = _fixtures()
    with pytest.raises(ValueError, match="Test rows missing"):
        summarize_encounter_volume_linkage(
            t.drop(columns="encounter_number"), v
        )
    conflict = t.iloc[[0]].copy()
    conflict.loc[:, "encounter_number"] = "new_encounter"
    with pytest.raises(ValueError, match="Conflicting test identifiers"):
        summarize_encounter_volume_linkage(
            pd.concat([t, conflict], ignore_index=True), v
        )
    with pytest.raises(ValueError, match="Volume rows missing"):
        summarize_encounter_volume_linkage(
            t, v.drop(columns="reported_unit")
        )
    with pytest.raises(ValueError, match="small_cell"):
        summarize_encounter_volume_linkage(t, v, small_cell=0)
