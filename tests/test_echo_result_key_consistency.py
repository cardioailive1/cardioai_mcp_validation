"""Synthetic source-test/result encounter and order alignment tests."""
import pandas as pd
import pytest

from cardioai_mcp.config import load_study
from echo_wm_training.echo_result_key_consistency import (
    _relationship,
    source_echo_result_key_rows_sql,
    summarize_source_key_consistency,
)

MC_EDV = "MC CV LV END DIASTOLIC VOLUME"
BP_EDV = "LV_vol_BP_ed_2D_by_MOD"
MC_ESV = "MC CV LV END SYSTOLIC VOLUME"


def _row(pid, *, name=MC_EDV, date="2021-07-01",
         t_enc="E", r_enc="E", t_order="O", r_order="O",
         value="120", unit=None):
    return {
        "person_id": pid, "index_date": date, "echo_test_fpk": f"t{pid}",
        "result_name": name, "raw_value": value,
        "reported_unit": unit if name == MC_EDV else (
            "mL" if unit is None else unit
        ),
        "test_encounter": t_enc, "result_encounter": r_enc,
        "test_order": t_order, "result_order": r_order,
    }


def test_exact_source_same_test_key_agreement_categories():
    rows = pd.DataFrame([
        _row(1),
        _row(2, t_enc="A", r_enc="B", t_order="A", r_order="B"),
        _row(3, t_enc=None, r_enc=None, t_order=None, r_order=None),
        _row(4, t_enc="A", r_enc=None, t_order=None, r_order="O"),
        _row(5),
        # A second result on the same test has conflicting key evidence.
        {**_row(5), "result_encounter": "different"},
        _row(6, name=BP_EDV),
        _row(7, name=MC_ESV, value="40", unit=""),
        # Ignore non-mL comparator and MC results with explicit units.
        _row(8, name=BP_EDV, unit="cm"),
        _row(9, name=MC_EDV, unit="mL"),
        _row(10, name=BP_EDV, value="99999"),
        _row(11, date="2018-01-02"),
    ])
    out = summarize_source_key_consistency(rows, small_cell=1)
    m = out.loc[
        out.metric.eq("lv_end_diastolic_volume")
        & out.period.eq("2019-2023")
        & out.family.eq("mc_cv_unitless")
        & out.key.eq("encounter")
    ].set_index("status").patient_days.to_dict()
    assert m == {
        "equal_nonblank": "1", "unequal_nonblank": "1",
        "both_missing": "1", "only_one_missing": "1",
        "mixed_evidence": "1",
    }
    other = out.loc[
        out.metric.eq("lv_end_diastolic_volume")
        & out.period.eq("2019-2023")
        & out.family.eq("explicit_ml")
        & out.key.eq("encounter")
        & out.status.eq("equal_nonblank")
    ]
    assert other.patient_days.iloc[0] == "1"
    assert "person_id" not in out
    assert "raw_value" not in out


def test_key_alignment_missing_values_are_not_false_conflicts():
    assert _relationship(None, None) == "both_missing"
    assert _relationship(pd.NA, float("nan")) == "both_missing"
    assert _relationship("", "E") == "only_one_missing"
    assert _relationship("E", None) == "only_one_missing"
    assert _relationship(" E ", "E") == "equal_nonblank"
    assert _relationship("E", "Different") == "unequal_nonblank"


def test_suppression_masks_all_positive_small_cell_partitions():
    rows = pd.DataFrame([
        _row(i, t_enc="A" if i == 1 else "E", r_enc="B" if i == 1 else "E")
        for i in range(1, 35)
    ])
    out = summarize_source_key_consistency(rows, small_cell=11)
    group = out.loc[
        out.metric.eq("lv_end_diastolic_volume")
        & out.period.eq("2019-2023")
        & out.family.eq("mc_cv_unitless")
        & out.key.eq("encounter")
    ]
    assert (group.patient_days == "suppressed").all()


def test_query_uses_same_source_test_fpk_and_no_writes():
    sql = source_echo_result_key_rows_sql(
        load_study()["modality"], start="2019-01-01", end="2023-12-31"
    )
    assert "test_encounter" in sql
    assert "result_encounter" in sql
    assert "x.ENCOUNTER_NUMBER" in sql
    assert "t.ENCOUNTER_NUMBER" in sql
    assert "x.ORDER_PROC_ID" in sql
    assert "ECHO_TEST_FPK" in sql
    assert "FACT_ECHO_RESULTS" in sql
    assert "UPDATE " not in sql and "INSERT " not in sql


def test_missing_required_fields_and_duplicate_source_keys():
    data = pd.DataFrame([_row(1)])
    with pytest.raises(ValueError, match="Missing source"):
        summarize_source_key_consistency(
            data.drop(columns="result_order")
        )
    with pytest.raises(ValueError, match="Missing exact"):
        summarize_source_key_consistency(
            data.assign(echo_test_fpk=None)
        )
    with pytest.raises(ValueError, match="small_cell"):
        summarize_source_key_consistency(data, small_cell=0)
