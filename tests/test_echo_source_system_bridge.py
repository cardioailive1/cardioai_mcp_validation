"""Synthetic Mayo source-system provenance bridge tests; no clinical data."""
import pandas as pd
import pytest

from cardioai_mcp.config import load_study
from echo_wm_training.source_system_bridge import (
    source_system_bridge_rows_sql,
    source_system_dimension_sql,
    summarize_source_system_bridge,
)

MC = "MC CV LV END DIASTOLIC VOLUME"
BP = "LV_vol_BP_ed_2D_by_MOD"


def _row(pid, *, result_name=MC, unit=None, test_id="t", 
         row_source="47918", test_source="47918", test_dk="dkA",
         test_key="keyA", result_key="keyR"):
    return dict(
        person_id=pid, index_date="2021-01-02",
        echo_test_fpk=f"{test_id}{pid}", result_name=result_name,
        raw_value="120", reported_unit=unit,
        test_row_source_id=test_source,
        result_row_source_id=row_source,
        test_source_system_dk=test_dk,
        test_source_system_key=test_key,
        result_source_system_key=result_key,
    )


def _dimension():
    return pd.DataFrame([
        dict(source_system_dk="dkA", source_system_name="Echo Interface A"),
        dict(source_system_dk="dkB", source_system_name="Echo Interface B"),
        # A source ROW_SOURCE_ID is NOT presumed to be a dimension DK.
        dict(source_system_dk="47918", source_system_name="Wrong assumed join"),
    ])


def test_source_system_lookup_uses_test_system_dk_not_row_source():
    data = pd.DataFrame([
        _row(1),
        _row(2, result_name=BP, unit="mL", row_source="12315",
             test_source="12315", test_dk="dkB"),
        _row(3, row_source="12315", test_source="12315",
             test_dk="unknown"),
        _row(4, test_dk="dkA"),
        _row(4, test_id="alt", test_dk="dkB"),
    ])
    summary = summarize_source_system_bridge(
        data, _dimension(), small_cell=1
    )
    mc = summary.loc[
        summary.metric.eq("lv_end_diastolic_volume")
        & summary.family.eq("mc_cv_unitless")
    ]
    by_id = mc.set_index("result_row_source_id")
    assert by_id.loc["47918", "test_source_system_name"] == (
        "Echo Interface A"
    )
    assert by_id.loc["12315", "test_source_system_name"] == (
        "UNKNOWN_DIMENSION_DK"
    )
    assert "MULTIPLE_KEY_TUPLES" in set(mc.result_row_source_id)
    ml = summary.loc[summary.family.eq("explicit_ml")
                     & summary.metric.eq("lv_end_diastolic_volume")]
    assert ml.loc[ml.result_row_source_id.eq("12315"),
                  "test_source_system_name"].iloc[0] == "Echo Interface B"
    assert "person_id" not in summary and "raw_value" not in summary


def test_small_source_tuple_suppresses_every_partition_cell():
    data = pd.DataFrame([
        _row(i, test_dk="dkA" if i < 30 else "dkB")
        for i in range(1, 31)
    ])
    summary = summarize_source_system_bridge(
        data, _dimension(), small_cell=11
    )
    current = summary.loc[
        summary.metric.eq("lv_end_diastolic_volume")
        & summary.period.eq("2019-2023")
        & summary.family.eq("mc_cv_unitless")
    ]
    assert (current.patient_days == "suppressed").all()


def test_source_system_dimension_must_not_infer_other_keys():
    dim = pd.concat([
        _dimension(), pd.DataFrame([
            dict(source_system_dk="dkA", source_system_name="Another name")
        ])
    ], ignore_index=True)
    result = summarize_source_system_bridge(
        pd.DataFrame([_row(1)]), dim, small_cell=1
    )
    group = result.loc[
        result.metric.eq("lv_end_diastolic_volume")
        & result.family.eq("mc_cv_unitless")
    ]
    assert group.test_source_system_name.iloc[0] == "AMBIGUOUS_DIMENSION_DK"


def test_sql_includes_exact_provenance_keys_and_uses_read_only_select():
    study = load_study()
    query = source_system_bridge_rows_sql(
        study["modality"], start="2019-01-01", end="2023-12-31"
    )
    for expected in (
        "t.ROW_SOURCE_ID", "x.ROW_SOURCE_ID",
        "t.SOURCE_SYSTEM_DK", "t.SOURCE_SYSTEM_KEY",
        "x.SOURCE_SYSTEM_KEY", "ECHO_TEST_FPK",
    ):
        assert expected in query
    dsql = source_system_dimension_sql(study["modality"])
    assert "DIM_SOURCE_SYSTEM" in dsql
    assert "SOURCE_SYSTEM_NAME" in dsql
    assert "UPDATE " not in query and "INSERT " not in query


def test_empty_source_data_and_invalid_schema_fail_closed():
    cols = pd.DataFrame([_row(1)]).columns
    out = summarize_source_system_bridge(
        pd.DataFrame(columns=cols), _dimension()
    )
    assert (out.patient_days == "0").all()
    with pytest.raises(ValueError, match="Missing echo volume"):
        summarize_source_system_bridge(
            pd.DataFrame([_row(1)]).drop(columns="test_source_system_dk"),
            _dimension()
        )
    with pytest.raises(ValueError, match="small_cell"):
        summarize_source_system_bridge(
            pd.DataFrame(columns=cols), _dimension(), small_cell=0
        )
