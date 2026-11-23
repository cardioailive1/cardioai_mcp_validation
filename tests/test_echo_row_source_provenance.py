"""Synthetic provenance tests: no patient data or source result values exported."""
import pandas as pd
import pytest

from cardioai_mcp.config import load_study
from echo_wm_training.row_source_provenance import (
    _source_code,
    source_id_metadata_sql,
    source_id_fields,
    summarize_volume_source_ids,
    volume_source_rows_sql,
)

MC_EDV = "MC CV LV END DIASTOLIC VOLUME"
BP_EDV = "LV_vol_BP_ed_2D_by_MOD"


def _row(pid, *, result_name=MC_EDV, value=120, unit=None,
         test_source=None, result_source="Epic", fpk=None):
    return {
        "person_id": pid, "index_date": "2021-03-05",
        "echo_test_fpk": fpk or f"t{pid}",
        "result_name": result_name, "raw_value": str(value),
        "reported_unit": unit,
        "test_source_id": test_source,
        "result_source_id": result_source,
    }


def test_metadata_identifies_supported_source_field_locations():
    schema = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
    sql = source_id_metadata_sql(schema)
    assert "INFORMATION_SCHEMA.COLUMNS" in sql
    assert "ROW_SOURCE" in sql
    assert "FACT_ECHO_RESULTS" in sql
    with pytest.raises(ValueError, match="project.dataset"):
        source_id_metadata_sql("something; DROP TABLE anything")
    meta = pd.DataFrame([
        dict(table_name="FACT_ECHO_TEST", column_name="ROW_SOURCE_ID"),
        dict(table_name="FACT_ECHO_RESULTS", column_name="ROW_SOURCE_ID"),
    ])
    assert source_id_fields(meta) == (True, True)
    assert source_id_fields(meta.iloc[:1]) == (True, False)
    with pytest.raises(ValueError, match="metadata"):
        source_id_fields(meta.drop(columns="column_name"))


def test_source_query_never_references_missing_optional_source_field():
    cfg = load_study()["modality"]
    query = volume_source_rows_sql(
        cfg, start="2019-01-01", end="2023-12-31",
        test_has_source=False, result_has_source=True,
    )
    assert "x.ROW_SOURCE_ID" in query
    assert "t.ROW_SOURCE_ID" not in query
    assert "CAST(NULL AS STRING) AS test_source_id" in query
    assert "ECHO_TEST_FPK" in query
    assert "echo_analytic" in query
    assert "UPDATE " not in query
    with pytest.raises(ValueError, match="Neither"):
        volume_source_rows_sql(
            cfg, start="2019-01-01", end="2023-12-31",
            test_has_source=False, result_has_source=False,
        )


def test_missing_and_disagreeing_test_vs_result_provenance_are_distinct():
    assert _source_code(None, "Epic") == "ID:Epic"
    assert _source_code("Epic", None) == "ID:Epic"
    assert _source_code("A", "B") == "TEST_RESULT_SOURCE_DIFFER"
    assert _source_code(pd.NA, float("nan")) == "NO_SOURCE_ID"
    assert _source_code("  A ", "A") == "ID:A"


def test_patient_day_source_codes_are_mutually_exclusive_and_count_gated():
    values = pd.DataFrame([
        _row(1, result_source="SYS_A"),
        _row(1, result_source="SYS_A", result_name=BP_EDV,
             value=125, unit="mL", fpk="another"),
        _row(2, result_source="SYS_A"),
        _row(3, result_source="SYS_B"),
        _row(4, result_source="SYS_A"),
        _row(4, result_source="SYS_B", fpk="other"),
        # This one is not a unitless MC CV source row.
        _row(5, result_source="SYS_B", unit="mL"),
        # Invalid numeric unit in mL-method result must be ignored.
        _row(6, result_name=BP_EDV, result_source="SYS_A",
             unit="cm"),
    ])
    report = summarize_volume_source_ids(values, small_cell=1)
    edv = report.loc[
        report.metric.eq("lv_end_diastolic_volume")
        & report.period.eq("2019-2023")
        & report.family.eq("mc_cv_unitless")
    ]
    got = edv.set_index("source_code").patient_days.to_dict()
    assert got == {
        "ID:SYS_A": "2",
        "ID:SYS_B": "1",
        "MULTIPLE_SOURCE_IDS": "1",
    }
    assert "person_id" not in report
    assert "echo_test_fpk" not in report
    assert "raw_value" not in report
    gated = summarize_volume_source_ids(values, small_cell=11)
    filtered = gated.loc[
        gated.metric.eq("lv_end_diastolic_volume")
        & gated.period.eq("2019-2023")
        & gated.family.eq("mc_cv_unitless")
    ]
    assert (filtered.patient_days == "suppressed").all()


def test_zero_observation_is_not_suppressed():
    report = summarize_volume_source_ids(
        pd.DataFrame(columns=[
            "person_id", "index_date", "echo_test_fpk",
            "result_name", "raw_value", "reported_unit",
            "test_source_id", "result_source_id",
        ])
    )
    assert (report.patient_days == "0").all()
    with pytest.raises(ValueError, match="small_cell"):
        summarize_volume_source_ids(
            pd.DataFrame(columns=[
                "person_id", "index_date", "echo_test_fpk",
                "result_name", "raw_value", "reported_unit",
                "test_source_id", "result_source_id",
            ]), small_cell=0
        )
