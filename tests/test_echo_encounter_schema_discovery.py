"""Synthetic metadata-only tests; no Mayo patient or clinical records."""
import pandas as pd
import pytest

from echo_wm_training.encounter_schema_discovery import (
    encounter_schema_sql, _normalize_metadata,
)


def test_query_uses_information_schema_not_patient_rows():
    sql = encounter_schema_sql(
        "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
    )
    assert "INFORMATION_SCHEMA.COLUMNS" in sql
    assert "table_name, column_name, data_type" in sql
    assert "FLOWSHEET" in sql
    assert "CLINICAL_DOCUMENT" in sql
    assert "ENCOUNTER" in sql
    assert "ECHO" in sql
    assert "JOIN " not in sql
    assert "COUNT(" not in sql


def test_only_configured_project_dataset_identifier():
    for invalid in (
        "other-dataset",
        "`p`.ds; DROP TABLE x",
        "`p`.`ds`",
        "",
    ):
        with pytest.raises(ValueError, match="project.dataset"):
            encounter_schema_sql(invalid)


def test_metadata_processing_keeps_only_column_names_and_types():
    data = pd.DataFrame([
        dict(table_name="FACT_ECHO_TEST", column_name="ENCOUNTER_NUMBER",
             data_type="STRING", ordinal_position=5),
        dict(table_name="FACT_ECHO_TEST", column_name="ECHO_TEST_FPK",
             data_type="STRING", ordinal_position=1),
    ])
    out = _normalize_metadata(data)
    assert out.column_name.tolist() == ["ECHO_TEST_FPK", "ENCOUNTER_NUMBER"]
    assert set(out.columns) == {
        "table_name", "column_name", "data_type", "ordinal_position"
    }
    with pytest.raises(ValueError, match="missing columns"):
        _normalize_metadata(data.drop(columns="ordinal_position"))
