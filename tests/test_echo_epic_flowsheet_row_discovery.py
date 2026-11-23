"""Synthetic metadata-only flowsheet vital row-name discovery tests."""
import pandas as pd
import pytest

from cardioai_mcp.config import load_study
from echo_wm_training.epic_flowsheet_row_discovery import (
    flowsheet_vital_row_dictionary_sql,
    normalize_vital_row_dictionary,
)


def test_selects_dimension_labels_only_without_fact_or_patient_data():
    query = flowsheet_vital_row_dictionary_sql(load_study()["modality"])
    assert "DIM_FLOWSHEET_ROW_NAME" in query
    assert "FLOWSHEET_ROW_DESCRIPTION" in query
    assert "FLOWSHEET_ROW_CODE" in query
    assert "REGEXP_CONTAINS" in query
    assert "height|weight|stature" in query
    assert "FACT_FLOWSHEETS" not in query
    assert "echo_analytic" not in query
    assert "PATIENT_DK" not in query
    assert "UPDATE " not in query and "INSERT " not in query


def test_deduplicated_result_contains_only_nonpatient_metadata():
    rows = pd.DataFrame([
        dict(flowsheet_row_name_dk="1", flowsheet_row_code="HT",
             flowsheet_row_description="Measured Height",
             patient_id="DO_NOT_PUBLISH"),
        dict(flowsheet_row_name_dk="1", flowsheet_row_code="HT",
             flowsheet_row_description="Measured Height",
             patient_id="DO_NOT_PUBLISH"),
        dict(flowsheet_row_name_dk="2", flowsheet_row_code=None,
             flowsheet_row_description="Weight",
             patient_id="DO_NOT_PUBLISH"),
    ])
    result = normalize_vital_row_dictionary(rows)
    assert list(result.columns) == [
        "flowsheet_row_name_dk", "flowsheet_row_code",
        "flowsheet_row_description",
    ]
    assert len(result) == 2
    assert result.flowsheet_row_description.tolist() == [
        "Measured Height", "Weight"
    ]
    assert result.flowsheet_row_code.iloc[1] == ""


def test_missing_expected_dimension_metadata_rejected():
    with pytest.raises(ValueError, match="Missing Mayo flowsheet"):
        normalize_vital_row_dictionary(
            pd.DataFrame({"flowsheet_row_name_dk": ["1"]})
        )
