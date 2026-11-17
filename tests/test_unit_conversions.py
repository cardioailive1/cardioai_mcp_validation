"""Regression tests for optional unit-conversion match criteria."""
import pytest

from cardioai_mcp.config import Element
from cardioai_mcp.extractors import _value_expr


def _element(name="weight"):
    return Element(name=name, domain="demographics", description="", required=True,
                   expected_completeness=0.95, src={"type": "measurement", "set": name})


def test_ounce_source_only_and_pound_conversion():
    study = {"unit_conversions": {
        "weight": [
            {"unit_source_like": ["oz"], "factor": 0.028349523125},
            {"unit_concept_id": 8739, "unit_source_like": ["lb%"], "factor": 0.45359237},
        ]
    }}
    sql = _value_expr(_element(), study)
    assert "LOWER(m.unit_source_value) LIKE 'oz'" in sql
    assert "m.unit_concept_id = 8739" in sql
    assert "0.028349523125" in sql
    assert "0.45359237" in sql
    assert sql.index("LIKE 'oz'") < sql.index("m.unit_concept_id = 8739")


def test_conversion_must_have_matching_condition():
    study = {"unit_conversions": {"weight": [{"factor": 100.0}]}}
    with pytest.raises(ValueError, match="requires unit_concept_id or unit_source_like"):
        _value_expr(_element(), study)
