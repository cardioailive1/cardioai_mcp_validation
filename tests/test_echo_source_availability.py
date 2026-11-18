"""Tests of aggregate-only LVEDV source availability; synthetic counts, no PHI."""
import pandas as pd
import pytest

from cardioai_mcp.config import load_study
from echo_wm_training.source_availability import (
    source_availability_sql,
    summarise_source_availability,
)


def test_sql_scans_all_source_tte_not_only_cohort():
    cfg = load_study()["modality"]
    sql = source_availability_sql(cfg, start="2020-01-01", end="2020-01-31")
    assert "all_tte AS" in sql
    assert "cohort_tte AS" in sql
    assert "FROM all_tte t" in sql
    assert "echo_analytic" in sql
    assert "'all_source'" in sql
    assert "'cohort_index_day'" in sql
    assert "MC CV LV END DIASTOLIC VOLUME" in sql
    assert "LV_vol_BP_ed_2D_by_MOD" in sql
    assert "SAFE_CAST" in sql
    assert "BETWEEN 20 AND 500" in sql
    assert "SELECT scope, record_source, test_type" in sql
    assert "GROUP BY scope, record_source, test_type" in sql
    assert "RLIKE" not in sql
    assert "CREATE " not in sql.upper()
    assert "UPDATE " not in sql.upper()


def test_scope_and_source_aggregates_are_record_rates():
    rows = pd.DataFrame([
        ["all_source", "ACC", "TTE", 100, 60, 40],
        ["all_source", "ORD", "TTE", 50, 30, 20],
        ["cohort_index_day", "ACC", "TTE", 20, 12, 8],
        ["all_source", "ACC", "TTE OTHER", 25, 10, 15],
    ], columns=["scope", "record_source", "test_type", "test_records",
                "records_with_lvedv", "records_without_lvedv"])
    totals, safe_types = summarise_source_availability(rows, min_cell=11)
    acc_all = totals[(totals.scope == "all_source") &
                     (totals.record_source == "ACC")].iloc[0]
    assert int(acc_all.test_records) == 125
    assert int(acc_all.records_with_lvedv) == 70
    assert float(acc_all.available_pct) == 56.0
    assert set(safe_types.test_type) == {"TTE"}
    assert "TTE OTHER" not in safe_types.test_type.tolist()


def test_reject_missing_fields():
    with pytest.raises(ValueError, match="Missing columns"):
        summarise_source_availability(pd.DataFrame({"scope": ["x"]}))
