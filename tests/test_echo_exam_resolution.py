"""Regression tests for conservative accession/order matching, using synthetic test records."""
import pandas as pd
import pytest

from cardioai_mcp.config import load_study
from echo_wm_training.exam_resolution import resolve_tte_groups, tte_record_rows_sql


def rec(person, fpk, when, acc=None, order=None):
    return {"person_id": person, "index_date": "2020-01-02",
            "echo_test_fpk": fpk, "exam_datetime": f"2020-01-02 {when}",
            "accession_number": acc, "order_proc_id": order}


def test_unique_mutual_pair_and_shared_accession_group():
    rows = pd.DataFrame([
        rec(1, "a1", "09:00:00", acc="A"),
        rec(1, "a2", "09:00:00", acc="A"),
        rec(1, "o1", "09:00:30", order="O"),
    ])
    out = resolve_tte_groups(rows)
    assert len(out) == 2
    assert set(out.status) == {"provisional_mutual_match"}
    assert out.test_records.sum() == 3
    assert set(out.closest_seconds) == {30.0}
    assert out.set_index("group_key").loc["ACC:A", "partner_key"] == "ORD:O"


def test_competing_matches_not_automatically_merged():
    rows = pd.DataFrame([
        rec(1, "a1", "09:00:00", acc="A"),
        rec(1, "a2", "09:00:20", acc="B"),
        rec(1, "o1", "09:00:10", order="O"),
    ])
    out = resolve_tte_groups(rows)
    assert (out.status == "ambiguous_competing_match").all()
    assert out.partner_key.isna().all()


def test_no_cross_patient_date_or_far_timestamp_matches():
    rows = pd.DataFrame([
        rec(1, "a", "09:00:00", acc="A"),
        rec(1, "o", "10:01:00", order="O"),
        rec(2, "b", "09:00:05", acc="A"),
        rec(2, "p", "09:00:10", order="P"),
    ])
    out = resolve_tte_groups(rows)
    assert set(out[out.person_id == 1].status) == {"unmatched_acc", "unmatched_ord"}
    assert set(out[out.person_id == 2].status) == {"provisional_mutual_match"}


def test_missing_keys_and_no_transitive_match():
    rows = pd.DataFrame([
        rec(1, "no-key", "09:00:00"),
        rec(1, "a", "09:00:00", acc="A"),
        rec(1, "o", "09:00:30", order="O"),
        rec(1, "other", "09:00:45", order="P"),
    ])
    out = resolve_tte_groups(rows)
    assert out[out.key_type == "TEST"].status.iloc[0] == "missing_group_identifier"
    assert (out[out.key_type != "TEST"].status == "ambiguous_competing_match").all()


def test_config_and_sql_restrict_to_index_date_and_tte():
    sql = tte_record_rows_sql(load_study()["modality"], start="2012-01-01", end="2023-12-31")
    assert "{{work}}.echo_analytic" in sql
    assert "ORDER_PROC_ID" in sql and "ACCESSION_NUMBER" in sql
    assert "ECHO_TEST_DESCRIPTION" in sql
    assert "CAST(a.index_date AS DATE)" in sql
    with pytest.raises(ValueError, match="start"):
        tte_record_rows_sql(load_study()["modality"], start="2024-01-01", end="2012-01-01")
    with pytest.raises(ValueError, match="max_seconds"):
        resolve_tte_groups(pd.DataFrame(), max_seconds=-1)
