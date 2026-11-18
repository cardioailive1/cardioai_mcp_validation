"""Synthetic next-day LVEDV validation; no BigQuery or PHI."""
import pandas as pd

from cardioai_mcp.config import load_study
from echo_wm_training.exam_resolution import tte_record_rows_sql
from echo_wm_training.next_day_lvedv import (
    next_day_lvedv_rows_sql, summarize_next_day_lvedv,
)


def test_next_day_sql_keeps_default_same_day_query_unchanged():
    config = load_study()["modality"]
    day0 = tte_record_rows_sql(config, start="2020-01-01", end="2020-01-02")
    day1 = tte_record_rows_sql(config, start="2020-01-01", end="2020-01-02", day_offset=1)
    assert "CAST(t.ECHO_PERFORMED_DTM AS DATE) = CAST(a.index_date AS DATE)" in day0
    assert "DATEDIFF(CAST(t.ECHO_PERFORMED_DTM AS DATE), CAST(a.index_date AS DATE)) = 1" in day1
    assert "DATEDIFF" not in day0
    query = next_day_lvedv_rows_sql(config, start="2020-01-01", end="2020-01-02")
    assert "MC CV LV END DIASTOLIC VOLUME" in query
    assert "LEFT JOIN volume_rows" in query
    assert "DATEDIFF(CAST(t.ECHO_PERFORMED_DTM AS DATE), CAST(a.index_date AS DATE)) = 1" in query


def test_next_day_counts_only_new_unambiguous_single_test():
    date = "2020-01-02"
    cohort = pd.DataFrame([{"person_id": i, "index_date": date} for i in range(1, 7)])
    candidates = pd.DataFrame([
        {"person_id": 1, "index_date": date, "candidate_status": "single_exam_candidate", "lvedv": 100},
        {"person_id": 2, "index_date": date, "candidate_status": "unresolved_method_disagreement", "lvedv": 150},
        {"person_id": 3, "index_date": date, "candidate_status": "single_exam_candidate", "lvedv": None},
        {"person_id": 4, "index_date": date, "candidate_status": "single_exam_candidate", "lvedv": None},
        {"person_id": 5, "index_date": date, "candidate_status": "single_exam_candidate", "lvedv": None},
        {"person_id": 6, "index_date": date, "candidate_status": "single_exam_candidate", "lvedv": None},
    ])
    def row(pid, test, vol, name="LV_vol_BP_ed_2D_by_MOD"):
        return {"person_id": pid, "index_date": date, "echo_test_fpk": test,
                "result_name": name, "value_raw": vol}
    next_day = pd.DataFrame([
        row(1, "prior", "110"),  # already covered
        row(3, "new", "120"),  # genuine new candidate
        row(4, "a", "130"), row(4, "b", "140"),  # two tests: not safe
        row(5, "x", "100"), row(5, "x", "115"),  # same-name conflict
        row(6, "no-volume", None, None),  # TTE with no valid volume
    ])
    details, metrics = summarize_next_day_lvedv(next_day, cohort, candidates)
    assert len(details) == 6
    assert metrics["same_day_potential_covered"] == 2
    assert metrics["same_day_potential_missing"] == 4
    assert metrics["next_day_any_valid_lvedv_while_missing"] == 3
    assert metrics["next_day_single_test_valid_no_same_name_conflict"] == 1
    assert metrics["next_day_multi_test_with_volume_while_missing"] == 1
    assert metrics["next_day_name_conflict_while_missing"] == 1
    assert metrics["potential_total_if_next_day_approved"] == 3


def test_recovered_mapping_does_not_double_count():
    cohort = pd.DataFrame([
        {"person_id": 1, "index_date": "2020-01-02"},
        {"person_id": 2, "index_date": "2020-01-02"},
    ])
    candidates = cohort.assign(candidate_status="single_exam_candidate", lvedv=[100, None])
    recovered = pd.DataFrame([{"person_id": 2, "index_date": "2020-01-02"}])
    _, metrics = summarize_next_day_lvedv(pd.DataFrame(), cohort, candidates, recovered)
    assert metrics["same_day_potential_covered"] == 2
    assert metrics["potential_total_if_next_day_approved"] == 2
