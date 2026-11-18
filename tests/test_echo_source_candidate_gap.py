"""PHI-free synthetic tests of source LVEDV vs candidate exclusions."""
import pandas as pd
import pytest

from cardioai_mcp.config import load_study
from echo_wm_training.cohort_type_availability import cohort_type_rows_sql
from echo_wm_training.source_candidate_gap import diagnose_source_not_candidate


def test_native_query_exposes_source_name_provenance():
    sql = cohort_type_rows_sql(
        load_study()["modality"], start="2020-01-01", end="2020-01-03"
    )
    assert "AS has_priority_lvedv" in sql
    assert "AS has_mayo_named_lvedv" in sql
    assert "COALESCE(v.has_priority_lvedv, FALSE) AS has_priority_lvedv" in sql


def test_disjoint_root_causes_and_patient_day_count():
    cohort = pd.DataFrame([
        {"person_id": i, "index_date": "2020-02-03"} for i in range(1, 9)
    ])
    records = [
        # mayo only -> unmapped by priority method extraction
        (1, "test1", 1, 1, 0),
        # separate priority and Mayo in same study: preserve priority cause
        (2, "test2", 1, 1, 1),
        # value exists but corresponding group has LVEDV conflict
        (3, "test3", 1, 0, 1),
        (4, "test4", 1, 0, 1),
        (5, "test5", 1, 0, 1),
        (6, "test6", 1, 0, 1),
        (7, "test7", 1, 0, 1),
        (8, "test8", 0, 0, 0),  # not targeted
    ]
    source = pd.DataFrame([
        {
            "person_id": pid, "index_date": "2020-02-03",
            "echo_test_fpk": fpk, "test_type": "Adult TTE",
            "record_source": "ACC",
            "has_mapped_lvedv": bool(mapped),
            "has_mayo_lvedv": bool(mayo),
            "has_priority_lvedv": bool(priority),
        }
        for pid, fpk, mapped, mayo, priority in records
    ])
    statuses = {
        1: ("single_exam_candidate", False, False),
        2: ("ambiguous_multiple_exams", False, False),
        3: ("conflicting_measurements", True, False),
        4: ("unresolved_identifier", False, False),
        5: ("ambiguous_multiple_exams", False, False),
        6: ("unresolved_method_disagreement", False, True),
        7: ("single_exam_candidate", False, False),
    }
    candidates = pd.DataFrame([
        {"person_id": pid, "index_date": "2020-02-03", "lvedv": None,
         "candidate_status": status, "lvedv_conflict": conflict,
         "lvef_method_disagreement": ef_disagreement}
        for pid, (status, conflict, ef_disagreement) in statuses.items()
    ])
    private, reasons, statuses_summary = diagnose_source_not_candidate(
        cohort, source, candidates
    )
    assert len(private) == 7
    assert reasons.patient_days.sum() == 7
    by_id = dict(zip(private.person_id, private.reason))
    assert by_id == {
        1: "mayo_result_name_only",
        2: "multiple_exam_candidates",
        3: "lvedv_conflict",
        4: "unresolved_identifier",
        5: "multiple_exam_candidates",
        6: "ef_method_disagreement",
        7: "priority_lvedv_not_carried_to_candidate",
    }
    assert int(statuses_summary.candidate_groups.sum()) == 7


def test_no_target_and_multiple_rows_per_source_test():
    cohort = pd.DataFrame([{"person_id": 1, "index_date": "2020-02-03"}])
    rows = pd.DataFrame([
        dict(person_id=1, index_date="2020-02-03", echo_test_fpk="t1",
             test_type="Adult TTE", record_source=src,
             has_mapped_lvedv=True, has_priority_lvedv=False,
             has_mayo_lvedv=True) for src in ("ACC", "ACC")
    ])
    candidates = pd.DataFrame([
        {"person_id": 1, "index_date": "2020-02-03",
         "candidate_status": "single_exam_candidate", "lvedv": 101.0}
    ])
    private, reasons, statuses = diagnose_source_not_candidate(
        cohort, rows, candidates
    )
    assert private.empty
    assert reasons.empty
    assert statuses.empty


def test_missing_source_provenance_rejected():
    cohort = pd.DataFrame([{"person_id": 1, "index_date": "2020-02-03"}])
    source = pd.DataFrame([{
        "person_id": 1, "index_date": "2020-02-03", "echo_test_fpk": "t",
        "test_type": "Adult TTE", "record_source": "ACC",
        "has_mapped_lvedv": True,
    }])
    candidates = pd.DataFrame([{
        "person_id": 1, "index_date": "2020-02-03",
        "candidate_status": "single_exam_candidate", "lvedv": None
    }])
    with pytest.raises(ValueError, match="missing columns"):
        diagnose_source_not_candidate(cohort, source, candidates)



def test_numeric_lvedv_in_excluded_candidate_status_is_diagnosed_not_rejected():
    # A candidate can retain a numeric LVEDV and still fail provisional
    # selection because the examination is ambiguous. This was observed in
    # live MCP data and must not raise ValueError.
    cohort = pd.DataFrame([
        {"person_id": 1, "index_date": "2020-02-03"},
        {"person_id": 2, "index_date": "2020-02-03"},
    ])
    source = pd.DataFrame([
        dict(person_id=pid, index_date="2020-02-03", echo_test_fpk=f"test{pid}",
             test_type="Adult TTE", record_source="ACC",
             has_mapped_lvedv=True, has_priority_lvedv=True,
             has_mayo_lvedv=False)
        for pid in (1, 2)
    ])
    candidates = pd.DataFrame([
        dict(person_id=1, index_date="2020-02-03",
             candidate_status="ambiguous_multiple_exams",
             lvedv=125.0),
        dict(person_id=2, index_date="2020-02-03",
             candidate_status="unresolved_identifier",
             lvedv=140.0),
    ])
    private, reasons, statuses = diagnose_source_not_candidate(
        cohort, source, candidates
    )
    assert len(private) == 2
    assert private.candidate_has_lvedv.all()
    assert dict(zip(private.person_id, private.reason)) == {
        1: "multiple_exam_candidates",
        2: "unresolved_identifier",
    }
    assert int(reasons.patient_days.sum()) == 2
    assert int(statuses.candidate_groups.sum()) == 2
