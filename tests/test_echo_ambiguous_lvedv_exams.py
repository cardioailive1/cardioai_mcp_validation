"""Synthetic tests: ambiguous source exams are not auto-approved."""
import pandas as pd

from echo_wm_training.ambiguous_lvedv_exams import classify_ambiguous_exams


def _rows():
    # Person 1: matched ACC/ORD (same minute) plus distinct ACC exam.
    # Person 2: two unmatched ACC examinations.
    return pd.DataFrame([
        (1, "a1", "2020-02-03 10:00:00", "ACC1", None),
        (1, "o1", "2020-02-03 10:00:20", None, "ORD1"),
        (1, "a2", "2020-02-03 14:00:00", "ACC2", None),
        (2, "a3", "2020-02-03 10:00:00", "ACC3", None),
        (2, "a4", "2020-02-03 12:00:00", "ACC4", None),
    ], columns=["person_id", "echo_test_fpk", "exam_datetime",
                "accession_number", "order_proc_id"]).assign(
                    index_date="2020-02-03", test_description="Adult TTE")


def _excluded():
    return pd.DataFrame([
        dict(person_id=1, index_date="2020-02-03",
             reason="multiple_exam_candidates"),
        dict(person_id=2, index_date="2020-02-03",
             reason="multiple_exam_candidates"),
        dict(person_id=3, index_date="2020-02-03",
             reason="mayo_result_name_only"),
    ])


def _candidates():
    return pd.DataFrame([
        dict(person_id=1, index_date="2020-02-03", group_key="ACC:ACC1",
             candidate_status="ambiguous_multiple_exams", lvedv=120.0),
        dict(person_id=1, index_date="2020-02-03", group_key="ACC:ACC2",
             candidate_status="ambiguous_multiple_exams", lvedv=130.0),
        dict(person_id=2, index_date="2020-02-03", group_key="ACC:ACC3",
             candidate_status="ambiguous_multiple_exams", lvedv=None),
        dict(person_id=2, index_date="2020-02-03", group_key="ACC:ACC4",
             candidate_status="ambiguous_multiple_exams", lvedv=140.0),
    ])


def test_pairing_does_not_auto_collapse_distinct_exam():
    private, summary, statuses = classify_ambiguous_exams(
        _excluded(), _candidates(), _rows()
    )
    assert len(private) == 2
    one = private.set_index("person_id").loc[1]
    assert int(one.raw_source_records) == 3
    assert int(one.identifier_groups) == 3
    assert int(one.provisional_pairs) == 1
    assert int(one.provisional_groups) == 2
    assert int(one.candidate_groups_with_lvedv) == 2
    assert one.review_category == "multiple_groups_after_provisional_pairing"
    two = private.set_index("person_id").loc[2]
    assert int(two.provisional_pairs) == 0
    assert int(two.provisional_groups) == 2
    assert two.review_category == "multiple_unmatched_identifier_groups"
    assert summary.patient_days.sum() == 2
    assert statuses.groups.sum() == 5


def test_no_ambiguous_exclusions():
    other = _excluded().query("reason == 'mayo_result_name_only'")
    private, summary, statuses = classify_ambiguous_exams(
        other, _candidates(), _rows()
    )
    assert private.empty
    assert summary.empty
    assert statuses.empty
