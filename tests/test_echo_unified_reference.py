"""Synthetic regression tests for unified read-only Echo candidate comparisons."""
import pandas as pd

from cardioai_mcp.config import load_study
from echo_wm_training.unified_reference import (
    baseline_lvef_rows_sql, unified_exam_candidates, compare_to_baseline,
)


def test_unified_keeps_unmatched_acc_ef_and_matched_order_measurements():
    cfg = load_study()["modality"]["echo"]["mcp"]["results"]
    def t(pid, fpk, when, acc=None, order=None):
        return dict(person_id=pid, index_date="2020-01-02",
                    echo_test_fpk=fpk, exam_datetime="2020-01-02 " + when,
                    accession_number=acc, order_proc_id=order)
    tests = pd.DataFrame([
        t(1, "acc1", "09:00:00", acc="A"),
        t(1, "ord1", "09:00:30", order="O"),
        t(2, "acc2", "10:00:00", acc="B"),
        t(3, "acc3", "11:00:00", acc="C"),
        t(3, "acc4", "12:00:00", acc="D"),
    ])
    def m(test, name, value, metric):
        return dict(**test, result_name=name, result_label=name,
                    result_value=value, metric=metric)
    measurements = pd.DataFrame([
        m(tests.iloc[1].to_dict(), "LVEF biplane", "52", "lvef"),
        m(tests.iloc[1].to_dict(), "LV_vol_BP_ed_2D_by_MOD", "120", "lvedv"),
        m(tests.iloc[2].to_dict(), "MC CV ECHO EJECTION FRACTION", "48", "lvef"),
        m(tests.iloc[3].to_dict(), "MC CV ECHO EJECTION FRACTION", "55", "lvef"),
        m(tests.iloc[4].to_dict(), "MC CV ECHO EJECTION FRACTION", "70", "lvef"),
    ])
    candidates = unified_exam_candidates(tests, measurements, cfg)
    matched = candidates[candidates.person_id == 1].iloc[0]
    assert matched.record_source == "ACC+ORD"
    assert matched.lvef == 52 and matched.lvedv == 120
    assert matched.lvef_source == "ORD" and matched.lvedv_source == "ORD"
    acc = candidates[candidates.person_id == 2].iloc[0]
    assert acc.record_source == "ACC" and acc.lvef == 48
    assert acc.lvef_source == "ACC"
    assert acc.candidate_status == "single_exam_candidate"
    multi = candidates[candidates.person_id == 3]
    assert len(multi) == 2 and set(multi.candidate_status) == {"ambiguous_multiple_exams"}
    baseline = pd.DataFrame([
        dict(person_id=1, index_date="2020-01-02", lvef_date="2020-01-02", baseline_lvef=52),
        dict(person_id=2, index_date="2020-01-02", lvef_date="2020-01-03", baseline_lvef=50),
        dict(person_id=3, index_date="2020-01-02", lvef_date="2020-01-02", baseline_lvef=55),
    ])
    detail, summary = compare_to_baseline(candidates, baseline)
    assert len(detail) == 2
    row = summary[summary.scope == "single_exam_candidates"].iloc[0]
    assert row.patients == 2 and row.lvef_available == 2
    assert row.equal_lvef == 1 and row.changed_lvef == 1
    assert row.median_lvef_abs_difference == 1


def test_conflicting_values_remain_unresolved():
    cfg = load_study()["modality"]["echo"]["mcp"]["results"]
    tests = pd.DataFrame([
        dict(person_id=1, index_date="2020-01-02", echo_test_fpk="a",
             exam_datetime="2020-01-02 09:00:00", accession_number="A", order_proc_id=None),
        dict(person_id=1, index_date="2020-01-02", echo_test_fpk="o",
             exam_datetime="2020-01-02 09:00:30", accession_number=None, order_proc_id="O"),
    ])
    measurements = pd.DataFrame([
        {**tests.iloc[0].to_dict(), "result_name": "MC CV ECHO EJECTION FRACTION",
         "result_label": "MC CV ECHO EJECTION FRACTION",
         "result_value": "55", "metric": "lvef"},
        {**tests.iloc[1].to_dict(), "result_name": "LVEF visual",
         "result_label": "LVEF visual",
         "result_value": "65", "metric": "lvef"},
    ])
    candidates = unified_exam_candidates(tests, measurements, cfg)
    assert len(candidates) == 1
    assert candidates.lvef.isna().all()
    assert candidates.candidate_status.iloc[0] == "conflicting_measurements"


def test_read_only_baseline_sql_uses_existing_echo_lvef():
    sql = baseline_lvef_rows_sql(start="2012-01-01", end="2023-12-31")
    assert "{{work}}.echo_lvef" in sql
    assert "DATEDIFF" in sql
    assert "BETWEEN 0 AND 1" in sql
    assert "WRITE" not in sql.upper()


def test_modified_quinones_is_not_biplane_and_large_ef_conflict_is_unresolved():
    """Regression: 59/13/63/63 was previously reduced to visual EF=13."""
    import re
    from echo_wm_training.exam_comparison import _measurement_groups
    cfg = load_study()["modality"]["echo"]["mcp"]["results"]
    methods = [(x["method"], re.compile(x["regex"], re.I))
               for x in cfg["method_priority"]]
    quin_label = "LV_EF_2D_by_mod_Quin | Left Ventricular EF Modified Quinones by 2-D"
    hits = [name for name, rx in methods if rx.search(quin_label)]
    assert hits[0] == "modified_quinones"
    assert "biplane" not in hits
    base = dict(person_id=11, index_date="2020-01-02",
                echo_test_fpk="a", exam_datetime="2020-01-02 10:00:00",
                accession_number="A", order_proc_id=None)
    labels = [
        (quin_label, "59"),
        ("LV_EF_2D_by_vis_est | Left Ventricular EF Visual Estimate by 2-D", "13"),
        ("LV_EF_2D_by_mod_Quin_Acf | Left Ventricular EF Modified Quinones with Apical Correction Factor by 2-D", "63"),
        ("MC CV ECHO EJECTION FRACTION | Ejection Fraction", "63"),
    ]
    rows = pd.DataFrame([
        {**base, "result_label": label, "result_name": label,
         "result_value": value, "metric": "lvef"}
        for label, value in labels
    ])
    selected = _measurement_groups(rows, cfg)
    assert selected.lvef.iloc[0] == 61.0
    assert selected.lvef_spread_pp.iloc[0] == 50.0
    assert bool(selected.lvef_method_disagreement.iloc[0])
    candidates = unified_exam_candidates(pd.DataFrame([base]), rows, cfg)
    assert candidates.candidate_status.iloc[0] == "unresolved_method_disagreement"
    assert pd.isna(candidates.lvef.iloc[0])


def test_consistent_modified_quinones_candidate_is_retained():
    cfg = load_study()["modality"]["echo"]["mcp"]["results"]
    base = dict(person_id=12, index_date="2020-01-02",
                echo_test_fpk="b", exam_datetime="2020-01-02 10:00:00",
                accession_number="B", order_proc_id=None)
    values = pd.DataFrame([
        {**base, "result_label": label, "result_name": label,
         "result_value": value, "metric": "lvef"}
        for label, value in [
            ("LV_EF_2D_by_mod_Quin", "59"),
            ("MC CV ECHO EJECTION FRACTION", "60"),
        ]
    ])
    out = unified_exam_candidates(pd.DataFrame([base]), values, cfg)
    assert out.candidate_status.iloc[0] == "single_exam_candidate"
    assert out.lvef.iloc[0] == 59
