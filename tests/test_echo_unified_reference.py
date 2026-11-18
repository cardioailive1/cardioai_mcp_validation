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


def test_same_day_strain_ef_cannot_override_conventional_tte():
    """MCP discovery: conventional contrast TTE 63%; separate TTE Strain 59/63/13%.

    A patient/date-level classifier could select the visual 13%. Eligible test
    identity must be checked *before* EF method selection.
    """
    from echo_wm_training.exam_resolution import eligible_tte_records
    cfg = load_study()["modality"]["echo"]["mcp"]["results"]
    tests = pd.DataFrame([
        dict(person_id=99, index_date="2021-11-30",
             echo_test_fpk="conventional", exam_datetime="2021-11-30 12:54:00",
             accession_number="CONV", order_proc_id=None,
             test_description="(TTE) 2D ECHO DOPPLER COLOR AND CONTRAST"),
        dict(person_id=99, index_date="2021-11-30",
             echo_test_fpk="strain", exam_datetime="2021-11-30 12:54:00",
             accession_number="STRAIN", order_proc_id=None,
             test_description="TTE Strain"),
    ])
    assert eligible_tte_records(tests).echo_test_fpk.tolist() == ["conventional"]
    def reading(record, name, value):
        return {**record, "result_name": name,
                "result_label": name, "result_value": value, "metric": "lvef"}
    rows = pd.DataFrame([
        reading(tests.iloc[0].to_dict(), "MC CV ECHO EJECTION FRACTION", "63"),
        reading(tests.iloc[1].to_dict(), "LV_EF_2D_by_mod_Quin", "59"),
        reading(tests.iloc[1].to_dict(), "LV_EF_2D_by_mod_Quin_Acf", "63"),
        reading(tests.iloc[1].to_dict(), "LV_EF_2D_by_vis_est", "13"),
    ])
    # Deliberately provide *both* test rows and measurement rows to exercise
    # the independent safety guard even if a caller bypasses query-side filters.
    candidate = unified_exam_candidates(tests, rows, cfg)
    assert len(candidate) == 1
    assert candidate.lvef.iloc[0] == 63.0
    assert candidate.candidate_status.iloc[0] == "single_exam_candidate"
    assert not bool(candidate.lvef_method_disagreement.iloc[0])
    assert candidate.record_source.iloc[0] == "ACC"


def test_strain_only_has_no_eligible_conventional_tte():
    cfg = load_study()["modality"]["echo"]["mcp"]["results"]
    strain = dict(person_id=99, index_date="2021-11-30",
                  echo_test_fpk="strain", exam_datetime="2021-11-30 12:54:00",
                  accession_number="S", order_proc_id=None,
                  test_description="TTE Strain")
    rows = pd.DataFrame([{**strain, "result_name": "LV_EF_2D_by_vis_est",
                          "result_label": "LV_EF_2D_by_vis_est",
                          "result_value": "13", "metric": "lvef"}])
    assert unified_exam_candidates(pd.DataFrame([strain]), rows, cfg).empty


def test_disagreement_categories_split_within_and_between_methods():
    from echo_wm_training.exam_comparison import _measurement_groups
    cfg = load_study()["modality"]["echo"]["mcp"]["results"]
    def measurements(pid, values):
        base = dict(person_id=pid, index_date="2020-01-02",
                    echo_test_fpk=str(pid), exam_datetime="2020-01-02 10:00:00",
                    accession_number=str(pid), order_proc_id=None)
        return base, [{**base, "result_label": method,
                       "result_name": method, "result_value": str(value),
                       "metric": "lvef"} for method, value in values]
    data = []
    tests = []
    samples = [
        (1, [("LVEF visual", 40), ("LVEF visual", 60)], "within_method"),
        (2, [("LVEF visual", 40), ("LVEF biplane", 60)], "between_method"),
        (3, [("LVEF visual", 20), ("LVEF visual", 40),
             ("LVEF biplane", 60)], "both"),
        (4, [("LVEF visual", 50), ("LVEF biplane", 55)], "none"),
    ]
    for pid, values, _ in samples:
        base, rows = measurements(pid, values)
        tests.append(base)
        data.extend(rows)
    groups = _measurement_groups(pd.DataFrame(data), cfg)
    candidates = unified_exam_candidates(pd.DataFrame(tests), pd.DataFrame(data), cfg)
    for pid, _, expected in samples:
        actual = groups.loc[groups.person_id == pid, "lvef_disagreement_category"].iloc[0]
        assert actual == expected
        candidate = candidates.loc[candidates.person_id == pid].iloc[0]
        assert candidate.lvef_disagreement_category == expected
        if expected == "none":
            assert candidate.candidate_status == "single_exam_candidate"
            assert pd.notna(candidate.lvef)
        else:
            assert candidate.candidate_status == "unresolved_method_disagreement"
            assert pd.isna(candidate.lvef)


def test_same_method_spread_not_inflated_by_different_method_labels():
    from echo_wm_training.exam_comparison import _measurement_groups
    cfg = load_study()["modality"]["echo"]["mcp"]["results"]
    rows = pd.DataFrame([
        dict(person_id=1, index_date="2020-01-02", echo_test_fpk="1",
             accession_number="A", order_proc_id=None,
             metric="lvef", result_name=label,
             result_label=label, result_value=value)
        for label, value in [
            ("LV_EF_2D_by_mod_Quin", "59"),
            ("LV_EF_2D_by_mod_Quin_Acf", "63"),
            ("LVEF visual", "13"),
        ]
    ])
    g = _measurement_groups(rows, cfg).iloc[0]
    assert g.lvef_within_method_spread_pp == 4
    assert g.lvef_between_method_spread_pp == 48
    assert g.lvef_disagreement_category == "between_method"
