"""Synthetic checks for read-only provisional TTE measurement comparisons."""
import pandas as pd

from cardioai_mcp.config import load_study
from echo_wm_training.exam_comparison import (
    compare_provisional_pairs, comparison_summary, tte_measurement_rows_sql,
)


def test_pair_measurement_comparison_keeps_methods_and_groups_separate():
    cfg = load_study()["modality"]["echo"]["mcp"]["results"]
    tests = pd.DataFrame([
        {"person_id": 1, "index_date": "2020-01-02", "echo_test_fpk": fpk,
         "exam_datetime": "2020-01-02 " + dt,
         "accession_number": acc, "order_proc_id": order}
        for fpk, dt, acc, order in [
            ("a1", "09:00:00", "A", None),
            ("a2", "09:00:02", "A", None),
            ("o1", "09:00:30", None, "O"),
            ("far", "12:00:00", "B", None),
        ]
    ])
    def m(fpk, name, value, metric):
        test = tests[tests.echo_test_fpk == fpk].iloc[0].to_dict()
        return {**test, "result_name": name, "result_label": name,
                "result_value": value, "metric": metric}
    values = pd.DataFrame([
        m("a1", "LVEF visual", "60", "lvef"),
        m("a2", "LVEF biplane", "52", "lvef"),
        m("o1", "LVEF visual", "54", "lvef"),
        m("a2", "LV_vol_BP_ed_2D_by_MOD", "120", "lvedv"),
        m("o1", "LV_vol_4C_ed_2D_by_MOD", "124", "lvedv"),
        m("far", "LV_vol_BP_ed_2D_by_MOD", "400", "lvedv"),
    ])
    pairs = compare_provisional_pairs(tests, values, cfg)
    assert len(pairs) == 1
    assert pairs.acc_lvef.iloc[0] == 52
    assert pairs.ord_lvef.iloc[0] == 54
    assert pairs.lvef_abs_difference.iloc[0] == 2
    assert pairs.acc_lvedv.iloc[0] == 120
    assert pairs.ord_lvedv.iloc[0] == 124
    assert pairs.lvedv_abs_difference.iloc[0] == 4
    summary = comparison_summary(pairs)
    assert summary.both.tolist() == [1, 1]
    assert summary.median_abs_difference.tolist() == [2, 4]


def test_ambiguous_competitors_never_enter_pair_comparison():
    cfg = load_study()["modality"]["echo"]["mcp"]["results"]
    tests = pd.DataFrame([
        dict(person_id=1, index_date="2020-01-02", echo_test_fpk=fpk,
             exam_datetime="2020-01-02 " + tm, accession_number=acc, order_proc_id=order)
        for fpk, tm, acc, order in [
            ("a", "10:00:00", "A", None),
            ("b", "10:00:10", "B", None),
            ("o", "10:00:05", None, "O"),
        ]
    ])
    assert compare_provisional_pairs(tests, pd.DataFrame(), cfg).empty
    assert comparison_summary(pd.DataFrame()).empty


def test_measurement_query_scoped_to_tte_and_fpk():
    sql = tte_measurement_rows_sql(load_study()["modality"], start="2012-01-01", end="2023-12-31")
    assert "WITH tte AS" in sql
    assert "ECHO_TEST_FPK" in sql
    assert "LV_vol_BP_ed_2D_by_MOD" in sql
    assert "tte.echo_test_fpk" in sql
