"""No-PHI synthetic regression tests for staged Mayo echo extraction."""
import pandas as pd

from cardioai_mcp.config import load_cohort
from echo_wm_training.cohort_extraction_preview import (
    compare_echo_completeness,
    echo_source_note_rows_sql,
    echo_volume_rows_sql,
    preview_echo_extraction,
)
from cardioai_mcp.config import load_study


def _tests(rows):
    return pd.DataFrame(rows, columns=[
        "person_id", "index_date", "echo_test_fpk", "exam_datetime",
        "accession_number", "order_proc_id", "test_description",
    ])


def _vol(rows):
    return pd.DataFrame(rows, columns=[
        "person_id", "index_date", "echo_test_fpk", "result_name",
        "raw_value", "reported_unit",
    ])


def _notes(rows):
    return pd.DataFrame(rows, columns=[
        "person_id", "index_date", "echo_test_fpk", "echo_note",
    ])


def _cohort(ids=(1, 2, 3, 4)):
    return pd.DataFrame([dict(person_id=i, index_date="2020-01-01")
                         for i in ids])


def test_sql_reuses_cohort_tte_filter_and_source_test_fpk():
    cfg = load_study()["modality"]
    volumes = echo_volume_rows_sql(cfg, start="2020-01-01", end="2020-01-02")
    notes = echo_source_note_rows_sql(cfg, start="2020-01-01", end="2020-01-02")
    assert "echo_analytic" in volumes and "echo_analytic" in notes
    for expected in ("ECHO_TEST_FPK", "FACT_ECHO_RESULTS",
                     "ECHO_RESULTS_VALUE_UNIT_OF_MEASURE",
                     "LV_vol_BP_ed_2D_by_MOD",
                     "MC CV LV END SYSTOLIC VOLUME"):
        assert expected in volumes
    assert "FACT_ECHO_TEST" in notes and "ECHO_NOTE" in notes
    assert "stress|strain|outside|review|fetal" in notes
    for sql in (volumes, notes):
        for op in ("CREATE TABLE", "INSERT INTO", "UPDATE ", "DELETE FROM"):
            assert op not in sql.upper()


def test_selects_only_one_unmatched_identifier_group_with_consistent_ml():
    day = "2020-01-01"
    tests = _tests([
        (1, day, "e1", day + " 11:00:00", "A1", None, "Adult TTE"),
        (1, day, "e2", day + " 11:00:01", "A1", None, "Adult TTE"),
        (2, day, "e3", day + " 10:00:00", "A2", None, "Adult TTE"),
        (2, day, "e4", day + " 13:00:00", "A3", None, "Adult TTE"),
        (3, day, "e5", day + " 10:00:00", None, None, "Adult TTE"),
        (4, day, "e6", day + " 10:00:00", "A4", None, "Adult TTE"),
        (4, day, "e7", day + " 10:00:05", None, "O4", "Adult TTE"),
    ])
    values = _vol([
        (1, day, "e1", "LV_vol_BP_ed_2D_by_MOD", "120", "mL"),
        (1, day, "e2", "LV_vol_BP_ed_2D_by_MOD", "120", "mL"),
        (1, day, "e1", "LV_vol_BP_es_2D_by_MOD", "44", "mL"),
        (2, day, "e3", "LV_vol_BP_ed_2D_by_MOD", "150", "mL"),
        (3, day, "e5", "LV_vol_BP_ed_2D_by_MOD", "150", "mL"),
        (4, day, "e6", "LV_vol_BP_ed_2D_by_MOD", "150", "mL"),
    ])
    notes = _notes([
        (1, day, "e1", "normal wall motion"),
        (1, day, "e2", "normal wall motion"),
        (2, day, "e3", "multiple exams"),
        (3, day, "e5", "no ID"),
        (4, day, "e6", "possible duplicate exam"),
    ])
    result = preview_echo_extraction(_cohort(), tests, values, notes).set_index("person_id")
    assert result.loc[1, "exam_status"] == "single_identifier_group"
    assert result.loc[1, "lv_end_diastolic_volume"] == 120.
    assert result.loc[1, "lv_end_systolic_volume"] == 44.
    assert result.loc[1, "echo_report_status"] == "unique_source_report"
    assert result.loc[2, "exam_status"] == "multiple_exams"
    assert pd.isna(result.loc[2, "lv_end_diastolic_volume"])
    assert result.loc[3, "exam_status"] == "unresolved_identifier"
    assert result.loc[4, "exam_status"] == "provisional_pair_requires_review"
    assert pd.isna(result.loc[4, "echo_report"])


def test_rejects_disagreeing_values_unknown_units_and_conflicting_reports():
    d = "2020-01-01"
    tests = _tests([
        (1, d, "e1", d + " 10:00:00", "A1", None, "Adult TTE"),
        (2, d, "e2", d + " 10:00:00", "A2", None, "Adult TTE"),
        (3, d, "e3", d + " 10:00:00", "A3", None, "Adult TTE"),
        (3, d, "e4", d + " 10:01:00", "A3", None, "Adult TTE"),
    ])
    vol = _vol([
        (1, d, "e1", "LV_vol_BP_ed_2D_by_MOD", "120", "mL"),
        (1, d, "e1", "LV_vol_4C_ed_2D_by_MOD", "130", "mL"),
        (2, d, "e2", "LV_vol_BP_ed_2D_by_MOD", "120", None),
        (3, d, "e3", "LV_vol_BP_ed_2D_by_MOD", "120", "mL"),
    ])
    notes = _notes([
        (3, d, "e3", "note one"),
        (3, d, "e4", "note two"),
    ])
    got = preview_echo_extraction(
        _cohort((1, 2, 3)), tests, vol, notes
    ).set_index("person_id")
    assert got.loc[1, "lv_end_diastolic_volume_status"] == "disagreeing_measurements_review"
    assert pd.isna(got.loc[1, "lv_end_diastolic_volume"])
    assert got.loc[2, "lv_end_diastolic_volume_status"] == "unknown_or_non_absolute_unit"
    assert got.loc[3, "echo_report_status"] == "conflicting_source_reports"


def test_completeness_comparison_is_read_only_and_only_changes_volumes():
    cohort = load_cohort("echo")
    analytic = _cohort((1, 2, 3))
    for e in cohort.deck_elements:
        analytic[e.name] = None
    analytic.loc[0, "lv_end_diastolic_volume"] = 110.
    private = analytic[["person_id", "index_date"]].copy()
    private["lv_end_diastolic_volume"] = [110., 125., None]
    private["lv_end_systolic_volume"] = [55., None, None]
    original = analytic.copy(deep=True)
    report, updated = compare_echo_completeness(cohort, analytic, private)
    assert len(report) == len(cohort.deck_elements) == 27
    assert len(updated) == len(analytic) == 3
    assert analytic.equals(original)
    r = report.set_index("element")
    assert r.loc["lv_end_diastolic_volume", "before"] == 1 / 3
    assert r.loc["lv_end_diastolic_volume", "after"] == 2 / 3
    assert r.loc["lv_end_systolic_volume", "after"] == 1 / 3
    assert r.loc["wall_motion", "after"] == r.loc["wall_motion", "before"]
    assert r.loc["image_quality", "after"] == r.loc["image_quality", "before"]
    assert r.loc["clinical_indication", "after"] == r.loc["clinical_indication", "before"]
