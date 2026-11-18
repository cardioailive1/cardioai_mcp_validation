"""Synthetic, no-PHI tests for pairwise EF method diagnostic."""
import pandas as pd

from cardioai_mcp.config import load_study
from echo_wm_training.ef_method_pair_diagnostics import diagnose_ef_method_pairs


def _c(pid, source="ACC", partner=None, category="between_method"):
    return dict(
        person_id=pid, index_date="2020-01-02",
        group_key=f"{source}:{pid}", partner_key=partner,
        record_source="ACC+ORD" if partner else source,
        lvef_disagreement_category=category,
    )


def _r(pid, source, name, value):
    return dict(
        person_id=pid, index_date="2020-01-02", echo_test_fpk=f"{source}-{pid}",
        accession_number=str(pid) if source == "ACC" else None,
        order_proc_id=str(pid) if source == "ORD" else None,
        metric="lvef", result_name=name, result_label=name,
        result_value=str(value),
    )


def test_pairwise_summary_tracks_each_method_pair_not_independent_patients():
    cfg = load_study()["modality"]["echo"]["mcp"]["results"]
    candidates = pd.DataFrame([_c(1), _c(2)])
    measurements = pd.DataFrame([
        _r(1, "ACC", "LV_EF_2C_2D_by_MOD", 40),
        _r(1, "ACC", "LV_EF_4C_2D_by_MOD", 60),
        _r(1, "ACC", "LV_EF_BP_2D_by_MOD", 65),
        _r(2, "ACC", "LV_EF_2C_2D_by_MOD", 45),
        _r(2, "ACC", "LV_EF_4C_2D_by_MOD", 58),
    ])
    detail, summary, coverage = diagnose_ef_method_pairs(measurements, candidates, cfg)
    assert len(detail) == 4
    # The 4C-vs-biplane pair is 5 pp and must not be flagged.
    assert int(detail.over_threshold.sum()) == 3
    pair = summary.loc[
        (summary.method_a == "single_plane_2c_mod") &
        (summary.method_b == "single_plane_4c_mod")
    ].iloc[0]
    assert pair.candidate_groups == 2
    assert pair.median_abs_difference_pp == 16.5
    assert coverage.candidates.sum() == 2
    # One exam contributes to several pairs; sum of pair counts exceeds cases.
    assert int(summary.candidate_groups.sum()) > 2


def test_pairing_does_not_cross_acc_ord_identifier_groups():
    cfg = load_study()["modality"]["echo"]["mcp"]["results"]
    candidates = pd.DataFrame([_c(4, partner="ORD:4")])
    measurements = pd.DataFrame([
        _r(4, "ACC", "LV_EF_2C_2D_by_MOD", 20),
        _r(4, "ORD", "LV_EF_BP_2D_by_MOD", 80),
    ])
    detail, summary, coverage = diagnose_ef_method_pairs(measurements, candidates, cfg)
    assert detail.empty and summary.empty
    assert coverage.candidates.sum() == 1


def test_pairwise_comparison_uses_variant_medians_not_raw_extremes():
    cfg = load_study()["modality"]["echo"]["mcp"]["results"]
    candidates = pd.DataFrame([_c(3)])
    measurements = pd.DataFrame([
        _r(3, "ACC", "LV_EF_2C_2D_by_MOD", 20),
        _r(3, "ACC", "LV_EF_2C_2D_by_MOD", 60),
        _r(3, "ACC", "LV_EF_4C_2D_by_MOD", 45),
    ])
    detail, summary, _ = diagnose_ef_method_pairs(measurements, candidates, cfg)
    assert len(detail) == 1
    assert detail.difference_pp.iloc[0] == 5
    assert not bool(detail.over_threshold.iloc[0])
    assert summary.empty


def test_only_requested_category_is_selected():
    cfg = load_study()["modality"]["echo"]["mcp"]["results"]
    candidates = pd.DataFrame([_c(5, category="within_method")])
    rows = pd.DataFrame([_r(5, "ACC", "LVEF visual", 10)])
    detail, summary, coverage = diagnose_ef_method_pairs(rows, candidates, cfg)
    assert detail.empty and summary.empty and coverage.empty


def test_all_pair_summary_includes_nonflagged_and_excludes_ambiguous():
    from echo_wm_training.ef_method_pair_diagnostics import summarize_all_ef_method_pairs
    cfg = load_study()["modality"]["echo"]["mcp"]["results"]
    candidates = pd.DataFrame([
        {**_c(1, category="none"), "candidate_status": "single_exam_candidate"},
        {**_c(2, category="between_method"),
         "candidate_status": "unresolved_method_disagreement"},
        {**_c(3, category="none"), "candidate_status": "ambiguous_multiple_exams"},
    ])
    readings = pd.DataFrame([
        _r(1, "ACC", "LV_EF_2C_2D_by_MOD", 50),
        _r(1, "ACC", "LV_EF_4C_2D_by_MOD", 55),  # 5 pp, not flagged
        _r(2, "ACC", "LV_EF_2C_2D_by_MOD", 40),
        _r(2, "ACC", "LV_EF_4C_2D_by_MOD", 60),  # 20 pp, flagged
        _r(3, "ACC", "LV_EF_2C_2D_by_MOD", 10),
        _r(3, "ACC", "LV_EF_4C_2D_by_MOD", 90),  # excluded ambiguous
    ])
    details, summary = summarize_all_ef_method_pairs(readings, candidates, cfg)
    assert len(details) == 2
    s = summary.iloc[0]
    assert s.candidates == 2
    assert s.source_groups == 2
    assert s.pair_observations == 2
    assert s.over_10pp == 1
    assert s.pct_over_10pp == 50
    assert s.median_abs_pp == 12.5
    assert s.median_signed_pp == -12.5
    assert s.method_b_higher == 2
    assert s.method_a_higher == 0


def test_all_pair_summary_keeps_acc_and_ord_separate():
    from echo_wm_training.ef_method_pair_diagnostics import summarize_all_ef_method_pairs
    cfg = load_study()["modality"]["echo"]["mcp"]["results"]
    candidates = pd.DataFrame([
        {**_c(7, partner="ORD:7", category="none"),
         "candidate_status": "single_exam_candidate"},
    ])
    readings = pd.DataFrame([
        _r(7, "ACC", "LV_EF_2C_2D_by_MOD", 30),
        _r(7, "ACC", "LV_EF_4C_2D_by_MOD", 50),
        _r(7, "ORD", "LV_EF_2C_2D_by_MOD", 60),
        _r(7, "ORD", "LV_EF_4C_2D_by_MOD", 80),
    ])
    details, summary = summarize_all_ef_method_pairs(readings, candidates, cfg)
    assert len(details) == 2
    assert summary.candidates.iloc[0] == 1
    assert summary.source_groups.iloc[0] == 2
    assert summary.pair_observations.iloc[0] == 2
    assert summary.over_10pp.iloc[0] == 2
    assert all(details.difference_pp == 20)
