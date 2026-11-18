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
