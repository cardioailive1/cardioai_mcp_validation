"""Read-only explicit-mL echo volume feasibility, not reference selection.

Only the saved cohort's conservatively selected *single identifier*
examination groups are considered. Unitless MC CV measurements are ignored,
not interpreted as mL. Diagnostic priority (biplane > 3D > single-plane)
is a WHAT-IF, not a clinical approval or production mapping.

Do not print/export patient detail, identifiers, numeric values or note text.
Two complementary-safe summary tables are returned, subject to MCP's
cumulative disclosure policy.
"""
from __future__ import annotations

import pandas as pd

from .cohort_extraction_preview import (
    _KEYS, _RANGE, _VOLUME_NAMES, echo_volume_rows_sql,
    preview_echo_extraction,
)
from .exam_resolution import tte_record_rows_sql


_METHODS = {
    "lv_end_diastolic_volume": {
        "biplane": ("LV_vol_BP_ed_2D_by_MOD",),
        "3d": ("LV_vol_ed_3D",),
        "single_plane": ("LV_vol_4C_ed_2D_by_MOD", "LV_vol_2C_ed_2D_by_MOD"),
    },
    "lv_end_systolic_volume": {
        "biplane": ("LV_vol_BP_es_2D_by_MOD",),
        "3d": ("LV_vol_es_3D",),
        "single_plane": ("LV_vol_4C_es_2D_by_MOD", "LV_vol_2C_es_2D_by_MOD"),
    },
}
# Mayo-specific unitless result types intentionally not included.
_UNIT_OK = frozenset(("ml", "milliliter", "milliliters"))
_TIER_ORDER = ("biplane", "3d", "single_plane")
_STATUS_ORDER = (
    "biplane_candidate",
    "3d_candidate",
    "single_plane_candidate",
    "conflicting_preferred_tier",
    "no_explicit_ml",
)


def _method_feasibility(rows: pd.DataFrame, metric: str) -> str:
    """Rank only recognized, numeric, explicitly mL values.

    Within the highest available tier, require consistency (after removing
    identical duplicates). No cross-method equality requirement. Differing
    2C/4C values in the same tier are flagged, not arbitrarily chosen.
    """
    if metric not in _METHODS:
        raise ValueError("Unrecognized volume metric")
    expected = set(_VOLUME_NAMES[metric])
    d = rows.loc[rows.result_name.isin(expected)].copy()
    d["number"] = pd.to_numeric(d.raw_value, errors="coerce")
    lo, hi = _RANGE[metric]
    d = d.loc[d.number.between(lo, hi)]
    unit = d.reported_unit.fillna("").astype(str).str.strip().str.lower()
    d = d.loc[unit.isin(_UNIT_OK)]
    if d.empty:
        return "no_explicit_ml"
    for method in _TIER_ORDER:
        candidates = d.loc[d.result_name.isin(_METHODS[metric][method])]
        if candidates.empty:
            continue
        if candidates.number.nunique() > 1:
            return "conflicting_preferred_tier"
        return method + "_candidate"
    return "no_explicit_ml"


def summarize_explicit_ml_feasibility(
    private: pd.DataFrame,
    tests: pd.DataFrame,
    volume_rows: pd.DataFrame,
    *,
    small_cell: int = 11,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (high-level presence, hypothetical priority tier) safe aggregates.

    Every patient-day is counted once per metric among selected exam groups.
    Do not sum distinct method populations or share small cells.
    """
    if small_cell < 1:
        raise ValueError("small_cell must be positive")
    if not set(_KEYS + ["exam_status"]).issubset(private):
        raise ValueError("Preview missing patient/date/examination status")
    if private.duplicated(_KEYS).any():
        raise ValueError("Duplicate preview patient-days")
    test_key = _KEYS + ["echo_test_fpk"]
    if not set(test_key).issubset(tests):
        raise ValueError("Tests missing examination keys")
    if not set(test_key + ["result_name", "raw_value", "reported_unit"]).issubset(volume_rows):
        raise ValueError("Volume rows missing examination or measurement fields")

    selected = private.loc[
        private.exam_status.eq("single_identifier_group"), _KEYS
    ].copy()
    selected.index_date = pd.to_datetime(selected.index_date).dt.date
    t = tests[test_key].copy()
    t.index_date = pd.to_datetime(t.index_date).dt.date
    t.echo_test_fpk = t.echo_test_fpk.astype("string").str.strip()
    t = t.merge(selected, on=_KEYS, how="inner", validate="many_to_one")
    t = t.drop_duplicates(test_key)
    v = volume_rows.copy()
    v.index_date = pd.to_datetime(v.index_date).dt.date
    v.echo_test_fpk = v.echo_test_fpk.astype("string").str.strip()
    v = v.merge(t, on=test_key, how="inner", validate="many_to_one")
    grouped = {k: g for k, g in v.groupby(_KEYS, dropna=False)}
    empty = pd.DataFrame(columns=["result_name", "raw_value", "reported_unit"])

    presence = []
    detail = []
    for metric in _METHODS:
        counts = {s: 0 for s in _STATUS_ORDER}
        for r in selected.itertuples(index=False):
            status = _method_feasibility(
                grouped.get((r.person_id, r.index_date), empty), metric)
            counts[status] += 1
        if sum(counts.values()) != len(selected):
            raise ValueError("Metric coverage did not reconcile to selected exams")
        exists = len(selected) - counts["no_explicit_ml"]
        none = counts["no_explicit_ml"]
        present_safe = all(n == 0 or n >= small_cell
                           for n in (exists, none))
        presence.append({
            "metric": metric, "selected_patient_days": len(selected),
            "with_explicit_ml": str(exists) if present_safe else "suppressed",
            "without_explicit_ml": str(none) if present_safe else "suppressed",
        })
        tier_safe = not any(0 < n < small_cell for n in counts.values())
        for s in _STATUS_ORDER:
            detail.append({
                "metric": metric, "priority_scenario": s,
                "patient_days": str(counts[s]) if tier_safe else "suppressed",
            })
    return pd.DataFrame(presence), pd.DataFrame(detail)


def run_explicit_ml_feasibility(eng, study: dict):
    """No source writes, no report text; only aggregate outputs leave function."""
    start, end = str(study["index_start"]), str(study["index_end"])
    t = eng.query(tte_record_rows_sql(study["modality"], start=start, end=end))
    v = eng.query(echo_volume_rows_sql(study["modality"], start=start, end=end))
    analytic = eng.read_table("echo_analytic")
    no_notes = pd.DataFrame(columns=_KEYS + ["echo_test_fpk", "echo_note"])
    private = preview_echo_extraction(analytic, t, v, no_notes)
    return summarize_explicit_ml_feasibility(
        private, t, v, small_cell=int(study.get("small_cell", 11))
    )
