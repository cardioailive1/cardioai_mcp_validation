"""Read-only second-stage diagnosis of LVEDV/LVESV preview holds.

Answers whether the conservative selected-exam preview is blocked by
incomplete/non-mL unit metadata, within-result-name disagreement, or
different measurement methods. These are *reason flags*, not clinical
reference-selection rules. No cohort changes and no raw values exported.
"""
from __future__ import annotations

import pandas as pd

from .cohort_extraction_preview import (
    _KEYS, _RANGE, _VOLUME_NAMES, echo_volume_rows_sql,
    preview_echo_extraction,
)
from .exam_resolution import tte_record_rows_sql


_HOLD_REASONS = (
    "unknown_or_non_absolute_unit",
    "disagreeing_measurements_review",
)
_CATEGORIES = (
    "unit_metadata_only",
    "different_methods_only",
    "same_method_conflict",
    "unit_and_value_disagreement",
)


def _classify_held_metric(rows: pd.DataFrame, metric: str) -> str:
    """Classify values the current preview withheld, without choosing one."""
    selected_names = _VOLUME_NAMES[metric]
    data = rows.loc[rows.result_name.isin(selected_names)].copy()
    data["number"] = pd.to_numeric(data.raw_value, errors="coerce")
    lo, hi = _RANGE[metric]
    data = data.loc[data.number.between(lo, hi)]
    if data.empty:
        raise ValueError("Held metric has no in-range source measurement")
    valid_unit = data.reported_unit.fillna("").astype(str).str.strip().str.lower().isin(
        ("ml", "milliliter", "milliliters")
    )
    bad_unit = bool((~valid_unit).any())
    within_name = any(
        sub.number.nunique() > 1
        for _, sub in data.groupby("result_name", dropna=False)
    )
    any_difference = data.number.nunique() > 1
    # Different named methods do not prove technical inconsistency; different
    # methods can legitimately produce different numerical values.
    if bad_unit and any_difference:
        return "unit_and_value_disagreement"
    if bad_unit:
        return "unit_metadata_only"
    if within_name:
        return "same_method_conflict"
    if any_difference:
        return "different_methods_only"
    raise ValueError("Held volume has neither unit nor numeric conflict")


def summarize_volume_hold_causes(
    private: pd.DataFrame,
    tests: pd.DataFrame,
    volume_rows: pd.DataFrame,
    *,
    small_cell: int = 11,
) -> pd.DataFrame:
    """Disclosure-guarded mutually exclusive causes within SELECTED cases.

    Redacts all breakdown cells for a metric when any positive category
    count is below small_cell; preserves the categorical labels.
    """
    required = {*_KEYS, "exam_status",
                *[m + "_status" for m in _VOLUME_NAMES]}
    if not required.issubset(private):
        raise ValueError(f"Missing preview fields: {sorted(required - set(private))}")
    if small_cell < 1:
        raise ValueError("small_cell must be positive")
    if private.duplicated(_KEYS).any():
        raise ValueError("Preview has duplicate cohort patient/index-days")
    t_cols = _KEYS + ["echo_test_fpk"]
    if not set(t_cols).issubset(tests) or not set(t_cols + [
        "result_name", "raw_value", "reported_unit"
    ]).issubset(volume_rows):
        raise ValueError("Source test or volume row is missing required columns")
    chosen = private.loc[
        private.exam_status.eq("single_identifier_group"), _KEYS
    ].copy()
    chosen["index_date"] = pd.to_datetime(chosen.index_date).dt.date
    t = tests.copy()
    t["index_date"] = pd.to_datetime(t.index_date).dt.date
    t["echo_test_fpk"] = t.echo_test_fpk.astype("string").str.strip()
    t = t.merge(chosen, on=_KEYS, how="inner", validate="many_to_one")
    t = t[t_cols].drop_duplicates()
    v = volume_rows.copy()
    v["index_date"] = pd.to_datetime(v.index_date).dt.date
    v["echo_test_fpk"] = v.echo_test_fpk.astype("string").str.strip()
    v = v.merge(t, on=t_cols, how="inner", validate="many_to_one")
    by_day = {k: d for k, d in v.groupby(_KEYS, dropna=False)}
    rows = []
    for metric in _VOLUME_NAMES:
        status_col = metric + "_status"
        held = private.loc[
            private.exam_status.eq("single_identifier_group")
            & private[status_col].isin(_HOLD_REASONS), _KEYS + [status_col]
        ].copy()
        held["index_date"] = pd.to_datetime(held.index_date).dt.date
        counts = {name: 0 for name in _CATEGORIES}
        for r in held.itertuples(index=False):
            reason = _classify_held_metric(by_day.get(
                (r.person_id, r.index_date), pd.DataFrame(
                    columns=["result_name", "raw_value", "reported_unit"]
                )), metric)
            status = getattr(r, status_col)
            has_bad_unit = reason in (
                "unit_metadata_only", "unit_and_value_disagreement")
            if (status == "unknown_or_non_absolute_unit") != has_bad_unit:
                raise ValueError("Preview and hold reason disagree about unit status")
            counts[reason] += 1
        if sum(counts.values()) != len(held):
            raise ValueError("Held volume counts failed reconciliation")
        small = any(0 < n < small_cell for n in counts.values())
        for name, number in counts.items():
            rows.append({
                "metric": metric,
                "hold_reason": name,
                "patient_days": "suppressed" if small else str(number),
            })
    return pd.DataFrame(rows)


def run_volume_hold_audit(eng, study: dict) -> pd.DataFrame:
    """Run within MCP; return ONLY disclosure-guarded aggregate categories."""
    start, end = str(study["index_start"]), str(study["index_end"])
    tests = eng.query(tte_record_rows_sql(study["modality"], start=start, end=end))
    volumes = eng.query(echo_volume_rows_sql(study["modality"], start=start, end=end))
    analytic = eng.read_table("echo_analytic")
    empty_notes = pd.DataFrame(columns=[
        *_KEYS, "echo_test_fpk", "echo_note"
    ])
    private = preview_echo_extraction(analytic, tests, volumes, empty_notes)
    return summarize_volume_hold_causes(
        private, tests, volumes, small_cell=int(study.get("small_cell", 11))
    )
