"""Read-only audit of echo LVEDV/LVESV result-unit metadata.

This audit investigates why the strict preview withheld 443/454 LVEDV
patient-days due to at least one absent or non-mL unit. It does not
infer/assign units, apply a clinical method hierarchy, or update the cohort.
Only source result names and disclosure-controlled counts are returned.
Raw measurements and identifiers stay inside MCP.

Per-result-name categories are mutually exclusive *within that name*
but multiple result names can appear on the same patient-day. Never add
counts across result names as distinct patients.
"""
from __future__ import annotations

import pandas as pd

from .cohort_extraction_preview import (
    _KEYS,
    _RANGE,
    _VOLUME_NAMES,
    echo_volume_rows_sql,
    preview_echo_extraction,
)
from .exam_resolution import tte_record_rows_sql

_UNIT_CLASSES = ("explicit_ml_only", "missing_only", "non_ml_only", "mixed_unit_metadata")
_ML = frozenset(("ml", "milliliter", "milliliters"))


def _normalized_units(units: pd.Series) -> pd.Series:
    s = units.fillna("").astype(str).str.strip().str.lower()
    return s.map(lambda value:
                 "missing" if not value else "ml" if value in _ML else "other")


def classify_numeric_unit_profile(
    selected_preview: pd.DataFrame,
    tests: pd.DataFrame,
    volume_rows: pd.DataFrame,
    *,
    small_cell: int = 11,
) -> pd.DataFrame:
    """Aggregate parseable, in-range source measurements by exact result name.

    The denominator for each method is the distinct selected patient-days
    with at least one in-range numeric row of that result name. Some names
    appear in many records on the same exam, hence group by patient/day/name.
    Any positive count < small_cell redacts the whole corresponding name row.
    Missing values in other columns are NOT published.
    """
    if small_cell < 1:
        raise ValueError("small_cell must be positive")
    needed = set(_KEYS + ["exam_status"])
    if needed - set(selected_preview):
        raise ValueError(f"Preview missing fields: {sorted(needed - set(selected_preview))}")
    if selected_preview.duplicated(_KEYS).any():
        raise ValueError("Duplicate preview index days")
    needed_t = set(_KEYS + ["echo_test_fpk"])
    if needed_t - set(tests):
        raise ValueError("Missing test identifiers")
    needed_v = needed_t | {"result_name", "raw_value", "reported_unit"}
    if needed_v - set(volume_rows):
        raise ValueError("Missing source result fields")

    chosen = selected_preview.loc[
        selected_preview.exam_status.eq("single_identifier_group"), _KEYS
    ].copy()
    chosen.index_date = pd.to_datetime(chosen.index_date).dt.date
    t = tests[_KEYS + ["echo_test_fpk"]].copy()
    t.index_date = pd.to_datetime(t.index_date).dt.date
    t.echo_test_fpk = t.echo_test_fpk.astype("string").str.strip()
    t = t.merge(chosen, on=_KEYS, how="inner", validate="many_to_one").drop_duplicates()

    v = volume_rows.copy()
    v.index_date = pd.to_datetime(v.index_date).dt.date
    v.echo_test_fpk = v.echo_test_fpk.astype("string").str.strip()
    v = v.merge(t, on=_KEYS + ["echo_test_fpk"], how="inner", validate="many_to_one")
    all_names = {n: metric for metric, seq in _VOLUME_NAMES.items() for n in seq}
    v = v.loc[v.result_name.isin(all_names)].copy()
    v["metric"] = v.result_name.map(all_names)
    v["number"] = pd.to_numeric(v.raw_value, errors="coerce")
    v = v.loc[v.apply(
        lambda r: (pd.notna(r.number)
                   and _RANGE[r.metric][0] <= r.number <= _RANGE[r.metric][1]),
        axis=1,
    )].copy()
    v["unit_class"] = _normalized_units(v.reported_unit)

    grouped = []
    for (metric, name, pid, day), g in v.groupby(
        ["metric", "result_name", *_KEYS], dropna=False
    ):
        codes = set(g.unit_class)
        if codes == {"ml"}:
            kind = "explicit_ml_only"
        elif codes == {"missing"}:
            kind = "missing_only"
        elif codes == {"other"}:
            kind = "non_ml_only"
        else:
            kind = "mixed_unit_metadata"
        grouped.append({"metric": metric, "result_name": name,
                        "unit_category": kind})
    data = pd.DataFrame(grouped, columns=["metric", "result_name", "unit_category"])
    out = []
    for metric, names in _VOLUME_NAMES.items():
        for name in names:
            subset = data.loc[
                data.metric.eq(metric) & data.result_name.eq(name)
            ]
            counts = subset.unit_category.value_counts()
            total = int(len(subset))
            amounts = [int(counts.get(cat, 0)) for cat in _UNIT_CLASSES]
            if sum(amounts) != total:
                raise ValueError("Unit categories failed reconciliation")
            # Count 0 is explicit zero; any small positive class or small
            # total redacts the entire row to prevent within-name subtraction.
            safe = (total == 0 or total >= small_cell) and not any(
                0 < n < small_cell for n in amounts
            )
            out.append({
                "metric": metric,
                "result_name": name,
                "patient_days_with_numeric_in_range": (
                    str(total) if safe else "suppressed"
                ),
                **{
                    cat: str(n) if safe else "suppressed"
                    for cat, n in zip(_UNIT_CLASSES, amounts)
                }
            })
    return pd.DataFrame(out)


def run_volume_unit_profile(eng, study: dict) -> pd.DataFrame:
    """MCP-only data retrieval; return only aggregate, redacted unit profile."""
    start, end = str(study["index_start"]), str(study["index_end"])
    tests = eng.query(tte_record_rows_sql(study["modality"], start=start, end=end))
    volume = eng.query(echo_volume_rows_sql(study["modality"], start=start, end=end))
    analytic = eng.read_table("echo_analytic")
    empty_notes = pd.DataFrame(
        columns=_KEYS + ["echo_test_fpk", "echo_note"]
    )
    private = preview_echo_extraction(analytic, tests, volume, empty_notes)
    return classify_numeric_unit_profile(
        private, tests, volume, small_cell=int(study.get("small_cell", 11))
    )
