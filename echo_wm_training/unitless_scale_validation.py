"""Read-only unit-scale check: unitless Mayo MC CV vs explicitly labeled mL.

Pair values only within the identical ECHO_TEST_FPK, patient and index date.
Never pair merely on encounter, ACC/ORD identifiers, or close timestamps.
This diagnostic provides evidence about *scale*, not approval to infer mL,
alter clinical selection policy, or rebuild the saved cohort.

Patient identifiers, test identifiers and source numerical values remain
in private MCP memory. Return only disclosure-guarded aggregate results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cohort_extraction_preview import (
    _KEYS, _RANGE, echo_volume_rows_sql,
)
from .exam_resolution import tte_record_rows_sql


_EXAM_KEYS = _KEYS + ["echo_test_fpk"]
_ML = frozenset(("ml", "milliliter", "milliliters"))
_UNITLESS_NAMES = {
    "lv_end_diastolic_volume": "MC CV LV END DIASTOLIC VOLUME",
    "lv_end_systolic_volume": "MC CV LV END SYSTOLIC VOLUME",
}
_COMPARATORS = {
    "lv_end_diastolic_volume": {
        "biplane_2d": "LV_vol_BP_ed_2D_by_MOD",
        "four_chamber_2d": "LV_vol_4C_ed_2D_by_MOD",
        "two_chamber_2d": "LV_vol_2C_ed_2D_by_MOD",
        "three_d": "LV_vol_ed_3D",
    },
    "lv_end_systolic_volume": {
        "biplane_2d": "LV_vol_BP_es_2D_by_MOD",
        "four_chamber_2d": "LV_vol_4C_es_2D_by_MOD",
        "two_chamber_2d": "LV_vol_2C_es_2D_by_MOD",
        "three_d": "LV_vol_es_3D",
    },
}
_PERIODS = ("2012-2018", "2019-2023")


def _period(value) -> str | None:
    year = pd.Timestamp(value).year
    if 2012 <= year <= 2018:
        return "2012-2018"
    if 2019 <= year <= 2023:
        return "2019-2023"
    return None


def _protect_complement(n: int, yes: int, small_cell: int) -> tuple[str, str, str]:
    """Protect both a count and its difference from a known total."""
    if any(0 < v < small_cell for v in (n, yes, n - yes)):
        return ("suppressed",) * 3
    return str(n), str(yes), str(n - yes)


def _protected_comparison(
    rows: pd.DataFrame, small_cell: int
) -> dict[str, str]:
    """Gate ALL comparison fields when any positive near/far cell is small."""
    n = len(rows)
    near10 = int(rows["within_10pct"].sum())
    near25 = int(rows["within_25pct"].sum())
    if (n < small_cell or any(
        0 < val < small_cell
        for val in (near10, n - near10, near25, n - near25)
    )):
        return {
            "patient_days": "suppressed",
            "median_mc_over_ml": "suppressed",
            "p25_mc_over_ml": "suppressed",
            "p75_mc_over_ml": "suppressed",
            "pct_within_10pct": "suppressed",
            "pct_within_25pct": "suppressed",
        }
    q = rows.ratio.quantile([.25, .50, .75])
    return {
        "patient_days": str(n),
        "median_mc_over_ml": f"{q.loc[.50]:.3f}",
        "p25_mc_over_ml": f"{q.loc[.25]:.3f}",
        "p75_mc_over_ml": f"{q.loc[.75]:.3f}",
        "pct_within_10pct": f"{100 * near10 / n:.1f}",
        "pct_within_25pct": f"{100 * near25 / n:.1f}",
    }


def compare_unitless_with_labeled_ml(
    tests: pd.DataFrame,
    volume_rows: pd.DataFrame,
    *,
    small_cell: int = 11,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (coverage, exact-FPK ratio profile) as aggregate-only tables.

    - Unitless MC CV result names require truly *blank* source unit metadata.
    - Comparators require explicit mL and an in-range numeric value.
    - Ambiguous different numeric values within a test/result name are
      excluded. Identical duplicate source rows are harmless.
    - A comparator-method ratio is excluded if one patient/index-day has
      more than one source test with that pair.
    - Cross-method disagreement is allowed; methods are reported separately.

    These are not same-method comparisons: clinical method differences can
    cause ratios != 1 even where both measurements use mL.
    """
    if small_cell < 1:
        raise ValueError("small_cell must be positive")
    required_t = set(_EXAM_KEYS)
    required_v = required_t | {"result_name", "raw_value", "reported_unit"}
    if not required_t.issubset(tests):
        raise ValueError("Tests missing exact echo source identifiers")
    if not required_v.issubset(volume_rows):
        raise ValueError("Volume rows missing result, value or unit")
    t = tests[_EXAM_KEYS].copy()
    t.index_date = pd.to_datetime(t.index_date).dt.date
    t.echo_test_fpk = t.echo_test_fpk.astype("string").str.strip()
    if t.echo_test_fpk.isna().any() or t.echo_test_fpk.eq("").any():
        raise ValueError("Missing echo test identifier")
    t = t.drop_duplicates()

    v = volume_rows.copy()
    v.index_date = pd.to_datetime(v.index_date).dt.date
    v.echo_test_fpk = v.echo_test_fpk.astype("string").str.strip()
    v = v.merge(t, on=_EXAM_KEYS, how="inner", validate="many_to_one")
    mapping = {}
    for metric, unitless_name in _UNITLESS_NAMES.items():
        mapping[unitless_name] = (metric, "mc_cv")
        for method, comparator_name in _COMPARATORS[metric].items():
            mapping[comparator_name] = (metric, method)
    v = v.loc[v.result_name.isin(mapping)].copy()
    v["number"] = pd.to_numeric(v.raw_value, errors="coerce")
    v = v.loc[np.isfinite(v.number) & v.number.gt(0)].copy()
    v["period"] = v.index_date.map(_period)
    v = v.loc[v.period.notna()].copy()
    v["metric"] = v.result_name.map(lambda name: mapping[name][0])
    v["method"] = v.result_name.map(lambda name: mapping[name][1])
    normalized_unit = v.reported_unit.fillna("").astype(str).str.strip().str.lower()
    v["unit_missing"] = normalized_unit.eq("")
    v["unit_ml"] = normalized_unit.isin(_ML)

    entries = []
    for group, rows in v.groupby(
        _EXAM_KEYS + ["period", "metric", "method"], dropna=False
    ):
        pid, date, test_id, period, metric, method = group
        if method == "mc_cv":
            valid_units = rows.unit_missing.all()
        else:
            lower, upper = _RANGE[metric]
            valid_units = (
                rows.unit_ml.all() and rows.number.between(lower, upper).all()
            )
        if not valid_units or rows.number.nunique() != 1:
            continue
        entries.append({
            "person_id": pid, "index_date": date, "echo_test_fpk": test_id,
            "period": period, "metric": metric, "method": method,
            "number": float(rows.number.iloc[0]),
        })

    exact = pd.DataFrame(entries, columns=_EXAM_KEYS + [
        "period", "metric", "method", "number"
    ])
    unitless = exact.loc[exact.method.eq("mc_cv")].rename(
        columns={"number": "mc_cv_number"}
    ).drop(columns="method")
    explicit = exact.loc[exact.method.ne("mc_cv")].rename(
        columns={"number": "ml_number"}
    )
    pairs = unitless.merge(
        explicit, on=_EXAM_KEYS + ["period", "metric"],
        how="inner", validate="one_to_many",
    )
    pairs["ratio"] = pairs.mc_cv_number / pairs.ml_number
    pairs["within_10pct"] = pairs.ratio.sub(1).abs().le(.10)
    pairs["within_25pct"] = pairs.ratio.sub(1).abs().le(.25)

    coverage = []
    ratio_profile = []
    for metric in _UNITLESS_NAMES:
        for period in _PERIODS:
            left = unitless.loc[
                unitless.metric.eq(metric) & unitless.period.eq(period)
            ]
            right = pairs.loc[
                pairs.metric.eq(metric) & pairs.period.eq(period)
            ]
            total = len(left[_KEYS].drop_duplicates())
            matched = len(right[_KEYS].drop_duplicates())
            all_count, yes_count, no_count = _protect_complement(
                total, matched, small_cell
            )
            coverage.append({
                "metric": metric, "period": period,
                "unitless_mc_cv_patient_days": all_count,
                "paired_same_test_fpk_patient_days": yes_count,
                "no_paired_test_fpk_patient_days": no_count,
            })
            for method in _COMPARATORS[metric]:
                method_pairs = right.loc[right.method.eq(method)].copy()
                # A single day with two tests is not one unequivocal
                # comparison, even if both tests' values happen to agree.
                if not method_pairs.empty:
                    count = method_pairs.groupby(_KEYS)[
                        "echo_test_fpk"
                    ].transform("nunique")
                    method_pairs = method_pairs.loc[count.eq(1)]
                report = _protected_comparison(method_pairs, small_cell)
                ratio_profile.append({
                    "metric": metric, "period": period,
                    "comparator": method, **report
                })
    return pd.DataFrame(coverage), pd.DataFrame(ratio_profile)


def run_unitless_scale_validation(eng, study: dict):
    """MCP-only read-only query runner. Return only redacted aggregate tables."""
    start, end = str(study["index_start"]), str(study["index_end"])
    tests = eng.query(tte_record_rows_sql(study["modality"], start=start, end=end))
    volume = eng.query(echo_volume_rows_sql(study["modality"], start=start, end=end))
    return compare_unitless_with_labeled_ml(
        tests, volume, small_cell=int(study.get("small_cell", 11))
    )
