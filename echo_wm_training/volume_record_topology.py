"""Read-only record-topology triage for unitless Mayo MC CV volume results.

No numeric comparison across different source ECHO_TEST_FPKs is performed.
Same patient/index-day, or reciprocal ACC/ORD timestamp candidate, does NOT
establish that two test records describe one echocardiogram. Results are
patient-day aggregates with primary and complementary suppression only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cohort_extraction_preview import _KEYS, _RANGE, _group_keys, echo_volume_rows_sql
from .exam_resolution import resolve_tte_groups, tte_record_rows_sql
from .unitless_scale_validation import (
    _COMPARATORS, _EXAM_KEYS, _ML, _PERIODS, _UNITLESS_NAMES,
    _period, _protect_complement,
)


def summarize_volume_record_topology(
    tests: pd.DataFrame,
    volume_rows: pd.DataFrame,
    *,
    small_cell: int = 11,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate only; record linkage cannot be treated as accepted identity."""
    if small_cell < 1:
        raise ValueError("small_cell must be positive")
    required_t = set(_EXAM_KEYS + [
        "exam_datetime", "accession_number", "order_proc_id"
    ])
    required_v = set(_EXAM_KEYS + [
        "result_name", "raw_value", "reported_unit"
    ])
    if not required_t.issubset(tests):
        raise ValueError("Missing test examination identifier columns")
    if not required_v.issubset(volume_rows):
        raise ValueError("Missing volume measurement columns")
    t = tests.copy()
    t.index_date = pd.to_datetime(t.index_date).dt.date
    t.echo_test_fpk = t.echo_test_fpk.astype("string").str.strip()
    if t.echo_test_fpk.isna().any() or t.echo_test_fpk.eq("").any():
        raise ValueError("Missing ECHO_TEST_FPK")
    t = t.drop_duplicates(_EXAM_KEYS)
    mapped = _group_keys(t)
    resolved = resolve_tte_groups(t)
    g = mapped[_EXAM_KEYS + ["group_key"]].merge(
        resolved[_KEYS + ["group_key", "status", "partner_key"]],
        on=_KEYS + ["group_key"], how="left", validate="many_to_one",
    )
    group_by_test = {
        tuple(getattr(row, k) for k in _EXAM_KEYS):
        (row.group_key, row.status, row.partner_key)
        for row in g.itertuples(index=False)
    }

    v = volume_rows.copy()
    v.index_date = pd.to_datetime(v.index_date).dt.date
    v.echo_test_fpk = v.echo_test_fpk.astype("string").str.strip()
    v = v.merge(t[_EXAM_KEYS], on=_EXAM_KEYS, how="inner", validate="many_to_one")
    names = {
        name: (metric, "mc_cv")
        for metric, name in _UNITLESS_NAMES.items()
    }
    for metric, methods in _COMPARATORS.items():
        for name in methods.values():
            names[name] = (metric, "explicit_ml")
    v = v.loc[v.result_name.isin(names)].copy()
    v["metric"] = v.result_name.map(lambda name: names[name][0])
    v["family"] = v.result_name.map(lambda name: names[name][1])
    v["number"] = pd.to_numeric(v.raw_value, errors="coerce")
    v["normalized_unit"] = v.reported_unit.fillna("").astype(str).str.strip().str.lower()
    v = v.loc[np.isfinite(v.number) & v.number.gt(0)].copy()

    # A named result on one exact test must have a unique numeric value and
    # an internally consistent unit. Do not choose between conflicting rows.
    entry_rows = []
    for keys, part in v.groupby(
        _EXAM_KEYS + ["metric", "family", "result_name"], dropna=False
    ):
        pid, day, fpk, metric, family, result_name = keys
        if part.number.nunique() != 1:
            continue
        if family == "mc_cv":
            if not part.normalized_unit.eq("").all():
                continue
        else:
            low, high = _RANGE[metric]
            if not (
                part.normalized_unit.isin(_ML).all()
                and part.number.between(low, high).all()
            ):
                continue
        entry_rows.append({
            "person_id": pid, "index_date": day, "echo_test_fpk": fpk,
            "metric": metric, "family": family,
        })

    valid = pd.DataFrame(entry_rows, columns=_EXAM_KEYS + ["metric", "family"])
    records = {
        key: sub for key, sub in valid.groupby(
            _KEYS + ["metric"], dropna=False
        )
    }
    # Group the distinct MC-bearing patient-days, even when no labelled
    # counterpart exists. A patient-day may have several MC source records.
    findings = []
    for (pid, day, metric), chunk in records.items():
        mc = set(chunk.loc[chunk.family.eq("mc_cv"), "echo_test_fpk"])
        if not mc:
            continue
        explicit = set(chunk.loc[
            chunk.family.eq("explicit_ml"), "echo_test_fpk"
        ])
        # Exact-test co-occurrence was separately examined; this diagnostic
        # intentionally asks about OTHER test records only. A source test
        # may contain both families, so compare each distinct FPK pair
        # instead of removing every MC-bearing FPK from explicit candidates.
        record_pairs = [
            (mc_test, ml_test)
            for mc_test in mc for ml_test in explicit
            if mc_test != ml_test
        ]
        has_other = bool(record_pairs)
        provisional = False
        for mc_test, ml_test in record_pairs:
            a = group_by_test.get((pid, day, mc_test))
            b = group_by_test.get((pid, day, ml_test))
            if (a is not None and b is not None
                and a[1] == "provisional_mutual_match"
                and b[1] == "provisional_mutual_match"
                and a[2] == b[0] and b[2] == a[0]):
                provisional = True
                break
        period = _period(day)
        if period is not None:
            findings.append({
                "metric": metric, "period": period,
                "other_test_ml": has_other,
                "provisional_pair_with_ml": provisional,
            })
    outcome = pd.DataFrame(
        findings, columns=["metric", "period", "other_test_ml",
                           "provisional_pair_with_ml"]
    )
    cooccur, pairs = [], []
    for metric in _UNITLESS_NAMES:
        for period in _PERIODS:
            sub = outcome.loc[
                outcome.metric.eq(metric) & outcome.period.eq(period)
            ]
            n = len(sub)
            yes = int(sub.other_test_ml.sum())
            reciprocal = int(sub.provisional_pair_with_ml.sum())
            if reciprocal > yes:
                raise ValueError("Reciprocal match cannot exceed cross-record co-occurrence")
            # These tables will be shown together. If reciprocal is a
            # subset of other-test co-occurrence, releasing both exact
            # totals could reveal a small (other-test MINUS reciprocal)
            # group even when each standalone table seems safe.
            joint = (n, yes, n - yes, reciprocal, n - reciprocal,
                     yes - reciprocal)
            if any(0 < x < small_cell for x in joint):
                a, b, c = ("suppressed",) * 3
                x, y, z = ("suppressed",) * 3
            else:
                a, b, c = _protect_complement(n, yes, small_cell)
                x, y, z = _protect_complement(n, reciprocal, small_cell)
            cooccur.append({
                "metric": metric, "period": period,
                "unitless_mc_cv_patient_days": a,
                "explicit_ml_other_test_patient_days": b,
                "no_explicit_ml_other_test_patient_days": c,
            })
            pairs.append({
                "metric": metric, "period": period,
                "unitless_mc_cv_patient_days": x,
                "provisional_acc_ord_pair_with_ml_days": y,
                "without_provisional_pair_with_ml_days": z,
            })
    return pd.DataFrame(cooccur), pd.DataFrame(pairs)


def run_volume_record_topology(eng, study: dict):
    """MCP-only source access. Return only suppressed aggregate tables."""
    start, end = str(study["index_start"]), str(study["index_end"])
    tests = eng.query(tte_record_rows_sql(study["modality"], start=start, end=end))
    volume = eng.query(echo_volume_rows_sql(study["modality"], start=start, end=end))
    return summarize_volume_record_topology(
        tests, volume, small_cell=int(study.get("small_cell", 11))
    )
