"""Read-only EIMS-only LVEDV/LVESV extraction feasibility on saved echo cohort.

Mayo DIM_SOURCE_SYSTEM verified: EIMS ROW_SOURCE_ID=12315, Epic=47918.
Only values explicitly labeled mL with source TEST *and* RESULT both EIMS
are considered. Exactly one EIMS ECHO_TEST_FPK is necessary for this
feasibility scenario, but it does NOT establish that this is the clinically
selected index examination, especially when another source has a TTE on
the same day. No clinical values are returned or written.

Result counts are disclosure-gated patient/index-day aggregates. They need
cumulative disclosure review before leaving Mayo Cloud PC.
"""
from __future__ import annotations

import pandas as pd

from .cohort_extraction_preview import _KEYS
from .exam_resolution import tte_record_rows_sql
from .explicit_ml_feasibility import _method_feasibility
from .mcp_echo import _cfg, _t
from .source_system_bridge import source_system_bridge_rows_sql
from .unitless_scale_validation import _PERIODS, _period

_EIMS = "12315"  # Mayo source dimension resolved EIMS in MCP, 2026-11-23
_TEST_KEYS = _KEYS + ["echo_test_fpk"]
_METRICS = ("lv_end_diastolic_volume", "lv_end_systolic_volume")
_TOPOLOGY = (
    "no_conventional_tte",
    "no_eims_tte",
    "multiple_eims_source_tests",
    "single_eims_test_only",
    "single_eims_test_other_source_present",
)
_EXPLICIT_ML = ("explicit_ml_candidate", "preferred_tier_conflict", "no_explicit_ml")


def eims_test_source_sql(modality_cfg: dict, *, start: str, end: str) -> str:
    """Same-day conventional-TTE index tests and verified source IDs.

    The join is on ECHO_TEST_FPK, never patient/date alone.
    """
    cfg = _cfg(modality_cfg)
    e = cfg["echo_test_columns"]
    return f"""
WITH tte AS ({tte_record_rows_sql(modality_cfg, start=start, end=end)})
SELECT DISTINCT tte.person_id, tte.index_date, tte.echo_test_fpk,
       CAST(t.ROW_SOURCE_ID AS STRING) AS test_row_source_id
FROM tte
JOIN {_t(cfg, 'echo_test_table')} t
  ON CAST(t.{e['test_key']} AS STRING) = tte.echo_test_fpk
"""


def _clean_source(value) -> str | None:
    if pd.isna(value):
        return None
    return str(value).strip() or None


def summarize_eims_volume_eligibility(
    cohort: pd.DataFrame,
    tests: pd.DataFrame,
    source_results: pd.DataFrame,
    *,
    small_cell: int = 11,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (index-day test topology, EIMS explicit-mL method feasibility).

    Only a sole EIMS FPK per patient/index-day may contribute a candidate.
    The same source-test FPK must carry an EIMS ROW_SOURCE_ID in both
    FACT_ECHO_TEST and FACT_ECHO_RESULTS; unit metadata must state mL.
    Non-EIMS same-day TTEs never contribute clinical values and are
    reported as a separate review status, not silently reconciled.

    Tier choice uses the existing *hypothetical* _method_feasibility
    (biplane > 3D > single plane); the categories are only candidate
    availability and never clinical method approval or final reference.
    """
    if small_cell < 1:
        raise ValueError("small_cell must be positive")
    if not set(_KEYS).issubset(cohort) or not set(_TEST_KEYS + ["test_row_source_id"]).issubset(tests):
        raise ValueError("Missing saved-cohort keys or EIMS source test fields")
    need_v = set(_TEST_KEYS + ["test_row_source_id", "result_row_source_id",
                                "result_name", "raw_value", "reported_unit"])
    if not need_v.issubset(source_results):
        raise ValueError("Missing source volume provenance or result fields")

    c = cohort[_KEYS].copy()
    c.index_date = pd.to_datetime(c.index_date).dt.date
    if c.duplicated(_KEYS).any() or c[_KEYS].isna().any().any():
        raise ValueError("Cohort must contain unique nonblank person/index-days")
    t = tests[_TEST_KEYS + ["test_row_source_id"]].copy()
    t.index_date = pd.to_datetime(t.index_date).dt.date
    t.echo_test_fpk = t.echo_test_fpk.astype("string").str.strip()
    if t[_TEST_KEYS].isna().any().any() or t.echo_test_fpk.eq("").any():
        raise ValueError("TTE test identifiers cannot be missing")
    t["test_row_source_id"] = t.test_row_source_id.map(_clean_source)
    if t.drop_duplicates(_TEST_KEYS + ["test_row_source_id"]).duplicated(_TEST_KEYS).any():
        raise ValueError("Conflicting source IDs for a TTE test")
    t = t.drop_duplicates(_TEST_KEYS)
    t = t.merge(c, on=_KEYS, how="inner", validate="many_to_one")

    v = source_results.copy()
    v.index_date = pd.to_datetime(v.index_date).dt.date
    v.echo_test_fpk = v.echo_test_fpk.astype("string").str.strip()
    if v[_TEST_KEYS].isna().any().any() or v.echo_test_fpk.eq("").any():
        raise ValueError("Source result test identifier missing")
    v["test_row_source_id"] = v.test_row_source_id.map(_clean_source)
    v["result_row_source_id"] = v.result_row_source_id.map(_clean_source)
    v = v.merge(
        t[_TEST_KEYS + ["test_row_source_id"]],
        on=_TEST_KEYS, how="inner", validate="many_to_one",
        suffixes=("_result", "_test")
    )
    # Independently verify that result provenance matches test provenance.
    # Do not silently accept source rows from a different test source.
    bad = v.loc[
        v.test_row_source_id_result.ne(v.test_row_source_id_test)
        | v.test_row_source_id_result.isna()
        | v.test_row_source_id_test.isna()
    ]
    if not bad.empty:
        raise ValueError("Inconsistent source test/result ROW_SOURCE_ID")
    v = v.loc[
        v.test_row_source_id_test.eq(_EIMS)
        & v.result_row_source_id.eq(_EIMS)
    ]

    tests_by_day = {
        k: f for k, f in t.groupby(_KEYS, dropna=False)
    }
    results_by_test = {
        k: f for k, f in v.groupby(_TEST_KEYS, dropna=False)
    }
    empty = pd.DataFrame(columns=["result_name", "raw_value", "reported_unit"])
    topology_counts = {(p, status): 0 for p in _PERIODS for status in _TOPOLOGY}
    metric_counts = {
        (p, metric, status): 0
        for p in _PERIODS for metric in _METRICS for status in _EXPLICIT_ML
    }

    for patient in c.itertuples(index=False):
        key = (patient.person_id, patient.index_date)
        period = _period(patient.index_date)
        if period not in _PERIODS:
            continue
        day = tests_by_day.get(key)
        fpk = None
        if day is None or day.empty:
            status = "no_conventional_tte"
        else:
            eims = day.loc[day.test_row_source_id.eq(_EIMS)]
            if eims.empty:
                status = "no_eims_tte"
            elif eims.echo_test_fpk.nunique() > 1:
                status = "multiple_eims_source_tests"
            else:
                fpk = str(eims.echo_test_fpk.iloc[0])
                status = (
                    "single_eims_test_only" if len(day) == 1
                    else "single_eims_test_other_source_present"
                )
        topology_counts[(period, status)] += 1
        if fpk is None:
            continue
        source = results_by_test.get((*key, fpk), empty)
        for metric in _METRICS:
            method_status = _method_feasibility(source, metric)
            label = (
                "no_explicit_ml" if method_status == "no_explicit_ml"
                else "preferred_tier_conflict"
                if method_status == "conflicting_preferred_tier"
                else "explicit_ml_candidate"
            )
            metric_counts[(period, metric, label)] += 1

    exams, volume = [], []
    for period in _PERIODS:
        exam_vals = [topology_counts[(period, s)] for s in _TOPOLOGY]
        eligible = sum(topology_counts[(period, s)] for s in (
            "single_eims_test_only", "single_eims_test_other_source_present"
        ))
        metric_vals = {
            metric: [metric_counts[(period, metric, s)] for s in _EXPLICIT_ML]
            for metric in _METRICS
        }
        for vals in metric_vals.values():
            if sum(vals) != eligible:
                raise ValueError("EIMS method counts do not reconcile to sole-FPK tests")
        # Protect the two tables together: the same eligible denominator
        # is shared by both metrics, and all exam totals reconcile to
        # the fixed saved cohort's period counts. A single small cell
        # suppresses both public tables for that period.
        all_vals = exam_vals + [n for vals in metric_vals.values() for n in vals]
        hide = any(0 < n < small_cell for n in all_vals)
        for status, n in zip(_TOPOLOGY, exam_vals):
            exams.append({
                "period": period, "source_test_category": status,
                "patient_days": "suppressed" if hide else str(n)
            })
        for metric in _METRICS:
            for status, n in zip(_EXPLICIT_ML, metric_vals[metric]):
                volume.append({
                    "period": period, "metric": metric,
                    "availability_category": status,
                    "sole_eims_fpk_patient_days": "suppressed" if hide else str(n)
                })
    return pd.DataFrame(exams), pd.DataFrame(volume)


def run_eims_volume_eligibility(eng, study: dict):
    """MCP-only: no writes or raw-ID export, only disclosure-gated counts."""
    start, end = str(study["index_start"]), str(study["index_end"])
    cohort = eng.read_table("echo_analytic")
    tests = eng.query(eims_test_source_sql(
        study["modality"], start=start, end=end
    ))
    volume = eng.query(source_system_bridge_rows_sql(
        study["modality"], start=start, end=end
    ))
    return summarize_eims_volume_eligibility(
        cohort, tests, volume, small_cell=int(study.get("small_cell", 11)),
    )
