"""Read-only same-FPK echo TEST versus RESULTS identifier consistency audit.

Distinct test keys with MC CV and explicit-mL results often have different
encounters. Before assuming they represent independent exams, check whether
FACT_ECHO_RESULTS identifiers agree with FACT_ECHO_TEST identifiers *on the
same ECHO_TEST_FPK*. No patient-level source data may leave MCP. This module
does not repair, merge, or infer units for any records.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cohort_extraction_preview import _KEYS, _RANGE
from .exam_resolution import tte_record_rows_sql
from .mcp_echo import _cfg, _macro, _t
from .unitless_scale_validation import (
    _COMPARATORS, _EXAM_KEYS, _ML, _PERIODS, _UNITLESS_NAMES, _period
)

_KEY_TYPES = ("encounter", "order")
_STATUSES = (
    "equal_nonblank",
    "unequal_nonblank",
    "both_missing",
    "only_one_missing",
    "mixed_evidence",
)


def source_echo_result_key_rows_sql(modality_cfg: dict, *, start: str, end: str) -> str:
    """Read-only source rows, restricted to existing cohort's same-day TTEs."""
    c = _cfg(modality_cfg)
    e = c["echo_test_columns"]
    r = c["results"]
    rc, ty = r["columns"], r["result_type"]
    wanted = [
        *_UNITLESS_NAMES.values(),
        *(n for methods in _COMPARATORS.values() for n in methods.values()),
    ]
    names = ", ".join("'" + n.replace("'", "''") + "'" for n in wanted)
    return f"""
WITH tte AS (
  {tte_record_rows_sql(modality_cfg, start=start, end=end)}
)
SELECT tte.person_id, tte.index_date, tte.echo_test_fpk,
       CAST(ty.{ty['name']} AS STRING) AS result_name,
       CAST(x.{rc['value']} AS STRING) AS raw_value,
       CAST(x.ECHO_RESULTS_VALUE_UNIT_OF_MEASURE AS STRING) AS reported_unit,
       CAST(t.ENCOUNTER_NUMBER AS STRING) AS test_encounter,
       CAST(x.ENCOUNTER_NUMBER AS STRING) AS result_encounter,
       CAST(t.ORDER_PROC_ID AS STRING) AS test_order,
       CAST(x.ORDER_PROC_ID AS STRING) AS result_order
FROM tte
JOIN {_t(c, "echo_test_table")} t
  ON CAST(t.{e['test_key']} AS STRING) = tte.echo_test_fpk
JOIN {_macro(r['table'])} x
  ON CAST(x.{rc['test_key']} AS STRING) = tte.echo_test_fpk
JOIN {_macro(ty['table'])} ty
  ON ty.{ty['key']} = x.{rc['result_type_key']}
WHERE ty.{ty['name']} IN ({names})
"""


def _identifier(value) -> str | None:
    if pd.isna(value):
        return None
    val = str(value).strip()
    return val if val else None


def _relationship(left, right) -> str:
    a, b = _identifier(left), _identifier(right)
    if a is None and b is None:
        return "both_missing"
    if a is None or b is None:
        return "only_one_missing"
    return "equal_nonblank" if a == b else "unequal_nonblank"


def summarize_source_key_consistency(
    rows: pd.DataFrame, *, small_cell: int = 11
) -> pd.DataFrame:
    """Aggregate unique patient-days per volume-family and source key.

    More than one distinct status over results and/or multiple same-family
    source records on a patient-day is classified as mixed_evidence instead
    of silently preferring a matching result.

    Each metric/period/family/key_type partition is fully suppressed if
    any positive status cell is < small_cell. Cumulative and cross-report
    disclosure still requires independent review prior to sharing.
    """
    if small_cell < 1:
        raise ValueError("small_cell must be positive")
    required = set(_EXAM_KEYS + [
        "result_name", "raw_value", "reported_unit",
        "test_encounter", "result_encounter", "test_order", "result_order"
    ])
    if not required.issubset(rows):
        raise ValueError("Missing source echo test/result comparison columns")
    d = rows.copy()
    d.index_date = pd.to_datetime(d.index_date).dt.date
    d.echo_test_fpk = d.echo_test_fpk.astype("string").str.strip()
    if d.echo_test_fpk.isna().any() or d.echo_test_fpk.eq("").any():
        raise ValueError("Missing exact source ECHO_TEST_FPK")

    names = {
        name: (metric, "mc_cv_unitless")
        for metric, name in _UNITLESS_NAMES.items()
    }
    for metric, methods in _COMPARATORS.items():
        for name in methods.values():
            names[name] = (metric, "explicit_ml")
    d = d.loc[d.result_name.isin(names)].copy()
    d["metric"] = d.result_name.map(lambda x: names[x][0])
    d["family"] = d.result_name.map(lambda x: names[x][1])
    d["number"] = pd.to_numeric(d.raw_value, errors="coerce")
    d = d.loc[np.isfinite(d.number) & d.number.gt(0)].copy()
    units = d.reported_unit.fillna("").astype(str).str.strip().str.lower()
    lower = d.metric.map(lambda x: _RANGE[x][0])
    upper = d.metric.map(lambda x: _RANGE[x][1])
    eligible_mc = d.family.eq("mc_cv_unitless") & units.eq("")
    eligible_ml = (
        d.family.eq("explicit_ml") & units.isin(_ML)
        & d.number.ge(lower) & d.number.le(upper)
    )
    d = d.loc[eligible_mc | eligible_ml].copy()
    d["period"] = d.index_date.map(_period)
    d = d.loc[d.period.isin(_PERIODS)].copy()
    for key in _KEY_TYPES:
        d[key + "_state"] = [
            _relationship(a, b) for a, b in
            zip(d["test_" + key], d["result_" + key])
        ]

    statuses = {}
    for group, chunk in d.groupby(
        _KEYS + ["metric", "period", "family"], dropna=False
    ):
        for key in _KEY_TYPES:
            distinct = set(chunk[key + "_state"])
            statuses[(*group, key)] = (
                next(iter(distinct)) if len(distinct) == 1 else "mixed_evidence"
            )

    public_rows = []
    for metric in _UNITLESS_NAMES:
        for period in _PERIODS:
            for family in ("mc_cv_unitless", "explicit_ml"):
                for key in _KEY_TYPES:
                    observed = [
                        state for (pid, day, m, p, f, k), state
                        in statuses.items()
                        if (m, p, f, k) == (metric, period, family, key)
                    ]
                    counts = pd.Series(observed, dtype="string").value_counts()
                    values = [int(counts.get(status, 0)) for status in _STATUSES]
                    if sum(values) != len(observed):
                        raise ValueError("Source key statuses do not reconcile")
                    hide = any(0 < val < small_cell for val in values)
                    for status, number in zip(_STATUSES, values):
                        public_rows.append({
                            "metric": metric, "period": period,
                            "family": family, "key": key,
                            "status": status,
                            "patient_days": (
                                "suppressed" if hide else str(number)
                            ),
                        })
    return pd.DataFrame(public_rows)


def run_source_echo_key_consistency(eng, study: dict) -> pd.DataFrame:
    """Queries stay inside Mayo MCP; return only disclosure-gated aggregates."""
    start, end = str(study["index_start"]), str(study["index_end"])
    rows = eng.query(source_echo_result_key_rows_sql(
        study["modality"], start=start, end=end
    ))
    return summarize_source_key_consistency(
        rows, small_cell=int(study.get("small_cell", 11))
    )
