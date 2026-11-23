"""Read-only Mayo source provenance bridge for the two LV volume families.

FACT_ECHO_TEST and FACT_ECHO_RESULTS expose ROW_SOURCE_ID but also
SOURCE_SYSTEM_DK/KEY. DIM_SOURCE_SYSTEM has names/descriptions keyed by DK.
The numerical ROW_SOURCE_ID is NOT assumed to equal SOURCE_SYSTEM_DK.

Summarize only stable source provenance (test/result ROW_SOURCE_ID and
test SOURCE_SYSTEM_DK), not high-cardinality SOURCE_SYSTEM_KEY values,
which fragmented previous results into effectively one row per patient-day.
Attach a dimension name ONLY through the TEST's actual SOURCE_SYSTEM_DK.
Do not use source IDs, common dates, or source names as an exam crosswalk.
All clinical rows and identifiers remain inside the restricted Mayo PC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cohort_extraction_preview import _KEYS, _RANGE
from .exam_resolution import tte_record_rows_sql
from .mcp_echo import _cfg, _macro, _t
from .unitless_scale_validation import (
    _COMPARATORS, _EXAM_KEYS, _ML, _PERIODS, _UNITLESS_NAMES, _period,
)

_KEY_COLUMNS = (
    "test_row_source_id", "result_row_source_id", "test_source_system_dk",
)
_FAMILY_ORDER = ("mc_cv_unitless", "explicit_ml")
_RESULT_COLUMNS = ("metric", "period", "family", *_KEY_COLUMNS,
                   "test_source_system_name", "patient_days")


def source_system_bridge_rows_sql(
    modality_cfg: dict, *, start: str, end: str,
) -> str:
    """Fetch relevant source keys for known echo volume values (read-only)."""
    c = _cfg(modality_cfg)
    e = c["echo_test_columns"]
    r = c["results"]
    rc, ty = r["columns"], r["result_type"]
    names = [
        *_UNITLESS_NAMES.values(),
        *(name for methods in _COMPARATORS.values() for name in methods.values()),
    ]
    quoted = ", ".join("'" + s.replace("'", "''") + "'" for s in names)
    return f"""
WITH tte AS (
  {tte_record_rows_sql(modality_cfg, start=start, end=end)}
)
SELECT tte.person_id, tte.index_date, tte.echo_test_fpk,
       CAST(ty.{ty['name']} AS STRING) AS result_name,
       CAST(x.{rc['value']} AS STRING) AS raw_value,
       CAST(x.ECHO_RESULTS_VALUE_UNIT_OF_MEASURE AS STRING) AS reported_unit,
       CAST(t.ROW_SOURCE_ID AS STRING) AS test_row_source_id,
       CAST(x.ROW_SOURCE_ID AS STRING) AS result_row_source_id,
       CAST(t.SOURCE_SYSTEM_DK AS STRING) AS test_source_system_dk
FROM tte
JOIN {_t(c, "echo_test_table")} t
  ON CAST(t.{e['test_key']} AS STRING) = tte.echo_test_fpk
JOIN {_macro(r['table'])} x
  ON CAST(x.{rc['test_key']} AS STRING) = tte.echo_test_fpk
JOIN {_macro(ty['table'])} ty
  ON ty.{ty['key']} = x.{rc['result_type_key']}
WHERE CAST(ty.{ty['name']} AS STRING) IN ({quoted})
"""


def source_system_dimension_sql(modality_cfg: dict) -> str:
    """Return nonpatient Mayo source-system dimension names and definitions."""
    cfg = _cfg(modality_cfg)
    return f"""
SELECT CAST(SOURCE_SYSTEM_DK AS STRING) AS source_system_dk,
       CAST(SOURCE_SYSTEM_CODE AS STRING) AS source_system_code,
       CAST(SOURCE_SYSTEM_NAME AS STRING) AS source_system_name,
       CAST(SOURCE_SYSTEM_DESCRIPTION AS STRING) AS source_system_description,
       CAST(SOURCE_SYSTEM_SITE AS STRING) AS source_system_site
FROM {_macro(cfg['schema'])}.DIM_SOURCE_SYSTEM
"""


def _key(value: object) -> str:
    if pd.isna(value):
        return "(missing)"
    return str(value).strip() or "(missing)"


def summarize_source_system_bridge(
    source_rows: pd.DataFrame,
    source_dim: pd.DataFrame,
    *, small_cell: int = 11,
) -> pd.DataFrame:
    """One stable provenance tuple per patient-day/metric/family.

    Distinct per-record source-system keys are intentionally excluded:
    those long values can vary across results within the same source
    system, leading to hundreds of disclosive suppressed rows.
    If a patient-day has multiple *stable* origin combinations,
    classify it as MULTIPLE_PROVENANCES rather than selecting one.
    Resolve the source-system name using only the TEST source DK.
    Whole-partition suppression protects small primary/complementary
    groups. Cumulative disclosure review is still required.
    """
    if small_cell < 1:
        raise ValueError("small_cell must be positive")
    required = set(_EXAM_KEYS + [
        "result_name", "raw_value", "reported_unit", *_KEY_COLUMNS
    ])
    if not required.issubset(source_rows):
        raise ValueError("Missing echo volume source-system bridge columns")
    required_dim = {"source_system_dk", "source_system_name"}
    if not required_dim.issubset(source_dim):
        raise ValueError("Missing Mayo source-system dimension columns")
    dim = source_dim.copy()
    dim.source_system_dk = dim.source_system_dk.map(_key)
    dim.source_system_name = dim.source_system_name.map(_key)
    dk_to_name = {}
    for dk, chunk in dim.groupby("source_system_dk", dropna=False):
        names = set(chunk.source_system_name)
        dk_to_name[dk] = (
            next(iter(names)) if len(names) == 1 else "AMBIGUOUS_DIMENSION_DK"
        )
    data = source_rows.copy()
    data.index_date = pd.to_datetime(data.index_date).dt.date
    data.echo_test_fpk = data.echo_test_fpk.astype("string").str.strip()
    if data.echo_test_fpk.isna().any() or data.echo_test_fpk.eq("").any():
        raise ValueError("Missing ECHO_TEST_FPK")
    names = {v: (k, "mc_cv_unitless") for k, v in _UNITLESS_NAMES.items()}
    for metric, items in _COMPARATORS.items():
        for v in items.values():
            names[v] = (metric, "explicit_ml")
    data = data.loc[data.result_name.isin(names)].copy()
    data["metric"] = data.result_name.map(lambda n: names[n][0])
    data["family"] = data.result_name.map(lambda n: names[n][1])
    data["number"] = pd.to_numeric(data.raw_value, errors="coerce")
    data = data.loc[np.isfinite(data.number) & data.number.gt(0)].copy()
    u = data.reported_unit.fillna("").astype(str).str.strip().str.lower()
    lo = data.metric.map(lambda m: _RANGE[m][0])
    hi = data.metric.map(lambda m: _RANGE[m][1])
    eligible = (
        (data.family.eq("mc_cv_unitless") & u.eq(""))
        | (data.family.eq("explicit_ml") & u.isin(_ML)
           & data.number.ge(lo) & data.number.le(hi))
    )
    data = data.loc[eligible].copy()
    data["period"] = data.index_date.map(_period)
    data = data.loc[data.period.isin(_PERIODS)].copy()
    for col in _KEY_COLUMNS:
        data[col] = data[col].map(_key)

    days = []
    for group, sub in data.groupby(
        _KEYS + ["metric", "period", "family"], dropna=False
    ):
        tuples = set(tuple(row) for row in sub[list(_KEY_COLUMNS)].itertuples(
            index=False, name=None
        ))
        if len(tuples) != 1:
            code = ("MULTIPLE_PROVENANCES",) * len(_KEY_COLUMNS)
            name = "not_applicable"
        else:
            code = next(iter(tuples))
            dk = code[2]
            name = (
                dk_to_name.get(dk, "UNKNOWN_DIMENSION_DK")
                if dk != "(missing)" else "UNKNOWN_DIMENSION_DK"
            )
        days.append({
            "metric": group[2], "period": group[3], "family": group[4],
            **dict(zip(_KEY_COLUMNS, code)),
            "test_source_system_name": name
        })
    cases = pd.DataFrame(days, columns=list(_RESULT_COLUMNS)[:-1])
    result = []
    for metric in _UNITLESS_NAMES:
        for period in _PERIODS:
            for family in _FAMILY_ORDER:
                matching = cases.loc[
                    cases.metric.eq(metric) & cases.period.eq(period)
                    & cases.family.eq(family)
                ]
                if matching.empty:
                    result.append({
                        "metric": metric, "period": period,
                        "family": family,
                        **{c: "(none)" for c in _KEY_COLUMNS},
                        "test_source_system_name": "not_applicable",
                        "patient_days": "0",
                    })
                    continue
                counts = matching.value_counts(
                    list(_KEY_COLUMNS) + ["test_source_system_name"],
                    dropna=False
                )
                hide = any(0 < int(n) < small_cell for n in counts.values)
                for fields, amount in counts.items():
                    result.append({
                        "metric": metric, "period": period, "family": family,
                        **dict(zip(_KEY_COLUMNS, fields[:-1])),
                        "test_source_system_name": fields[-1],
                        "patient_days": (
                            "suppressed" if hide else str(int(amount))
                        ),
                    })
    return pd.DataFrame(result, columns=_RESULT_COLUMNS)


def run_source_system_bridge(eng, study: dict) -> pd.DataFrame:
    """Run inside MCP only; output counts and nonpatient source labels."""
    start, end = str(study["index_start"]), str(study["index_end"])
    raw = eng.query(source_system_bridge_rows_sql(
        study["modality"], start=start, end=end
    ))
    dimension = eng.query(source_system_dimension_sql(study["modality"]))
    return summarize_source_system_bridge(
        raw, dimension, small_cell=int(study.get("small_cell", 11))
    )
