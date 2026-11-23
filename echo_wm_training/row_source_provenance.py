"""Read-only Mayo ROW_SOURCE_ID exploration on saved echo cohort source tests.

Per Mayo support (2026-11-20), ROW_SOURCE_ID—not the naming prefix—
identifies source provenance. Neither an ID nor a shared encounter proves
two different ECHO_TEST_FPKs represent one clinical examination.

The two outputs contain only source schema metadata and count-gated
patient/index-day totals. Raw patient IDs, source results, echo notes and
examination identifiers stay in the authorized MCP Cloud PC.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .cohort_extraction_preview import _KEYS, _RANGE
from .exam_resolution import tte_record_rows_sql
from .mcp_echo import _cfg, _macro, _t
from .unitless_scale_validation import (
    _COMPARATORS, _EXAM_KEYS, _ML, _PERIODS, _UNITLESS_NAMES, _period,
)

_SCHEMA = re.compile(r"^`?[A-Za-z0-9_-]+`?\.[A-Za-z_][A-Za-z0-9_]*$")
_PUBLIC_COLUMNS = [
    "metric", "period", "family", "source_code", "patient_days",
]
_FAMILIES = ("mc_cv_unitless", "explicit_ml")


def source_id_metadata_sql(schema: str) -> str:
    """Query INFORMATION_SCHEMA only; do not assume where ROW_SOURCE_ID lives."""
    if not _SCHEMA.fullmatch(schema):
        raise ValueError("Expected configured project.dataset for source metadata")
    return f"""
SELECT table_name, column_name, data_type
FROM {schema}.INFORMATION_SCHEMA.COLUMNS
WHERE (
        table_name IN ('FACT_ECHO_TEST', 'FACT_ECHO_RESULTS')
        AND REGEXP_CONTAINS(UPPER(column_name), r'ROW_SOURCE|SOURCE_SYSTEM')
      )
   OR (
        REGEXP_CONTAINS(UPPER(table_name), r'SOURCE')
        AND REGEXP_CONTAINS(UPPER(column_name), r'SOURCE|NAME|DESCRIPTION|CODE')
      )
ORDER BY table_name, column_name
"""


def source_id_fields(metadata: pd.DataFrame) -> tuple[bool, bool]:
    """Return (TEST source column exists, RESULTS source column exists)."""
    required = {"table_name", "column_name"}
    if not required.issubset(metadata):
        raise ValueError("Missing source metadata table/column identifiers")
    t = metadata.table_name.astype(str).str.upper()
    c = metadata.column_name.astype(str).str.upper()
    return (
        bool((t.eq("FACT_ECHO_TEST") & c.eq("ROW_SOURCE_ID")).any()),
        bool((t.eq("FACT_ECHO_RESULTS") & c.eq("ROW_SOURCE_ID")).any()),
    )


def volume_source_rows_sql(
    modality_cfg: dict,
    *,
    start: str,
    end: str,
    test_has_source: bool,
    result_has_source: bool,
) -> str:
    """Read source IDs for named LV volumes on existing cohort TTE test keys."""
    if not (test_has_source or result_has_source):
        raise ValueError("Neither source table has ROW_SOURCE_ID")
    cfg = _cfg(modality_cfg)
    e = cfg["echo_test_columns"]
    r = cfg["results"]
    rc, ty = r["columns"], r["result_type"]
    all_names = [
        *_UNITLESS_NAMES.values(),
        *(name for methods in _COMPARATORS.values() for name in methods.values()),
    ]
    quoted = ", ".join("'" + s.replace("'", "''") + "'" for s in all_names)
    test_source = "CAST(t.ROW_SOURCE_ID AS STRING)" if test_has_source else "CAST(NULL AS STRING)"
    result_source = "CAST(x.ROW_SOURCE_ID AS STRING)" if result_has_source else "CAST(NULL AS STRING)"
    return f"""
WITH tte AS (
  {tte_record_rows_sql(modality_cfg, start=start, end=end)}
)
SELECT tte.person_id, tte.index_date, tte.echo_test_fpk,
       CAST(ty.{ty['name']} AS STRING) AS result_name,
       CAST(x.{rc['value']} AS STRING) AS raw_value,
       CAST(x.ECHO_RESULTS_VALUE_UNIT_OF_MEASURE AS STRING) AS reported_unit,
       {test_source} AS test_source_id,
       {result_source} AS result_source_id
FROM tte
JOIN {_t(cfg, "echo_test_table")} t
  ON CAST(t.{e['test_key']} AS STRING) = tte.echo_test_fpk
JOIN {_macro(r['table'])} x
  ON CAST(x.{rc['test_key']} AS STRING) = tte.echo_test_fpk
JOIN {_macro(ty['table'])} ty
  ON ty.{ty['key']} = x.{rc['result_type_key']}
WHERE CAST(ty.{ty['name']} AS STRING) IN ({quoted})
"""


def _value(value) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    return s or None


def _source_code(test_source, result_source) -> str:
    t, r = _value(test_source), _value(result_source)
    if t and r and t != r:
        # May be distinct *provenance layers*; do not call it an ETL error.
        return "TEST_RESULT_SOURCE_DIFFER"
    if r:
        return "ID:" + r
    if t:
        return "ID:" + t
    return "NO_SOURCE_ID"


def summarize_volume_source_ids(
    rows: pd.DataFrame, *, small_cell: int = 11
) -> pd.DataFrame:
    """Disjoint per-patient-day source-code categories by era/metric/family.

    If several source IDs occur in the same patient-day/family, classify
    that patient-day as MULTIPLE_SOURCE_IDS rather than count it in each.
    Suppress all categories in a partition if any positive category
    would disclose < small_cell patient-days; cross-report review remains
    necessary before transmission outside MCP.
    """
    if small_cell < 1:
        raise ValueError("small_cell must be positive")
    needed = set(_EXAM_KEYS + [
        "result_name", "raw_value", "reported_unit",
        "test_source_id", "result_source_id",
    ])
    if not needed.issubset(rows):
        raise ValueError("Source rows missing required columns")
    d = rows.copy()
    d.index_date = pd.to_datetime(d.index_date).dt.date
    d.echo_test_fpk = d.echo_test_fpk.astype("string").str.strip()
    if d.echo_test_fpk.isna().any() or d.echo_test_fpk.eq("").any():
        raise ValueError("Missing exact ECHO_TEST_FPK")
    names = {name: (metric, "mc_cv_unitless")
             for metric, name in _UNITLESS_NAMES.items()}
    for metric, methods in _COMPARATORS.items():
        for name in methods.values():
            names[name] = (metric, "explicit_ml")
    d = d.loc[d.result_name.isin(names)].copy()
    d["metric"] = d.result_name.map(lambda s: names[s][0])
    d["family"] = d.result_name.map(lambda s: names[s][1])
    d["number"] = pd.to_numeric(d.raw_value, errors="coerce")
    d = d.loc[np.isfinite(d.number) & d.number.gt(0)].copy()
    units = d.reported_unit.fillna("").astype(str).str.strip().str.lower()
    low = d.metric.map(lambda metric: _RANGE[metric][0])
    high = d.metric.map(lambda metric: _RANGE[metric][1])
    is_mc = d.family.eq("mc_cv_unitless") & units.eq("")
    is_ml = (
        d.family.eq("explicit_ml") & units.isin(_ML)
        & d.number.between(low, high)
    )
    d = d.loc[is_mc | is_ml].copy()
    d["period"] = d.index_date.map(_period)
    d = d.loc[d.period.isin(_PERIODS)].copy()
    d["source_code"] = [
        _source_code(t, r)
        for t, r in zip(d.test_source_id, d.result_source_id)
    ]

    day_codes = []
    for group, sub in d.groupby(_KEYS + ["metric", "period", "family"], dropna=False):
        codes = set(sub.source_code)
        day_codes.append({
            "metric": group[2], "period": group[3], "family": group[4],
            "source_code": (
                next(iter(codes)) if len(codes) == 1 else "MULTIPLE_SOURCE_IDS"
            ),
        })
    by_day = pd.DataFrame(
        day_codes, columns=["metric", "period", "family", "source_code"]
    )
    report = []
    for metric in _UNITLESS_NAMES:
        for period in _PERIODS:
            for family in _FAMILIES:
                subset = by_day.loc[
                    by_day.metric.eq(metric) & by_day.period.eq(period)
                    & by_day.family.eq(family)
                ]
                counts = subset.source_code.value_counts()
                # No source IDs observed: zero is not a privacy suppression.
                if counts.empty:
                    report.append({
                        "metric": metric, "period": period,
                        "family": family, "source_code": "NO_OBSERVATIONS",
                        "patient_days": "0",
                    })
                    continue
                hide = any(0 < int(n) < small_cell for n in counts.values)
                for code, count in counts.items():
                    report.append({
                        "metric": metric, "period": period,
                        "family": family, "source_code": str(code),
                        "patient_days": (
                            "suppressed" if hide else str(int(count))
                        ),
                    })
    return pd.DataFrame(report, columns=_PUBLIC_COLUMNS)


def run_volume_source_provenance(eng, study: dict):
    """Return (source metadata, protected volume source counts), no writes."""
    sql = source_id_metadata_sql(study["schemas"]["modality"])
    metadata = eng.client.query(
        eng.render(sql, {}), location=eng.location
    ).to_dataframe()
    has_test, has_result = source_id_fields(metadata)
    if not (has_test or has_result):
        return metadata, pd.DataFrame(columns=_PUBLIC_COLUMNS)
    start, end = str(study["index_start"]), str(study["index_end"])
    values = eng.query(volume_source_rows_sql(
        study["modality"], start=start, end=end,
        test_has_source=has_test, result_has_source=has_result,
    ))
    return metadata, summarize_volume_source_ids(
        values, small_cell=int(study.get("small_cell", 11))
    )
