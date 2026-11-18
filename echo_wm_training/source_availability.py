"""Read-only, source-wide conventional-TTE LVEDV availability (GoogleSQL).

Unlike cohort candidate selection, this query scans conventional TTE source
test records in the configured index-year window and then flags whether each
source record also appears among the saved cohort's same-day TTE source tests.

Units: FACT_ECHO_TEST/ECHO_TEST_FPK records, NOT unique clinical examinations.
ACC/ORD representations can refer to the same clinical exam. No patient-level
data or identifiers are returned to the caller.
"""
from __future__ import annotations

from datetime import date
import pandas as pd

from .exam_results import LVEDV_PRIORITY
from .mcp_echo import _cfg, _macro, _t
from .next_day_lvedv import ADDITIONAL_LVEDV_NAME


def source_availability_sql(modality_cfg: dict, *, start: str, end: str) -> str:
    """Generate native BigQuery SQL returning aggregate test-record counts only.

    Must be executed through BigQuery client directly, NOT Engine.query, because
    the latter treats input as Spark SQL and cannot reliably transpile SAFE_CAST.
    """
    if date.fromisoformat(start) > date.fromisoformat(end):
        raise ValueError("start must not exceed end")
    c = _cfg(modality_cfg)
    e, tt, r = c["echo_test_columns"], c["test_type"], c["results"]
    d, pm = c["dim_patient"], c["person_map"]
    rc, ty = r["columns"], r["result_type"]
    names = (*LVEDV_PRIORITY, ADDITIONAL_LVEDV_NAME)
    quoted = ", ".join("'" + n.replace("'", "''") + "'" for n in names)
    return f"""
WITH all_tte AS (
  SELECT DISTINCT
    CAST(t.{e['test_key']} AS STRING) AS echo_test_fpk,
    t.{e['patient_dk']} AS patient_dk,
    DATE(t.{e['datetime']}) AS exam_date,
    CAST(tt.{tt['description']} AS STRING) AS test_type,
    CASE
      WHEN NULLIF(TRIM(CAST(t.{e['link']} AS STRING)), '') IS NOT NULL THEN 'ACC'
      WHEN NULLIF(TRIM(CAST(t.ORDER_PROC_ID AS STRING)), '') IS NOT NULL THEN 'ORD'
      ELSE 'TEST'
    END AS record_source
  FROM {_t(c, 'echo_test_table')} t
  JOIN {_macro(tt['table'])} tt
    ON tt.{tt['key']} = t.{e['performed_test_dk']}
  WHERE DATE(t.{e['datetime']}) BETWEEN DATE('{start}') AND DATE('{end}')
    AND REGEXP_CONTAINS(LOWER(CAST(tt.{tt['description']} AS STRING)),
                        r'tte|transthoracic|echocardiogram')
    AND NOT REGEXP_CONTAINS(LOWER(CAST(tt.{tt['description']} AS STRING)),
                            r'tee|transesophageal|stress|strain|outside|review|fetal')
),
cohort_tte AS (
  SELECT DISTINCT t.echo_test_fpk
  FROM all_tte t
  JOIN {_macro(d['table'])} d
    ON d.{d['patient_dk']} = t.patient_dk
  JOIN {_macro(pm['table'])} pm
    ON CAST(pm.{pm['clinic_number_column']} AS STRING)
     = CAST(d.{d['clinic_number']} AS STRING)
  JOIN {{{{work}}}}.echo_analytic a
    ON a.person_id = pm.{pm['person_id_column']}
   AND DATE(a.index_date) = t.exam_date
   AND DATE(a.index_date) BETWEEN DATE('{start}') AND DATE('{end}')
),
edv_by_test AS (
  SELECT CAST(r.{rc['test_key']} AS STRING) AS echo_test_fpk,
         COUNTIF(
           ty.{ty['name']} IN ({quoted})
           AND SAFE_CAST(TRIM(CAST(r.{rc['value']} AS STRING)) AS FLOAT64)
               BETWEEN 20 AND 500
         ) > 0 AS has_mapped_absolute_lvedv,
         COUNTIF(
           ty.{ty['name']} = '{ADDITIONAL_LVEDV_NAME}'
           AND SAFE_CAST(TRIM(CAST(r.{rc['value']} AS STRING)) AS FLOAT64)
               BETWEEN 20 AND 500
         ) > 0 AS has_additional_mayo_lvedv
  FROM {_macro(r['table'])} r
  JOIN {_macro(ty['table'])} ty
    ON ty.{ty['key']} = r.{rc['result_type_key']}
  WHERE ty.{ty['name']} IN ({quoted})
  GROUP BY 1
),
classified AS (
  SELECT a.echo_test_fpk, a.test_type, a.record_source,
         c.echo_test_fpk IS NOT NULL AS in_cohort_index_day,
         COALESCE(v.has_mapped_absolute_lvedv, FALSE) AS has_lvedv,
         COALESCE(v.has_additional_mayo_lvedv, FALSE) AS has_mayo_lvedv
  FROM all_tte a
  LEFT JOIN cohort_tte c USING (echo_test_fpk)
  LEFT JOIN edv_by_test v USING (echo_test_fpk)
),
scoped AS (
  SELECT 'all_source' AS scope, record_source, test_type,
         has_lvedv, has_mayo_lvedv FROM classified
  UNION ALL
  SELECT 'cohort_index_day' AS scope, record_source, test_type,
         has_lvedv, has_mayo_lvedv FROM classified
  WHERE in_cohort_index_day
)
SELECT scope, record_source, test_type,
       COUNT(*) AS test_records,
       COUNTIF(has_lvedv) AS records_with_lvedv,
       COUNTIF(NOT has_lvedv) AS records_without_lvedv,
       COUNTIF(has_mayo_lvedv) AS records_with_mayo_named_lvedv
FROM scoped
GROUP BY scope, record_source, test_type
ORDER BY scope, record_source, test_records DESC
"""


def summarise_source_availability(grouped: pd.DataFrame, *, min_cell: int = 11):
    """Aggregate to scope/source without identifying patients.

    Return a printable high-level table and type-specific counts. All output
    represents SOURCE TEST RECORDS, not unique exams/patients. Suppress small
    cells and complementary breakdown detail before sharing externally.
    """
    required = {"scope", "record_source", "test_type", "test_records",
                "records_with_lvedv", "records_without_lvedv"}
    if not required.issubset(grouped.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(grouped.columns))}")
    if grouped.empty:
        return pd.DataFrame(), pd.DataFrame()
    g = grouped.copy()
    for key in ("test_records", "records_with_lvedv", "records_without_lvedv"):
        g[key] = pd.to_numeric(g[key], errors="raise").astype("int64")
    totals = (g.groupby(["scope", "record_source"], as_index=False)
              [["test_records", "records_with_lvedv", "records_without_lvedv"]].sum())
    totals["available_pct"] = (
        totals.records_with_lvedv.div(totals.test_records).mul(100).round(2)
    )
    # Do not output a disclosive type breakdown; all types are local/private
    # unless the site's aggregation disclosure rules explicitly permit it.
    types = (g.groupby(["scope", "test_type"], as_index=False)
             [["test_records", "records_with_lvedv", "records_without_lvedv"]].sum())
    types["available_pct"] = (
        types.records_with_lvedv.div(types.test_records).mul(100).round(2)
    )
    safe_types = types.loc[
        (types.test_records >= min_cell)
        & (types.records_with_lvedv >= min_cell)
        & (types.records_without_lvedv >= min_cell)
    ].copy()
    return totals, safe_types


def run_source_availability_audit(engine, study: dict):
    """Execute a native read-only GoogleSQL aggregate query.

    Returns (by_source, safe_by_test_type) aggregate DataFrames. The work
    table is read but never modified; all detailed analysis stays in MCP.
    """
    sql = engine.render(source_availability_sql(
        study["modality"],
        start=str(study["index_start"]),
        end=str(study["index_end"]),
    ), {})
    # Direct GoogleSQL path bypasses Spark SQLGlot translation.
    rows = engine.client.query(sql, location=engine.location).to_dataframe()
    return summarise_source_availability(
        rows, min_cell=int(study.get("small_cell", 11))
    )
