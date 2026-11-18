"""Cohort-specific TTE LVEDV source availability, read-only.

All record/patient keys stay inside the restricted MCP workspace. This audit
distinguishes (a) raw structured result availability, (b) acceptance into the
existing provisional examination matcher, and (c) no conventional same-day
TTE. Rates by test type are descriptive and NOT clinical eligibility rules.

Only same-day records are included, consistent with unified_reference.py.
ACC and ORD may represent the same clinical examination: source record counts
are not examination or patient counts. Test-type patient-days can overlap.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .exam_results import LVEDV_PRIORITY
from .next_day_lvedv import ADDITIONAL_LVEDV_NAME
from .mcp_echo import _cfg, _macro, _t


KEYS = ["person_id", "index_date"]


def cohort_type_rows_sql(modality_cfg: dict, *, start: str, end: str) -> str:
    """Native BigQuery SQL; ONLY selected cohort's conventional index-day tests.

    Return one row per person/index_date/ECHO_TEST_FPK/test_type. The saved
    analytic cohort is read, never modified. Execute through BigQuery client,
    not Spark-SQL-transpiling engine.query().
    """
    if date.fromisoformat(start) > date.fromisoformat(end):
        raise ValueError("start must not exceed end")
    c = _cfg(modality_cfg)
    e, d, pm, tt, result = (c["echo_test_columns"], c["dim_patient"],
                            c["person_map"], c["test_type"], c["results"])
    rc, ty = result["columns"], result["result_type"]
    names = (*LVEDV_PRIORITY, ADDITIONAL_LVEDV_NAME)
    names_sql = ", ".join("'" + n.replace("'", "''") + "'" for n in names)
    return f"""
WITH cohort AS (
  SELECT DISTINCT person_id, DATE(index_date) AS index_date
  FROM {{{{work}}}}.echo_analytic
  WHERE DATE(index_date) BETWEEN DATE('{start}') AND DATE('{end}')
),
source_tests AS (
  SELECT DISTINCT
    a.person_id, a.index_date,
    CAST(t.{e['test_key']} AS STRING) AS echo_test_fpk,
    CAST(tt.{tt['description']} AS STRING) AS test_type,
    CASE
      WHEN NULLIF(TRIM(CAST(t.{e['link']} AS STRING)), '') IS NOT NULL
        THEN 'ACC'
      WHEN NULLIF(TRIM(CAST(t.ORDER_PROC_ID AS STRING)), '') IS NOT NULL
        THEN 'ORD'
      ELSE 'TEST'
    END AS record_source
  FROM cohort a
  JOIN {_macro(pm['table'])} pm
    ON pm.{pm['person_id_column']} = a.person_id
  JOIN {_macro(d['table'])} d
    ON CAST(d.{d['clinic_number']} AS STRING)
     = CAST(pm.{pm['clinic_number_column']} AS STRING)
  JOIN {_t(c, 'echo_test_table')} t
    ON t.{e['patient_dk']} = d.{d['patient_dk']}
   AND DATE(t.{e['datetime']}) = a.index_date
  JOIN {_macro(tt['table'])} tt
    ON tt.{tt['key']} = t.{e['performed_test_dk']}
  WHERE REGEXP_CONTAINS(
      LOWER(CAST(tt.{tt['description']} AS STRING)),
      r'tte|transthoracic|echocardiogram')
    AND NOT REGEXP_CONTAINS(
      LOWER(CAST(tt.{tt['description']} AS STRING)),
      r'tee|transesophageal|stress|strain|outside|review|fetal')
),
volume_tests AS (
  SELECT CAST(r.{rc['test_key']} AS STRING) AS echo_test_fpk,
         COUNTIF(
           ty.{ty['name']} IN ({names_sql})
           AND SAFE_CAST(TRIM(CAST(r.{rc['value']} AS STRING)) AS FLOAT64)
             BETWEEN 20 AND 500
         ) > 0 AS has_mapped_absolute_lvedv,
         COUNTIF(
           ty.{ty['name']} = '{ADDITIONAL_LVEDV_NAME}'
           AND SAFE_CAST(TRIM(CAST(r.{rc['value']} AS STRING)) AS FLOAT64)
             BETWEEN 20 AND 500
         ) > 0 AS has_mayo_named_lvedv
  FROM {_macro(result['table'])} r
  JOIN {_macro(ty['table'])} ty
    ON ty.{ty['key']} = r.{rc['result_type_key']}
  WHERE ty.{ty['name']} IN ({names_sql})
  GROUP BY 1
)
SELECT t.person_id, t.index_date, t.echo_test_fpk,
       t.test_type, t.record_source,
       COALESCE(v.has_mapped_absolute_lvedv, FALSE) AS has_mapped_lvedv,
       COALESCE(v.has_mayo_named_lvedv, FALSE) AS has_mayo_lvedv
FROM source_tests t
LEFT JOIN volume_tests v USING (echo_test_fpk)
"""


def summarize_cohort_types(
    cohort: pd.DataFrame,
    source_rows: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    small_cell: int = 11,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (private patient-level disposition, overall counts, type counts).

    Source availability is independent of EF-method disagreement and exam
    eligibility. 'candidate_lvedv' refers to a non-null recognized LVEDV
    on at least one provisional single-exam or EF-disagreement candidate,
    not a clinically approved reference.

    A patient-day can appear under >1 test_type. Such rows must not be
    summed to produce cohort population totals. Type rows with small
    counts in any outcome cell are withheld from the printed report.
    """
    c = cohort[KEYS].copy()
    c["index_date"] = pd.to_datetime(c.index_date).dt.date
    if c.duplicated(KEYS).any():
        raise ValueError("Cohort contains duplicate person/index-date keys")
    if small_cell < 1:
        raise ValueError("small_cell must be positive")

    tests = source_rows.copy()
    required = {*KEYS, "echo_test_fpk", "test_type",
                "record_source", "has_mapped_lvedv"}
    if required - set(tests.columns):
        raise ValueError(f"Source data missing columns: {sorted(required - set(tests.columns))}")
    tests["index_date"] = pd.to_datetime(tests.index_date).dt.date
    tests = tests.merge(c, on=KEYS, how="inner", validate="many_to_one")
    tests["has_mapped_lvedv"] = tests.has_mapped_lvedv.fillna(False).astype(bool)
    tests = tests.drop_duplicates(KEYS + ["echo_test_fpk", "test_type"])
    test_days = (tests.groupby(KEYS, as_index=False)
                 .agg(n_source_tests=("echo_test_fpk", "nunique"),
                      source_lvedv=("has_mapped_lvedv", "any")))

    eligible = candidates.copy()
    expected = {*KEYS, "candidate_status", "lvedv"}
    if expected - set(eligible.columns):
        raise ValueError(f"Candidate data missing columns: {sorted(expected - set(eligible.columns))}")
    eligible["index_date"] = pd.to_datetime(eligible.index_date).dt.date
    accepted_status = eligible.candidate_status.isin([
        "single_exam_candidate", "unresolved_method_disagreement"
    ])
    eligible["candidate_lvedv"] = accepted_status & eligible.lvedv.notna()
    matched_days = (eligible.groupby(KEYS, as_index=False)
                    .agg(candidate_lvedv=("candidate_lvedv", "any")))

    disposition = c.merge(test_days, on=KEYS, how="left", validate="one_to_one")
    disposition = disposition.merge(matched_days, on=KEYS, how="left", validate="one_to_one")
    source_present = disposition.n_source_tests.notna()
    disposition["source_lvedv"] = disposition.source_lvedv.fillna(False).astype(bool)
    disposition["candidate_lvedv"] = disposition.candidate_lvedv.fillna(False).astype(bool)
    disposition["coverage_class"] = "no_conventional_tte"
    disposition.loc[source_present, "coverage_class"] = "tte_no_mapped_lvedv"
    disposition.loc[source_present & disposition.source_lvedv,
                    "coverage_class"] = "source_lvedv_not_candidate"
    disposition.loc[disposition.candidate_lvedv,
                    "coverage_class"] = "provisional_candidate_lvedv"
    # A candidate value cannot be counted if it had no same-day source records.
    if (disposition.candidate_lvedv & ~source_present).any():
        raise ValueError("Candidate LVEDV has no matching same-day TTE source records")

    overall = (disposition.coverage_class.value_counts()
               .rename_axis("coverage_class").reset_index(name="patient_days"))
    type_days = (tests.groupby(KEYS + ["test_type"], as_index=False)
                 .agg(source_lvedv=("has_mapped_lvedv", "any"),
                      source_test_records=("echo_test_fpk", "nunique")))
    typed = (type_days.groupby("test_type", as_index=False)
             .agg(patient_days=("person_id", "size"),
                  with_mapped_lvedv=("source_lvedv", "sum"),
                  source_test_records=("source_test_records", "sum")))
    typed["without_mapped_lvedv"] = typed.patient_days - typed.with_mapped_lvedv
    typed["availability_pct"] = (
        100 * typed.with_mapped_lvedv / typed.patient_days).round(2)
    typed = typed.loc[
        (typed.patient_days >= small_cell)
        & (typed.with_mapped_lvedv >= small_cell)
        & (typed.without_mapped_lvedv >= small_cell)
    ].sort_values("patient_days", ascending=False).reset_index(drop=True)
    return disposition, overall, typed


def run_cohort_type_audit(eng, study: dict):
    """Execute diagnostic, return private records and print-safe aggregates."""
    from .unified_reference import build_unified_echo_diagnostics

    start, end = str(study["index_start"]), str(study["index_end"])
    cohort_sql = (f"SELECT person_id, DATE(index_date) AS index_date "
                  f"FROM {{{{work}}}}.echo_analytic "
                  f"WHERE DATE(index_date) BETWEEN DATE('{start}') AND DATE('{end}')")
    cohort = eng.query(eng.render(cohort_sql, {}))
    candidates, _, _ = build_unified_echo_diagnostics(
        eng, study["modality"], start=start, end=end, max_seconds=60,
        log=lambda _: None
    )
    query = eng.render(cohort_type_rows_sql(
        study["modality"], start=start, end=end), {})
    # BigQuery native SQL. Avoid SQLGlot's SAFE_CAST/INTERVAL translation.
    source_rows = eng.client.query(query, location=eng.location).to_dataframe()
    return summarize_cohort_types(
        cohort, source_rows, candidates,
        small_cell=int(study.get("small_cell", 11))
    )
