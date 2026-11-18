"""Read-only cohort-scoped echo source discovery for extraction review.

No patient rows, note text, result values, accessions, or test IDs leave SQL.
The saved echo_analytic cohort is read only. These are source availability
signals, not accepted clinical references and not proposed extraction rules.

Uses native BigQuery GoogleSQL through client.query(), not Engine.query()
(which expects Spark SQL and transpiles it).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .mcp_echo import _cfg, _macro, _t


RESULT_PATTERN = (
    "edv|esv|diastolic.{0,25}vol|systolic.{0,25}vol|lv.?vol|"
    "left.{0,12}atri|la.{0,10}(diam|dim|vol)|wall.{0,8}motion|"
    "wmsi|segment|image.{0,8}quality|indication|reason|"
    "height|weight|body surface|bsa"
)


def _window(start: str, end: str) -> None:
    if date.fromisoformat(start) > date.fromisoformat(end):
        raise ValueError("index_start must not exceed index_end")


def _cohort_tte_cte(modality_cfg: dict, *, start: str, end: str) -> str:
    """BigQuery CTE with one eligible conventional index-day test per key.

    Mirrors cohort_type_rows_sql filters, but retains only computed flags for
    ECHO_NOTE. Report contents are never returned to Python.
    """
    _window(start, end)
    c = _cfg(modality_cfg)
    e, d, pm, tt = (c["echo_test_columns"], c["dim_patient"],
                    c["person_map"], c["test_type"])
    note = f"CAST(t.{e['note']} AS STRING)"
    report = f"LOWER(COALESCE({note}, ''))"
    return f"""
cohort AS (
  SELECT DISTINCT person_id, DATE(index_date) AS index_date,
         LENGTH(TRIM(COALESCE(CAST(echo_report AS STRING), ''))) > 0
           AS current_report_present
  FROM {{{{work}}}}.echo_analytic
  WHERE DATE(index_date) BETWEEN DATE('{start}') AND DATE('{end}')
),
source_tests AS (
  SELECT a.person_id, a.index_date,
         CAST(t.{e['test_key']} AS STRING) AS echo_test_fpk,
         LOGICAL_OR(LENGTH(TRIM(COALESCE({note}, ''))) > 0) AS has_echo_note,
         LOGICAL_OR(REGEXP_CONTAINS({report},
             r'wall motion|hypokines|akines|dyskines')) AS has_wall_motion_cue,
         LOGICAL_OR(REGEXP_CONTAINS({report},
             r'image quality|technically difficult|suboptimal|poor (acoustic|image)|limited (study|views|windows|image)'))
             AS has_quality_cue,
         LOGICAL_OR(REGEXP_CONTAINS({report},
             r'indications?:|reason for (study|exam|test)|clinical indication'))
             AS has_indication_cue
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
  GROUP BY 1, 2, 3
)
"""


def cohort_result_types_sql(
    modality_cfg: dict, *, start: str, end: str, small_cell: int = 11
) -> str:
    """Result names observed on THIS cohort's index-day conventional TTEs."""
    if small_cell < 1:
        raise ValueError("small_cell must be positive")
    c = _cfg(modality_cfg)
    r = c["results"]
    rc, ty = r["columns"], r["result_type"]
    cte = _cohort_tte_cte(modality_cfg, start=start, end=end)
    distinct_day = "TO_JSON_STRING(STRUCT(t.person_id, t.index_date))"
    return f"""
WITH {cte}
SELECT CAST(ty.{ty['name']} AS STRING) AS result_name,
       CAST(ty.{ty['description']} AS STRING) AS result_description,
       COUNT(*) AS result_rows,
       COUNT(DISTINCT t.echo_test_fpk) AS source_test_records,
       COUNT(DISTINCT {distinct_day}) AS cohort_patient_days
FROM source_tests t
JOIN {_macro(r['table'])} r
  ON CAST(r.{rc['test_key']} AS STRING) = t.echo_test_fpk
JOIN {_macro(ty['table'])} ty
  ON ty.{ty['key']} = r.{rc['result_type_key']}
WHERE REGEXP_CONTAINS(
  LOWER(CONCAT(COALESCE(CAST(ty.{ty['name']} AS STRING), ''),
    ' ', COALESCE(CAST(ty.{ty['description']} AS STRING), ''))),
  r'{RESULT_PATTERN}')
GROUP BY 1, 2
HAVING COUNT(DISTINCT {distinct_day}) >= {small_cell}
   AND COUNT(DISTINCT t.echo_test_fpk) >= {small_cell}
ORDER BY cohort_patient_days DESC, result_rows DESC
LIMIT 80
"""


def cohort_echo_note_coverage_sql(
    modality_cfg: dict, *, start: str, end: str
) -> str:
    """One aggregate row, no note text or identifier in the result."""
    cte = _cohort_tte_cte(modality_cfg, start=start, end=end)
    return f"""
WITH {cte},
per_day AS (
  SELECT person_id, index_date,
         COUNT(*) AS source_test_records,
         LOGICAL_OR(has_echo_note) AS has_echo_note,
         LOGICAL_OR(has_wall_motion_cue) AS has_wall_motion_cue,
         LOGICAL_OR(has_quality_cue) AS has_quality_cue,
         LOGICAL_OR(has_indication_cue) AS has_indication_cue
  FROM source_tests
  GROUP BY 1, 2
)
SELECT COUNT(*) AS saved_cohort_days,
       COUNTIF(d.source_test_records IS NOT NULL) AS days_with_conventional_tte,
       SUM(COALESCE(d.source_test_records, 0)) AS conventional_test_records,
       COUNTIF(COALESCE(c.current_report_present, FALSE))
           AS days_with_current_extracted_report,
       COUNTIF(COALESCE(d.has_echo_note, FALSE))
           AS days_with_source_echo_note,
       COUNTIF(COALESCE(d.has_wall_motion_cue, FALSE))
           AS days_with_wall_motion_cue,
       COUNTIF(COALESCE(d.has_quality_cue, FALSE))
           AS days_with_quality_cue,
       COUNTIF(COALESCE(d.has_indication_cue, FALSE))
           AS days_with_indication_cue
FROM cohort c
LEFT JOIN per_day d USING (person_id, index_date)
"""


def echo_schema_sql(modality_schema: str) -> str:
    """Metadata only: no data values."""
    return f"""
SELECT table_name, column_name, data_type
FROM {modality_schema}.INFORMATION_SCHEMA.COLUMNS
WHERE table_name IN ('FACT_ECHO_TEST', 'FACT_ECHO_RESULTS',
                     'DIM_ECHO_TEST_RESULT_TYPE')
ORDER BY table_name, ordinal_position
"""


def echo_note_titles_sql(cdm_schema: str, *, small_cell: int = 11) -> str:
    """Source-wide OMOP titles, complementary to FACT_ECHO_TEST.ECHO_NOTE."""
    if small_cell < 1:
        raise ValueError("small_cell must be positive")
    return f"""
SELECT CAST(note_title AS STRING) AS note_title, COUNT(*) AS note_rows
FROM {cdm_schema}.note
WHERE REGEXP_CONTAINS(LOWER(COALESCE(CAST(note_title AS STRING), '')),
  r'echo|\\btte\\b|transthoracic|ultrasound.*heart|cardiac us')
GROUP BY 1
HAVING COUNT(*) >= {small_cell}
ORDER BY note_rows DESC
LIMIT 40
"""


def display_safe_coverage(
    coverage: pd.DataFrame, *, small_cell: int = 11
) -> pd.DataFrame:
    """Suppress exact small and complementary cohort-day counts for sharing.

    This redacted display excludes the source-test-record total because
    suppressing cohorts based on test counts requires independent review.
    No percentages are derived from disclosive small counts.
    """
    columns = [
        "saved_cohort_days", "days_with_conventional_tte",
        "days_with_current_extracted_report", "days_with_source_echo_note",
        "days_with_wall_motion_cue", "days_with_quality_cue",
        "days_with_indication_cue",
    ]
    if len(coverage) != 1 or not set(columns).issubset(coverage):
        raise ValueError("Expected one aggregate coverage row with documented columns")
    if small_cell < 1:
        raise ValueError("small_cell must be positive")
    total = int(coverage.iloc[0].saved_cohort_days)
    rows = [dict(measure="saved_cohort_days",
                 patient_days=(str(total) if total >= small_cell else "suppressed"),
                 pct=(100.0 if total >= small_cell else None))]
    for col in columns[1:]:
        n = int(coverage.iloc[0][col])
        if not 0 <= n <= total:
            raise ValueError(f"Invalid count for {col}")
        safe = (n == 0 or n >= small_cell) and (
            total - n == 0 or total - n >= small_cell)
        rows.append(dict(measure=col,
                         patient_days=str(n) if safe else "suppressed",
                         pct=(round(100 * n / total, 2) if safe and total else None)))
    return pd.DataFrame(rows)


def run_echo_source_discovery(eng, study: dict):
    """Return aggregate result types, schema metadata, titles, safe coverage.

    All SQL is read-only. Print-safe here means only primary suppression;
    apply MCP's additional complementary disclosure rules before exporting.
    """
    cfg = study["modality"]
    start, end = str(study["index_start"]), str(study["index_end"])
    min_cell = int(study.get("small_cell", 11))
    def query(sql):
        return eng.client.query(eng.render(sql, {}), location=eng.location).to_dataframe()

    schema = query(echo_schema_sql(study["schemas"]["modality"]))
    results = query(cohort_result_types_sql(
        cfg, start=start, end=end, small_cell=min_cell))
    titles = query(echo_note_titles_sql(
        study["schemas"]["cdm"], small_cell=min_cell))
    coverage = query(cohort_echo_note_coverage_sql(cfg, start=start, end=end))
    safe_coverage = display_safe_coverage(coverage, small_cell=min_cell)
    return results, schema, titles, safe_coverage
