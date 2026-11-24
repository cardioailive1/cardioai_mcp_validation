"""Read-only cohort-scoped Epic same-encounter height/weight source feasibility.

Mayo (2026-11-20) confirmed for the Epic source system:
  FACT_ECHO_TEST.ENCOUNTER_NUMBER = FACT_FLOWSHEETS.ENCOUNTER_NBR
  = FACT_ENCOUNTERS.EHR_ENCOUNTER_NUMBER
when the readings belong to the same encounter.

Only ONE Epic conventional-TTE source test with one nonblank
patient/encounter identity per 2019+ index-day qualifies for this
preview. Match on PATIENT_DK AND encounter number AND index date; no
patient/date-only fallback. Row label screening is PROVISIONAL:
review DIM_FLOWSHEET_ROW_NAME labels and unit meanings before any
clinical height/weight selection. No source values or patient IDs are
returned to Python: the SQL returns one aggregate row only.

No writes, cohort rebuild, or clinical value/unit inference.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .cohort_source_discovery import _cohort_tte_cte
from .mcp_echo import _cfg, _t, _macro

_EPIC = "47918"
_REPORT_COLUMNS = (
    "cohort_days", "single_epic_encounter_days",
    "height_numeric_days", "weight_numeric_days",
    "baseline_height_missing_days", "baseline_weight_missing_days",
    "height_missing_with_numeric_candidate",
    "weight_missing_with_numeric_candidate",
)


def epic_flowsheet_candidate_sql(modality_cfg: dict, *, start: str, end: str) -> str:
    """Native GoogleSQL, returning only 2019+ patient-day aggregate counts.

    Candidate labels are intentionally broad and should be reviewed
    against the separate nonpatient dictionary before interpretation.
    Excludes rows referring simultaneously to height AND weight.
    Numeric-only screening does not establish a valid clinical value.
    """
    first = max(date.fromisoformat(start), date(2019, 1, 1))
    last = date.fromisoformat(end)
    if last < first:
        raise ValueError("No 2019+ Epic-era dates in configured study")
    c = _cfg(modality_cfg)
    e = c["echo_test_columns"]
    # Reuse the already-tested patient/clinic and conventional TTE join
    # instead of creating a new cross-source patient identity mapping.
    tte = _cohort_tte_cte(
        modality_cfg, start=first.isoformat(), end=last.isoformat()
    )
    return f"""
WITH {tte},
epic_tests AS (
  SELECT DISTINCT s.person_id, s.index_date, s.echo_test_fpk,
         NULLIF(TRIM(CAST(t.{e['patient_dk']} AS STRING)), '') AS patient_dk,
         NULLIF(TRIM(CAST(t.ENCOUNTER_NUMBER AS STRING)), '') AS encounter_number
  FROM source_tests s
  JOIN {_t(c, 'echo_test_table')} t
    ON CAST(t.{e['test_key']} AS STRING) = s.echo_test_fpk
  WHERE TRIM(CAST(t.ROW_SOURCE_ID AS STRING)) = '{_EPIC}'
),
epic_days AS (
  SELECT person_id, index_date,
         COUNT(DISTINCT echo_test_fpk) AS source_test_count,
         COUNT(DISTINCT patient_dk) AS patient_key_count,
         COUNT(DISTINCT encounter_number) AS encounter_count,
         COUNTIF(patient_dk IS NULL OR encounter_number IS NULL) AS missing_identity_rows,
         ANY_VALUE(patient_dk) AS patient_dk,
         ANY_VALUE(encounter_number) AS encounter_number
  FROM epic_tests
  GROUP BY person_id, index_date
),
eligible_days AS (
  SELECT a.person_id, a.index_date,
         a.height IS NULL AS baseline_height_missing,
         a.weight IS NULL AS baseline_weight_missing,
         CASE
           WHEN e.source_test_count = 1
             AND e.patient_key_count = 1
             AND e.encounter_count = 1
             AND e.missing_identity_rows = 0
           THEN TRUE ELSE FALSE
         END AS unique_epic_encounter,
         e.patient_dk, e.encounter_number
  FROM {{{{work}}}}.echo_analytic a
  LEFT JOIN epic_days e
    ON e.person_id = a.person_id
   AND e.index_date = DATE(a.index_date)
  WHERE DATE(a.index_date)
    BETWEEN DATE('{first.isoformat()}') AND DATE('{last.isoformat()}')
),
row_labels AS (
  SELECT CAST(FLOWSHEET_ROW_NAME_DK AS STRING) AS row_dk,
         CASE
           WHEN REGEXP_CONTAINS(
             LOWER(CONCAT(
               COALESCE(CAST(FLOWSHEET_ROW_CODE AS STRING), ''), ' ',
               COALESCE(CAST(FLOWSHEET_ROW_DESCRIPTION AS STRING), '')
             )), r'height|stature|(^|[^a-z])ht([^a-z]|$)'
           ) AND NOT REGEXP_CONTAINS(
             LOWER(CONCAT(
               COALESCE(CAST(FLOWSHEET_ROW_CODE AS STRING), ''), ' ',
               COALESCE(CAST(FLOWSHEET_ROW_DESCRIPTION AS STRING), '')
             )), r'weight|weigh|(^|[^a-z])wt([^a-z]|$)'
           ) THEN 'height'
           WHEN REGEXP_CONTAINS(
             LOWER(CONCAT(
               COALESCE(CAST(FLOWSHEET_ROW_CODE AS STRING), ''), ' ',
               COALESCE(CAST(FLOWSHEET_ROW_DESCRIPTION AS STRING), '')
             )), r'weight|weigh|(^|[^a-z])wt([^a-z]|$)'
           ) AND NOT REGEXP_CONTAINS(
             LOWER(CONCAT(
               COALESCE(CAST(FLOWSHEET_ROW_CODE AS STRING), ''), ' ',
               COALESCE(CAST(FLOWSHEET_ROW_DESCRIPTION AS STRING), '')
             )), r'height|stature|(^|[^a-z])ht([^a-z]|$)'
           ) THEN 'weight'
           ELSE NULL
         END AS possible_vital
  FROM {_macro(c['schema'])}.DIM_FLOWSHEET_ROW_NAME
),
candidate_rows AS (
  SELECT e.person_id, e.index_date, r.possible_vital,
         SAFE_CAST(TRIM(CAST(f.FLOWSHEET_RESULT_TXT AS STRING)) AS FLOAT64)
            AS numeric_result
  FROM eligible_days e
  JOIN {_macro(c['schema'])}.FACT_FLOWSHEETS f
    ON e.unique_epic_encounter
   AND CAST(f.PATIENT_DK AS STRING) = e.patient_dk
   AND CAST(f.ENCOUNTER_NBR AS STRING) = e.encounter_number
   AND DATE(f.FLOWSHEET_ASSESSMENT_DTM) = DATE(e.index_date)
  JOIN row_labels r
    ON CAST(f.FLOWSHEET_ROW_NAME_DK AS STRING) = r.row_dk
  WHERE r.possible_vital IS NOT NULL
),
day_flags AS (
  SELECT person_id, index_date,
         LOGICAL_OR(possible_vital = 'height' AND numeric_result > 0)
             AS height_numeric,
         LOGICAL_OR(possible_vital = 'weight' AND numeric_result > 0)
             AS weight_numeric
  FROM candidate_rows
  GROUP BY person_id, index_date
)
SELECT COUNT(*) AS cohort_days,
       COUNTIF(e.unique_epic_encounter) AS single_epic_encounter_days,
       COUNTIF(e.unique_epic_encounter
           AND COALESCE(f.height_numeric, FALSE)) AS height_numeric_days,
       COUNTIF(e.unique_epic_encounter
           AND COALESCE(f.weight_numeric, FALSE)) AS weight_numeric_days,
       COUNTIF(e.baseline_height_missing) AS baseline_height_missing_days,
       COUNTIF(e.baseline_weight_missing) AS baseline_weight_missing_days,
       COUNTIF(e.unique_epic_encounter AND e.baseline_height_missing
           AND COALESCE(f.height_numeric, FALSE))
           AS height_missing_with_numeric_candidate,
       COUNTIF(e.unique_epic_encounter AND e.baseline_weight_missing
           AND COALESCE(f.weight_numeric, FALSE))
           AS weight_missing_with_numeric_candidate
FROM eligible_days e
LEFT JOIN day_flags f USING (person_id, index_date)
"""


def suppress_epic_flowsheet_counts(
    raw: pd.DataFrame, *, small_cell: int = 11
) -> pd.DataFrame:
    """Protect cohort and all nested source/missingness count complements.

    All displayed values are suppressed if any positive count *or its
    natural complement* within the nested binary partitions is < threshold.
    Also review cross-report disclosure against prior cohort figures.
    """
    if small_cell < 1:
        raise ValueError("small_cell must be positive")
    if len(raw) != 1 or not set(_REPORT_COLUMNS).issubset(raw):
        raise ValueError("Expected one cohort-scoped flowsheet aggregate row")
    values = {key: int(raw.iloc[0][key]) for key in _REPORT_COLUMNS}
    n = values["cohort_days"]
    e = values["single_epic_encounter_days"]
    h = values["height_numeric_days"]
    w = values["weight_numeric_days"]
    bh = values["baseline_height_missing_days"]
    bw = values["baseline_weight_missing_days"]
    nh = values["height_missing_with_numeric_candidate"]
    nw = values["weight_missing_with_numeric_candidate"]
    partitions = [
        (e, n - e),
        (h, e - h),
        (w, e - w),
        (bh, n - bh),
        (bw, n - bw),
        (nh, bh - nh, h - nh, n - bh - h + nh),
        (nw, bw - nw, w - nw, n - bw - w + nw),
    ]
    if n < 1 or any(x < 0 for parts in partitions for x in parts):
        raise ValueError("Impossible cohort or nested candidate counts")
    if any(sum(parts) != total for parts, total in [
        (partitions[0], n), (partitions[1], e), (partitions[2], e),
        (partitions[3], n), (partitions[4], n),
        (partitions[5], n), (partitions[6], n),
    ]):
        raise ValueError("Cohort source candidate counts fail reconciliation")
    hidden = any(
        0 < item < small_cell for parts in partitions for item in parts
    )
    # We never print source units or individual values. These are
    # possible numeric values identified using unapproved row labels.
    return pd.DataFrame([
        {"measure": col,
         "patient_days": "suppressed" if hidden else str(values[col])}
        for col in _REPORT_COLUMNS
    ])


def run_epic_flowsheet_feasibility(eng, study: dict) -> pd.DataFrame:
    """Run native BigQuery read-only SQL; return disclosure-gated counts."""
    sql = epic_flowsheet_candidate_sql(
        study["modality"], start=study["index_start"], end=study["index_end"]
    )
    result = eng.client.query(
        eng.render(sql, {}), location=eng.location
    ).to_dataframe()
    return suppress_epic_flowsheet_counts(
        result, small_cell=int(study.get("small_cell", 11))
    )
