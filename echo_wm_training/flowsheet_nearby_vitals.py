"""Read-only Mayo flowsheet height/weight recovery feasibility near echo dates.

2026-11-24: Same-Epic-encounter audit on 600 index-days (2019–23)
found no candidate measurements among baseline-missing height (41) and
weight (38), despite broad label matching. The DIM contains many
unrelated rows (exercise, diet, calculations, diapers, questionnaire).

This module checks only direct height/weight candidate row labels and
only baseline-missing patient/index-days. It uses the ORIGINAL study
windows: height [-1095,+30], weight [-90,+30] days, restricted
to Epic-era source dates >=2019-01-01. The patient identity
crosswalk must resolve to EXACTLY ONE Mayo PATIENT_DK. The encounter
need not match the echo (this is an explicit separate-encounter,
date-bounded sensitivity analysis Mayo suggested, not accepted
clinical measurement selection). No patient-level records or source
values leave BigQuery. No writes or target/window modifications.

Clinical units, value plausibility, duplicate readings, and source
workflow provenance are NOT adjudicated; a positive numeric string
is only a candidate signal.
"""
from __future__ import annotations

from datetime import date
import re

import pandas as pd

from .mcp_echo import _cfg, _macro

# Strict whole-description labels observed in the Mayo DIM inventory.
# Do not infer clinical meaning from substring matches like
# 'Calculation Weight for Calories' or 'Standing Height of Machinery'.
_HEIGHT_DESCRIPTION = (
    r"^(?:actual height|measured height|current height|height|"
    r"height/length|admit height|admission height(?:/length)?|"
    r"stature|\(ext\)height)"
    r"(?:\s*\((?:cm|m|in|inch|inches|ft)\))?$"
)
_WEIGHT_DESCRIPTION = (
    r"^(?:actual weight|measured weight|clinical weight|"
    r"current weight|admit weight|admission weight|"
    r"admission body weight|daily weight|assessment weight|"
    r"body weight|weight|\(ext\)weight)"
    r"(?:\s*\((?:kg|g|lb|lbs|in lbs)\))?$"
)
_COLUMNS = (
    "cohort_days", "baseline_height_missing",
    "baseline_weight_missing", "height_missing_with_candidate",
    "weight_missing_with_candidate",
)


def classify_direct_vital_label(description: str | None) -> str | None:
    """Pure-Python test mirror of the strict SQL descriptor filter."""
    if description is None:
        return None
    label = re.sub(r"\s+", " ", str(description).strip().lower())
    if re.fullmatch(_HEIGHT_DESCRIPTION, label):
        return "height"
    if re.fullmatch(_WEIGHT_DESCRIPTION, label):
        return "weight"
    return None


def nearby_direct_vital_sql(
    modality_cfg: dict, *, start: str, end: str,
) -> str:
    """Return one nonpatient numeric aggregate row, native BigQuery SQL.

    Only cohort missing values can contribute to a candidate; patient
    crosswalk multiplicity is explicitly excluded, as are non-Epic-era
    flowsheet dates. Existing study windows are NOT changed.
    """
    first = max(date.fromisoformat(start), date(2019, 1, 1))
    last = date.fromisoformat(end)
    if last < first:
        raise ValueError("No Epic-era cohort dates to audit")
    c = _cfg(modality_cfg)
    pm = c["person_map"]
    d = c["dim_patient"]
    schema = _macro(c["schema"])
    pm_table = _macro(pm["table"])
    d_table = _macro(d["table"])
    return f"""
WITH cohort AS (
  SELECT person_id, DATE(index_date) AS index_date,
         height IS NULL AS missing_height,
         weight IS NULL AS missing_weight
  FROM {{{{work}}}}.echo_analytic
  WHERE DATE(index_date) BETWEEN DATE('{first.isoformat()}')
                              AND DATE('{last.isoformat()}')
),
patient_keys AS (
  SELECT a.person_id, a.index_date,
         COUNT(DISTINCT CAST(d.{d['patient_dk']} AS STRING))
           AS distinct_patient_dk_count,
         MIN(NULLIF(TRIM(CAST(d.{d['patient_dk']} AS STRING)), ''))
           AS patient_dk
  FROM cohort a
  JOIN {pm_table} pm
    ON pm.{pm['person_id_column']} = a.person_id
  JOIN {d_table} d
    ON CAST(d.{d['clinic_number']} AS STRING) =
       CAST(pm.{pm['clinic_number_column']} AS STRING)
  WHERE a.missing_height OR a.missing_weight
  GROUP BY a.person_id, a.index_date
),
unique_patient_keys AS (
  SELECT person_id, index_date, patient_dk
  FROM patient_keys
  WHERE distinct_patient_dk_count = 1 AND patient_dk IS NOT NULL
),
direct_rows AS (
  SELECT CAST(FLOWSHEET_ROW_NAME_DK AS STRING) AS row_dk,
         CASE
           WHEN REGEXP_CONTAINS(
             LOWER(TRIM(CAST(FLOWSHEET_ROW_DESCRIPTION AS STRING))),
             r'{_HEIGHT_DESCRIPTION}') THEN 'height'
           WHEN REGEXP_CONTAINS(
             LOWER(TRIM(CAST(FLOWSHEET_ROW_DESCRIPTION AS STRING))),
             r'{_WEIGHT_DESCRIPTION}') THEN 'weight'
           ELSE NULL
         END AS kind
  FROM {schema}.DIM_FLOWSHEET_ROW_NAME
),
unambiguous_rows AS (
  SELECT row_dk, MIN(kind) AS kind
  FROM direct_rows
  WHERE kind IS NOT NULL
  GROUP BY row_dk
  HAVING COUNT(DISTINCT kind) = 1 AND COUNTIF(kind IS NULL) = 0
),
candidate AS (
  SELECT a.person_id, a.index_date, r.kind
  FROM cohort a
  JOIN unique_patient_keys k
    ON k.person_id = a.person_id AND k.index_date = a.index_date
  JOIN {schema}.FACT_FLOWSHEETS f
    ON CAST(f.PATIENT_DK AS STRING) = k.patient_dk
   AND DATE(f.FLOWSHEET_ASSESSMENT_DTM) >= DATE('2019-01-01')
   AND DATE(f.FLOWSHEET_ASSESSMENT_DTM) BETWEEN
       DATE_SUB(a.index_date, INTERVAL 1095 DAY)
       AND DATE_ADD(a.index_date, INTERVAL 30 DAY)
  JOIN unambiguous_rows r
    ON CAST(f.FLOWSHEET_ROW_NAME_DK AS STRING) = r.row_dk
  WHERE (
       (a.missing_height AND r.kind = 'height'
          AND DATE_DIFF(DATE(f.FLOWSHEET_ASSESSMENT_DTM),
                        a.index_date, DAY) BETWEEN -1095 AND 30)
       OR
       (a.missing_weight AND r.kind = 'weight'
          AND DATE_DIFF(DATE(f.FLOWSHEET_ASSESSMENT_DTM),
                        a.index_date, DAY) BETWEEN -90 AND 30)
  )
  AND SAFE_CAST(TRIM(CAST(f.FLOWSHEET_RESULT_TXT AS STRING)) AS FLOAT64) > 0
  GROUP BY a.person_id, a.index_date, r.kind
),
candidate_days AS (
  SELECT person_id, index_date,
         LOGICAL_OR(kind = 'height') AS candidate_height,
         LOGICAL_OR(kind = 'weight') AS candidate_weight
  FROM candidate
  GROUP BY person_id, index_date
)
SELECT COUNT(*) AS cohort_days,
       COUNTIF(a.missing_height) AS baseline_height_missing,
       COUNTIF(a.missing_weight) AS baseline_weight_missing,
       COUNTIF(a.missing_height AND COALESCE(c.candidate_height, FALSE))
         AS height_missing_with_candidate,
       COUNTIF(a.missing_weight AND COALESCE(c.candidate_weight, FALSE))
         AS weight_missing_with_candidate
FROM cohort a
LEFT JOIN candidate_days c USING (person_id, index_date)
"""


def protect_nearby_direct_vital_counts(
    raw: pd.DataFrame, *, small_cell: int = 11,
) -> pd.DataFrame:
    """Joint complementary suppression including missing/no-candidate.

    Cross-report release still requires cumulative disclosure review.
    """
    if small_cell < 1:
        raise ValueError("small_cell must be positive")
    if len(raw) != 1 or not set(_COLUMNS).issubset(raw):
        raise ValueError("Expected one cohort aggregate row")
    v = {c: int(raw.iloc[0][c]) for c in _COLUMNS}
    n = v["cohort_days"]
    hm, wm = v["baseline_height_missing"], v["baseline_weight_missing"]
    hc, wc = (v["height_missing_with_candidate"],
              v["weight_missing_with_candidate"])
    partitions = ((hm, n - hm), (wm, n - wm), (hc, hm - hc),
                  (wc, wm - wc))
    if n < 1 or any(x < 0 for pair in partitions for x in pair):
        raise ValueError("Inconsistent height/weight candidate counts")
    if any(0 < count < small_cell for pair in partitions for count in pair):
        return pd.DataFrame(
            [{"measure": name, "patient_days": "suppressed"}
             for name in _COLUMNS]
        )
    return pd.DataFrame(
        [{"measure": name, "patient_days": str(v[name])}
         for name in _COLUMNS]
    )


def run_nearby_direct_vital_feasibility(eng, study: dict) -> pd.DataFrame:
    """Return only privacy-gated, nonpatient aggregate statistics."""
    sql = nearby_direct_vital_sql(
        study["modality"],
        start=study["index_start"], end=study["index_end"],
    )
    raw = eng.client.query(
        eng.render(sql, {}), location=eng.location
    ).to_dataframe()
    return protect_nearby_direct_vital_counts(
        raw, small_cell=int(study.get("small_cell", 11))
    )
