"""Second-pass read-only echo source feasibility checks.

Uses the saved cohort and conventional same-day TTE test membership.
Returns only aggregates; no report text, result values or identifiers leave SQL.
Candidate results are NOT clinically accepted measurements, and patient/date
matching of OMOP notes is NOT validated examination linkage.
"""
from __future__ import annotations

import pandas as pd

from .cohort_source_discovery import _cohort_tte_cte
from .mcp_echo import _cfg, _macro

# Exact names observed in the MCP cohort discovery; these are source screening
# hypotheses, not a clinically approved hierarchy or interchangeable units.
_RESULT_NAMES = {
    "lvedv": (
        "LV_vol_BP_ed_2D_by_MOD",
        "LV_vol_4C_ed_2D_by_MOD",
        "LV_vol_2C_ed_2D_by_MOD",
        "LV_vol_ed_3D",
        "MC CV LV END DIASTOLIC VOLUME",
    ),
    "lvesv": (
        "LV_vol_BP_es_2D_by_MOD",
        "LV_vol_4C_es_2D_by_MOD",
        "LV_vol_2C_es_2D_by_MOD",
        "LV_vol_es_3D",
        "MC CV LV END SYSTOLIC VOLUME",
    ),
    "la_diameter": ("LA_dim_s_MM", "LA_dim_es_2D"),
    # These two are for source context, NOT substitutes for LA diameter or
    # required height and weight.
    "la_volume_not_diameter": (
        "LA_max_vol_es_2D_by_MOD_BP",
        "LA_max_vol_es_2D_by_MOD_4C",
    ),
    "bsa_not_height_weight": ("Body_Surface_area",),
    "wmsi_not_wall_motion_label": (
        "LV_WMS_index_2D", "MC CV WALL MOTION SCORE INDEX",
    ),
}

# Restrict OMOP source to titles actually seen in the discovery and exclude
# echo stress, transesophageal and unrelated letters. Same-day/+1 candidate
# note availability is exploratory; it cannot establish exam identity.
_NOTE_TITLE = (
    r"echocardiology image exam|echo report text|"
    r"echocardiogram report|hosp echocardiogram|echocardiogram complete"
)
_WALL_CUE = r"wall motion|hypokines|akines|dyskines"
_QUALITY_CUE = (
    r"image quality|technically difficult|suboptimal|"
    r"poor (acoustic|image)|limited (study|views|windows|image)"
)
_INDICATION_CUE = r"indications?:|reason for (study|exam|test)|clinical indication"


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def cohort_numeric_source_sql(
    modality_cfg: dict, *, start: str, end: str
) -> str:
    """Counts unique cohort days with result names and parseable numbers.

    The preliminary 20..500/5..450 ranges for LVEDV/LVESV are diagnostics
    only. LA diameter has no screening-range count pending unit verification.
    """
    c = _cfg(modality_cfg)
    r = c["results"]
    rc, ty = r["columns"], r["result_type"]
    names = {name: metric for metric, kinds in _RESULT_NAMES.items()
             for name in kinds}
    cases = " ".join(
        f"WHEN CAST(ty.{ty['name']} AS STRING) = {_quote(name)} THEN {_quote(metric)}"
        for name, metric in names.items()
    )
    sql_names = ", ".join(_quote(name) for name in names)
    cte = _cohort_tte_cte(modality_cfg, start=start, end=end)
    return f"""
WITH {cte},
values_by_test AS (
  SELECT t.person_id, t.index_date,
         CASE {cases} END AS metric,
         SAFE_CAST(TRIM(CAST(r.{rc['value']} AS STRING)) AS FLOAT64) AS number,
         NULLIF(TRIM(CAST(r.ECHO_RESULTS_VALUE_UNIT_OF_MEASURE AS STRING)), '')
           AS reported_unit
  FROM source_tests t
  JOIN {_macro(r['table'])} r
    ON CAST(r.{rc['test_key']} AS STRING) = t.echo_test_fpk
  JOIN {_macro(ty['table'])} ty
    ON ty.{ty['key']} = r.{rc['result_type_key']}
  WHERE CAST(ty.{ty['name']} AS STRING) IN ({sql_names})
),
per_day AS (
  SELECT person_id, index_date, metric,
         LOGICAL_OR(number IS NOT NULL) AS has_numeric,
         LOGICAL_OR(CASE
           WHEN metric = 'lvedv' THEN number BETWEEN 20 AND 500
           WHEN metric = 'lvesv' THEN number BETWEEN 5 AND 450
           ELSE FALSE END) AS in_preliminary_range
  FROM values_by_test
  GROUP BY 1, 2, 3
)
SELECT metric,
       COUNT(*) AS patient_days_with_named_result,
       COUNTIF(has_numeric) AS patient_days_with_numeric_result,
       COUNTIF(in_preliminary_range) AS patient_days_in_screening_range
FROM per_day
GROUP BY metric
ORDER BY metric
"""


def cohort_la_diameter_unit_sql(
    modality_cfg: dict, *, start: str, end: str, small_cell: int = 11
) -> str:
    """Unit-name metadata with cohort-day counts, suppressed at SQL level."""
    if small_cell < 1:
        raise ValueError("small_cell must be positive")
    c = _cfg(modality_cfg)
    r = c["results"]
    rc, ty = r["columns"], r["result_type"]
    names = ", ".join(_quote(n) for n in _RESULT_NAMES["la_diameter"])
    cte = _cohort_tte_cte(modality_cfg, start=start, end=end)
    distinct_day = "TO_JSON_STRING(STRUCT(t.person_id, t.index_date))"
    return f"""
WITH {cte}
SELECT CAST(ty.{ty['name']} AS STRING) AS result_name,
       COALESCE(NULLIF(TRIM(CAST(r.ECHO_RESULTS_VALUE_UNIT_OF_MEASURE AS STRING)), ''),
                '(missing)') AS reported_unit,
       COUNT(DISTINCT {distinct_day}) AS cohort_patient_days_with_result,
       COUNT(DISTINCT IF(
         SAFE_CAST(TRIM(CAST(r.{rc['value']} AS STRING)) AS FLOAT64) IS NOT NULL,
         {distinct_day}, NULL)) AS cohort_patient_days_with_numeric
FROM source_tests t
JOIN {_macro(r['table'])} r
  ON CAST(r.{rc['test_key']} AS STRING) = t.echo_test_fpk
JOIN {_macro(ty['table'])} ty
  ON ty.{ty['key']} = r.{rc['result_type_key']}
WHERE CAST(ty.{ty['name']} AS STRING) IN ({names})
GROUP BY 1, 2
HAVING cohort_patient_days_with_result >= {small_cell}
   AND (cohort_patient_days_with_numeric = 0
        OR cohort_patient_days_with_numeric >= {small_cell})
   AND (cohort_patient_days_with_result = cohort_patient_days_with_numeric
        OR cohort_patient_days_with_result - cohort_patient_days_with_numeric >= {small_cell})
ORDER BY cohort_patient_days_with_result DESC
"""


def cohort_omop_note_overlap_sql(
    modality_cfg: dict, *, start: str, end: str
) -> str:
    """Candidate OMOP note availability; do not infer clinical report linkage."""
    cte = _cohort_tte_cte(modality_cfg, start=start, end=end)
    return f"""
WITH {cte},
source_per_day AS (
  SELECT person_id, index_date,
         LOGICAL_OR(has_echo_note) AS has_echo_note,
         LOGICAL_OR(has_wall_motion_cue) AS has_wall_motion_cue
  FROM source_tests
  GROUP BY 1, 2
),
omop_note_day AS (
  SELECT a.person_id, a.index_date,
         COUNTIF(LENGTH(TRIM(COALESCE(CAST(n.note_text AS STRING), ''))) > 0)
           > 0 AS has_candidate_note_text,
         COUNTIF(REGEXP_CONTAINS(LOWER(COALESCE(CAST(n.note_text AS STRING), '')),
               r'{_WALL_CUE}')) > 0 AS has_wall_motion_cue,
         COUNTIF(REGEXP_CONTAINS(LOWER(COALESCE(CAST(n.note_text AS STRING), '')),
               r'{_QUALITY_CUE}')) > 0 AS has_quality_cue,
         COUNTIF(REGEXP_CONTAINS(LOWER(COALESCE(CAST(n.note_text AS STRING), '')),
               r'{_INDICATION_CUE}')) > 0 AS has_indication_cue
  FROM cohort a
  JOIN {{{{cdm}}}}.note n
    ON n.person_id = a.person_id
   AND CAST(n.note_date AS DATE)
       BETWEEN a.index_date AND DATE_ADD(a.index_date, INTERVAL 1 DAY)
  WHERE REGEXP_CONTAINS(LOWER(COALESCE(CAST(n.note_title AS STRING), '')),
                        r'{_NOTE_TITLE}')
  GROUP BY 1, 2
)
SELECT COUNT(*) AS saved_cohort_days,
       COUNTIF(COALESCE(c.current_report_present, FALSE))
         AS days_current_report,
       COUNTIF(COALESCE(s.has_echo_note, FALSE)) AS days_source_echo_note,
       COUNTIF(COALESCE(o.has_candidate_note_text, FALSE))
         AS days_omop_echo_candidate_note,
       COUNTIF(
         COALESCE(c.current_report_present, FALSE)
         OR COALESCE(s.has_echo_note, FALSE)
         OR COALESCE(o.has_candidate_note_text, FALSE)
       ) AS days_any_report_text_candidate,
       COUNTIF(
         NOT COALESCE(c.current_report_present, FALSE)
         AND COALESCE(o.has_candidate_note_text, FALSE)
       ) AS days_omop_candidate_not_current,
       COUNTIF(
         COALESCE(s.has_wall_motion_cue, FALSE)
         OR COALESCE(o.has_wall_motion_cue, FALSE)
       ) AS days_source_or_omop_wall_cue,
       COUNTIF(COALESCE(o.has_quality_cue, FALSE))
         AS days_omop_quality_cue,
       COUNTIF(COALESCE(o.has_indication_cue, FALSE))
         AS days_omop_indication_cue
FROM cohort c
LEFT JOIN source_per_day s USING (person_id, index_date)
LEFT JOIN omop_note_day o USING (person_id, index_date)
"""


def suppress_aggregate_counts(
    row: pd.DataFrame, *, small_cell: int = 11
) -> pd.DataFrame:
    """Suppress complementary and pairwise small cells from a one-row audit.

    This is deliberately conservative: if one pair of counts differs by a
    positive cell smaller than the threshold, suppress all counts in the
    affected row, retaining only the measure labels. Further site disclosure
    policy still applies to combinations with other published reports.
    """
    if len(row) != 1 or "saved_cohort_days" not in row.columns:
        raise ValueError("Expected one cohort aggregate row")
    total = int(row.iloc[0].saved_cohort_days)
    values = {col: int(row.iloc[0][col]) for col in row.columns}
    if small_cell < 1 or total < 0 or any(not 0 <= v <= total for v in values.values()):
        raise ValueError("Invalid count or disclosure limit")
    nums = list(values.values())
    pair_small = any(0 < abs(a - b) < small_cell
                     for i, a in enumerate(nums)
                     for b in nums[i + 1:])
    lines = []
    for metric, n in values.items():
        safe = (not pair_small
                and (n == 0 or n >= small_cell)
                and (total - n == 0 or total - n >= small_cell))
        lines.append(dict(metric=metric,
                          patient_days=str(n) if safe else "suppressed",
                          pct=round(n / total * 100, 2) if safe and total else None))
    return pd.DataFrame(lines)


def screen_numeric_counts(
    totals: pd.DataFrame, *, denominator: int, small_cell: int = 11
) -> pd.DataFrame:
    """Suppress metric counts if small cells arise from differences or totals."""
    cols = ["patient_days_with_named_result",
            "patient_days_with_numeric_result",
            "patient_days_in_screening_range"]
    if not {"metric", *cols}.issubset(totals):
        raise ValueError("Numeric coverage totals missing expected columns")
    out = []
    for row in totals.itertuples(index=False):
        values = [int(getattr(row, col)) for col in cols]
        if any(not 0 <= n <= denominator for n in values):
            raise ValueError("Numeric coverage count exceeds cohort")
        all_counts = [denominator, *values]
        safe = all(
            x == 0 or x >= small_cell
            for x in (*values, *(denominator - x for x in values),
                      *(abs(a - b) for i, a in enumerate(all_counts)
                        for b in all_counts[i + 1:]))
        )
        out.append(dict(metric=row.metric,
                        **{col: (str(n) if safe else "suppressed")
                           for col, n in zip(cols, values)}))
    return pd.DataFrame(out)


def run_echo_source_followup(eng, study: dict):
    """Return (safe_numeric_counts, LA diameter units, safe_note_counts)."""
    cfg = study["modality"]
    start, end = str(study["index_start"]), str(study["index_end"])
    small = int(study.get("small_cell", 11))

    def query(sql):
        return eng.client.query(eng.render(sql, {}), location=eng.location).to_dataframe()

    note = query(cohort_omop_note_overlap_sql(cfg, start=start, end=end))
    safe_note = suppress_aggregate_counts(note, small_cell=small)
    n = int(note.iloc[0].saved_cohort_days)
    raw_numeric = query(cohort_numeric_source_sql(cfg, start=start, end=end))
    safe_numeric = screen_numeric_counts(raw_numeric,
                                         denominator=n, small_cell=small)
    la_units = query(cohort_la_diameter_unit_sql(
        cfg, start=start, end=end, small_cell=small))
    return safe_numeric, la_units, safe_note
