"""EIMS echo-report source availability for missing required narrative fields.

Read-only BigQuery aggregate. The existing analytic cohort is unchanged.
A single EIMS ECHO_TEST_FPK on the patient/index-day is the only eligible
source-report candidate. Never join text to a different exam by date alone,
never conflate an EIMS source test with same-day Epic echo, and never
declare clinical labels from regex keyword cues.

All patient identifiers and ECHO_NOTE narrative text stay inside BigQuery:
the SQL returns one aggregate row, then joint primary/complementary
suppression is applied. Cumulative small-cell release review is required.
"""
from __future__ import annotations

import pandas as pd

from .eims_volume_eligibility import _EIMS
from .cohort_source_discovery import _cohort_tte_cte
from .mcp_echo import _cfg, _t

_METRICS = ("wall_motion", "image_quality", "clinical_indication")
# These are *cues*, not valid classification or completeness.
_CUES = {
    "wall_motion": r"wall motion|hypokines|akines|dyskines",
    "image_quality": (
        r"image quality|technically difficult|suboptimal|"
        r"poor (acoustic|image)|limited (study|views|windows|image)"
    ),
    "clinical_indication": (
        r"indications?:|reason for (study|exam|test)|clinical indication"
    ),
}
_REQUIRED = (
    "cohort_days", "sole_eims_test_days", "sole_eims_with_note",
    *[f"baseline_missing_{field}" for field in _METRICS],
    *[f"missing_{field}_with_note" for field in _METRICS],
    *[f"missing_{field}_with_cue" for field in _METRICS],
)


def eims_note_gap_sql(modality_cfg: dict, *, start: str, end: str) -> str:
    """Native GoogleSQL: only aggregate source note existence and cues."""
    cfg = _cfg(modality_cfg)
    ec = cfg["echo_test_columns"]
    q = _cohort_tte_cte(modality_cfg, start=start, end=end)
    base_flags = ",\n         ".join(
        f"{field} IS NULL AS missing_{field}" for field in _METRICS
    )
    cols = [
        "COUNT(*) AS cohort_days",
        "COUNTIF(e.sole_eims_test) AS sole_eims_test_days",
        "COUNTIF(e.sole_eims_test AND e.has_note) AS sole_eims_with_note",
    ]
    for field in _METRICS:
        cols += [
            f"COUNTIF(a.missing_{field}) AS baseline_missing_{field}",
            f"""COUNTIF(a.missing_{field}
                  AND e.sole_eims_test AND e.has_note)
                AS missing_{field}_with_note""",
            f"""COUNTIF(a.missing_{field}
                  AND e.sole_eims_test AND e.has_{field}_cue)
                AS missing_{field}_with_cue""",
        ]
    return f"""
WITH {q},
cohort AS (
  SELECT person_id, DATE(index_date) AS index_date,
         {base_flags}
  FROM {{{{work}}}}.echo_analytic
  WHERE DATE(index_date) BETWEEN DATE('{start}') AND DATE('{end}')
),
source_notes AS (
  -- Reuse the existing native GoogleSQL TTE and note cue
  -- classification from cohort_source_discovery. Its predicates
  -- are not Spark SQL and require no SQLGlot conversion.
  SELECT s.person_id, s.index_date, s.echo_test_fpk,
         TRIM(CAST(t.ROW_SOURCE_ID AS STRING)) AS test_row_source_id,
         s.has_echo_note AS has_note,
         s.has_wall_motion_cue,
         s.has_quality_cue AS has_image_quality_cue,
         s.has_indication_cue AS has_clinical_indication_cue
  FROM source_tests s
  JOIN {_t(cfg, "echo_test_table")} t
    ON CAST(t.{ec['test_key']} AS STRING) = s.echo_test_fpk
),
eims_per_day AS (
  SELECT person_id, index_date,
         COUNT(DISTINCT IF(test_row_source_id = '{_EIMS}',
                           echo_test_fpk, NULL)) = 1 AS sole_eims_test,
         LOGICAL_OR(test_row_source_id = '{_EIMS}' AND has_note) AS has_note,
         {", ".join(
             "LOGICAL_OR(test_row_source_id = '" + _EIMS
             + "' AND has_" + field + "_cue) AS has_" + field + "_cue"
             for field in _METRICS
         )}
  FROM source_notes
  GROUP BY person_id, index_date
)
SELECT {", ".join(cols)}
FROM cohort a
LEFT JOIN eims_per_day e USING (person_id, index_date)
"""


def protect_eims_note_gap_counts(
    raw: pd.DataFrame, *, small_cell: int = 11
) -> pd.DataFrame:
    """Suppress entire report if any nested intersection/complement is small.

    The 'with_cue' group must be nested within the 'with_note' group.
    This is source *feasibility*, not wall-motion, image quality or
    indication acceptance, and no clinical note text is returned.
    """
    if small_cell < 1:
        raise ValueError("small_cell must be positive")
    if len(raw) != 1 or not set(_REQUIRED).issubset(raw):
        raise ValueError("Expected one EIMS note-gap aggregate row")
    v = {key: int(raw.iloc[0][key]) for key in _REQUIRED}
    n, e, t = (v["cohort_days"], v["sole_eims_test_days"],
               v["sole_eims_with_note"])
    parts = [(e, n-e), (t, e-t)]
    if n < 1:
        raise ValueError("Invalid EIMS cohort count")
    for field in _METRICS:
        missing = v[f"baseline_missing_{field}"]
        with_text = v[f"missing_{field}_with_note"]
        with_cue = v[f"missing_{field}_with_cue"]
        # Intersection among (missing yes/no) and (source-note yes/no).
        # Include constraints vs total with EIMS source and note.
        parts.extend([
            (missing, n - missing),
            (with_text, missing - with_text,
             t - with_text, n - missing - t + with_text),
            (with_cue, with_text - with_cue),
        ])
    if any(x < 0 for part in parts for x in part):
        raise ValueError("Impossible EIMS report overlap counts")
    if any(0 < x < small_cell for part in parts for x in part):
        return pd.DataFrame([
            {"measure": key, "patient_days": "suppressed"}
            for key in _REQUIRED
        ])
    return pd.DataFrame([
        {"measure": key, "patient_days": str(v[key])}
        for key in _REQUIRED
    ])


def run_eims_note_gap_audit(eng, study: dict) -> pd.DataFrame:
    """Run source query inside MCP; return only suppressed aggregate counts."""
    sql = eims_note_gap_sql(
        study["modality"],
        start=str(study["index_start"]), end=str(study["index_end"]),
    )
    raw = eng.client.query(
        eng.render(sql, {}), location=eng.location
    ).to_dataframe()
    return protect_eims_note_gap_counts(
        raw, small_cell=int(study.get("small_cell", 11))
    )
