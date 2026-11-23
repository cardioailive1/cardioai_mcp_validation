"""Read-only discovery of Mayo structured flowsheet height/weight row labels.

Mayo confirmed on November 20 that for the Epic system
FACT_ECHO_TEST.ENCOUNTER_NUMBER == FACT_FLOWSHEETS.ENCOUNTER_NBR ==
FACT_ENCOUNTERS.EHR_ENCOUNTER_NUMBER when captured in one encounter.

This module discovers candidate row labels and metadata ONLY. It does not
query clinical result values or patient encounters. Source-row selection and
units must be reviewed before an Epic encounter-scoped vitals feasibility
audit. No patient data or cohort tables are modified.
"""
from __future__ import annotations

import pandas as pd

from .mcp_echo import _cfg, _macro

_VITAL_PATTERN = r"height|weight|stature|weigh|body mass|\bht\b|\bwt\b"
_FIELDS = (
    "flowsheet_row_name_dk", "flowsheet_row_code",
    "flowsheet_row_description"
)


def flowsheet_vital_row_dictionary_sql(modality_cfg: dict) -> str:
    """Select nonpatient row names/codes from the published Mayo dimension."""
    c = _cfg(modality_cfg)
    schema = _macro(c["schema"])
    return f"""
SELECT CAST(FLOWSHEET_ROW_NAME_DK AS STRING) AS flowsheet_row_name_dk,
       CAST(FLOWSHEET_ROW_CODE AS STRING) AS flowsheet_row_code,
       CAST(FLOWSHEET_ROW_DESCRIPTION AS STRING) AS flowsheet_row_description
FROM {schema}.DIM_FLOWSHEET_ROW_NAME
WHERE REGEXP_CONTAINS(
  LOWER(CONCAT(
    COALESCE(CAST(FLOWSHEET_ROW_CODE AS STRING), ''), ' ',
    COALESCE(CAST(FLOWSHEET_ROW_DESCRIPTION AS STRING), '')
  )), r'{_VITAL_PATTERN}'
)
ORDER BY flowsheet_row_description, flowsheet_row_code
LIMIT 300
"""


def normalize_vital_row_dictionary(rows: pd.DataFrame) -> pd.DataFrame:
    """Return only schema labels: never patients, encounters or vital values."""
    if not set(_FIELDS).issubset(rows):
        raise ValueError("Missing Mayo flowsheet row dictionary columns")
    result = rows[list(_FIELDS)].copy()
    result = result.fillna("").drop_duplicates()
    return result.sort_values(
        ["flowsheet_row_description", "flowsheet_row_code",
         "flowsheet_row_name_dk"]
    ).reset_index(drop=True)


def run_epic_flowsheet_row_discovery(eng, study: dict) -> pd.DataFrame:
    """Nonpatient Mayo DIM metadata query, no clinical-row or cohort query."""
    sql = flowsheet_vital_row_dictionary_sql(study["modality"])
    result = eng.client.query(
        eng.render(sql, {}), location=eng.location
    ).to_dataframe()
    return normalize_vital_row_dictionary(result)
