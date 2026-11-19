"""Read-only Mayo encounter/link-key schema discovery (no patient data).

Before attempting any encounter-based join, inspect the published FACT/DIM
column names and types in the shared Mayo dataset. An encounter can contain
multiple echo tests; a shared ENCOUNTER_NUMBER alone is *not* exam identity.
Use native BigQuery INFORMATION_SCHEMA rather than Spark SQL transpilation.
"""
from __future__ import annotations

import re

import pandas as pd


_SCHEMA = re.compile(r"^`?[A-Za-z0-9_-]+`?\.[A-Za-z_][A-Za-z0-9_]*$")
_TABLE_PATTERN = (
    "ECHO|ENCOUNTER|FLOWSHEET|CLINICAL_DOCUMENT|"
    "PROCEDURE|ORDER"
)
_COLUMN_PATTERN = (
    "ENCOUNTER|ACCESSION|ORDER|PROC|PATIENT|TEST_FPK|TEST_DK|"
    "DOCUMENT|FLOWSHEET|PERFORMED|VISIT|RECORD|RESULT"
)


def encounter_schema_sql(modality_schema: str) -> str:
    """Return metadata only; never SELECT from actual patient source rows."""
    if not _SCHEMA.fullmatch(modality_schema):
        raise ValueError("Expected a configured project.dataset modality schema")
    return f"""
SELECT table_name, column_name, data_type, ordinal_position
FROM {modality_schema}.INFORMATION_SCHEMA.COLUMNS
WHERE REGEXP_CONTAINS(UPPER(table_name), r'({_TABLE_PATTERN})')
  AND REGEXP_CONTAINS(UPPER(column_name), r'({_COLUMN_PATTERN})')
ORDER BY table_name, ordinal_position
"""


def _normalize_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    needed = {"table_name", "column_name", "data_type", "ordinal_position"}
    if not needed.issubset(metadata):
        raise ValueError("Mayo INFORMATION_SCHEMA metadata has missing columns")
    frame = metadata[list(sorted(needed))].copy()
    if frame[["table_name", "column_name"]].isna().any().any():
        raise ValueError("Schema metadata cannot contain unnamed columns")
    return frame.sort_values(["table_name", "ordinal_position"]).reset_index(
        drop=True
    )


def run_encounter_schema_discovery(eng, study: dict) -> pd.DataFrame:
    """Return a code/schema-only table suitable for review with Mayo.

    Runs inside MCP using the same native BigQuery metadata access path
    already used by cohort_source_discovery. Does not load clinical rows.
    """
    sql = encounter_schema_sql(study["schemas"]["modality"])
    raw = eng.client.query(eng.render(sql, {}), location=eng.location).to_dataframe()
    return _normalize_metadata(raw)
