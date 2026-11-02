"""The de-identification request MCP needs before it releases imaging (MCP TSM): a CSV of the selected studies with
exactly STUDY_INSTANCE_UID, PATIENT_DK, MODALITY and STMO_NAME_IN_ADL.

Shared by the MRI (mri_training.mcp_imaging) and echo (echo_wm_training.mcp_echo) request builders; run it all with
    python scripts/build_deid_request.py --modality mri echo --out /workspace/deid

PATIENT_DK: the DICOM inventory and FACT_IMAGINGSTUDY carry PATIENT_CLINIC_NUMBER, not PATIENT_DK (MCP Radiology
slides). DIM_PATIENT has one row per medical-record system; PATIENT_MERGED_FLAG = 'N' gives one row per person
(Adam Resnick, 2026-10-21). So PATIENT_DK is looked up there from the study's clinic number, and a clinic number
with no row, or with several unmerged rows, stops the request instead of guessing.

Only adults are requested (age >= study.yaml min_age at the study, from OMOP person; dates are shifted by the same
amount across a patient's record, so age is preserved). Studies with no OMOP person or birth date are not requested.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import pandas as pd

DEID_COLUMNS = ["STUDY_INSTANCE_UID", "PATIENT_DK", "MODALITY", "STMO_NAME_IN_ADL"]
EXPECTED_MODALITY = {"mri": "MR", "echo": "US"}


def _macro(name: str) -> str:
    return re.sub(r"(?<!\{)\{(\w+)\}(?!\})", r"{{\1}}", name)


def _q(v) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def _chunks(values: List[str], n: int = 2000):
    for i in range(0, len(values), n):
        yield values[i:i + n]


# ------------------------------------------------------------------------------------------------ PATIENT_DK
def dim_patient_sql(dim: dict, clinic_numbers: List[str]) -> str:
    flag = dim.get("merged_flag_column", "PATIENT_MERGED_FLAG")
    keep = dim.get("merged_flag_keep", "N")
    return f"""
    SELECT CAST(d.{dim['clinic_number']} AS STRING) AS clinic_number, d.{dim['patient_dk']} AS patient_dk
    FROM {_macro(dim['table'])} d
    WHERE d.{flag} = {_q(keep)}
      AND CAST(d.{dim['clinic_number']} AS STRING) IN ({', '.join(_q(c) for c in clinic_numbers)})
    """


def attach_patient_dk(engine, mcp_cfg: dict, studies: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Set studies.patient_dk from DIM_PATIENT (unmerged row of the study's clinic number).

    Returns (studies with patient_dk, counts). Clinic numbers with no unmerged row are counted and their studies
    dropped; clinic numbers with several unmerged rows raise, because MCP would get an ambiguous patient key."""
    if mcp_cfg.get("patient_dk_source") != "dim_patient":
        return studies, {"patient_dk_from": "study query"}
    dim = mcp_cfg["dim_patient"]
    s = studies.copy()
    s["clinic_number"] = s["clinic_number"].astype(str)
    clinics = sorted(s.clinic_number.dropna().unique().tolist())
    parts = [engine.query(engine.render(dim_patient_sql(dim, c), {})) for c in _chunks(clinics)]
    dk = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["clinic_number", "patient_dk"])
    dk["clinic_number"] = dk["clinic_number"].astype(str)
    dk = dk.drop_duplicates()
    n_per = dk.groupby("clinic_number").patient_dk.nunique()
    if (n_per > 1).any():
        raise ValueError(f"{int((n_per > 1).sum())} clinic numbers have more than one unmerged DIM_PATIENT row "
                         f"({dim.get('merged_flag_column', 'PATIENT_MERGED_FLAG')} = "
                         f"'{dim.get('merged_flag_keep', 'N')}'); ask MCP which PATIENT_DK to send")
    s = s.drop(columns=["patient_dk"], errors="ignore").merge(dk, on="clinic_number", how="left")
    missing = s.patient_dk.isna()
    counts = {"patient_dk_from": "DIM_PATIENT", "studies_without_patient_dk": int(missing.sum())}
    return s[~missing].reset_index(drop=True), counts


# ------------------------------------------------------------------------------------------------ adults only
def person_birth_sql(person_table: str, person_ids: List) -> str:
    ids = ", ".join(str(int(p)) for p in person_ids)
    return f"""
    SELECT p.person_id, p.year_of_birth, p.month_of_birth, p.day_of_birth
    FROM {_macro(person_table)} p
    WHERE p.person_id IN ({ids})
    """


def adults_only(engine, mcp_cfg: dict, studies: pd.DataFrame, min_age: float = 18) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Keep studies done at age >= min_age (OMOP person birth date; missing month/day -> 1 July / 15th)."""
    s = studies.copy()
    table = (mcp_cfg.get("person_map") or {}).get("table", "{cdm}.person")
    ids = pd.to_numeric(s.get("person_id"), errors="coerce").dropna().astype("int64").unique().tolist()
    parts = [engine.query(engine.render(person_birth_sql(table, c), {})) for c in _chunks(ids)]
    b = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["person_id", "year_of_birth"])
    if len(b):
        dob = pd.to_datetime(dict(year=b.year_of_birth.astype("Int64"),
                                  month=b.month_of_birth.fillna(7).astype("Int64"),
                                  day=b.day_of_birth.fillna(15).astype("Int64")), errors="coerce")
        b = pd.DataFrame({"person_id": b.person_id.astype("int64"), "_dob": dob})
    else:
        b = pd.DataFrame({"person_id": pd.Series(dtype="int64"), "_dob": pd.Series(dtype="datetime64[ns]")})
    s["_pid"] = pd.to_numeric(s.get("person_id"), errors="coerce").astype("Int64")
    s = s.merge(b.rename(columns={"person_id": "_pid"}).astype({"_pid": "Int64"}), on="_pid", how="left")
    when = pd.to_datetime(s.get("study_datetime", s.get("study_date")), errors="coerce")
    age = (when - s._dob).dt.days / 365.25
    unknown = age.isna()
    child = ~unknown & (age < min_age)
    counts = {"studies_under_min_age": int(child.sum()), "studies_age_unknown": int(unknown.sum()), "min_age": min_age}
    return s[~unknown & ~child].drop(columns=["_pid", "_dob"]).reset_index(drop=True), counts


# ------------------------------------------------------------------------------------------------ the file
def validate(df: pd.DataFrame, modality: str | None = None) -> List[str]:
    """Problems that would make MCP reject or mis-process the file (empty list = ready to send)."""
    problems = []
    if list(df.columns) != DEID_COLUMNS:
        problems.append(f"columns must be exactly {DEID_COLUMNS}, got {list(df.columns)}")
        return problems
    blank = df.isna() | (df.astype(str).apply(lambda c: c.str.strip()) == "")
    if blank.any().any():
        problems.append(f"{int(blank.any(axis=1).sum())} rows have an empty cell")
    dup = df.duplicated(["STUDY_INSTANCE_UID", "STMO_NAME_IN_ADL"])
    if dup.any():
        problems.append(f"{int(dup.sum())} duplicate studies")
    allowed = {EXPECTED_MODALITY[modality]} if modality else set(EXPECTED_MODALITY.values())
    bad = ~df.MODALITY.isin(allowed)
    if bad.any():
        problems.append(f"{int(bad.sum())} rows with MODALITY not in {sorted(allowed)}")
    if not df.STUDY_INSTANCE_UID.astype(str).str.fullmatch(r"[0-9.]+").all():
        problems.append("some STUDY_INSTANCE_UID values are not DICOM UIDs (digits and dots)")
    if len(df) == 0:
        problems.append("no studies selected")
    return problems


def write_csv(df: pd.DataFrame, path: str) -> None:
    df[DEID_COLUMNS].to_csv(path, index=False)
