"""Find echocardiograms on Mayo Clinic Platform, request their DICOM, and build the study and reference tables.

Echo reports are in FACT_ECHO_TEST (ECHO_NOTE). Echo DICOM, like CMR, is inventoried in
FACT_RADIOLOGY_DICOM_INVENTORY (MODALITY 'US') and REQUESTED for de-identification (still to confirm with MCP).
MCP (2026-10-21): there is no shared key; link a report to its DICOM study by PATIENT_CLINIC_NUMBER and date
(echo.mcp.link.mode = clinic_date). Ambiguous patient-days (two reports or two studies) are dropped and counted.

    from echo_wm_training import mcp_echo as me
    studies = me.find_echo_studies(eng, study["modality"])                 # echo exams with a report
    req, report = me.build_request(eng, study["modality"], studies)         # studies with de-identifiable instances
    me.deid_request(studies, req).to_csv("/workspace/echo/deid_request.csv", index=False)   # -> TSM
    ... after delivery:
    table = me.build_study_table(studies, "/workspace/echo/dicom")          # one row per delivered study folder
    ref = me.reference_table(studies)                                       # report-derived wall-motion reference

Echo studies are requested whole: view labels are rarely in the inventory, so clips are selected after delivery
(io.list_clips + views.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from mri_training.mcp_imaging import DEID_COLUMNS, _macro, _q

from .labels import label_columns, parse_report


def _cfg(modality_cfg: dict) -> dict:
    return modality_cfg["echo"]["mcp"]


def _t(c: dict, name: str) -> str:
    return f"{_macro(c['schema'])}.{c[name]}"


def echo_studies_sql(modality_cfg: dict, start: str = "1900-01-01", end: str = "2100-01-01") -> str:
    """Echo reports joined to their DICOM studies.

    link.mode 'clinic_date' (MCP, 2026-10-21): no shared key exists, so a FACT_ECHO_TEST row and an US study in
    the inventory are linked by PATIENT_CLINIC_NUMBER (through DIM_PATIENT) and the same date; dates are shifted
    by the same amount across a patient's record, so same-day still means same day. Days with more than one
    report or more than one US study are dropped by find_echo_studies (ambiguous). link.mode 'accession': the
    older key-based join (link / inventory_link columns)."""
    c = _cfg(modality_cfg)
    e = c["echo_test_columns"]
    link = c.get("link") or {"mode": "accession"}
    pm = c.get("person_map") or {}
    if link["mode"] == "accession":
        if pm.get("table"):
            person = f"pm.{pm['person_id_column']}"
            pjoin = f"LEFT JOIN {_macro(pm['table'])} pm ON pm.{pm['clinic_number_column']} = i.PATIENT_CLINIC_NUMBER"
        else:
            person, pjoin = "i.PATIENT_CLINIC_NUMBER", ""
        return f"""
    WITH e AS (
      SELECT t.{e['link']} AS link_key, t.{e['datetime']} AS study_datetime, t.{e['note']} AS report_text,
             t.{e['patient_dk']} AS patient_dk
      FROM {_t(c, 'echo_test_table')} t
      WHERE CAST(t.{e['datetime']} AS DATE) BETWEEN CAST('{start}' AS DATE) AND CAST('{end}' AS DATE)
        AND t.{e['note']} IS NOT NULL
    ),
    i AS (
      SELECT i.STMO_NAME_IN_ADL, i.MODALITY, i.STUDY_INSTANCE_UID, i.{c['inventory_link']} AS link_key,
             MAX(i.PATIENT_CLINIC_NUMBER) AS PATIENT_CLINIC_NUMBER, COUNT(*) AS n_instances,
             SUM(CASE WHEN i.IMAGE_DEID_STATUS IN ({', '.join(_q(x) for x in c['deid_ok'])}) THEN 1 ELSE 0 END) AS n_deid_ok
      FROM {_t(c, 'inventory_table')} i
      WHERE i.MODALITY = 'US'
      GROUP BY i.STMO_NAME_IN_ADL, i.MODALITY, i.STUDY_INSTANCE_UID, i.{c['inventory_link']}
    )
    SELECT {person} AS person_id, i.PATIENT_CLINIC_NUMBER AS clinic_number, e.patient_dk, i.STUDY_INSTANCE_UID AS study_uid,
           i.STMO_NAME_IN_ADL AS stmo, i.MODALITY AS modality, e.study_datetime, e.report_text,
           i.n_instances, i.n_deid_ok
    FROM e JOIN i ON i.link_key = e.link_key
    {pjoin}
    """
    d = c["dim_patient"]
    sd = link["study_date"]                       # where the DICOM study's date lives
    if pm.get("table"):
        person = f"pm.{pm['person_id_column']}"
        pjoin = f"LEFT JOIN {_macro(pm['table'])} pm ON CAST(pm.{pm['clinic_number_column']} AS STRING) = e.clinic_number"
    else:
        person, pjoin = "e.clinic_number", ""
    if sd.get("table"):
        sjoin = (f"JOIN {_macro(sd['table'])} s ON s.STMO_NAME_IN_ADL = i.STMO_NAME_IN_ADL AND s.MODALITY_1 = i.MODALITY "
                 f"AND s.STUDY_INSTANCE_UID = i.STUDY_INSTANCE_UID")
        sdate = f"s.{sd['column']}"
    else:
        sjoin, sdate = "", f"i.{sd['column']}"
    return f"""
    WITH e AS (
      SELECT CAST(d.{d['clinic_number']} AS STRING) AS clinic_number, t.{e['datetime']} AS study_datetime,
             CAST(t.{e['datetime']} AS DATE) AS study_date, t.{e['note']} AS report_text, t.{e['patient_dk']} AS patient_dk
      FROM {_t(c, 'echo_test_table')} t
      JOIN {_macro(d['table'])} d ON d.{d['patient_dk']} = t.{e['patient_dk']}
      WHERE CAST(t.{e['datetime']} AS DATE) BETWEEN CAST('{start}' AS DATE) AND CAST('{end}' AS DATE)
        AND t.{e['note']} IS NOT NULL
    ),
    i AS (
      SELECT i.STMO_NAME_IN_ADL, i.MODALITY, i.STUDY_INSTANCE_UID,
             CAST(MAX(i.PATIENT_CLINIC_NUMBER) AS STRING) AS clinic_number, MAX(CAST({sdate} AS DATE)) AS study_date,
             COUNT(*) AS n_instances,
             SUM(CASE WHEN i.IMAGE_DEID_STATUS IN ({', '.join(_q(x) for x in c['deid_ok'])}) THEN 1 ELSE 0 END) AS n_deid_ok
      FROM {_t(c, 'inventory_table')} i
      {sjoin}
      WHERE i.MODALITY = 'US'
      GROUP BY i.STMO_NAME_IN_ADL, i.MODALITY, i.STUDY_INSTANCE_UID
    )
    SELECT {person} AS person_id, e.clinic_number, e.patient_dk, i.STUDY_INSTANCE_UID AS study_uid,
           i.STMO_NAME_IN_ADL AS stmo, i.MODALITY AS modality, e.study_datetime, e.study_date, e.report_text,
           i.n_instances, i.n_deid_ok
    FROM e JOIN i ON i.clinic_number = e.clinic_number AND i.study_date = e.study_date
    {pjoin}
    """


def find_echo_studies(engine, modality_cfg: dict, start: str = "1900-01-01", end: str = "2100-01-01",
                      log=print) -> pd.DataFrame:
    df = engine.query(engine.render(echo_studies_sql(modality_cfg, start, end), {}))
    df, report = drop_ambiguous(df)
    if report["ambiguous_days"]:
        log(f"dropped {report['ambiguous_days']} patient-days with more than one echo report or US study "
            f"({report['ambiguous_rows']} candidate links): no time or study-type match yet (MCP open item F5)")
    df.attrs["link_report"] = report
    return df.drop_duplicates("study_uid").reset_index(drop=True)


def drop_ambiguous(df: pd.DataFrame):
    """Keep only patient-days with exactly one echo report and one US study (date-based link)."""
    if "study_date" not in df or df.empty:
        return df, {"ambiguous_days": 0, "ambiguous_rows": 0}
    key = [df.clinic_number.astype(str), pd.to_datetime(df.study_date)]
    n_rep = df.groupby(key).report_text.transform("nunique")
    n_std = df.groupby(key).study_uid.transform("nunique")
    ok = (n_rep == 1) & (n_std == 1)
    bad = df[~ok]
    return df[ok].copy(), {"ambiguous_days": int(bad.groupby([bad.clinic_number.astype(str), bad.study_date]).ngroups) if len(bad) else 0,
                           "ambiguous_rows": int(len(bad))}


def build_request(studies: pd.DataFrame, min_instances: int = 10) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Studies worth requesting: de-identifiable, enough instances to hold apical cines, and a wall-motion
    statement in the report (otherwise they carry no label)."""
    s = studies.copy()
    s["wm_described"] = [bool(parse_report(t)["wm_described"]) for t in s.report_text]
    ok = (s.n_deid_ok >= min_instances) & s.wm_described
    report = {"studies": len(s), "not_deidentifiable": int((s.n_deid_ok < min_instances).sum()),
              "no_wall_motion_statement": int((~s.wm_described).sum()), "requested": int(ok.sum())}
    return s[ok][["study_uid", "stmo", "modality"]].reset_index(drop=True), report


def deid_request(studies: pd.DataFrame, req: pd.DataFrame) -> pd.DataFrame:
    """The CSV MCP asks for (STUDY_INSTANCE_UID, PATIENT_DK, MODALITY, STMO_NAME_IN_ADL), one row per study."""
    keep = studies[studies.study_uid.isin(req.study_uid)]
    out = pd.DataFrame({"STUDY_INSTANCE_UID": keep.study_uid.astype(str), "PATIENT_DK": keep.patient_dk,
                        "MODALITY": keep.get("modality", pd.Series("US", index=keep.index)).fillna("US"),
                        "STMO_NAME_IN_ADL": keep.stmo}).drop_duplicates(["STUDY_INSTANCE_UID", "STMO_NAME_IN_ADL"])
    missing = out[DEID_COLUMNS].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} studies lack a value in {DEID_COLUMNS} (check echo.mcp columns)")
    return out[DEID_COLUMNS].reset_index(drop=True)


def build_study_table(studies: pd.DataFrame, delivered_root: str) -> pd.DataFrame:
    root = Path(delivered_root)
    found = {p.name: p for p in root.rglob("*") if p.is_dir()}
    rows = [{"person_id": r.person_id, "study_uid": r.study_uid, "study_datetime": r.study_datetime,
             "dicom_path": str(found[str(r.study_uid)]), "report_text": r.report_text}
            for _, r in studies.iterrows() if str(r.study_uid) in found]
    return pd.DataFrame(rows, columns=["person_id", "study_uid", "study_datetime", "dicom_path", "report_text"])


def reference_table(studies: pd.DataFrame) -> pd.DataFrame:
    """Report-derived wall-motion reference for every echo study (cardioai_work.echo_wm_reference): read by the
    Phase 2 echo cohort (config/cohorts/echo.yaml, modality echo_wm) and audited with annotation/label_audit.py."""
    labs = pd.DataFrame([parse_report(t) for t in studies.report_text], index=studies.index)
    keep = ["wm_described", "wm_uncertain"] + label_columns()
    out = pd.concat([studies[["person_id", "study_uid", "study_datetime"]], labs[keep]], axis=1)
    out["ref_wall_motion"] = out.wma_any.map({1.0: "abnormal", 0.0: "normal"})
    return out
