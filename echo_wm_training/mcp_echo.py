"""Find echocardiograms on Mayo Clinic Platform, request their DICOM, and build the study and reference tables.

Echo reports are in FACT_ECHO_TEST (ECHO_NOTE). Echo DICOM, like CMR, is inventoried in
FACT_RADIOLOGY_DICOM_INVENTORY (MODALITY 'US', echo named in STUDY_DESCRIPTION) and REQUESTED for
de-identification the same way (MCP confirmed 2026-10-28).
MCP (2026-10-21): there is no shared key; link a report to its DICOM study by PATIENT_CLINIC_NUMBER and date
(echo.mcp.link.mode = clinic_date). MCP (2026-10-28): both sides are first kept to transthoracic echo (the ordered
test in DIM_ECHO_TEST, the inventory STUDY_DESCRIPTION); patient-days still holding two reports or two studies are
dropped and counted. LVEF comes from FACT_ECHO_RESULTS (lvef_rows / lvef_table), not from OMOP measurement.

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

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from mri_training.mcp_imaging import DEID_COLUMNS, _macro, _q

from .labels import label_columns, parse_report


def _cfg(modality_cfg: dict) -> dict:
    return modality_cfg["echo"]["mcp"]


def _t(c: dict, name: str) -> str:
    return f"{_macro(c['schema'])}.{c[name]}"


def _rx(pattern: str) -> str:
    """A regex as a Spark SQL string literal: backslashes doubled (Spark unescapes '\\b' to a backspace)."""
    return _q(pattern.replace("\\", "\\\\"))


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
            pjoin = (f"LEFT JOIN {_macro(pm['table'])} pm ON CAST(pm.{pm['clinic_number_column']} AS STRING) "
                 f"= CAST(i.PATIENT_CLINIC_NUMBER AS STRING)")
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
    # patient key of FACT_ECHO_TEST: its own PATIENT_CLINIC_NUMBER when configured (PATIENT_DK is deprecated in some
    # MCP tables), else PATIENT_DK -> DIM_PATIENT
    if e.get("clinic_number"):
        clinic, djoin = f"t.{e['clinic_number']}", ""
        dk = f"t.{e['patient_dk']}" if e.get("patient_dk") else "CAST(NULL AS STRING)"
    else:
        clinic, djoin = f"d.{d['clinic_number']}", f"JOIN {_macro(d['table'])} d ON d.{d['patient_dk']} = t.{e['patient_dk']}"
        dk = f"t.{e['patient_dk']}"
    # MCP (2026-10-28): FACT_ECHO_TEST.ECHO_PERFORMED_TEST_DK -> DIM_ECHO_TEST names the exact test; the inventory's
    # STUDY_DESCRIPTION names the imaged study. Both are filtered to transthoracic echo, so a same-day TEE, stress
    # echo or non-cardiac ultrasound no longer makes the day ambiguous.
    tt, f = c.get("test_type") or {}, c.get("filters") or {}
    if tt.get("table"):
        tsel = f"CAST(tt.{tt['description']} AS STRING)"
        tjoin = f"LEFT JOIN {_macro(tt['table'])} tt ON tt.{tt['key']} = t.{e['performed_test_dk']}"
    else:
        tsel, tjoin = "CAST(NULL AS STRING)", ""
    twhere = ""
    if tt.get("table") and f.get("test_keep_regex"):
        twhere += f"\n        AND {tsel} RLIKE {_rx(f['test_keep_regex'])}"
    if tt.get("table") and f.get("exclude_regex"):
        twhere += f"\n        AND NOT ({tsel} RLIKE {_rx(f['exclude_regex'])})"
    iwhere = ""
    if f.get("study_keep_regex"):
        iwhere += f"\n        AND COALESCE(i.STUDY_DESCRIPTION, '') RLIKE {_rx(f['study_keep_regex'])}"
    if f.get("exclude_regex"):
        iwhere += f"\n        AND NOT (COALESCE(i.STUDY_DESCRIPTION, '') RLIKE {_rx(f['exclude_regex'])})"
    return f"""
    WITH e AS (
      SELECT CAST({clinic} AS STRING) AS clinic_number, t.{e['datetime']} AS study_datetime,
             CAST(t.{e['datetime']} AS DATE) AS study_date, t.{e['note']} AS report_text, {dk} AS patient_dk,
             {tsel} AS test_description
      FROM {_t(c, 'echo_test_table')} t
      {djoin}
      {tjoin}
      WHERE CAST(t.{e['datetime']} AS DATE) BETWEEN CAST('{start}' AS DATE) AND CAST('{end}' AS DATE)
        AND t.{e['note']} IS NOT NULL{twhere}
    ),
    i AS (
      SELECT i.STMO_NAME_IN_ADL, i.MODALITY, i.STUDY_INSTANCE_UID,
             CAST(MAX(i.PATIENT_CLINIC_NUMBER) AS STRING) AS clinic_number, MAX(CAST({sdate} AS DATE)) AS study_date,
             MAX(i.STUDY_DESCRIPTION) AS study_description, COUNT(*) AS n_instances,
             SUM(CASE WHEN i.IMAGE_DEID_STATUS IN ({', '.join(_q(x) for x in c['deid_ok'])}) THEN 1 ELSE 0 END) AS n_deid_ok
      FROM {_t(c, 'inventory_table')} i
      {sjoin}
      WHERE i.MODALITY = 'US'{iwhere}
      GROUP BY i.STMO_NAME_IN_ADL, i.MODALITY, i.STUDY_INSTANCE_UID
    )
    SELECT {person} AS person_id, e.clinic_number, e.patient_dk, i.STUDY_INSTANCE_UID AS study_uid,
           i.STMO_NAME_IN_ADL AS stmo, i.MODALITY AS modality, e.study_datetime, e.study_date, e.report_text,
           e.test_description, i.study_description, i.n_instances, i.n_deid_ok
    FROM e JOIN i ON i.clinic_number = e.clinic_number AND i.study_date = e.study_date
    {pjoin}
    """


def find_echo_studies(engine, modality_cfg: dict, start: str = "1900-01-01", end: str = "2100-01-01",
                      log=print) -> pd.DataFrame:
    df = engine.query(engine.render(echo_studies_sql(modality_cfg, start, end), {}))
    df, report = drop_ambiguous(df)
    if report["ambiguous_days"]:
        log(f"dropped {report['ambiguous_days']} patient-days with more than one echo report or US study "
            f"({report['ambiguous_rows']} candidate links): two transthoracic reports or studies on one day")
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


# ------------------------------------------------------------------------------------------------ LVEF
# MCP (Adam Resnick, 2026-10-28): LVEF is in FACT_ECHO_RESULTS.ECHO_RESULT_VALUE; the result type is named in
# DIM_ECHO_TEST_RESULT_TYPE (ECHO_TEST_RESULT_NAME / ECHO_TEST_RESULT_DESCRIPTION). The OMOP ETL does not load the
# echo tables, so OMOP measurement holds no echo LVEF: the Phase 2 echo cohort reads {work}.echo_lvef built here
# (config/cohorts/echo.yaml macros.lvef_rows).
_NUM = re.compile(r"(\d{1,2}(?:\.\d+)?)\s*(?:-|–|to)\s*(\d{1,2}(?:\.\d+)?)|(\d{1,2}(?:\.\d+)?)")


def parse_lvef(value) -> Optional[float]:
    """'55' / '55%' / '0.55' / '55-60' (midpoint) -> percent; None when no number in 5-90 %."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float, np.number)):
        v = float(value)
    else:
        m = _NUM.search(str(value))
        if not m:
            return None
        v = (float(m.group(1)) + float(m.group(2))) / 2 if m.group(1) else float(m.group(3))
        if m.group(3) and re.search(r"0\.\d", str(value)) and v < 1:      # '0.55'
            v = float(re.search(r"0\.\d+", str(value)).group())
    if 0 < v < 1:
        v *= 100.0                                                       # fraction -> percent
    return round(v, 1) if 5.0 <= v <= 90.0 else None


def lvef_rows_sql(modality_cfg: dict) -> str:
    """Every LVEF-type result with the echo's clinic number and date (one row per stored value)."""
    c = _cfg(modality_cfg)
    r, e, d, pm = c["results"], c["echo_test_columns"], c["dim_patient"], c.get("person_map") or {}
    rc, ty = r["columns"], r["result_type"]
    if e.get("clinic_number"):
        clinic, djoin = f"t.{e['clinic_number']}", ""
    else:
        clinic, djoin = f"d.{d['clinic_number']}", f"JOIN {_macro(d['table'])} d ON d.{d['patient_dk']} = t.{e['patient_dk']}"
    person, pjoin = ("CAST(NULL AS BIGINT)", "")
    if pm.get("table"):
        person = f"pm.{pm['person_id_column']}"
        pjoin = f"LEFT JOIN {_macro(pm['table'])} pm ON CAST(pm.{pm['clinic_number_column']} AS STRING) = CAST({clinic} AS STRING)"
    label = f"CONCAT(COALESCE(CAST(ty.{ty['name']} AS STRING), ''), ' | ', COALESCE(CAST(ty.{ty['description']} AS STRING), ''))"
    return f"""
    SELECT {person} AS person_id, CAST({clinic} AS STRING) AS clinic_number,
           CAST(t.{e['datetime']} AS DATE) AS lvef_date, {label} AS result_label,
           CAST(x.{rc['value']} AS STRING) AS result_value
    FROM {_macro(r['table'])} x
    JOIN {_macro(ty['table'])} ty ON ty.{ty['key']} = x.{rc['result_type_key']}
    JOIN {_t(c, 'echo_test_table')} t ON t.{e['test_key']} = x.{rc['test_key']}
    {djoin}
    {pjoin}
    WHERE {label} RLIKE {_rx(r['lvef_regex'])} AND NOT ({label} RLIKE {_rx(r['exclude_regex'])})
    """


def lvef_table(rows: pd.DataFrame, results_cfg: dict) -> pd.DataFrame:
    """One LVEF per patient and echo date. Several stored values (biplane, 3-D, visual) -> the first method in
    results.method_priority (proposed; the clinical lead confirms on the sign-off sheet); several values of that
    method -> their median. Columns: person_id, clinic_number, lvef_date, lvef, lvef_method, n_values, n_methods."""
    cols = ["person_id", "clinic_number", "lvef_date", "lvef", "lvef_method", "n_values", "n_methods"]
    if rows.empty:
        return pd.DataFrame(columns=cols)
    pri = [(m["method"], re.compile(m["regex"], re.I)) for m in results_cfg["method_priority"]]

    def method(label: str) -> Tuple[int, str]:
        for i, (name, rx) in enumerate(pri):
            if rx.search(label or ""):
                return i, name
        return len(pri), "other"

    d = rows.copy()
    d["lvef"] = d.result_value.map(parse_lvef)
    d = d[d.lvef.notna()]
    if d.empty:
        return pd.DataFrame(columns=cols)
    m = d.result_label.map(method)
    d["rank"], d["lvef_method"] = m.map(lambda x: x[0]), m.map(lambda x: x[1])
    d["clinic_number"] = d.clinic_number.astype(str)
    out = []
    for (cn, day), g in d.groupby(["clinic_number", "lvef_date"]):
        best = g[g["rank"] == g["rank"].min()]
        out.append({"person_id": g.person_id.dropna().iloc[0] if g.person_id.notna().any() else None,
                    "clinic_number": cn, "lvef_date": day, "lvef": round(float(best.lvef.median()), 1),
                    "lvef_method": best.lvef_method.iloc[0], "n_values": int(len(g)),
                    "n_methods": int(g.lvef_method.nunique())})
    return pd.DataFrame(out, columns=cols)


def build_lvef_table(engine, modality_cfg: dict, log=print) -> pd.DataFrame:
    """Save the result as {work}.echo_lvef, e.g. spark.createDataFrame(t).write.mode('overwrite').saveAsTable(...)."""
    rows = engine.query(engine.render(lvef_rows_sql(modality_cfg), {}))
    t = lvef_table(rows, _cfg(modality_cfg)["results"])
    log(f"echo LVEF: {len(t)} patient-days from {len(rows)} stored values; methods {t.lvef_method.value_counts().to_dict()}")
    missing = int(t.person_id.isna().sum())
    if missing:
        log(f"WARNING: {missing} patient-days have no OMOP person_id (person.person_source_value)")
    return t
