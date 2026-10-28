"""Find cardiac MRI studies on Mayo Clinic Platform, request their DICOM, and build the study table.

MCP keeps DICOM pixel data outside tables: every instance is listed in FACT_RADIOLOGY_DICOM_INVENTORY and
is REQUESTED for de-identification (IMAGE_DEID_STATUS Ready / Available). Reports are in FACT_RADIOLOGY
(linked by accession number); curated study headers are in FACT_IMAGINGSTUDY.

    from mri_training import mcp_imaging as mi
    studies = mi.find_cmr_studies(eng, study["modality"])          # CMR studies with a final report
    series  = mi.list_series(eng, study["modality"], studies)        # every series of those studies
    req, report = mi.build_request(series)                           # short-axis LGE + cine series only
    mi.deid_request(studies, req).to_csv("/workspace/mri/deid_request.csv", index=False)   # -> send to the TSM
    ... after delivery:
    cmr = mi.build_study_table(studies, "/workspace/mri/dicom")      # one row per study with its folder
    spark.createDataFrame(cmr).write.saveAsTable("cardioai_work.cmr_study")

Only the series the model uses are requested: short-axis LGE (required) and short-axis cine (for LVEF),
picked with the same rules as io.select_series, so localisers, long-axis cines, mapping and other series
are never pulled. Studies without an LGE series are not requested (counted in the report).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .io import CINE_RE, LGE_RE, SAX_RE

LONG_AXIS_RE = re.compile(r"(?i)(2ch|3ch|4ch|lvot|long.?axis|\bla\b|\bhla\b|\bvla\b|axial|\btra\b|cor(onal)?\b|sag(ittal)?\b|loc(alizer|aliser)?\b|scout|survey|tagging|t1.?map|t2.?map|molli|shmolli|ecv|perf(usion)?|flow|phase.?contrast|\bpc\b)")


def _cfg(modality_cfg: dict) -> dict:
    return modality_cfg["mri"]["mcp"]


def _macro(name: str) -> str:
    """'{modality}.X' -> '{{modality}}.X' (engine schema macro)."""
    return re.sub(r"(?<!\{)\{(\w+)\}(?!\})", r"{{\1}}", name)


def _t(c: dict, name: str) -> str:
    return f"{_macro(c['schema'])}.{c[name]}"


def _q(v) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def cmr_studies_sql(modality_cfg: dict, start: str = "1900-01-01", end: str = "2100-01-01") -> str:
    c = _cfg(modality_cfg)
    cpt = ", ".join(_q(x) for x in c["cmr_cpt"])
    pm = c.get("person_map") or {}
    if pm.get("table"):                       # OMOP person: person_source_value = PATIENT_CLINIC_NUMBER (MCP 2026-10-28)
        person = f"pm.{pm['person_id_column']}"
        pm_table = _macro(pm["table"])
        pjoin = (f"LEFT JOIN {pm_table} pm ON CAST(pm.{pm['clinic_number_column']} AS STRING) "
                 f"= CAST(s.PATIENT_CLINIC_NUMBER AS STRING)")
    else:
        person, pjoin = "s.PATIENT_CLINIC_NUMBER", ""
    return f"""
    WITH s AS (
      SELECT * FROM {_t(c, 'study_table')} s
      WHERE (s.MODALITY_1 = 'MR' OR s.MODALITY_2 = 'MR')
        AND (CAST(s.PROCEDURE_CODE_CODING_CODE AS STRING) IN ({cpt})
             OR s.STUDY_DESCRIPTION RLIKE '{c['cmr_description_regex']}'
             OR s.PROCEDURE_CODE_TEXT RLIKE '{c['cmr_description_regex']}')
    ),
    acc AS (   -- accession number of each study (from the inventory) -> its report
      SELECT DISTINCT i.STMO_NAME_IN_ADL, i.MODALITY, i.STUDY_INSTANCE_UID, i.DICOM_ACCESSION_NUMBER
      FROM {_t(c, 'inventory_table')} i
      WHERE i.MODALITY = 'MR'
    )
    SELECT {person} AS person_id, s.PATIENT_CLINIC_NUMBER AS clinic_number, {c.get('patient_dk_source', 'r.PATIENT_DK')} AS patient_dk,
           s.STUDY_INSTANCE_UID AS study_uid,
           s.STMO_NAME_IN_ADL AS stmo, a.MODALITY AS modality, a.DICOM_ACCESSION_NUMBER AS accession,
           r.RADIOLOGY_DTM AS study_datetime, s.STUDY_DESCRIPTION AS study_description,
           CAST(s.PROCEDURE_CODE_CODING_CODE AS STRING) AS procedure_code,
           CONCAT_WS(' ', r.RADIOLOGY_REPORT, r.RADIOLOGY_IMPRESSION) AS report_text,
           r.SERVICE_RESULT_STATUS AS report_status
    FROM s
    JOIN acc a ON a.STMO_NAME_IN_ADL = s.STMO_NAME_IN_ADL AND a.STUDY_INSTANCE_UID = s.STUDY_INSTANCE_UID
    JOIN {_t(c, 'report_table')} r ON r.ACCESSION_NBR = a.DICOM_ACCESSION_NUMBER
    {pjoin}
    WHERE CAST(r.RADIOLOGY_DTM AS DATE) BETWEEN CAST('{start}' AS DATE) AND CAST('{end}' AS DATE)
      AND r.SERVICE_RESULT_STATUS RLIKE '{c['final_report_status_regex']}'
    """


def find_cmr_studies(engine, modality_cfg: dict, start: str = "1900-01-01", end: str = "2100-01-01") -> pd.DataFrame:
    df = engine.query(engine.render(cmr_studies_sql(modality_cfg, start, end), {}))
    return df.drop_duplicates("study_uid").reset_index(drop=True)


def series_sql(modality_cfg: dict, study_uids) -> str:
    c = _cfg(modality_cfg)
    uids = ", ".join(_q(u) for u in study_uids)
    return f"""
    SELECT i.STMO_NAME_IN_ADL AS stmo, i.MODALITY AS modality, i.STUDY_INSTANCE_UID AS study_uid,
           i.SERIES_INSTANCE_UID AS series_uid, MAX(i.SERIES_DESCRIPTION) AS series_description,
           COUNT(*) AS n_instances,
           SUM(CASE WHEN i.IMAGE_DEID_STATUS IN ({', '.join(_q(x) for x in c['deid_ok'])}) THEN 1 ELSE 0 END) AS n_deid_ok
    FROM {_t(c, 'inventory_table')} i
    WHERE i.MODALITY = 'MR' AND i.STUDY_INSTANCE_UID IN ({uids})
    GROUP BY i.STMO_NAME_IN_ADL, i.MODALITY, i.STUDY_INSTANCE_UID, i.SERIES_INSTANCE_UID
    """


def list_series(engine, modality_cfg: dict, studies: pd.DataFrame, batch: int = 2000) -> pd.DataFrame:
    uids = studies.study_uid.astype(str).tolist()
    parts = [engine.query(engine.render(series_sql(modality_cfg, uids[i:i + batch]), {})) for i in range(0, len(uids), batch)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def classify_series(desc: str, n_instances: int) -> str:
    """'lge' / 'cine' / 'other' from the series description (the DICOM headers decide again on delivery)."""
    d = str(desc or "")
    if LONG_AXIS_RE.search(d) and not SAX_RE.search(d):
        return "other"
    if LGE_RE.search(d):
        return "lge"
    if CINE_RE.search(d) and n_instances >= 18:            # several slices x many phases
        return "cine"
    return "other"


def build_request(series: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Series to request (SAX LGE + cine of studies that have an LGE series) and counts for the record."""
    s = series.copy()
    s["kind"] = [classify_series(d, n) for d, n in zip(s.series_description, s.n_instances)]
    s["sax"] = [bool(SAX_RE.search(str(d))) for d in s.series_description]
    s["deid_ready"] = s.n_deid_ok == s.n_instances
    has_lge = s[(s.kind == "lge") & s.deid_ready].groupby("study_uid").size()
    keep = s[s.study_uid.isin(has_lge.index) & s.kind.isin(["lge", "cine"]) & s.deid_ready]
    # several candidate series of a kind: prefer short axis, then the most instances (all kept if tied on SAX;
    # io.select_series picks the final one from the delivered headers)
    keep = keep.sort_values(["study_uid", "kind", "sax", "n_instances"], ascending=[True, True, False, False])
    keep = keep[keep.sax | ~keep.duplicated(["study_uid", "kind"])]
    report = {"studies": int(s.study_uid.nunique()), "studies_with_lge": int(len(has_lge)),
              "studies_without_lge_or_not_deidentifiable": int(s.study_uid.nunique() - len(has_lge)),
              "series_total": int(len(s)), "series_requested": int(len(keep)),
              "instances_requested": int(keep.n_instances.sum()), "instances_total": int(s.n_instances.sum())}
    cols = ["stmo", "modality", "study_uid", "series_uid", "series_description", "kind", "n_instances"]
    return keep[cols].reset_index(drop=True), report


DEID_COLUMNS = ["STUDY_INSTANCE_UID", "PATIENT_DK", "MODALITY", "STMO_NAME_IN_ADL"]


def deid_request(studies: pd.DataFrame, req: pd.DataFrame) -> pd.DataFrame:
    """The file MCP asks for (TSM, 2026-10-15): one row per selected study, with STUDY_INSTANCE_UID,
    PATIENT_DK, MODALITY and STMO_NAME_IN_ADL. Selected = the studies build_request kept (a de-identifiable
    short-axis LGE series). MCP de-identifies whole studies; io.select_series picks the LGE and cine series
    from the delivered files."""
    keep = studies[studies.study_uid.isin(req.study_uid)]
    out = pd.DataFrame({"STUDY_INSTANCE_UID": keep.study_uid.astype(str), "PATIENT_DK": keep.patient_dk,
                        "MODALITY": keep.get("modality", pd.Series("MR", index=keep.index)).fillna("MR"),
                        "STMO_NAME_IN_ADL": keep.stmo})
    out = out.drop_duplicates(["STUDY_INSTANCE_UID", "STMO_NAME_IN_ADL"]).reset_index(drop=True)
    missing = out[DEID_COLUMNS].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} studies lack a value in {DEID_COLUMNS} (check patient_dk_source)")
    return out[DEID_COLUMNS]


def build_study_table(studies: pd.DataFrame, delivered_root: str) -> pd.DataFrame:
    """After delivery: one row per study whose folder exists (<root>/<study_uid>/ or found by name below root)."""
    root = Path(delivered_root)
    found = {p.name: p for p in root.rglob("*") if p.is_dir()}
    rows = []
    for _, r in studies.iterrows():
        f = found.get(str(r.study_uid))
        if f is not None:
            rows.append({"person_id": r.person_id, "study_uid": r.study_uid, "study_datetime": r.study_datetime,
                         "dicom_path": str(f), "report_text": r.report_text, "accession": r.accession})
    return pd.DataFrame(rows, columns=["person_id", "study_uid", "study_datetime", "dicom_path", "report_text", "accession"])
