"""Accession-aware Echo reference candidates for MCP validation.

This module is deliberately NON-DESTRUCTIVE: it reads only the currently saved
Echo analytic cohort and returns a candidate reference DataFrame. It never
overwrites echo_lvef or echo_analytic. Review accession ambiguity and clinical
method priorities before promoting any output to the Phase 2 cohort.
"""
from __future__ import annotations

from datetime import date
import re
from typing import Optional

import pandas as pd

from .mcp_echo import _cfg, _macro, _rx, _t, parse_lvef

# Absolute LVEDV (mL) only. Exclude indexed volumes, percentages and ratios.
LVEDV_PRIORITY = (
    "LV_vol_BP_ed_2D_by_MOD",
    "LV_vol_ed_3D",
    "LV_vol_4C_ed_2D_by_MOD",
    "LV_vol_2C_ed_2D_by_MOD",
)


def exam_reference_rows_sql(modality_cfg: dict, *, start: str, end: str) -> str:
    """Candidate LVEF/LVEDV rows for patients in the saved Phase 2 Echo cohort.

    A cohort index date may match a performed echo on that date or the next.
    Each row retains its accession and ECHO_TEST_FPK; unlike the old LVEF
    query, this does not group different exams together by patient/day.
    """
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last:
        raise ValueError("start must be on or before end")

    c = _cfg(modality_cfg)
    e, d, pm, results = c["echo_test_columns"], c["dim_patient"], c.get("person_map") or {}, c["results"]
    rc, ty = results["columns"], results["result_type"]
    if not pm.get("table"):
        raise ValueError("An OMOP person_map is required for cohort-scoped echo references")
    result_name = f"ty.{ty['name']}"
    label = (f"CONCAT(COALESCE(CAST({result_name} AS STRING), ''), ' | ', "
             f"COALESCE(CAST(ty.{ty['description']} AS STRING), ''))")
    lvef_test = (f"({label} RLIKE {_rx(results['lvef_regex'])} "
                 f"AND NOT ({label} RLIKE {_rx(results['exclude_regex'])}))")
    lvedv_names = ", ".join("'" + name + "'" for name in LVEDV_PRIORITY)
    return f"""
    SELECT DISTINCT
           a.person_id,
           CAST(a.index_date AS DATE) AS index_date,
           CAST(d.{d['clinic_number']} AS STRING) AS clinic_number,
           CAST(t.{e['link']} AS STRING) AS accession_number,
           CAST(t.{e['test_key']} AS STRING) AS echo_test_fpk,
           CAST(t.{e['datetime']} AS DATE) AS exam_date,
           CAST({result_name} AS STRING) AS result_name,
           {label} AS result_label,
           CAST(r.{rc['value']} AS STRING) AS result_value,
           CASE WHEN {result_name} IN ({lvedv_names}) THEN 'lvedv' ELSE 'lvef' END AS metric
    FROM {{{{work}}}}.echo_analytic a
    JOIN {_macro(pm['table'])} pm
      ON pm.{pm['person_id_column']} = a.person_id
    JOIN {_macro(d['table'])} d
      ON CAST(d.{d['clinic_number']} AS STRING) = CAST(pm.{pm['clinic_number_column']} AS STRING)
    JOIN {_t(c, 'echo_test_table')} t
      ON t.{e['patient_dk']} = d.{d['patient_dk']}
    JOIN {_macro(results['table'])} r
      ON r.{rc['test_key']} = t.{e['test_key']}
    JOIN {_macro(ty['table'])} ty
      ON ty.{ty['key']} = r.{rc['result_type_key']}
    WHERE CAST(a.index_date AS DATE) BETWEEN CAST('{start}' AS DATE) AND CAST('{end}' AS DATE)
      AND DATEDIFF(CAST(t.{e['datetime']} AS DATE), CAST(a.index_date AS DATE)) BETWEEN 0 AND 1
      AND ({result_name} IN ({lvedv_names}) OR {lvef_test})
    """


def select_exam_references(rows: pd.DataFrame, results_cfg: dict) -> pd.DataFrame:
    """One candidate row per patient/index-date/accession, with ambiguity status.

    Does not decide which accession is the index TTE. The caller must review
    status == 'unique_accession' and establish TTE eligibility before use.
    """
    cols = ["person_id", "index_date", "accession_number", "exam_dates",
            "candidate_accessions", "status", "lvef", "lvef_method",
            "lvedv", "lvedv_method"]
    if rows.empty:
        return pd.DataFrame(columns=cols)
    required = {"person_id", "index_date", "accession_number", "exam_date",
                "result_label", "result_name", "result_value", "metric"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Missing echo reference columns: {sorted(missing)}")

    d = rows.copy()
    d["index_date"] = pd.to_datetime(d.index_date).dt.date
    d["exam_date"] = pd.to_datetime(d.exam_date).dt.date
    d["accession_number"] = d.accession_number.astype("string").str.strip().replace("", pd.NA)
    priorities = [(x["method"], re.compile(x["regex"], re.I))
                  for x in results_cfg["method_priority"]]

    def lvef_pick(g: pd.DataFrame):
        if g.empty:
            return None, None
        g = g.copy()
        g["value"] = g.result_value.map(parse_lvef)
        g = g[g.value.notna()]
        if g.empty:
            return None, None
        def classify(label):
            for idx, (name, pattern) in enumerate(priorities):
                if pattern.search(str(label)):
                    return idx, name
            return len(priorities), "other"
        classified = g.result_label.map(classify)
        g["priority"] = classified.map(lambda x: x[0])
        g["method"] = classified.map(lambda x: x[1])
        selected = g[g.priority == g.priority.min()]
        return round(float(selected.value.median()), 1), selected.method.iloc[0]

    def lvedv_pick(g: pd.DataFrame):
        if g.empty:
            return None, None
        g = g.copy()
        g["value"] = pd.to_numeric(g.result_value, errors="coerce")
        g = g[g.value.between(20, 500) & g.result_name.isin(LVEDV_PRIORITY)]
        if g.empty:
            return None, None
        rank = {name: idx for idx, name in enumerate(LVEDV_PRIORITY)}
        g["priority"] = g.result_name.map(rank)
        selected = g[g.priority == g.priority.min()]
        return round(float(selected.value.median()), 1), selected.result_name.iloc[0]

    output = []
    for (person_id, index_date), patient in d.groupby(["person_id", "index_date"], dropna=False):
        accs = patient.accession_number.dropna().unique()
        n_acc = len(accs)
        for accession, g in patient.groupby("accession_number", dropna=False):
            dates = g.exam_date.nunique()
            if pd.isna(accession):
                status = "missing_accession"
            elif n_acc > 1 or patient.accession_number.isna().any():
                status = "multiple_or_missing_accessions"
            elif dates > 1:
                status = "multi_date_accession"
            else:
                status = "unique_accession"
            lv, lv_method = lvef_pick(g[g.metric == "lvef"])
            vol, vol_method = lvedv_pick(g[g.metric == "lvedv"])
            output.append({
                "person_id": person_id, "index_date": index_date,
                "accession_number": accession, "exam_dates": int(dates),
                "candidate_accessions": int(n_acc), "status": status,
                "lvef": lv, "lvef_method": lv_method,
                "lvedv": vol, "lvedv_method": vol_method,
            })
    return pd.DataFrame(output, columns=cols)


def build_exam_reference_candidates(engine, modality_cfg: dict, *, start: str, end: str,
                                    log=print) -> pd.DataFrame:
    """Read candidate rows and return accession-aware references; no writes."""
    sql = exam_reference_rows_sql(modality_cfg, start=start, end=end)
    rows = engine.query(engine.render(sql, {}))
    out = select_exam_references(rows, _cfg(modality_cfg)["results"])
    log(f"Echo exam reference candidates: {len(out)} accession groups from {len(rows)} result rows")
    if not out.empty:
        log(f"Status counts: {out.status.value_counts(dropna=False).to_dict()}")
        log(f"Non-null LVEF: {int(out.lvef.notna().sum())}; LVEDV: {int(out.lvedv.notna().sum())}")
    return out
