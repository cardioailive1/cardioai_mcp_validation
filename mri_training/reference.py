"""Phase 2 MRI reference standard from the signed CMR report, by the SAME code as the training labels.

The Phase 2 MRI cohort (config/cohorts/mri.yaml) reads its reference (ref_infarct, ref_nonischemic_lge, ref_mvo,
ref_lv_thrombus, ref_lv_aneurysm, lvef_report, infarct_size_report) from the table built here, so hedged,
historical and resolved statements are handled exactly as in training (mri_training.dataset.report_labels,
cardioai_mcp/report_context.py). Without the table the cohort falls back to its older word-level SQL rules, and
scripts/evaluate_cohorts.py warns.

Built from the cohort's OWN report text, so the reference and the report shown in the analytic table can't drift:

    res = build_all(eng, ["mri"], study, export_dir)                 # pass 1 (modality mri_ref disabled)
    ref = reference_table(res["mri"]["analytic"])
    spark.createDataFrame(ref).write.mode("overwrite").saveAsTable("cardioai_work.mri_reference")
    # config/modality_tables.yaml: mri_ref.enabled: true
    res = build_all(eng, ["mri"], study, export_dir)                 # pass 2: the cohort now reads this reference
    check_reference(res["mri"]["analytic"])                          # before locking (snapshot mri_analytic_locked)
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

import pandas as pd

from .dataset import load_config, report_labels

COLUMNS = ["infarct", "nonischemic_lge", "lv_thrombus", "lv_aneurysm", "report_mvo", "report_lvef", "report_infarct_size"]


def rules_version(cfg: dict) -> str:
    """Short hash of the label rules (config labels block): recorded with every reference row."""
    return hashlib.sha256(json.dumps(cfg["labels"], sort_keys=True).encode()).hexdigest()[:12]


def reference_table(analytic: pd.DataFrame, cfg: Optional[dict] = None, text_col: str = "cmr_report",
                    date_col: str = "index_date") -> pd.DataFrame:
    """One row per cohort patient with a report: person_id, study_datetime (= the index date), ref_found = 1,
    the label columns (1 / 0 / NaN = masked: not stated or only hedged) and rules_version."""
    cfg = cfg or load_config()
    d = analytic[["person_id", date_col, text_col]].dropna(subset=[text_col])
    d = d[d[text_col].astype(str).str.strip().str.len() > 0]
    labs = pd.DataFrame([report_labels(t, cfg) for t in d[text_col].astype(str)], index=d.index)
    out = pd.DataFrame({"person_id": d.person_id, "study_datetime": pd.to_datetime(d[date_col]), "ref_found": 1})
    out = pd.concat([out, labs[COLUMNS].astype(float)], axis=1)
    out["rules_version"] = rules_version(cfg)
    return out.reset_index(drop=True)


def check_reference(analytic: pd.DataFrame, text_col: str = "cmr_report") -> dict:
    """Before locking: every patient with a report must carry the Python reference (py_ref_found = 1)."""
    has_text = analytic[text_col].notna() if text_col in analytic else pd.Series(False, index=analytic.index)
    found = analytic.get("py_ref_found")
    if found is None:
        raise AssertionError("py_ref_found missing: config/cohorts/mri.yaml is older than mri_training/reference.py")
    missing = int((has_text & pd.to_numeric(found, errors="coerce").ne(1)).sum())
    if missing:
        raise AssertionError(f"{missing} patients with a CMR report have no Python reference row: build "
                             "mri_reference from this analytic table, enable mri_ref, and rebuild the cohort")
    return {"patients_with_report": int(has_text.sum()), "missing_reference": 0}
