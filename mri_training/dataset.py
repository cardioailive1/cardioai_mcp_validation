"""MRI training manifest: one row per cardiac MRI study with report labels and split.

    from mri_training.dataset import load_config, build_manifest
    m = build_manifest(engine, study, cfg, study["modality"], exclude_persons=...)

Report labels (config.yaml labels:) for the study heads, plus report values used for evaluation
(LVEF, infarct size, MVO). Splits are patient-disjoint by time and site (shared with the ECG agent).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yaml

from ecg_training.dataset import assign_splits, check_disjoint

HERE = Path(__file__).parent


def load_config(path: Optional[str] = None) -> dict:
    return yaml.safe_load(Path(path or HERE / "config.yaml").read_text())


def _flag(text: str, pos: str, neg: Optional[str]) -> Optional[float]:
    clean = re.sub(neg, " ", text) if neg else text
    return 1.0 if re.search(pos, clean) else 0.0


def report_labels(text: Optional[str], cfg: dict) -> dict:
    """One CMR report -> study labels (1 / 0 / NaN) and stated values (regexes capture the number as `val`)."""
    lab = cfg["labels"]
    nan = float("nan")
    if not isinstance(text, str) or len(text.strip()) < 20:
        return {"infarct": nan, "nonischemic_lge": nan, "lv_thrombus": nan, "lv_aneurysm": nan,
                "report_mvo": nan, "report_lvef": nan, "report_infarct_size": nan}
    no_lge = bool(re.search(lab["no_lge_regex"], text))
    body = re.sub(lab["no_lge_regex"], " ", text)
    infarct = 1.0 if re.search(lab["infarct_regex"], body) else (0.0 if no_lge else nan)
    nonisch = 1.0 if re.search(lab["nonischemic_regex"], body) else (0.0 if (no_lge or infarct == 1.0) else nan)
    if no_lge and infarct != 1.0 and nonisch != 1.0:
        infarct, nonisch = 0.0, 0.0
    if nonisch == 1.0 and infarct != 1.0:
        infarct = 0.0          # LGE described, pattern non-ischaemic only -> no infarct
    default_neg = lab.get("thrombus_aneurysm_default_negative", True)
    thr = _flag(text, lab["thrombus_regex"], lab["thrombus_negation_regex"])
    ane = _flag(text, lab["aneurysm_regex"], lab["aneurysm_negation_regex"])
    mvo = _flag(text, lab["mvo_regex"], lab["mvo_negation_regex"])
    ef = re.search(lab["lvef_regex"], text)
    size = re.search(lab["infarct_size_regex"], text)
    return {"infarct": infarct, "nonischemic_lge": nonisch,
            "lv_thrombus": thr if (thr == 1.0 or default_neg) else nan,
            "lv_aneurysm": ane if (ane == 1.0 or default_neg) else nan,
            "report_mvo": mvo,
            "report_lvef": float(ef.group("val")) if ef else nan,
            "report_infarct_size": float(size.group("val")) if size else nan}


def make_labels(ev: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    labs = pd.DataFrame([report_labels(t, cfg) for t in ev.report_text], index=ev.index)
    return pd.concat([ev, labs], axis=1)


def evidence_sql(engine, cfg: dict, modality_cfg: dict) -> str:
    mri = modality_cfg["mri"]
    extra = {"mri_table": mri["table"].replace("{modality}", "{{modality}}")}
    extra.update({f"c_{k}": v for k, v in mri["columns"].items()})
    return engine.render((HERE / "labels.sql").read_text(), extra)


def build_manifest(engine, study: dict, cfg: dict, modality_cfg: dict, exclude_persons: Iterable[int] = ()) -> pd.DataFrame:
    from cardioai_mcp.concepts import resolve_concept_sets
    resolve_concept_sets(engine, study["concept_sets"], only={"stemi", "mi_acute", "cardiac_mri_contrast"})
    ev = engine.query(evidence_sql(engine, cfg, modality_cfg))
    m = assign_splits(make_labels(ev, cfg), cfg, exclude_persons, date_col="study_date")
    bad = {k: v for k, v in check_disjoint(m).items() if v}
    if bad:
        raise AssertionError(f"splits share patients: {bad}")
    return m


def summary(m: pd.DataFrame, heads: Iterable[str]) -> pd.DataFrame:
    rows = []
    for split, g in m.groupby("split"):
        r = {"split": split, "studies": len(g), "patients": g.person_id.nunique(),
             "contoured": int(g.get("label_uri", pd.Series(dtype=object)).notna().sum()) if "label_uri" in g else 0}
        for h in heads:
            r[f"{h}_pos"] = int((g[h] == 1).sum())
            r[f"{h}_labelled"] = int(g[h].notna().sum())
        rows.append(r)
    return pd.DataFrame(rows)
