"""MRI training manifest: one row per cardiac MRI study with report labels and split.

    from mri_training.dataset import load_config, build_manifest
    m = build_manifest(engine, study, cfg, study["modality"], exclude_persons=...)

Report labels (config.yaml labels:) for the study heads, plus report values used for evaluation
(LVEF, infarct size, MVO). Splits are patient-disjoint by time and site (shared with the ECG agent).
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yaml

from ecg_training.dataset import assign_splits, check_disjoint

HERE = Path(__file__).parent


def load_config(path: Optional[str] = None) -> dict:
    return yaml.safe_load(Path(path or HERE / "config.yaml").read_text())


def report_labels(text: Optional[str], cfg: dict) -> dict:
    """One CMR report -> study labels (1 / 0 / NaN) and stated values (regexes capture the number as `val`).

    Each mention is judged in its sentence (cardioai_mcp/report_context.py): affirmed, negated, uncertain
    ("possible", "cannot be excluded", "versus", "differential includes"), historical ("history of", "was reported
    on the referring echo") or resolved ("has resolved"). A label is 1 when affirmed anywhere, masked (NaN) when its
    only mentions are uncertain, otherwise by the rules below.
      infarct / nonischemic_lge: affirmed -> 1; uncertain -> NaN; "no LGE" or (the other pattern affirmed) -> 0;
        otherwise NaN (LGE not assessed). "Prior infarct" counts (old scar is still there); an infarct-pattern
        mention in a sentence about diffuse / amyloid enhancement with no coronary wording is ignored.
      lv_thrombus, report_mvo: affirmed -> 1; uncertain -> NaN; otherwise 0 (silence = absent, model card).
      lv_aneurysm: as thrombus, and "aneurysmal" or dyskinesis without an aneurysm named -> NaN (rule sheet).
    """
    from cardioai_mcp.report_context import Context
    lab = cfg["labels"]
    nan = float("nan")
    if not isinstance(text, str) or len(text.strip()) < 20:
        return {"infarct": nan, "nonischemic_lge": nan, "lv_thrombus": nan, "lv_aneurysm": nan,
                "report_mvo": nan, "report_lvef": nan, "report_infarct_size": nan}
    ctx = Context(lab.get("context"))
    no_lge = ctx.combine(ctx.statuses(text, lab["no_lge_regex"])) == 1.0          # "no LGE" stated (not hedged)
    excl = re.compile(lab["infarct_exclude_regex"]) if lab.get("infarct_exclude_regex") else None
    coronary = re.compile(lab.get("infarct_keep_regex", r"(?!x)x"))
    skip = (lambda sent: bool(excl.search(sent)) and not coronary.search(sent)) if excl else None
    inf_st = ctx.statuses(text, lab["infarct_regex"], skip)
    non_st = ctx.statuses(text, lab["nonischemic_regex"])
    inf, non = ctx.combine(inf_st), ctx.combine(non_st)

    def resolve(own, other):
        if own == 1.0:
            return 1.0
        if own is not None and math.isnan(own):
            return nan                                     # only hedged mentions
        if no_lge or other == 1.0:
            return 0.0
        return nan
    infarct, nonisch = resolve(inf, non), resolve(non, inf)

    def present(pattern) -> float:
        v = ctx.label(text, pattern)
        return 0.0 if v is None else v                     # silence = absent (thrombus_aneurysm_default_negative)

    default_neg = lab.get("thrombus_aneurysm_default_negative", True)
    thr = present(lab["thrombus_regex"])
    ane = present(lab["aneurysm_regex"])
    if ane == 0.0 and lab.get("aneurysm_uncertain_regex") and re.search(lab["aneurysm_uncertain_regex"], text):
        ane = nan                                          # "aneurysmal segment" / dyskinesis, no aneurysm named
    mvo = present(lab["mvo_regex"])
    ef = re.search(lab["lvef_regex"], text)
    size = re.search(lab["infarct_size_regex"], text)
    return {"infarct": infarct, "nonischemic_lge": nonisch,
            "lv_thrombus": thr if (thr != 0.0 or default_neg) else nan,
            "lv_aneurysm": ane if (ane != 0.0 or default_neg) else nan,
            "report_mvo": mvo,
            "report_lvef": float(ef.group("val")) if ef else nan,
            "report_infarct_size": float(size.group("val")) if size else nan}


def make_labels(ev: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    labs = pd.DataFrame([report_labels(t, cfg) for t in ev.report_text], index=ev.index)
    return pd.concat([ev, labs], axis=1)


def evidence_sql(engine, cfg: dict, modality_cfg: dict) -> str:
    mri = modality_cfg["mri"]
    extra = {"mri_table": re.sub(r"(?<!\{)\{(\w+)\}(?!\})", r"{{\1}}", mri["table"])}   # {work} -> {{work}} macro
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
