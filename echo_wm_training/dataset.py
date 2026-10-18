"""Echo wall-motion training manifest (one row per study) and clip table (one row per 2-D cine).

    from echo_wm_training.dataset import load_config, build_manifest, build_clip_table
    m = build_manifest(eng, study, cfg, study["modality"], exclude_persons=excl)   # labels + patient-disjoint splits
    clips = build_clip_table(m)                                                     # every 2-D cine of every study

Labels come from the echo report (labels.parse_report). Studies with no wall-motion statement keep NaN labels
(masked) and are used only for the LVEF head when an LVEF is known. Splits follow the ECG / MRI calendar.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd
import yaml

from ecg_training.dataset import assign_splits, check_disjoint

from .io import list_clips
from .labels import label_columns, parse_report

HERE = Path(__file__).parent


def load_config(path: Optional[str] = None) -> dict:
    return yaml.safe_load(Path(path or HERE / "config.yaml").read_text())


def make_labels(ev: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    un = bool(cfg["labels"].get("unmentioned_normal", True))
    labs = pd.DataFrame([parse_report(t, un) for t in ev.report_text], index=ev.index)
    out = pd.concat([ev, labs[["wm_described", "wm_uncertain"] + label_columns()]], axis=1)
    if "lvef_measured" in out:                       # structured echo measurement first (labels.lvef_source)
        out["lvef"] = out.lvef_measured.where(out.lvef_measured.notna(), out.report_lvef)
    else:
        out["lvef"] = out.report_lvef
    return out


def evidence_sql(engine, modality_cfg: dict) -> str:
    w = modality_cfg["echo_wm"]
    extra = {"echo_wm_table": w["table"].replace("{work}", "{{work}}")}
    extra.update({f"c_{k}": v for k, v in w["columns"].items()})
    return engine.render((HERE / "labels.sql").read_text(), extra)


def build_manifest(engine, study: dict, cfg: dict, modality_cfg: dict, exclude_persons: Iterable[int] = ()) -> pd.DataFrame:
    ev = engine.query(evidence_sql(engine, modality_cfg))
    m = assign_splits(make_labels(ev, cfg), cfg, exclude_persons, date_col="study_date")
    bad = {k: v for k, v in check_disjoint(m).items() if v}
    if bad:
        raise AssertionError(f"splits share patients: {bad}")
    return m


def build_clip_table(m: pd.DataFrame, lister: Callable[[str, str], pd.DataFrame] = list_clips,
                     log: Callable[[str], None] = print) -> pd.DataFrame:
    parts = []
    for i, r in enumerate(m.itertuples()):
        try:
            parts.append(lister(str(r.dicom_uri), str(r.study_id)))
        except Exception as e:  # noqa: BLE001
            log(f"{r.study_id}: {e}")
        if (i + 1) % 500 == 0:
            log(f"{i + 1}/{len(m)} studies listed")
    c = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["study_id", "clip_uri", "kind"])
    return c[c.kind == "cine"].reset_index(drop=True)


def summary(m: pd.DataFrame, heads: Iterable[str]) -> pd.DataFrame:
    rows = []
    for split, g in m.groupby("split"):
        r = {"split": split, "studies": len(g), "patients": g.person_id.nunique(),
             "wm_described": int(g.get("wm_described", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
             "lvef_known": int(g["lvef"].notna().sum()) if "lvef" in g else 0}
        for h in heads:
            r[f"{h}_pos"] = int((g[h] == 1).sum())
            r[f"{h}_labelled"] = int(g[h].notna().sum())
        rows.append(r)
    return pd.DataFrame(rows)
