"""Training manifest for the ECG model: one row per ECG with labels and split.

    from ecg_training.dataset import build_manifest
    manifest = build_manifest(engine, study, cfg)   # inside the MCP workspace (Spark engine)
    manifest.to_parquet(".../ecg_manifest.parquet")

Labels follow config.yaml (labels:); splits are patient-disjoint by time and site.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).parent


def load_config(path: Optional[str] = None) -> dict:
    return yaml.safe_load(Path(path or HERE / "config.yaml").read_text())


def _ecg_macros(modality_cfg: dict) -> dict:
    ecg = modality_cfg["ecg"]
    out = {"ecg_table": ecg["table"].replace("{modality}", "{{modality}}")}
    for k, v in ecg["columns"].items():
        out[f"c_{k}"] = v
    return out


def evidence_sql(engine, study: dict, cfg: dict, modality_cfg: dict) -> str:
    lab = cfg["labels"]
    t = lab["troponin_url_fallback"]
    ecg = modality_cfg["ecg"]
    b64 = ecg.get("waveform_storage", "file") == "base64"
    extra = {**_ecg_macros(modality_cfg), "win_lo": lab["window_days"][0], "win_hi": lab["window_days"][1],
             "clear_days": lab["clear_days"], "trop_ng_per_l": t["ng_per_l"],
             "trop_ng_per_l_male": t.get("ng_per_l_male", t["ng_per_l"]),
             "trop_ng_per_l_female": t.get("ng_per_l_female", t["ng_per_l"]), "trop_ng_per_ml": t["ng_per_ml"],
             # Base64 waveforms are not pulled into the manifest (about 160 kB each): extract_waveforms adds the paths
             "waveform_select": "CAST(NULL AS STRING)" if b64 else f"x.{ecg['columns']['waveform_uri']}"}
    return engine.render((HERE / "labels.sql").read_text(), extra)


def make_labels(ev: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Evidence columns -> acute_mi / stemi / af in {1, 0, NaN} (NaN = excluded from that head's loss)."""
    lab = cfg["labels"]
    d = ev.copy()
    pos_mi = (d.mi_dx == 1) & (d.trop_pos == 1)
    neg_mi = d.mi_any_near == 0
    d["acute_mi"] = np.where(pos_mi, 1.0, np.where(neg_mi, 0.0, np.nan))
    d["stemi"] = np.where(pos_mi & (d.stemi_dx == 1), 1.0,
                          np.where(neg_mi | (pos_mi & (d.nstemi_dx == 1) & (d.stemi_dx == 0)), 0.0, np.nan))
    af_re, neg_re = re.compile(lab["af_regex"]), re.compile(lab["af_negation_regex"])

    def af(row):
        txt = row.read_text
        if not isinstance(txt, str) or not txt.strip():
            return np.nan
        if lab.get("require_confirmed_read", True) and row.confirmed != 1:
            return np.nan
        clean = neg_re.sub(" ", txt)
        return 1.0 if af_re.search(clean) else 0.0

    d["af"] = d.apply(af, axis=1) if len(d) else pd.Series(dtype=float)
    return d


def _site_held_out(site, cfg: dict) -> bool:
    sp = cfg["splits"]
    if sp.get("heldout_sites"):
        return site in set(sp["heldout_sites"])
    h = int(hashlib.sha256(f"{sp['seed']}:{site}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < float(sp.get("heldout_site_fraction", 0.0))


def assign_splits(d: pd.DataFrame, cfg: dict, exclude_persons: Iterable[int] = (), date_col: str = "ecg_date") -> pd.DataFrame:
    """Patient-disjoint split.

    test_site : every ECG at a held-out site; those patients appear nowhere else
    otherwise by the patient's FIRST ECG date: train (< tune_start), tune (< test_start), test_time
    Each patient keeps only ECGs dated inside their split's window. Phase 2 validation
    patients (exclude_persons) are removed from train and tune.
    """
    sp = cfg["splits"]
    tune_start, test_start = pd.Timestamp(sp["tune_start"]), pd.Timestamp(sp["test_start"])
    d = d.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d["_held_site"] = d.site.map(lambda s: _site_held_out(s, cfg))
    site_pts = set(d.loc[d._held_site, "person_id"])
    site_part = d[d._held_site].assign(split="test_site")
    rest = d[~d.person_id.isin(site_pts)].copy()
    first = rest.groupby("person_id")[date_col].transform("min")
    rest["split"] = np.where(first < tune_start, "train", np.where(first < test_start, "tune", "test_time"))
    in_window = np.select([rest.split == "train", rest.split == "tune"],
                          [rest[date_col] < tune_start, (rest[date_col] >= tune_start) & (rest[date_col] < test_start)],
                          default=rest[date_col] >= test_start)
    rest = rest[in_window]
    excl = set(exclude_persons)
    rest = rest[~(rest.split.isin(["train", "tune"]) & rest.person_id.isin(excl))]
    cap = int(sp.get("max_ecgs_per_patient", 0) or 0)
    if cap:
        trn = rest.split.isin(["train", "tune"])
        shuffled = rest[trn].sample(frac=1.0, random_state=sp["seed"])           # deterministic random subset
        capped = shuffled[shuffled.groupby("person_id").cumcount() < cap]
        rest = pd.concat([capped, rest[~trn]])
    out = pd.concat([rest, site_part]).drop(columns="_held_site")
    return out.sort_values(["split", "person_id", date_col]).reset_index(drop=True)


def check_disjoint(m: pd.DataFrame) -> dict:
    """{pair: number of shared patients} - all must be 0."""
    s = {k: set(g.person_id) for k, g in m.groupby("split")}
    keys = sorted(s)
    return {f"{a}&{b}": len(s[a] & s[b]) for i, a in enumerate(keys) for b in keys[i + 1:]}


def build_manifest(engine, study: dict, cfg: dict, modality_cfg: dict, exclude_persons: Iterable[int] = ()) -> pd.DataFrame:
    from cardioai_mcp.concepts import resolve_concept_sets
    resolve_concept_sets(engine, study["concept_sets"], only={"mi_acute", "mi_any", "stemi", "nstemi", "troponin"})
    ev = engine.query(evidence_sql(engine, study, cfg, modality_cfg))
    m = assign_splits(make_labels(ev, cfg), cfg, exclude_persons)
    bad = {k: v for k, v in check_disjoint(m).items() if v}
    if bad:
        raise AssertionError(f"splits share patients: {bad}")
    return m


def summary(m: pd.DataFrame, heads: Iterable[str]) -> pd.DataFrame:
    """ECGs, patients and positives / labelled per split and head (for the model card)."""
    rows = []
    for split, g in m.groupby("split"):
        r = {"split": split, "ecgs": len(g), "patients": g.person_id.nunique()}
        for h in heads:
            r[f"{h}_pos"] = int((g[h] == 1).sum())
            r[f"{h}_labelled"] = int(g[h].notna().sum())
        rows.append(r)
    return pd.DataFrame(rows)
