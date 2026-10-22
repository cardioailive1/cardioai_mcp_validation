"""Decode the MCP ECG waveform table (Base64 strings) once into .npz files that training reads.

    from ecg_training.extract_waveforms import extract_waveforms
    m = extract_waveforms(engine, m, study["modality"], "/workspace/ecg/waveforms")   # fills waveform_uri
    m.to_parquet("/workspace/ecg/manifest.parquet")

Run it after build_manifest() and probe_waveforms. Each ECG becomes <out>/<ecg_id>.npz (signal in the
waveform's own units, fs, lead names), the format ecg_training.io.read_npz / read_any read. Waveforms are
read in batches by ecg_id, so the manifest never holds the Base64 strings. ECGs that fail to decode keep
waveform_uri empty and the reason in waveform_error; train() should drop them (counted in the model card).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .io import decode_waveform


def _safe_name(ecg_id: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(ecg_id))
    return keep if keep == str(ecg_id) and len(keep) <= 100 else hashlib.sha256(str(ecg_id).encode()).hexdigest()[:32]


def _quote(v: str) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def waveform_batch_sql(modality_cfg: dict, ids) -> str:
    e = modality_cfg["ecg"]
    c = e["columns"]
    cols = [f"CAST(x.{c['ecg_id']} AS STRING) AS ecg_id", f"x.{c['waveform_uri']} AS waveform"]
    for key in ("sample_rate_hz", "n_leads"):
        if c.get(key):
            cols.append(f"x.{c[key]} AS {key}")
    table = e["table"].replace("{modality}", "{{modality}}").replace("{work}", "{{work}}")
    return f"SELECT {', '.join(cols)} FROM {table} x WHERE CAST(x.{c['ecg_id']} AS STRING) IN ({', '.join(_quote(i) for i in ids)})"


def decode_row(row: pd.Series, raw_cfg: dict):
    return decode_waveform(row["waveform"], n_leads=row.get("n_leads"), fs=row.get("sample_rate_hz"),
                           units_per_bit=float(raw_cfg.get("units_per_bit", 4.88)), units=raw_cfg.get("units", "uV"),
                           dtype=raw_cfg.get("dtype", "<i2"), layout=raw_cfg.get("layout", "auto"))


def extract_waveforms(engine, manifest: pd.DataFrame, modality_cfg: dict, out_dir: str, batch_size: int = 1000,
                      log: Callable[[str], None] = print, overwrite: bool = False,
                      query: Optional[Callable[[str], pd.DataFrame]] = None) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if modality_cfg["ecg"].get("waveform_storage") == "base64_per_lead":
        return _extract_per_lead(engine, manifest, modality_cfg, out, batch_size, log, overwrite, query)
    raw_cfg = modality_cfg["ecg"].get("waveform_raw", {})
    q = query or (lambda sql: engine.query(engine.render(sql, {})))
    m = manifest.copy()
    m["waveform_uri"] = m.get("waveform_uri", pd.Series(index=m.index, dtype=object)).astype(object)
    m["waveform_error"] = None
    ids = m.ecg_id.astype(str).tolist()
    done = 0
    for start in range(0, len(ids), batch_size):
        chunk = ids[start:start + batch_size]
        todo = [i for i in chunk if overwrite or not (out / f"{_safe_name(i)}.npz").exists()]
        rows = q(waveform_batch_sql(modality_cfg, todo)) if todo else pd.DataFrame(columns=["ecg_id"])
        got = {str(r.ecg_id): r for _, r in rows.iterrows()}
        for i in chunk:
            path = out / f"{_safe_name(i)}.npz"
            idx = m.index[m.ecg_id.astype(str) == i]
            if path.exists() and i not in got:
                m.loc[idx, "waveform_uri"] = str(path)
                continue
            if i not in got:
                m.loc[idx, "waveform_error"] = "not in waveform table"
                continue
            try:
                sig, fs, leads, units = decode_row(got[i], raw_cfg)
                np.savez_compressed(path, signal=sig.astype(np.float32), fs=fs, leads=np.asarray(leads), units=units)
                m.loc[idx, "waveform_uri"] = str(path)
            except Exception as e:  # noqa: BLE001  (one bad waveform must not stop the extraction)
                m.loc[idx, "waveform_error"] = str(e)[:200]
        done += len(chunk)
        log(f"{done}/{len(ids)} ECGs; {int(m.waveform_error.notna().sum())} failed so far")
    return m


def _extract_per_lead(engine, manifest: pd.DataFrame, modality_cfg: dict, out: Path, batch_size: int,
                      log: Callable[[str], None], overwrite: bool,
                      query: Optional[Callable[[str], pd.DataFrame]]) -> pd.DataFrame:
    """MCP layout: one FACT_ECG_WAVEFORM row per lead (ecg_training.mcp_waveforms). Needs the manifest's
    clinic_number and ecg_dtm (labels.sql); batches by patient so each query fetches whole acquisitions."""
    from .mcp_waveforms import assemble, match, waveform_rows_sql
    e = modality_cfg["ecg"]
    q = query or (lambda sql: engine.query(engine.render(sql, {})))
    m = manifest.copy()
    for col in ("clinic_number", "ecg_dtm"):
        if col not in m:
            raise ValueError(f"manifest lacks {col}: rebuild it with waveform_storage base64_per_lead")
    m["waveform_uri"] = m.get("waveform_uri", pd.Series(index=m.index, dtype=object)).astype(object)
    m["waveform_error"] = None
    patients = sorted(m.clinic_number.dropna().astype(str).unique())
    tol = float(e.get("timestamp_tolerance_s", 0))
    done = 0
    for start in range(0, len(patients), batch_size):
        batch = patients[start:start + batch_size]
        sub = m[m.clinic_number.astype(str).isin(batch)]
        todo = sub[[overwrite or not (out / f"{_safe_name(i)}.npz").exists() for i in sub.ecg_id.astype(str)]]
        rows = q(waveform_rows_sql(modality_cfg, todo.clinic_number)) if len(todo) else pd.DataFrame()
        hits = match(todo, rows, tol) if len(todo) else {}
        for idx, r in sub.iterrows():
            i = str(r.ecg_id)
            path = out / f"{_safe_name(i)}.npz"
            if path.exists() and not overwrite:
                m.at[idx, "waveform_uri"] = str(path)
                continue
            if i not in hits:
                m.at[idx, "waveform_error"] = "no Rhythm rows at this ECG_DTM"
                continue
            try:
                sig, fs, leads, units = assemble(hits[i], e)
                np.savez_compressed(path, signal=sig.astype(np.float32), fs=fs, leads=np.asarray(leads), units=units)
                m.at[idx, "waveform_uri"] = str(path)
            except Exception as ex:  # noqa: BLE001  (one bad acquisition must not stop the extraction)
                m.at[idx, "waveform_error"] = str(ex)[:200]
        done += len(batch)
        log(f"{done}/{len(patients)} patients; {int(m.waveform_error.notna().sum())} ECGs failed so far")
    return m
