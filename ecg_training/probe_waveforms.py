"""Look at a few rows of the MCP ECG waveform table and report how the Base64 strings are laid out.

    python -m ecg_training.probe_waveforms --n 5          # inside the MCP workspace (Spark engine)
    from ecg_training.probe_waveforms import probe; probe(engine, study["modality"], n=5)

For each row: decoded size, what is inside (MUSE XML / gzip / npz / JSON / raw samples), and for raw
samples the lead layout (lead-major vs interleaved: the smoother reading wins), seconds implied by the
table's sample rate, and the amplitude range in mV at the configured units_per_bit. A 12-lead resting
ECG should give about 10 s and R waves around 0.5-2 mV; anything else means waveform_raw in
config/modality_tables.yaml needs changing before extract_waveforms is run. Prints numbers only, no
patient data.
"""
from __future__ import annotations

import argparse
import gzip
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .io import _b64, container_of, decode_waveform, raw_layout


def describe(value, n_leads: Optional[int], fs: Optional[float], raw_cfg: Dict) -> Dict:
    r: Dict = {}
    try:
        raw = _b64(value)
    except ValueError as e:
        return {"container": "not base64", "error": str(e)}
    r["decoded_bytes"] = len(raw)
    kind = container_of(raw)
    if kind == "gzip":
        raw = gzip.decompress(raw)
        r["gunzipped_bytes"] = len(raw)
        kind = "gzip+" + container_of(raw)
    r["container"] = kind
    if kind.endswith("raw") and n_leads:
        flat = np.frombuffer(raw[: len(raw) - len(raw) % np.dtype(raw_cfg.get("dtype", "<i2")).itemsize],
                             dtype=raw_cfg.get("dtype", "<i2")).astype(float)
        try:
            _, layout, scores = raw_layout(flat, int(n_leads), raw_cfg.get("layout", "auto"))
            r.update(layout=layout, smooth_lead_major=round(scores["lead_major"], 3), smooth_interleaved=round(scores["interleaved"], 3),
                     samples_per_lead=flat.size // int(n_leads))
            if fs:
                r["seconds"] = round(flat.size / int(n_leads) / float(fs), 2)
        except ValueError as e:
            r["error"] = str(e)
    try:
        sig, rate, leads, units = decode_waveform(value, n_leads=n_leads, fs=fs, units_per_bit=float(raw_cfg.get("units_per_bit", 4.88)),
                                                  units=raw_cfg.get("units", "uV"), dtype=raw_cfg.get("dtype", "<i2"),
                                                  layout=raw_cfg.get("layout", "auto"))
        mv = sig / 1000.0 if units.lower() in ("uv", "microvolts") else sig
        r.update(n_leads=sig.shape[0], fs=rate, leads=",".join(leads), seconds=round(sig.shape[1] / rate, 2),
                 p99_abs_mv=round(float(np.percentile(np.abs(mv), 99)), 2), max_abs_mv=round(float(np.abs(mv).max()), 2))
    except Exception as e:  # noqa: BLE001
        r["decode_error"] = str(e)[:200]
    return r


def probe(engine, modality_cfg: dict, n: int = 5) -> pd.DataFrame:
    e = modality_cfg["ecg"]
    if e.get("waveform_storage") == "base64_per_lead":       # MCP FACT_ECG_WAVEFORM: one row per lead
        from .mcp_waveforms import probe as probe_per_lead
        return probe_per_lead(engine, modality_cfg, n)
    c = e["columns"]
    cols = [f"x.{c['waveform_uri']} AS waveform"] + [f"x.{c[k]} AS {k}" for k in ("sample_rate_hz", "n_leads") if c.get(k)]
    sql = f"SELECT {', '.join(cols)} FROM {e['table'].replace('{modality}', '{{modality}}').replace('{work}', '{{work}}')} x WHERE x.{c['waveform_uri']} IS NOT NULL LIMIT {int(n)}"
    rows = engine.query(engine.render(sql, {}))
    raw_cfg = e.get("waveform_raw", {})
    out = pd.DataFrame([describe(r.waveform, r.get("n_leads"), r.get("sample_rate_hz"), raw_cfg) for _, r in rows.iterrows()])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--engine", choices=["bigquery", "spark"], default="bigquery")
    a = ap.parse_args()
    from cardioai_mcp.config import load_study
    from cardioai_mcp.engine import make_engine
    study = load_study()
    print(probe(make_engine(a.engine, study), study["modality"], a.n).to_string())


if __name__ == "__main__":
    main()
