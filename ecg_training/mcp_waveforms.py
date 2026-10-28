"""MCP ECG waveforms stored one row per lead (FACT_ECG_WAVEFORM), as Mayo Clinic Platform described them.

MCP (Adam Resnick and Tommy Stretch, 2026-10-21):
  * FACT_ECG_WAVEFORM has one row per PATIENT_DK, ACQUISITION_TS, LEAD_ID and WAVEFORM_TYPE. Use
    WAVEFORM_TYPE = 'Rhythm' (the full 10 s recording); 'Median' is a synthesized representative beat.
  * Eight leads are stored: I, II, V1-V6 (III, aVR, aVL, aVF are derived by runtime.preprocess).
  * Each WAVEFORM value is URL-encoded Base64 that decodes on its own: signed 16-bit little-endian counts;
    counts x 4.88 = microvolts. Records are NOT ordered pieces of one recording (STRING_AGG order is not
    chronological), so each lead must come from exactly one row: an acquisition with several Rhythm rows
    for one lead is skipped with a reason until MCP says how such rows relate (open item F2).
  * The sample rate is ECGSAMPLEBASE in FACT_ECG_WAVEFORM_RESTING_MEASUREMENT; it is cross-checked against
    the sample count (10 s), and inferred from the count (2,500 -> 250 Hz, 5,000 -> 500 Hz) when missing.
  * Link to FACT_ECG: PATIENT_DK -> DIM_PATIENT -> PATIENT_CLINIC_NUMBER (the person-level key across
    medical-record systems), then ACQUISITION_TS = ECG_DTM. ECG_WAVEFORM_FPK does not join to ECG_FPK.
  * MCP OMOP ETL guide (2026-10-15 release): PATIENT_DK is deprecated in some MCP tables and PATIENT_CLINIC_NUMBER
    is the patient key. If a table carries PATIENT_CLINIC_NUMBER, name it as `clinic_number` in its column map
    and it is used directly (no DIM_PATIENT join). OMOP person.person_source_value = PATIENT_CLINIC_NUMBER
    (Adam Resnick, 2026-10-28), so person_map is the OMOP person table.

Config: config/modality_tables.yaml -> ecg (waveform_storage: base64_per_lead).
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote

import numpy as np
import pandas as pd

from .io import LEADS_8, _b64

Record = Tuple[np.ndarray, float, List[str], str]


def _t(name: str) -> str:
    """'{modality}.X' / '{work}.X' / '{cdm}.person' -> engine schema macros."""
    return re.sub(r"(?<!\{)\{(\w+)\}(?!\})", r"{{\1}}", name)


def _key(cols: dict) -> Tuple[str, str]:
    """(kind, column) of a table's patient key: PATIENT_CLINIC_NUMBER when the table has it, else PATIENT_DK."""
    return ("clinic_number", cols["clinic_number"]) if cols.get("clinic_number") else ("patient_dk", cols["patient_dk"])


def _q(v) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def decode_lead(value, dtype: str = "<i2", units_per_bit: float = 4.88) -> np.ndarray:
    """One WAVEFORM cell -> samples in microvolts (URL-decoded first, then strict Base64)."""
    text = value.decode("ascii") if isinstance(value, (bytes, bytearray)) else str(value)
    if "%" in text:
        text = unquote(text)              # unquote, not unquote_plus: '+' is a Base64 character
    raw = _b64(text)
    if len(raw) % np.dtype(dtype).itemsize:
        raise ValueError(f"{len(raw)} bytes is not a whole number of {dtype} samples")
    return np.frombuffer(raw, dtype=dtype).astype(np.float64) * float(units_per_bit)


def sample_rate(n_samples: int, stated: Optional[float], seconds: float = 10.0, tol_s: float = 0.5) -> float:
    if stated is not None and not (isinstance(stated, float) and np.isnan(stated)) and float(stated) > 0:
        fs = float(stated)
        if abs(n_samples / fs - seconds) > tol_s:
            raise ValueError(f"{n_samples} samples at ECGSAMPLEBASE {fs:g} Hz = {n_samples / fs:.2f} s, not {seconds:g} s")
        return fs
    fs = n_samples / seconds
    if fs not in (250.0, 500.0, 1000.0):
        raise ValueError(f"no ECGSAMPLEBASE and {n_samples} samples imply {fs:g} Hz")
    return fs


def assemble(rows: pd.DataFrame, cfg: dict) -> Record:
    """Rows of ONE acquisition (lead_id, waveform, sample_rate) -> (signal uV (8, N), fs, leads, 'uV')."""
    raw_cfg = cfg.get("waveform_raw", {})
    want = list(cfg.get("leads", LEADS_8))
    lead = rows.lead_id.astype(str).str.strip()
    dup = sorted(set(lead[lead.duplicated()]))
    if dup:
        raise ValueError(f"several Rhythm rows for lead(s) {', '.join(dup)} (MCP open item F2)")
    missing = [x for x in want if x not in set(lead)]
    if missing:
        raise ValueError(f"missing lead(s) {', '.join(missing)}")
    by = dict(zip(lead, rows.waveform))
    sigs = [decode_lead(by[x], raw_cfg.get("dtype", "<i2"), raw_cfg.get("units_per_bit", 4.88)) for x in want]
    n = {len(s) for s in sigs}
    if len(n) != 1:
        raise ValueError(f"leads have different lengths {sorted(n)}")
    stated = pd.to_numeric(rows.get("sample_rate", pd.Series(dtype=float)), errors="coerce").dropna().unique()
    if len(stated) > 1:
        raise ValueError(f"conflicting ECGSAMPLEBASE values {sorted(stated)}")
    fs = sample_rate(n.pop(), float(stated[0]) if len(stated) else None, float(cfg.get("seconds", 10)))
    return np.stack(sigs), fs, want, raw_cfg.get("units", "uV")


def waveform_rows_sql(modality_cfg: dict, clinic_numbers: Iterable) -> str:
    """Rhythm rows (with clinic number and sample rate) for a batch of patients; matched to ECGs in pandas."""
    e = modality_cfg["ecg"]
    w, d, sr = e["waveform_columns"], e["dim_patient"], e.get("sample_rate") or {}
    ids = ", ".join(_q(c) for c in sorted({str(x) for x in clinic_numbers}))
    kind, wk = _key(w)
    if kind == "clinic_number":                       # the waveform table carries the clinic number itself
        clinic, djoin = f"w.{wk}", ""
    else:
        clinic, djoin = f"d.{d['clinic_number']}", f"JOIN {_t(d['table'])} d ON d.{d['patient_dk']} = w.{wk}"
    if sr.get("table"):
        sk = sr.get(kind) or sr.get("patient_dk")
        rate, rjoin = (f"r.{sr['column']}",
                       f"LEFT JOIN {_t(sr['table'])} r ON r.{sk} = w.{wk} "
                       f"AND r.{sr['acquisition_ts']} = w.{w['acquisition_ts']}")
    else:
        rate, rjoin = "CAST(NULL AS DOUBLE)", ""
    return f"""
    SELECT CAST({clinic} AS STRING) AS clinic_number,
           CAST(w.{w['acquisition_ts']} AS TIMESTAMP) AS acquisition_ts,
           CAST(w.{w['lead_id']} AS STRING) AS lead_id, w.{w['waveform']} AS waveform,
           CAST({rate} AS DOUBLE) AS sample_rate
    FROM {_t(e['waveform_table'])} w
    {djoin}
    {rjoin}
    WHERE w.{w['waveform_type']} = {_q(e.get('waveform_type', 'Rhythm'))}
      AND CAST({clinic} AS STRING) IN ({ids})
    """


def match(ecgs: pd.DataFrame, rows: pd.DataFrame, tol_s: float = 0.0) -> Dict[str, pd.DataFrame]:
    """{ecg_id: its waveform rows}: same clinic number and ACQUISITION_TS = ECG_DTM (within tol_s seconds)."""
    out: Dict[str, pd.DataFrame] = {}
    if rows.empty:
        return out
    rows = rows.assign(acquisition_ts=pd.to_datetime(rows.acquisition_ts), clinic_number=rows.clinic_number.astype(str))
    for cn, g in rows.groupby("clinic_number"):
        mine = ecgs[ecgs.clinic_number.astype(str) == cn]
        for _, e in mine.iterrows():
            t = pd.Timestamp(e.ecg_dtm)
            hit = g[(g.acquisition_ts - t).abs() <= pd.Timedelta(seconds=tol_s)]
            if len(hit):
                out[str(e.ecg_id)] = hit
    return out


def probe(engine, modality_cfg: dict, n: int = 20) -> pd.DataFrame:
    """Numbers only, for n acquisitions: Rhythm rows per lead, samples, stated rate, seconds, amplitude."""
    e = modality_cfg["ecg"]
    w = e["waveform_columns"]
    _, wk = _key(w)
    keys = engine.query(engine.render(
        f"SELECT DISTINCT w.{wk} AS dk, w.{w['acquisition_ts']} AS ts FROM {_t(e['waveform_table'])} w "
        f"WHERE w.{w['waveform_type']} = {_q(e.get('waveform_type', 'Rhythm'))} LIMIT {int(n)}", {}))
    out = []
    for _, k in keys.iterrows():
        sql = (f"SELECT CAST(w.{w['lead_id']} AS STRING) AS lead_id, w.{w['waveform']} AS waveform FROM {_t(e['waveform_table'])} w "
               f"WHERE w.{wk} = {_q(k.dk)} AND w.{w['acquisition_ts']} = CAST({_q(k.ts)} AS TIMESTAMP) "
               f"AND w.{w['waveform_type']} = {_q(e.get('waveform_type', 'Rhythm'))}")
        rows = engine.query(engine.render(sql, {}))
        r = {"rows": len(rows), "max_rows_per_lead": int(rows.lead_id.value_counts().max()) if len(rows) else 0,
             "leads": ",".join(sorted(rows.lead_id.astype(str).unique()))}
        try:
            sig, fs, _, _ = assemble(rows.assign(sample_rate=np.nan), e)
            mv = sig / 1000.0
            r.update(samples=sig.shape[1], fs=fs, seconds=round(sig.shape[1] / fs, 2),
                     p99_abs_mv=round(float(np.percentile(np.abs(mv), 99)), 2))
        except Exception as ex:  # noqa: BLE001
            r["error"] = str(ex)[:160]
        out.append(r)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ ECG index
INDEX_COLUMNS = ["read_statement", "is_confirmed", "vent_rate", "pr_interval", "qrs_duration", "qt_interval",
                 "qtc_interval", "qrs_axis"]


def ecg_index_sql(modality_cfg: dict) -> str:
    """One row per FACT_ECG record with the standard ecg_index columns (config ecg.columns names):
    OMOP person_id (person_map: OMOP person, person_source_value = clinic number), clinic number, ecg_id, acquisition time, read, measurements.
    FACT_ECG columns not yet confirmed (null in ecg.fact_ecg.columns) come through as NULL."""
    e = modality_cfg["ecg"]
    f, d, pm, out = e["fact_ecg"], e["dim_patient"], e.get("person_map") or {}, e["columns"]
    fc = f["columns"]
    if not pm.get("table"):
        raise ValueError("ecg.person_map.table is not set: OMOP person_id comes from person.person_source_value "
                         "= PATIENT_CLINIC_NUMBER (config/modality_tables.yaml)")
    kind, fk = _key(fc)
    if kind == "clinic_number":                       # FACT_ECG carries the clinic number itself
        clinic, djoin = f"f.{fk}", ""
    else:
        clinic, djoin = f"d.{d['clinic_number']}", f"JOIN {_t(d['table'])} d ON d.{d['patient_dk']} = f.{fk}"
    sel = [f"CAST(pm.{pm['person_id_column']} AS BIGINT) AS {out['person_id']}",
           f"CAST({clinic} AS STRING) AS {out['clinic_number']}",
           f"CAST(f.{fc['ecg_id']} AS STRING) AS {out['ecg_id']}",
           f"CAST(f.{fc['acquisition_datetime']} AS TIMESTAMP) AS {out['acquisition_datetime']}"]
    for k in INDEX_COLUMNS:
        if k in out:
            sel.append((f"f.{fc[k]}" if fc.get(k) else "CAST(NULL AS STRING)") + f" AS {out[k]}")
    return f"""
    SELECT {', '.join(sel)}
    FROM {_t(f['table'])} f
    {djoin}
    LEFT JOIN {_t(pm['table'])} pm ON CAST(pm.{pm['clinic_number_column']} AS STRING) = CAST({clinic} AS STRING)
    """


def build_ecg_index(engine, modality_cfg: dict) -> pd.DataFrame:
    """Run ecg_index_sql; save the result as ecg.table (e.g. spark.createDataFrame(idx).write.saveAsTable(...))."""
    idx = engine.query(engine.render(ecg_index_sql(modality_cfg), {}))
    out = modality_cfg["ecg"]["columns"]
    missing = int(idx[out["person_id"]].isna().sum())
    if missing:
        print(f"WARNING: {missing} ECGs have no OMOP person_id through person_map; they cannot be labelled")
    return idx
