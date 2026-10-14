"""Read 12-lead ECG waveforms from the formats an MCP workspace may hold.

Every reader returns (signal (n_leads, N) in the file's units, fs, lead names, units),
which goes straight into runtime.preprocess.preprocess(). Which format MCP uses is
confirmed at onboarding (config/modality_tables.yaml, ecg).

Mayo Clinic Platform (confirmed by MCP, 2026-10-14): the ECG waveform table holds each waveform as a
Base64-encoded string. decode_waveform() reads that string whatever it wraps (MUSE XML, gzip, .npz,
JSON arrays, or raw little-endian integers); probe_waveforms.py reports which, on a few real rows.
"""
from __future__ import annotations

import base64
import gzip
import io as _io
import json
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

Record = Tuple[np.ndarray, float, List[str], str]


def read_muse_xml(data: bytes) -> Record:
    """GE MUSE RestingECG XML: rhythm waveform, base64 little-endian int16 per lead.

    MUSE stores 8 leads (I, II, V1-V6); preprocess() derives III, aVR, aVL, aVF.
    Amplitude = sample * LeadAmplitudeUnitsPerBit (normally 4.88 microvolts).
    """
    root = ET.fromstring(data)
    wf = None
    for w in root.iter("Waveform"):
        if (w.findtext("WaveformType") or "").strip().lower() == "rhythm":
            wf = w
    if wf is None:
        raise ValueError("no Rhythm waveform in MUSE XML")
    fs = float(wf.findtext("SampleBase") or 500)
    leads, sigs, units = [], [], "uV"
    for ld in wf.iter("LeadData"):
        raw = base64.b64decode((ld.findtext("WaveFormData") or "").strip())
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        x *= float(ld.findtext("LeadAmplitudeUnitsPerBit") or 4.88)
        u = (ld.findtext("LeadAmplitudeUnits") or "MICROVOLTS").upper()
        units = "mV" if u.startswith("MILLI") else "uV"
        leads.append((ld.findtext("LeadID") or "").strip())
        sigs.append(x)
    n = min(len(s) for s in sigs)
    return np.stack([s[:n] for s in sigs]), fs, leads, units


def read_wfdb(base_path: str) -> Record:
    import wfdb
    rec = wfdb.rdrecord(base_path)
    units = (rec.units or ["mV"])[0]
    return rec.p_signal.T, float(rec.fs), list(rec.sig_name), units


def read_npz(data: bytes) -> Record:
    """np.savez(signal=(n_leads, N), fs=..., leads=[...], units='mV')."""
    z = np.load(_io.BytesIO(data), allow_pickle=False)
    return z["signal"], float(z["fs"]), [str(x) for x in z["leads"]], str(z["units"]) if "units" in z else "mV"


def read_any(uri: str) -> Record:
    """Dispatch on file extension. WFDB records are given by their base path (no extension)."""
    ext = os.path.splitext(uri)[1].lower()
    if ext == ".xml":
        with open(uri, "rb") as f:
            return read_muse_xml(f.read())
    if ext == ".npz":
        with open(uri, "rb") as f:
            return read_npz(f.read())
    if ext in (".hea", ".dat") or os.path.exists(uri + ".hea"):
        return read_wfdb(uri[:-4] if ext in (".hea", ".dat") else uri)
    raise ValueError(f"unsupported ECG file: {uri}")


# --------------------------------------------------------------------------- Base64 strings (MCP waveform table)
LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEADS_8 = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]          # GE MUSE storage


def _b64(text: Union[str, bytes]) -> bytes:
    if isinstance(text, (bytes, bytearray)):
        text = bytes(text).decode("ascii", errors="strict")
    t = "".join(str(text).split())
    if t.startswith("data:") and "," in t:                         # data:...;base64,<payload>
        t = t.split(",", 1)[1]
    t += "=" * (-len(t) % 4)
    try:
        if "-" in t or "_" in t:                                    # URL-safe alphabet
            return base64.b64decode(t.replace("-", "+").replace("_", "/"), validate=True)
        return base64.b64decode(t, validate=True)
    except Exception as e:  # noqa: BLE001  (never decode leniently: it would turn junk into a waveform)
        raise ValueError(f"not a Base64 string ({e})") from e


def container_of(raw: bytes) -> str:
    head = raw.lstrip(b"\xef\xbb\xbf \t\r\n")[:4]
    if raw[:2] == b"\x1f\x8b":
        return "gzip"
    if head.startswith(b"<"):
        return "xml"
    if raw[:2] == b"PK":
        return "npz"
    if head[:1] in (b"[", b"{"):
        return "json"
    return "raw"


def _smoothness(x: np.ndarray) -> float:
    """Mean |first difference| relative to spread: low for real ECG leads, high for scrambled samples."""
    d = np.abs(np.diff(x, axis=1)).mean()
    return float(d / (x.std() + 1e-9))


def raw_layout(flat: np.ndarray, n_leads: int, layout: str = "auto") -> Tuple[np.ndarray, str, Dict[str, float]]:
    """Flat integer samples -> (n_leads, N). 'lead_major' = all of lead 1 then lead 2 ...; 'interleaved' =
    sample 1 of every lead, then sample 2 ... 'auto' picks the smoother reading (reported)."""
    if flat.size % n_leads:
        raise ValueError(f"{flat.size} samples do not split into {n_leads} leads")
    lm = flat.reshape(n_leads, -1)
    il = flat.reshape(-1, n_leads).T
    scores = {"lead_major": _smoothness(lm), "interleaved": _smoothness(il)}
    if layout == "auto":
        layout = min(scores, key=scores.get)
    return (lm if layout == "lead_major" else il), layout, scores


def decode_waveform(value: Union[str, bytes, Dict[str, str]], n_leads: Optional[int] = None, fs: Optional[float] = None,
                    leads: Optional[List[str]] = None, units_per_bit: float = 4.88, units: str = "uV",
                    dtype: str = "<i2", layout: str = "auto") -> Record:
    """One waveform cell from the MCP table -> Record.

    value: the Base64 string; or {lead name: Base64 string} when each lead has its own column.
    Containers inside the Base64 are detected (MUSE XML, gzip, .npz, JSON). Raw integers need n_leads
    and fs from the table's columns; units_per_bit (default 4.88 uV, the GE MUSE value) and dtype must
    be CONFIRMED on real rows with probe_waveforms.py: a wrong scale is refused later by preprocess().
    """
    if isinstance(value, dict):
        names = list(value)
        sigs = [np.frombuffer(_b64(v), dtype=dtype).astype(np.float64) * units_per_bit for v in value.values()]
        n = min(len(x) for x in sigs)
        if fs is None:
            raise ValueError("per-lead waveform columns need the sample rate")
        return np.stack([x[:n] for x in sigs]), float(fs), names, units
    raw = _b64(value)
    kind = container_of(raw)
    if kind == "gzip":
        return decode_waveform(base64.b64encode(gzip.decompress(raw)).decode(), n_leads, fs, leads, units_per_bit, units, dtype, layout)
    if kind == "xml":
        return read_muse_xml(raw)
    if kind == "npz":
        return read_npz(raw)
    if kind == "json":
        obj = json.loads(raw.decode("utf-8"))
        if isinstance(obj, dict):
            names = [k for k in obj if isinstance(obj[k], list)]
            sig = np.asarray([obj[k] for k in names], float)
            fs = float(obj.get("fs") or obj.get("sample_rate") or fs or 0) or None
            if fs is None:
                raise ValueError("JSON waveform without a sample rate")
            return sig, fs, names, str(obj.get("units", units))
        sig = np.asarray(obj, float)
        if fs is None:
            raise ValueError("JSON waveform without a sample rate")
        return sig, float(fs), leads or (LEADS_12 if sig.shape[0] == 12 else LEADS_8), units
    if not n_leads or not fs:
        raise ValueError("raw Base64 samples need the lead count and sample rate from the waveform table")
    flat = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    sig, _, _ = raw_layout(flat, int(n_leads), layout)
    names = leads or (LEADS_12 if int(n_leads) == 12 else LEADS_8 if int(n_leads) == 8 else None)
    if names is None:
        raise ValueError(f"lead names needed for {n_leads} leads")
    return sig * units_per_bit, float(fs), names, units
