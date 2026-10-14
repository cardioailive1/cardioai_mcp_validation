"""Read 12-lead ECG waveforms from the formats an MCP workspace may hold.

Every reader returns (signal (n_leads, N) in the file's units, fs, lead names, units),
which goes straight into runtime.preprocess.preprocess(). Which format MCP uses is
confirmed at onboarding (config/modality_tables.yaml, ecg.columns.waveform_uri).
"""
from __future__ import annotations

import base64
import io as _io
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple

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
