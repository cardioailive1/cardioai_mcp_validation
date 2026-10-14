"""12-lead ECG preprocessing shared by training, the validation study and serving.

This file (with model.py) is the model's input contract. It is kept byte-identical
wherever the model runs and locked by fingerprint (runtime.fingerprint()).

Pipeline (preprocess):
  1. leads -> standard order I, II, III, aVR, aVL, aVF, V1-V6; the 4 limb leads are
     derived from I and II when only 8 leads are stored (GE MUSE stores I, II, V1-V6)
  2. units -> millivolts (checked: implausible amplitudes are refused, catching uV/mV mix-ups)
  3. resample to 500 Hz (polyphase, anti-aliased)
  4. zero-phase high-pass 0.67 Hz (baseline wander) and low-pass 100 Hz on the full
     recording. A bidirectional filter at 0.67 Hz keeps the ST segment undistorted,
     which the STEMI output depends on
  5. 10 s window: centre crop; 9.5-10 s recordings are edge-padded; shorter is refused
  6. fixed scaling (mV, clipped to +/-10 mV); NO per-lead z-scoring, so absolute and
     between-lead amplitudes (ST elevation in mm, voltage) are preserved
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional, Sequence

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

PREPROCESS_VERSION = "1.0.0"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 500
SECONDS = 10
N_SAMPLES = FS * SECONDS
MIN_SECONDS = 9.5
HP_HZ, LP_HZ = 0.67, 100.0
CLIP_MV = 10.0

_ALIASES = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF", "LEAD I": "I", "LEAD II": "II", "LEAD III": "III"}


class ECGInputError(ValueError):
    """The recording cannot be prepared the way the model was trained (client error)."""


def _canon(name: str) -> str:
    n = str(name).strip().replace("-", "").replace("_", " ")
    up = n.upper()
    if up in _ALIASES:
        return _ALIASES[up]
    if up.startswith("V") and up[1:].isdigit():
        return "V" + up[1:]
    return {"I": "I", "II": "II", "III": "III"}.get(up, n)


def standard_leads(signal: np.ndarray, leads: Sequence[str]) -> np.ndarray:
    """(n_leads, N) in any order -> (12, N) in LEADS order; derives III/aVR/aVL/aVF if absent."""
    sig = np.asarray(signal, dtype=np.float64)
    if sig.ndim != 2 or sig.shape[0] != len(leads):
        raise ECGInputError(f"signal shape {sig.shape} does not match {len(leads)} lead names")
    by = {_canon(l): sig[i] for i, l in enumerate(leads)}
    if "I" in by and "II" in by:
        i, ii = by["I"], by["II"]
        by.setdefault("III", ii - i)
        by.setdefault("aVR", -(i + ii) / 2)
        by.setdefault("aVL", i - ii / 2)
        by.setdefault("aVF", ii - i / 2)
    missing = [l for l in LEADS if l not in by]
    if missing:
        raise ECGInputError(f"missing leads: {missing}")
    return np.stack([by[l] for l in LEADS])


def to_millivolts(sig: np.ndarray, units: str = "mV") -> np.ndarray:
    u = units.strip().lower().replace("µ", "u")
    scale = {"mv": 1.0, "uv": 1e-3, "v": 1e3}.get(u)
    if scale is None:
        raise ECGInputError(f"unknown amplitude unit {units!r}")
    out = sig * scale
    p99 = float(np.percentile(np.abs(out), 99))
    if not 0.02 <= p99 <= 20.0:
        raise ECGInputError(f"implausible ECG amplitude (99th percentile {p99:.3g} mV); check units")
    return out


def resample(sig: np.ndarray, fs: float) -> np.ndarray:
    if fs <= 0:
        raise ECGInputError("sample rate must be positive")
    if abs(fs - FS) < 1e-6:
        return sig
    frac = Fraction(FS / fs).limit_denominator(1000)
    return resample_poly(sig, frac.numerator, frac.denominator, axis=1)


def window(sig: np.ndarray) -> np.ndarray:
    n = sig.shape[1]
    if n < MIN_SECONDS * FS:
        raise ECGInputError(f"recording is {n / FS:.2f} s; at least {MIN_SECONDS} s required")
    if n >= N_SAMPLES:
        start = (n - N_SAMPLES) // 2
        return sig[:, start:start + N_SAMPLES]
    return np.pad(sig, ((0, 0), (0, N_SAMPLES - n)), mode="edge")


_SOS_HP = butter(2, HP_HZ, btype="highpass", fs=FS, output="sos")
_SOS_LP = butter(4, LP_HZ, btype="lowpass", fs=FS, output="sos")


def filter_(sig: np.ndarray) -> np.ndarray:
    return sosfiltfilt(_SOS_LP, sosfiltfilt(_SOS_HP, sig, axis=1), axis=1)


def preprocess(signal: np.ndarray, fs: float, leads: Optional[Sequence[str]] = None, units: str = "mV") -> np.ndarray:
    """Raw recording (n_leads, N) -> model input float32 (12, 5000)."""
    sig = np.asarray(signal, dtype=np.float64)
    if sig.ndim == 2 and leads is not None and sig.shape[0] != len(leads) and sig.shape[1] == len(leads):
        sig = sig.T                                   # accept (N, n_leads)
    sig = standard_leads(sig, leads or LEADS)
    if not np.isfinite(sig).all():
        raise ECGInputError("signal contains NaN or infinite values")
    sig = to_millivolts(sig, units)
    sig = filter_(resample(sig, fs))                  # filter the full recording, then crop:
    sig = window(sig)                                 # filter edge effects stay outside the window
    return np.clip(sig, -CLIP_MV, CLIP_MV).astype(np.float32)
