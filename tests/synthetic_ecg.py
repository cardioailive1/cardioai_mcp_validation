"""Synthetic 12-lead ECGs with known findings, for tests (no patient data).

Beats are sums of Gaussian waves (P, Q, R, S, T) with lead-specific QRS polarity.
  normal : sinus rhythm 55-95 bpm, mild RR variability, P waves
  af     : irregularly irregular RR, no P waves, 5-7 Hz fibrillatory waves
  stemi  : ST elevation 0.2-0.35 mV from J point to T in one territory
           (anterior V1-V4 or inferior II/III/aVF) with reciprocal depression
  nstemi : ST depression 0.1-0.2 mV and T inversion in V4-V6, I, aVL
Baseline wander and noise are added. Amplitudes in mV.
"""
from __future__ import annotations

import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
QRS_GAIN = np.array([0.9, 1.2, 0.5, -0.9, 0.5, 0.8, -0.7, 0.4, 0.9, 1.3, 1.2, 1.0])
ANTERIOR = [6, 7, 8, 9]
INFERIOR = [1, 2, 5]
LATERAL = [0, 4, 10, 11]


def _g(t, mu, sd, a):
    return a * np.exp(-0.5 * ((t - mu) / sd) ** 2)


def ecg(kind: str = "normal", fs: float = 500.0, seconds: float = 10.0, seed: int = 0) -> np.ndarray:
    rnd = np.random.default_rng(seed)
    n = int(round(fs * seconds))
    t = np.arange(n) / fs
    sig = np.zeros((12, n))
    if kind == "af":
        beats, x = [], rnd.uniform(0, 0.4)
        while x < seconds:
            beats.append(x)
            x += rnd.uniform(0.38, 1.15)
    else:
        rr = 60.0 / rnd.uniform(55, 95)
        grid = np.arange(rnd.uniform(0, rr), seconds, rr)
        beats = list(grid + rnd.normal(0, 0.015, len(grid)))           # mild sinus variability
    st = np.zeros(12)
    tw = np.full(12, 0.25) * np.sign(QRS_GAIN).clip(min=-0.4)
    if kind == "stemi":
        up, down = (ANTERIOR, INFERIOR) if rnd.random() < 0.5 else (INFERIOR, [4, 0])
        st[up] = rnd.uniform(0.2, 0.35)
        st[down] = -rnd.uniform(0.05, 0.15)
        tw[up] += 0.2
    elif kind == "nstemi":
        st[[9, 10, 11, 0, 4]] = -rnd.uniform(0.1, 0.2)
        tw[[9, 10, 11, 0, 4]] = -0.2
    for b in beats:
        for li in range(12):
            g = QRS_GAIN[li]
            s = 0.0
            if kind != "af":
                s += _g(t, b - 0.16, 0.022, 0.12 * np.sign(g + 0.3))
            s = s + _g(t, b - 0.02, 0.008, -0.1 * abs(g)) + _g(t, b, 0.011, g) + _g(t, b + 0.025, 0.009, -0.25 * abs(g))
            if st[li] != 0:
                seg = (t >= b + 0.05) & (t <= b + 0.26)
                ramp = np.clip((t - (b + 0.04)) / 0.02, 0, 1) * np.clip((b + 0.30 - t) / 0.06, 0, 1)
                s = s + st[li] * np.where(seg | (ramp > 0), ramp, 0)
            s = s + _g(t, b + 0.30, 0.045, tw[li])
            sig[li] += s
    if kind == "af":
        f = rnd.uniform(5, 7)
        sig += 0.05 * np.sin(2 * np.pi * f * t + rnd.uniform(0, 6.28))[None] * np.array([0.6, 1, 0.8, 0.6, 0.4, 0.8, 1.2, 0.8, 0.5, 0.4, 0.3, 0.3])[:, None]
    sig += 0.2 * np.sin(2 * np.pi * rnd.uniform(0.15, 0.4) * t + rnd.uniform(0, 6.28))[None]
    sig += rnd.normal(0, 0.02, sig.shape)
    i, ii = sig[0].copy(), sig[1].copy()                      # limb leads obey Einthoven / Goldberger, as in real ECGs
    sig[2], sig[3], sig[4], sig[5] = ii - i, -(i + ii) / 2, i - ii / 2, ii - i / 2
    return sig


LABELS = {"normal": (0, 0, 0), "af": (0, 0, 1), "stemi": (1, 1, 0), "nstemi": (1, 0, 0)}   # acute_mi, stemi, af


def dataset(n: int, fs: float = 500.0, seed: int = 0, mix=(0.4, 0.25, 0.2, 0.15)):
    """n synthetic ECGs -> list of (signal, fs, labels tuple, kind)."""
    rnd = np.random.default_rng(seed)
    kinds = rnd.choice(["normal", "af", "stemi", "nstemi"], size=n, p=list(mix))
    return [(ecg(k, fs, seed=int(seed * 100000 + i)), fs, LABELS[k], k) for i, k in enumerate(kinds)]
