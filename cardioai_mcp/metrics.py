"""Validation statistics for the pre-specified endpoints.

Decision rule (all endpoints): success when the 95% two-sided CI bound on the
side of the goal clears the goal — i.e. a one-sided 97.5% test against the
performance goal (same convention as Mayo Clinic Platform ECG-AI external
validations, npj Digit Med 2026).  Proportions use Wilson intervals (exact
Clopper-Pearson available); everything else uses a (site-clustered optional)
percentile bootstrap.  Within a cohort all primary endpoints must succeed
(intersection-union test), so no alpha adjustment is needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Result:
    endpoint: str
    label: str
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    n: int
    goal: float | None
    direction: str
    phase1: float | None
    success: bool | None
    note: str = ""

    def as_dict(self):
        return asdict(self)


# ------------------------------------------------------------------ intervals
def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (lo, hi)


def bootstrap_ci(fn, df: pd.DataFrame, n_boot: int = 2000, seed: int = 20261012, cluster: str | None = None,
                 alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    if cluster:
        groups = {k: g for k, g in df.groupby(cluster)}
        keys = list(groups)
        for _ in range(n_boot):
            pick = rng.choice(len(keys), len(keys), replace=True)
            b = pd.concat([groups[keys[i]] for i in pick], ignore_index=True)
            try:
                vals.append(fn(b))
            except Exception:  # noqa: BLE001
                continue
    else:
        n = len(df)
        arr = df.reset_index(drop=True)
        for _ in range(n_boot):
            b = arr.iloc[rng.integers(0, n, n)]
            try:
                vals.append(fn(b))
            except Exception:  # noqa: BLE001
                continue
    vals = np.array([v for v in vals if v is not None and not np.isnan(v)])
    if len(vals) == 0:
        return (math.nan, math.nan)
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


# ------------------------------------------------------------------ point metrics
def _bin(series: pd.Series, positive) -> np.ndarray:
    return (series.astype(str) == str(positive)).astype(int).to_numpy()


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def delong(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> tuple[float, float, float]:
    """Paired DeLong: returns (auc1 - auc2, se, two-sided p)."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y)
    y, s1, s2 = y[order], np.asarray(s1, float)[order], np.asarray(s2, float)[order]
    m = int(y.sum())
    n = len(y) - m

    def comps(s):
        x, z = s[:m], s[m:]
        v10 = np.array([(np.sum(xi > z) + 0.5 * np.sum(xi == z)) / n for xi in x])
        v01 = np.array([(np.sum(x > zj) + 0.5 * np.sum(x == zj)) / m for zj in z])
        return v10.mean(), v10, v01

    a1, v10a, v01a = comps(s1)
    a2, v10b, v01b = comps(s2)
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    se = math.sqrt(max(var, 1e-12))
    d = a1 - a2
    return d, se, 2 * (1 - stats.norm.cdf(abs(d) / se))


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    a = np.asarray(a).astype(str)
    b = np.asarray(b).astype(str)  # positional (bootstrap samples carry duplicate index labels)
    cats = sorted(set(a) | set(b))
    if len(cats) < 2:
        return math.nan
    idx = {c: i for i, c in enumerate(cats)}
    ct = np.zeros((len(cats), len(cats)))
    np.add.at(ct, (np.array([idx[x] for x in a]), np.array([idx[x] for x in b])), 1)
    n = ct.sum()
    po = np.trace(ct) / n
    pe = (ct.sum(0) * ct.sum(1)).sum() / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else math.nan


def f1_macro(y: pd.Series, p: pd.Series) -> float:
    y, p = np.asarray(y).astype(str), np.asarray(p).astype(str)
    f1s = []
    for c in sorted(set(y)):
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f1s))


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, groups: int = 10) -> tuple[float, float]:
    d = pd.DataFrame({"y": y, "p": p}).sort_values("p")
    d["g"] = pd.qcut(d["p"].rank(method="first"), groups, labels=False)
    g = d.groupby("g").agg(obs=("y", "sum"), exp=("p", "sum"), n=("y", "size"))
    hl = (((g.obs - g.exp) ** 2) / (g.exp * (1 - g.exp / g.n)).clip(lower=1e-9)).sum()
    return float(hl), float(1 - stats.chi2.cdf(hl, groups - 2))


def nri_categorical(y: np.ndarray, p_new: np.ndarray, p_old: np.ndarray, cuts: list[float]) -> float:
    c_new = np.digitize(p_new, cuts)
    c_old = np.digitize(p_old, cuts)
    up, down = c_new > c_old, c_new < c_old
    ev, ne = y == 1, y == 0
    nri_e = (up[ev].mean() - down[ev].mean()) if ev.any() else 0.0
    nri_ne = (down[ne].mean() - up[ne].mean()) if ne.any() else 0.0
    return float(nri_e + nri_ne)


def harrell_c(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    time, event, risk = map(np.asarray, (time, event, risk))
    order = np.argsort(time)
    time, event, risk = time[order], event[order], risk[order]
    conc = ties = comp = 0.0
    for i in range(len(time)):
        if not event[i]:
            continue
        later = time > time[i]
        comp += later.sum()
        conc += (risk[i] > risk[later]).sum()
        ties += (risk[i] == risk[later]).sum()
    return float((conc + 0.5 * ties) / comp) if comp else math.nan


# ------------------------------------------------------------------ alert (window-level) metrics
def alert_metrics(pred: pd.DataFrame, events: pd.DataFrame, horizon: int, alert_col: str,
                  monitoring_days: int = 180) -> dict:
    """Event- and window-level metrics for rolling alerts.

    pred:   person_id, day, <alert_col>  (agent run each day; uses only data up to `day`)
    events: person_id, day               (event days, e.g. HF exacerbation admission/ED dates)
    * sensitivity: share of events with >=1 alert on days [event-horizon, event-1]
    * specificity: share of patient-days with NO event in (day, day+horizon] that have no alert
    * PPV:         share of alert-days followed by an event in (day, day+horizon]
    * lead time:   mean (event day - first alert day) among detected events
    """
    p = pred[["person_id", "day", alert_col]].rename(columns={alert_col: "alert"})
    ev = events[["person_id", "day"]].drop_duplicates()
    # label each patient-day: event within (day, day+h]
    lab = p.merge(ev.rename(columns={"day": "eday"}), on="person_id", how="left")
    lab["hit"] = (lab.eday > lab.day) & (lab.eday <= lab.day + horizon)
    lab = lab.groupby(["person_id", "day", "alert"], as_index=False)["hit"].max()
    neg = lab[~lab.hit]
    spec = float((neg.alert == 0).mean()) if len(neg) else math.nan
    al = lab[lab.alert == 1]
    ppv = float(al.hit.mean()) if len(al) else math.nan
    # event level
    det, leads = 0, []
    alerts = p[p.alert == 1]
    for r in ev.itertuples():
        a = alerts[(alerts.person_id == r.person_id) & (alerts.day >= r.day - horizon) & (alerts.day < r.day)]
        if len(a):
            det += 1
            leads.append(r.day - a.day.min())
    sens = det / len(ev) if len(ev) else math.nan
    return {"sensitivity": sens, "n_events": int(len(ev)), "specificity": spec, "n_neg_windows": int(len(neg)),
            "ppv": ppv, "n_alerts": int(len(al)), "lead_time": float(np.mean(leads)) if leads else math.nan,
            "n_detected": det}


# ------------------------------------------------------------------ decision helper
def decide(est, lo, hi, goal, direction) -> bool | None:
    if goal is None or est is None or (isinstance(est, float) and math.isnan(est)):
        return None
    if direction in ("ge", "gt"):
        return bool(lo >= goal) if direction == "ge" else bool(lo > goal)
    return bool(hi <= goal) if direction == "le" else bool(hi < goal)
