"""Compute every pre-specified endpoint for a cohort from analytic data + agent predictions."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import metrics as M
from .config import Cohort
from .qc import completeness
from .risk_scores import pce


def _weighted_accuracy(d: pd.DataFrame, pred: str, ref: str) -> float:
    w = d["sampling_weight"].astype(float)
    return float(((d[pred].astype(str) == d[ref].astype(str)) * w).sum() / w.sum())


def evaluate(cohort: Cohort, analytic: pd.DataFrame, predictions: pd.DataFrame | None = None,
             series: pd.DataFrame | None = None, daily: pd.DataFrame | None = None,
             n_boot: int = 2000, cluster_by_site: bool = False) -> pd.DataFrame:
    """analytic: {k}_analytic as pandas; predictions: one row per person_id (agent outputs);
    daily: longitudinal day-level predictions; series: {k}_series long table."""
    d = analytic.copy()
    if predictions is not None:
        d = d.merge(predictions, on=["person_id"], how="left", suffixes=("", "_pred"))
    if cohort.key == "risk":
        d["pce_risk"] = pce(d)
    if cohort.key == "echo" and "agent_lv_size" not in d and "agent_edv_ml" in d:
        # same rule as ref_lv_size: EDV indexed to BSA vs ASE upper limits (74 men / 61 women mL/m2)
        idx = pd.to_numeric(d["agent_edv_ml"], errors="coerce") / pd.to_numeric(d.get("bsa"), errors="coerce")
        lim = d["gender"].map({"Male": 74.0}).fillna(61.0)
        d["agent_lv_size"] = np.where(idx.isna(), None, np.where(idx > lim, "dilated", "normal"))
    cl = "site" if cluster_by_site else None
    out: list[M.Result] = []
    extra = []
    for tier in ("primary", "secondary"):
        for ep in cohort.endpoints[tier]:
            thr, rule = success_threshold(cohort, ep)
            r = _one(cohort, dict(ep, goal=thr), d, series, daily, n_boot, cl, tier)
            point = None
            if r.estimate is not None and ep.get("goal") is not None and not (isinstance(r.estimate, float) and math.isnan(r.estimate)):
                point = bool(r.estimate >= ep["goal"]) if ep.get("direction", "ge") in ("ge", "gt") else bool(r.estimate <= ep["goal"])
            extra.append(dict(deck_goal=ep.get("goal"), rule=rule, deck_goal_point_met=point))
            out.append(r)
    res = pd.DataFrame([r.as_dict() for r in out]).rename(columns={"goal": "success_threshold"})
    res = pd.concat([res, pd.DataFrame(extra)], axis=1)
    res.insert(0, "tier", [t for t in ("primary", "secondary") for _ in cohort.endpoints[t]])
    res.insert(1, "plan", cohort.raw.get("plan", {}).get("key"))
    na = cohort.endpoints.get("not_applicable") or []
    if na:   # deck endpoints the agent cannot produce: listed, never counted as failures
        res = pd.concat([res, pd.DataFrame([dict(tier="not_applicable", plan=res["plan"].iloc[0], endpoint=e["id"],
                                                 label=e["label"], deck_goal=e.get("deck_goal"), note="N/A: " + e["reason"])
                                            for e in na])], ignore_index=True)
    return res


PROPORTION_METRICS = {"sensitivity", "specificity", "accuracy", "binary_accuracy", "ppv", "npv",
                      "alert_sensitivity", "alert_specificity", "alert_ppv"}


def phase1_scaled(ep: dict) -> float | None:
    p1, goal = ep.get("phase1"), ep.get("goal")
    if p1 is None:
        return None
    return p1 / 100.0 if (p1 > 1 and goal is not None and goal <= 1) else p1  # Phase 1 values given in %


def success_threshold(cohort: Cohort, ep: dict) -> tuple[float | None, str]:
    """Deck rule: lower CI >= deck goal.  Margin rule: lower CI >= min(deck goal, Phase 1 - margin)
    (only for '>=' endpoints with a Phase 1 value and a margin for their metric)."""
    plan = cohort.raw.get("plan", {"rule": "goal", "margins": {}})
    p1 = phase1_scaled(ep)
    if plan["rule"] != "margin" or p1 is None or ep.get("direction", "ge") != "ge":
        return ep.get("goal"), "deck goal"
    m = ep["metric"]
    key = "proportion" if m in PROPORTION_METRICS else m
    margin = plan["margins"].get(key)
    if margin is None:
        return ep.get("goal"), "deck goal"
    thr = round(p1 - margin, 4)
    goal = ep.get("goal")
    if goal is not None and goal < thr:  # never stricter than the deck goal
        return goal, "deck goal (more lenient than Phase 1 - margin)"
    return thr, f"Phase 1 - {margin}"


def _one(c: Cohort, ep: dict, d: pd.DataFrame, series, daily, n_boot, cl, tier) -> M.Result:
    m, goal, direc = ep["metric"], ep.get("goal"), ep.get("direction", "ge")
    base = dict(endpoint=ep["id"], label=ep["label"], goal=goal, direction=direc, phase1=ep.get("phase1"))
    if ep.get("population"):
        d = d.query(ep["population"].replace("=", "==").replace("<==", "<=").replace(">==", ">=").replace("!==", "!="), engine="python")
    try:
        if m == "completeness":
            comp = completeness(c, d)
            req = comp[comp.required]
            est = float(req.observed.mean())
            return M.Result(**base, estimate=est, ci_low=est, ci_high=est, n=len(d), success=M.decide(est, est, est, goal, direc),
                            note="mean observed completeness of required elements")
        if m in ("alert_sensitivity", "alert_specificity", "alert_ppv", "alert_lead_time", "temporal_cv"):
            return _alerts(c, ep, base, series, daily)
        pred, ref = ep.get("pred"), ep.get("ref")
        cols = [x for x in (pred, ref, ep.get("comparator"), ep.get("time"), ep.get("event")) if x]
        sub = d.dropna(subset=cols)
        n = len(sub)
        if n == 0:
            return M.Result(**base, estimate=None, ci_low=None, ci_high=None, n=0, success=None, note="no evaluable rows")
        if m in ("sensitivity", "specificity", "ppv", "npv", "binary_accuracy"):
            pos = ep.get("positive", 1)
            y, p = M._bin(sub[ref], pos), M._bin(sub[pred], pos)
            if m == "sensitivity":
                k, nn = int(((p == 1) & (y == 1)).sum()), int((y == 1).sum())
            elif m == "specificity":
                k, nn = int(((p == 0) & (y == 0)).sum()), int((y == 0).sum())
            elif m == "ppv":
                k, nn = int(((p == 1) & (y == 1)).sum()), int((p == 1).sum())
            elif m == "npv":
                k, nn = int(((p == 0) & (y == 0)).sum()), int((p == 0).sum())
            else:
                k, nn = int((p == y).sum()), len(y)
            est = k / nn if nn else math.nan
            lo, hi = M.wilson(k, nn)
            note = ""
            if m in ("ppv", "npv") and sub["sampling_weight"].nunique() > 1:
                note = "enriched sample: PPV/NPV also re-weighted to natural prevalence in supplementary table"
            return M.Result(**base, estimate=est, ci_low=lo, ci_high=hi, n=nn, success=M.decide(est, lo, hi, goal, direc), note=note)
        if m == "accuracy":
            if ep.get("weighted") and sub["sampling_weight"].nunique() > 1:
                f = lambda b: _weighted_accuracy(b, pred, ref)  # noqa: E731
                est = f(sub)
                lo, hi = M.bootstrap_ci(f, sub, n_boot, cluster=cl)
                note = "weighted to natural prevalence (inverse sampling fraction); bootstrap CI"
            else:
                k = int((sub[pred].astype(str) == sub[ref].astype(str)).sum())
                est = k / n
                lo, hi = M.wilson(k, n)
                note = ""
            return M.Result(**base, estimate=est, ci_low=lo, ci_high=hi, n=n, success=M.decide(est, lo, hi, goal, direc), note=note)
        if m == "auc":
            f = lambda b: M.auc(b[ref].astype(int).to_numpy(), b[pred].astype(float).to_numpy())  # noqa: E731
        elif m == "pearson":
            f = lambda b: float(np.corrcoef(b[pred].astype(float), b[ref].astype(float))[0, 1])  # noqa: E731
        elif m == "mae":
            f = lambda b: float((b[pred].astype(float) - b[ref].astype(float)).abs().mean())  # noqa: E731
        elif m == "kappa":
            f = lambda b: M.cohen_kappa(b[pred], b[ref])  # noqa: E731
        elif m == "f1_macro":
            f = lambda b: M.f1_macro(b[ref], b[pred])  # noqa: E731
        elif m == "median":
            f = lambda b: float(b[pred].astype(float).median())  # noqa: E731
        elif m == "c_index":
            f = lambda b: M.harrell_c(b[ep["time"]].astype(float).to_numpy(), b[ep["event"]].astype(int).to_numpy(),  # noqa: E731
                                      b[pred].astype(float).to_numpy())
        elif m == "nri":
            cuts = c.raw.get("comparator", {}).get("categories", [0.05, 0.075, 0.20])
            f = lambda b: M.nri_categorical(b[ref].astype(int).to_numpy(), b[pred].astype(float).to_numpy(),  # noqa: E731
                                            b[ep["comparator"]].astype(float).to_numpy(), cuts)
        elif m == "auc_difference":
            y = sub[ref].astype(int).to_numpy()
            dlt, se, p = M.delong(y, sub[pred].astype(float).to_numpy(), sub[ep["comparator"]].astype(float).to_numpy())
            lo, hi = dlt - 1.959964 * se, dlt + 1.959964 * se
            return M.Result(**base, estimate=dlt, ci_low=lo, ci_high=hi, n=n, success=M.decide(dlt, lo, hi, goal, direc),
                            note=f"paired DeLong; AUC(agent) - AUC(PCE); p={p:.3g}")
        elif m == "hosmer_lemeshow":
            stat, p = M.hosmer_lemeshow(sub[ref].astype(int).to_numpy(), sub[pred].astype(float).to_numpy())
            return M.Result(**base, estimate=p, ci_low=None, ci_high=None, n=n, success=bool(p > goal),
                            note=f"HL chi2={stat:.2f}, 10 groups; success = p > {goal}")
        else:
            raise ValueError(f"unknown metric {m}")
        est = f(sub)
        lo, hi = M.bootstrap_ci(f, sub, n_boot, cluster=cl)
        return M.Result(**base, estimate=est, ci_low=lo, ci_high=hi, n=n, success=M.decide(est, lo, hi, goal, direc),
                        note=f"percentile bootstrap ({n_boot})" + (", site-clustered" if cl else ""))
    except Exception as ex:  # noqa: BLE001
        return M.Result(**base, estimate=None, ci_low=None, ci_high=None, n=0, success=None, note=f"ERROR: {ex}")


def _alerts(c: Cohort, ep: dict, base: dict, series: pd.DataFrame | None, daily: pd.DataFrame | None) -> M.Result:
    if daily is None or series is None:
        return M.Result(**base, estimate=None, ci_low=None, ci_high=None, n=0, success=None, note="needs daily predictions + series")
    h = int(ep.get("horizon", 7))
    if ep.get("event") == "acute":
        events = series[series.element == "acute_cardiac_dates"][["person_id", "day"]]
        col = "alert_acute_24h"
    else:
        events = series[series.element == "hf_exacerbation_dates"][["person_id", "day"]]
        col = "alert_hf_7d"
    r = M.alert_metrics(daily, events, h, col)
    m = ep["metric"]
    if m == "temporal_cv":
        vals = []
        for lo, hi in [(0, 30), (31, 90), (91, 179)]:
            dd = daily[(daily.day >= lo) & (daily.day <= hi)]
            ev = events[(events.day > lo) & (events.day <= hi + h)]
            vals.append(M.alert_metrics(dd, ev, h, col)["sensitivity"])
        vals = np.array([v for v in vals if not math.isnan(v)])
        cv = float(vals.std(ddof=1) / vals.mean()) if len(vals) > 1 and vals.mean() > 0 else math.nan
        return M.Result(**base, estimate=cv, ci_low=None, ci_high=cv, n=len(vals), success=bool(cv < ep["goal"]) if not math.isnan(cv) else None,
                        note="CV of 7-d sensitivity across days 0-30 / 31-90 / 91-179")
    key, nkey = {"alert_sensitivity": ("sensitivity", "n_events"), "alert_specificity": ("specificity", "n_neg_windows"),
                 "alert_ppv": ("ppv", "n_alerts"), "alert_lead_time": ("lead_time", "n_detected")}[m]
    est, n = r[key], r[nkey]
    if m == "alert_lead_time":
        return M.Result(**base, estimate=est, ci_low=None, ci_high=None, n=n, success=bool(est >= ep["goal"]) if not math.isnan(est) else None,
                        note="mean days first alert -> event among detected events")
    k = int(round(est * n)) if not math.isnan(est) else 0
    lo, hi = M.wilson(k, n)
    return M.Result(**base, estimate=est, ci_low=lo, ci_high=hi, n=n, success=M.decide(est, lo, hi, ep["goal"], ep["direction"]),
                    note="Wilson CI treats windows as independent; final analysis uses patient-clustered bootstrap")
