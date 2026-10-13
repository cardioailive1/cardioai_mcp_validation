"""Power of each study plan (A / B / C in config/study.yaml) for every endpoint with a closed-form power.

Usage:  python scripts/sample_size_report.py [--plans A B C] [--out exports/sample_size_plans.csv]

Success = 95% CI bound on the goal side clears the plan's threshold (deck goal, or
min(deck goal, Phase 1 - margin) under the margin rule).  Power is shown for two
assumptions about TRUE external performance:
  * phase1   — equals the Phase 1 estimate
  * degraded — halfway between Phase 1 and the deck goal (typical external shrinkage)
Denominators scale with each plan's N using the planning rates below (confirm with MCP
feasibility counts before locking N).
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cardioai_mcp import power as P  # noqa: E402
from cardioai_mcp.config import load_cohort  # noqa: E402
from cardioai_mcp.evaluate import phase1_scaled, success_threshold  # noqa: E402

RATES = dict(
    cad_obstructive=0.60,      # obstructive CAD among suspected-CAD ICA
    risk_ref_avail=0.90,       # angiography report with a readable stenosis grade
    risk_hd_prev=0.55,         # >= 50% stenosis among chest-pain patients referred to angiography
    hf_events_per_pt=0.35,     # HF exacerbations per patient per 180 d
    mi_30d=0.025, event_30d=0.15, acute_events_per_pt=0.30,
    echo_size_avail=0.85,
)


def denominators(key: str, c) -> dict:
    R = RATES
    q = c.sampling.get("quotas", {})
    N = c.target_n
    if key == "echo":
        return {"ef_correlation": N, "ef_mae": N, "clinical_agreement": N,
                "chamber_size": round(N * R["echo_size_avail"])}
    if key == "ecg":
        return {"overall_accuracy": N, "stemi_sensitivity": q.get("stemi"), "af_sensitivity": q.get("af"),
                "vt_sensitivity": q.get("vt"), "bbb_accuracy": N, "nsr_accuracy": N}
    if key == "cad":
        ica = q.get("ICA", N)
        pos, neg = round(ica * R["cad_obstructive"]), round(ica * (1 - R["cad_obstructive"]))
        return {"accuracy": ica, "sensitivity": pos, "specificity": neg, "auc": {"n_pos": pos, "n_neg": neg},
                "ppv": pos, "npv": neg, "multivessel": pos, "risk_agreement": ica}
    if key == "risk":
        comp = round(N * R["risk_ref_avail"])
        pos = round(comp * R["risk_hd_prev"])
        return {"auc_hd": {"n_pos": pos, "n_neg": comp - pos}, "accuracy": comp, "sens_hd": pos, "spec_hd": comp - pos}
    if key == "longitudinal":
        return {"hf_sens_7d": round(N * R["hf_events_per_pt"]),
                "mi_auc_30d": {"n_pos": round(N * R["mi_30d"]), "n_neg": round(N * (1 - R["mi_30d"]))},
                "readmission": N, "acute_24h_sens": round(N * R["acute_events_per_pt"]),
                "event_auc_30d": {"n_pos": round(N * R["event_30d"]), "n_neg": round(N * (1 - R["event_30d"]))},
                "risk_90d": N}
    return {}


PROP = {"sensitivity", "specificity", "accuracy", "binary_accuracy", "ppv", "npv", "alert_sensitivity"}


def power_of(metric: str, t: float, thr: float, denom) -> float | None:
    if metric in PROP:
        return P.power_proportion(int(denom), t, thr)
    if metric == "auc":
        return P.power_auc(t, thr, denom["n_pos"], denom["n_neg"])
    if metric == "pearson":
        return P.power_correlation(t, thr, int(denom))
    if metric == "kappa":
        return P.power_kappa(t, thr, int(denom))
    if metric == "mae":
        need = P.n_for_mae(t, thr, 0.603 * t / 0.798)
        return 1.0 if denom >= need else 0.5
    return None


def run(plans: list[str]) -> pd.DataFrame:
    rows = []
    for plan in plans:
        for key in ["echo", "ecg", "cad", "risk", "longitudinal"]:
            c = load_cohort(key, plan=plan)
            dens = denominators(key, c)
            prim = {e["id"] for e in c.endpoints["primary"]}
            for ep in c.endpoints["primary"] + c.endpoints["secondary"]:
                if ep["id"] not in dens or phase1_scaled(ep) is None:
                    continue
                thr, rule = success_threshold(c, ep)
                p1, goal, d = phase1_scaled(ep), ep["goal"], dens[ep["id"]]
                deg = (p1 + goal) / 2
                rows.append(dict(plan=plan, rule=c.plan["rule"], cohort=key, cohort_n=c.target_n, endpoint=ep["id"],
                                 label=ep["label"], primary=ep["id"] in prim, phase1=p1, deck_goal=goal, threshold=thr,
                                 denominator=d if not isinstance(d, dict) else f"{d['n_pos']}+/{d['n_neg']}-",
                                 power_phase1=round(power_of(ep["metric"], p1, thr, d), 3),
                                 power_degraded=round(power_of(ep["metric"], deg, thr, d), 3)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--out", default="exports/sample_size_plans.csv")
    a = ap.parse_args()
    df = run(a.plans)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    print(df[["plan", "cohort", "cohort_n", "endpoint", "primary", "threshold", "denominator", "power_phase1", "power_degraded"]].to_string(index=False))
    s = df[df.primary].groupby("plan").agg(primary_powered=("power_phase1", lambda x: int((x >= 0.8).sum())),
                                          n_primary=("power_phase1", "size"),
                                          primary_powered_degraded=("power_degraded", lambda x: int((x >= 0.8).sum())))
    print("\n", s.to_string())


if __name__ == "__main__":
    main()
