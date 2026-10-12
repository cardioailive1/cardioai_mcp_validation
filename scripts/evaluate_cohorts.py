"""Score one or more agents against their pre-specified endpoints (run inside the MCP workspace).

    python scripts/evaluate_cohorts.py --cohorts risk            # one agent
    python scripts/evaluate_cohorts.py --cohorts cad risk --boot 5000

Reads (from the Spark catalog, work schema first):
    {cohort}_analytic_locked     locked snapshot (runbook step 7c); falls back to {cohort}_analytic
    {cohort}_predictions         written by the agent (contract: `predictions` in the cohort YAML)
    longitudinal_predictions_patient, longitudinal_series   (longitudinal only)
Writes aggregate-only exports: exports/{cohort}_endpoints.csv
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cardioai_mcp.config import load_cohort, load_study  # noqa: E402
from cardioai_mcp.evaluate import evaluate  # noqa: E402
from cardioai_mcp.qc import export  # noqa: E402


def check_contract(c, preds, patient_level: bool = False) -> None:
    spec = c.predictions.get("patient_level" if patient_level else "columns", {})
    missing = [col for col in spec if col not in preds.columns]
    if "person_id" not in preds.columns:
        missing.insert(0, "person_id")
    if missing:
        raise SystemExit(f"{c.key}: predictions table is missing columns {missing} (see `predictions` in config/cohorts/{c.key}.yaml)")


def main():
    from pyspark.sql import SparkSession

    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", required=True)
    ap.add_argument("--boot", type=int, default=5000, help="bootstrap resamples (protocol: 5000 for the final analysis)")
    ap.add_argument("--cluster-by-site", action="store_true")
    a = ap.parse_args()
    spark = SparkSession.builder.getOrCreate()
    study = load_study()
    work = study["schemas"]["work"]

    def table(name):
        for full in (f"{work}.{name}", name):
            try:
                return spark.table(full).toPandas()
            except Exception:  # noqa: BLE001
                continue
        raise SystemExit(f"table {name} not found in {work} or as a temp view")

    for key in a.cohorts:
        c = load_cohort(key)
        try:  # the locked snapshot from runbook step 7c takes precedence
            analytic = table(f"{key}_analytic_locked")
        except SystemExit:
            print(f"WARNING: {key}_analytic_locked not found - scoring the unlocked {key}_analytic")
            analytic = table(f"{key}_analytic")
        series = daily = None
        if key == "longitudinal":
            daily = table(c.predictions["table"])
            check_contract(c, daily)
            preds = table("longitudinal_predictions_patient")
            check_contract(c, preds, patient_level=True)
            series = table("longitudinal_series")
        else:
            preds = table(c.predictions["table"])
            check_contract(c, preds)
        res = evaluate(c, analytic, preds, series=series, daily=daily, n_boot=a.boot, cluster_by_site=a.cluster_by_site)
        path = export(c, "endpoints", res, study.get("export_dir", "exports"))
        prim = res[res.tier == "primary"]
        verdict = "PASS" if prim.success.fillna(False).all() else "NOT MET"
        print(f"\n=== {c.title} (plan {res.plan.iloc[0]}): primary endpoints {verdict}")
        print(prim[["endpoint", "estimate", "ci_low", "ci_high", "success_threshold", "success"]].round(3).to_string(index=False))
        print(f"-> {path}")


if __name__ == "__main__":
    main()
