"""Build the Phase 2 validation cohorts.

Inside a Mayo Clinic Platform workspace (Spark session available):
    python scripts/run_cohorts.py --engine spark --cohorts echo ecg cad risk longitudinal

Offline smoke test on synthetic OMOP (DuckDB):
    python scripts/run_cohorts.py --engine duckdb-synthetic
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cardioai_mcp import COHORTS  # noqa: E402
from cardioai_mcp.config import load_study  # noqa: E402
from cardioai_mcp.pipeline import build_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["spark", "duckdb-synthetic"], default="spark")
    ap.add_argument("--cohorts", nargs="+", default=COHORTS)
    ap.add_argument("--export-dir", default=None)
    ap.add_argument("--save-tables", action="store_true", help="persist analytic + series tables to the work schema (spark)")
    a = ap.parse_args()
    study = load_study()
    if a.engine == "spark":
        from pyspark.sql import SparkSession

        from cardioai_mcp.engine import SparkEngine

        spark = SparkSession.builder.getOrCreate()
        eng = SparkEngine(spark, study)
    else:
        from cardioai_mcp.engine import DuckDBEngine
        from tests.synthetic_omop import build

        study["schemas"].update(cdm="main", vocab="main", modality="main", work="main")
        eng = DuckDBEngine(build(6000), study)
    res = build_all(eng, a.cohorts, study, a.export_dir)
    for k, r in res.items():
        print(f"\n=== {k}: {r['cohort'].title}")
        print(r["attrition"].to_string(index=False))
        comp = r["completeness"]
        print(f"required elements below expected completeness: {int((comp.required & (comp.flag != '')).sum())}")
        if a.save_tables and a.engine == "spark":
            for v in (f"{k}_analytic", f"{k}_series"):
                if eng.table_exists(v):
                    spark.table(v).write.mode("overwrite").saveAsTable(f"{study['schemas']['work']}.{v}")


if __name__ == "__main__":
    main()
