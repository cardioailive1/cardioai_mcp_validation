"""Build the Phase 2 validation cohorts.

Inside the Mayo Clinic Platform workspace (BigQuery client):
    python scripts/run_cohorts.py --cohorts echo ecg cad risk longitudinal --save-tables

Offline smoke test on synthetic OMOP (DuckDB):
    python scripts/run_cohorts.py --engine duckdb-synthetic
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cardioai_mcp import COHORTS, EXTRA_COHORTS  # noqa: E402
from cardioai_mcp.config import load_study  # noqa: E402
from cardioai_mcp.pipeline import build_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["bigquery", "spark", "duckdb-synthetic"], default="bigquery")
    ap.add_argument("--cohorts", nargs="+", default=COHORTS, choices=COHORTS + EXTRA_COHORTS)
    ap.add_argument("--export-dir", default=None)
    ap.add_argument("--save-tables", action="store_true", help="persist analytic + series tables to the work schema")
    a = ap.parse_args()
    study = load_study()
    if a.engine in ("bigquery", "spark"):
        from cardioai_mcp.engine import make_engine

        eng = make_engine(a.engine, study)
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
        if a.save_tables and a.engine != "duckdb-synthetic":
            for v in (f"{k}_analytic", f"{k}_series"):
                if eng.table_exists(v):
                    print(f"saved {eng.save_view(v)}")


if __name__ == "__main__":
    main()
