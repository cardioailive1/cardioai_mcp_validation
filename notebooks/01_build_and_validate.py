# %% [markdown]
# # CardioAI Phase 2 — cohort build & agent validation on Mayo Clinic Platform
# Study **CA-AGENTS-VAL-PHASE2** · 5 agent cohorts (Echo, ECG, CAD, Risk, Longitudinal)
#
# Run inside an MCP Discover **JupyterLab workspace** (Spark session available).
# Open as a notebook with Jupytext, or run the cells in order.
#
# 0. Onboarding: `python scripts/discover_schema.py` → fill `config/modality_tables.yaml`, `config/study.yaml`
# 1. Resolve concept sets → clinical sign-off of `exports/concept_sets_detail_for_review.csv`
# 2. Build cohorts → attrition, completeness, Table 1 (aggregate exports only)
# 3. Run each agent inside the workspace → `{cohort}_predictions`
# 4. Evaluate endpoints → `exports/{cohort}_endpoints.csv`

# %%
import sys, os
sys.path.insert(0, os.path.abspath(".."))
from pyspark.sql import SparkSession
from cardioai_mcp.config import load_study, load_cohort
from cardioai_mcp.engine import SparkEngine
from cardioai_mcp.pipeline import build_all

spark = SparkSession.builder.getOrCreate()
study = load_study()
eng = SparkEngine(spark, study)

# %% Build all five cohorts
res = build_all(eng, ["echo", "ecg", "cad", "risk", "longitudinal"])
for k, r in res.items():
    print(k); display(r["attrition"])  # noqa: F821  (Jupyter display)

# %% [markdown]
# ## Agent inference (inside the workspace — data never leaves)
# Each agent reads its cohort's index rows and writes a predictions table matching
# `predictions.columns` in the cohort YAML.  Model weights are uploaded into the
# workspace; only aggregate metrics are exported.
#
# * echo: A4C cine for each index echo (modality table `echo.dicom_uri`) → `echo_predictions`
# * ecg: 12-lead waveform (`ecg.waveform_uri`, resampled to 500 Hz/10 s) → `ecg_predictions`
# * cad / risk: tabular features = the cohort's input elements → `*_predictions`
# * longitudinal: daily landmark runs over `longitudinal_series` (day <= t only) → `longitudinal_predictions` (+ patient-level)

# %% Evaluate
from cardioai_mcp.evaluate import evaluate
from cardioai_mcp.qc import export
for k, r in res.items():
    c = r["cohort"]
    preds = spark.table(c.predictions["table"]).toPandas() if eng.table_exists(c.predictions["table"]) else None
    daily = series = None
    if k == "longitudinal" and preds is not None:
        daily = preds
        preds = spark.table("longitudinal_predictions_patient").toPandas()
        series = spark.table("longitudinal_series").toPandas()
    ep = evaluate(c, r["analytic"], preds, series=series, daily=daily, cluster_by_site=False)
    export(c, "endpoints", ep, study["export_dir"])
    display(ep)  # noqa: F821
