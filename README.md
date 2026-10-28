# CardioAI Phase 2 — Agent Validation Cohorts on Mayo Clinic Platform

Study **CA-AGENTS-VAL-PHASE2**. This toolkit builds and evaluates the five agent-validation cohorts from the Phase 2 data-elements deck: Echo, ECG, CAD, Risk Stratification and Longitudinal Monitoring. The Women's CVD cohort and the five risk calculators are out of scope.

It runs inside a **Mayo Clinic Platform (MCP) Discover workspace**. There the EHR data is in **OMOP CDM** format, provisioned in **BigQuery** and queried with the BigQuery client from JupyterLab. The SQL is written in Spark SQL and translated to BigQuery SQL by the engine (sqlglot). The same code runs offline on DuckDB against synthetic OMOP data for testing.

| Cohort | Study ID | N | Sites | Elements (REQ) | Reference standard |
|---|---|---|---|---|---|
| Echo | …-ECHO | 1,500 | 12 | 27 (19) | Expert LVEF and report findings on the index TTE |
| ECG | …-ECG | 2,000 | 12 | 30 (22) | Confirmed ECG over-read, plus clinical STEMI/NSTEMI diagnosis |
| CAD | …-CAD | 600 (Plan C: 1,300) | 8 | 41 (27) | Invasive angiography ≥50% stenosis (CCTA CAD-RADS ≥3 secondary) |
| Risk | …-RISK | 1,000 (Plan C: 1,500) | 12 | 41 (33) | Heart disease at angiography (≥ 50% stenosis), Cleveland model; MACE out of scope |
| Longitudinal | …-LONG | 1,000 (Plan C: 2,000) | 12 | 40 (27) | HF exacerbations and events over a 180-day monitoring window |

All 179 elements are mapped. The loader refuses to start if any cohort's element count, REQ count or average expected completeness differs from the deck.

A sixth cohort, **Cardiac MRI** (`…-MRI`, 1,000 patients, `config/cohorts/mri.yaml`), validates the MRI agent added after the deck. It covers infarct confirmation after STEMI, infarct size, MVO and LVEF, STEMI mimics, LV thrombus and aneurysm, and 1-year MACE. It has no deck targets, so it is listed in `EXTRA_COHORTS` and is built with `--cohorts mri`.

Models trained inside MCP, with MobileNetV3 weights approved for export, live next to the toolkit:
- `ecg_training/`: acute MI, STEMI and AF from 12-lead ECGs;
- `mri_training/`: infarct, scar extent and STEMI work-up from cardiac MRI.
- `echo_wm_training/`: regional (17 AHA segments, LAD / RCA / LCx territories) and global (LVEF, global hypokinesis, WMSI) wall motion from apical echo cines; extends the echo agent.

Each has its own README. `annotation/` prepares Mayo images for readers inside the workspace (MRI contours; optional echo traces), checks what they draw and writes the training labels. It also audits the automatic labels (data scientists first, cardiologist adjudication): [annotation/LABEL_AUDIT.md](annotation/LABEL_AUDIT.md).

## Layout
```
config/
  study.yaml              schemas, study window, sites, small-cell rule, unit conversions
  concept_sets.yaml       92 concept sets (ICD-10-CM/ICD-9-CM -> SNOMED via 'Maps to', LOINC, CPT4, ATC -> RxNorm)
  modality_tables.yaml    MCP ECG-waveform / echo-DICOM table names (fill at onboarding)
  cohorts/{echo,ecg,cad,risk,longitudinal,mri}.yaml
                          index event, exclusions, sampling quotas, every data element, agent
                          output contract, endpoints with goals and Phase 1 values
cardioai_mcp/
  engine.py      BigQueryEngine (MCP) / SparkEngine / DuckDBEngine (tests); Spark SQL transpiled with sqlglot
  concepts.py    set-based concept-set resolution -> cs_members
  extractors.py  14 element extractors (measurement, flag, outcome, death, visit_count, note NLP, modality, series ...)
  builder.py     candidates -> exclusions (attrition) -> 1/patient -> sites -> stratified sampling + weights -> elements
  evaluate.py    computes every endpoint (Wilson / bootstrap CI, DeLong, HL, NRI, C-index, alert-window metrics)
  metrics.py, power.py, risk_scores.py (ACC/AHA PCE, descriptive only), qc.py (completeness, Table 1, small-cell suppression)
scripts/
  discover_schema.py      onboarding inventory (tables, modality candidates, care sites, note titles)
  run_cohorts.py          build cohorts (bigquery | spark | duckdb-synthetic)
  sample_size_report.py   power of the planned N for every endpoint -> exports/sample_size.csv
notebooks/01_build_and_validate.py   end-to-end MCP notebook (Jupytext percent format)
docs/data_dictionary_omop_mapping.csv   179 elements -> OMOP source, window, fill / transform
tests/            synthetic OMOP generator, pytest end-to-end, local-Spark check
```

## Running on MCP
**Step-by-step validation guide: [VALIDATION_RUNBOOK.md](VALIDATION_RUNBOOK.md).** Agent runbooks for the models trained in MCP: [RUNBOOK_ECG.md](RUNBOOK_ECG.md), [RUNBOOK_MRI.md](RUNBOOK_MRI.md) and [RUNBOOK_ECHO_WM.md](RUNBOOK_ECHO_WM.md). Summary:

1. **Onboard.** Run `python scripts/discover_schema.py`, then set `schemas` and `site_map` in `config/study.yaml` and the ECG/echo table names in `config/modality_tables.yaml` (`enabled: true`). Until these are set, the builder falls back to OMOP proxies: procedure codes, LOINC ECG/echo measurements, and NOTE text.
2. **Concept sign-off.** `exports/concept_sets_detail_for_review.csv` lists every concept. Rows with `needs_review = 1` came from concept-name searches (the echo dimensions and pack-years) and need a clinician's sign-off.
3. **Build.** Run `python scripts/run_cohorts.py --save-tables` or the notebook. Only aggregate exports are written: attrition, completeness, sites and Table 1, with cells under 11 suppressed.
4. **Run the agents inside the workspace.** Each agent writes `{cohort}_predictions` following the YAML `predictions` contract.
5. **Evaluate.** `evaluate()` writes `exports/{cohort}_endpoints.csv`, with each endpoint's estimate, 95% CI, goal and pass/fail.

## Testing
```
python -m pytest -q tests/                    # synthetic OMOP -> 5 cohorts -> simulated agents -> endpoints (DuckDB)
PYTHONPATH=<pyspark> python tests/spark_local_check.py   # same pipeline as native Spark SQL 3.5
```
Both engines give identical attrition counts and concept-set resolution on the same synthetic data.

## Known limitations / decisions before database lock
* **Study plan (power).** `plan:` in `config/study.yaml` selects the sizes and the success rule; `CARDIOAI_PLAN=A|B|C` overrides it for one run.
  - **A:** larger cohorts (CAD 2,200, Risk 2,000, Longitudinal 4,000) with the deck rule, lower CI ≥ deck goal.
  - **B:** deck sizes with the margin rule, lower CI ≥ min(deck goal, Phase 1 − margin).
  - **C (adopted):** CAD 1,300, Risk 1,500, Longitudinal 2,000 with the margin rule. This powers 13 of 13 primary endpoints at Phase 1 performance with 8,300 patients.

  Run `python scripts/sample_size_report.py` to compare the plans (writes `exports/sample_size_plans.csv`). The endpoint tables report the plan, the threshold, the rule, and whether the deck goal was met as a point estimate.
* **Reference standards from notes.** Cath-report stenosis %, the echo report findings and the ECG over-read text are extracted with regex rules. The protocol requires double manual adjudication of a 10% random sample (κ is reported).
* **Death causes.** Cause of death is not populated on MCP (Adam Resnick, 2026-10-21), so every death outcome uses all-cause death; the Risk cohort's `cv_mortality` element is defined as all-cause death (a proxy that over-counts CV death). This is a protocol change for the clinical lead to approve.
