# Validation Runbook: CardioAI Phase 2 on Mayo Clinic Platform

**Study CA-AGENTS-VAL-PHASE2 · Plan C · 5 agents · 8,300 patients**

This runbook takes you from an accepted MCP account to a signed-off validation report for each agent. Follow the steps in order: steps 1–6 are done **once** for all agents, and step 7 is repeated **for each agent**.

---

## One at a time, or all at once?

**Each agent is validated individually.** Each one has its own cohort, its own reference standard, its own pre-specified endpoints and its own pass/fail decision. One agent failing does not affect another. Each agent gets its own report.

**The setup is shared and done once.** Workspace setup, schema mapping, concept sign-off, the feasibility check and locking everything happen a single time, before any agent is run.

**The agents then run in three waves,** ordered by the data each one needs:

| Wave | Agents | Why this order | Needs |
|---|---|---|---|
| 1 | **CAD**, **Risk** | Tabular inputs that are pure OMOP; no waveform or DICOM mapping; runs on CPU | Cath / CCTA report adjudication (CAD) |
| 2 | **Longitudinal** | OMOP time series, but a daily-landmark inference loop (≈ 2,000 × 180 runs) | Serial-data checks |
| 3 | **ECG**, **Echo** | Need the MCP ECG-waveform and echo-DICOM tables mapped, plus a GPU | `modality_tables.yaml` filled in; GPU workspace |

Agents within a wave can run in parallel. Wave 1 also proves the pipeline end-to-end on real MCP data before the heavier imaging work begins.

> **Blinding rule.** No one looks at any agent's predictions until that agent's reference standard is locked (step 7c). Adjudicators never see agent outputs.

---

## Step 1: Set up the workspace (once)

1. In the MCP portal, open a **workspace**. Use a medium CPU configuration for waves 1–2; you'll need one with a GPU for wave 3. Choose **JupyterLab**.
2. Bring the toolkit in. Use the workspace's Git integration to clone this repo (or upload a zip of it). The app repo is **not** needed: the agents' input encoders are included here (`cardioai_mcp/encoders`, a byte-identical copy of the app's `app/validated_agents/preprocessing`).
   ```bash
   git clone https://github.com/cardioailive1/cardioai_mcp_validation.git
   cd cardioai_mcp_validation
   pip install -r requirements.txt
   python -c "from cardioai_mcp import encoders; print(encoders.fingerprint())"
   ```
   The printed fingerprint must equal `model_lock.encoders_fingerprint` in `config/study.yaml` (the tests check this too). Record it and this repo's commit SHA (`git rev-parse HEAD`) in the analysis log; they are part of the validation lock.
3. Confirm Spark works from a notebook cell:
   ```python
   from pyspark.sql import SparkSession
   spark = SparkSession.builder.getOrCreate(); spark.sql("SHOW DATABASES").show()
   ```
4. Confirm with Mayo that the study is not human-subjects research, and file the determination letter in `docs/governance/`.

## Step 2: Map the MCP schema (once)

```bash
python scripts/discover_schema.py --out exports/schema_inventory
```
This writes `tables.csv` (every table, with ECG / echo / waveform candidates flagged), `care_sites.csv`, `note_titles.csv` and `vocabularies.csv`. Check them against the **Schema Visualizer**, then edit:

| File | Set |
|---|---|
| `config/study.yaml` | `schemas.cdm`, `schemas.vocab`, `schemas.work` (your writable schema), `schemas.modality`; `data_cutoff` = the MCP data-refresh date; `site_map` (care_site_id → site name) so each cohort has ≥ 12 sites (CAD ≥ 8) |
| `config/modality_tables.yaml` | ECG waveform table + columns; echo study / DICOM table + columns; `enabled: true` (needed for wave 3 only) |
| `config/modality_tables.yaml` → `notes` | Note-title patterns, adjusted to what `note_titles.csv` shows (e.g. how cath reports are titled) |

## Step 3: Sign off the concept sets (once)

```bash
python scripts/run_cohorts.py --engine spark --cohorts risk     # any cohort; this resolves all concept sets
```
Send `exports/concept_sets_detail_for_review.csv` to the study cardiologist, who:
- checks each set's codes (MI, HF, STEMI, CPT angiography, LOINC labs …)
- **must** approve every row flagged `needs_review = 1` (echo dimensions, pack-years)

Edits go in `config/concept_sets.yaml`. Rerun until it's signed. Record the sign-off date in the git log.

## Step 4: Check feasibility (once; no agent outputs)

```bash
python scripts/run_cohorts.py --engine spark --save-tables
```
For each cohort, review the files in `exports/`:
- `*_attrition.csv`: the eligible pool at each step; any **SHORTFALL** in the sampling line (e.g. VT ECGs)
- `*_sites.csv`: the number of sites
- `*_completeness.csv`: elements below the deck's expected completeness (flag `BELOW TARGET`)

Then put the **real rates** into `RATES` in `scripts/sample_size_report.py`: obstructive-CAD share among angiography patients, the share of Risk-cohort patients with ≥ 50% stenosis at angiography, HF exacerbations per patient. Rerun:
```bash
python scripts/sample_size_report.py
```
If Plan C still powers every primary endpoint, keep it. If not, enlarge that cohort in `config/study.yaml → plans.C.sizes`, rerun, and record the change.

## Step 5: Lock everything (once, before any inference)

1. **Agents:** for each agent, record the model file's SHA-256 and its decision thresholds exactly as in Phase 1:
   ```bash
   sha256sum models/*.pt models/*.pkl > docs/governance/agent_lock.sha256
   ```
   Put the thresholds (e.g. the CAD probability cut-off, the alert thresholds) in `docs/governance/agent_lock.yaml`.
2. **Plan and code:** `plan: C` in `config/study.yaml`, then
   ```bash
   git add -A && git commit -m "Lock Phase 2 analysis (Plan C)" && git tag phase2-analysis-lock
   ```
3. Optional but recommended: pre-register the protocol (e.g. OSF), with the tag hash.

From here on, a change to the config, the concepts or a threshold is a **protocol deviation** and must be documented.

## Step 6: Upload the agents into the workspace (once)

Patient data cannot leave MCP, so the agents must come to the data. Upload each agent's weights and inference code into the workspace (Git, or workspace file upload), check the SHA-256 against `agent_lock.sha256`, and confirm each agent loads.

---

## Step 6b: Train the ECG model (ECG agent only, before its validation)

The ECG agent is a new MobileNetV3-1D model (acute MI, STEMI, AF), trained here on Mayo ECGs. MobileNetV3 export is approved under the agreement. Follow [`ecg_training/README.md`](ecg_training/README.md). Two rules:
- **Lock the ECG validation cohort before training.** Its patients are excluded from training, and it only uses ECGs from 2021 on, while training uses earlier years.
- **Export only `weights.pt`, `thresholds.json` and `model_card.json`.**

## Step 7: Validate each agent (repeat per agent, by wave)

Each agent goes through the same six sub-steps, **a → f**. Swap `<cohort>` for `cad`, `risk`, `longitudinal`, `ecg` or `echo`.

### 7a. Build the cohort
```bash
python scripts/run_cohorts.py --engine spark --cohorts <cohort> --save-tables
```
This creates `<work>.<cohort>_analytic` (one row per patient, every deck data element) and, for Longitudinal, `<work>.longitudinal_series`.

### 7b. Check data quality
Review `exports/<cohort>_attrition.csv`, `_completeness.csv`, `_sites.csv` and `_table1.csv`. Fix mapping problems now; once the reference standard is locked, the data are frozen.

### 7c. Adjudicate and lock the reference standard
Several references come from report text: CAD stenosis %, echo wall motion / RV / valves, ECG over-read classes. For these:
1. Draw a 10% random sample (seed fixed in `study.yaml`). Two cardiologists independently review the source report, **blinded to the extracted value and to the agent**. A third reviewer resolves disagreements.
2. Record agreement (κ) between the regex rules and the adjudicated value. If it's poor (κ < 0.8), fix the rule and redo 7a–7c.
3. Snapshot the table: `CREATE TABLE <work>.<cohort>_analytic_locked AS SELECT * FROM <work>.<cohort>_analytic`.

| Cohort | Reference that needs adjudication (Plan C, 10%) |
|---|---|
| CAD | ~130 cath / CCTA reports: per-vessel stenosis %, CAD present |
| Risk | ~135 cath reports: maximum stenosis % → heart disease (≥ 50%) |
| Longitudinal | HF exacerbation dates; review a sample of events |
| ECG | ~200 ECG reads: superclass, rhythm, STEMI |
| Echo | ~150 echo reports: wall motion, RV, valves (LVEF is a structured value) |

### 7d. Run the agent
The agent reads the locked analytic table and writes a predictions table that follows the cohort's contract (the `predictions:` block in `config/cohorts/<cohort>.yaml`):

| Agent | Input | Output table and columns |
|---|---|---|
| CAD | The 54 Z-Alizadeh Sani inputs (`model_inputs` in `cad.yaml`, `docs/model_input_mapping.csv`), measured before the procedure, encoded with `app/validated_agents/preprocessing` | `cad_predictions`: `person_id, agent_cad_prob, agent_cad, agent_multivessel, agent_risk_category` |
| Risk | The 13 Cleveland inputs (`model_inputs` in `risk.yaml`); report **1 − model output** (training target 1 = no disease) | `risk_predictions`: `person_id, agent_hd_prob, agent_hd` |
| Longitudinal | `longitudinal_series`, only data with `day ≤ t`, run daily for t = 0…179 | `longitudinal_predictions`: `person_id, day, risk_hf_7d, alert_hf_7d, risk_acute_24h, alert_acute_24h`; plus `longitudinal_predictions_patient`: `person_id, risk_mi_30d, risk_event_30d, pred_readmit_30d, pred_risk_90d` |
| ECG | The 12-lead waveform from the MCP ECG table (10 s, resampled to 500 Hz) | `ecg_predictions`: `person_id, agent_superclass, agent_stemi, agent_rhythm, agent_bbb, runtime_s` |
| Echo | The A4C cine from the echo DICOM (112 px, ≥ 32 frames) | `echo_predictions`: `person_id, agent_lvef, agent_ef_category, agent_wall_motion, agent_lv_size, runtime_s` |

Encode the model inputs with the **same code the app serves with**, so the validated encoding is the deployed one. `encode_cohort` renames the analytic columns into the app's request fields, applies the pre-specified missing-data rules and calls the shared encoders:
```python
from cardioai_mcp.agent_inputs import encode_cohort
analytic = spark.table("cardioai_work.risk_analytic_locked").toPandas()
enc = encode_cohort("risk", analytic)        # person_id, scorable, reason, vector, imputed   ("cad" for CAD)
X = list(enc.vector[enc.scorable])           # send each vector to the locked model; Risk: report 1 - output
```
Patients with `scorable = False` (the `reason` column says why, e.g. more than 5 optional CAD labs missing) count as model failures in the accuracy denominators, per the protocol.
The Longitudinal EHR vitals model can't be scored until its raw-unit normalization and label definition are known (`model_inputs.blockers` in `longitudinal.yaml`).

A template for writing the table:
```python
import pandas as pd
cohort = spark.table("cardioai_work.cad_analytic_locked").toPandas()
preds = my_cad_agent.predict(cohort)          # your locked agent -> DataFrame with the contract columns
spark.createDataFrame(preds).write.mode("overwrite").saveAsTable("cardioai_work.cad_predictions")
```
The **Longitudinal** agent must never see data after day t when predicting for day t. Build each day's input by filtering `longitudinal_series` to `day <= t`.

### 7e. Score the agent
```bash
python scripts/evaluate_cohorts.py --cohorts <cohort> --boot 5000
```
For each endpoint it prints the estimate, the 95% CI, the Plan C threshold and pass/fail. The agent **passes** only if **all** its primary endpoints pass. It writes `exports/<cohort>_endpoints.csv`, which also states whether the deck goal was met as a point estimate.

Then run the pre-specified sensitivity analyses and subgroups, all with the same command and settings:
```bash
python scripts/evaluate_cohorts.py --cohorts <cohort> --cluster-by-site
```
- the adjudicated-subset reference
- CCTA-referenced CAD
- all-cause death in place of CV death
- weighted vs unweighted ECG accuracy
- subgroups by sex, age, race/ethnicity and site

### 7f. Export and write the report
1. Only aggregate files leave the workspace: `exports/*.csv`, where cells under 11 are already suppressed. Have them reviewed under MCP's export process.
2. Write the agent's report using **TRIPOD+AI**: attrition flow, Table 1, completeness against the deck, the endpoint table, subgroups, adjudication κ, and any deviations from the lock.

---

## Step 8: Close-out (once)

- Combine the five agent reports into the Phase 2 summary (Study ID CA-AGENTS-VAL-PHASE2): one pass/fail line per agent.
- Archive the git tag, `agent_lock.*`, the concept-set sign-off, the adjudication logs and the exports.
- Optional: submit the agents to **MCP_Validate** for an independent bias report.

## Checklist

- [ ] 1 Workspace + toolkit + Spark verified; IRB determination filed
- [ ] 2 `study.yaml` / `modality_tables.yaml` mapped; ≥ 12 sites
- [ ] 3 Concept sets signed off
- [ ] 4 Feasibility run; Plan C confirmed with real rates
- [ ] 5 Agents, thresholds, plan and code locked (git tag)
- [ ] 6 Agents uploaded; hashes verified
- [ ] 7 Wave 1: CAD ☐ Risk ☐ · Wave 2: Longitudinal ☐ · Wave 3: ECG ☐ Echo ☐ (each a → f)
- [ ] 8 Summary report and archive
