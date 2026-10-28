# ECG agent runbook (MobileNetV3-1D) on Mayo Clinic Platform

How to train, validate and deploy the new ECG agent. It flags **acute MI, STEMI and AF/flutter** on a resting 12-lead ECG. The model is trained only on Mayo data inside the MCP workspace. Labels come from diagnosis codes, troponin and confirmed reads, so no images or tracings need annotating.

| | |
|---|---|
| Model | MobileNetV3-1D (`ecg_training/`); weights approved for export |
| Periods | ECGs before 2020-01-01 train it, 2020 tunes it, and 2021 onward (plus held-out sites) tests it. The Phase 2 cohort comes from 2021 onward, and its patients are excluded from training |
| Mayo data | The ECG waveform table, which holds Base64-encoded strings; diagnoses, troponin and confirmed reads come from OMOP |
| Phase 2 goals | Primary: STEMI sensitivity ≥ 92%, AF/flutter sensitivity ≥ 94%. Secondary: acute MI sensitivity ≥ 92%, STEMI and AF specificity ≥ 93%, AUROC ≥ 0.93 for each, processing ≤ 3 s |
| Exported | `weights.pt`, `thresholds.json`, `model_card.json` only |

## 1. Before you start

| Item | What to do |
|---|---|
| Accounts | Every team member submits the MCP DUA form and completes onboarding |
| Trainings | Data Environment training; the ECG and Lab Test trainings in the MCP Resource Hub |
| GPU | Switch on an A100 40 GB machine from the shared reservation (it starts if one is free) |
| Code | Clone `cardioailive1/cardioai_mcp_validation` (main) into the workspace |
| Packages | `pip install -r requirements.txt -r requirements-ecg.txt`, with the CUDA build of torch for the workspace driver |
| Configuration | Schemas in `config/study.yaml`; the ECG table in `config/modality_tables.yaml` → `ecg` |

### Querying the data (BigQuery)

MCP provisions Mayo's data in BigQuery, and the workspace queries it with the BigQuery client (MCP, 2026-10-28). The SQL in this repository is written once, in Spark SQL, and `BigQueryEngine` translates every statement to BigQuery SQL (sqlglot) before running it.
- **Templates:** the SQL is written with placeholders such as `{{cdm}}` (OMOP dataset), `{{modality}}` and `{{cs:stemi}}` (a concept set).
- **Filling them in:** the engine replaces the placeholders with the dataset names in `config/study.yaml` and `config/modality_tables.yaml`, runs the query in one BigQuery session, and returns a pandas table. Intermediate steps are session temp tables.
- **When names change:** edit the config, not the SQL.

First time in the workspace:
```bash
python scripts/discover_schema.py --out exports/schema_inventory   # datasets, tables, columns and their types
```
```python
# put the dataset names into config/study.yaml (schemas: cdm, vocab, modality, work as "`<project>`.<dataset>"), then:
from cardioai_mcp.config import load_study; from cardioai_mcp.engine import make_engine
study = load_study(); eng = make_engine("bigquery", study)
eng.query("SELECT COUNT(*) AS n FROM {{cdm}}.person")         # placeholders filled from the config
eng.query("SELECT COUNT(*) AS n FROM {{modality}}.FACT_ECG_WAVEFORM")   # one test query
```
Saving and reading tables: `eng.write_table(df, "name")` writes `{work}.name`; `eng.read_table("name")` reads it back.
Model training does not use BigQuery: the queries build the training list and pull the data, then PyTorch trains on the GPU from files.

## 2. Steps

1. **Lock the Phase 2 ECG cohort first.**
   ```bash
   python scripts/run_cohorts.py --cohorts ecg --save-tables
   ```
   Snapshot the analytic table as `cardioai_work.ecg_analytic_locked`. Its window starts on 2021-01-01 (`splits.test_start`), and training excludes its patients.

2. **Map and probe the MCP ECG tables.** MCP (2026-10-21): `FACT_ECG_WAVEFORM` holds one row per lead; use `WAVEFORM_TYPE = 'Rhythm'` (full 10 s), leads I, II, V1–V6; each value is URL-encoded Base64 → signed 16-bit little-endian counts × 4.88 = µV; the sample rate is `ECGSAMPLEBASE` in `FACT_ECG_WAVEFORM_RESTING_MEASUREMENT`; waveforms link to `FACT_ECG` through `DIM_PATIENT` (clinic number) and `ACQUISITION_TS = ECG_DTM`. This is `waveform_storage: base64_per_lead` in `config/modality_tables.yaml` → `ecg`, already filled in.
   - `person_map` is filled in: OMOP `person.person_source_value` holds `PATIENT_CLINIC_NUMBER` (MCP 2026-10-28). Still to fill once MCP answers: the over-read columns in `fact_ecg.columns` (F3). Confirm the resting-measurement join keys. If `FACT_ECG` or `FACT_ECG_WAVEFORM` carries `PATIENT_CLINIC_NUMBER` (MCP is retiring `PATIENT_DK` in some tables), add it as `clinic_number` in that table's column map: it is then used directly.
   - Build the ECG index once and save it as `ecg.table`, then set `enabled: true`:
     ```python
     from ecg_training.mcp_waveforms import build_ecg_index
     eng = make_engine("bigquery", study)
     idx = build_ecg_index(eng, study["modality"])
     eng.write_table(idx, "ecg_index")
     ```
   - Run `python -m ecg_training.probe_waveforms --n 20`. Per acquisition it reports the Rhythm rows per lead, samples, the implied rate, seconds and the amplitude in mV. Expect 8 leads, one row each, about 10 s and R waves of about 0.5–2 mV. Acquisitions with more than one Rhythm row for a lead are skipped with a reason until MCP says how such rows relate (F2).

3. **Check troponin.** Mayo's test is **Troponin T, 5th generation** (TRPS, "Troponin T, 5th gen, P", LOINC 67151-1), reported in ng/L.
   - A row is raised if it is above its own `range_high`. Rows without one use Mayo's sex-specific limits: **males > 15 ng/L, females > 10 ng/L**.
   - For older rows in ng/mL, look up the assay by reporting name on [mayocliniclabs.com](https://www.mayocliniclabs.com/search#q=troponin), under "Clinical and Interpretive" and "Fees & Codes". Then confirm `labels.troponin_url_fallback.ng_per_ml` in `ecg_training/config.yaml`.

4. **Build the manifest and decode the waveforms once.**
   ```python
   from cardioai_mcp.config import load_study; from cardioai_mcp.engine import make_engine
   from ecg_training.dataset import load_config, build_manifest, summary
   from ecg_training.extract_waveforms import extract_waveforms
   study, cfg = load_study(), load_config(); eng = make_engine("bigquery", study)
   excl = eng.read_table("ecg_analytic_locked").person_id
   m = build_manifest(eng, study, cfg, study["modality"], exclude_persons=excl)
   m = extract_waveforms(eng, m, study["modality"], "/workspace/ecg/waveforms")   # Base64 -> one .npz per ECG
   summary(m, cfg["heads"]); m.to_parquet("/workspace/ecg/manifest.parquet")
   ```
   - The build refuses to continue if any patient falls in two splits.
   - Extraction can be re-run; it resumes where it stopped.
   - ECGs that fail to decode keep the reason in `waveform_error`.

5. **Audit the automatic labels** before training. The protocol is in [`annotation/LABEL_AUDIT.md`](annotation/LABEL_AUDIT.md).
   ```bash
   # troponin / acute MI / STEMI: recomputed independently from the raw lab rows (must match 100%)
   python -m annotation.label_audit troponin --manifest /workspace/ecg/manifest.parquet --out /workspace/audit/troponin
   # AF: rhythm cross-check from the waveform, then the blinded text audit of the over-reads
   python -m annotation.label_audit af-rhythm --manifest /workspace/ecg/manifest.parquet --out /workspace/audit/ecg/rr_flags.csv
   python -m annotation.label_audit sample ecg --manifest /workspace/ecg/manifest.parquet --auditors ds1,ds2 \
          --extra /workspace/audit/ecg/rr_flags.csv --auto triage --out /workspace/audit/ecg      # tier 0 required: triage (default) | assist
   python -m annotation.label_audit score ecg --dir /workspace/audit/ecg     # twice: before and after the cardiologist
   ```
   - The cardiologist reads `troponin_concepts.csv` (which assays the troponin concept set really contains, with units and reference limits) and signs it off.
   - AF must PASS (PPV and NPV at least 0.95, lower bound at least 0.90). On a FAIL, fix `af_regex` or the `context` trigger words (`cardioai_mcp/report_context.py`), rebuild and re-audit on a fresh sample.

6. **Train on the A100.**
   ```bash
   python -m ecg_training.train --manifest /workspace/ecg/manifest.parquet --out /workspace/ecg/runs/v1
   ```
   - Thresholds are fitted on 2020 only, with headroom above the goals: target sensitivity 0.96 for STEMI, 0.97 for AF and 0.96 for acute MI.
   - ECGs with no decoded waveform are dropped and counted in the model card.

7. **Review `model_card.json`** before opening the test set:
   - data summary and label definitions;
   - thresholds; check **STEMI and AF specificity on the tuning split against the 93% goal**;
   - test metrics (`test_time`, `test_site`), `ecgs_without_waveform`, runtime fingerprint, weights hash and code commit.

   Copy the `test_time` values into the `phase1` fields of `config/cohorts/ecg.yaml`, then rerun `scripts/sample_size_report.py`.

8. **Export** only `weights.pt`, `thresholds.json` and `model_card.json`. Manifests, waveforms and per-ECG predictions stay in MCP.

## 3. Phase 2 validation

1. Score the locked cohort inside MCP with the exported bundle and the same `ecg_training/runtime` code. Write `cardioai_work.ecg_predictions` with these columns:
   - `person_id`;
   - `agent_acute_mi_prob`, `agent_acute_mi`, `agent_stemi_prob`, `agent_stemi`, `agent_af_prob`, `agent_af`;
   - `runtime_s` and `runtime_fingerprint`.

   The fingerprint must equal the model card's. The contract is the `predictions:` block in `config/cohorts/ecg.yaml`.
2. Evaluate:
   ```bash
   python scripts/evaluate_cohorts.py --cohorts ecg --boot 5000
   ```
   The result goes to `exports/ecg_endpoints.csv`. Success needs the lower 95% CI bound to clear each primary goal.

## 4. Deployment

1. Deploy the exported bundle to Vertex, outside MCP:
   ```bash
   gsutil cp weights.pt thresholds.json model_card.json gs://<bucket>/ecg/v1/
   BUNDLE_URI=gs://<bucket>/ecg/v1 IMAGE_TAG=v1 ecg_training/serve/deploy.sh
   ```
   The container refuses to start if its runtime fingerprint or weights hash differs from the model card.
2. Lock it in the app (`app/validated_agents/registry.yaml` → `ecg` in cardioaimedconnect).
   - Fill in `model_id`, `endpoint_id`, `weights_sha256` and the three thresholds from `thresholds.json`, then set `status: research_only`.
   - The runtime fingerprint is already pinned, and the app answers 409 to any container that differs.
3. After Phase 2 passes, open a PR that changes only `status: validated` and the validation report link. The clinical lead and the QA/regulatory owner sign it off.

## 5. Checks and stop conditions

| Step | Must hold | If not |
|---|---|---|
| Cohort lock | Phase 2 cohort snapshotted before any training run | Stop and re-lock |
| Manifest | No patient in two splits (the build refuses otherwise) | Fix the patient IDs |
| Waveform probe | About 10 s and R waves of 0.5–2 mV on 5 rows | Adjust `waveform_raw`, probe again |
| Extraction | Few ECGs with `waveform_error` | Inspect the errors before training |
| Label audit | Troponin recomputation 100% match; AF PASS in `audit_results.csv`; troponin concept list signed off | Fix the SQL or rule, rebuild, re-audit |
| Thresholds | Fitted on 2020 only; specificity reviewed before the test set is opened | Never refit on test data |
| Export | Only the three bundle files leave MCP | Stop |
| Serving | Fingerprint and weights hash match the model card | Rebuild from the exact commit |

## 6. Open items

- [ ] Probe 5 real waveform rows and confirm `waveform_raw`.
- [ ] Confirm the assay and cut-off of any older ng/mL troponin rows.
- [ ] Write the scoring script that produces `ecg_predictions` from the locked bundle.

See also: `ecg_training/README.md` (model details and label definitions), `VALIDATION_RUNBOOK.md` (the full Phase 2 study), and `RUNBOOK_MRI.md`.
