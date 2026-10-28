# MRI agent runbook (MobileNetV3-UNet) on Mayo Clinic Platform

How to train, validate and deploy the cardiac MRI agent, on short-axis LGE plus cine for LVEF.
- **Measures:** confirms the infarct after a STEMI and measures infarct size, transmurality per AHA segment, MVO and LVEF.
- **Flags:** STEMI mimics (non-ischaemic LGE), LV thrombus, LV aneurysm, and low LVEF as red flags.

The model is trained **only on Mayo data** inside the MCP workspace. No public datasets or outside pretrained weights are used.

| | |
|---|---|
| Model | MobileNetV3 encoder + U-Net decoder (`mri_training/`); export approved |
| Periods | Studies before 2020-01-01 train it, 2020 tunes it, and 2021 onward (plus held-out sites) tests it. The Phase 2 cohort comes from 2021 onward, and its patients are excluded from training |
| Mayo data | **Reports:** `FACT_RADIOLOGY` (`RADIOLOGY_REPORT`, `RADIOLOGY_IMPRESSION`). **Images:** DICOM requested for de-identification from `FACT_RADIOLOGY_DICOM_INVENTORY`, linked by accession number. **Studies:** `FACT_IMAGINGSTUDY` |
| Labels | Report labels for every study, plus about 300 studies outlined by CardioAI readers |
| Phase 2 goals (proposed; clinical lead to confirm) | Infarct sensitivity ≥ 90%, infarct specificity ≥ 85%, LVEF mean absolute error against the report ≤ 5 EF points |
| Exported | `weights.pt`, `thresholds.json`, `model_card.json` only |

## 1. Before you start

| Item | What to do |
|---|---|
| Accounts | Every team member and every reader submits the MCP DUA form and completes onboarding. Readers get up to 5 annotator seats with cardioailive.com emails; Mayo readers can't annotate in our environment |
| Trainings | Data Environment and Radiology trainings ("Radiology" slides in the MCP Resource Hub) |
| GPU | Switch on an A100 40 GB machine from the shared reservation |
| Code | Clone `cardioailive1/cardioai_mcp_validation` (main) into the workspace |
| Packages | `pip install -r requirements.txt -r requirements-mri.txt` (CUDA torch). Add `monailabel` only if you want model-assisted annotation |
| Reader tools | 3D Slicer and ITK-SNAP are already installed |
| Workspace check | `python -m annotation.check_workspace` |

### Querying the data (Spark SQL)

Mayo's data on MCP sit in Apache Spark. Every query in this repository is Spark SQL, run through the `spark` session that JupyterLab provides.
- **Templates:** the SQL is written with placeholders such as `{{cdm}}` (OMOP schema), `{{modality}}` and `{{cs:stemi}}` (a concept set).
- **Filling them in:** `SparkEngine` replaces the placeholders with the real names from `config/study.yaml` and `config/modality_tables.yaml`, runs the query on Spark, and returns a pandas table.
- **When names change:** edit the config, not the SQL.

First time in the workspace:
```python
spark.sql("SHOW DATABASES").show(100, False)                 # find the OMOP and FACT_ schemas
spark.sql("SHOW TABLES IN <schema>").show(200, False)
spark.sql("SELECT COUNT(*) FROM <schema>.FACT_IMAGINGSTUDY WHERE MODALITY_1 = 'MR'").show()   # one test query
# put the schema names into config/study.yaml (schemas: cdm, vocab, modality, work), then:
from cardioai_mcp.config import load_study; from cardioai_mcp.engine import SparkEngine
study = load_study(); eng = SparkEngine(spark, study)
eng.query("SELECT COUNT(*) AS n FROM {{cdm}}.person")         # placeholders filled from the config
```
Model training does not use Spark: Spark builds the training list and pulls the data, then PyTorch trains on the GPU from files.

## 2. Steps

1. **Lock the Phase 2 MRI cohort first.** The clinical lead confirms the proposed goals and the red-flag cut-offs (`mri_training/config.yaml` → `red_flags`) beforehand.
   ```bash
   python scripts/run_cohorts.py --engine spark --cohorts mri --save-tables
   ```
   Snapshot the analytic table as `cardioai_work.mri_analytic_locked`.

2. **Find the CMR studies and build the DICOM request.** First fill in `config/modality_tables.yaml` → `mri.mcp`:
   - the schema of the FACT tables;
   - the final-report `SERVICE_RESULT_STATUS` values;
   - `person_map` is filled in: OMOP `person.person_source_value` holds `PATIENT_CLINIC_NUMBER` (MCP 2026-10-28).

   Then run:
   ```python
   from cardioai_mcp.config import load_study; from cardioai_mcp.engine import SparkEngine
   from mri_training import mcp_imaging as mi
   study = load_study(); eng = SparkEngine(spark, study)
   studies = mi.find_cmr_studies(eng, study["modality"])     # MR cardiac studies with a final report
   series = mi.list_series(eng, study["modality"], studies)
   req, report = mi.build_request(series)                     # short-axis LGE + cine series only
   deid = mi.deid_request(studies, req)                       # one row per selected study
   deid.to_csv("/workspace/mri/deid_request.csv", index=False); report
   ```
   Only de-identifiable series (status Ready or Available) are included. Studies without an LGE series are skipped and counted in `report`.

3. **Send `deid_request.csv` to the MCP Technical Success Manager (TSM).** It has one row per selected study, with the columns MCP asked for: `STUDY_INSTANCE_UID`, `PATIENT_DK`, `MODALITY`, `STMO_NAME_IN_ADL`.
   - MCP de-identifies **whole studies**. After delivery, the pipeline itself picks the short-axis LGE and cine series.
   - `PATIENT_DK` is read from `FACT_RADIOLOGY` by default (`mri.mcp.patient_dk_source`). If the workspace has it in another table, change that setting.
   - Check the file before sending: no empty cells, and the row count matches `report["studies_with_lge"]`.

4. **After delivery, build the study table and the manifest.**
   ```python
   cmr = mi.build_study_table(studies, "/workspace/mri/dicom")      # one row per delivered study folder
   spark.createDataFrame(cmr).write.saveAsTable("cardioai_work.cmr_study")
   from mri_training.dataset import load_config, build_manifest
   cfg = load_config()
   excl = spark.table("cardioai_work.mri_analytic_locked").select("person_id").toPandas().person_id
   m = build_manifest(eng, study, cfg, study["modality"], exclude_persons=excl)
   m.to_parquet("/workspace/mri/manifest.parquet")
   ```
   - Check one delivered study with `mri_training.io.read_study_path`.
   - The build refuses to continue if any patient falls in two splits.

5. **Audit the automatic report labels** before any training. Data scientists check them first; the cardiologist adjudicates. The protocol is in [`annotation/LABEL_AUDIT.md`](annotation/LABEL_AUDIT.md).
   ```bash
   python -m annotation.label_audit sample mri --manifest /workspace/mri/manifest.parquet --auditors ds1,ds2 --auto triage --out /workspace/audit/mri
   #   tier 0 (required): --auto triage (default; automatic auditor settles agreeing reports, 25% still verified) | assist (people read everything, relevant sentences shown)
   #   data scientists fill sheet_ds1.xlsx / sheet_ds2.xlsx (blinded: they never see the automatic labels)
   python -m annotation.label_audit score mri --dir /workspace/audit/mri    # -> adjudication_1.xlsx for the cardiologist
   #   cardiologist fills adjudication_1.xlsx (every disagreement and U, plus a blinded spot-check)
   python -m annotation.label_audit score mri --dir /workspace/audit/mri    # -> audit_results.csv: PASS / FAIL per label
   ```
   - With `--auto triage`, the automatic auditor must also match the reference on at least 98% of the verification reports, or that label is re-audited with `--auto assist`.
   - Every label must PASS: PPV and NPV at least 0.95, with the lower 95% bound at least 0.90. Stated LVEF and infarct size must match in at least 98% of reports.
   - On a FAIL, fix the rule in `mri_training/config.yaml` → `labels`, rebuild the manifest and re-audit on a **fresh** sample (new `seed` in `annotation/tasks.yaml`).
   - Record the audit results and the rule version in the model card.

6. **Pretrain the encoder on Mayo images** (self-supervised; training period only).
   ```bash
   python -m mri_training.pretrain --manifest /workspace/mri/manifest.parquet --out /workspace/mri/runs/ssl_v1 --device cuda
   ```
   - It writes `encoder.pt` and `pretrain_card.json` (period, studies, slices, loss).
   - It refuses studies dated on or after 2020-01-01.

7. **Outline about 300 studies in rounds.** Readers use 3D Slicer; the protocol and checks are in `annotation/README.md`.
   ```bash
   python -m annotation.select mri --manifest /workspace/mri/manifest.parquet --readers r01,r02,r03 --out worklist.csv
   # round 1: the first ~40 studies, outlined from scratch
   python -m annotation.export mri --worklist worklist_round1.csv --out /workspace/annotation
   python -m annotation.collect mri --root /workspace/annotation --out /workspace/mri/contours
   #   join contours/label_uri.csv into the manifest (label_uri column), then train the pre-drawing model:
   python -m mri_training.train --manifest /workspace/mri/manifest_labels.parquet --init /workspace/mri/runs/ssl_v1 \
          --stage-a-only --out /workspace/mri/runs/prelabel_r1 --device cuda
   # round 2 onward: readers correct pre-drawn outlines; retrain about every 100 studies
   python -m annotation.export mri --worklist worklist_round2.csv --out /workspace/annotation --bundle /workspace/mri/runs/prelabel_r1
   ```
   - `collect` refuses unreviewed or invalid files.
   - About 10% of studies are read twice; agreement goes to `agreement.csv`.
   - **Outline QC.** Data scientists check every batch of pre-drawn (before readers) and corrected (after `collect`) outlines. The cardiologist reviews grade C, grader disagreements and a spot-check:
     ```bash
     python -m annotation.label_audit outlines --root /workspace/annotation --manifest /workspace/mri/manifest.parquet --graders ds1,ds2 --out /workspace/audit/outlines_r2
     python -m annotation.label_audit score-outlines --dir /workspace/audit/outlines_r2
     ```

8. **Train the full model.**
   ```bash
   python -m mri_training.train --manifest /workspace/mri/manifest_labels.parquet --init /workspace/mri/runs/ssl_v1 \
          --out /workspace/mri/runs/v1 --cache-dir /workspace/mri/cache --device cuda
   ```
   - Stage A learns segmentation from the reader outlines.
   - Stage B adds report labels for every training study.
   - Thresholds are fitted on 2020 only.

9. **Review `model_card.json`** before opening the test set:
   - `initialised_from`: the pretraining period and the hash of the starting weights;
   - data summary and thresholds;
   - finding AUROC, sensitivity and specificity;
   - Dice per structure;
   - infarct size, MVO and LVEF against the reader outlines and the reports;
   - double-read agreement, runtime fingerprint and weights hash.

   Copy the `test_time` values into the `phase1` fields of `config/cohorts/mri.yaml`.

10. **Export** only `weights.pt`, `thresholds.json` and `model_card.json`. Images, outlines, manifests and per-study predictions stay in MCP.

## 3. Phase 2 validation

1. Score the locked cohort inside MCP with the exported bundle, using `mri_training.runtime.pipeline.analyse` and `interpret` (the code the container runs). Write `cardioai_work.mri_predictions` with these columns:
   - `person_id`;
   - `agent_infarct_prob`, `agent_infarct`, `agent_nonischemic_prob`, `agent_nonischemic`;
   - `agent_thrombus_prob`, `agent_thrombus`, `agent_aneurysm_prob`, `agent_aneurysm`;
   - `agent_infarct_size_pct`, `agent_mvo_pct`, `agent_mvo`, `agent_lvef`, `agent_red_flags`;
   - `runtime_s` and `runtime_fingerprint`.

   The fingerprint must equal the model card's. The contract is the `predictions:` block in `config/cohorts/mri.yaml`.
2. Evaluate:
   ```bash
   python scripts/evaluate_cohorts.py --cohorts mri --boot 5000
   ```
   The result goes to `exports/mri_endpoints.csv`. Success needs the lower 95% CI bound to clear each primary goal.

## 4. Deployment

1. Deploy to Vertex, outside MCP. Studies are passed by `gs://` reference, because Vertex limits a request to 1.5 MB.
   ```bash
   gsutil cp weights.pt thresholds.json model_card.json gs://<bucket>/mri/v1/
   BUNDLE_URI=gs://<bucket>/mri/v1 STUDY_PREFIX=gs://<bucket>/mri-studies/ IMAGE_TAG=v1 mri_training/serve/deploy.sh
   ```
   The container refuses to start if its runtime fingerprint, weights hash or thresholds differ from the model card. It reads studies only under `STUDY_PREFIX`.
2. Lock it in the app (`app/validated_agents/registry.yaml` → `mri` in cardioaimedconnect).
   - Fill in `model_id`, `endpoint_id`, `study_prefix`, `weights_sha256`, the four thresholds and the clinical lead's red-flag cut-offs, then set `status: research_only`.
   - The runtime fingerprint is already pinned. The app answers 409 to any container that differs, and it re-applies its own thresholds and red-flag rules.
3. Put a timeout of at least 120 s on `/mri/predict`, because every cine phase is segmented.
4. After Phase 2 passes, open a PR that changes only `status: validated` and the validation report link. The clinical lead and the QA/regulatory owner sign it off.

## 5. Checks and stop conditions

| Step | Must hold | If not |
|---|---|---|
| Cohort lock | Phase 2 cohort snapshotted before any training run | Stop and re-lock |
| Patient IDs | Every study has an OMOP `person_id` through `person.person_source_value`; no patient in two splits | Stop: splits and STEMI links depend on it |
| DICOM request | Only SA LGE and cine series; studies without LGE counted | Review the series descriptions |
| Label audit | Every report label PASS in `audit_results.csv`; no adjudication pending | Fix the rule, rebuild, re-audit on a fresh sample |
| Pretraining | Card shows only studies before 2020 | Rebuild the manifest |
| Annotation | `collect` rejects unreviewed or invalid files; double-read Dice reviewed; outline QC graded, grade C redrawn | Return the studies to readers |
| Thresholds | Fitted on 2020 only; reviewed before the test set is opened | Never refit on test data |
| Export | Only the three bundle files leave MCP | Stop |
| Serving | Fingerprint, weights hash and thresholds match the model card | Rebuild from the exact commit |

## 6. Open items

- [ ] MCP: how `PATIENT_CLINIC_NUMBER` maps to the OMOP `person_id`; the de-identification request steps, turnaround and delivery folder; the final `SERVICE_RESULT_STATUS` values.
- [ ] Clinical lead: MRI goals, red-flag cut-offs, and the reader protocol (scar threshold, basal slice).
- [ ] Hire up to 5 outside cardiac MRI readers; each submits the DUA form.
- [ ] Write the scoring script that produces `mri_predictions` from the locked bundle.

See also: `mri_training/README.md` (model details), `annotation/README.md` (reader protocol and tools), `VALIDATION_RUNBOOK.md` (the full Phase 2 study), and `RUNBOOK_ECG.md`.
