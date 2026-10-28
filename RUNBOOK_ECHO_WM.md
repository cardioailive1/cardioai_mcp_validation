# Echo wall-motion agent runbook (MobileNetV3-WM) on Mayo Clinic Platform

How to train, validate and deploy the echo wall-motion model. It extends the echo agent, whose LVEF comes from the EchoNet-Dynamic segmentation model and which assesses no wall motion. The new model reads the apical 4-, 2- and 3-chamber cines of a transthoracic echo and gives two kinds of output.

- **Regional (coronary territory mapping):**
  - a wall-motion score for each of the 17 AHA segments (normal, hypokinetic, akinetic, dyskinetic/aneurysmal);
  - each abnormality mapped to the territory it is consistent with: LAD, RCA or LCx.
- **Global (heart-failure assessment):**
  - an LVEF estimate, global hypokinesis, and the wall motion score index (WMSI);
  - in the app, these sit beside EchoNet's volumetric LVEF, which stays the primary EF and sets the HFrEF / HFmrEF / HFpEF category.

The model is trained **only on Mayo data** inside the MCP workspace, like the ECG and MRI models. No public datasets or outside pretrained weights are used.

| | |
|---|---|
| Model | MobileNetV3-Small frame encoder + temporal motion block + one learned "slot" per segment position in each apical view (`echo_wm_training/`); export approved (MobileNetV3) |
| Input | Greyscale apical cines (A4C, A2C, A3C), at least one, all three for all 17 segments; 1.6 s at 20 fps, 112 px |
| Periods | Echos before 2020-01-01 train it, 2020 tunes it, 2021 onward (plus held-out sites) tests it. Patients in the locked Phase 2 echo cohort never train it |
| Mayo data | **Reports:** `FACT_ECHO_TEST.ECHO_NOTE`. **Images:** echo DICOM requested for de-identification from `FACT_RADIOLOGY_DICOM_INVENTORY` (`MODALITY = 'US'`) |
| Labels | Automatic: the report's wall-motion statement parsed into 17 segment scores (`labels.parse_report`), audited by data scientists and the cardiologist (`annotation/LABEL_AUDIT.md`). Views: automatic first (DICOM tags, the machine's measurement reports, clusters whose tagged clips agree); people label only the remaining clusters (about 60 pictures) and, if needed, a few hundred single clips |
| Phase 2 goals | Any wall-motion abnormality: sensitivity and specificity ≥ 0.85 (deck). Proposed (clinical-lead sign-off sheet): each territory ≥ 0.80, global hypokinesis ≥ 0.85, WMSI error ≤ 0.20 |
| Exported | `weights.pt`, `thresholds.json`, `model_card.json` only |

## 1. Before you start

| Item | What to do |
|---|---|
| Accounts, trainings, GPU | As for the MRI agent (RUNBOOK_MRI.md §1): DUA forms, Radiology training, an A100 40 GB from the shared reservation |
| Code | `cardioai_mcp_validation` (main) in the workspace; `pip install -r requirements.txt -r requirements-echo-wm.txt` (CUDA torch) |
| Echo link | MCP (2026-10-21): no shared key; a `FACT_ECHO_TEST` row links to its DICOM study by `PATIENT_CLINIC_NUMBER` and date (`echo.mcp.link.mode: clinic_date`). MCP (2026-10-28): echo DICOM is in `FACT_RADIOLOGY_DICOM_INVENTORY` (`MODALITY = 'US'`, echo named in `STUDY_DESCRIPTION`) and is requested like CMR; `ECHO_PERFORMED_TEST_DK` → `DIM_ECHO_TEST` names the ordered test. Both sides are kept to transthoracic echo (`echo.mcp.filters`, proposed) before the date link; patient-days still holding two TTE reports or studies are dropped and counted (`find_echo_studies`). Confirm the `DIM_ECHO_TEST` key and description names, then check the filters against the real description counts |
| Patient IDs | OMOP `person.person_source_value` = `PATIENT_CLINIC_NUMBER` (MCP 2026-10-28): `echo.mcp.person_map` is filled in |
| Expert LVEF | MCP (2026-10-28): `FACT_ECHO_RESULTS.ECHO_RESULT_VALUE`, type in `DIM_ECHO_TEST_RESULT_TYPE` (`ECHO_TEST_RESULT_NAME` / `_DESCRIPTION`). The OMOP ETL loads no echo tables, so this is the LVEF source: `echo.mcp.results` (confirm the key names; the method order biplane > 3-D > single-plane > visual is proposed for the clinical lead) |
| Clinical lead | Signs the sign-off sheet "Echo Wall-Motion Agent: Clinical Lead Sign-off": the echo rule sheet (LABEL_AUDIT.md §1.3), the A3C apical slots (proposed: not scored) and the AHA territory map (`runtime/segments.py`), the red-flag cut-offs (`config.yaml` → `red_flags`) and the Phase 2 goals (`config/cohorts/echo.yaml`). The code already holds the proposals; amendments are applied before training |

Queries run in BigQuery through `make_engine("bigquery", study)` (RUNBOOK_MRI.md, "Querying the data").

## 2. Steps

1. **Lock the Phase 2 echo cohort first.** The wall-motion endpoints are evaluated on it, so its patients must never train the model.
   First build the expert-LVEF table (the cohort's reference standard) from `FACT_ECHO_RESULTS`, and point the cohort at it:
   ```python
   from echo_wm_training import mcp_echo as me
   lvef = me.build_lvef_table(eng, study["modality"])        # one LVEF per patient and echo date, with its method
   eng.write_table(lvef, "echo_lvef")
   # config/cohorts/echo.yaml -> macros.lvef_rows: switch to the {{work}}.echo_lvef form
   ```
   Then:
   ```bash
   python scripts/run_cohorts.py --cohorts echo --save-tables
   ```
   Snapshot the analytic table as `cardioai_work.echo_analytic_locked`.

2. **Find echo studies and build the de-identification request.**
   ```python
   from cardioai_mcp.config import load_study; from cardioai_mcp.engine import make_engine
   from echo_wm_training import mcp_echo as me
   study = load_study(); eng = make_engine("bigquery", study)
   studies = me.find_echo_studies(eng, study["modality"])           # echo reports joined to their DICOM study
   req, report = me.build_request(studies)                           # de-identifiable + a wall-motion statement
   me.deid_request(studies, req).to_csv("/workspace/echo/deid_request.csv", index=False); report
   ```
   Send `deid_request.csv` (STUDY_INSTANCE_UID, PATIENT_DK, MODALITY, STMO_NAME_IN_ADL) to the MCP TSM. Echo studies are requested whole; the cines are selected after delivery.

3. **After delivery, build the tables.**
   ```python
   t = me.build_study_table(studies, "/workspace/echo/dicom")
   eng.write_table(t, "echo_wm_study")
   ref = me.reference_table(studies)                                 # report-derived reference (Phase 2 and audit)
   eng.write_table(ref, "echo_wm_reference")
   from echo_wm_training.dataset import load_config, build_manifest, build_clip_table
   cfg = load_config()
   excl = eng.read_table("echo_analytic_locked").person_id
   m = build_manifest(eng, study, cfg, study["modality"], exclude_persons=excl)
   m.to_parquet("/workspace/echo/manifest.parquet")
   clips = build_clip_table(m)                                       # every 2-D greyscale cine (colour, Doppler, M-mode skipped)
   clips.to_parquet("/workspace/echo/clips.parquet")
   ```
   Adults only: tests done before age 18 (`splits.min_age`, the Phase 2 cohort rule) are dropped from every split, and `m.attrs["n_under_min_age"]` gives the count.
   Set `echo_wm.enabled: true` in `config/modality_tables.yaml` once `echo_wm_reference` exists.

4. **Audit the automatic labels** (data scientists first, then the cardiologist; LABEL_AUDIT.md).
   ```bash
   python -m annotation.label_audit sample echo --manifest /workspace/echo/manifest.parquet --auditors ds1,ds2 --auto triage --out /workspace/audit/echo
   python -m annotation.label_audit score echo --dir /workspace/audit/echo    # before and after the cardiologist
   ```
   Every label must PASS (PPV and NPV ≥ 0.95, lower bound ≥ 0.90). On a FAIL, fix `labels.py`, rebuild and re-audit on a fresh sample.

5. **Get the clip views, automatic sources first.**
   ```bash
   # tags + the machine's measurement reports (SR): what share of clips and studies they already label
   python -m echo_wm_training.view_sources coverage --clips /workspace/echo/clips.parquet --studies /workspace/echo/dicom --out /workspace/echo/views
   ```
   - **Tags:** what the scanner wrote in the DICOM.
   - **SR:** a clip that a sonographer measured on (biplane EF, volumes) is referenced by the study's measurement report together with its view.
   - `coverage.json` shows how many clips and studies each source covers.

   Then group the remaining clips (after step 6, which builds the encoder):
   ```bash
   python -m echo_wm_training.view_sources clusters --clips /workspace/echo/clips.parquet --sr /workspace/echo/views/sr_views.csv \
          --encoder /workspace/echo/runs/ssl_v1 --k 60 --out /workspace/echo/views/clusters
   #   a group whose tagged / SR clips agree (>= 90%, >= 10 clips) takes their view automatically;
   #   the data scientists label the other groups in clusters.xlsx from one picture of 8 clips each ('mixed' = leave out)
   python -m echo_wm_training.view_sources read-clusters --dir /workspace/echo/views/clusters --out /workspace/echo/views/cluster_views.csv
   ```
   ```python
   from echo_wm_training.views import assign_views
   c = assign_views(pd.read_parquet("/workspace/echo/clips.parquet"), sr=pd.read_csv("/workspace/echo/views/sr_views.csv"),
                    clusters=pd.read_csv("/workspace/echo/views/cluster_views.csv"))
   c.to_parquet("/workspace/echo/clips_views.parquet")     # precedence: person > tag > SR > cluster; the view model fills the rest
   ```
   - **Per-clip labels:** only when a view is still short, label clips one at a time (`python -m echo_wm_training.views sheet --n 300 ...`).
   - **Spot-check:** the cardiologist checks 100 clips across the sources. Agreement per source goes into the model card.

6. **Pretrain the encoder on Mayo cines** (self-supervised, training period only).
   ```bash
   python -m echo_wm_training.pretrain --manifest /workspace/echo/manifest.parquet --clips /workspace/echo/clips.parquet \
          --out /workspace/echo/runs/ssl_v1 --cache-dir /workspace/echo/cache --device cuda
   ```
   It refuses studies dated on or after 2020-01-01.

7. **Train.**
   ```bash
   python -m echo_wm_training.train --manifest /workspace/echo/manifest.parquet --clips /workspace/echo/clips_views.parquet \
          --init /workspace/echo/runs/ssl_v1 --out /workspace/echo/runs/wm_v1 --cache-dir /workspace/echo/cache --device cuda
   ```
   - Stage V learns views from the labelled and tagged clips, then labels the remaining clips (only predictions with P ≥ 0.9 are used).
   - Stage W learns segment scores, LVEF and global hypokinesis; early stopping on the 2020 tuning split.
   - Thresholds for any abnormality, each territory and global hypokinesis are fitted on 2020 only.

8. **Review `model_card.json`** before opening the test set:
   - data summary, clip views by source, unreadable clips;
   - thresholds; AUROC, sensitivity and specificity per head (territories, any abnormality, global);
   - segment-score accuracy and weighted kappa, WMSI and LVEF against the report, EF-category agreement;
   - runtime fingerprint, weights hash, `initialised_from` (pretraining period).

9. **Export** only `weights.pt`, `thresholds.json` and `model_card.json`. Cines, reports, manifests and per-study predictions stay in MCP.

## 3. Phase 2 validation

1. Score the locked echo cohort inside MCP with `echo_wm_training.runtime.pipeline.analyse` and `interpret` (the code the container runs) on each index echo's apical cines. Add these columns to `cardioai_work.echo_predictions`:
   - `agent_wall_motion`, `agent_terr_lad`, `agent_terr_rca`, `agent_terr_lcx`, `agent_global_hypokinesis` (abnormal / normal);
   - `agent_wmsi`, `agent_wm_lvef`, `agent_wm_views`, `wm_runtime_fingerprint`.

   The contract is the `predictions:` block in `config/cohorts/echo.yaml`. EchoNet's `agent_lvef` stays the primary EF.
2. Evaluate:
   ```bash
   python scripts/evaluate_cohorts.py --cohorts echo --boot 5000
   ```
   The wall-motion endpoints (`wall_motion_sens/spec`, territory and global endpoints, `wmsi_mae`) are reported with the EF endpoints.

## 4. Deployment

1. Deploy to Vertex, outside MCP. Cines are passed by `gs://` reference (Vertex limits a request to 1.5 MB).
   ```bash
   gsutil cp weights.pt thresholds.json model_card.json gs://<bucket>/echo-wm/v1/
   BUNDLE_URI=gs://<bucket>/echo-wm/v1 STUDY_PREFIX=gs://<bucket>/echo-studies/ IMAGE_TAG=v1 echo_wm_training/serve/deploy.sh
   ```
   The container refuses to start if its runtime fingerprint, weights hash or thresholds differ from the model card.
2. Lock it in the app (`app/validated_agents/registry.yaml` → `echo_wall_motion` in cardioaimedconnect): `model_id`, `endpoint_id`, `clip_prefix`, `runtime_fingerprint`, `weights_sha256`, the five thresholds and the signed-off red flags; then `status: research_only`.
3. The app's `/echo/predict` then returns the wall-motion block beside EchoNet's EF when A2C / A3C cines (or `wall_motion: true`) are sent; `/echo/wall-motion/predict` runs it alone.

## 5. Checks and stop conditions

| Step | Must hold | If not |
|---|---|---|
| Cohort lock | Phase 2 echo cohort snapshotted before any training run | Stop and re-lock |
| Echo link | Clinic-number + date link run; the share of patient-days dropped as ambiguous is small and reported | Stop: labels would attach to the wrong images |
| Label audit | Every echo label PASS; no adjudication pending | Fix `labels.py`, rebuild, re-audit on a fresh sample |
| Views | `coverage.json` reviewed; view labels spot-checked by the cardiologist per source; most training clips labelled by a source or confidently predicted | Label more clusters or clips |
| Pretraining | Card shows only studies before 2020 | Rebuild the manifest |
| Thresholds | Fitted on 2020 only; reviewed before the test set is opened | Never refit on test data |
| Export | Only the three bundle files leave MCP | Stop |
| Serving | Fingerprint, weights hash and thresholds match the model card | Rebuild from the exact commit |

## 6. Open items

- Answered by MCP (2026-10-28): echo DICOM is in the inventory; `person_id` via `person.person_source_value`; LVEF in `FACT_ECHO_RESULTS`; same-day studies told apart by `DIM_ECHO_TEST` and `STUDY_DESCRIPTION`.
- To confirm in the workspace: the `DIM_ECHO_TEST`, `FACT_ECHO_RESULTS` and `DIM_ECHO_TEST_RESULT_TYPE` key names; whether `FACT_ECHO_TEST` still carries `PATIENT_DK` (MCP is retiring it in some tables) or `PATIENT_CLINIC_NUMBER`.
- Clinical lead: sign the sign-off sheet (rule sheet incl. rules 4 and 8, A3C apical slots, territory map, red flags, goals). Proposals are in the code, marked proposed.
- Territory output is "consistent with" an AHA territory, not a coronary diagnosis: coronary anatomy varies (dominance, wrap-around LAD).
