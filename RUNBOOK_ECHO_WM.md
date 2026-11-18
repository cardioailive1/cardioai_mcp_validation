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
| Echo link | MCP (2026-10-21): no shared key; a `FACT_ECHO_TEST` row links to its DICOM study by `PATIENT_CLINIC_NUMBER` and date (`echo.mcp.link.mode: clinic_date`). MCP (2026-10-28): echo DICOM is in `FACT_RADIOLOGY_DICOM_INVENTORY` (`MODALITY = 'US'`, echo named in `STUDY_DESCRIPTION`) and is requested like CMR; `ECHO_PERFORMED_TEST_DK` → `DIM_ECHO_TEST` names the ordered test. Both sides are kept to transthoracic echo (`echo.mcp.filters`, proposed) before the date link; patient-days still holding two TTE reports or studies are dropped and counted (`find_echo_studies`). Confirm the `DIM_ECHO_TEST` key and description names, then check the filters against the real description counts. Adam Resnick (2026-11-04): the choice of descriptions is ours; use the Term Finder in the My First Query notebook (`/data/intro/Introduction_Notebooks/Accelerate`) to list the echo descriptions and settle `echo.mcp.filters`. `DIM_ECHO_TEST` is a lookup of test types only (no patient or exam), which is how the code uses it |
| Patient IDs | OMOP `person.person_source_value` = `PATIENT_CLINIC_NUMBER` (MCP 2026-10-28): `echo.mcp.person_map` is filled in |
| Expert LVEF | MCP (2026-10-28): `FACT_ECHO_RESULTS.ECHO_RESULT_VALUE`, type in `DIM_ECHO_TEST_RESULT_TYPE` (`ECHO_TEST_RESULT_NAME` / `_DESCRIPTION`). The OMOP ETL loads no echo tables, so this is the LVEF source: `echo.mcp.results` (confirm the key names; the method order biplane > 3-D > single-plane > visual is proposed for the clinical lead; Adam: the preference is ours to set) |
| Clinical lead | Signs the sign-off sheet "Echo Wall-Motion Agent: Clinical Lead Sign-off": the echo rule sheet (LABEL_AUDIT.md §1.3), the A3C apical slots (proposed: not scored) and the AHA territory map (`runtime/segments.py`), the red-flag cut-offs (`config.yaml` → `red_flags`) and the Phase 2 goals (`config/cohorts/echo.yaml`). The code already holds the proposals; amendments are applied before training |

Queries run in BigQuery through `make_engine("bigquery", study)` (RUNBOOK_MRI.md, "Querying the data").

## 2. Steps

1. **Lock the Phase 2 echo cohort first.** The wall-motion endpoints are evaluated on it, so its patients must never train the model.
   First build the expert-LVEF table (the cohort's reference standard) from `FACT_ECHO_RESULTS`, and point the cohort at it:
   ```python
   from datetime import date, timedelta
   from cardioai_mcp.config import load_study
   from cardioai_mcp.engine import make_engine
   from echo_wm_training import mcp_echo as me

   study = load_study()
   eng = make_engine("bigquery", study)
   # The echo cohort accepts an LVEF measurement on the index day or the following day.
   # Restrict the source SQL before transferring rows into pandas.
   start = str(study["index_start"])
   end = (date.fromisoformat(str(study["index_end"])) + timedelta(days=1)).isoformat()
   lvef = me.build_lvef_table(eng, study["modality"], start=start, end=end)
   print(f"LVEF patient-days: {len(lvef):,}")
   print(lvef["lvef"].describe())
   print(lvef["lvef_method"].value_counts())
   # After reviewing these checks, persist the source for the Phase 2 echo cohort.
   eng.write_table(lvef, "echo_lvef")
   ```
   The extraction uses `FACT_ECHO_TEST.ECHO_PERFORMED_DTM`, confirmed in the MCP
   workspace. If the schema changes, verify the date column in
   `config/modality_tables.yaml` before running. The date range is inclusive;
   omitting both date arguments retains the historical unbounded extraction for
   other consumers, but is not recommended for this cohort.

   In `config/cohorts/echo.yaml`, replace the active `macros.lvef_rows` with
   the `{{work}}.echo_lvef` form already shown as a commented example. Do this
   **before** running `run_cohorts.py`; otherwise it continues to read OMOP
   `measurement` instead of the MCP echo result reference.
   Then:
   ```bash
   python scripts/run_cohorts.py --cohorts echo --save-tables
   ```
   Snapshot the analytic table as `cardioai_work.echo_analytic_locked`.

### Accession-aware Echo reference candidates (validation only)

The original `echo_lvef` is patient/day aggregated and remains the active cohort
reference. Do **not** overwrite it or lock the cohort until examination identity
and test-type filtering are independently reviewed. Mayo's verified exam
identifier in `FACT_ECHO_TEST` is `ACCESSION_NUMBER`, and a single accession
can contain several `ECHO_TEST_FPK` rows.

The first comparison extractor is read-only. It queries only the already saved
`{{work}}.echo_analytic` patients and matching index-day/next-day echo results.
Its returned rows preserve accession and count ambiguities among
**measurement-bearing candidate accessions**. They are not yet proof of a
unique eligible TTE or a DICOM match. Its LVEDV method order and LVEF priorities
remain proposed, pending clinical review.

```python
from echo_wm_training.exam_results import build_exam_reference_candidates
candidates = build_exam_reference_candidates(
    eng, study["modality"],
    start=str(study["index_start"]), end=str(study["index_end"])
)
print(candidates.status.value_counts(dropna=False))
# Aggregate only: do not copy/export patient IDs or accession numbers from MCP.
print(candidates.groupby("status")[["lvef", "lvedv"]].count())
```

This function makes no changes to saved tables. Before promotion, implement
complete TTE eligibility and uniqueness checks across **all** candidate echo
records (including exams lacking these measurements), verify multi-date
accessions and the LVEF/LVEDV method definitions, and compare the result
against the current cohort on a stable sample within MCP.

### Provisional TTE accession/order matching (read-only)

MCP validation found that accession-bearing and order-only same-day TTE records
frequently occur within 60 seconds, but timestamps are NOT proof that they
refer to the same clinical examination. The `exam_resolution` diagnostic uses
all TTE-classified `FACT_ECHO_TEST` records on the index date, not just
measurement-bearing tests. It groups records by patient/date and accession
(or `ORDER_PROC_ID` when accession is missing), then identifies candidate
ACC/ORD pairs with a **mutually unique** match within 60 seconds. Groups
with competing candidates are explicitly ambiguous. There is **no** automatic
propagation into LVEF, LVEDV, the DICOM inventory, or the cohort.

```python
from echo_wm_training.exam_resolution import build_tte_match_diagnostics
matches = build_tte_match_diagnostics(
    eng, study["modality"], start=str(study["index_start"]),
    end=str(study["index_end"]), max_seconds=60
)
print(matches.status.value_counts(dropna=False))
print(matches.groupby("key_type").status.value_counts())
```

Only report aggregates outside the MCP workspace; `matches` has patient and
examination identifiers. A provisional matching pair still requires an
independent clinical linkage validation before using it as reference identity.
This query currently compares tests on the **index date** only; the cohort's
one-day LVEF allowance is a separate policy to evaluate before promotion.
The TTE description regex is provisional and subject to clinical review.

### Compare LVEF and LVEDV on provisional ACC/ORD pairs

After checking `tests/test_echo_exam_resolution.py` and
`tests/test_echo_exam_comparison.py`, run this **read-only** comparison in MCP:

```python
from echo_wm_training.exam_comparison import build_tte_measurement_comparison
pairs, summary = build_tte_measurement_comparison(
    eng, study["modality"],
    start=str(study["index_start"]), end=str(study["index_end"]),
    max_seconds=60,
)
print(summary.to_string(index=False))
```

The returned `pairs` DataFrame contains identifiers and must stay in MCP.
`summary` reports LVEF and LVEDV availability on both sides of each
provisional pair, and median/p90 absolute disagreement where **both** sides
have a valid value. A low disagreement supports consistency but does not
confirm two records represent the same clinical examination. No merging or
overwriting occurs. The query checks TTE-classified records on **index day
only**; it does not address next-day reference eligibility or clinically
validate test-type/measurement-method rules. Never use these candidate
measurements in the production cohort before completing those checks.

### Unified ACC/ORD reference coverage versus active LVEF (read-only)

MCP diagnostics confirm that both Mayo catalogs store numeric EF in the same
percentage scale. The ACC catalog includes `MC CV ECHO EJECTION FRACTION`;
the ORD catalog includes named LVEF methods. The following diagnostic reads
**index-day TTE-classified records only** and groups by verified identifier
or provisional mutually unique ACC/ORD time match. A person with more than
one unresolved examination candidate is excluded from value comparison.

```bash
python -m pytest tests/test_echo_exam_resolution.py tests/test_echo_exam_comparison.py tests/test_echo_unified_reference.py -q
```

```python
from echo_wm_training.unified_reference import build_unified_echo_diagnostics

candidate_exams, comparison_rows, summary = build_unified_echo_diagnostics(
    eng, study["modality"],
    start=str(study["index_start"]), end=str(study["index_end"]),
    max_seconds=60,
)
print(candidate_exams.candidate_status.value_counts(dropna=False))
print(candidate_exams.lvef_source.value_counts(dropna=False))
print(summary.to_string(index=False))
```

The `candidate_exams` and `comparison_rows` DataFrames contain patient and
examination identifiers and must remain inside the MCP workspace. `summary`
reports available LVEF/LVEDV, the number with both new and saved LVEF, and
absolute EF differences for single-exam candidates. It does **not** replace
`{{work}}.echo_lvef` or modify the saved cohort.

Limitations: the cohort may have no qualifying index-day TTE, so the
`all_index_dates` row is only the set of indexes represented by extracted TTE
candidates, not necessarily all 1,500; next-day echo matches are not yet
evaluated. Provisional ACC/ORD matches require independent clinical review.
EF/EDV method priorities and measurement equivalence also need clinical
sign-off. Do not promote these results automatically.

### Modified Quinones method recognition and EF disagreement review

Observed MCP source case: four EF values for the same patient/day were
59 (Modified Quinones 2D), 13 (visual estimate), 63 (Modified Quinones
with apical correction), and 63 (MC CV ECHO EJECTION FRACTION). The
previous priority regex did not recognize `mod_Quin` as biplane, and the
visual estimate won. Modified Quinones is **not** Simpson biplane.

`config/modality_tables.yaml` now recognizes `modified_quinones` as
its own method, provisionally ranked after single-plane and before
visual. **Clinical review is required** before treating that priority
as definitive. The read-only unified candidate comparison additionally
flags a within-identifier-group range greater than 10 EF percentage
points as `unresolved_method_disagreement`. Such a candidate's
LVEF is withheld (null) from the single-exam baseline comparison.
The threshold is a provisional diagnostic cutoff, not a validated
clinical eligibility rule. This is different from the existing
`echo_lvef` writer: do not rerun/persist that writer or modify the
locked cohort based on this provisional method priority.

After pulling the branch, rerun synthetic tests:
```bash
python -m pytest tests/test_echo_exam_results.py tests/test_echo_exam_comparison.py tests/test_echo_unified_reference.py -q
```
Then rerun `build_unified_echo_diagnostics` from the preceding section
and inspect only aggregate candidate statuses and counts in MCP.

### Regression: conventional TTE versus same-day TTE Strain (2026-11-17)

MCP investigation resolved the previous 50 percentage-point discrepancy in the
saved patient/day LVEF reference. Two distinct `ECHO_TEST_FPK` records existed
for the same patient/date: a conventional `(TTE) 2D ECHO DOPPLER COLOR AND
CONTRAST` record with `MC CV ECHO EJECTION FRACTION = 63`, and a separate
`TTE Strain` record with Modified Quinones 59 and 63 plus a visual estimate
of 13. The historical patient/date extractor had classified visual 13 ahead
of those differently named methods and did not enforce conventional exam
eligibility before choosing EF.

The read-only unified candidate builder now defensively filters eligible
`ECHO_TEST_FPK` records *before* measurement extraction, even if an upstream
caller passes unfiltered test/results rows. The SQL-level TTE query already
excludes `strain`; the Python guard adds regression protection. Two synthetic
tests exercise this exact dual-record scenario and strain-only input.
A separate `TTE Strain` measurement cannot overwrite conventional TTE EF
solely because the index date is identical. This does not determine whether
the two records clinically describe related imaging procedures.

```bash
python -m pytest tests/test_echo_exam_resolution.py tests/test_echo_exam_comparison.py tests/test_echo_unified_reference.py -q
```

This policy remains a **read-only diagnostic**. Existing `echo_lvef` and the
cohort are untouched. The 302 provisional within-group disagreements remain
reviewable rather than being accepted or permanently discarded; neither the
10-point review threshold nor Modified Quinones priority has clinical sign-off.

### Separate within-method and between-method EF disagreement (read-only)

For all plausible LVEF values within each eligible exam identifier group,
the new diagnostic records:

- `lvef_within_method_spread_pp`: the largest max-minus-min value
  **inside one classified method** (e.g., two visual estimates).
- `lvef_between_method_spread_pp`: the difference between the largest
  and smallest **per-method medians**, when more than one method exists.
- `lvef_disagreement_category`: `none`, `within_method`,
  `between_method`, `both`, or `mixed_spread`. Matched ACC/ORD
  pairs can additionally be `multiple_categories`.

This is a diagnostic categorization, NOT a revised clinical policy. The
existing >10-percentage-point **total EF spread** remains the exclusion
trigger; `unresolved_method_disagreement` stays unchanged until clinical
review. There is no change to method priority, patient eligibility, or any
saved BigQuery table.

```bash
python -m pytest tests/test_echo_exam_comparison.py tests/test_echo_unified_reference.py -q
```

After restarting IPython and running the prior
`build_unified_echo_diagnostics` call, use the **aggregate-only** summary:

```python
print(candidate_exams.loc[
    candidate_exams["lvef_method_disagreement"],
    "lvef_disagreement_category",
].value_counts(dropna=False).to_string())
print(candidate_exams.groupby(
    ["record_source", "lvef_disagreement_category"]
).size().to_string())
```

Detailed candidate rows contain sensitive identifiers and must stay inside MCP.
The source test labels and method-priority mappings remain subject to clinical
review.

### Result-name diversity inside within-method disagreements (read-only)

MCP reported 144 within-method, 74 between-method, 67 mixed-spread,
and 17 both-type flagged examination groups. This follow-up examines the
144 `within_method` candidates. It distinguishes an exact
`ECHO_TEST_RESULT_NAME` repeated with discordant values from two different
result names grouped into the same configured method family. A matched ACC/ORD
pair is analyzed **within each original group**, not across the pair.
A different result name does not necessarily mean a clinically distinct method.

```bash
python -m pytest tests/test_echo_ef_name_diagnostics.py tests/test_echo_unified_reference.py -q
```

After restarting IPython and running `build_unified_echo_diagnostics`
from above, use this additional **read-only** MCP query and aggregation:

```python
from echo_wm_training.exam_comparison import tte_measurement_rows_sql
from echo_wm_training.ef_name_diagnostics import diagnose_within_method_names

rows_sql = tte_measurement_rows_sql(
    study["modality"], start=str(study["index_start"]),
    end=str(study["index_end"]),
)
measurement_rows = eng.query(eng.render(rows_sql, {}))
name_details, name_summary = diagnose_within_method_names(
    measurement_rows, candidate_exams,
    study["modality"]["echo"]["mcp"]["results"],
)
print(name_details.name_conflict_class.value_counts(dropna=False).to_string())
print(name_summary.to_string(index=False))
```

`name_details` and `measurement_rows` contain examination/patient identifiers
and must stay in the restricted MCP environment. The aggregate
`name_summary` reports `same_name_only`, `different_names_only`,
`both`, and `neither` grouped by original ACC/ORD provenance. The
comparison uses a strict >10 percentage-point difference between numeric
values and unchanged method-regex precedence. It does not update the EF
classification, exclusion rule, saved `echo_lvef`, or cohort.
`neither` means the threshold was not exceeded *within an individual
original group and method family* by these checks, not that an examination
is clinically resolved. The original 302 flags remain unchanged.

### Mayo EF method taxonomy correction (read-only diagnostics)

MCP examination of actual `result_label` values confirmed that three separate
Mayo MOD result types were classified as `biplane` by the old broad
`\\bmod\\b` match: `LV_EF_2C_2D_by_MOD`,
`LV_EF_4C_2D_by_MOD`, and `LV_EF_BP_2D_by_MOD`.
The first two are single-plane views; only `BP` is explicitly biplane.

The taxonomy now recognizes those exact Mayo names, retains the existing
provisional priority **tiers** (biplane > 3D > single-plane >
Modified Quinones > visual > other), and introduces **diagnostic variants**
within the single-plane and Modified Quinones tiers:
`single_plane_2c_mod`, `single_plane_4c_mod`,
`quinones_standard`, and `quinones_apical_correction`. The
`MC CV ECHO EJECTION FRACTION` name remains `other` for clinical
selection priority while receiving the diagnostic label `unspecified_ef`.

The diagnostic within-method spread now measures within each variant;
differences between variants are classified as between-variant disagreements
even when their priority tier is the same. This improves attribution and
**does not change** the >10-point total-range review flag, automatic
withholding of discordant unified candidates, clinical approval requirements,
or saved tables. Correcting the falsely-biplane single-plane regex can change
which value a *read-only* candidate chooses when no review flag applies.
Do not rerun the historical `echo_lvef` table writer or promote the
provisional taxonomy without clinical review.

```bash
python -m pytest tests/test_echo_unified_reference.py tests/test_echo_ef_name_diagnostics.py tests/test_echo_exam_comparison.py -q
```

Recompute `build_unified_echo_diagnostics` after restarting IPython and
compare category counts with the previous snapshot of 302 flags:
144 within-method, 74 between-method, 67 mixed, 17 both. Changes to
subcategory counts are expected; a large shift in the *total number of
flagged groups* or eligibility warrants investigation before proceeding.

### EF method-pair disagreement ranking (read-only)

After correcting the Mayo 2C, 4C, biplane, and Quinones variant taxonomy,
the prior 302 EF review flags redistributed to **293 between-method**,
**8 mixed-spread**, and **1 within-method**. The 293 between-method groups
are the subject of a separate **read-only** pair diagnostic. This does not
alter the total-range threshold, the reference policy, or any saved table.

```bash
python -m pytest tests/test_echo_ef_method_pair_diagnostics.py -q
```

In a fresh MCP IPython session after running the previous
`build_unified_echo_diagnostics`, execute the following. It reads only
eligible TTE-linked measurement rows, and summarizes absolute differences
between the *median* EF for each diagnostic method variant **within the same
original ACC or ORD identifier group**. It never compares measurements
across an unconfirmed ACC/ORD pair.

```python
from echo_wm_training.exam_comparison import tte_measurement_rows_sql
from echo_wm_training.ef_method_pair_diagnostics import diagnose_ef_method_pairs

sql = tte_measurement_rows_sql(
    study["modality"],
    start=str(study["index_start"]),
    end=str(study["index_end"]),
)
rows = eng.query(eng.render(sql, {}))
pair_details, pair_summary, source_coverage = diagnose_ef_method_pairs(
    rows, candidate_exams, study["modality"]["echo"]["mcp"]["results"]
)
print("\\nTop method pairs with >10 EF-point median difference:")
print(pair_summary.head(20).to_string(index=False))
print("\\nSource coverage:")
print(source_coverage.to_string(index=False))
```

`pair_details` is **patient-level confidential data** and must remain in
MCP. The aggregate summary shows method pairs ranked by number of
candidate examinations affected, number of original identifier groups,
median/p90/max absolute EF difference. A single examination can have multiple
discordant pairs; **sum of pair counts is not the number of unique
examinations**. `source_coverage.candidates` is the number of groups *in
scope* from the 293, while `source_groups` and `flagged_pairs` count
underlying original groups and pair occurrences exceeding the same strict
10 EF-point review threshold. Small-cell reporting must follow MCP policy.
The diagnostic also retains pairs not exceeding the threshold in private
`pair_details` so their presence can be audited locally.

2. **Find echo studies and build the de-identification request.**
   ```bash
   python scripts/deid_request.py --modality echo --dry-run             # counts per step and year; nothing written
   python scripts/deid_request.py --modality echo --out /workspace/deid  # deid_request_echo.csv + selection + summary
   ```
   Echo reports are joined to their DICOM study (clinic number + date, transthoracic only); a study is selected when it is de-identifiable and its report has a wall-motion statement. Send `deid_request_echo.csv` (STUDY_INSTANCE_UID, PATIENT_DK, MODALITY, STMO_NAME_IN_ADL) to the MCP TSM; `PATIENT_DK` comes from `DIM_PATIENT` by clinic number. Echo studies are requested whole; the cines are selected after delivery. Run this **after step 1**: the script reads the Phase 2 echo cohort (`echo_analytic_locked`, else `echo_analytic`) and always requests each cohort patient's index echo, including those with no wall-motion statement, because the EF agent is scored on them. It stops if the cohort table is missing. Echo is far more common than CMR: run the dry run first and agree the number with the TSM; `--max-studies N` caps only the training studies. Only adults are requested (age ≥ 18 at the study; unknown or redacted birth year: not requested). A clinic number with no unmerged `DIM_PATIENT` row, or with several, is left out and counted. The file is checked (exact columns, no empty cells or duplicates, MR / US, DICOM UIDs) and nothing is written if a check fails.

3. **After delivery, build the tables.**
   ```python
   import pandas as pd
   from cardioai_mcp.config import load_study; from cardioai_mcp.engine import make_engine
   from echo_wm_training import mcp_echo as me
   study = load_study(); eng = make_engine("bigquery", study)
   studies = pd.read_parquet("/workspace/deid/deid_selection_echo.parquet")   # written by scripts/deid_request.py
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
