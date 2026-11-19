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

### Population-wide EF method pair comparison and clinical-review package

The above flagged-only pair summary (823 flagged pair occurrences in 293
candidate groups) is conditioned on >10 EF-point disagreement and cannot
estimate overall method bias. A separate read-only summary now includes
**all** observed variant pairs from the `single_exam_candidate` and
`unresolved_method_disagreement` groups, regardless of their EF difference.
Ambiguous/unresolved identifier candidates are excluded from that comparison.

```bash
python -m pytest tests/test_echo_ef_method_pair_diagnostics.py tests/test_echo_unified_reference.py -q
```

In a fresh MCP IPython session, this complete code computes the additional
aggregate pair summary. Results remain unverified until run in MCP:

```python
from cardioai_mcp.config import load_study
from cardioai_mcp.engine import make_engine
from echo_wm_training.unified_reference import build_unified_echo_diagnostics
from echo_wm_training.exam_comparison import tte_measurement_rows_sql
from echo_wm_training.ef_method_pair_diagnostics import summarize_all_ef_method_pairs

study = load_study()
eng = make_engine("bigquery", study)
start, end = str(study["index_start"]), str(study["index_end"])
cfg = study["modality"]["echo"]["mcp"]["results"]

candidate_exams, comparison_rows, candidate_summary = build_unified_echo_diagnostics(
    eng, study["modality"], start=start, end=end, max_seconds=60
)
sql = tte_measurement_rows_sql(study["modality"], start=start, end=end)
measurement_rows = eng.query(eng.render(sql, {}))

all_pair_details, all_pair_summary = summarize_all_ef_method_pairs(
    measurement_rows, candidate_exams, cfg
)
print(all_pair_summary.head(25).to_string(index=False))
print("Pair observations:", len(all_pair_details))
print("Distinct candidates with >=2 methods:",
      len(all_pair_details[["person_id", "index_date", "candidate_key"]].drop_duplicates()))
```

Only the aggregated summary should be reviewed outside MCP, subject to
small-cell disclosure rules. `all_pair_details`, `measurement_rows`,
`candidate_exams`, and `comparison_rows` contain patient/exam identifiers.
Each pair's direction is `method_a - method_b`; positive values mean
method A higher. A candidate may contribute more than one pair and,
for provisional ACC/ORD matching, measurements are compared *separately*
within each original side. This report is still not a clinically validated
sample: reference hierarchy, cross-system identity, and +1-day allowance
need clinical approval.

See `docs/ECHO_EF_CLINICAL_REVIEW.md` for the evidence summary and
clinical sign-off questions. Neither script writes to BigQuery.

### Next-day LVEDV coverage audit (read-only, not a cohort rebuild)

The conventional TTE SQL defaults to strict index-day matching. The
`day_offset=1` switch is exclusively for investigating tests performed
the following day; it does **not** change the current reference selection,
acceptability decisions, or persisted `echo_analytic` rows.

Previously observed in MCP: 664 conservative index-day LVEDV candidates,
287 index-day patients with LVEDV but excluded because of EF disagreements
(and no detected LVEDV-specific conflict), and 18 additional patient-days
from `MC CV LV END DIASTOLIC VOLUME`, each linked to a single-exam
candidate. The resulting 969 *potential* covered patient-days are not
approved reference values. We must not assume +1-day tests are the index
examination merely because they belong to the same patient.

```bash
python -m pytest tests/test_echo_next_day_lvedv.py tests/test_echo_exam_resolution.py -q
```

In the existing MCP IPython session with `study`, `eng`, `cohort`,
`candidates`, and the prior `recovered` 18-row DataFrame:

```python
from echo_wm_training.next_day_lvedv import (
    next_day_lvedv_rows_sql, summarize_next_day_lvedv,
)

next_sql = next_day_lvedv_rows_sql(
    study["modality"],
    start=str(study["index_start"]),
    end=str(study["index_end"]),
)
next_day_rows = eng.query(eng.render(next_sql, {}))
next_day_evidence, next_day_summary = summarize_next_day_lvedv(
    next_day_rows, cohort, candidates, recovered,
)
for label, value in next_day_summary.items():
    print(f"{label}: {value}")
```

**Keep** `next_day_rows` and `next_day_evidence` within MCP (they
contain patient/exam identifiers). Aggregate numbers are evidence only.
A conservative next-day candidate has precisely one next-day
`ECHO_TEST_FPK`, a numeric absolute LVEDV of 20–500 mL from one of
the five candidate result names, and no disagreement among values under
the *same* result name. This is deliberately conservative and still
requires clinical examination identity and timing review. A broader
`next_day_any_valid_lvedv_while_missing` count is an availability
ceiling in the limited mapped source, not a validated reference.

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

### Source-wide conventional-TTE LVEDV availability (read-only)

After the next-day/indexed-volume audits, distinguish **source-data
availability** from cohort-selection effects. The source-wide audit
measures FACT_ECHO_TEST records (identified by ECHO_TEST_FPK) from the
configured study window and compares them with records linked to the
existing echo_analytic index day. It preserves the same conventional
TTE inclusion/exclusion predicates and checks five known absolute LVEDV
result names for numeric values 20–500 mL.

**Important:** ACC and ORD can describe the same clinical examination;
record-level percentages are **not** clinical examination-level or
patient-level completeness and do not validate the 90% cohort threshold.
No LVEF disagreement gate is applied to the availability measure. The
extra Mayo result name and numeric screening range remain provisional,
subject to clinical review. The query is native GoogleSQL and executes
through the BigQuery client, bypassing Spark/SQLGlot date and SAFE_CAST
transpilation problems.

In the MCP Cloud PC (after merging this branch into the MCP source
repository):

```bash
python -m pytest tests/test_echo_source_availability.py -q
```

```python
from cardioai_mcp.config import load_study
from cardioai_mcp.engine import make_engine
from echo_wm_training.source_availability import run_source_availability_audit

study = load_study()
eng = make_engine("bigquery", study)
by_source, safe_by_type = run_source_availability_audit(eng, study)
print("Source-record availability by scope/source:")
print(by_source.to_string(index=False))
print("Test types with non-small cells in both outcome categories:")
print(safe_by_type.to_string(index=False))
```

Query output is aggregate only. Follow MCP disclosure controls when sharing
aggregate counts, including complementary suppression if relevant. Do not
export row-level identifiers or change `echo_analytic` based on these rates.


### Cohort-specific examination-type LVEDV audit (read-only)

The follow-up to the **source-wide** availability report compares
conventional TTE records on the **saved cohort's index day** against
the same patient's provisional examination-matching status. It answers:

- How many unique cohort patient/index-days have a mapped structured
  absolute LVEDV, independently of EF disagreement?
- How many have source LVEDV but no eligible provisional candidate,
  versus source TTE records without mapped LVEDV?
- Which TTE descriptions occur among patient-days with/without mapped
  LVEDV? (A patient-day can have multiple descriptions; do **not**
  sum examination-type patient-days as a cohort denominator.)

The diagnostic uses the same TTE include/exclude filters, five candidate
LVEDV result names (including the additional Mayo label), and a
**provisional** numeric 20–500 mL filter as the source-wide audit. It
does not authorize clinical method substitutions, matching, relaxing
requirements, or changing the existing analytic cohort.

Run after merging `validation/echo-wm-mcp` into MCP source:

```bash
python -m pytest tests/test_echo_cohort_type_availability.py -q
```

Inside the restricted MCP Cloud PC, in fresh IPython:

```python
from cardioai_mcp.config import load_study
from cardioai_mcp.engine import make_engine
from echo_wm_training.cohort_type_availability import run_cohort_type_audit

study = load_study()
eng = make_engine("bigquery", study)
private_detail, overall, by_type = run_cohort_type_audit(eng, study)

print("Mutually exclusive patient-day dispositions:")
print(overall.to_string(index=False))
print("Examination-type availability (overlapping patient-days):")
print(by_type.to_string(index=False))
print("Cohort index-days:", len(private_detail))
```

**Do not print, export or publish `private_detail`** (patient IDs and
index dates). Type rows whose mapped or unmapped cell is smaller than
`small_cell` are withheld. Still apply MCP disclosure rules, including
secondary/complementary suppression when distributing aggregate output.
Tests are synthetic; BigQuery execution must be verified inside MCP.

### Why source LVEDV was not retained as an examination candidate

Follow the cohort-specific availability audit with a diagnosis of its
`source_lvedv_not_candidate` patient-days (41 in the latest MCP audit).
This comparison distinguishes the additionally mapped Mayo result label
from priority-name measurements, checks the provisional candidate's
status, and partitions these patient-days into mutually exclusive
mechanical exclusion reasons. This is **not** automatic recovery or
clinical approval. The source SQL now provides an explicit
`has_priority_lvedv` flag to avoid incorrectly treating tests
containing *both* naming conventions as Mayo-only.

Run in MCP after synchronizing this validation branch:

```bash
python -m pytest \
  tests/test_echo_source_candidate_gap.py \
  tests/test_echo_cohort_type_availability.py -q
```

Fresh IPython session (all dependencies initialized here):

```python
from cardioai_mcp.config import load_study
from cardioai_mcp.engine import make_engine
from echo_wm_training.source_candidate_gap import run_source_candidate_gap_audit

study = load_study()
eng = make_engine("bigquery", study)
private_detail, reasons, candidate_statuses = run_source_candidate_gap_audit(eng, study)

print("Source LVEDV present but no provisional candidate:", len(private_detail))
print("Mutually exclusive patient-day exclusion reasons:")
print(reasons.to_string(index=False))
print("Diagnostic candidate-group status occurrences:")
print(candidate_statuses.to_string(index=False))
```

The patient-day reason totals should sum to **41** for the current
saved cohort and mapping. Candidate-group status occurrence counts
may sum to more than 41 because an excluded patient-day can have
multiple candidate groups. Review reasons before inferring recoverable
measurements. All row-level patient/exam identifiers stay within MCP.
Follow cell suppression and complementary suppression policies before
sharing any small counts outside MCP. Do **not** rebuild
`echo_analytic` or revise the 90% requirement based on this audit.

### Examination topology for the 22 ambiguous source-LVEDV patient-days

After the 41-case source-to-candidate audit, examine the subset whose
reason is `multiple_exam_candidates` (22 in the latest MCP run). This
read-only audit reuses the same 60-second **mutually unique** ACC/ORD
matching logic used by the unified candidate builder. For each cohort
patient/index-day it separately counts raw ECHO_TEST_FPK records,
identifier groups, provisional ACC/ORD pairings, provisional groups
*after* pairing, and candidate groups retaining a numeric LVEDV.

```bash
python -m pytest tests/test_echo_ambiguous_lvedv_exams.py -q
```

Fresh IPython session in MCP (after merging into the MCP source repository):

```python
from cardioai_mcp.config import load_study
from cardioai_mcp.engine import make_engine
from echo_wm_training.ambiguous_lvedv_exams import run_ambiguous_lvedv_audit

study = load_study()
eng = make_engine("bigquery", study)
private_detail, topology, group_statuses = run_ambiguous_lvedv_audit(eng, study)

print("Ambiguous patient-days investigated:", len(private_detail))
print("Mutually exclusive examination-topology categories:")
print(topology.to_string(index=False))
print("ACC/ORD identifier group statuses (NOT patient counts):")
print(group_statuses.to_string(index=False))
print("READ-ONLY: no cohort modifications")
```

`private_detail` contains protected identifiers; keep it within MCP.
Any smaller aggregate cell may also be disclosive and must be
suppressed per MCP policy before external distribution. **No provisional
pair establishes one clinical examination**, and no multiple-exam case
should be promoted automatically. Compare result-count categories
before the clinical review report; no 90% threshold change.

### Clinical review handoff after LVEDV feasibility audit (2026-11-18)

The aggregate findings across the source, cohort, mapping-gap, and
22-case examination-topology audits are consolidated in
`docs/ECHO_EF_CLINICAL_REVIEW.md` under *LVEDV structured-availability
assessment*. The report records 1,500 saved cohort index-days,
951 provisional mapped candidate values, 18 additional Mayo-label
potential values (969 total, not yet clinically approved), and the
381-patient-day shortfall against the unchanged 90% target. Of the
41 source-present/noncandidate patient-days, 23 require examination
identity review; none is automatically approved.

**Before another cohort build or a rule change:** obtain documented
clinical/study-owner decisions on absolute LVEDV result names and units,
ACC/ORD exam identity evidence, index-day alignment, whether other
validated data sources may be used, and whether 90% patient-level
availability is feasible under the approved study denominator.
This is a review checkpoint, not a reason to relax thresholds or select
high-completeness test types retrospectively.

The clinical-review Markdown contains **aggregates and analytical
interpretation only**, not patient IDs or DICOM/clinical rows. Its
source-record rates are not examination rates; its technical
provisional coverage is not clinically validated completeness.
Keep any patient/exam-level adjudication and small-cell findings
inside MCP. The latest audit did not write or rebuild any BigQuery
table.

### Read-only pre-PR echo source discovery on the saved cohort

Before changing `config/cohorts/echo.yaml`, verify actual source-result
names, schema fields, and report coverage. This **does not rebuild the
cohort or extract raw patient/note/result rows**. Unlike a source-wide
result-name inventory, the result-type counts in this discovery are
restricted to the saved cohort's **same-day conventional TTE records**.

After merging the fork's `validation/echo-wm-mcp` changes into MCP's
corporate source repository and pulling the result:

```bash
python -m pytest tests/test_echo_cohort_source_discovery.py -q
```

Run in fresh IPython inside the MCP Cloud PC:

```python
from cardioai_mcp.config import load_study
from cardioai_mcp.engine import make_engine
from echo_wm_training.cohort_source_discovery import run_echo_source_discovery

study = load_study()
eng = make_engine("bigquery", study)
result_types, source_columns, note_titles, report_coverage = (
    run_echo_source_discovery(eng, study)
)
print("\\nRESULT TYPES on cohort index-day conventional TTEs (count-gated):")
print(result_types.to_string(index=False))
print("\\nMAYO ECHO source schema (metadata only):")
print(source_columns.to_string(index=False))
print("\\nOMOP echo-like NOTE titles (source-wide, count-gated):")
print(note_titles.to_string(index=False))
print("\\nCOHORT ECHO_NOTE and existing report-coverage flags:")
print(report_coverage.to_string(index=False))
```

**Interpretation:** Result-type rows are aggregated by result name and
description, with counts of result rows, source tests and distinct
patient/index-days; the current screen excludes types below the configured
minimum count. Source tests can be multiple representations of the same
clinical exam. ECHO_NOTE keyword flags indicate *only text availability /
candidate phrases*, **not clinically valid wall-motion, image-quality or
indication labels**. Echo test table height/weight columns, if present,
still require unit and provenance checks; do not silently extend OMOP
lookback windows.

`source_columns` and the count-gated tables contain no clinical values,
but the metadata, small cells and combinations still require the site's
disclosure rules before screenshots are sent outside MCP. Do not export
unredacted `ECHO_NOTE`, `FACT_ECHO_RESULTS` values, patient identifiers
or accessions. Do not make new mappings, change completeness goals,
or rebuild `echo_analytic` until the source names and clinical identity
requirements have been reviewed.

### Findings and read-only follow-up after 2026-11-18 echo-source discovery

MCP output from the first discovery:

- Saved `echo_analytic` denominator: **1,500**; same-day conventional
  TTE source present: **1,477** (98.47%).
- Current extracted `echo_report` nonblank: **282** (18.80%).
- Same-day `FACT_ECHO_TEST.ECHO_NOTE` nonblank: **592** (39.47%);
  source-note cue counts: wall motion **440**, quality **55**,
  indication **0**. **Cues are not valid labels**; note overlap with
  existing reports and OMOP notes has not been measured.
- Observed structured source test-result names include LVEDV
  `LV_vol_4C_ed_2D_by_MOD` (942 patient-days with a result row),
  LVESV `LV_vol_4C_es_2D_by_MOD` (936), and LA-dimension
  `LA_dim_s_MM` (26) and `LA_dim_es_2D` (13).
  **A result row is not proof of a numeric/valid, exam-matched value.**
  LA *volumes* are common but are not substitutes for LA *diameter*.
- `FACT_ECHO_TEST` has `ECHO_NOTE`, but **no direct height/weight
  columns** among its current schema fields. `Body_Surface_area`
  appears among result types but cannot independently fill height and
  weight.
- The initial source-wide OMOP title regex `echo|tte|...` also matched
  unrelated *Letter* titles (the substring `tte`). This has been
  corrected to a word-bounded `tte` search.

To establish numeric feasibility and candidate report coverage without
writing a cohort, use the second-pass audit:

```bash
python -m pytest tests/test_echo_cohort_source_followup.py tests/test_echo_cohort_source_discovery.py -q
```

Fresh IPython inside MCP:

```python
from cardioai_mcp.config import load_study
from cardioai_mcp.engine import make_engine
from echo_wm_training.cohort_source_followup import run_echo_source_followup

study = load_study()
eng = make_engine("bigquery", study)
numeric_counts, la_diameter_units, note_coverage = run_echo_source_followup(eng, study)

print("Numeric result-type coverage (patient-days; read-only):")
print(numeric_counts.to_string(index=False))
print("LA diameter source unit names (count-gated):")
print(la_diameter_units.to_string(index=False))
print("Existing vs source/OMOP echo-note candidate availability:")
print(note_coverage.to_string(index=False))
```

The new note audit checks a *curated* list of observed OMOP echo report
titles, linked by patient and index date or +1 day, and compares their
availability with the current extracted report and same-day
`FACT_ECHO_TEST.ECHO_NOTE`. This join is **provisional patient/date
linkage**, not an approved exam-specific report choice. Numeric counts
check parseability; screening ranges for LVEDV and LVESV are provisional.
LA diameter units must be confirmed before any mm/cm conversion.
These aggregate screens do **not** select clinical reference measurements.

The displayed output applies conservative primary, complementary and
within-row difference suppression. MCP's cumulative/disclosure policy
still applies across reports; do not export patient rows, note text,
result values, accession keys or unsuppressed small cells.

Next PR only after discovery and clinical review: choose source columns
and matching policies, preserve ambiguity checks, test against synthetic
records, then compare a rebuilt cohort to the **unchanged saved baseline**.
The separate wall-motion endpoint reference is `ref_wma` (from
`echo_wm_reference`), whereas deck element `wall_motion` derives
from `echo_report`; treat them as different fields.

### Staged, read-only LVEDV/LVESV and ECHO_NOTE extraction preview

Following Sampson's 2026-11-18 review, stage the first two extraction
improvements **without changing the saved cohort, thresholds, or clinical
reference**. The original MCP corporate repository target branch is
`echo-wm-mcp`. Our changes are prepared on GitHub fork
`galaxog/cardioai_mcp_validation` branch `validation/echo-wm-mcp`;
merging fork changes into the corporate source repository is a separate
step from running `git pull --ff-only` within MCP.

`echo_wm_training/cohort_extraction_preview.py` uses the existing
`resolve_tte_groups` identifier/mutual-match logic. It conservatively
reports `multiple_exams`, `unresolved_identifier` and
`provisional_pair_requires_review`, and does not promote the associated
measurements. It also withholds discordant numeric volume values, unknown
units, conflicting test reports, and absent source texts. It accepts a
staged *single unpaired identifier group* and **only** consistent
absolute-mL LVEDV/LVESV measurements in preliminary numeric ranges.

It separately selects `FACT_ECHO_TEST.ECHO_NOTE` only from that same
single group. It does **not** yet apply report NLP to produce clinically
validated wall-motion, image-quality or indication labels, and it does
not use OMOP patient/date note matches lacking exam-level linkage.
Those require clinical/source review. This first preview may be
intentionally more conservative than the earlier 969 potential-LVEDV
diagnostic, which used different provisional status assumptions.

After merging and pulling code into MCP:

```bash
python -m pytest tests/test_echo_cohort_extraction_preview.py -q
```

In a **fresh IPython session in MCP**:

```python
from cardioai_mcp.config import load_study
from cardioai_mcp.engine import make_engine
from echo_wm_training.cohort_extraction_preview import run_echo_extraction_preview

study = load_study()
eng = make_engine("bigquery", study)
private_detail, completeness_preview, exam_status_summary = (
    run_echo_extraction_preview(eng, study)
)
print("EXAM SELECTION STATUS (disclosure-guarded):")
print(exam_status_summary.to_string(index=False))
print("ALL 27 COHORT ELEMENTS — CURRENT VS CONSERVATIVE VOLUME PREVIEW:")
print(completeness_preview.to_string(index=False))
print("No cohort, reference, or acceptance-criteria change was made")
```

**Never print or export `private_detail`**: it contains patient IDs
and potentially raw `ECHO_NOTE` text. The exposed comparison and status
reports apply count, complement and small-change suppression. Still
check MCP's cross-report disclosure policies before sharing.
Only LVEDV/LVESV can change in the initial 27-element comparison;
other elements remain at their *existing* observed completeness.
Do not interpret unchanged `wall_motion` as a test of new report NLP.

When the preview and unit/name/examination policy have been reviewed,
introduce a dedicated Mayo echo result extractor in
`cardioai_mcp/extractors.py`, configured from
`config/cohorts/echo.yaml` / `config/modality_tables.yaml`. Preserve
source provenance and the conservative resolution gate. Integrate
the selected exam's report with a separately validated text parsing
policy. Then compare the locked before/after 27-field reports after
an explicitly approved rebuild. Do **not** enable the generic placeholder
`echo.study_table` source or loosen matching to increase coverage.

### Diagnosing unexpectedly low strict extraction preview (2026-11-18)

The MCP conservative preview executed against the unchanged 1,500-patient
cohort. Its selected `single_identifier_group` population is
**703** patient-days; **797** were withheld for additional examination
identity review. This is a *stricter* gate than the earlier provisional
LVEDV diagnostic; the outcomes must not be compared as if they used the
same selection criteria. The 27-element completeness preview showed
LVESV improving from 0.00% to **4.07%** (still below 90%).
The exact LVEDV before/after was **suppressed** because the saved
baseline has a complementary small cell. No model reference or cohort
table was changed. Other 25 fields remain unchanged because report
NLP has not been applied in the preview.

The strict volume selector presently demands every in-range measurement
across *all* methods in the selected identifier group have compatible
explicit mL units and the **exact same numeric value**. Biplane and
single-plane volume estimates may reasonably differ. This creates a
potentially over-restrictive preview condition, not evidence that the
source contains little usable LVESV. Do not loosen it or choose a method
solely to improve coverage; get the rejection-reason breakdown and
confirm a clinical measurement hierarchy first.

The new aggregate-only audit characterizes:
- selected single identifier vs provisional ACC/ORD pairing vs other
  examination/identifier review;
- among selected patient-days, consistent absolute-mL volume, no
  recognized/usable numeric volume, and values held for method or
  unit review.

```bash
python -m pytest tests/test_echo_cohort_preview_diagnostics.py -q
```

Fresh IPython inside MCP:

```python
from cardioai_mcp.config import load_study
from cardioai_mcp.engine import make_engine
from echo_wm_training.cohort_preview_diagnostics import run_preview_barrier_audit

study = load_study()
eng = make_engine("bigquery", study)
private_detail, exam_barriers, volume_barriers = run_preview_barrier_audit(eng, study)

print("EXAMINATION SELECTION BARRIERS")
print(exam_barriers.to_string(index=False))
print("VOLUME REASONS WITHIN SINGLE IDENTIFIER GROUPS")
print(volume_barriers.to_string(index=False))
```

**Keep `private_detail` inside MCP**, where it includes patient keys,
numeric values, and potentially full echo report text. The detailed
within-MCP helper `within_mcp_reason_counts(private_detail)` returns
PRIVATE status breakdowns that are **not approved to export or screenshot**
without disclosure review. The two printed aggregate tables redact
all counts in a category group if any positive cell is below the
small-cell threshold, protecting against subtraction from the known
denominator. Cross-report disclosure checks may require further redaction.

Next step: if method/unit review dominates, distinguish within-method
duplicate conflicts from legitimate differences across biplane/3D/
single-plane measurements in a separate *diagnostic*; seek clinical
approval for a priority policy before altering production extraction.
If provisional ACC/ORD pairing dominates, establish authoritative exam
identity with Mayo, rather than automatically accepting timestamp pairs.

### Unit metadata vs method differences in 454/448 held volumes (read-only)

The most recent MCP preview-barrier audit reconciled all 1,500 saved
patient/index-days: 703 selected single-identifier groups, 718
provisional ACC/ORD pairings withheld, and 79 other identifier/exam
review cases. Among the 703, LVEDV had 58 accepted by the deliberately
strict preview, 191 without recognized usable results, and **454 held
for unit/method review**; LVESV respectively 61, 194, **448**.

This does **not** demonstrate a 4% source-availability ceiling. The
strict preview requires every plausible in-range value from biplane,
3D, and single-plane methods to match exactly, and disallows unknown
or non-mL units. It can reject clinically plausible differing methods.

To distinguish these technical exclusion causes without altering
selection, `echo_wm_training/volume_hold_detail.py` produces **one
mutually exclusive primary reason** per held metric in a selected
identifier group:

- `unit_metadata_only`: one or more in-range values has absent or
  non-mL unit metadata, but numeric values otherwise agree
- `different_methods_only`: all in-range values carry mL units;
  values differ **between** result names but not within each name
- `same_method_conflict`: mL values disagree **within** a result name,
  regardless of whether other methods also differ
- `unit_and_value_disagreement`: both unknown/non-mL unit metadata
  and differing values occur

```bash
python -m pytest tests/test_echo_volume_hold_detail.py -q
```

Fresh IPython inside MCP:

```python
from cardioai_mcp.config import load_study
from cardioai_mcp.engine import make_engine
from echo_wm_training.volume_hold_detail import run_volume_hold_audit

study = load_study()
eng = make_engine("bigquery", study)

hold_causes = run_volume_hold_audit(eng, study)
print("LVEDV/LVESV HOLD REASONS WITHIN SELECTED EXAMS")
print(hold_causes.to_string(index=False))
```

All source values, identifiers and notes stay in MCP; this function
returns only reason/count aggregates with whole-metric suppression if
any category has a positive count below the configured threshold.
Apply platform cross-report disclosure rules before sharing results.

**No value-selection change occurs.** In particular, never assume a
blank unit is mL merely because the result name looks like a volume.
Follow up with Mayo/developer/clinical reviewer for units, trusted
method selection and authoritative examination identity.

### Identify unit-metadata omissions by exact volume result name

The 2026-11-18 MCP volume-hold audit found, **within the 703
single-identifier patient-days**, LVEDV 454 held for unit/method
review: 24 `unit_metadata_only`, 11 `different_methods_only`,
zero isolated `same_method_conflict`, and 419
`unit_and_value_disagreement`. Thus **443/454** of the held LVEDV
patient-days have at least one in-range measurement whose
`ECHO_RESULTS_VALUE_UNIT_OF_MEASURE` is absent or not recognized as
mL. This does *not* establish that all 443 have blank units (some
could have a different unit), nor whether those source values are
clinically valid. The four LVESV categories were suppressed by the
disclosure guard; no numerical breakdown is available for LVESV.

The diagnostic classifies an exact result name's **numeric, in-range
source rows** within the selected 703 patient-days, grouping by
patient/index-day/name so duplicate result records cannot inflate the
patient-day counts. It separates `explicit_ml_only`,
`missing_only`, `non_ml_only` and `mixed_unit_metadata`.
Result-name populations can overlap; never sum them to determine
unique patient-day coverage. The unit categories must not be
interpreted as a clinically approved conversion or inferred unit.

After merging changes into original MCP source branch `echo-wm-mcp`:

```bash
python -m pytest tests/test_echo_volume_unit_profile.py -q
```

Fresh IPython inside the restricted MCP Cloud PC:

```python
from cardioai_mcp.config import load_study
from cardioai_mcp.engine import make_engine
from echo_wm_training.volume_unit_profile import run_volume_unit_profile

study = load_study()
eng = make_engine("bigquery", study)
unit_profile = run_volume_unit_profile(eng, study)
print("VOLUME UNIT METADATA BY EXACT MAYO RESULT NAME")
print(unit_profile.to_string(index=False))
```

The output is aggregate only and suppresses *all* counts within each
result-name group if any category has a positive cell below the
configured small-cell minimum. As with prior audit outputs,
cross-report disclosures must be reviewed before sharing outside
MCP. Do not print/export the internal source rows or numeric values.

**Next implementation gate:** confirm with Mayo's data dictionary/
Adam which exact result names always represent absolute mL
(as opposed to indexed, method-specific or converted volumes),
whether the unit field is intentionally unpopulated for those
names, and whether other trustworthy metadata can certify units.
Ask the model developer/clinical lead for the approved volume method
hierarchy. Do not silently fill blank units or collapse
ACC/ORD pairs based solely on timestamps.

### Explicit-mL-only feasibility after Mayo result-name unit audit (2026-11-18)

The prior MCP source-unit audit found method-specific, explicitly
labeled mL LVEDV in source result names
`LV_vol_BP_ed_2D_by_MOD` (431 patient-days),
`LV_vol_4C_ed_2D_by_MOD` (476),
`LV_vol_2C_ed_2D_by_MOD` (432),
and `LV_vol_ed_3D` (49), all within the *703 selected
single-identifier patient-days*. LVESV counterpart counts were
426, 474, 427 and 48, all explicitly labeled mL.

Mayo-specific `MC CV LV END DIASTOLIC VOLUME` (443 patient-days)
and `MC CV LV END SYSTOLIC VOLUME` (439) had **entirely missing
unit metadata** on numeric, in-range source rows. These result-name
counts overlap across patient-days. Do not assume a unit for the
Mayo-specific source names until Mayo confirms their definition.

An additional no-write study
`echo_wm_training/explicit_ml_feasibility.py` quantifies how many of
the 703 strictly selected single-identifier patient-days contain
**at least one numeric, explicitly mL-labeled LVEDV or LVESV**
without allowing the unitless `MC CV` values. It also reports an
illustrative source-method ranking:
biplane > 3D > single-plane (2C and 4C are the same tier). It
does **not** require biplane results to equal single-plane results,
does **not** select between discordant 2C/4C values, and **does
not** change the clinical cohort.

The ranking is an **unapproved sensitivity scenario only**,
not a new clinical hierarchy or operational extractor.
Patient-level values stay inside MCP.

After merging the fork's validation branch into the **original
repository branch `echo-wm-mcp`** and pulling:

```bash
python -m pytest tests/test_echo_explicit_ml_feasibility.py -q
```

Fresh IPython:

```python
from cardioai_mcp.config import load_study
from cardioai_mcp.engine import make_engine
from echo_wm_training.explicit_ml_feasibility import run_explicit_ml_feasibility

study = load_study()
eng = make_engine("bigquery", study)
presence, priority_scenario = run_explicit_ml_feasibility(eng, study)

print("EXPLICIT-mL AVAILABILITY IN SELECTED EXAM GROUPS")
print(presence.to_string(index=False))
print("UNAPPROVED METHOD-PRIORITY SENSITIVITY")
print(priority_scenario.to_string(index=False))
```

Counts are grouped per patient/index-day and metric, with primary and
complementary small-cell safeguards. The second output is wholly
suppressed by metric when a positive method tier has fewer than
`study.small_cell` members. Review aggregate results against MCP's
cumulative disclosure rules before sharing screenshots or numbers.

**Important denominator:** The report is about the 703 conservative
single-identifier cases, *not* all 1,500 patient-days. The separate
718 provisional ACC/ORD pairings and 79 other cases remain outside
this sensitivity scenario. Any eventual extraction rules must also
address those examination-identity categories with clinical/Mayo
approval rather than silently promoting them.

### Outcome of explicit-mL-only feasibility and next ACC/ORD pairing screen (2026-11-18)

The new source-method sensitivity ran successfully in MCP. Among
**703** conservative `single_identifier_group` patient/index-days:

| Selected preferred source tier | LVEDV | LVESV |
|---|---:|---:|
| Biplane candidate | 431 | 426 |
| 3D candidate | 21 | 21 |
| Single-plane candidate | 42 | 44 |
| Conflicting top-tier values | 0 | 0 |
| No explicit-mL numeric volume in range | 209 | 212 |
| **At least one explicit-mL preferred-tier candidate** | **494** | **491** |

This is **70.27%/69.84% of the 703 selected** but only
**32.93%/32.73% of the full 1,500** because the staged examination
identity gate excludes 797 other index days. Neither count is an
approved clinical completeness result. The hypothetical hierarchy
(biplane > 3D > single plane) must receive clinical review before
being applied in production. The source-unit investigation also
showed the two `MC CV` volume results have empty unit metadata;
they are **not** assumed to be mL here.

The previous preview disposition was:
- `single_identifier_group`: 703 patient/index-days
- `provisional_pair_requires_review`: 718 patient/index-days
- all other exam/identifier review: 79 patient/index-days

The next **read-only** audit asks whether any of the 718 provisional
ACC/ORD patient-days have exactly **one reciprocal** ACC/ORD pair
and **no competing examination group**, and what the explicit-mL
source-data availability would be *if* that pair eventually were
adjudicated as one exam. The audit refuses a pair plus any additional
identifier group, multiple pairs, nonreciprocal pair and any conflicting
highest-tier measurement.

The source grouping is from `resolve_tte_groups` with the same
mutually unique 60-second rule already used. Being a reciprocal
60-second pair is **not** evidence of clinical identity, and **no**
pair is merged into a clinical reference or written to BigQuery.

After merging the fork branch `validation/echo-wm-mcp` into the
original MCP repo branch `echo-wm-mcp` and pulling:

```bash
python -m pytest tests/test_echo_provisional_pair_feasibility.py -q
```

Fresh IPython **inside MCP**:

```python
from cardioai_mcp.config import load_study
from cardioai_mcp.engine import make_engine
from echo_wm_training.provisional_pair_feasibility import (
    run_provisional_pair_feasibility,
)

study = load_study()
eng = make_engine("bigquery", study)
pair_topology, pair_volume_scenarios = run_provisional_pair_feasibility(
    eng, study
)
print("PROVISIONAL ACC/ORD PAIR TOPOLOGY — NO CLINICAL PROMOTION")
print(pair_topology.to_string(index=False))
print("EXPLICIT-mL CANDIDATES IN SOLE PROVISIONAL PAIRS")
print(pair_volume_scenarios.to_string(index=False))
```

The denominators differ: pair topology covers the 718 provisional
pair-withheld patient-days, while pair-volume scenarios cover *only*
the mutually unique sole pair subset. Count-gated suppression is
applied, but consult MCP cumulative-disclosure policy before
sharing aggregates. Follow up with Mayo/clinical leadership on the
identity of paired records and method priorities before any
production cohort rebuild.

### Result of the sole ACC/ORD pairing audit (2026-11-18)

The MCP run of `run_provisional_pair_feasibility` completed. Its
`pair_topology` table (sole reciprocal ACC/ORD pair versus
pair with additional identifier groups) and both LVEDV and LVESV
`pair_volume_scenarios` tables returned **suppressed** for all
category counts. This is due to the primary/complementary disclosure
rules; **suppressed is neither zero nor an estimate**. We cannot
derive a numeric count of sole pairs or additional explicit-mL
candidates from the released output. The previously established
**718** patient/index-days with any provisional pairing remain
withheld from clinical reference selection.

Do **not** weaken suppression or reconstruct a small complementary
cell using the known 718 denominator. Authorized examination-level
review and disclosure decisions must take place within MCP.
If an aggregate will be sent to Mayo or the model developer,
obtain platform-approved disclosure clearance rather than
using a more revealing query.

**Clinical/technical sign-off needed before extraction changes:**
1. Authoritative way to link ACC and ORD records to one clinical
   examination (identifier crosswalk, order/accession linkage,
   report provenance); timestamp proximity alone is not sufficient.
2. Approved method hierarchy for absolute LVEDV/LVESV: the
   biplane > 3D > single-plane ordering was a *hypothetical
   feasibility scenario*, not clinical policy.
3. Whether the unitless `MC CV LV END DIASTOLIC VOLUME` and
   `MC CV LV END SYSTOLIC VOLUME` source results are documented
   as absolute mL. Until answered, do not infer units or use
   them for automatic selection.
4. Exam/report selection and handling of multiple exams, conflicting
   within-method values, source-system differences and missing
   index-day echo notes.

Keep `echo_analytic` unchanged, retain the 90% individual
volume-completeness requirement, and treat the existing
27-element before/after report as a dry-run comparison.


### Assess whether unitless Mayo MC CV volumes use the same scale as explicit mL (2026-11-19)

Following Mayo's reply, a source-name with blank
`ECHO_RESULTS_VALUE_UNIT_OF_MEASURE` **does not necessarily imply mL**,
even where its numeric magnitude looks physiologically reasonable. Mayo
recommended broader result-description searches in Term Finder (to be
reviewed with Adam), and cautioned that source formatting is more
consistent from **2019 onward**.

A read-only audit `echo_wm_training.unitless_scale_validation` now
checks two hypotheses independently:

1. Does a unitless `MC CV LV END DIASTOLIC VOLUME` or
   `MC CV LV END SYSTOLIC VOLUME` co-occur with an **explicit-mL**
   2D biplane/4C/2C or 3D volume on the **same
   `person_id` + `index_date` + `ECHO_TEST_FPK`**?
   Never infer this link from an encounter, timestamp, accession, or
   pairing of two distinct tests.
2. Within exact-test co-occurrences, what is the **ratio of
   unitless-to-explicit-mL numeric values** by comparison method
   and 2012–2018 versus 2019–2023? A median near one and broad
   agreement would support a hypothesis of the same *scale*, but
   different clinical measurement methods need not agree exactly.
   This is not proof of units, and must not be used to autoassign mL.

Only one distinct positive numeric value per result name/test is
eligible; conflicting duplicates are excluded. Unitless MC CV rows
must have empty unit metadata and comparison rows must explicitly
say mL and pass the existing volume screening ranges. The results
contain **no raw values or identifiers**, and suppress whole comparison
rows when their counts (or 10%/25% near/far partitions) are smaller
than the required cell threshold. Review **cumulative cross-report
disclosure** before sharing.

The previous 443 LVEDV / 439 LVESV source-unit counts came from
**703 selected single-identifier examinations**. This new exact-FPK
audit samples every cohort-scoped same-day conventional TTE source
test and reports only **unambiguous numeric MC CV** groups by
period; its denominator and meaning are therefore different.
An exact-FPK comparison is strong source-record evidence but not
proof that the examination was clinically selected as the index
exam, and the 2019+ split does not change the cohort.

After integrating fork validation commits into the original
`echo-wm-mcp` branch:

~~~bash
python -m pytest tests/test_echo_unitless_scale_validation.py -q
~~~

Fresh IPython **inside MCP**:

~~~python
from cardioai_mcp.config import load_study
from cardioai_mcp.engine import make_engine
from echo_wm_training.unitless_scale_validation import (
    run_unitless_scale_validation,
)

study = load_study()
eng = make_engine("bigquery", study)

unitless_coverage, unitless_scale = run_unitless_scale_validation(
    eng, study
)

print("UNAMBIGUOUS UNITLESS MC CV VOLUME COVERAGE BY PERIOD")
print(unitless_coverage.to_string(index=False))

print("EXACT-ECHO-TEST UNIT SCALE COMPARISONS")
print(unitless_scale.to_string(index=False))
~~~

**Pending:** these tests and diagnostic have not yet been executed
in the restricted Mayo Cloud PC. Do not modify
`echo_analytic`, the volume unit conversion policy, or the
clinical method-selection hierarchy based on a matching ratio.
Ask Adam whether the source result-type metadata/documents can
authoritatively establish the expected unit or source convention.


### Outcome: no exact ECHO_TEST_FPK unitless-to-mL matches; investigate cross-record topology (2026-11-19)

The 2026-11-19 read-only MCP run of
`run_unitless_scale_validation` had **zero co-occurring unitless MC CV
and explicitly mL-labeled results on the exact same
`ECHO_TEST_FPK`**, for both LVEDV and LVESV in both time periods.

| Source result family | 2012–2018 unambiguous unitless patient-days | 2019–2023 unambiguous unitless patient-days | Same-FPK comparator |
|---|---:|---:|---:|
| MC CV LV end-diastolic volume | 46 | 400 | 0 in both periods |
| MC CV LV end-systolic volume | 46 | 397 | 0 in both periods |

The printed ratio rows were all `suppressed` in the initial code
because the reporting helper treated an empty sample as a
small group. This was a *reporting issue*, not a data-disclosure
suppression of actual ratios. The helper now shows
`patient_days=0` and `not_applicable` quantiles/percentages
for a genuinely empty comparison group. Ratios cannot establish
mL units because no exact-test comparisons exist.

The next read-only module `echo_wm_training/volume_record_topology.py`
checks how often a unitless result and an explicitly mL-labeled
result for the same metric occur on **different
`ECHO_TEST_FPK`s on the same cohort/index day**. It separately
flags a **provisional reciprocal ACC/ORD group match** between
those source tests. It deliberately does **not** compare values
across those records or imply those tests are one clinical
examination. Its patient-day outputs apply joint
primary/complementary suppression across both tables; when a
small cell can be inferred from their difference, all cells for
that metric/period are suppressed. Even these counts need
MCP cumulative-release review before sharing.

After merging the fork validation branch into the original
`echo-wm-mcp` branch, run synthetic tests **inside MCP**:

~~~bash
python -m pytest -q \
  tests/test_echo_unitless_scale_validation.py \
  tests/test_echo_volume_record_topology.py
~~~

Fresh IPython in MCP:

~~~python
from cardioai_mcp.config import load_study
from cardioai_mcp.engine import make_engine
from echo_wm_training.volume_record_topology import (
    run_volume_record_topology,
)

study = load_study()
eng = make_engine("bigquery", study)

cross_record_overlap, provisional_pair_overlap = (
    run_volume_record_topology(eng, study)
)

print("UNITLESS MC CV VS EXPLICIT-mL ON DIFFERENT SOURCE TESTS")
print(cross_record_overlap.to_string(index=False))

print("RECIPROCAL ACC/ORD CANDIDATES — NOT EXAM IDENTITY")
print(provisional_pair_overlap.to_string(index=False))
~~~

This new script has not yet been run on MCP patient data.
The source records and their numeric values remain inside MCP.
Do not send patient-level output, use results to assign the
unit of MC CV values, change the approved cohort selection,
or collapse ACC/ORD records based only on time proximity.

**For Adam:** Ask where `ENCOUNTER_NUMBER` is stored, and
whether any source accession↔order crosswalk / report provenance
can prove two echo test records correspond to the *same exam*.
A common encounter can include multiple independent exams and
does not by itself validate record identity.


### Outcome of cross-record Mayo MC CV versus explicitly-mL test topology (2026-11-19)

The read-only `run_volume_record_topology` MCP audit completed. For
**2019–2023**, among the saved cohort's conventional same-day TTE
source-test patient-days:

| Metric | Unitless MC CV result patient-days | Explicit-mL volume on a **different** ECHO_TEST_FPK on same patient/index-day | No different-FPK explicit-mL source |
|---|---:|---:|---:|
| LVEDV | 400 | 385 | 15 |
| LVESV | 397 | 382 | 15 |

For both metrics, **0** of these 2019–2023 patient-days had an
explicit-mL source record linked to the unitless source record
by our *provisional reciprocal ACC/ORD matching algorithm*.
Pre-2019 detailed cells are suppressed and cannot be inferred.

Thus the majority of the newer unitless MC CV patient-days have
a separate source-test record with a named explicit-mL volume, but
**that is NOT evidence they are the same clinical echo examination**.
The different-FPK co-occurrence alone cannot verify the numeric
unit of the unitless values, and the absence of a 60-second reciprocal
pair does not prove they are separate exams. Do not compare or use
their numerical measurements as a pair until provenance is established.

The previous `run_unitless_scale_validation` displayed ratio
statistics as suppressed when its true exact-test pair count was
**zero**; it now returns `patient_days=0` and
`not_applicable` ratio statistics for empty comparisons. The new
presentation fix has not yet been rerun in MCP.

#### Identify authoritative encounter and source-exam linkage columns

Mayo advised checking `ENCOUNTER_NUMBER` and using its native FACT/DIM
sources. Before guessing the location of that field or building an
encounter join, query the published **metadata only**. This returns
table names, column names and data types from the Mayo source schema,
including echo, encounter, clinical documents and flowsheets.
It reads no patient data and does not modify any cohort data.

After merging the fork's validation branch into the original
`echo-wm-mcp` branch, run:

~~~bash
python -m pytest -q \
  tests/test_echo_unitless_scale_validation.py \
  tests/test_echo_volume_record_topology.py \
  tests/test_echo_encounter_schema_discovery.py
~~~

Fresh IPython, inside MCP:

~~~python
from cardioai_mcp.config import load_study
from cardioai_mcp.engine import make_engine
from echo_wm_training.encounter_schema_discovery import (
    run_encounter_schema_discovery,
)

study = load_study()
eng = make_engine("bigquery", study)

source_keys = run_encounter_schema_discovery(eng, study)
print("MAYO FACT/DIM ENCOUNTER AND SOURCE LINKAGE COLUMNS")
print(source_keys.to_string(index=False))
~~~

The metadata query uses native BigQuery
`INFORMATION_SCHEMA.COLUMNS` rather than the Spark-to-GoogleSQL
transpiler. If metadata permissions or object naming differ in
a workspace, ask Adam to use the Term Finder and Schema Visualizer
rather than guessing from error messages. This table is metadata,
not clinical results, but still follow platform sharing restrictions.

**Next question:** Which source table and column actually supplies
`ENCOUNTER_NUMBER` to the echo tests, and does joining on encounter
plus reliable exam-specific identifiers yield a one-to-one echo
study mapping? An encounter may contain multiple echoes, so
same-encounter agreement alone does NOT establish the same exam.
Do not use patient/date or encounter-only joins to populate clinical
reference measurements or clinical reports without adjudication.
