# Mayo Clinic Platform support: reproduce echo cohort source-mapping gaps

**Project:** CardioAI Phase 2 echo external validation  
**Authoritative repository/branch:** original MCP repository, `echo-wm-mcp`  
**Data scope:** existing 1,500 patient/index-day `echo_analytic` cohort  
**Purpose:** Read-only reproduction for Mayo schema/ETL and clinical-source support  
**Status:** No cohort rebuild, production extractor update, reference promotion, or threshold change has been approved.

## 1. Handoff contents and environment

Use the code and study configuration at the recorded commit of the **original**
`echo-wm-mcp` branch. The files in this handoff are a **subset of the
repository**. They rely on the original package-relative layout (and the
supporting import modules in the archive). Run from the repository root
inside an authorized Mayo Clinic Platform (MCP) Cloud PC.

**IMPORTANT — requirements file provenance:** The working Cloud PC has a
**locally modified** `requirements.txt`, which differs from the file committed
in the GitHub fork. The **Cloud PC version actually used to execute the
diagnostics** must accompany the support archive as
`requirements-cloudpc.txt`. Do not substitute the older committed file or
claim it reproduces the active environment. The customer should inspect the
local dependency file and confirm what is actually installed. If useful,
provide a separately reviewed `pip list --format=freeze` output, which is
*environment evidence* rather than the minimum supported dependency set.

The local requirements file and software-inventory output should be
screened for internal package URLs, tokens, paths, private package
names or other material not approved for sharing.

Record the following with the ticket:

- Commit SHA of original-repository `echo-wm-mcp` used for the archive
- Python version, exact **Cloud PC** `requirements.txt` version
- Dataset delivery/version and whether `echo_analytic` is the existing
  saved cohort
- Date the read-only queries were run and any environment-specific failures

MCP BigQuery access is needed: code by itself does not contain source data.
The managed environment must have `google-cloud-bigquery` and
`db-dtypes` installed (they may be provided by MCP even when not listed
as active requirements). The optional BigQuery Storage module warning
does not invalidate the successful REST-based queries.

## 2. Current configuration and failing fields

The cohort specification is `config/cohorts/echo.yaml`; concept sets
in `config/concept_sets.yaml` and Mayo table mapping in
`config/modality_tables.yaml`. The extraction implementation is
`cardioai_mcp/extractors.py`, assembled by
`cardioai_mcp/builder.py`.

| Required field | Current extraction in saved cohort | Observed completeness | Target |
|---|---|---:|---:|
| `lv_end_diastolic_volume` | OMOP `measurement`: `lv_edv`, day 0/+1 | approx. 0.7% | 90% |
| `lv_end_systolic_volume` | OMOP `measurement`: `lv_esv`, day 0/+1 | 0% | 90% |
| `la_diameter` | OMOP `measurement`: `la_diam`, day 0/+1 | 0% | 85% |
| `wall_motion` | Regex parsed from auxiliary `echo_report` | 12.8% | 95% |
| `image_quality` | Echo modality image-quality field with report-text fill | 18.8% | 95% |
| `clinical_indication` | Report-text indication regex, fallback to index visit-concept classification | 45.67% | 90% |
| `height` | OMOP `measurement`: `height`, −1095/+30 days | 93.87% | 95% |
| `weight` | OMOP `measurement`: `weight`, −90/+30 days | 93.60% | 95% |

The original required-field average was approximately **71.3%**
versus a 95% aggregate goal, with **eight required fields** below
their individual targets. This table reports the **current saved
cohort**, not projected improvements.

The configured generic `echo` modality study table is a **disabled
placeholder**, not a verified Mayo source. The saved report fallback
uses OMOP `note`; the Mayo source also has `FACT_ECHO_TEST.ECHO_NOTE`.
Do not enable a placeholder or assume that a patient/date note joins
to the specific index examination.

The cohort's report-derived `wall_motion` and the model's
separately built `ref_wma` are different reference-related fields;
report keyword matches are not clinical ground-truth adjudications.

## 3. Authorized, read-only reproduction

All commands below run inside the restricted MCP Cloud PC, with the
existing `config/study.yaml` and `echo_analytic`. **Do not run
`scripts/run_cohorts.py` or any builder that persists/replaces the
cohort merely to reproduce this issue.** Do not export patient-level
data, note bodies, measurements, examination keys or DICOM metadata.

### A. Verify environment and synthetic diagnostics

~~~bash
python --version
python -m pip show pyyaml pandas sqlglot google-cloud-bigquery db-dtypes

python -m pytest -q \
  tests/test_echo_cohort_source_discovery.py \
  tests/test_echo_cohort_source_followup.py \
  tests/test_echo_cohort_extraction_preview.py \
  tests/test_echo_volume_unit_profile.py \
  tests/test_echo_explicit_ml_feasibility.py \
  tests/test_echo_provisional_pair_feasibility.py
~~~

These tests use synthetic data; they verify query generation, unit
screening, examination safeguards and disclosure handling, not the
clinical validity of the selected measurement methods.

### B. Discover actual Mayo results, table columns and echo reports

~~~python
from cardioai_mcp.config import load_study
from cardioai_mcp.engine import make_engine
from echo_wm_training.cohort_source_discovery import run_echo_source_discovery
from echo_wm_training.cohort_source_followup import run_echo_source_followup

study = load_study()
eng = make_engine("bigquery", study)

result_types, source_columns, note_titles, source_notes = (
    run_echo_source_discovery(eng, study)
)
numeric_counts, la_units, combined_notes = (
    run_echo_source_followup(eng, study)
)

print("COHORT-INDEX-DAY STRUCTURED RESULT TYPES")
print(result_types.to_string(index=False))
print("MAYO ECHO SOURCE TABLE SCHEMA")
print(source_columns.to_string(index=False))
print("SOURCE-WIDE OMOP ECHO REPORT TITLES")
print(note_titles.to_string(index=False))
print("COHORT ECHO_NOTE AND EXISTING REPORT COVERAGE")
print(source_notes.to_string(index=False))
print("COHORT NUMERIC RESULT AVAILABILITY")
print(numeric_counts.to_string(index=False))
print("LA DIAMETER RESULT UNITS")
print(la_units.to_string(index=False))
print("COMBINED CANDIDATE NOTE COVERAGE")
print(combined_notes.to_string(index=False))
~~~

Observed November 2026 results: conventional same-day TTE records
present for **1,477/1,500** patient-days; existing extracted
`echo_report` **282/1,500**; nonblank Mayo `ECHO_NOTE` **592/1,500**;
candidate same/next-day OMOP echo notes **429/1,500**; union of any
report candidates **980/1,500 (65.33%)**. Candidate sources can overlap;
**980 is the union, not their sum**. Report presence does not
establish that a valid wall-motion, quality or indication label
is extractable. Selected Mayo/OMOP note wall-motion phrase cues
were **754/1,500**, not validated labels.

Observed numerical LA diameter from the two identified result
names `LA_dim_s_MM` and `LA_dim_es_2D` was **39/1,500 (2.60%)**,
with source unit **mm**. LA *volume* is more common but must not
be substituted for LA *diameter* without study/clinical approval.
The inspected `FACT_ECHO_TEST` schema does not provide separate
height or weight columns.

### C. Reproduce conservative extraction and all-27-field comparison

~~~python
from cardioai_mcp.config import load_study
from cardioai_mcp.engine import make_engine
from echo_wm_training.cohort_extraction_preview import run_echo_extraction_preview

study = load_study()
eng = make_engine("bigquery", study)

private_detail, comparison, exam_status = (
    run_echo_extraction_preview(eng, study)
)
# DO NOT print, save, screenshot or transmit private_detail.
print("EXAMINATION SELECTION — DISCLOSURE-GUARDED")
print(exam_status.to_string(index=False))
print("ALL 27 ELEMENTS: CURRENT VS STRICT VOLUME-ONLY PREVIEW")
print(comparison.to_string(index=False))
~~~

Observed: **703** patient-days selected with a single identifier
group, **797** withheld for exam/identifier review. The deliberately
strict volume-only preview yielded **4.07%** LVESV versus 0%
in the saved cohort; LVEDV's before/after comparison is
**SUPPRESSED** by small-cell protections. The other 25 elements
are unchanged **by design**. This preview rejects provisional
ACC/ORD identity, missing units and differences across methods
and **underestimates potential source availability**.

### D. Reproduce source-unit and explicitly-labeled-mL findings

~~~python
from cardioai_mcp.config import load_study
from cardioai_mcp.engine import make_engine
from echo_wm_training.volume_unit_profile import run_volume_unit_profile
from echo_wm_training.explicit_ml_feasibility import run_explicit_ml_feasibility
from echo_wm_training.provisional_pair_feasibility import run_provisional_pair_feasibility

study = load_study()
eng = make_engine("bigquery", study)

unit_profile = run_volume_unit_profile(eng, study)
presence, method_scenario = run_explicit_ml_feasibility(eng, study)
pair_topology, pair_volumes = run_provisional_pair_feasibility(eng, study)

print("MAYO RESULT-NAME UNIT METADATA (count gated)")
print(unit_profile.to_string(index=False))
print("SELECTED EXAMS WITH EXPLICIT-mL VOLUMES")
print(presence.to_string(index=False))
print("HYPOTHETICAL METHOD PRIORITY — NOT CLINICALLY APPROVED")
print(method_scenario.to_string(index=False))
print("PROVISIONAL ACC/ORD SOLE-PAIR TOPOLOGY")
print(pair_topology.to_string(index=False))
print("PROVISIONAL-PAIR CANDIDATES — SUPPRESSION EXPECTED")
print(pair_volumes.to_string(index=False))
~~~

Observed: within the **703 selected identifier groups**, an
explicit mL-labeled, numeric LVEDV was present in **494**
patient-days, and LVESV in **491**, before clinical exam/method
sign-off. The hypothetical biplane → 3D → single-plane
method-priority sensitivity found no conflicts at the preferred
tier. **Do not treat that method priority as an approved extractor.**

Among selected exam groups, `MC CV LV END DIASTOLIC VOLUME` had
**443** patient-days with numeric results but **missing units**;
`MC CV LV END SYSTOLIC VOLUME` had **439** such patient-days,
also with missing units. Named `LV_vol_*_2D_by_MOD` and
`LV_vol_*_3D` results reported explicit **mL**. Counts by
result name **overlap** and cannot be summed.

A total of **718** patient-days had provisional ACC/ORD
pairing and another **79** had other exam/identifier review
needs. The follow-up sole-pair breakdown was **fully suppressed**
to avoid disclosing small cells. **Suppressed is not zero**;
do not reconstruct counts using the known 718 denominator.

## 4. Mayo support questions

We need Mayo's authoritative table/ETL guidance on:

1. **Exam identity:** Which stable key, crosswalk or provenance
   relationship joins ACC and ORD records, `FACT_ECHO_TEST`,
   `FACT_ECHO_RESULTS`, finalized echo reports and any DICOM
   study? Time proximity alone is not verified identity.
2. **Volume units:** Are `MC CV LV END DIASTOLIC VOLUME` and
   `MC CV LV END SYSTOLIC VOLUME` defined as *absolute mL*
   when `ECHO_RESULTS_VALUE_UNIT_OF_MEASURE` is blank?
   Are there other authoritative absolute-volume sources?
3. **LA diameter:** Other LA anteroposterior, PLAX, M-mode
   dimension/result names, units, site differences, or authoritative
   alternative table? LA indexed volume is a different measure.
4. **Wall motion:** Per-segment coded or numeric scores, regional
   assessments, and any clinically usable label sources separate
   from WMSI or report-text keywords?
5. **Report/quality/indication:** Which source contains complete
   finalized TTE reports with exam-specific join keys? Where are
   structured image-quality fields, ordered reasons/diagnoses
   and encounter links? The fallback index visit type isn't
   necessarily the actual echo indication.
6. **Height/weight:** Echo-time values available in a source
   other than the OMOP measurements?

Please return authoritative **table names, column/result-type
names, unit definitions, join relationships and relevant ETL
coverage caveats**. Clinical choices (e.g., acceptable
method priority or changing a protocol threshold) remain with
the study owner.

## 5. Privacy and stop conditions

- All runs are **read-only** against the saved cohort and source
  views. Do not execute cohort rebuilds, save clinical references,
  modify thresholds, or select uncertain values to raise coverage.
- No patient/exam-level datasets or raw report texts belong in
  ticket attachments. Neither `private_detail` nor derived
  unsuppressed internal diagnostics are permissible exports.
- The printouts use primary/complementary disclosure safeguards,
  but Mayo's **cumulative cross-report** policy still applies.
  Review every table, screenshot and attached document before
  transmitting through the approved support channel.
- The numbers above are observed in the stated workspace and
  code version, not guaranteed across future data refreshes.
  A schema failure should be reported with its non-PHI
  diagnostic error, not fixed by guessing joins or changing
  clinical thresholds.
