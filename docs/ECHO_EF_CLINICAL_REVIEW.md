# Echo EF reference selection — clinical-review evidence (provisional)

**Status:** Technical diagnostic only; clinical approval required before altering
the active reference. Initially compiled 2026-11-17; LVEDV audit appended 2026-11-18 from aggregate outputs observed inside
the Mayo Clinic Platform (MCP) Cloud PC. No patient or accession identifiers,
clinical rows, or local MCP exports are included in this document.

## Purpose and scope

Assess the suitability of the current patient/date expert-LVEF reference,
and review a proposed examination-aware conventional TTE reference. The
wall-motion model's reported EF reference is separate from the EchoNet
segmentation model's primary predicted EF. The existing saved `echo_lvef`
table and active cohort have **not** been replaced.

The read-only comparison uses index-day conventional TTE candidates. It does
not yet validate next-day eligibility, DICOM linkage, or whether provisional
ACC/ORD time matches are the same clinical examination.

## Reproducible aggregate findings (observed in MCP)

- Index-date patients with a saved baseline: **1,477**.
- Candidate statuses after corrected taxonomy: **1,132**
  `single_exam_candidate`, **302** `unresolved_method_disagreement`,
  **96** `ambiguous_multiple_exams`, **12** `unresolved_identifier`.
- Among the 302 above-threshold EF groups: **293 between-method**,
  **8 mixed-spread**, **1 within-method**. The flag is based on total
  EF spread **greater than 10 percentage points**; this is a provisional
  diagnostic threshold, not a clinically approved cutoff.
- Of **1,098** comparable values for single-exam candidates: **943**
  exactly match the historical saved LVEF; **54** differ by >0 to 1 point,
  **98** by >1 to 5, **2** by >5 to 10, **1** by >10. The largest
  absolute disagreement is **50 points**, previously traced to a 13%
  TTE Strain visual EF from a separate test record versus 63% from a
  conventional contrast TTE. The old patient/date extraction mixed those
  records.
- Method-pair analysis **restricted to the 293 between-method flagged
  candidates** found **823** pair occurrences above 10 points.
  Multiple pairs can arise from one group; pairs are not independent
  patients and must not be summed as distinct examinations.

### Priority method pairs (flagged-only subset)

| Comparison (method A − B) | Flagged pairs | Median absolute difference (pp) | Median signed (pp) | A higher | B higher |
|---|---:|---:|---:|---:|---:|
| 2C MOD − 4C MOD | 149 | 13 | −11 | 70 | 79 |
| Quinones standard − 4C MOD | 132 | 14 | −13 | 32 | 100 |
| Quinones standard − 2C MOD | 127 | 14 | −13 | 34 | 93 |
| Biplane MOD − Quinones standard | 99 | 13 | +12 | 73 | 26 |
| Quinones apical correction − 2C MOD | 66 | 13 | +11 | 37 | 29 |
| Quinones apical correction − 4C MOD | 65 | 13 | −10.5 | 32 | 33 |
| Quinones standard − unspecified EF | 54 | 13.5 | −12 | 17 | 37 |

**Selection bias:** This table only includes pairs *already differing
by more than 10 points*. Neither its signed medians nor its direction
frequencies establish population-wide measurement bias. Two method pairs
may originate from the same candidate group.

## Code corrections / safeguards implemented on validation branch

- Identify conventional TTE tests before extracting EF measurements; use
  `ECHO_TEST_FPK` membership to prevent same-date TTE Strain contamination.
- Correct Mayo `LV_EF_2C_2D_by_MOD` and `LV_EF_4C_2D_by_MOD` as
  single-plane, rather than falsely biplane; recognize `BP` as biplane.
- Separate diagnostic variants of Modified Quinones with and without
  apical correction, without changing their priority tier.
- Preserve source provenance and do not automatically merge ACC and ORD on
  timestamp proximity alone.
- Flag EF values with a full within-identifier-group spread >10 points
  instead of silently selecting a preferred method and promoting it.
- Preserve ambiguous examinations and identifiers as unresolved.

These are code-level safeguards; they do not confer clinical validation.

## Additional analysis: all available method pairs

The accompanying function
`echo_wm_training.ef_method_pair_diagnostics.summarize_all_ef_method_pairs`
compares *all* variant pairs (including differences <=10 points) among
`single_exam_candidate` and `unresolved_method_disagreement` groups.
It excludes `ambiguous_multiple_exams` and `unresolved_identifier`
from this analysis. It compares medians **within** original ACC or ORD
groups; not across tentative ACC/ORD pairings.

The output includes pair-observation counts, distinct candidate and source
group counts, proportion of pairs above 10 points, median signed difference,
median and p90 absolute difference, and direction frequencies. Results have
**not yet been run/confirmed inside MCP**. These should be reviewed before
making any claims about general method-specific bias.

## LVEDV structured-availability assessment — MCP audits, 2026-11-18

**Scope / status:** Observed aggregate outputs from read-only MCP Cloud PC
audits. The 1,500-person/index-day saved `echo_analytic` cohort and its
**90% LVEDV completeness target** (1,350 patient-days) have not changed.
The existing table had **10/1,500 (0.67%)** non-null
`lv_end_diastolic_volume` values at the time of the completeness check.
The calculations below are *potential source availability*, not validated
clinical coverage, approved extraction values, or permission to rebuild.

### Mutually exclusive cohort patient/index-day disposition

| Source and provisional matching state | Patient-days | Interpretation |
|---|---:|---|
| `provisional_candidate_lvedv` | 951 | Recognized LVEDV in an eligible provisional examination-candidate category; still unapproved clinically |
| `source_lvedv_not_candidate` | 41 | Numeric mapped absolute LVEDV present on a conventional index-day TTE record, not accepted into the provisional candidate count |
| `tte_no_mapped_lvedv` | 485 | Conventional index-day TTE source record present, but no recognized numeric mapped absolute LVEDV |
| `no_conventional_tte` | 23 | No conventional index-day TTE record under current description/date filters |
| **Cohort total** | **1,500** | Counts are mutually exclusive patient/index-days |

The 41-case root-cause audit identified **18** patient-days using the
additional Mayo result-name mapping; those are already included in the
earlier exploratory **969/1,500 (64.60%) potential** estimate (951 + 18),
so **do not double count**. The other **23** require examination-identity
review; they are *not* currently eligible. Their finer-grained
small-cell breakdown must remain in MCP under disclosure policy.

An exploratory next-day TTE audit found **12** previously uncovered
patient-days with next-day TTE source records and **5** with a numeric
next-day LVEDV, but **zero** met the audit's conservative unique-test
rule. Thus it added **zero** to the 969 provisional potential count.
A subsequent identifier-group review did not establish clinical
examination identity and did not authorize next-day inclusion.

The indexed-volume/BSA audit found **no** patient-days with a valid
indexed LVEDV and matching BSA in the same source test under its
screening rules. It therefore added **zero** potential derived values.
The indexed-volume numeric range and BSA formula need clinical/unit
approval before any use.

### Examination-type and source-wide context

Raw source-record availability is not patient-level completeness. ACC
and ORD may represent separate records for one examination; records
should not be summed as distinct clinical exams. Broad
conventional-TTE source-record availability was:

| Source | All-source mapped absolute LVEDV | Cohort index-day mapped absolute LVEDV |
|---|---:|---:|
| ACC | 711,843 / 1,715,748 (41.49%) | 937 / 2,105 (44.51%) |
| ORD | 146,437 / 320,344 (45.71%) | 483 / 785 (61.53%) |

In the cohort-specific type audit, observed structured LVEDV
availability varied by test description (patient-days may belong to
more than one test type):

| Conventional TTE description | Patient-days | With mapped LVEDV | Availability |
|---|---:|---:|---:|
| Adult TTE | 794 | 509 | 64.11% |
| (TTE) 2D ECHO DOPPLER COLOR | 424 | 268 | 63.21% |
| (TTE) 2D ECHO DOPPLER COLOR AND CONTRAST | 213 | 161 | 75.59% |
| MCF Adult TTE | 130 | 109 | 83.85% |
| MCJ Adult TTE | 103 | 53 | 51.46% |
| FSH - Adult TTE | 64 | 42 | 65.62% |
| MJH Adult TTE | 30 | 17 | 56.67% |

The larger source-wide dataset contains some examination-type
categories exceeding 90% source-record availability (for example
MCA, MCS, and MPH Adult TTE), but that **does not** establish 90%
availability for a clinically appropriate subset of this cohort.
The distribution supports a *source/workflow availability hypothesis*;
it does not prove structural impossibility of the 90% patient-level
target.

### Ambiguous examination-resolution audit

All **22** patient-days assigned `multiple_exam_candidates` were
reviewed with the existing mutually unique, 60-second ACC/ORD
provisional matching logic. The audit observed **21 unmatched ACC
groups, 21 unmatched ORD groups, and four provisional mutual-match
groups** (two potential ACC/ORD pairs). The patient-day topology
included cases with multiple unmatched identifiers and cases with
multiple remaining groups after pairing. Small patient-day subgroup
cells are omitted from this document.

**No one of these 22 cases was automatically accepted.** Proximity
of ACC/ORD timestamps does not establish a single clinical exam,
and matching groups may still leave genuinely separate examinations.

### Source and report discovery on the same saved cohort (2026-11-18)

A follow-up **read-only** MCP metadata/count discovery identified why
the currently built cohort has low report-derived completeness:

| Cohort evidence | Observed count | Interpretation |
|---|---:|---|
| Conventional index-day TTE source | 1,477 / 1,500 | Source-test presence, not validated one-exam linkage |
| Existing nonempty `echo_report` column | 282 / 1,500 (18.80%) | Current extracted report coverage |
| Nonempty conventional-TTE `FACT_ECHO_TEST.ECHO_NOTE` | 592 / 1,500 (39.47%) | Available alternative report text; overlap with existing reports unknown |
| Source `ECHO_NOTE` mentioning wall-motion keywords | 440 / 1,500 (29.33%) | Keyword cue only, **not** clinically adjudicated wall-motion labels |
| Source `ECHO_NOTE` mentioning quality keywords | 55 / 1,500 (3.67%) | Keyword cue only, not automatic adequate/poor image-quality classification |
| Source `ECHO_NOTE` mentioning indication keywords | 0 | Current exploratory phrase search detected no indication cues |

`FACT_ECHO_TEST` has `ECHO_NOTE` but **no dedicated height or
weight fields** in the inspected schema. The source result type
`Body_Surface_area` appears in **1,210** cohort patient-days with
a result row; BSA is not individually measured height or weight.

Source result-name discovery also found LVESV
`LV_vol_4C_es_2D_by_MOD` in **936** patient-days with a result row,
LVEDV `LV_vol_4C_ed_2D_by_MOD` in **942**, and candidate LA
diameter result names `LA_dim_s_MM` in **26** and
`LA_dim_es_2D` in **13**. Rows have not yet been screened here
for numeric values, units or unambiguous exam identity. Left-atrial
**volumes** occur much more often, but cannot automatically substitute
for a requested **diameter**.

The source-wide OMOP note-title screen found `Echo Report Text`,
`ECHOCARDIOLOGY IMAGE EXAM`, `Echocardiogram Report`, etc.; title
totals were **not** cohort-specific. The initial broad regex also
matched unrelated *Letter* titles through substring `tte`, which
was corrected. A cohort-scoped OMOP note overlap and numeric-source
audit has now run inside MCP. See the 2026-11-18 results immediately
below.

**Do not conflate** the deck's `wall_motion` field (derived from
`echo_report`) with the model-performance reference
`ref_wma` (from the separate `echo_wm_reference` table). Improving
report extraction does not itself establish that the latter has
been built and clinically adjudicated.

### Second-pass numeric-result and report-overlap findings (2026-11-18)

The following aggregates were observed on the **unchanged saved cohort**
after applying the curated OMOP echo-note title matching and numeric
parseability screens. They measure *potential availability*, not validated
clinical reference values or approved exam assignment.

| Discovery measure | Patient/index-days | Denominator |
|---|---:|---:|
| Current extracted report text | 282 (18.80%) | 1,500 |
| Same-day conventional `FACT_ECHO_TEST.ECHO_NOTE` | 592 (39.47%) | 1,500 |
| Curated OMOP echo note, index day or +1 | 429 (28.60%) | 1,500 |
| **Union: at least one report text candidate** | **980 (65.33%)** | **1,500** |
| Curated OMOP candidate not already extracted | 156 (10.40%) | 1,500 |
| Source or OMOP note containing wall-motion keywords | 754 (50.27%) | 1,500 |
| OMOP note containing image-quality keywords | 52 (3.47%) | 1,500 |
| OMOP note containing indication keywords | 0 | 1,500 |
| Numeric `LA_dim_s_MM` or `LA_dim_es_2D` | **39 (2.60%)** | 1,500 |
| Numeric `Body_Surface_area` | 1,210 (80.67%) | 1,500 |
| Numeric `LV_WMS_index_2D` / `MC CV WALL MOTION SCORE INDEX` | 346 (23.07%) | 1,500 |

The 980 potential report-text patient-days are a union across three
sources, **not their sum**; their overlap with distinct clinical exams
and with the model's wall-motion reference is not validated. The
available source/OMOP wall-motion keywords (754) are not equivalent
to validated wall-motion labels. Even if all 980 note candidates
were linked correctly and yielded labels, note text alone could
supply no more than 65.33% of this fixed cohort, below the individual
wall-motion target of 95% (1,425/1,500). More complete or different
reference sources, improved mapping, or a clinically authorized
protocol decision would be needed.

The numeric LA-diameter rows were observed only under the two screened
result names, with explicit units **mm**: `LA_dim_s_MM` (26) and
`LA_dim_es_2D` (13), totaling 39 unique cohort patient-days in the
audit. This is only 2.60% against the configured 85% (1,275/1,500)
target. This **does not prove** all other potential LA diameter
sources are absent; it means the currently identified source names
do not supply the requirement. LA volume, LA area, and indexed LA
volume are not equivalent to LA diameter and must not be silently
substituted.

The numeric-result table withheld the LVEDV, LVESV and LA-volume
counts as **suppressed**, due to the audit's within-row complementary
small-difference rule. The suppression is **not evidence of low
coverage**, nor does it overturn the separately conducted provisional
969-patient-day LVEDV examination-aware audit. LVESV should be
verified under an approved disclosure-safe approach if exact
aggregate counts are necessary.

`FACT_ECHO_TEST` still has no dedicated height/weight columns.
A numeric BSA in 1,210 patient-days does not recover either
individual measurement. The WMSI result name reflects a numeric
**score index** and is not an approved replacement for the
categorical wall-motion reference or a complete 17-segment label.

**Decision gate:** Request developer/clinical input on whether
a separate note/report or exam-linked measurement source exists for
wall-motion, image-quality, indication and LA diameter; independently
validate any new join and measurement-unit definitions. Do not
rebuild `echo_analytic`, promote a reference, substitute endpoints,
or change the locked completeness targets based on these exploratory
counts.

### Conservative measurement/report preview results — 2026-11-18

A subsequent **read-only** trial of a strict Mayo echo source extractor
(`echo_wm_training/cohort_extraction_preview.py`) showed:

| Selection / field | Output | Interpretation |
|---|---:|---|
| Cohort denominator | 1,500 | Unchanged saved `echo_analytic` |
| One unmatched identifier group | 703 (46.87%) | Passes the strict preview examination identity gate |
| Not selected for immediate extraction | 797 (53.13%) | Includes provisional ACC/ORD matches and multiple/unresolved groups |
| LVESV completeness before | 0.00% | Saved cohort field |
| LVESV completeness in preview | **4.07%** | Strict candidate value filling only, not clinical acceptance |
| LVEDV before / after | suppressed | Small-cell/cross-report disclosure safeguard |

The current preview **withholds every provisional ACC/ORD pair** until
identity is independently validated, even when mutually unique within
60 seconds. It also rejects an entire volume metric if *any* recorded
value differs across method names, or if a candidate's unit is not
explicitly mL. These are conservative engineering decisions for an
interim diagnostic; they **do not establish** that the LVEDV/LVESV
source is only 4.07% complete, and are not comparable to the
earlier 969-patient-day potential LVEDV estimate.

Only LVEDV and LVESV are allowed to change in this staged preview.
The other 25 fields, including `wall_motion`, `clinical_indication`
and `image_quality`, remain unchanged by design. Raw
`FACT_ECHO_TEST.ECHO_NOTE` may be selected privately but has not
been converted into validated clinical labels.

A further read-only barrier-disposition audit was added under
`echo_wm_training/cohort_preview_diagnostics.py` to distinguish
exam-gating from absent numeric names and method/unit rejection.
Its BigQuery run is pending. **No cohort rebuild or threshold change
is supported by this intermediate preview.**

### Held LVEDV/LVESV measurement review: unit metadata (2026-11-18)

A subsequent read-only audit of the strict preview split 1,500 saved
cohort patient-days into **703 selected single-identifier groups**,
**718 provisional ACC/ORD pairings needing examination-identity
review**, and **79 other exam/identifier review cases**.

Within the 703 selected patient-days:

| Volume disposition | LVEDV | LVESV |
|---|---:|---:|
| Unique consistent, explicit absolute-mL value | 58 | 61 |
| No recognized, usable numeric result | 191 | 194 |
| Held for unit or method review | 454 | 448 |

For the **454 held LVEDV cases** a further non-overlapping diagnostic
found: **24** unit metadata issues only, **11** different method
values only, **0** isolated same-method conflicts with otherwise
recognized units, and **419** both unit metadata and numeric
differences. Thus **443 of the 454** held LVEDV cases have at least
one in-range measurement with a missing or non-mL unit field.
This is not a statement that all 443 have *blank* units, nor that
they are invalid clinical measurements; the combined category may
also contain within-method disagreement. LVESV's finer-grained
breakdown was suppressed in compliance with the audit's small-cell
safeguard.

**Implication:** Do not equate the strict 4.07% preview LVESV
completeness to source availability. Confirm actual unit codes and
the unit conventions of each specific Mayo volume result name before
relaxing this engineering check. The aggregate-only `echo_wm_training.volume_unit_profile`
diagnostic has now been executed inside MCP. Among the 703 selected
single-identifier patient-days, method-specific 2D/3D LVEDV results
(`LV_vol_BP_ed_2D_by_MOD`: 431,
`LV_vol_4C_ed_2D_by_MOD`: 476,
`LV_vol_2C_ed_2D_by_MOD`: 432,
`LV_vol_ed_3D`: 49)
had **explicit mL** for all in-range numeric results. Their
LVESV counterparts had 426, 474, 427, and 48 patient-days,
respectively, also all explicit mL.

In contrast, the Mayo-specific
`MC CV LV END DIASTOLIC VOLUME` had 443 patient-days with
numeric in-range results and **missing unit metadata in all 443**.
`MC CV LV END SYSTOLIC VOLUME` likewise had 439 patient-days
and **missing unit metadata in all 439**. These are per-result-name
patient-day counts which overlap, and must not be summed into a
cohort completeness estimate.

This strongly localizes the metadata problem to the `MC CV`
result-name family. It does **not** establish that missing units mean
mL. It also does not prove that these 443/439 cases are the same
patient-days as those withheld in the strict preview.

The further **read-only, explicit-mL-only** method-feasibility
scenario (`echo_wm_training.explicit_ml_feasibility`) was run inside
MCP. Within 703 conservatively selected single-identifier groups,
it found **494** patient-days with an LVEDV explicit-mL candidate
(**431** biplane, **21** 3D, **42** single-plane; **209** without
explicit mL), and **491** with an LVESV candidate (**426** biplane,
**21** 3D, **44** single-plane; **212** without explicit mL).
The hypothetical method hierarchy produced **zero** conflicts
within the preferred available tier for either measurement.

These candidate counts represent **70.27% LVEDV** and **69.84%
LVESV** among 703 selected examinations, or **32.93%** and
**32.73%** of the unchanged 1,500 patient-day cohort, respectively.
They are not approved clinical reference completeness. The
apparent improvement over the strict 58/61 preview is due to
testing explicit-mL measurements under a per-method sensitivity
hierarchy rather than rejecting all different method values.

The other **718** patient/index-days with provisional ACC/ORD
pairing remain withheld, alongside 79 other identity/exam
review cases. The read-only `provisional_pair_feasibility` audit
has **now been run in MCP**. Both the sole-pair versus additional-group
topology and the explicit-mL candidate/noncandidate breakdown
were reported as **suppressed** under the configured small-cell and
complementary disclosure safeguards. These outputs cannot be
interpreted as either zero availability or a numeric count of
recoverable pairs. Do not infer the suppressed cells by subtraction
from the previously published 718 patient-day denominator.

No pair was automatically approved, merged, or promoted. A reviewer
authorized inside MCP may inspect the examination-level evidence
without exporting identifiers, source values, or disclosive
subgroup counts. The next operational dependency is a documented
rule for establishing ACC/ORD exam identity independently of
a nearby timestamp, and a clinically agreed measurement-method
selection hierarchy.

This is a **hypothetical method hierarchy**, not an approved
clinical selection or a production cohort change.

No cohort rebuild, reference promotion or threshold change has occurred.

### Interpretation, constraints, and decision requests

- **Source absence dominates:** At least 485 index-day patient-days
  had a conventional TTE record without a recognized numeric LVEDV
  under the current five result-name mappings and screening range.
  This is an observation about *mapped structured results*, not proof
  that no LVEDV exists in unstructured reports, DICOM-derived outputs,
  other tables, or alternative conventions.
- **Potential is not validated completeness:** Even the 969 provisional
  potential patient-days leave **381** below the 1,350 requirement;
  clinical examination identification, measurement hierarchy and
  image alignment are still pending.
- **Mechanical upper bound:** If *all* 23 additional source-present but
  identity-unresolved patient-days were accepted after independent
  review, the potential would be no more than **992/1,500 (66.13%)**
  under the current mapping and cohort; this is not a recommendation
  to count them.
- **Study requirement:** Clinical/study leadership should determine
  whether an alternate validated source (including report review
  or imaging-based LVEDV) is available, or whether the 90% requirement
  requires a formal feasibility/denominator discussion. Do not
  redefine cohort eligibility after seeing completeness rates
  without approved protocol rationale and bias assessment.
- **Technical next step:** Freeze the extractor's matching rules pending
  adjudication. Maintain the existing dataset, perform a small,
  approved *inside-MCP* chart/examination review to validate source
  records and availability categories, and document reviewer
  conclusions before any revised reference/cohort build.

**Reproducibility:** In the MCP Cloud PC (not outside it), run the
read-only modules
`echo_wm_training.source_availability`,
`echo_wm_training.cohort_type_availability`,
`echo_wm_training.source_candidate_gap`, and
`echo_wm_training.ambiguous_lvedv_exams`, as documented in
`RUNBOOK_ECHO_WM.md`. This document records the user-provided aggregate
outputs; no new MCP query or clinical verification was performed when
assembling it. All protected per-examination data and small cells
remain inside MCP.

## Clinical decisions requiring sign-off

1. **Conventional TTE definition:** confirm permitted test descriptions,
   and whether TTE Strain should always be excluded or handled separately.
2. **True exam identity:** decide what independent evidence, beyond near
   timestamps, validates an ACC/ORD pair as one examination.
3. **Measurement hierarchy:** approve or revise the provisional biplane
   > 3D > single-plane > Modified Quinones > visual > unspecified hierarchy.
   Clarify any special role for Modified Quinones with apical correction.
4. **Discordance policy:** decide which measurements define the reference
   when methods disagree. Do not use the technical >10-point flag as a
   clinical rejection rule without review.
5. **Clinical acceptability:** establish handling for repeated values,
   outliers, missing methods, and clinically suspicious EF differences.
6. **Cohort timing:** validate index-day versus index-day/+1 selection.
7. **DICOM alignment:** verify that the selected measurement/exam truly
   corresponds to the images to be used for evaluation or training.
8. **Change control:** require sign-off on a locked sample and documented
   selection rules before any new reference table or cohort promotion.
9. **LVEDV source feasibility:** confirm mapping, units, alternative report or
   image sources, and applicability of the 90% patient-level requirement.
10. **Review eligibility/bias:** assess source-system and site differences
    before proposing any prospective exam-type eligibility restriction.

## Stop conditions

Do not overwrite `echo_lvef`, update the analytic cohort, or lock/promote
new reference values solely on the aggregate agreement rate. Keep
patient-level diagnosis, outlier review, and per-examination outputs inside
MCP. No individual patient or accession data may leave that environment.


### Mayo support guidance received 2026-11-19; exact-test unit-scale screen pending

Mayo support recommends the **Term Finder** from
`My_First_Query.ipynb`, with broad result-description searches:
`end%diastolic`, `end%systolic`, and `left%dimension`, rather
than only previously screened exact `ECHO_TEST_RESULT_NAME` values.
Their cardiology results are centralized in `FACT_ECHO_RESULTS`
across contributing systems, but Mayo reports greater format
consistency in the **Epic-era 2019+ records**. This is a
source-format consideration, **not approval** to restrict the
pre-existing study window or reselect the 1,500-person cohort.

Other newly identified source avenues are:
`ENCOUNTER_NUMBER` for an echo's originating encounter (an
encounter is **not automatically a unique echo exam**);
`FACT_CLINICAL_DOCUMENTS` and titles such as `Echo Report Text`
for earlier narrative reports; `FACT_ECHO_TEST.ECHO_NOTE`
for newer reports; and `FACT_FLOWSHEETS` for height/weight,
preferably same-encounter, otherwise nearest dated values.
Mayo recommends native FACT/DIM cardiology sources rather than
the sparser OMOP transformations when OMOP is not required.
Clinical method choice should be reviewed with the
**Advisor Network / study clinical lead**, not inferred by code.

**Unitless source test — MCP results 2026-11-19:**
The read-only `echo_wm_training.unitless_scale_validation` audit
found the following cohort-index-day source availability by period:

| Unitless source metric | Period | Unambiguous unitless MC CV patient-days | Explicit-mL comparator on the **same ECHO_TEST_FPK** |
|---|---|---:|---:|
| LVEDV | 2012–2018 | 46 | 0 |
| LVEDV | 2019–2023 | 400 | 0 |
| LVESV | 2012–2018 | 46 | 0 |
| LVESV | 2019–2023 | 397 | 0 |

**There were no same-test comparison pairs** and therefore no
ratios, quartiles or agreement statistics to interpret. The first
implementation displayed all ratio rows as `suppressed` even for
empty samples; that reporting issue was corrected so a subsequent
run will show **0** observations and **not_applicable** ratios.
These source-level counts use all conventional same-day TTE tests
in the saved cohort, **not** only the 703 strictly selected cases,
and are not accepted clinical coverage. We cannot conclude that
unitless source values are in mL, nor that they have a different
scale.

The next read-only `echo_wm_training.volume_record_topology`
audit checks whether these source result families occur on
**distinct ECHO_TEST_FPK records within a patient/index-day**, and
whether the associated accession/order groups form merely a
**provisional reciprocal match**. That audit does not compare
raw measurements across different tests, approve an ACC/ORD pair
or change `echo_analytic`. Its MCP run is pending.

**Subsequent cross-record topology result (MCP, 2026-11-19):**
Among 2019–2023 patient/index-days with an unambiguous
unitless MC CV LVEDV result, **385/400** (96.25%) had a
numeric explicit-mL LVEDV on **another ECHO_TEST_FPK** on
the same patient/index-day. For LVESV, this was **382/397**
(96.22%). In both analyses, **zero** unitless/explicit-mL
cross-record pairs met the provisional reciprocal ACC/ORD
matching rule. The corresponding pre-2019 detailed cells
were suppressed. These are *co-present source records*,
**not** verified duplicate studies or clinical measurement pairs.

This explains why the exact-FPK scale check produced no ratios:
the two source result families commonly reside on different test
keys. It provides **no basis** for comparing their numeric values
or interpreting the unitless family as mL without a verified
exam identity crosswalk and source-unit definition.

A next read-only module `echo_wm_training.encounter_schema_discovery`
will enumerate the exact source table/column names and types
related to `ENCOUNTER_NUMBER`, echo tests, reports and flowsheets
via BigQuery INFORMATION_SCHEMA. This metadata query has **not
yet been executed in MCP**; its output will guide an
exam-identity/cardinality audit without assuming a field location.

**Next technical checkpoints:** Validate result names and source
metadata with Adam's Term Finder walkthrough, inspect the actual
source-schema linkage columns, and establish one-to-one exam
identity before interpreting cross-record values. Obtain clinical
sign-off for method selection before promoting any reference
or changing cohort extraction.


### Mayo encounter/source-linkage schema confirmed in MCP (2026-11-19)

An INFORMATION_SCHEMA-only metadata audit confirmed the following
**existing source columns** (exact spelling):

| Source table | Columns relevant to echo source identity |
|---|---|
| `FACT_ECHO_TEST` | `ECHO_TEST_FPK`, `PATIENT_DK`, `ENCOUNTER_NUMBER`, `ORDER_PROC_ID`, `ACCESSION_NUMBER`, `ECHO_PERFORMED_DTM` |
| `FACT_ECHO_RESULTS` | `ECHO_TEST_FPK`, `PATIENT_DK`, `ENCOUNTER_NUMBER`, `ORDER_PROC_ID`, `ECHO_RESULT_VALUE`, `ECHO_RESULTS_VALUE_UNIT_OF_MEASURE` |
| `FACT_CLINICAL_DOCUMENTS` | `PATIENT_DK`, `ENCOUNTER_NUMBER`, `ORDER_NUMBER`, `CLINICAL_DOCUMENT_TITLE`, `CLINICAL_DOCUMENT_TEXT` |
| `FACT_FLOWSHEETS` | `PATIENT_DK`, `ENCOUNTER_NBR`, `FLOWSHEET_ASSESSMENT_DTM`, `FLOWSHEET_RESULT_TXT`, `FLOWSHEET_UNIT_OF_MEASURE_TXT` |
| `FACT_ENCOUNTERS` | `PATIENT_DK`, `EHR_ENCOUNTER_NUMBER`, `BILLING_ENCOUNTER_NUMBER`, `VISIT_NUMBER` |

The schema reveals **potential join keys, not confirmed relationships**.
In particular, `FACT_FLOWSHEETS.ENCOUNTER_NBR` and
`FACT_ECHO_TEST.ENCOUNTER_NUMBER` have different names and have **not**
been proven to use the same identifier namespace. Similarly,
`FACT_ENCOUNTERS` has several encounter identifier types.

A new read-only `echo_wm_training.encounter_volume_linkage` audit
checks the **same patient/index-day and distinct ECHO_TEST_FPK** source
records for exact nonblank order/accession matches, shared encounter,
missing linkage, conflicting nonblank encounters and multiple possible
test pairs. Results are mutually exclusive, threshold-gated aggregates.
A shared encounter is insufficient by itself to establish clinical
exam identity; no values are automatically compared or assigned
to the cohort. MCP execution of this new audit is pending.

With Adam, confirm how `FACT_ECHO_TEST.ENCOUNTER_NUMBER` and
`FACT_ECHO_RESULTS.ENCOUNTER_NUMBER` are populated, whether
`ORDER_PROC_ID` can provide an authoritative test-level crosswalk,
which `FACT_ENCOUNTERS` identifier corresponds to those values,
and how flowsheet/document encounter keys map. Confirm the exact
source key before attempting an echo-time vitals or report join.


### Verified separate-test encounter disagreement: 2019–2023 (2026-11-19)

A subsequent read-only `run_encounter_volume_linkage` audit of
the unchanged saved echo cohort found that **all** 2019–2023
patient/index-days containing both a unitless `MC CV` volume and
an explicit-mL volume on **two different `ECHO_TEST_FPK`s**
were classified as having **different nonblank
`FACT_ECHO_TEST.ENCOUNTER_NUMBER` values**.

| Source-record classification | LVEDV | LVESV |
|---|---:|---:|
| Unitless MC CV without another explicit-mL source test | 15 | 15 |
| Exactly one different-FPK explicit-mL test; **conflicting encounters** | **385** | **382** |
| Multiple possible source-test pairs | 0 | 0 |
| Shared order or accession | 0 | 0 |
| Shared encounter only | 0 | 0 |
| No shared identifiers / missing encounter evidence | 0 | 0 |

Pre-2019 detail was **suppressed**. This is a strong source-record
**provenance difference**, not proof of two separate clinical
examinations and **not** evidence of numerical unit equivalence.
It remains unsafe to numerically compare those values as the
same examination without an authoritative source crosswalk.

Both `FACT_ECHO_TEST` and `FACT_ECHO_RESULTS` carry
`ENCOUNTER_NUMBER` and `ORDER_PROC_ID`. Before interpreting
different echo-test encounter numbers, a follow-up read-only
`echo_wm_training.echo_result_key_consistency` audit will check
whether the TEST and RESULTS attributes agree on the **same
`ECHO_TEST_FPK`**, separately for unitless MC CV versus
explicit-mL source families, and before/after 2019.

The cross-source identifiers are not to be inferred by date
proximity. A genuine same-FPK field mismatch would suggest
source ETL/identifier translation, whereas same-FPK agreement
with different-FPK conflicts would strengthen the evidence of
distinct source records, without proving separate clinical exams.
The new follow-up code is committed but has **not yet run in MCP**.

**Question for Adam:** Why are the `MC CV` and
`LV_vol_*` result families associated with systematically
different encounters despite appearing on the same patient/index-day?
Are they generated through different clinical workflows or ETL
sources, and is there a source-defined crosswalk between them?


### Same-FPK Mayo test versus result identifier audit completed (2026-11-19)

The read-only `echo_wm_training.echo_result_key_consistency` diagnostic
completed in the restricted MCP workspace. For each named result
family, the `FACT_ECHO_TEST` versus `FACT_ECHO_RESULTS`
`ENCOUNTER_NUMBER` comparison and the `ORDER_PROC_ID`
comparison were **equal and nonblank across all observed
patient/index-days** in both study eras:

| Result family | Period | LVEDV patient-days | LVESV patient-days |
|---|---|---:|---:|
| unitless `MC CV` | 2012–2018 | 46 | 46 |
| unitless `MC CV` | 2019–2023 | 400 | 397 |
| explicitly labeled mL | 2012–2018 | 531 | 526 |
| explicitly labeled mL | 2019–2023 | 443 | 441 |

Each patient-day is counted **once per family/metric/period**; the
encounter and order counts cover the *same* patient-days, so should
not be added. For all eight populations, no unequal nonblank
identifiers, both-missing identifiers, one-missing identifiers,
or mixed identifier evidence was observed.

**Interpretation:** No same-`ECHO_TEST_FPK` key discrepancy was
found between `FACT_ECHO_TEST` and `FACT_ECHO_RESULTS`.
This supports internal consistency of Mayo's test/result linkage
and suggests that the previously observed cross-record encounter
disagreements are not caused by a mismatched identifier field
between the source test and its own results. However, it
**does not prove** that two distinct-FPK records represent
two independent clinical exams, nor establish an implied
unit for `MC CV` volume results.

**Clinical extraction implication:** Explicitly labeled mL source
volume results can be investigated independently of the
unitless `MC CV` values. The blocking issue is selecting the
correct source clinical examination and clinically approved
measurement method, not a demonstrated test/result key mismatch.
No cohort, clinical value or protocol threshold was changed.

**Question for Adam:** Why can source tests for the two volume
families coexist on one patient/index-day but have distinct
`ECHO_TEST_FPK` and `ENCOUNTER_NUMBER` values even though each
test/result pair agrees internally? Is there a source-defined
crosswalk or workflow distinction, and which test corresponds
to the intended clinical echo examination?


### Mayo follow-up and November 20 walkthrough: ROW_SOURCE_ID and Epic encounters

Mayo's latest written follow-up, composed during the November 20
walkthrough, clarifies several earlier data-provenance questions:

- **Source system** must be investigated through `ROW_SOURCE_ID`,
  **not inferred from the `MC CV` or `LV_vol_*` prefix**.
  The prefix's meaning has not been authoritatively established.
- The same patient can have **separate encounters and separate echo
  studies on the same date** (for example, a repeat examination
  later in the day). Consequently, the different nonblank encounter
  identifiers previously observed between the unitless and labeled
  volume families must **not** be collapsed into one exam on date
  proximity alone.
- Mayo's assessment from observed numerical magnitudes is that the
  `MC CV LV END ... VOLUME` results **appear to be mL**, but Mayo
  explicitly suggested consulting a clinical advisor. This is
  **not a formal source unit definition or permission to infer units
  in production**.
- For **Epic-system encounters**, Mayo confirms the mapping
  `FACT_ECHO_TEST.ENCOUNTER_NUMBER` =
  `FACT_FLOWSHEETS.ENCOUNTER_NBR` =
  `FACT_ENCOUNTERS.EHR_ENCOUNTER_NUMBER` **when information was
  captured in the same encounter**. Match patient identity too,
  check encounter cardinality and duplicates, and do not assume
  the same equality holds for every historical source.
- Wall motion, image quality and clinical-indication sources
  remain open for further data exploration. The clinical
  measurement-method hierarchy remains a decision for the
  study's advisors, not the Mayo data-engineering support team.

**Engineering follow-up:** A new *read-only*
`echo_wm_training.row_source_provenance` module first looks for
`ROW_SOURCE_ID` in Mayo source-table metadata; if present on
`FACT_ECHO_TEST` and/or `FACT_ECHO_RESULTS`, it generates
privacy-gated patient-day counts for each candidate LVEDV/LVESV
source-code/family, separately for pre-2019 and 2019–2023.
It does not automatically translate source IDs into clinical
study relationships. This diagnostic has not yet been executed
inside MCP. A dedicated Epic same-encounter
`FACT_FLOWSHEETS` vital-sign study can follow with
cardinality checks and clinical time-window decisions.

**Decision state:** Preserve `echo_analytic`, its volume unit policy,
and all study requirements. Source-system provenance, source-unit
validation, and clinical selection remain separate approvals.
