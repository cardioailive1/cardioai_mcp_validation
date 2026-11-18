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
audit has now been added, but its BigQuery results **have not been
received yet**. Therefore the incremental recoverable coverage for
report-derived labels, LVESV or LA diameter is still unknown.

**Do not conflate** the deck's `wall_motion` field (derived from
`echo_report`) with the model-performance reference
`ref_wma` (from the separate `echo_wm_reference` table). Improving
report extraction does not itself establish that the latter has
been built and clinically adjudicated.

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
