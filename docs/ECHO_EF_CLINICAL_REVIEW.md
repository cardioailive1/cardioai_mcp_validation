# Echo EF reference selection — clinical-review evidence (provisional)

**Status:** Technical diagnostic only; clinical approval required before altering
the active reference. Compiled 2026-11-17 from aggregate outputs observed inside
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

## Stop conditions

Do not overwrite `echo_lvef`, update the analytic cohort, or lock/promote
new reference values solely on the aggregate agreement rate. Keep
patient-level diagnosis, outlier review, and per-examination outputs inside
MCP. No individual patient or accession data may leave that environment.
