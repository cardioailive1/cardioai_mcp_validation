# Echo wall-motion validation — completeness decision brief

**As of:** 2026-11-24
**Purpose:** Sampson / clinical-advisor readiness decision before the next
validation or training step. This is a **decision request**, not a request to
silently loosen data-quality criteria or rebuild the locked cohort.

## Decision requested

The saved 1,500-patient external-validation cohort is clinically
purpose-specific. Should we **proceed to the next gated step with an
explicitly documented, model-specific completeness exception**, or are
there fields that must be remedied before progression?

Please identify which requirements are **hard gates** for this particular
echo/wall-motion model (especially clinical reference labels, matching
the source echo exam to imaging, and expert LVEF), which are contextual
covariates/metadata, and which can be documented as limitations. The
existing requirements remain unchanged pending explicit approval.

The cohort configuration currently defines:
- **27 elements**: 19 required, 8 optional.
- An **overall required-element data-completeness validation endpoint of
  95%** (`config/cohorts/echo.yaml` `endpoints.secondary`,
  `id: data_completeness`).
- **Element-specific expected completeness**, not universally 95%:
  90% for LVEDV and LVESV, 85% for LA diameter, 95% for
  height/weight/wall motion/image quality, and 90% for clinical
  indication.
- The index cohort and its excluded training subjects are a held-out
  validation reference: changes to `echo_analytic`, locked records or
  target requirements require a documented authorization and
  reproducible before/after audit.

## Findings established on the saved cohort

| Element or screen | Current / candidate result | Existing individual target | Interpretation |
|---|---:|---:|---|
| Overall baseline mean completeness across required fields | **71.3%** | Overall required-element endpoint: **95%** | Original completeness report, not recomputed after staging EIMS volumes |
| Expert LVEF | **100%** | 98% | Existing source extraction passed |
| LVEDV EIMS explicit-mL candidates | **964/1,500 (64.3%)** | 90% | Source candidate availability only; full before/after completeness was **suppressed** |
| LVESV EIMS preview | **955/1,500 (63.67%)** | 90% | Read-only before/after changed from 0% to 63.67%, confirming EIMS feasibility count |
| LA diameter in saved baseline | **0%** | 85% | Mayo source search found only ~39 numeric patient/index-days under recognized LA-dimension names; LA volume/index is **not** a drop-in replacement without clinical approval |
| Wall motion field | **12.8%** | 95% | Existing report-derived cohort element, distinct from `ref_wma` clinical wall-motion reference |
| Image quality | **18.8%** | 95% | Report-derived completeness remains poor |
| Clinical indication | **45.67%** | 90% | Report-derived completeness remains poor |
| Height | **93.87%** | 95% | Approximately 17 additional valid patient-days needed for target |
| Weight | **93.60%** | 95% | 21 additional valid patient-days needed for target |

**Important distinctions:** There are 1,404/1,500 (93.6%)
patient/index-days with exactly **one EIMS source test FPK**, but this
is **examination-identity feasibility, not volume completeness**.
Some of those days may also have an Epic TTE. The separately
reported 964/955 EIMS measurement candidates are what matter for
volume completeness. Do not infer the suppressed LVEDV before/after
count from the candidate figures or other releases.

The configured 90% LV volume target is 1,350/1,500. Under current
recognized EIMS result names and provisional method rules, even
resolving all 96 days without one EIMS source test would not reach
that threshold. This does **not** establish absence of all other
potentially valid Mayo volume results.

## Implementation and extraction strategy

**Implemented on `validation/echo-wm-mcp`, NOT promoted into the
production `echo_analytic` extraction:**

- `echo_wm_training/eims_first_extractor.py` stages EIMS
  `FACT_ECHO_RESULTS` LVEDV/LVESV values.
- EIMS source `ROW_SOURCE_ID=12315` is verified on both test and
  result; the measurement must belong to the **same**
  `ECHO_TEST_FPK`, be within the existing numeric range, and have
  **explicit mL** unit metadata.
- One EIMS source examination per index-day is required. Multiple EIMS
  tests are withheld. Any additional same-day Epic/source exam is
  flagged for clinical examination-selection review.
- The biplane > 3D > single-plane preference is a **provisional
  sensitivity scenario**, **not an approved clinical preference**.
  Conflicting preferred-tier values are withheld rather than averaged
  or silently replaced.
- The original 27-element completeness report was re-evaluated in
  memory. LVESV increased to 63.67%; LVEDV results were appropriately
  disclosure-suppressed; all other fields were unchanged.
- No clinical source data, saved cohort, reference values, locked
  validation population, or thresholds were modified.
- Synthetic regression tests are present. The read-only Mayo cohort
  preview completed; a specific test-suite pass for the latest
  EIMS-first integration has **not** been confirmed in this record.

**Height/weight follow-up is complete for the tested Epic sources:**
For 2019–2023 (600 original patient/index-days), baseline missing
height/weight counts were 41 and 38. Both an exact
`FACT_ECHO_TEST.ENCOUNTER_NUMBER` =
`FACT_FLOWSHEETS.ENCOUNTER_NBR` check and a separate-encounter
direct-measurement search within the ORIGINAL height [-1095,+30]
and weight [-90,+30] date windows found **zero additional numeric
candidates among baseline-missing rows**. The current OMOP extraction
should remain. This does not prove the source data is absent from
every system.

A separate EIMS-exam-keyed `ECHO_NOTE` availability/keyword-cue audit
is committed as `echo_wm_training/eims_note_gap_audit.py`, but
**has not yet been run in MCP**, and keyword cues are **not**
validated clinical wall-motion/quality/indication labels.
Do not claim that this report extraction problem has been solved.

## Model-specific questions for Sampson and clinical advisor

1. **What are the next-stage prerequisites?** Does moving to the next
   phase depend on the global 95% endpoint, each configured
   per-element target, or confirmed availability of a *smaller set*
   of clinically essential validation/reference labels?
2. **Which labels/fields are indispensable?** Please distinguish
   expert LVEF; the independently derived `ref_wma`, territory
   references and WMSI; patient covariates; and report-derived
   wall_motion/image_quality/clinical_indication. The 12.8%
   `wall_motion` cohort field must **not** be misread as an
   evaluation of independent `ref_wma` availability.
3. **Is an explicit exception acceptable?** If so, specify the
   *exact* element(s), a justified threshold or waiver,
   affected endpoint(s), scope/dates, a waiver owner, and required
   limitations/sensitivity analysis. Do **not** globally change
   95% or silently demote required fields.
4. **Clinical selection sign-off:** Is EIMS accepted as the
   preferred source for the relevant index echocardiogram?
   Which LV volume method should be preferred; what happens with
   same-day Epic+EIMS echo exams and conflicting methods? Is LA
   diameter required, or can another clinically appropriate LA
   feature be separately approved with revised endpoint definition?
5. **Guardrails:** Confirm the validation cohort remains
   excluded from wall-motion model training and that clinical
   references and model outputs are matched to the same exam.
   Decide whether the next action is to authorize a controlled
   cohort integration/rebuild, continue label-specific source
   work, or proceed with a documented limitation.

## Files and repeatability

- Main study definition and targets:
  `config/cohorts/echo.yaml` and `config/study.yaml`
- EIMS staged extraction and synthetic tests:
  `echo_wm_training/eims_first_extractor.py`;
  `tests/test_echo_eims_first_extractor.py`
- EIMS source eligibility and source-system provenance:
  `echo_wm_training/eims_volume_eligibility.py`;
  `echo_wm_training/source_system_bridge.py`
- Read-only gap audits:
  `echo_wm_training/epic_flowsheet_feasibility.py`,
  `echo_wm_training/flowsheet_nearby_vitals.py`,
  `echo_wm_training/eims_note_gap_audit.py`
- Detailed query/reproduction steps: `RUNBOOK_ECHO_WM.md`
- Prior Mayo consultation and clinical questions:
  `docs/ECHO_EF_CLINICAL_REVIEW.md`

Clinical data and patient-level candidate frames remain in the Mayo
Cloud PC. Only separately reviewed, disclosure-safe aggregates may
be shared outside Mayo; avoid combining suppressed reports in a way
that reveals protected small cells.

**Status:** Code staged; Mayo read-only feasibility runs completed;
decision and clinical sign-off **pending**. No configuration thresholds
or production cohort mutation authorized yet.
