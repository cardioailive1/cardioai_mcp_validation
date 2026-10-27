# Auditing the automatic labels

Most training and test labels for the ECG, MRI and echo wall-motion agents are produced by code. Regular-expression rules read CMR reports, echo reports and ECG over-reads, the troponin and diagnosis-code logic builds the acute MI and STEMI labels, and the pre-drawing model draws MRI outlines. This protocol checks those labels before any training. It runs in three tiers, **always in this order, for every agent**:

- **Tier 0, automatic audit (required):** independent automatic checks run on the automatic labels before any person reads them (see [Part 1b](#part-1b-tier-0-automatic-audit-required)):
  - **all three agents:** the automatic auditor (`annotation/auto_audit.py`) re-reads every sampled report with its own vocabulary;
  - **ECG:** also the troponin, acute MI and STEMI recomputation from the raw lab rows (part 2) and the AF rhythm cross-check (part 3);
  - **MRI:** also the automatic outline checks (part 4.1).
- **Tier 1, data scientists:** they check the sampled labels against their source (the report text, the lab rows, the image), **without seeing the automatic value**, using written rule sheets.
- **Tier 2, cardiologist:** they adjudicate everything the data scientists can't settle. That means every disagreement (automatic vs data scientist, or data scientist vs data scientist) and every case marked uncertain. The cardiologist also re-reads a **blinded 10% spot-check** of rows that were settled without them, which measures whether the earlier tiers read the reports the way a cardiologist would.

Data scientists do the volume and the cardiologist's time goes to the hard cases: typically 10–20% of the sample. Tier 0 cannot be skipped: `label_audit sample` has no "off" mode, and `label_audit score` refuses a sample drawn without it. Everything below runs **inside the MCP workspace**. The sheets contain report text and never leave it.

Code: `annotation/label_audit.py`. Numbers and pass rules: `annotation/tasks.yaml` → `label_audit`.

## Who does what

| Role | Does | Signs off |
|---|---|---|
| Data scientist 1 and 2 (codes `ds1`, `ds2`, no names) | Fill the blinded audit sheets; grade outlines; run the troponin and rhythm checks | — |
| Cardiologist | Adjudication sheets (disagreements, `U`, spot-check); outline review list; troponin concept list | The rule sheets below, before the audit starts |
| Clinical lead | Accepts the pass rules and the final `audit_results.csv` | The audit result goes in the model card |

## What is audited

| Agent | Automatic label | Source | Check |
|---|---|---|---|
| MRI | `infarct`, `nonischemic_lge`, `lv_thrombus`, `lv_aneurysm`, `report_mvo` | CMR report text (FACT_RADIOLOGY) | Blinded text audit (part 1) |
| MRI | `report_lvef`, `report_infarct_size` | CMR report text | Blinded text audit: number as stated (part 1) |
| MRI | Pre-drawn and reader-corrected outlines | Images | Automatic checks, report comparison, visual grading (part 4) |
| ECG | `af` | Confirmed over-read statement | Blinded text audit (part 1) plus rhythm cross-check (part 3) |
| ECG | `trop_pos`, `acute_mi`, `stemi` | Lab rows plus diagnosis codes | Independent recomputation (part 2) |
| Echo wall motion | `wma_any`, `terr_lad`, `terr_rca`, `terr_lcx`, `global_hypokinesis` (from 17 segment scores) | Echo report (FACT_ECHO_TEST.ECHO_NOTE) | Blinded text audit (part 1), `sample echo` |
| Echo wall motion | `report_lvef` | Echo report | Blinded text audit: number as stated (part 1) |
| Echo wall motion | Clip views (A4C / A2C / A3C) | Images | View labels by people, then a confident view model (`echo_wm_training/views.py`) |

## Part 1: report-derived labels

### 1.1 Sample

```bash
python -m annotation.label_audit sample mri --manifest /workspace/mri/manifest.parquet --auditors ds1,ds2 --auto triage --out /workspace/audit/mri
```

The sample is seeded, so the same manifest gives the same sample. It contains:

- **300 random reports.** These give the NPV of every label and the PPV of the common ones.
- **Top-ups.** Every label gets at least **100 automatic positives**, drawn at random from all its positives, so rare labels such as thrombus and aneurysm still get a meaningful PPV.
- **25 "hard" reports per label.** These are reports where negation or hedging words ("no", "possible", "cannot exclude", "history of", "prior") sit near the label's words. Most rule errors happen here.
- **25 reports per number** where the keyword is present but no number was extracted. This finds missed LVEF and infarct-size values.
- **ECG only:** every ECG the rhythm check flagged (`--extra rr_flags.csv`).

Expect about **600 MRI reports**: 1–2 minutes each, split between two data scientists, with about 20% (around 120) read by both.

The output folder holds:

- `sheet_ds1.xlsx` and `sheet_ds2.xlsx`: the blinded sheets, with an audit id and the text only, in random order. Each has a `rules` tab.
- `key.csv`: the automatic labels. **Not for the auditors or the cardiologist.**

### 1.2 Fill the sheet

Read the text only, never the images. For each label, enter one of the following.

| Entry | Means |
|---|---|
| `1` | The text says it is present |
| `0` | The text says it is absent (or the rule says silence means absent) |
| `NM` | The text does not mention it (only where the rule allows it) |
| `U` | Uncertain, hedged, contradictory, or you can't tell. **Never guess**: `U` goes to the cardiologist |

For numbers, type the value as stated, without the % sign. Leave the cell blank if no number is stated, and enter `U` for a range or for conflicting numbers. Use the `note` column for anything odd, such as a wrong patient or a truncated report.

### 1.3 Rule sheets

These must match what the code is meant to do. **The cardiologist signs them off before the audit starts.** Where the cardiologist disagrees with a rule, the code is changed first.

**CMR report (MRI):**

| Label | 1 | 0 | NM / U |
|---|---|---|---|
| `infarct` | LGE or scar in an ischaemic pattern (subendocardial or transmural, coronary territory), or "consistent with infarct" (prior, old, recent or acute all count) | No LGE; or LGE only in a non-ischaemic pattern | NM: LGE not assessed or not described |
| `nonischemic_lge` | Mid-wall, subepicardial, patchy or insertion-point LGE, or a named cause (myocarditis, sarcoid, amyloid, HCM) | No LGE; or ischaemic LGE only | NM: LGE not assessed or not described |
| `lv_thrombus` | LV (apical or mural) thrombus present now | Stated absent, **or not mentioned**; resolved thrombus | U: "possible", "cannot exclude" |
| `lv_aneurysm` | True LV aneurysm or pseudoaneurysm | Stated absent, or not mentioned | U: "aneurysmal segment" or dyskinesis without the word aneurysm |
| `report_mvo` | Microvascular obstruction or no-reflow present | Stated absent, or not mentioned | U: "possible", "may represent" |
| `report_lvef` | LVEF % measured on **this** CMR | — | Blank: not stated. U: a range, or echo or prior-study values only |
| `report_infarct_size` | Infarct size or scar burden as % of LV mass | — | Blank: grams only or segment counts |

**ECG confirmed over-read:**

| Label | 1 | 0 | U |
|---|---|---|---|
| `af` | Atrial fibrillation or flutter on **this** tracing, including paced rhythm with underlying AF | Any other rhythm; "history of AF" with sinus rhythm now; "no AF"; AF not mentioned | "Possible AF", "consider AF" |

**Echo report (wall motion):**

| Label | 1 | 0 | NM / U |
|---|---|---|---|
| `wma_any` | Hypokinesis, akinesis, dyskinesis or aneurysm of any segment, regional or global | "Normal wall motion", "no regional wall motion abnormalities" | NM: wall motion not described. U: "possible", "cannot exclude" |
| `terr_lad` | An abnormal anterior, anteroseptal, apical anterior / septal segment or the apex (AHA 1, 2, 7, 8, 13, 14, 17), or global hypokinesis | Wall motion described, none of these abnormal | "Septal" alone counts for LAD and RCA |
| `terr_rca` | An abnormal inferior, inferoseptal or apical inferior segment (AHA 3, 4, 9, 10, 15), or global hypokinesis | Described, none abnormal | |
| `terr_lcx` | An abnormal inferolateral / posterior, anterolateral / lateral or apical lateral segment (AHA 5, 6, 11, 12, 16), or global hypokinesis | Described, none abnormal | |
| `global_hypokinesis` | Global / diffuse / generalised LV hypokinesis | Wall motion described without it | NM: not described |
| `report_lvef` | LVEF % of **this** echo; a range is entered as its midpoint | — | Blank: not stated or a prior study only |

When abnormal walls are listed, the other segments count as normal (`unmentioned_normal`, stated in the model card). The clinical lead signs these rules, the AHA segment-to-territory map and the A3C apical slots (proposed: not scored) on the echo wall-motion sign-off sheet.

### 1.4 Score, adjudicate, score again

```bash
python -m annotation.label_audit score mri --dir /workspace/audit/mri
```

The first run refuses to continue while any label cell is empty. Then, for each row and label:

- **The data scientists' entry becomes the reference when:**
  - the automatic label and data scientist 1 agree;
  - data scientist 2 also agrees, where they read the row;
  - nobody entered `U`.
- **Everything else goes to the cardiologist:**
  - automatic vs data scientist disagreements;
  - disagreements between the two data scientists;
  - `U` entries;
  - stated numbers the code missed.

  These rows, plus a random **10% spot-check** of the settled rows, are written to `adjudication_1.xlsx`. It is blinded in the same way: the cardiologist does not see the automatic label or the data scientists' entries. Running `score` again before the cardiologist answers does not create another sheet.
- **The cardiologist's entry is the reference** wherever they filled it. Run `score` again. Any new disagreements, for example after a sheet is corrected, go to `adjudication_2.xlsx`.

An automatic label the code left unset (NaN, masked from training), where the text does state it, isn't adjudicated. It is counted as `masked_but_stated`: data the model could use but doesn't get.

### 1.5 Pass rules

These are set in `tasks.yaml` and are fixed before the audit.

| Measure | Rule |
|---|---|
| **PPV** = share of automatic positives that the reference confirms | ≥ 0.95, and the lower bound of the Wilson 95% CI ≥ 0.90 |
| **NPV** = share of automatic negatives that the reference confirms | ≥ 0.95, and the lower bound ≥ 0.90 |
| A reference of `U` or `NM` against an automatic 1 or 0 | Counts as **wrong**: training on a hedged report is an error |
| Kappa between data scientist 1 and 2 | ≥ 0.80. Below this, the rule sheet is too vague: rewrite it and re-audit |
| Data scientists vs cardiologist on the spot-check | ≥ 0.90 agreement. Below this, the data scientists' reading can't be trusted: the cardiologist reviews more |
| Stated numbers | Match within 0.5 in ≥ 98% of reports; missed values ≤ 5% |
| Adjudication | None pending |

PPV and NPV are estimated within the automatic-positive and automatic-negative groups. That is why the sample is drawn separately from each group. Sensitivity and specificity aren't reported, because the enriched sample would bias them.

**On a FAIL:**

1. `errors.csv` lists every row where the automatic label was wrong, with the reference.
2. Fix the rule in the agent's `config.yaml`.
3. Rebuild the manifest.
4. Change `label_audit.seed` and re-audit on a **fresh sample**. Re-scoring the same rows would only show the fix on the rows it was written for.

The output files are:

- `audit_results.csv`: one row per label, with PPV and NPV and their CIs, kappa, the spot-check, hard-case accuracy and PASS/FAIL.
- `resolved.csv`: every row × label, with who settled it.
- `errors.csv`: the errors listed above.

## Part 1b: tier 0, automatic audit (required)

The automatic auditor is a **second, independent reader** of the report text. It runs on every sampled report of every agent, after the automatic labels are made and before any person reads the sample. It doesn't reuse the training rules. It uses its own vocabulary (`tasks.yaml` → `label_audit.auto_terms`) and reads sentence by sentence, deciding with NegEx-style triggers whether each mention is:

- **affirmed:** stated as present;
- **negated:** "no", "without", "not seen", "resolved";
- **uncertain:** "possible", "cannot be excluded", "suggestive of", "vs", "?";
- **historical** (AF): "history of".

It skips indication and history lines. A single hedged mention makes the whole sentence uncertain. It returns the same entries as the rule sheet: `1`, `0`, `U`, `NM` and numbers, where LVEF ranges and conflicting values give `U`. It runs in the workspace on the standard library only: no model, no network, and no text leaves MCP.

**What tier 0 produces.** `sample` stores the auditor's reading of every row in `key.csv` (not for auditors) and writes `auto_audit_summary.csv`: per label, how often the auditor agrees with the code, finds the text uncertain or silent, and how many rows it settled. The summary holds counts only, so it can be shared. A label with low agreement is an early warning that a rule is wrong, before anyone reads a report.

`--auto` on the `sample` command chooses how the data scientists use tier 0 (there is no "off"):

| `--auto` | What the data scientists read | What the automatic auditor does |
|---|---|---|
| `triage` (default) | Reports it **cannot** settle, plus a **blinded 25%** of the ones it did settle (mixed in, indistinguishable) | Settles a report only when it agrees with the code on **every** label and number, and the report wasn't picked as hard, missing-value or rhythm-flagged. Also adds the evidence column |
| `assist` (fallback) | Every sampled report | Adds an **evidence** column to the sheets: the sentences that mention any label's words. This is not its answer, so the sheets stay blinded. Used when triage FAILs for a label |

```bash
python -m annotation.label_audit sample mri --manifest /workspace/mri/manifest.parquet --auditors ds1,ds2 --auto triage --out /workspace/audit/mri
```

In triage mode:

- **Who reads what:**
  - Every disagreement with the code, every `U`, every hard or flagged report and every missed number goes to the data scientists, then on to the cardiologist as before.
  - The cardiologist's 10% spot-check also draws from the auto-settled reports.
- **Extra pass rules:**
  - On the verification reports, the automatic auditor must match the final reference on **≥ 98%** of entries for each label (`auto_audit.verify_min`).
  - On the auto-settled reports the cardiologist spot-checks, it must match the cardiologist on ≥ 90%.

  Below either level the label FAILs with "re-run with --auto assist". The automatic auditor isn't trusted for that label, so the audit is redone with every report read by a person.
- **What gets reported:** `audit_results.csv` gains `auto_settled`, `auto_verify_n`, `auto_vs_reference` and `auto_vs_cardiologist`. Put them in the model card with the rest.
- **The limit:** the automatic auditor and the code can share blind spots, because both read words. Agreement between them is evidence, not proof. That's why the verification share is blinded, random and scored, and why the rules for hard cases still send those reports to people.

On synthetic reports, triage cut the data-scientist reading from about 530 reports to about 210. Real reports are more varied, so expect a smaller saving; the verification numbers show whether it was safe.

## Part 2: troponin, acute MI and STEMI (ECG)

These labels involve no reading. The data scientist recomputes them **independently** from the raw lab rows and checks that the result matches the SQL.

```bash
python -m annotation.label_audit troponin --manifest /workspace/ecg/manifest.parquet --out /workspace/audit/troponin
```

- 100 ECGs with `acute_mi = 1` and 100 with `acute_mi = 0`.
- **The recomputation:**
  1. Pulls every troponin row within ±30 days.
  2. Applies the window, the row's own upper reference limit, or else the sex-specific limit (males > 15 ng/L, females > 10 ng/L).
  3. Applies the unit rules, then the acute MI and STEMI rules.
- **Pass:** a **100% match** for `trop_pos`, `acute_mi` and `stemi`. Any mismatch is a code or unit bug, so fix it before training.
- **`troponin_units.csv`:** every unit seen. An unexpected unit means a conversion rule is missing.
- **`troponin_concepts.csv`:** which assays the troponin concept set actually contains on the platform, with units, median values and reference limits. **The cardiologist reads and signs it off**, to confirm for example that Troponin I or point-of-care tests are meant to be included.

The clinical validity of the acute MI definition itself (diagnosis code plus raised troponin within the window) is a clinical-lead decision. It is already fixed in `ecg_training/config.yaml`.

## Part 3: AF rhythm cross-check (ECG)

```bash
python -m annotation.label_audit af-rhythm --manifest /workspace/ecg/manifest.parquet --out /workspace/audit/ecg/rr_flags.csv
```

- **The measure:** R peaks are detected on lead II, and the code computes the coefficient of variation (CV) of the RR intervals, a measure of how irregular the rhythm is.
- **What gets flagged:**
  - AF-labelled ECGs with a regular rhythm (CV < 0.08);
  - non-AF ECGs with a very irregular rhythm (CV > 0.20, which can also be ectopy).
- **What happens to flagged ECGs:** they join the blinded text audit (`sample ecg --extra rr_flags.csv`).
- **What it isn't:** a diagnosis. It only finds the over-reads most worth checking. The flag rates go in the model card.

## Part 4: MRI outlines

Run this on each batch: on the **pre-drawn** outlines before readers see them, to triage reader time, and on the **reader-corrected** outlines after `annotation.collect`, to catch slips before training.

```bash
python -m annotation.label_audit outlines --root /workspace/annotation --manifest /workspace/mri/manifest.parquet --graders ds1,ds2 --out /workspace/audit/outlines_r2
python -m annotation.label_audit score-outlines --dir /workspace/audit/outlines_r2
```

### 4.1 Automatic checks (every study)

- **Structural**, the same checks as `collect`:
  - an LV pool exists;
  - myocardium is drawn on at least 3 slices;
  - label values are allowed;
  - MVO sits inside scar;
  - the RV is present;
  - the end-systole cavity is smaller than end-diastole.
- **Plausible ranges:**
  - LVEF between 10% and 80%;
  - infarct ≤ 60% of the LV.
- **Against the report:**
  - LVEF within 10 points of the stated LVEF;
  - infarct size within 10 points;
  - "infarct" in the report but no scar drawn;
  - "no LGE" in the report but scar drawn;
  - MVO in the report but not drawn, or the reverse.

### 4.2 Visual grading (data scientists, in 3D Slicer)

Each grader gets `grades_<code>.xlsx`, listing the study folder and its automatic flags. A second grader grades 10% of the studies. For every study:

| Check (Y / N / NA) | Look for |
|---|---|
| `ring_follows_wall` | Endocardial and epicardial borders follow the bright and dark edges on every slice |
| `no_blood_in_myocardium` | No bright blood pool inside the ring; papillary muscles count as pool |
| `scar_bright_and_in_wall` | Scar sits on clearly bright myocardium and doesn't spill into the blood pool or outside the epicardium |
| `mvo_dark_core_in_scar` | MVO is a dark core inside scar, not cavity thrombus |
| `base_apex_ok` | No myocardium drawn in the LV outflow tract or above the valve plane; the apex is covered |
| `rv_ok` | The RV pool is drawn where the RV is visible |
| `ed_es_phases_ok` | ED is the largest cavity and ES the smallest |

| Grade | Means | Action |
|---|---|---|
| **A** | All checks Y (or NA) | Usable as is |
| **B** | Small local errors (one or two slices, a few pixels) | Minor correction by a reader |
| **C** | Wrong structure, missing slices, or scar or MVO clearly wrong | Full correction or redraw by a reader |

### 4.3 Cardiologist review

`cardiologist_review.csv` lists:

- every grade C;
- every study where the two graders disagree;
- every study graded A despite automatic flags;
- a random 10% of clean A studies (spot-check).

The cardiologist confirms or changes the grade. Data scientists don't judge borderline scar, MVO versus thrombus, or ischaemic versus non-ischaemic patterns: those always go to the cardiologist.

`outline_summary.json` holds the share of A, B and C grades, the kappa between graders and the review count. It goes in the model card.

## What this does and doesn't establish

- The data-scientist tier shows that the automatic labels are **faithful to their source** and that the rules are applied consistently.
- The cardiologist tier settles what that can't: hedged or contradictory reports, image judgements, and whether the rules themselves are clinically right.
- Reports can be wrong. The audit measures agreement with what the report says, not with the patient's true state. For the MRI contour endpoints (infarct size, MVO, transmurality), the reference remains the reader outlines.
- The **Phase 2 reference standard** (VALIDATION_RUNBOOK step 7) is separate from this audit and keeps its own adjudication.

## What goes into the model card

For each label, record:

- the rule-sheet version (a commit hash of this file and of the agent's `config.yaml`);
- the sample size;
- PPV and NPV with their CIs;
- kappa between the data scientists;
- spot-check agreement with the cardiologist;
- the number of adjudicated rows;
- PASS/FAIL.

For ECG, also record the troponin match and the rhythm-flag rates. For MRI, also record the outline grade shares.
