# CardioAI MRI agent: MobileNetV3-UNet on cardiac MRI (infarct, scar extent, STEMI work-up)

A cardiac MRI model trained inside the Mayo Clinic Platform workspace. It confirms a STEMI on late gadolinium enhancement (LGE), measures how much heart muscle is scarred, and flags findings that need further investigation. Under the CardioAI MCP agreement, the MobileNetV3 encoder with this segmentation decoder is approved for export.

| | |
|---|---|
| Input | One CMR study as a zip of its DICOM files. Short-axis LGE is required; short-axis cine is optional and needed for LVEF. Series are picked by `io.select_series` (localisers, long-axis and axial stacks are ignored; PSIR real images are preferred) |
| Preprocessing | `runtime/preprocess.py`: slices ordered along the stack normal, 1.25 mm pixels, 192 × 192 (240 mm), per-slice 1st–99.9th percentile scaling, plus a sequence channel (LGE = 1, cine = 0) |
| Model | `runtime/model.py`: MobileNetV3-Large encoder in 2-D, U-Net decoder (skips at 1/2–1/16), GroupNorm, about 7.2 M parameters. It outputs per-pixel classes (LV pool, myocardium, scar, MVO, RV pool) and four study-level probabilities |
| Measurements | `runtime/quantify.py`: infarct size and MVO (% of LV myocardium), LV mass, transmurality per AHA segment 1–16, scar depth in the wall (ischaemic = from the endocardium; mid-wall or epicardial = non-ischaemic), and EDV, ESV and LVEF from cine (Simpson's discs) |
| Study findings | `infarct`, `nonischemic_lge` (myocarditis, cardiomyopathy, infiltrative: the STEMI mimics), `lv_thrombus`, `lv_aneurysm`. Thresholds are locked in `thresholds.json` |
| Red flags | Locked clinical rules in `config.yaml` → `red_flags`, copied into the model card: infarct ≥ 10% LV, MVO present, LVEF < 35% and < 50%, possible thrombus, aneurysm, non-ischaemic pattern. **The cut-offs are placeholders for the clinical lead to set before training** |
| Lock | `runtime.fingerprint()` covers `model.py`, `preprocess.py`, `quantify.py` and `pipeline.py`. It and `weights_sha256` are recorded in `model_card.json` and must match wherever the model runs |

`runtime/pipeline.py` (`analyse` + `interpret`) turns one study into findings, probabilities, flags and red flags. Training evaluation, the Phase 2 study and the serving container all call it, so the numbers clinicians see come from the code that was validated.

GroupNorm replaces MobileNetV3's BatchNorm. Each training step is one study's slices from one sequence, so BatchNorm would learn per-study statistics in training but use pooled running statistics in serving. It would also wipe out the constant sequence channel. GroupNorm behaves the same in both.

## Training data (hybrid)

| Stage | Data | Loss |
|---|---|---|
| A: segmentation | Public annotated sets (EMIDEC, MyoPS LGE; ACDC cine) and about 300 Mayo studies contoured by readers (LGE, plus cine at ED and ES; 10% double-read) | Class-balanced cross-entropy plus Dice. Classes a dataset doesn't draw are counted under the class that contains them (MyoPS MVO → scar, ACDC scar → myocardium, EMIDEC RV → background) |
| B: whole model | Every Mayo study, with report labels for the four findings (masked where the report is silent), plus the contoured and public studies for segmentation | Segmentation loss plus masked BCE on the study findings |

**Before training, confirm** each public dataset's label values against its README and check its licence for commercial use (`config.yaml` → `public_datasets`).

## Order of work in the MCP workspace

1. **Lock the Phase 2 MRI cohort first** (`config/cohorts/mri.yaml`, studies from 2021-01-01 on). Run `scripts/run_cohorts.py --cohorts mri --save-tables` and snapshot the result as `mri_analytic_locked`. Training excludes these patients. Have the clinical lead confirm the proposed endpoint goals and the red-flag cut-offs.
2. **Map the CMR table.** Fill in `config/modality_tables.yaml` → `mri` (study table, DICOM location, report column) and set `enabled: true`. Check one study with `mri_training.io.read_study`.
3. **Build the manifest:**
   ```python
   from cardioai_mcp.config import load_study; from cardioai_mcp.engine import SparkEngine
   from mri_training.dataset import load_config, build_manifest, summary
   study, cfg = load_study(), load_config()
   excl = spark.table("cardioai_work.mri_analytic_locked").select("person_id").toPandas().person_id
   m = build_manifest(SparkEngine(spark, study), study, cfg, study["modality"], exclude_persons=excl)
   summary(m, cfg["study_heads"]); m.to_parquet("/workspace/mri/manifest.parquet")
   ```
   The build refuses to continue if any patient appears in two splits.
4. **Contour about 300 studies** with the annotation kit (`annotation/README.md`: worklist, pre-drawn outlines, tool options, reader protocol, checks and double-read agreement). Readers draw LV pool, myocardium, scar, MVO and RV on the short-axis LGE, and LV pool, myocardium and RV on cine at ED and ES. Spread the studies across splits and over-sample post-STEMI studies. Save one `.npz` per study: `lge` (S, H, W) in `read_study()` slice order, and optionally `cine` (K, S, H, W) with `cine_phases` (K,). Put its path in the manifest's `label_uri` column.
5. **List the public sets** in a CSV with columns `source,image_uri,label_uri` (NIfTI), on the machine **outside** MCP where they are stored.
6. **Train.** MCP asks that the public datasets stay **outside** the workspace (2026-10-14), so Stage A on public data runs outside and only its weights come in:
   ```bash
   # outside MCP (CardioAI GPU): segmentation on the public sets only
   python -m mri_training.train --public public.csv --out runs/public_v1
   # bring runs/public_v1/weights.pt + model_card.json into the workspace (MCP import process), then on an A100:
   pip install -r requirements-mri.txt      # CUDA build of torch matching the workspace driver
   python -m mri_training.train --manifest /workspace/mri/manifest.parquet --init runs/public_v1 \
          --out /workspace/mri/runs/v1 --cache-dir /workspace/mri/cache --device cuda
   ```
   - `--init` loads the public-data weights strictly. Their hash and the public sets they came from are recorded in the model card (`initialised_from`).
   - Inside MCP, Stage A then fine-tunes on the Mayo reader outlines, and Stage B adds the report labels.
   - The same public-data bundle pre-draws outlines for the readers (`annotation/README.md`).
   - Thresholds are fitted on the `tune` split only, with headroom above the Phase 2 goals. Metrics are reported on `test_time` (2021 on) and `test_site`:
     - finding AUROC, sensitivity and specificity;
     - Dice per structure;
     - infarct size, MVO and LVEF against reader contours;
     - LVEF and infarct size against the report;
     - MVO presence against the report.
7. **Review `model_card.json`** and copy the `test_time` values into the `phase1` fields of `config/cohorts/mri.yaml`.
8. **Export** only `weights.pt`, `thresholds.json` and `model_card.json`. Manifests, contours and per-study predictions stay in MCP.
9. **Run Phase 2 on the locked cohort.** Run `pipeline.analyse` and `interpret` with the exported bundle and write `mri_predictions` following the cohort's `predictions` contract. Add the EMIDEC test set as the external check.

## Serving on Vertex (`serve/`)

A CMR study is far larger than Vertex AI's 1.5 MB online-request limit. The app therefore uploads the study zip to a bucket and sends a reference:

```json
{"instances": [{"study_uri": "gs://<bucket>/mri-studies/<id>.zip"}]}
```

Small inline zips are also accepted as `study_zip_b64`. The container reads only under `ALLOWED_STUDY_PREFIXES`.

`serve/predictor.py` refuses to start in three cases:
- its runtime fingerprint differs from the model card;
- `weights.pt` doesn't match the model card's hash;
- `thresholds.json` doesn't match the model card's findings.

Each answer contains:
- `findings`: infarct size, MVO, mass, AHA segments, scar depth, volumes and LVEF;
- `probabilities` and `flags`;
- `red_flags`;
- `notes`: which series were used and what couldn't be measured;
- `runtime_fingerprint` and `weights_sha256`.

```bash
gsutil cp weights.pt thresholds.json model_card.json gs://<bucket>/mri/v1/
BUNDLE_URI=gs://<bucket>/mri/v1 STUDY_PREFIX=gs://<bucket>/mri-studies/ IMAGE_TAG=v1 mri_training/serve/deploy.sh
```

Every cine phase is segmented, so allow tens of seconds per study on CPU (`n1-standard-8`), or add a GPU.

## Report labels (config.yaml → labels)

| Finding | Positive | Negative | Masked |
|---|---|---|---|
| infarct | Ischaemic pattern, subendocardial or transmural LGE, "consistent with infarction", or LGE in a coronary territory | "No LGE / scar / infarct", or a non-ischaemic pattern only | Otherwise |
| nonischemic_lge | Mid-wall, subepicardial, patchy, non-ischaemic or insertion-point LGE; myocarditis, sarcoid, amyloid or HCM | No LGE, or an infarct only | Otherwise |
| lv_thrombus, lv_aneurysm | Mentioned and not negated | Negated, or not mentioned in a full LGE report (an assumption recorded in the model card) | Short or empty report |

LVEF, infarct size and MVO stated in the report are kept to evaluate the measurements. `config/cohorts/mri.yaml` uses the same rules in SQL, and `tests/test_mri_cohort.py` checks that the two agree report by report.

## Tests

`pytest tests/test_mri_training.py tests/test_mri_cohort.py` checks the following:
- **Report labels**, and the label SQL on synthetic OMOP.
- **DICOM series selection:** short-axis LGE and cine are picked; the axial localiser, 4-chamber cine and non-DICOM files are ignored.
- **Preprocessing:** slice order, scar contrast kept, refusals.
- **Quantification against known synthetic anatomy:**
  - infarct size, MVO and LVEF;
  - anterior AHA segments, with the image flipped too;
  - subendocardial vs mid-wall scar depth.
- **Red-flag rules.**
- **Model:** GroupNorm and strict loading.
- **The partial-label loss.**
- **Fingerprint coverage.**
- **End to end:** training on synthetic studies writes a bundle; the serving container gives the same answer as `pipeline.analyse` and refuses tampered bundles and out-of-prefix or invalid studies.
- **The MRI cohort:** it builds, and its endpoints evaluate.
