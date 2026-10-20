# CardioAI ECG model: MobileNetV3-1D (acute MI, STEMI, AF)

A 12-lead ECG model trained on Mayo ECGs inside the Mayo Clinic Platform workspace. Under the CardioAI MCP agreement, MobileNetV3 weights are approved for export.

| | |
|---|---|
| Input | Resting 12-lead ECG, at least 9.5 s, any sample rate, mV or µV. The MCP waveform table's Base64 strings are decoded (`io.decode_waveform`); GE MUSE XML (8 stored leads), WFDB and `.npz` are also read |
| Preprocessing | `runtime/preprocess.py`: lead order and derived limb leads, mV, 500 Hz, zero-phase 0.67–100 Hz filter, 10 s window, no per-lead normalisation |
| Model | `runtime/model.py`: MobileNetV3-Small layout in 1-D, about 1.05 M parameters, one sigmoid per finding |
| Outputs | `acute_mi`, `stemi`, `af` (AF or flutter); thresholds are locked in `thresholds.json` |
| Lock | `runtime.fingerprint()` (preprocessing + model code) and `weights_sha256` are recorded in `model_card.json` and must match wherever the model runs |

## Order of work in the MCP workspace

1. **Lock the Phase 2 ECG cohort first.** Run step 7a of the runbook for `ecg`. Its index window starts on 2021-01-01, the same date as `splits.test_start`. Snapshot the result as `ecg_analytic_locked`. Training excludes these patients.
2. **Map the ECG waveform table and probe it.** MCP confirmed (2026-10-14) that the waveform table stores each waveform as a **Base64-encoded string** (`waveform_storage: base64`).
   - Fill in `config/modality_tables.yaml` → `ecg`: the table, the waveform column, sample rate, lead count and the confirmed-read column. Then set `enabled: true`.
   - Run `python -m ecg_training.probe_waveforms --n 5`. It reports what the Base64 holds (MUSE XML, gzip, npz, JSON, or raw samples) and, for raw samples, the lead layout, the seconds per recording and the amplitude in mV. Expect about 10 s and R waves of about 0.5–2 mV. If not, adjust `waveform_raw` (`dtype`, `units_per_bit`, `layout`) and probe again.
   - The MCP Resource Hub's ECG training covers the table.
3. **Troponin.** Mayo's troponin is **Troponin T, 5th generation**: [TRPS](https://www.testcatalog.org/show/TRPS), reporting name "Troponin T, 5th gen, P", LOINC 67151-1, in ng/L.
   - Rows above the row's own `range_high` count as raised. Rows without one fall back to the sex-specific Mayo reference values: **males ≤ 15 ng/L, females ≤ 10 ng/L** (`labels.troponin_url_fallback`).
   - For older troponin rows in other units, look up the assay by its reporting name on [mayocliniclabs.com](https://www.mayocliniclabs.com/search#q=troponin), under the "Clinical and Interpretive" and "Fees & Codes" tabs. Then confirm `ng_per_ml`.
4. **Build the training manifest, then decode the waveforms once:**
   ```python
   from cardioai_mcp.config import load_study; from cardioai_mcp.engine import SparkEngine
   from ecg_training.dataset import load_config, build_manifest, summary
   from ecg_training.extract_waveforms import extract_waveforms
   study, cfg = load_study(), load_config()
   eng = SparkEngine(spark, study)
   excl = spark.table("cardioai_work.ecg_analytic_locked").select("person_id").toPandas().person_id
   m = build_manifest(eng, study, cfg, study["modality"], exclude_persons=excl)
   m = extract_waveforms(eng, m, study["modality"], "/workspace/ecg/waveforms")   # Base64 -> one .npz per ECG
   summary(m, cfg["heads"]); m.to_parquet("/workspace/ecg/manifest.parquet")
   ```
   - The manifest never holds the Base64 strings (about 160 kB each). `extract_waveforms` reads them in batches by `ecg_id`, writes `.npz` files and can be re-run: it resumes where it stopped.
   - ECGs that fail to decode keep the reason in `waveform_error`. Training drops them and records the count in the model card.
   - The build refuses to continue if any patient appears in two splits.
   - Adults only: tests done before age 18 (`splits.min_age`, the Phase 2 cohort rule) are dropped from every split, and `m.attrs["n_under_min_age"]` gives the count.
5. **Train on the GPU workspace:**
   ```bash
   pip install -r requirements-ecg.txt   # CUDA build of torch matching the workspace driver
   python -m ecg_training.train --manifest /workspace/ecg/manifest.parquet --out /workspace/ecg/runs/v1
   ```
   Thresholds are fitted on the `tune` split only. The targets include headroom above the Phase 2 goals, because success needs the lower 95% CI bound to clear each goal. Metrics are reported on `test_time` (2021 on) and `test_site` (held-out sites).
6. **Review `model_card.json`.** Check the data summary, label definitions, tuning and test metrics, fingerprints and the commit. Copy the `test_time` values into the `phase1` fields of `config/cohorts/ecg.yaml`, then rerun `scripts/sample_size_report.py`.
7. **Export** only `weights.pt`, `thresholds.json` and `model_card.json`. Manifests and per-ECG predictions stay in MCP.
8. **Run Phase 2 on the locked cohort** with the same runtime and bundle (step 7d of the runbook). Then add the external public test: MIMIC-IV-ECG for acute MI, and CODE-15 or Chapman-Shaoxing for AF.

## Serving on Vertex (`serve/`)

`serve/predictor.py` runs the same `runtime/` code as training. It **refuses to start** in three cases:
- its runtime fingerprint differs from the model card;
- `weights.pt` doesn't match the model card's hash;
- the weights don't load strictly.

Each answer carries `runtime_fingerprint` and `weights_sha256`. The app checks both against its lock file and refuses (409) on any mismatch.

```bash
gsutil cp weights.pt thresholds.json model_card.json gs://<bucket>/ecg/v1/
BUNDLE_URI=gs://<bucket>/ecg/v1 IMAGE_TAG=v1 ecg_training/serve/deploy.sh    # builds from the repo root
```

The request is `{"instances": [{"signal": n_leads x N, "fs": Hz, "leads": [...], "units": "mV"}]}`. The container does all preprocessing, so callers send the raw recording.

## Label definitions (config.yaml)

| Head | Positive | Negative | Not used (masked) |
|---|---|---|---|
| acute_mi | Acute MI diagnosis **and** troponin above the upper reference limit (the row's own limit, else Mayo's: males > 15, females > 10 ng/L), both within −1 to +3 days of the ECG | No MI diagnosis of any kind within ±30 days | Diagnosis without troponin, or the reverse |
| stemi | acute_mi = 1 **and** a STEMI diagnosis within −1 to +3 days | acute_mi = 0, or NSTEMI without a STEMI diagnosis | Otherwise |
| af | Confirmed read says atrial fibrillation or flutter on this tracing | Confirmed read without it, or negated, historical ("history of", "on the prior tracing") or resolved ("no longer present") | No confirmed read; only hedged mentions ("possible", "consider", "versus") |

## Tests

`pytest tests/test_ecg_training.py` checks:
- sample-rate invariance, 8-lead MUSE derivation, units and lead order, and refusals;
- the MUSE XML reader;
- label rules, and patient-disjoint time and site splits with Phase 2 exclusions;
- the label SQL on synthetic OMOP;
- strict loading: a wrapped or mismatched checkpoint is refused;
- the threshold rule;
- end to end, that training learns synthetic AF, STEMI and NSTEMI patterns and writes a bundle whose fingerprint matches.
