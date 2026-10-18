# echo_wm_training: echo wall-motion model (MobileNetV3-WM)

Regional and global wall motion from apical echo cines, trained only on Mayo data inside MCP.
Runbook: [RUNBOOK_ECHO_WM.md](../RUNBOOK_ECHO_WM.md).

| File | What it does |
|---|---|
| `runtime/` | Fingerprinted code shared by training, validation and serving: `preprocess.py` (DICOM / AVI decode, colour refusal, sector crop, 112 px, 20 fps, 1.6 s clips), `model.py` (MobileNetV3-Small frame encoder, temporal motion block, per-view segment slots, view / LVEF / global heads), `segments.py` (AHA 17 segments, apical views, LAD / RCA / LCx, WMSI), `pipeline.py` (`analyse`, `interpret`: flags, pattern, red flags) |
| `labels.py` | Echo report -> 17 segment scores, territories, global hypokinesis, LVEF (the automatic labels; audited) |
| `mcp_echo.py` | Echo studies from `FACT_ECHO_TEST` + the DICOM inventory; de-identification CSV; study and reference tables |
| `dataset.py` | Manifest with labels and patient-disjoint splits; clip table (2-D greyscale cines only) |
| `view_sources.py` | Automatic view labels: measurement reports (SR) and DICOM tags, coverage report, clusters of similar clips (auto-labelled when their tagged clips agree, else one picture per group for a person) |
| `views.py` | Per-clip labelling sheet (last resort), view assignment (person > tag > SR > cluster > confident model) |
| `pretrain.py` | Self-supervised encoder (SimCLR on nearby frames), training period only |
| `train.py` | Stage V (views), stage W (segments, LVEF, global), thresholds on 2020, metrics, export bundle |
| `serve/` | Vertex container (`predictor.py`, `Dockerfile`, `deploy.sh`) |

Outputs per study: `segments` (score 1-4 and class probabilities for each assessed segment), `territories`
(P(abnormal), abnormal segments), `wmsi`, `lvef_pct`, `pattern` (normal / regional / global), `flags`
(locked thresholds), `red_flags` (locked rules) and notes (views used, segments not assessed).
