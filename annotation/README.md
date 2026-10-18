# Imaging annotation in the MCP workspace

Mayo images can't leave the Mayo Clinic Platform workspace, so annotation happens inside it. This kit prepares studies for readers, takes their finished outlines back, checks them, and turns them into training labels. It works with any tool that edits NIfTI label maps.

| Agent | Annotation | Why | Size |
|---|---|---|---|
| **MRI** | **Required.** Outlines on short-axis LGE (LV cavity, myocardium, scar, MVO, RV) and on cine at end-diastole and end-systole (LV cavity, myocardium, RV) | Stage A training on Mayo images, and contour-based test metrics: infarct size, MVO, transmurality per segment and LVEF against readers | 300 studies, 10% read twice |
| **Echo (wall motion)** | **View labels** only, and mostly automatic: DICOM tags, the echo machine's measurement reports, and clusters of similar clips (`echo_wm_training/view_sources.py`); people label the clusters no source settles (about 60 pictures of 8 clips) and, if needed, a few hundred single clips. Wall-motion labels come from the reports and are audited | The wall-motion model needs to know which clips are apical views | ~60 cluster pictures (+ up to 300 clips); cardiologist spot-checks 100 |
| **Echo** | Optional QC. LV cavity traced on two frames (end-diastole, end-systole) of the A4C cine | Checks the EchoNet segmentation behind EF and volumes on Mayo data. The Phase 2 echo endpoints use the report reference, so they don't depend on this | 100 studies, 10% read twice |
| ECG, CAD, Risk, Longitudinal | None | Labels come from reports, over-reads and EHR codes | — |

The **automatic labels** (report rules, troponin logic, pre-drawn outlines) are checked before training by data scientists, with a cardiologist adjudicating: see [LABEL_AUDIT.md](LABEL_AUDIT.md) and `annotation/label_audit.py`.

## Set-up agreed with MCP (Adam Resnick, 2026-10-14)

| Item | Answer | What it means for us |
|---|---|---|
| Desktop, 3D Slicer, ITK-SNAP | Already installed in the workspace | **Route A is the plan**: readers correct outlines in 3D Slicer (or ITK-SNAP) |
| pip install (`monailabel`, `highdicom`) | Allowed | Option A2 (Slicer + MONAI Label on localhost) is open once the pre-drawing model is good |
| Local web apps | Run on localhost in the environment | OHIF / MONAI Label server possible |
| Docker | Runs, but `docker pull` doesn't fetch dependencies: images must be self-contained | Only needed for CVAT; build the image outside and import it as a tar, or skip CVAT |
| Readers | Up to **5 annotator seats**, each with a **cardioailive.com** email and a signed **DUA**; **Mayo readers can't annotate** in our environment | CardioAI hires outside cardiac MRI readers, and each submits the DUA form before access; `select` refuses more than 5 readers |
| GPU | Shared reservation of **A100 40 GB** machines; other types on request | Training and pre-drawing run on an A100 (`--device cuda`) |
| Reports | Radiology (incl. cardiac MRI) in **FACT_RADIOLOGY**; echo notes in **FACT_ECHO_TEST.ECHO_NOTE** | Report labels come from these (`config/modality_tables.yaml`; columns to confirm) |
| Existing annotations | None: labels are what reports and echo notes say | As planned: report labels for every study, reader outlines for the 300 |
| Public datasets | **Keep them outside the environment** if possible | **Not used** (2026-10-15): the MRI model is trained on Mayo data only. Outlines are pre-drawn by a Mayo model trained on the first reader outlines |

Answered on 2026-10-15:
- **Images:** DICOM files are requested through `FACT_RADIOLOGY_DICOM_INVENTORY` and linked to `FACT_RADIOLOGY` by accession number (`mri_training.mcp_imaging` builds the request).
- **Imports:** a self-contained Docker image (tar) goes through the same import process as files.
- **Readers:** each annotator completes the same onboarding as any data user, including the **DUA form**.
- **GPUs:** the A100 reservation is used by switching the machine on; it starts if a machine is free.

Still open with MCP: how `PATIENT_CLINIC_NUMBER` maps to the OMOP `person_id`; the de-identification
request steps, turnaround and where delivered files land; whether echo DICOM is in the same inventory
(modality US); and the process for importing model weights.

## Workflow

```
1. check    python -m annotation.check_workspace                       -> which tools can run here (send to MCP)
2. select   python -m annotation.select mri --manifest manifest.parquet --readers r01,r02,r03 --out worklist.csv
3. pre-draw (MRI, Mayo only) bootstrap in rounds:
            round 1: export the first ~40 studies WITHOUT --bundle; readers outline them from scratch
            then:    python -m mri_training.pretrain --manifest manifest.parquet --out runs/ssl_v1          (once)
                     python -m mri_training.train --manifest manifest_with_labels.parquet --init runs/ssl_v1 \
                            --stage-a-only --out runs/prelabel_r1     # segmentation model from the outlines so far
            round 2+: export the next studies WITH --bundle runs/prelabel_rN; retrain every ~100 studies
4. export   python -m annotation.export mri --worklist worklist.csv --out /workspace/annotation [--bundle runs/prelabel_rN]
5. annotate readers open <out>/<reader>/<study>/ in the tool and correct the *_label.nii.gz files (protocol below)
6. collect  python -m annotation.collect mri --root /workspace/annotation --out /workspace/mri/contours
7. use      join contours/label_uri.csv into the MRI manifest (label_uri column) and train
```

- **What `collect` checks:**
  - It **refuses** a study whose label file is unchanged from the export (not reviewed), has label values that aren't allowed (for example scar on cine), has no LV pool, or has myocardium on fewer than 3 slices.
  - It **warns** about MVO with no surrounding scar, scar with no viable myocardium, a missing RV (needed to place the AHA segments), and an ES cavity that isn't smaller than ED.
- **Double reads:** these go to `agreement.csv`, with Dice per structure plus the infarct-size and EF differences between readers. The numbers go into the model card. Training uses the primary reader (`reader_1`).
- **Reader codes:** use codes, not names, because they end up in folder names.
- **Outline QC:** grade each batch with `python -m annotation.label_audit outlines ...` (pre-drawn before readers, corrected after `collect`). See [LABEL_AUDIT.md](LABEL_AUDIT.md), part 4.

## Choosing the tool

Run `annotation.check_workspace` first. It installs nothing. It reports Python, GPU, disk, whether pip can fetch packages, whether a local web app can run and be opened through JupyterLab, whether there is a desktop session, and whether docker is available.

| Option | What readers use | Needs in the workspace | Best for |
|---|---|---|---|
| **A. 3D Slicer** (Segment Editor) | Desktop app: paint, erase, threshold and islands tools on the pre-drawn outlines | A desktop session (VNC or remote desktop) and Slicer installed | MRI (3-D stacks); echo also works |
| **A2. 3D Slicer + MONAI Label** | Option A, plus a server on localhost that re-runs the model as readers correct (active learning) | Option A, plus `pip install monailabel` | MRI at scale, once the pre-drawing model is good |
| **B. ITK-SNAP** | A lighter desktop editor | A desktop session and ITK-SNAP | MRI, echo |
| **C. OHIF + MONAI Label** | A browser viewer opened through JupyterLab | jupyter-server-proxy (or similar), pip, an OHIF build; labels as DICOM-SEG (convertible) | Teams that can't get a desktop |
| **D. CVAT** | Browser, 2-D | docker or podman, plus a browser proxy | Echo frames |
| **E. Mayo or MCP service** | Their own environment or annotators | Agreement with MCP | When none of A–D is allowed |

Preferred order: A (or A2), then B, then C, then D for echo, then E. All of them read the NIfTI files `export` writes. C and D need a format conversion; ask and I'll add it.

## Questions for MCP

Send these together with `workspace_check.json`.

1. Are there **annotated cardiac MRI or echo images** on the platform (segmentations or curated labels), or only DICOM plus reports?
2. Can we get a **desktop session** (VNC or remote desktop) in our workspace, and **install 3D Slicer or ITK-SNAP** there?
3. Can we **pip-install packages** (monailabel, highdicom) and **open a localhost web app** in the browser through JupyterLab (jupyter-server-proxy)?
4. Is **docker or podman** available in the workspace (for CVAT, or OHIF with Orthanc)?
5. If none of these is possible: does MCP or Mayo offer an **approved annotation environment or service**, and can **Mayo cardiac imaging readers** be engaged under our agreement?
6. Can **our own readers** (non-Mayo cardiologists or radiologists) get workspace accounts to annotate de-identified images?
7. How many workspace seats and **GPU hours** are included? A GPU makes the pre-drawing and the MONAI Label option much faster.

## Reader protocol: cardiac MRI

Readers should be cardiologists or radiologists who report cardiac MRI. **The clinical lead signs off these rules before the pilot**; the scar threshold and the basal-slice rule especially vary between centres. Run a pilot of 10 studies with every reader, review it together, then start.

**LGE (short axis), every slice that shows LV myocardium, from base to apex:**

| Value | Structure | Rules |
|---|---|---|
| 1 | LV blood pool | Endocardial border; papillary muscles and trabeculations count as **blood pool** |
| 2 | Myocardium (viable) | Between the endocardial and epicardial borders, excluding scar. Don't include RV insertion-point enhancement as LV scar unless it extends into the LV wall |
| 3 | Scar (hyperenhanced) | Visually hyperenhanced myocardium, judged against remote myocardium. Borderline cases follow one pre-agreed rule, for example signal above 5 SD of remote myocardium, applied the same way on every slice. Ischaemic scar starts at the endocardium |
| 4 | MVO | The dark core **inside** an infarct (no-reflow). Always surrounded by scar; never blood-pool thrombus |
| 5 | RV blood pool | Endocardial RV border on the same slices; needed to place the AHA segments |

- **LV thrombus:** a dark mass in the cavity, outside the myocardium. Leave it as background (0) and record it in the study's notes; don't label it as MVO.
- **Basal slice:** include it only if myocardium surrounds at least 50% of the cavity, and exclude the LV outflow tract.

**Cine at end-diastole and end-systole:** labels 1, 2 and 5 only (no scar on cine). End-diastole is the largest LV cavity and end-systole the smallest. The export picks both phases, using the model when there is one and otherwise a guess. If a phase is wrong, say so in the notes. Don't draw on another phase.

**Time:** correcting pre-drawn outlines should take about 20 minutes per study. Confirm this in the pilot.

## Reader protocol: echo LV traces (optional)

- On the A4C cine, choose end-diastole (largest LV, around the QRS) and end-systole (smallest LV) in **one** heartbeat.
- Trace the LV endocardial border on those two frames only, with label 1. Include papillary muscles in the cavity, close the contour straight across the mitral annulus, and exclude the LA.
- Frames with any other drawing are refused.
