"""Build the de-identification request MCP needs before it gives access to the images (TSM, 2026-11-02).

MCP: "Provide a CSV file with STUDY_INSTANCE_UID, PATIENT_DK, MODALITY and STMO_NAME_IN_ADL to the TSM for the
selected imaging studies to submit for de-identification." Run INSIDE the MCP workspace (BigQuery):

    python scripts/deid_request.py --modality mri echo --dry-run          # counts only: decide the scope first
    python scripts/deid_request.py --modality mri echo --out /workspace/deid

Selection (one row per study; MCP de-identifies whole studies):
  mri   cardiac MR studies (CPT 75557-75565 or a cardiac description in FACT_IMAGINGSTUDY) with a final report and
        a de-identifiable short-axis LGE series (mri_training.mcp_imaging)
  echo  transthoracic echo studies (US, echo in the study description) linked to an echo report by clinic number
        and date, de-identifiable, with a wall-motion statement in the report (echo_wm_training.mcp_echo)
PATIENT_DK comes from DIM_PATIENT by clinic number (the radiology tables carry PATIENT_CLINIC_NUMBER).

Writes:
  deid_request_<modality>.csv   the four columns MCP asked for (to the TSM, the way MCP asks)
  deid_request_all.csv          both modalities in one file
  deid_selection_<modality>.parquet  the selected studies with their reports, dates and counts (stays in the
                                workspace: build_study_table reads it after delivery)
  deid_summary.json             counts per step and per year (aggregate; fine to share)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mri_training.mcp_imaging import DEID_COLUMNS  # noqa: E402


def _years(df: pd.DataFrame, col: str) -> dict:
    y = pd.to_datetime(df[col], errors="coerce").dt.year
    return {str(int(k)): int(v) for k, v in y.value_counts().sort_index().items() if pd.notna(k)}


def select_mri(eng, mod: dict, start: str, end: str):
    from mri_training import mcp_imaging as mi
    studies = mi.find_cmr_studies(eng, mod, start, end)
    series = mi.list_series(eng, mod, studies) if len(studies) else pd.DataFrame(columns=["study_uid"])
    req, rep = mi.build_request(series) if len(series) else (pd.DataFrame(columns=["study_uid"]), {"studies": 0})
    keep = studies[studies.study_uid.isin(req.study_uid)]
    return studies, keep, mi.deid_request(studies, req) if len(keep) else pd.DataFrame(columns=DEID_COLUMNS), rep


def select_echo(eng, mod: dict, start: str, end: str):
    from echo_wm_training import mcp_echo as me
    logs = []
    studies = me.find_echo_studies(eng, mod, start, end, log=logs.append)
    req, rep = me.build_request(studies) if len(studies) else (pd.DataFrame(columns=["study_uid"]), {"studies": 0})
    rep = {**rep, **studies.attrs.get("link_report", {}), "notes": logs}
    keep = studies[studies.study_uid.isin(req.study_uid)]
    return studies, keep, me.deid_request(studies, req) if len(keep) else pd.DataFrame(columns=DEID_COLUMNS), rep


def sample(keep: pd.DataFrame, n: int, seed: int, always: set) -> pd.DataFrame:
    """At most n studies: every study of a patient in `always` (e.g. the Phase 2 cohort), then a seeded random
    sample of the rest."""
    if not n or len(keep) <= n:
        return keep
    must = keep[keep.person_id.isin(always)] if always else keep.iloc[:0]
    rest = keep.drop(must.index)
    k = max(0, n - len(must))
    return pd.concat([must, rest.sample(n=min(k, len(rest)), random_state=seed)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", nargs="+", choices=["mri", "echo"], default=["mri", "echo"])
    ap.add_argument("--start", default=None, help="first study date (default: study.yaml index_start)")
    ap.add_argument("--end", default=None, help="last study date (default: study.yaml data_cutoff)")
    ap.add_argument("--max-studies", type=int, default=0, help="cap per modality (0 = all); Phase 2 patients first")
    ap.add_argument("--phase2-table", nargs="*", default=[],
                    help="work tables whose person_ids are always kept, e.g. mri_analytic_locked echo_analytic_locked")
    ap.add_argument("--out", default="exports/deid")
    ap.add_argument("--dry-run", action="store_true", help="counts only; no CSV")
    ap.add_argument("--engine", choices=["bigquery", "spark"], default="bigquery")
    a = ap.parse_args()

    from cardioai_mcp.config import load_study
    from cardioai_mcp.engine import make_engine
    study = load_study()
    eng, mod = make_engine(a.engine, study), study["modality"]
    start, end = a.start or str(study["index_start"]), a.end or str(study["data_cutoff"])
    always = set()
    for t in a.phase2_table:
        always |= set(eng.read_table(t).person_id.dropna().astype(int))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary, frames = {"window": [start, end], "max_studies": a.max_studies}, []
    for m in a.modality:
        studies, keep, csv, rep = (select_mri if m == "mri" else select_echo)(eng, mod, start, end)
        keep = sample(keep, a.max_studies, int(study.get("seed", 0)), always)
        csv = csv[csv.STUDY_INSTANCE_UID.isin(keep.study_uid.astype(str))]
        multi = int((pd.to_numeric(keep.get("n_patient_dk"), errors="coerce") > 1).sum()) if "n_patient_dk" in keep else 0
        summary[m] = {**{k: v for k, v in rep.items()}, "candidate_studies": int(len(studies)),
                      "selected_studies": int(len(csv)), "selected_patients": int(keep.clinic_number.nunique()),
                      "phase2_patients_selected": int(keep.person_id.isin(always).sum()) if always else None,
                      "clinic_numbers_with_several_patient_dk": multi,
                      "selected_by_year": _years(keep, "study_datetime")}
        print(f"\n{m}: {len(studies)} candidate studies -> {len(csv)} selected ({keep.clinic_number.nunique()} patients)")
        print(json.dumps(summary[m], indent=1, default=str))
        if multi:
            print(f"NOTE: {multi} selected studies belong to a clinic number with several active PATIENT_DK values "
                  "(the smallest is used): mention this to the TSM")
        if not a.dry_run:
            csv.to_csv(out / f"deid_request_{m}.csv", index=False)
            keep.to_parquet(out / f"deid_selection_{m}.parquet", index=False)
            frames.append(csv)
    if not a.dry_run and frames:
        pd.concat(frames)[DEID_COLUMNS].to_csv(out / "deid_request_all.csv", index=False)
    (out / "deid_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"\n{'counts only' if a.dry_run else 'files'} written to {out}/")


if __name__ == "__main__":
    main()
