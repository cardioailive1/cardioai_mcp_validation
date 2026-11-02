"""Build the imaging de-identification request for the MCP TSM (run inside the Mayo Clinic Platform workspace).

MCP releases DICOM only after a de-identification request: one CSV row per selected study with
STUDY_INSTANCE_UID, PATIENT_DK, MODALITY, STMO_NAME_IN_ADL.

    python scripts/build_deid_request.py --modality mri echo --out /workspace/deid

Writes, in --out:
    deid_request_mri.csv, deid_request_echo.csv   one file per modality
    deid_request.csv                              both together (send this one unless the TSM wants them apart)
    deid_request_summary.json                     counts at each selection step (no patient-level data)

Selection (same rules as training uses, so nothing is pulled that the models would not read):
    MRI  : cardiac MR with a final report, a de-identifiable short-axis LGE series (mcp_imaging.build_request)
    echo : transthoracic echo linked to its report by clinic number + date, de-identifiable, with a wall-motion
           statement (mcp_echo.build_request)
    both : adults (>= study.yaml min_age at the study); PATIENT_DK from DIM_PATIENT (unmerged row).
The script stops with an error, and writes nothing, if a file fails its checks.

Order (runbooks MRI / echo, step 1 -> 3): the Phase 2 cohort is built and LOCKED first ({work}.mri_analytic_locked,
{work}.echo_analytic_locked); the script refuses to run before that. The request then covers both the training
candidates and the locked cohort's index studies (Phase 2 scores their images too); the summary reports how many
locked index studies are in the request and how many have no requestable images. The training pool (manifest) is
built AFTER delivery, from the delivered studies, with the locked cohort's patients excluded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from cardioai_mcp import deid  # noqa: E402
from cardioai_mcp.config import load_study  # noqa: E402


LOCKED = {"mri": "mri_analytic_locked", "echo": "echo_analytic_locked"}


def locked_cohort(engine, modality: str) -> pd.DataFrame:
    try:
        return engine.read_table(LOCKED[modality])
    except KeyError:
        raise SystemExit(f"Lock the Phase 2 {modality} cohort first (runbook step 1: run_cohorts.py --cohorts {modality} "
                         f"--save-tables, then snapshot {LOCKED[modality]}). The de-identification request comes after it.")


def cohort_coverage(locked: pd.DataFrame, sel: pd.DataFrame, window_days: int = 1) -> dict:
    """Locked cohort index studies (person_id + index_date) that have a requested study within +/- window_days."""
    lk = pd.DataFrame({"person_id": pd.to_numeric(locked.person_id, errors="coerce"),
                       "index_date": pd.to_datetime(locked.index_date, errors="coerce")}).dropna().drop_duplicates()
    when = pd.to_datetime(sel.get("study_datetime", sel.get("study_date")), errors="coerce").dt.normalize()
    rq = pd.DataFrame({"person_id": pd.to_numeric(sel.person_id, errors="coerce"), "study_date": when}).dropna()
    m = lk.merge(rq, on="person_id", how="left")
    m["hit"] = (m.study_date - m.index_date).abs().dt.days <= window_days
    hit = m.groupby(["person_id", "index_date"]).hit.any()
    return {"phase2_locked_index_studies": int(len(lk)), "phase2_index_studies_requested": int(hit.sum()),
            "phase2_index_studies_not_requestable": int((~hit).sum())}


def build(engine, study: dict, modality: str, start: str, end: str, min_age: float, log=print):
    cfg = study["modality"]
    locked = locked_cohort(engine, modality)
    if modality == "mri":
        from mri_training import mcp_imaging as mod
        studies = mod.find_cmr_studies(engine, cfg, start, end)
        req, rep = mod.build_request(mod.list_series(engine, cfg, studies))
        mcp = cfg["mri"]["mcp"]
    else:
        from echo_wm_training import mcp_echo as mod
        studies = mod.find_echo_studies(engine, cfg, start, end, log=log)
        req, rep = mod.build_request(studies)
        mcp = cfg["echo"]["mcp"]
    rep = {"found": int(studies.study_uid.nunique()), **rep}
    sel = studies[studies.study_uid.isin(req.study_uid)]
    sel, a = deid.adults_only(engine, mcp, sel, min_age)
    sel, p = deid.attach_patient_dk(engine, mcp, sel)
    out = mod.deid_request(sel, req[req.study_uid.isin(sel.study_uid)])
    problems = deid.validate(out, modality)
    cov = cohort_coverage(locked, sel)
    if cov["phase2_index_studies_not_requestable"]:
        log(f"WARNING {modality}: {cov['phase2_index_studies_not_requestable']} of {cov['phase2_locked_index_studies']} "
            "locked Phase 2 index studies have no requestable images (not de-identifiable, no LGE / wall-motion "
            "statement, or no PATIENT_DK); Phase 2 cannot score them. Review before sending.")
    return out, {**rep, **a, **p, **cov, "rows": int(len(out)), "patients": int(sel.clinic_number.nunique()),
                 "problems": problems}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--modality", nargs="+", choices=["mri", "echo"], default=["mri", "echo"])
    ap.add_argument("--out", required=True, help="output folder inside the workspace")
    ap.add_argument("--engine", choices=["bigquery", "spark"], default="bigquery")
    ap.add_argument("--start", default="1900-01-01", help="first study date (default: all)")
    ap.add_argument("--end", default="2100-01-01", help="last study date (default: all)")
    a = ap.parse_args(argv)
    study = load_study()
    from cardioai_mcp.engine import make_engine
    eng = make_engine(a.engine, study)
    run(eng, study, a.modality, a.out, a.start, a.end)


def run(eng, study, modalities, out_dir, start="1900-01-01", end="2100-01-01"):
    os.makedirs(out_dir, exist_ok=True)
    min_age = float(study.get("min_age", 18))
    files, summary = {}, {}
    for m in modalities:
        df, rep = build(eng, study, m, start, end, min_age)
        summary[m] = rep
        files[m] = df
    bad = {m: r["problems"] for m, r in summary.items() if r["problems"]}
    if bad:
        print(json.dumps(summary, indent=2, default=str))
        raise SystemExit(f"Not written: the request fails its checks {bad}")
    for m, df in files.items():
        deid.write_csv(df, os.path.join(out_dir, f"deid_request_{m}.csv"))
    both = pd.concat(files.values(), ignore_index=True)
    deid.write_csv(both, os.path.join(out_dir, "deid_request.csv"))
    summary["total_rows"] = int(len(both))
    with open(os.path.join(out_dir, "deid_request_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {len(both)} studies to {os.path.join(out_dir, 'deid_request.csv')}: send it to the MCP TSM "
          "through the workspace's approved channel.")
    return summary


if __name__ == "__main__":
    main()
