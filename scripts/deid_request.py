"""Build the de-identification request MCP needs before it gives access to the images (TSM, 2026-11-02).

MCP: "Provide a CSV file with STUDY_INSTANCE_UID, PATIENT_DK, MODALITY and STMO_NAME_IN_ADL to the TSM for the
selected imaging studies to submit for de-identification." Run INSIDE the MCP workspace (BigQuery):

    python scripts/deid_request.py --modality mri echo --dry-run          # counts only: decide the scope first
    python scripts/deid_request.py --modality mri echo --out /workspace/deid

ORDER: run it AFTER the Phase 2 MRI and echo cohorts are built and saved (run_cohorts.py --save-tables; locked
snapshots preferred). Each cohort patient's index study is then requested even if the training rules would skip
it (e.g. an echo report with no wall-motion statement): the agents are scored on those images. The script stops if
a cohort table is missing (--no-cohort overrides, for training studies only).

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
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mri_training.mcp_imaging import DEID_COLUMNS  # noqa: E402


def _years(df: pd.DataFrame, col: str) -> dict:
    y = pd.to_datetime(df[col], errors="coerce").dt.year
    return {str(int(k)): int(v) for k, v in y.value_counts().sort_index().items() if pd.notna(k)}


def select_mri(eng, mod: dict, start: str, end: str):
    """(candidate studies with a deid_ok flag, the study_uids the training rules select, counts)."""
    from mri_training import mcp_imaging as mi
    studies = mi.find_cmr_studies(eng, mod, start, end)
    series = mi.list_series(eng, mod, studies) if len(studies) else pd.DataFrame(columns=["study_uid", "n_deid_ok"])
    req, rep = mi.build_request(series) if len(series) else (pd.DataFrame(columns=["study_uid"]), {"studies": 0})
    ok = series.groupby("study_uid").n_deid_ok.sum() if len(series) else pd.Series(dtype=float)
    studies = studies.assign(deid_ok=studies.study_uid.map(ok).fillna(0) > 0, modality=studies.get("modality", "MR"))
    return studies, set(req.study_uid.astype(str)), rep


def select_echo(eng, mod: dict, start: str, end: str):
    from echo_wm_training import mcp_echo as me
    logs = []
    studies = me.find_echo_studies(eng, mod, start, end, log=logs.append)
    req, rep = me.build_request(studies) if len(studies) else (pd.DataFrame(columns=["study_uid"]), {"studies": 0})
    rep = {**rep, **studies.attrs.get("link_report", {}), "notes": logs}
    studies = studies.assign(deid_ok=pd.to_numeric(studies.get("n_deid_ok"), errors="coerce").fillna(0) > 0)
    return studies, set(req.study_uid.astype(str)), rep


def to_csv(rows: pd.DataFrame, default_modality: str) -> pd.DataFrame:
    out = pd.DataFrame({"STUDY_INSTANCE_UID": rows.study_uid.astype(str), "PATIENT_DK": rows.patient_dk,
                        "MODALITY": rows.get("modality", pd.Series(default_modality, index=rows.index)).fillna(default_modality),
                        "STMO_NAME_IN_ADL": rows.stmo}).drop_duplicates(["STUDY_INSTANCE_UID", "STMO_NAME_IN_ADL"])
    missing = out[DEID_COLUMNS].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} studies lack a value in {DEID_COLUMNS} (no active DIM_PATIENT row "
                         "for the clinic number?)")
    return out[DEID_COLUMNS].reset_index(drop=True)


def cohort_index_studies(studies: pd.DataFrame, cohort: pd.DataFrame, date_col: str, days: int = 1):
    """The Phase 2 cohort's index studies: each cohort patient's de-identifiable study on the index date (within
    `days`; the nearest one). They are requested whatever the training rules say: the agent is scored on them.
    Returns (rows, number of cohort patients with no such study)."""
    if cohort is None or not len(cohort):
        return studies.iloc[:0], 0
    c = cohort[["person_id", "index_date"]].dropna().assign(person_id=lambda x: x.person_id.astype("int64"),
                                                          index_date=lambda x: pd.to_datetime(x.index_date).dt.normalize())
    s = studies[studies.deid_ok & studies.person_id.notna()].assign(
        person_id=lambda x: x.person_id.astype("int64"), _d=lambda x: pd.to_datetime(x[date_col]).dt.normalize())
    j = s.merge(c, on="person_id")
    j = j[(j._d - j.index_date).abs() <= pd.Timedelta(days=days)]
    j = j.assign(_gap=(j._d - j.index_date).abs()).sort_values(["person_id", "_gap"]).drop_duplicates("person_id")
    return j.drop(columns=["_d", "_gap", "index_date"]), int(c.person_id.nunique() - j.person_id.nunique())


def sample(rows: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """At most n rows, a seeded random sample (n = 0: all)."""
    if not n or len(rows) <= n:
        return rows
    return rows.sample(n=n, random_state=seed)


COHORT_TABLES = {"mri": ["mri_analytic_locked", "mri_analytic"], "echo": ["echo_analytic_locked", "echo_analytic"]}


def load_cohort(eng, m: str, name: Optional[str]):
    """The Phase 2 cohort table for modality m: the given name, else the locked snapshot, else the saved build."""
    for t in ([name] if name else COHORT_TABLES[m]):
        try:
            return t, eng.read_table(t)
        except KeyError:
            continue
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", nargs="+", choices=["mri", "echo"], default=["mri", "echo"])
    ap.add_argument("--start", default=None, help="first study date (default: study.yaml index_start)")
    ap.add_argument("--end", default=None, help="last study date (default: study.yaml data_cutoff)")
    ap.add_argument("--max-studies", type=int, default=0,
                    help="cap on the TRAINING studies per modality (0 = all); the Phase 2 index studies are always in")
    ap.add_argument("--mri-cohort", default=None, help="Phase 2 MRI cohort table (default mri_analytic_locked, else mri_analytic)")
    ap.add_argument("--echo-cohort", default=None, help="Phase 2 echo cohort table (default echo_analytic_locked, else echo_analytic)")
    ap.add_argument("--no-cohort", action="store_true", help="run before the cohorts exist (training studies only)")
    ap.add_argument("--out", default="exports/deid")
    ap.add_argument("--dry-run", action="store_true", help="counts only; no CSV")
    ap.add_argument("--engine", choices=["bigquery", "spark"], default="bigquery")
    a = ap.parse_args()

    from cardioai_mcp.config import load_study
    from cardioai_mcp.engine import make_engine
    study = load_study()
    eng, mod = make_engine(a.engine, study), study["modality"]
    start, end = a.start or str(study["index_start"]), a.end or str(study["data_cutoff"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary, frames = {"window": [start, end], "max_training_studies": a.max_studies}, []
    for m in a.modality:
        cname, cohort = (None, None) if a.no_cohort else load_cohort(eng, m, getattr(a, f"{m}_cohort"))
        if cohort is None and not a.no_cohort:
            raise SystemExit(f"{m}: no Phase 2 cohort table found ({' / '.join(COHORT_TABLES[m])}). Build the cohort "
                             f"first (run_cohorts.py --cohorts {m} --save-tables), so its index studies are in the "
                             "request, or pass --no-cohort to request training studies only.")
        studies, train_uids, rep = (select_mri if m == "mri" else select_echo)(eng, mod, start, end)
        idx, unmatched = cohort_index_studies(studies, cohort, "study_datetime")
        train = studies[studies.study_uid.astype(str).isin(train_uids) & ~studies.study_uid.isin(idx.study_uid)]
        train = sample(train, a.max_studies, int(study.get("seed", 0)))
        keep = pd.concat([idx.assign(purpose="phase2_index"), train.assign(purpose="training")], ignore_index=True)
        csv = to_csv(keep, "MR" if m == "mri" else "US") if len(keep) else pd.DataFrame(columns=DEID_COLUMNS)
        multi = int((pd.to_numeric(keep.get("n_patient_dk"), errors="coerce") > 1).sum()) if "n_patient_dk" in keep else 0
        summary[m] = {**rep, "cohort_table": cname, "candidate_studies": int(len(studies)),
                      "phase2_index_studies": int(len(idx)), "phase2_patients_without_a_deidentifiable_study": unmatched,
                      "training_studies": int(len(train)), "selected_studies": int(len(csv)),
                      "selected_patients": int(keep.clinic_number.nunique()) if len(keep) else 0,
                      "clinic_numbers_with_several_patient_dk": multi,
                      "selected_by_year": _years(keep, "study_datetime") if len(keep) else {}}
        print(f"\n{m}: cohort {cname or '(none)'}; {len(studies)} candidate studies -> {len(idx)} Phase 2 index + "
              f"{len(train)} training = {len(csv)} selected")
        print(json.dumps(summary[m], indent=1, default=str))
        if unmatched:
            print(f"NOTE: {unmatched} Phase 2 {m} patients have no de-identifiable study on their index date: they "
                  "cannot be scored on images; review before locking")
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
