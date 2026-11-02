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
  both  adults only (age >= study.yaml min_age at the study, from OMOP person; unknown age is not requested);
        PATIENT_DK from DIM_PATIENT by clinic number, unmerged row (the radiology tables carry PATIENT_CLINIC_NUMBER);
        a clinic number with no unmerged row, or with several, is left out and counted (ask the TSM about them)
Checks before anything is written: exact columns in order, no empty cells, no duplicate studies, MODALITY MR / US,
DICOM-style UIDs. If a file fails, nothing is written.

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

from cardioai_mcp import deid  # noqa: E402
from cardioai_mcp.deid import DEID_COLUMNS  # noqa: E402


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
    return out[DEID_COLUMNS].reset_index(drop=True)


def patient_key_ok(rows: pd.DataFrame):
    """Rows with exactly one unmerged DIM_PATIENT row for their clinic number; counts of the rest."""
    n = pd.to_numeric(rows.get("n_patient_dk", pd.Series(1, index=rows.index)), errors="coerce")
    missing = rows.patient_dk.isna() | n.isna() | (n == 0)
    several = ~missing & (n > 1)
    return rows[~missing & ~several], {"studies_without_patient_dk": int(missing.sum()),
                                       "studies_with_several_patient_dk": int(several.sum())}


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


def run(eng, study: dict, modalities, out_dir: str, start: Optional[str] = None, end: Optional[str] = None,
        max_studies: int = 0, cohorts: Optional[dict] = None, no_cohort: bool = False, dry_run: bool = False,
        log=print) -> dict:
    mod = study["modality"]
    start, end = start or str(study["index_start"]), end or str(study["data_cutoff"])
    min_age = float(study.get("min_age", 18))
    out = Path(out_dir)
    summary, files, sel = {"window": [start, end], "max_training_studies": max_studies, "min_age": min_age}, {}, {}
    for m in modalities:
        cname, cohort = (None, None) if no_cohort else load_cohort(eng, m, (cohorts or {}).get(m))
        if cohort is None and not no_cohort:
            raise SystemExit(f"{m}: no Phase 2 cohort table found ({' / '.join(COHORT_TABLES[m])}). Build the cohort "
                             f"first (run_cohorts.py --cohorts {m} --save-tables), so its index studies are in the "
                             "request, or pass --no-cohort to request training studies only.")
        studies, train_uids, rep = (select_mri if m == "mri" else select_echo)(eng, mod, start, end)
        studies, ages = deid.adults_only(eng, mod[m]["mcp"], studies, min_age)
        studies, keys = patient_key_ok(studies)
        idx, unmatched = cohort_index_studies(studies, cohort, "study_datetime")
        train = studies[studies.study_uid.astype(str).isin(train_uids) & ~studies.study_uid.isin(idx.study_uid)]
        train = sample(train, max_studies, int(study.get("seed", 0)))
        keep = pd.concat([idx.assign(purpose="phase2_index"), train.assign(purpose="training")], ignore_index=True)
        csv = to_csv(keep, deid.EXPECTED_MODALITY[m]) if len(keep) else pd.DataFrame(columns=DEID_COLUMNS)
        problems = deid.validate(csv, m)
        summary[m] = {**rep, **ages, **keys, "cohort_table": cname, "candidate_studies": int(len(studies)),
                      "phase2_index_studies": int(len(idx)), "phase2_patients_without_a_requestable_study": unmatched,
                      "training_studies": int(len(train)), "selected_studies": int(len(csv)),
                      "selected_patients": int(keep.clinic_number.nunique()) if len(keep) else 0,
                      "selected_by_year": _years(keep, "study_datetime") if len(keep) else {}, "problems": problems}
        log(f"\n{m}: cohort {cname or '(none)'}; {len(idx)} Phase 2 index + {len(train)} training = {len(csv)} studies")
        if unmatched:
            log(f"NOTE: {unmatched} Phase 2 {m} patients have no requestable study on their index date (not "
                "de-identifiable, or no single PATIENT_DK): they cannot be scored on images; review before locking")
        if keys["studies_with_several_patient_dk"] or keys["studies_without_patient_dk"]:
            log(f"NOTE: {keys['studies_with_several_patient_dk']} studies left out for several unmerged PATIENT_DK rows, "
                f"{keys['studies_without_patient_dk']} for none: ask the TSM which key to use")
        files[m], sel[m] = csv, keep
    log(json.dumps(summary, indent=1, default=str))
    bad = {m: summary[m]["problems"] for m in modalities if summary[m]["problems"]}
    if dry_run:
        return summary
    if bad:
        raise SystemExit(f"Not written: the request fails its checks {bad}")
    out.mkdir(parents=True, exist_ok=True)
    for m in modalities:
        deid.write_csv(files[m], str(out / f"deid_request_{m}.csv"))
        sel[m].to_parquet(out / f"deid_selection_{m}.parquet", index=False)
    deid.write_csv(pd.concat(list(files.values()), ignore_index=True), str(out / "deid_request_all.csv"))
    (out / "deid_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log(f"\nfiles written to {out}/: send deid_request_all.csv (or the per-modality files) to the MCP TSM the way the "
        "TSM asks; it holds study and patient keys")
    return summary


def main(argv=None):
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
    ap.add_argument("--dry-run", action="store_true", help="counts only; nothing written")
    ap.add_argument("--engine", choices=["bigquery", "spark"], default="bigquery")
    a = ap.parse_args(argv)
    from cardioai_mcp.config import load_study
    from cardioai_mcp.engine import make_engine
    study = load_study()
    run(make_engine(a.engine, study), study, a.modality, a.out, a.start, a.end, a.max_studies,
        {"mri": a.mri_cohort, "echo": a.echo_cohort}, a.no_cohort, a.dry_run)


if __name__ == "__main__":
    main()
