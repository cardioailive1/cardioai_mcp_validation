"""Choose the studies to annotate and assign readers (deterministic, seeded).

    python -m annotation.select mri  --manifest /workspace/mri/manifest.parquet --readers dr_a,dr_b,dr_c --out worklist.csv
    python -m annotation.select echo --manifest echo_studies.csv --readers dr_a,dr_b --out echo_worklist.csv

MRI: target_studies spread over the splits (tasks.yaml split_share), enriched for post-STEMI and
non-ischaemic LGE studies, at most one study per patient. Echo: a random sample of the given studies.
double_read_fraction of studies get a second, independent reader (reader_2); readers are balanced.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).parent


def load_tasks(path: Optional[str] = None) -> dict:
    return yaml.safe_load(Path(path or HERE / "tasks.yaml").read_text())


def _assign(w: pd.DataFrame, readers: List[str], double: float, rng: np.random.Generator, max_readers: int = 5) -> pd.DataFrame:
    if len(readers) > max_readers:
        raise ValueError(f"{len(readers)} readers but MCP allows {max_readers} annotator seats")
    if len(readers) < (2 if double > 0 else 1):
        raise ValueError("need at least two readers for double reads")
    w = w.reset_index(drop=True)
    w["reader_1"] = [readers[i % len(readers)] for i in range(len(w))]
    n2 = int(round(double * len(w)))
    pick = set(rng.choice(len(w), n2, replace=False).tolist()) if n2 else set()
    w["reader_2"] = [readers[(i + 1 + (i // len(readers)) % (len(readers) - 1)) % len(readers)] if i in pick else ""
                     for i in range(len(w))]
    return w


def select_mri(m: pd.DataFrame, readers: List[str], task: dict, seed: int = 20261014, n: Optional[int] = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = int(n or task["target_studies"])
    m = m[m.split.isin(list(task["split_share"]))].copy()
    m = m.sample(frac=1.0, random_state=seed).drop_duplicates("person_id")            # one study per patient
    m["stratum"] = np.where(m.get("recent_stemi", 0).fillna(0).astype(int) == 1, "recent_stemi",
                            np.where(m.get("nonischemic_lge", 0).fillna(0) == 1, "nonischemic_lge", "other"))
    picks = []
    for split, share in task["split_share"].items():
        part, k = m[m.split == split], int(round(n * share))
        chosen = []
        for stratum, frac in task["enrich"].items():
            s = part[(part.stratum == stratum) & ~part.study_id.isin(chosen)]
            chosen += s.study_id.head(int(round(k * frac))).tolist()
        rest = part[~part.study_id.isin(chosen)]
        chosen += rest.study_id.head(k - len(chosen)).tolist()
        picks.append(part[part.study_id.isin(chosen)])
    w = pd.concat(picks)[["study_id", "person_id", "split", "stratum", "dicom_uri"]]
    return _assign(w, readers, float(task["double_read_fraction"]), rng, int(task.get("max_readers", 5)))


def select_echo(studies: pd.DataFrame, readers: List[str], task: dict, seed: int = 20261014, n: Optional[int] = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = int(n or task["target_studies"])
    s = studies.sample(frac=1.0, random_state=seed).drop_duplicates("person_id").head(n)
    w = s[["study_id", "person_id", "dicom_uri"]].assign(split="qc", stratum="random")
    return _assign(w, readers, float(task["double_read_fraction"]), rng, int(task.get("max_readers", 5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["mri", "echo"])
    ap.add_argument("--manifest", required=True, help="MRI: build_manifest output; echo: CSV/parquet with study_id, person_id, dicom_uri")
    ap.add_argument("--readers", required=True, help="comma-separated reader codes (no names: codes go into file paths)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    tasks = load_tasks()
    path = Path(a.manifest)
    m = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    readers = [r.strip() for r in a.readers.split(",") if r.strip()]
    w = select_mri(m, readers, tasks["mri_contours"], n=a.n) if a.task == "mri" else select_echo(m, readers, tasks["echo_lv_traces"], n=a.n)
    w.to_csv(a.out, index=False)
    print(w.groupby(["split", "stratum"]).size().to_string())
    print(f"{len(w)} studies, {int((w.reader_2 != '').sum())} double-read")


if __name__ == "__main__":
    main()
