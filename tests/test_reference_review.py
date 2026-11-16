"""Phase 2 MRI reference review (MRI 8, approved 2026-11-16): 100 reports, automatic proposals the cardiologist
approves or corrects, 20 blinded, kappa per label against the reference, anchoring check, LVEF match."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
pytest.importorskip("openpyxl")

from annotation import reference_review as R  # noqa: E402
from mri_training.dataset import load_config, report_labels  # noqa: E402
from test_label_audit import T  # noqa: E402  (report templates + what a careful reader enters)

REF = {"infarct": "ref_infarct", "nonischemic_lge": "ref_nonischemic_lge", "report_mvo": "ref_mvo",
       "lv_thrombus": "ref_lv_thrombus", "lv_aneurysm": "ref_lv_aneurysm"}


def _cohort(n=1000, seed=0):
    """A locked-cohort stand-in whose ref_* come from the real reference code (report_labels)."""
    rnd = np.random.default_rng(seed)
    kinds = rnd.choice(list(T), n, p=[0.25, 0.35, 0.10, 0.05, 0.05, 0.20])
    cfg = load_config()
    rows = []
    for i, k in enumerate(kinds):
        lab = report_labels(T[k][0], cfg)
        rows.append({"person_id": i, "cmr_report": T[k][0], "kind": k, **{REF[x]: lab[x] for x in REF},
                     "lvef_report": lab["report_lvef"]})
    return pd.DataFrame(rows)


def _truth(kind, col):
    v = T[kind][1].get(col)
    if col == "report_lvef":
        return v if v is not None else "none"
    return "NM" if v is None else v


def _norm(col, v):
    if col == "report_lvef":
        return None if v == "none" else R.norm_value(v)
    return R.norm_label(v)


def _fill(d, cohort, blind_wrong=False):
    """The cardiologist: approve right proposals, correct wrong ones, fill blinded rows from the text."""
    sheet = pd.read_excel(d / "review_sheet.xlsx", sheet_name="review", dtype=object)
    key = pd.read_csv(d / "key.csv")
    kind = dict(zip(key.review_id, cohort.set_index("person_id").loc[key.person_id, "kind"]))
    for i, r in sheet.iterrows():
        k = kind[r.review_id]
        cols = list(REF) + ["report_lvef"]
        if isinstance(r.blinded, str) and r.blinded:
            for c in cols:
                v = _truth(k, c)
                if blind_wrong and c == "infarct" and v in (0, 1):
                    v = 1 - v
                sheet.loc[i, f"final_{c}"] = "" if v == "none" else v
            continue
        wrong = [c for c in cols if _norm(c, r[f"proposed_{c}"]) != _norm(c, _truth(k, c))]
        sheet.loc[i, "decision"] = "correct" if wrong else "approve"
        for c in wrong:
            sheet.loc[i, f"final_{c}"] = _truth(k, c)
    with pd.ExcelWriter(d / "review_sheet.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        sheet.to_excel(w, sheet_name="review", index=False)


def test_draw_is_seeded_blinded_and_topped_up():
    c = _cohort()
    a, b = R.draw(c), R.draw(c)
    assert a.person_id.tolist() == b.person_id.tolist()
    assert len(a) >= 100 and a.blinded.sum() == 20
    for ref in REF.values():
        assert (a[ref] == 1).sum() >= min(10, int((c[ref] == 1).sum()))
    assert R.draw(c, {**R.DEFAULTS, "seed": 1}).person_id.tolist() != a.person_id.tolist()


def test_sheet_hides_the_reference_and_blinded_rows_have_no_proposal(tmp_path):
    c = _cohort()
    meta = R.build(c, str(tmp_path))
    sheet = pd.read_excel(tmp_path / "review_sheet.xlsx", sheet_name="review", dtype=object)
    assert meta["blinded"] == 20 and len(sheet) == meta["reports"]
    assert not any(col.startswith("ref_") or col == "lvef_report" for col in sheet.columns)
    blind = sheet[sheet.blinded.notna()]
    assert len(blind) == 20 and blind.filter(like="proposed_").isna().all().all()
    assert sheet[sheet.blinded.isna()].proposed_infarct.notna().all()


def test_score_passes_when_the_reference_matches_the_cardiologist(tmp_path):
    c = _cohort()
    R.build(c, str(tmp_path))
    _fill(tmp_path, c)
    r = R.score(str(tmp_path))
    res = r["results"].set_index("label")
    assert r["passed"], res.to_string()
    assert res.loc["infarct", "kappa"] == 1.0 and res.loc["report_lvef", "agreement"] == 1.0
    assert (tmp_path / "review_results.csv").exists()


def test_anchoring_on_the_blinded_rows_asks_for_a_blind_reread(tmp_path):
    c = _cohort()
    R.build(c, str(tmp_path))
    _fill(tmp_path, c, blind_wrong=True)
    res = R.score(str(tmp_path))["results"].set_index("label")
    assert res.loc["infarct", "result"].startswith("RE-READ")


def test_incomplete_sheet_is_refused(tmp_path):
    c = _cohort()
    R.build(c, str(tmp_path))
    with pytest.raises(SystemExit, match="not complete"):
        R.score(str(tmp_path))
