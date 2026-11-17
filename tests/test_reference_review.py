"""Phase 2 MRI reference review (MRI 8a): the cardiologist approves or rejects each report's reference values with a
note and signs off; 10 planted errors check the review is not simply agreeing; the auditor flags rows it reads
differently."""
import json
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


SHOW = R.SHOW


def _truth_row(kind):
    t = T[kind][1]
    row = {SHOW[lab]: R._word(t.get(lab)) for lab in R.LABELS}
    row[SHOW["report_lvef"]] = R._lvef(t.get("report_lvef"))
    return row


def _review(d, cohort, approve_all=False, sign=True):
    """The cardiologist: approve a row when every value matches the report, else reject with a note."""
    from openpyxl import load_workbook
    key = pd.read_csv(d / "key.csv")
    kind = dict(zip(key.review_id, cohort.set_index("person_id").loc[key.person_id, "kind"]))
    wb = load_workbook(d / "review_sheet.xlsx")
    ws = wb["review"]
    cols = [c.value for c in ws[1]]
    for i in range(2, ws.max_row + 1):
        rid = ws.cell(i, cols.index("review_id") + 1).value
        truth = _truth_row(kind[rid])
        wrong = [c for c, v in truth.items() if str(ws.cell(i, cols.index(c) + 1).value) != v]
        ok = approve_all or not wrong
        ws.cell(i, cols.index("decision") + 1).value = "approve" if ok else "reject"
        if not ok:
            ws.cell(i, cols.index("note") + 1).value = "wrong: " + ", ".join(wrong)
    if sign:
        so = wb["sign_off"]
        so["B1"], so["B2"], so["B3"] = "Tamanna Nahar", "2026-11-17", "approve"
    wb.save(d / "review_sheet.xlsx")


def test_draw_is_seeded_and_topped_up():
    c = _cohort()
    a, b = R.draw(c), R.draw(c)
    assert a.person_id.tolist() == b.person_id.tolist() and len(a) >= 110
    for ref in REF.values():
        assert (a[ref] == 1).sum() >= min(10, int((c[ref] == 1).sum()))
    assert R.draw(c, {**R.DEFAULTS, "seed": 1}).person_id.tolist() != a.person_id.tolist()


def test_sheet_shows_values_and_plants_ten_errors(tmp_path):
    c = _cohort()
    meta = R.build(c, str(tmp_path))
    sheet = pd.read_excel(tmp_path / "review_sheet.xlsx", sheet_name="review", dtype=object)
    key = pd.read_csv(tmp_path / "key.csv")
    assert meta["planted"] == 10 and (key.planted_label.notna()).sum() == 10 and len(sheet) == meta["reports"]
    assert set(sheet["infarct"]) <= {"present", "absent", "unknown"}
    planted = sheet[sheet.review_id.isin(key.loc[key.planted_label.notna(), "review_id"])]
    assert (planted.auditor_check == "auditor agrees").all()            # planted rows look like any other
    assert "planted_label" not in sheet.columns and "sign_off" in pd.ExcelFile(tmp_path / "review_sheet.xlsx").sheet_names


def test_careful_review_passes_and_writes_the_signoff(tmp_path):
    c = _cohort()
    R.build(c, str(tmp_path))
    _review(tmp_path, c)
    r = R.score(str(tmp_path))
    assert r["passed"], r["results"].to_string()
    rec = json.loads((tmp_path / "signoff.json").read_text())
    assert rec["sign_off"]["cardiologist"] == "Tamanna Nahar" and rec["passed"] and rec["protocol"] == "MRI 8a"


def test_approving_everything_is_caught_by_the_planted_errors(tmp_path):
    c = _cohort()
    R.build(c, str(tmp_path))
    _review(tmp_path, c, approve_all=True)
    res = R.score(str(tmp_path))["results"].set_index("check")
    assert res.loc["planted errors rejected", "result"].startswith("REVIEW AGAIN")


def test_incomplete_or_unsigned_sheet_is_refused(tmp_path):
    c = _cohort()
    R.build(c, str(tmp_path))
    with pytest.raises(SystemExit, match="not complete"):
        R.score(str(tmp_path))
    _review(tmp_path, c, sign=False)
    with pytest.raises(SystemExit, match="sign_off"):
        R.score(str(tmp_path))


def test_browser_page_and_its_decisions_file(tmp_path):
    """review_app.html (no Excel) embeds the same rows; the decisions file it saves is scored like the sheet."""
    import re as _re
    c = _cohort()
    R.build(c, str(tmp_path))
    html = (tmp_path / "review_app.html").read_text()
    data = json.loads(_re.search(r'<script id="data" type="application/json">(.*?)</script>', html, _re.S).group(1))
    sheet = pd.read_excel(tmp_path / "review_sheet.xlsx", sheet_name="review", dtype=object)
    assert [r["review_id"] for r in data["rows"]] == sheet.review_id.tolist() and "__DATA__" not in html
    assert "planted" not in html.lower().replace("planted error even", "")   # nothing marks the planted rows
    key = pd.read_csv(tmp_path / "key.csv")
    kind = dict(zip(key.review_id, c.set_index("person_id").loc[key.person_id, "kind"]))
    dec = []
    for r in data["rows"]:
        truth = _truth_row(kind[r["review_id"]])
        wrong = [lab for lab, v in r["values"] if truth[lab] != v]
        dec.append({"review_id": r["review_id"], "decision": "reject" if wrong else "approve",
                    "note": ", ".join(wrong)})
    out = {"sheet_id": data["sheet_id"], "signoff": {"cardiologist": "Tamanna Nahar", "date": "2026-11-17",
                                                    "decision": "approve", "comments": ""}, "decisions": dec}
    p = tmp_path / "dl" / "review_decisions.json"
    p.parent.mkdir()
    p.write_text(json.dumps(out))
    r = R.score(str(tmp_path), decisions=str(p))
    assert r["passed"], r["results"].to_string()
    p.write_text(json.dumps({**out, "sheet_id": "other"}))
    with pytest.raises(SystemExit, match="different review sheet"):
        R.score(str(tmp_path), decisions=str(p))
