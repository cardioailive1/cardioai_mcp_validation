"""Phase 2 MRI reference review: the cardiologist approves or rejects, with notes, and signs off.

Sign-off sheet MRI 8 (approved 2026-11-16), amended as MRI 8a (2026-11-17, pending the clinical lead): the
cardiologist no longer types values. Each report is shown with the reference values the cohort will be scored
against; the cardiologist approves the row or rejects it with a note, then signs the sign_off tab.

  1. sheet : 110 reports from the locked cohort (fixed seed), topped up so each label has at least 10 positives.
             Every row shows the report, the evidence sentences and the reference values (present / absent /
             unknown, LVEF). The automatic auditor (annotation/auto_audit.py, separate code) checks each row: where it
             reads the report differently, the row says so, so those rows get a closer look.
             10 rows carry one planted error (a flipped label or an LVEF off by 15 points) and say "auditor agrees".
             The cardiologist knows some rows are planted, not which: catching them shows the review is not
             simply agreeing with what is shown.
  2. the cardiologist reviews inside the workspace, either way:
       - review_app.html (no Excel needed): open it in the workspace web browser; one report at a time with the
         evidence highlighted, Approve / Reject buttons, a note on rejects, then a sign-off screen that saves
         review_decisions.json (usually to ~/Downloads). Progress is kept in the browser between sessions.
       - review_sheet.xlsx: decision = approve or reject (+ note), then the sign_off tab.
  3. score : PASS when >= 95% of the 100 genuine reports are approved, at least 8 of the 10 planted errors are
             rejected, and the sign_off tab is filled and approved. Rejected genuine reports are listed with their
             notes (rejected_reports.csv) to fix the rule in mri_training/config.yaml, rebuild the reference,
             re-lock and review a fresh sample (--seed). Fewer than 8 planted errors caught: review again.

    python -m annotation.reference_review sheet --out ~/cardioai_exports/ref_review      # reads mri_analytic_locked
    python -m annotation.reference_review score --dir ~/cardioai_exports/ref_review [--decisions ~/Downloads/review_decisions.json]

Everything stays in the workspace: the sheet holds report text. key.csv (true values, planted rows) is for the
scoring step only; the cardiologist does not open it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .auto_audit import AutoAuditor
from .label_audit import NM, U
from .review_page import file_id, read_decisions, write_page

# sheet label -> reference column in the MRI analytic table
LABELS: Dict[str, str] = {"infarct": "ref_infarct", "nonischemic_lge": "ref_nonischemic_lge", "report_mvo": "ref_mvo",
                          "lv_thrombus": "ref_lv_thrombus", "lv_aneurysm": "ref_lv_aneurysm"}
VALUE, VALUE_REF = "report_lvef", "lvef_report"
TEXT = "cmr_report"
SHOW = {"infarct": "infarct", "nonischemic_lge": "non-ischaemic LGE", "report_mvo": "MVO", "lv_thrombus": "LV thrombus",
        "lv_aneurysm": "LV aneurysm", "report_lvef": "LVEF %"}
DEFAULTS = {"n": 100, "n_planted": 10, "min_positives": 10, "seed": 20261116, "approve_pass": 0.95,
            "planted_caught_pass": 8, "lvef_shift": 15.0, "lvef_tolerance": 1.0}


def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _word(v) -> str:
    """Reference / auditor label -> the word on the sheet."""
    if _isnan(v) or v in (U, NM):
        return "unknown"
    return "present" if float(v) == 1.0 else "absent"


def _lvef(v) -> str:
    return "not stated" if _isnan(v) or v in (U, NM) else f"{float(v):g}"


def _txt(v) -> str:
    return "" if _isnan(v) else str(v).strip()


# --------------------------------------------------------------------------- 1. sheet
def draw(analytic: pd.DataFrame, cfg: dict = DEFAULTS) -> pd.DataFrame:
    """n genuine + n_planted reports with a report text, topped up to min_positives per label. Seeded."""
    rng = np.random.default_rng(int(cfg["seed"]))
    pool = analytic[analytic[TEXT].notna() & (analytic[TEXT].astype(str).str.strip().str.len() > 0)].copy()
    pool = pool.sort_values("person_id").reset_index(drop=True)
    n = min(int(cfg["n"]) + int(cfg["n_planted"]), len(pool))
    pick = list(rng.choice(len(pool), n, replace=False))
    topped = {}
    for lab, ref in LABELS.items():
        need = int(cfg["min_positives"]) - int((pool.loc[pick, ref] == 1).sum())
        if need > 0:
            extra = [i for i in rng.permutation(len(pool)) if pool.loc[i, ref] == 1 and i not in pick][:need]
            pick += extra
            topped[lab] = len(extra)
    s = pool.loc[pick].sample(frac=1.0, random_state=int(cfg["seed"])).reset_index(drop=True)   # planted rows mixed in
    s["review_id"] = [f"R{i + 1:03d}" for i in range(len(s))]
    s.attrs["topped_up"] = topped
    return s


def _plant(row: pd.Series, rng: np.random.Generator, cfg: dict):
    """(label, shown value) of one planted error: flip a stated label, or move a stated LVEF."""
    choices = [lab for lab, ref in LABELS.items() if row[ref] in (0, 1, 0.0, 1.0)]
    if not _isnan(row[VALUE_REF]):
        choices.append(VALUE)
    if not choices:
        return None, None
    lab = choices[int(rng.integers(len(choices)))]
    if lab == VALUE:
        v = float(row[VALUE_REF])
        return lab, v - cfg["lvef_shift"] if v > 40 else v + cfg["lvef_shift"]
    return lab, 1.0 - float(row[LABELS[lab]])


def build(analytic: pd.DataFrame, out: str, cfg: dict = DEFAULTS, auditor: Optional[AutoAuditor] = None) -> Dict:
    """Write review_sheet.xlsx (the cardiologist) and key.csv (scoring only)."""
    aa = auditor or AutoAuditor("mri")
    s = draw(analytic, cfg)
    rng = np.random.default_rng(int(cfg["seed"]) + 1)
    planted: Dict[str, tuple] = {}
    for i in rng.permutation(len(s)):
        if len(planted) >= int(cfg["n_planted"]):
            break
        lab, v = _plant(s.loc[i], rng, cfg)
        if lab:
            planted[s.loc[i, "review_id"]] = (lab, v)
    rows, keys = [], []
    for _, r in s.iterrows():
        text = str(r[TEXT])
        shown = {lab: r[ref] for lab, ref in LABELS.items()}
        shown[VALUE] = r[VALUE_REF]
        p_lab, p_val = planted.get(r.review_id, (None, None))
        if p_lab:
            shown[p_lab] = p_val
        auto = aa.read(text)
        diffs = []
        for lab in LABELS:
            if _word(auto.get(lab)) != _word(shown[lab]):
                diffs.append(f"{SHOW[lab]} {_word(auto.get(lab))}")
        a_ef, s_ef = auto.get(VALUE), shown[VALUE]
        if _lvef(a_ef) != _lvef(s_ef) and not (isinstance(a_ef, float) and not _isnan(s_ef)
                                                and abs(a_ef - float(s_ef)) <= cfg["lvef_tolerance"]):
            diffs.append(f"LVEF {_lvef(a_ef)}")
        check = "auditor agrees" if (p_lab or not diffs) else "CHECK - auditor reads: " + "; ".join(diffs)
        row = {"review_id": r.review_id, "report": text, "evidence": aa.evidence(text)}
        row.update({SHOW[lab]: _word(shown[lab]) for lab in LABELS})
        row[SHOW[VALUE]] = _lvef(shown[VALUE])
        row.update({"auditor_check": check, "decision": "", "note": ""})
        rows.append(row)
        keys.append({"review_id": r.review_id, "person_id": r.person_id, "planted_label": p_lab or "",
                     "planted_shown": "" if p_lab is None else (_lvef(p_val) if p_lab == VALUE else _word(p_val)),
                     "auditor_flag": check.startswith("CHECK")})
    sheet, key = pd.DataFrame(rows), pd.DataFrame(keys)
    d = Path(out).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    _write_xlsx(sheet, d / "review_sheet.xlsx")
    key.to_csv(d / "key.csv", index=False)
    write_app(sheet, d / "review_app.html", _sheet_id(d))
    meta = {**cfg, "reports": int(len(s)), "planted": len(planted), "auditor_flagged": int(key.auditor_flag.sum()),
            "topped_up": s.attrs["topped_up"], "auto_auditor": "annotation/auto_audit.py",
            "reference": "mri_training/reference.py via the locked cohort", "protocol": "MRI 8a (approve / reject + notes)"}
    (d / "review_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def _sheet_id(d: Path) -> str:
    """Ties a decisions file to the sheet it was made from (hash of key.csv)."""
    return file_id(d / "key.csv")


def write_app(sheet: pd.DataFrame, path: Path, sheet_id: str) -> None:
    """review_app.html: approve / reject in a web browser (annotation/review_page.py)."""
    labels = [SHOW[x] for x in list(LABELS) + [VALUE]]
    rows = [{"review_id": r.review_id, "report": r.report, "evidence": [x for x in str(r.evidence).split("\n") if x.strip()],
             "values": [[lab, str(r[lab])] for lab in labels], "check": r.auditor_check} for _, r in sheet.iterrows()]
    write_page(path, "Phase 2 MRI reference review (MRI 8a)", "MRI 8a", sheet_id, rows)


def _write_xlsx(sheet: pd.DataFrame, path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ws = wb.active
    ws.title = "review"
    cols = list(sheet.columns)
    ws.append(cols)
    for _, r in sheet.iterrows():
        ws.append([r[c] for c in cols])
    for c in ws[1]:
        c.font = Font(bold=True)
    wrap = Alignment(wrap_text=True, vertical="top")
    green, amber = PatternFill("solid", fgColor="E5F4E4"), PatternFill("solid", fgColor="F9EEDA")
    for j, name in enumerate(cols, start=1):
        ws.column_dimensions[ws.cell(1, j).column_letter].width = {"report": 80, "evidence": 45, "note": 40,
                                                                  "auditor_check": 32}.get(name, 13)
        for i in range(2, ws.max_row + 1):
            cell = ws.cell(i, j)
            cell.alignment = wrap
            if name in ("decision", "note"):
                cell.fill = green
            elif name == "auditor_check" and str(cell.value).startswith("CHECK"):
                cell.fill = amber
    dec = DataValidation(type="list", formula1='"approve,reject"', allow_blank=True)
    ws.add_data_validation(dec)
    col = ws.cell(1, cols.index("decision") + 1).column_letter
    dec.add(f"{col}2:{col}{ws.max_row}")
    ws.freeze_panes = "C2"

    so = wb.create_sheet("sign_off")
    for r in (["Cardiologist (name)", ""], ["Date (YYYY-MM-DD)", ""], ["Decision", ""], ["Comments", ""]):
        so.append(r)
    so.column_dimensions["A"].width, so.column_dimensions["B"].width = 24, 60
    for i in range(1, 5):
        so.cell(i, 1).font = Font(bold=True)
        so.cell(i, 2).fill = green
    sdec = DataValidation(type="list", formula1='"approve,reject"', allow_blank=True)
    so.add_data_validation(sdec)
    sdec.add("B3")

    how = wb.create_sheet("how to fill")
    for line in [
        "Each row is one CMR report with the reference values the Phase 2 MRI cohort will be scored against.",
        "Rule sheet: annotation/LABEL_AUDIT.md (CMR report), signed 2026-11-16. present / absent / unknown;",
        "  thrombus, aneurysm and MVO are absent when not mentioned in a study with LGE imaging, unknown otherwise.",
        "decision = approve when every value in the row matches the report.",
        "decision = reject when any value is wrong, and write in note what is wrong (e.g. 'MVO present, not absent').",
        "auditor_check: amber rows are where a second, independent reading differs: look at those closely.",
        "Some rows contain a planted error even though they say 'auditor agrees'. Reject them like any other error.",
        "When every row is done: sign_off tab, your name, the date, decision approve (or reject, with comments).",
        "Save as .xlsx with the same name in the same folder. Do not open key.csv.",
    ]:
        how.append([line])
    how.column_dimensions["A"].width = 120
    wb.save(path)


# --------------------------------------------------------------------------- 2. score
def read_signoff(path: Path) -> Dict[str, str]:
    from openpyxl import load_workbook
    ws = load_workbook(path, data_only=True)["sign_off"]
    vals = [_txt(ws.cell(i, 2).value) for i in range(1, 5)]
    return dict(zip(["cardiologist", "date", "decision", "comments"], vals))


def apply_decisions(sheet: pd.DataFrame, path: Path, sheet_id: str):
    """Decisions saved by review_app.html -> the sheet's decision / note columns, and the sign-off."""
    rec = read_decisions(path, sheet_id)
    dec = rec["decisions"]
    sheet = sheet.copy()
    sheet["decision"] = sheet.review_id.map(lambda r: dec.get(r, {}).get("decision", ""))
    sheet["note"] = sheet.review_id.map(lambda r: dec.get(r, {}).get("note", ""))
    return sheet, rec["signoff"]


def score(d: str, cfg: dict = DEFAULTS, decisions: Optional[str] = None) -> Dict:
    d = Path(d).expanduser()
    sheet = pd.read_excel(d / "review_sheet.xlsx", sheet_name="review", dtype=object)
    key = pd.read_csv(d / "key.csv", dtype={"planted_label": str}).fillna({"planted_label": ""})
    if decisions is None and (d / "review_decisions.json").exists():
        decisions = str(d / "review_decisions.json")
    if decisions:
        sheet, sign = apply_decisions(sheet, Path(decisions), _sheet_id(d))
    else:
        sign = read_signoff(d / "review_sheet.xlsx")
    m = sheet.merge(key, on="review_id")
    m["decision"] = m.decision.map(lambda v: _txt(v).lower())
    m["note"] = m.note.map(_txt)
    problems = [f"{r.review_id}: decision empty" for _, r in m.iterrows() if r.decision not in ("approve", "reject")]
    problems += [f"{r.review_id}: rejected without a note" for _, r in m.iterrows() if r.decision == "reject" and not r.note]
    if not sign["cardiologist"] or not sign["date"] or sign["decision"].lower() not in ("approve", "reject"):
        problems.append("sign_off tab: name, date and decision (approve / reject) are needed")
    if problems:
        raise SystemExit("review sheet not complete:\n  " + "\n  ".join(problems[:30]))

    gen, pla = m[m.planted_label == ""], m[m.planted_label != ""]
    approved = float((gen.decision == "approve").mean()) if len(gen) else float("nan")
    flagged = gen[gen.auditor_flag.astype(bool)]
    caught = int((pla.decision == "reject").sum())
    rows = [
        {"check": "genuine reports approved", "value": f"{int((gen.decision == 'approve').sum())} / {len(gen)} ({approved:.0%})",
         "goal": f">= {cfg['approve_pass']:.0%}", "result": "PASS" if approved >= cfg["approve_pass"] else
         "FAIL: fix the rule (see rejected_reports.csv), rebuild the reference, re-lock, review a fresh sample"},
        {"check": "planted errors rejected", "value": f"{caught} / {len(pla)}", "goal": f">= {cfg['planted_caught_pass']}",
         "result": "PASS" if caught >= min(int(cfg["planted_caught_pass"]), len(pla)) else
         "REVIEW AGAIN: too many planted errors approved"},
        {"check": "auditor-flagged genuine reports approved", "value": f"{int((flagged.decision == 'approve').sum())} / {len(flagged)}",
         "goal": "for information", "result": ""},
        {"check": "cardiologist sign-off", "value": f"{sign['cardiologist']}, {sign['date']}: {sign['decision']}",
         "goal": "approve", "result": "PASS" if sign["decision"].lower() == "approve" else "NOT SIGNED: see comments"},
    ]
    res = pd.DataFrame(rows)
    passed = bool((res.result.isin(["PASS", ""])).all())
    res.to_csv(d / "review_results.csv", index=False)
    rej = gen[gen.decision == "reject"][["review_id", "note", "auditor_check"] + [SHOW[x] for x in list(LABELS) + [VALUE]]]
    rej.to_csv(d / "rejected_reports.csv", index=False)
    record = {"protocol": "MRI 8a", "decisions_from": decisions or "review_sheet.xlsx", "scored_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "sign_off": sign, "passed": passed, "results": rows, "rejected_genuine": int(len(rej))}
    (d / "signoff.json").write_text(json.dumps(record, indent=1))
    return {"results": res, "rejected": rej, "passed": passed, "sign_off": sign}


# --------------------------------------------------------------------------- CLI
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("--cohort", default="mri_analytic_locked", help="locked MRI analytic table (work dataset)")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=DEFAULTS["seed"], help="new seed for a fresh sample after a rule fix")
    s.add_argument("--engine", choices=["bigquery", "spark"], default="bigquery")
    sc = sub.add_parser("score")
    sc.add_argument("--dir", required=True)
    sc.add_argument("--decisions", default=None, help="review_decisions.json saved by review_app.html (e.g. ~/Downloads/...)")
    a = ap.parse_args(argv)
    if a.cmd == "sheet":
        from cardioai_mcp.config import load_study
        from cardioai_mcp.engine import make_engine
        study = load_study()
        analytic = make_engine(a.engine, study).read_table(a.cohort)
        meta = build(analytic, a.out, {**DEFAULTS, "seed": a.seed})
        print(f"{meta['reports']} reports ({meta['planted']} planted, {meta['auditor_flagged']} flagged by the auditor; "
              f"topped up {meta['topped_up'] or 'none'})")
        print(f"  open in the workspace web browser: {Path(a.out).expanduser() / 'review_app.html'}")
        print(f"  or the spreadsheet:                {Path(a.out).expanduser() / 'review_sheet.xlsx'}")
    else:
        r = score(a.dir, decisions=a.decisions)
        print(r["results"].to_string(index=False))
        print("\nSIGNED AND PASSED: the Phase 2 MRI reference stands (signoff.json)" if r["passed"]
              else "\nNot passed: see review_results.csv and rejected_reports.csv")


if __name__ == "__main__":
    main()
