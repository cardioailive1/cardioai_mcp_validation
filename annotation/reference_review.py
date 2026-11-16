"""Phase 2 MRI reference review (sign-off sheet MRI 8, approved by the clinical lead 2026-11-16).

The Phase 2 MRI reference (ref_* columns of the locked cohort) is read from each CMR report by the training label
code. Before the cohort is scored, one cardiologist checks it on 100 reports:

  1. sheet : 100 reports drawn at random from the locked cohort (fixed seed), topped up so each label has at least
             10 positives where the cohort has them. The automatic auditor (annotation/auto_audit.py, written
             separately from the reference code) PROPOSES every label and the LVEF, with the sentences it used. The
             reference value is never shown. 20 of the 100 reports carry no proposal (blinded): the cardiologist
             fills those from the text, which measures how far the proposals sway the reading.
  2. the cardiologist works through review_sheet.xlsx in the workspace: decision = approve (the proposals are
     right) or correct (type the right value in the final_ columns for the labels that are wrong). Blinded rows:
     fill every final_ column.
  3. score : kappa per label between the reference and the cardiologist's final values (PASS at >= 0.80),
             blinded vs shown kappa (more than 0.10 lower on the blinded rows -> re-read all 100 blind), stated
             LVEF within 1 point in >= 98% of reports. A failing label: fix the rule in mri_training/config.yaml,
             rebuild the reference, re-lock the cohort, review a fresh sample (new --seed).

    python -m annotation.reference_review sheet --out ~/cardioai_exports/ref_review      # reads mri_analytic_locked
    python -m annotation.reference_review score --dir ~/cardioai_exports/ref_review

Everything stays in the workspace: the sheet holds report text. key.csv (reference values, blinded rows) is for
the scoring step only; the cardiologist does not open it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .auto_audit import AutoAuditor
from .label_audit import NM, U, cohen_kappa, norm_label, norm_value

# sheet label -> reference column in the MRI analytic table
LABELS: Dict[str, str] = {"infarct": "ref_infarct", "nonischemic_lge": "ref_nonischemic_lge", "report_mvo": "ref_mvo",
                          "lv_thrombus": "ref_lv_thrombus", "lv_aneurysm": "ref_lv_aneurysm"}
VALUE, VALUE_REF = "report_lvef", "lvef_report"
TEXT = "cmr_report"
DEFAULTS = {"n": 100, "n_blinded": 20, "min_positives": 10, "seed": 20261116, "kappa_pass": 0.80,
            "anchoring_max_drop": 0.10, "lvef_tolerance": 1.0, "lvef_match_pass": 0.98}
UNKNOWN = "unknown"                       # reference NaN and a reader's U / NM are the same class for kappa


def _cls(v) -> str:
    """1 / 0 / unknown for kappa."""
    if v is None or v in (U, NM) or (isinstance(v, float) and math.isnan(v)):
        return UNKNOWN
    return "1" if float(v) == 1.0 else "0"


def _fmt(v) -> object:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return int(v) if isinstance(v, float) and v in (0.0, 1.0) else v


# --------------------------------------------------------------------------- 1. sheet
def draw(analytic: pd.DataFrame, cfg: dict = DEFAULTS) -> pd.DataFrame:
    """The review sample: random n reports with a report text, topped up to min_positives per label; n_blinded of
    them blinded. Deterministic for a seed."""
    rng = np.random.default_rng(int(cfg["seed"]))
    pool = analytic[analytic[TEXT].notna() & (analytic[TEXT].astype(str).str.strip().str.len() > 0)].copy()
    pool = pool.sort_values("person_id").reset_index(drop=True)
    n = min(int(cfg["n"]), len(pool))
    pick = list(rng.choice(len(pool), n, replace=False))
    topped = {}
    for lab, ref in LABELS.items():
        have = int((pool.loc[pick, ref] == 1).sum())
        need = int(cfg["min_positives"]) - have
        if need > 0:
            extra = [i for i in rng.permutation(len(pool)) if pool.loc[i, ref] == 1 and i not in pick][:need]
            pick += extra
            topped[lab] = len(extra)
    s = pool.loc[pick].reset_index(drop=True)
    s["review_id"] = [f"R{i + 1:03d}" for i in range(len(s))]
    s["topped_up"] = [i >= n for i in range(len(s))]
    blind = set(rng.choice(len(s), min(int(cfg["n_blinded"]), len(s)), replace=False))
    s["blinded"] = [i in blind for i in range(len(s))]
    s.attrs["topped_up"] = topped
    return s


def build(analytic: pd.DataFrame, out: str, cfg: dict = DEFAULTS, auditor: Optional[AutoAuditor] = None) -> Dict:
    """Write review_sheet.xlsx (for the cardiologist) and key.csv (for scoring)."""
    aa = auditor or AutoAuditor("mri")
    s = draw(analytic, cfg)
    rows = []
    for _, r in s.iterrows():
        text = str(r[TEXT])
        prop = aa.read(text)
        row = {"review_id": r.review_id, "report": text, "evidence": aa.evidence(text),
               "blinded": "yes: fill every final_ column" if r.blinded else "", "decision": ""}
        for lab in list(LABELS) + [VALUE]:
            row[f"proposed_{lab}"] = "" if r.blinded else _fmt(prop.get(lab))
            row[f"final_{lab}"] = ""
        row["note"] = ""
        rows.append(row)
    sheet = pd.DataFrame(rows)
    key = s[["review_id", "person_id", "blinded", "topped_up"] + list(LABELS.values()) + [VALUE_REF]].copy()
    for lab in list(LABELS) + [VALUE]:
        key[f"proposed_{lab}"] = sheet[f"proposed_{lab}"].values
    d = Path(out).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    _write_xlsx(sheet, d / "review_sheet.xlsx")
    key.to_csv(d / "key.csv", index=False)
    meta = {**cfg, "reports": int(len(s)), "blinded": int(s.blinded.sum()), "topped_up": s.attrs["topped_up"],
            "auto_auditor": "annotation/auto_audit.py", "reference": "mri_training/reference.py via the locked cohort"}
    (d / "review_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


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
    fill = PatternFill("solid", fgColor="E5F4E4")
    for j, name in enumerate(cols, start=1):
        letter = ws.cell(1, j).column_letter
        ws.column_dimensions[letter].width = {"report": 80, "evidence": 50, "note": 30, "blinded": 18}.get(name, 14)
        for i in range(2, ws.max_row + 1):
            ws.cell(i, j).alignment = wrap
            if name.startswith("final_") or name == "decision":
                ws.cell(i, j).fill = fill
    last = ws.max_row
    dec = DataValidation(type="list", formula1='"approve,correct"', allow_blank=True)
    lab = DataValidation(type="list", formula1='"1,0,U,NM"', allow_blank=True)
    ws.add_data_validation(dec)
    ws.add_data_validation(lab)
    for j, name in enumerate(cols, start=1):
        letter = ws.cell(1, j).column_letter
        if name == "decision":
            dec.add(f"{letter}2:{letter}{last}")
        elif name.startswith("final_") and name != f"final_{VALUE}":
            lab.add(f"{letter}2:{letter}{last}")
    ws.freeze_panes = "C2"
    rules = wb.create_sheet("how to fill")
    for line in [
        "Read the report text only. Rule sheet: annotation/LABEL_AUDIT.md (CMR report), signed 2026-11-16.",
        "1 = present; 0 = stated absent (thrombus, aneurysm, MVO: also not mentioned in a study with LGE imaging);",
        "U = hedged, contradictory or unclear; NM = not mentioned (infarct / non-ischaemic: LGE not assessed).",
        "report_lvef: the LVEF % measured on this CMR; U for a range. Blinded rows: blank if none is stated.",
        "  To correct a proposed LVEF that the report does not state, type none.",
        "Rows with a proposal: decision = approve if every proposed value is right. Otherwise decision = correct",
        "  and type the right value in final_<label> for each wrong one (leave the others blank).",
        "Blinded rows (no proposal): fill every final_ column; decision can stay blank.",
    ]:
        rules.append([line])
    rules.column_dimensions["A"].width = 120
    wb.save(path)


# --------------------------------------------------------------------------- 2. score
def _txt(v) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v).strip()


NONE_WORDS = ("none", "no lvef", "not stated")      # final_report_lvef: the report states no LVEF


def final_values(sheet: pd.DataFrame) -> pd.DataFrame:
    """The cardiologist's final value per report and label (approved proposal, or the typed correction)."""
    out, problems = [], []
    for _, r in sheet.iterrows():
        rid, blind = r["review_id"], bool(_txt(r.get("blinded")))
        dec = _txt(r.get("decision")).lower()
        row = {"review_id": rid, "blinded": blind}
        for lab in list(LABELS) + [VALUE]:
            norm = norm_value if lab == VALUE else norm_label
            raw = r.get(f"final_{lab}")
            if lab == VALUE and _txt(raw).lower() in NONE_WORDS:
                row[lab] = None                           # correction: no LVEF stated
                continue
            typed = norm(raw)
            if typed is not None or (blind and lab == VALUE):
                row[lab] = typed                          # a blank LVEF on a blinded row = no LVEF stated
            elif blind:
                problems.append(f"{rid}: blinded row, final_{lab} empty")
                row[lab] = None
            elif dec in ("approve", "correct"):
                row[lab] = norm(r.get(f"proposed_{lab}"))  # not corrected -> the proposal stands
            else:
                problems.append(f"{rid}: decision empty")
                row[lab] = None
        out.append(row)
    f = pd.DataFrame(out)
    f.attrs["problems"] = sorted(set(problems))
    return f


def score(d: str, cfg: dict = DEFAULTS) -> Dict:
    d = Path(d).expanduser()
    sheet = pd.read_excel(d / "review_sheet.xlsx", sheet_name="review", dtype=object)
    key = pd.read_csv(d / "key.csv")
    f = final_values(sheet)
    if f.attrs["problems"]:
        raise SystemExit("review sheet not complete:\n  " + "\n  ".join(f.attrs["problems"][:30]))
    m = key.merge(f.drop(columns="blinded"), on="review_id")
    rows, dis = [], []
    for lab, ref in LABELS.items():
        a, b = [_cls(v) for v in m[ref]], [_cls(v) for v in m[lab]]
        k = cohen_kappa(a, b)
        agree = float(np.mean([x == y for x, y in zip(a, b)]))
        bl = m.blinded.astype(bool).to_numpy()
        kb = cohen_kappa([x for x, z in zip(a, bl) if z], [y for y, z in zip(b, bl) if z])
        ks = cohen_kappa([x for x, z in zip(a, bl) if not z], [y for y, z in zip(b, bl) if not z])
        # kappa is undefined when a column holds one class only; perfect agreement then counts as a pass
        ok = (k >= cfg["kappa_pass"]) if not math.isnan(k) else agree == 1.0
        anchored = (not math.isnan(kb) and not math.isnan(ks) and ks - kb > cfg["anchoring_max_drop"])
        rows.append({"label": lab, "reports": len(m), "positives_reference": int(sum(x == "1" for x in a)),
                     "agreement": round(agree, 3), "kappa": round(k, 3) if not math.isnan(k) else None,
                     "kappa_blinded": round(kb, 3) if not math.isnan(kb) else None,
                     "kappa_shown": round(ks, 3) if not math.isnan(ks) else None,
                     "result": "PASS" if ok and not anchored else
                               ("RE-READ BLIND: blinded kappa more than 0.10 below shown" if anchored else
                                "FAIL: fix the rule, rebuild the reference, re-lock, review a fresh sample")})
        dis += [{"review_id": r.review_id, "label": lab, "reference": x, "cardiologist": y}
                for (_, r), x, y in zip(m.iterrows(), a, b) if x != y]
    # stated LVEF: compared where either side states a number
    ev = []
    for _, r in m.iterrows():
        ref = r[VALUE_REF]
        ref = None if ref is None or (isinstance(ref, float) and math.isnan(ref)) else float(ref)
        car = r[VALUE]
        car = None if car is None or (isinstance(car, float) and math.isnan(car)) else car
        if ref is None and car is None:
            continue
        hit = isinstance(car, float) and ref is not None and abs(car - ref) <= cfg["lvef_tolerance"]
        ev.append(hit)
        if not hit:
            dis.append({"review_id": r.review_id, "label": VALUE, "reference": ref, "cardiologist": car})
    match = float(np.mean(ev)) if ev else float("nan")
    rows.append({"label": VALUE, "reports": len(ev), "agreement": round(match, 3) if ev else None,
                 "result": "PASS" if ev and match >= cfg["lvef_match_pass"] else ("FAIL: check lvef_regex" if ev else "no LVEF stated")})
    res = pd.DataFrame(rows)
    res.to_csv(d / "review_results.csv", index=False)
    pd.DataFrame(dis).to_csv(d / "disagreements.csv", index=False)
    return {"results": res, "disagreements": pd.DataFrame(dis), "passed": bool((res.result == "PASS").all())}


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
    a = ap.parse_args(argv)
    if a.cmd == "sheet":
        from cardioai_mcp.config import load_study
        from cardioai_mcp.engine import make_engine
        study = load_study()
        analytic = make_engine(a.engine, study).read_table(a.cohort)
        meta = build(analytic, a.out, {**DEFAULTS, "seed": a.seed})
        print(f"{meta['reports']} reports ({meta['blinded']} blinded; topped up {meta['topped_up'] or 'none'}): "
              f"{Path(a.out).expanduser() / 'review_sheet.xlsx'}")
    else:
        r = score(a.dir)
        print(r["results"].to_string(index=False))
        print("\nALL PASS: the reference stands" if r["passed"] else "\nNot passed: see review_results.csv and disagreements.csv")


if __name__ == "__main__":
    main()
