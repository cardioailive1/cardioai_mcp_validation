"""Audit the AUTOMATIC labels: data scientists first, a cardiologist for what they can't settle.

Protocol, rule sheets and pass rules: annotation/LABEL_AUDIT.md; numbers: tasks.yaml label_audit.
Everything stays in the MCP workspace (the sheets contain report text).

Report-derived labels (MRI report labels, ECG AF from confirmed reads)
    python -m annotation.label_audit sample mri --manifest /workspace/mri/manifest.parquet --auditors ds1,ds2 --out /workspace/audit/mri
        -> key.csv (automatic labels: NOT for the auditors), sheet_ds1.xlsx, sheet_ds2.xlsx (blinded),
           auto_audit_summary.csv (tier 0: the automatic auditor vs the code, per label; aggregate only)
        The automatic audit (tier 0, annotation/auto_audit.py) is REQUIRED for every agent and always runs first:
            --auto triage (default): it settles rows where it agrees with the code on every label; the data
                             scientists read the rest plus a blinded 25% of those rows to verify it
            --auto assist  : the fallback when triage FAILs for a label: every row is read by people, with the
                             auditor's evidence sentences on the sheets
    ... data scientists fill their sheets by the rule sheet (1 / 0 / U; numbers as stated) ...
    python -m annotation.label_audit score mri --dir /workspace/audit/mri
        -> adjudication_1.xlsx for the cardiologist (every disagreement, every U, plus a blinded spot-check)
    ... cardiologist fills adjudication_1.xlsx ...
    python -m annotation.label_audit score mri --dir /workspace/audit/mri
        -> audit_results.csv (PPV, NPV, Wilson CIs, kappa, value match, PASS / FAIL), errors.csv

ECG troponin / acute MI labels (recomputed independently from the raw lab rows)
    python -m annotation.label_audit troponin --manifest /workspace/ecg/manifest.parquet --out /workspace/audit/troponin

ECG AF cross-check from the waveform (rows whose rhythm disagrees with the label join the next text audit)
    python -m annotation.label_audit af-rhythm --manifest /workspace/ecg/manifest.parquet --out /workspace/audit/ecg/rr_flags.csv
    python -m annotation.label_audit sample ecg ... --extra /workspace/audit/ecg/rr_flags.csv

MRI outlines (model pre-drawn, or reader-corrected after annotation.collect)
    python -m annotation.label_audit outlines --root /workspace/annotation --manifest /workspace/mri/manifest.parquet --graders ds1,ds2 --out /workspace/audit/outlines
    python -m annotation.label_audit score-outlines --dir /workspace/audit/outlines
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .select import load_tasks

U = "U"
NM = "NM"      # the text does not mention it (the code leaves these labels unset / masked)


# --------------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.959964) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    a, b = [str(x) for x in a], [str(x) for x in b]
    if not a:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = float(np.mean([x == y for x, y in zip(a, b)]))
    pe = sum((a.count(c) / len(a)) * (b.count(c) / len(b)) for c in cats)
    return 1.0 if pe >= 1.0 else (po - pe) / (1 - pe)


# --------------------------------------------------------------------------- entries
def norm_label(v) -> Optional[object]:
    """Sheet cell -> 1.0 / 0.0 / 'U' / 'NM' / None (not filled)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip().upper()
    if s in ("1", "1.0", "Y", "YES"):
        return 1.0
    if s in ("0", "0.0", "N", "NO"):
        return 0.0
    if s in ("U", "?", "UNCERTAIN"):
        return U
    if s in ("NM", "NOT MENTIONED"):
        return NM
    if s == "":
        return None
    raise ValueError(f"label cell {v!r}: use 1, 0, U or NM")


def norm_value(v) -> Optional[object]:
    """Sheet cell -> float / 'U' / None (= the source states no value)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip().rstrip("%").strip()
    if s == "" or s == "-":
        return None
    if s.upper() in ("U", "?"):
        return U
    return float(s)


def _auto(v) -> Optional[float]:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def same(a, b, tol: Optional[float] = None) -> bool:
    if a is None or b is None or isinstance(a, str) or isinstance(b, str):
        return a == b
    return abs(float(a) - float(b)) <= tol if tol is not None else float(a) == float(b)


# --------------------------------------------------------------------------- sampling
def _hard_mask(text: pd.Series, keywords: str, hedge: str) -> pd.Series:
    pat = re.compile(f"(?is)(?:{hedge}).{{0,60}}(?:{keywords})|(?:{keywords}).{{0,40}}(?:{hedge})")
    return text.fillna("").map(lambda t: bool(pat.search(t)))


AUTO_MODES = ("triage", "assist")       # the automatic audit is required: there is no "off"


def sample(m: pd.DataFrame, modality: str, auditors: List[str], cfg: Optional[dict] = None,
           extra_ids: Iterable[str] = (), auto: str = "triage") -> pd.DataFrame:
    """Rows to audit, with why each was picked. Deterministic (seeded).

    The automatic auditor (tier 0) always reads every sampled row first; its reading is stored in key.csv.
    auto = "triage": it settles rows where it agrees with the code on EVERY label and value, except hard,
                     flagged and missing-value rows; data scientists read the rest plus a blinded
                     verify_fraction of the auto-settled rows (tasks.yaml label_audit.auto_audit).
           "assist": data scientists read every row, with the auditor's evidence sentences on their sheets
                     (the fallback when triage FAILs for a label).
    """
    if auto not in AUTO_MODES:
        raise ValueError(f"auto must be one of {AUTO_MODES}: the automatic audit is a required step before the "
                         "data scientists and the cardiologist (use assist to have people read every row)")
    cfg = cfg or load_tasks()["label_audit"]
    c = cfg[modality]
    if not auditors:
        raise ValueError("give at least one auditor code")
    rng = np.random.default_rng(cfg["seed"])
    idc, txt = c["id"], c["text"]
    m = m[m[txt].fillna("").str.strip().str.len() > 0].copy()
    m[idc] = m[idc].astype(str)
    m = m.drop_duplicates(idc).set_index(idc, drop=False)
    picked: Dict[str, List[str]] = {}

    def take(ids, n, why):
        ids = [i for i in ids]
        if n is not None and len(ids) > n:
            ids = list(rng.choice(ids, n, replace=False))
        for i in ids:
            picked.setdefault(str(i), []).append(why)

    take(m.index.tolist(), int(c["random"]), "random")
    # top-ups draw at random from ALL automatic positives of the label (a row already picked for another reason
    # is simply tagged again), so each label's positives remain a random sample for its PPV
    for lab in (c.get("labels") or {}):
        own = [i for i in picked if "random" in picked[i] and m.at[i, lab] == 1]
        pos = [i for i in m.index[m[lab] == 1] if "random" not in picked.get(i, [])]
        take(pos, max(0, int(c["min_positive"]) - len(own)), f"topup:{lab}")
    for lab, lc in (c.get("labels") or {}).items():
        hard = m.index[_hard_mask(m[txt], lc["keywords"], cfg["hedge"]).values]
        take(list(hard), int(c["hard"]), f"hard:{lab}")
    for val, vc in (c.get("values") or {}).items():
        kw_re = re.compile(vc["keywords"], re.I)
        kw = m[txt].fillna("").map(lambda t: bool(kw_re.search(t)))
        take(list(m.index[kw & m[val].isna()]), int(c["missing"]), f"missing:{val}")
    extra = [str(i) for i in extra_ids if str(i) in m.index and str(i) not in picked]
    take(extra, None, "flagged")

    ids = sorted(picked, key=lambda i: (picked[i][0] != "random", i))
    order = rng.permutation(len(ids))                       # auditors see rows in random order
    key = pd.DataFrame({"audit_id": [f"A{k + 1:05d}" for k in range(len(ids))], idc: [ids[k] for k in order]})
    key["picked"] = [";".join(picked[i]) for i in key[idc]]
    for col in list((c.get("labels") or {})) + list((c.get("values") or {})):
        key[f"auto_{col}"] = [m.at[i, col] for i in key[idc]]
    key["text"] = [m.at[i, txt] for i in key[idc]]
    labels, values = _columns(cfg, modality)
    key["auto_mode"] = auto
    key["auto_settled"] = False
    key["ds_verify"] = False
    if True:                                            # tier 0: always
        from .auto_audit import AutoAuditor
        aa = AutoAuditor(modality, cfg)
        reads = [aa.read(t) for t in key.text]
        for col in labels + values:
            key[f"aa_{col}"] = [r[col] for r in reads]
        key["evidence"] = [aa.evidence(t) for t in key.text]
        if auto == "triage":
            tol = cfg["pass"]["value_tolerance"]

            def agrees(r):
                if any(w.split(":")[0] in ("hard", "missing", "flagged") for w in r.picked.split(";")):
                    return False
                for col in labels:
                    code = _auto(r[f"auto_{col}"])
                    if not same(NM if code is None else code, norm_label(r[f"aa_{col}"])):
                        return False
                return all(same(_auto(r[f"auto_{v}"]), norm_value(r[f"aa_{v}"]), tol) for v in values)
            key["auto_settled"] = key.apply(agrees, axis=1)
            settled = np.flatnonzero(key.auto_settled.values)
            n_v = int(round(cfg["auto_audit"]["verify_fraction"] * len(settled)))
            if n_v:
                key.loc[rng.choice(settled, n_v, replace=False), "ds_verify"] = True
    to_ds = np.flatnonzero((~key.auto_settled | key.ds_verify).values)
    key["auditor_1"] = ""
    key["auditor_2"] = ""
    for j, k in enumerate(to_ds):
        key.at[k, "auditor_1"] = auditors[j % len(auditors)]
    if len(auditors) >= 2 and len(to_ds):
        n2 = int(round(cfg["second_auditor_fraction"] * len(to_ds)))
        for k in sorted(rng.choice(to_ds, n2, replace=False).tolist()):
            a1 = auditors.index(key.at[k, "auditor_1"])
            key.at[k, "auditor_2"] = auditors[(a1 + 1) % len(auditors)]
    key["cardiologist_round"] = 0
    key["spot_check"] = False
    return key


# --------------------------------------------------------------------------- sheets
def _columns(cfg: dict, modality: str):
    c = cfg[modality]
    return list((c.get("labels") or {})), list((c.get("values") or {}))


def rule_text(modality: str) -> List[tuple]:
    """(column, rule) pairs shown on the 'rules' tab. The full rule sheet is LABEL_AUDIT.md."""
    common = [("Every label", "Read the text only, never the images. 1 = the text says it is present; 0 = the text says "
                              "it is absent (or the rule below says silence means absent); NM = the text does not "
                              "mention it at all; U = uncertain, hedged, contradictory or you can't tell. Never guess: "
                              "U goes to the cardiologist."),
              ("Numbers", "Type the number exactly as stated (no %). Blank = no number stated. U = more than one "
                          "different number, a range, or you can't tell which is final.")]
    if modality == "mri":
        return common + [
            ("infarct", "1 = LGE/scar in an ischaemic pattern (subendocardial or transmural, coronary territory) or "
                        "'consistent with infarct' (prior, old, recent or acute all count). 0 = no LGE, or LGE only in "
                        "a non-ischaemic pattern. NM = LGE not assessed or not described."),
            ("nonischemic_lge", "1 = mid-wall, subepicardial, patchy or insertion-point LGE, or a named non-ischaemic "
                                "cause (myocarditis, sarcoid, amyloid, HCM). 0 = no LGE, or ischaemic LGE only. "
                                "NM = LGE not assessed or not described."),
            ("lv_thrombus", "1 = LV (apical, mural) thrombus present now. 0 = stated absent OR not mentioned (silence "
                            "= absent). 'Cannot exclude' / 'possible' -> U. Resolved thrombus -> 0."),
            ("lv_aneurysm", "1 = true LV aneurysm or pseudoaneurysm. 0 = stated absent OR not mentioned. 'Aneurysmal "
                            "segment' or dyskinesis without the word aneurysm -> U."),
            ("report_mvo", "1 = microvascular obstruction / no-reflow present. 0 = stated absent OR not mentioned "
                           "(silence = absent). 'Possible' / 'small area may represent' -> U."),
            ("report_lvef", "LVEF in % measured on THIS CMR (not echo, not a prior study). Range (e.g. 50-55) -> U."),
            ("report_infarct_size", "Infarct size / scar burden as % of LV mass. Grams only, or segment counts -> blank."),
        ]
    if modality == "echo":
        return common + [
            ("wma_any", "1 = any wall-motion abnormality on THIS echo: hypokinesis, akinesis, dyskinesis or aneurysm of any "
                        "segment, regional or global. 0 = 'normal wall motion' / 'no regional wall motion abnormalities'. "
                        "NM = wall motion not described. 'Possible' / 'cannot exclude' -> U."),
            ("terr_lad", "1 = an abnormal segment in the LAD territory: anterior, anteroseptal, apical anterior / septal or the "
                         "apex (AHA 1, 2, 7, 8, 13, 14, 17), or global hypokinesis. 0 = wall motion described and none of these "
                         "abnormal. 'Septal' without anterior/inferior counts for LAD and RCA."),
            ("terr_rca", "1 = an abnormal inferior or inferoseptal segment, or apical inferior (AHA 3, 4, 9, 10, 15), or global "
                         "hypokinesis. 0 = described and none abnormal."),
            ("terr_lcx", "1 = an abnormal inferolateral / posterior or anterolateral / lateral segment, or apical lateral "
                         "(AHA 5, 6, 11, 12, 16), or global hypokinesis. 0 = described and none abnormal."),
            ("global_hypokinesis", "1 = global / diffuse / generalised hypokinesis of the LV. 0 = wall motion described without "
                                   "it (regional abnormalities only, or normal). NM = not described."),
            ("report_lvef", "LVEF in % of THIS echo (biplane or visual). A range (e.g. 35-40) -> type the midpoint and note "
                            "'range'; a prior study's value -> blank."),
        ]
    return common + [
        ("af", "Confirmed (over-read) statement. 1 = atrial fibrillation or atrial flutter on THIS tracing. 0 = any "
               "other rhythm stated, including 'history of AF' with sinus rhythm now and 'no AF'. Paced rhythm with "
               "underlying AF -> 1. 'Possible AF' / 'consider AF' -> U."
               " No AF mentioned anywhere -> 0 (an over-read names AF when present)."),
    ]


def write_sheet(rows: pd.DataFrame, path: Path, labels: List[str], values: List[str], modality: str,
                title: str) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ws = wb.active
    ws.title = "audit"
    ev = "evidence" in rows.columns
    fixed = ["audit_id", "text"] + (["evidence"] if ev else [])
    head = fixed + labels + values + ["note"]
    ws.append(head)
    for _, r in rows.iterrows():
        t = str(r["text"])
        cells = [r["audit_id"], t[:32000] + (" [...truncated]" if len(t) > 32000 else "")]
        if ev:
            cells.append(r["evidence"] or "")
        ws.append(cells + [None] * (len(head) - len(fixed)))
    f0 = len(fixed) + 1                                 # first column to fill
    bold, fill = Font(bold=True), PatternFill("solid", fgColor="FFF2CC")
    for cell in ws[1]:
        cell.font = bold
    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 100
    if ev:
        ws.column_dimensions["C"].width = 60
    for k in range(f0, len(head) + 1):
        col = ws.cell(1, k).column_letter
        ws.column_dimensions[col].width = 14 if k < len(head) else 40
        for r in range(2, ws.max_row + 1):
            ws.cell(r, k).fill = fill
    for r in range(2, ws.max_row + 1):
        for k in range(2, f0):
            ws.cell(r, k).alignment = Alignment(wrap_text=True, vertical="top")
    if labels and ws.max_row > 1:
        dv = DataValidation(type="list", formula1='"1,0,U,NM"', allow_blank=True)
        ws.add_data_validation(dv)
        a, b = ws.cell(1, f0).column_letter, ws.cell(1, f0 - 1 + len(labels)).column_letter
        dv.add(f"{a}2:{b}{ws.max_row}")
    ws.freeze_panes = ws.cell(2, f0).coordinate
    rs = wb.create_sheet("rules")
    rs.append([title])
    rs.append([])
    for k, v in rule_text(modality):
        rs.append([k, v])
    if ev:
        rs.append(["evidence", "The sentences that mention any label's words, picked by the automatic auditor to save "
                               "reading time. They are NOT its answer. Still read the whole text when the evidence "
                               "is empty, cut short or unclear."])
    rs.column_dimensions["A"].width = 22
    rs.column_dimensions["B"].width = 120
    for row in rs.iter_rows(min_row=3):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(path)


def read_sheet(path: Path, labels: List[str], values: List[str]) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="audit", dtype=object) if path.suffix == ".xlsx" else pd.read_csv(path, dtype=object)
    df["audit_id"] = df["audit_id"].astype(str)
    out = pd.DataFrame({"audit_id": df.audit_id})
    for col in labels:
        out[col] = [norm_label(v) for v in df.get(col, pd.Series([None] * len(df)))]
    for col in values:
        out[col] = [norm_value(v) for v in df.get(col, pd.Series([None] * len(df)))]
    return out.set_index("audit_id")


def _find(d: Path, stem: str) -> Optional[Path]:
    for ext in (".xlsx", ".csv"):
        if (d / f"{stem}{ext}").exists():
            return d / f"{stem}{ext}"
    return None


def auto_audit_summary(key: pd.DataFrame, modality: str, cfg: Optional[dict] = None) -> pd.DataFrame:
    """Tier 0 result, per label / value: how often the automatic auditor agrees with the code on the sampled
    rows, how often it finds the text uncertain or silent, and (triage) how many rows it settled.
    Aggregate counts only: safe to share; the row-level readings stay in key.csv."""
    cfg = cfg or load_tasks()["label_audit"]
    labels, values = _columns(cfg, modality)
    tol = cfg["pass"]["value_tolerance"]
    rows = []
    for col in labels + values:
        is_val = col in values
        code = [_auto(v) for v in key[f"auto_{col}"]]
        aa = [(norm_value if is_val else norm_label)(v) for v in key[f"aa_{col}"]]
        if not is_val:
            code = [NM if c is None else c for c in code]
        agree = [same(c, a, tol if is_val else None) for c, a in zip(code, aa)]
        rows.append({"column": col, "kind": "value" if is_val else "label", "rows": len(key),
                     "auditor_agrees_with_code": int(sum(agree)),
                     "agreement": float(np.mean(agree)) if len(agree) else float("nan"),
                     "auditor_uncertain": int(sum(a == U for a in aa)),
                     "auditor_not_mentioned": int(sum(a == NM for a in aa)) if not is_val else None,
                     "auditor_differs": int(len(agree) - sum(agree)),
                     "auto_mode": key["auto_mode"].iloc[0] if len(key) else None,
                     "rows_settled_automatically": int(key.auto_settled.sum()),
                     "rows_read_by_people": int((key.auditor_1 != "").sum())})
    return pd.DataFrame(rows)


def write_auditor_sheets(key: pd.DataFrame, out: Path, modality: str, cfg: Optional[dict] = None) -> List[Path]:
    cfg = cfg or load_tasks()["label_audit"]
    labels, values = _columns(cfg, modality)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for a in sorted((set(key.auditor_1) | set(key.auditor_2)) - {""}):
        rows = key[(key.auditor_1 == a) | (key.auditor_2 == a)]
        rows = rows[["audit_id", "text"] + (["evidence"] if "evidence" in rows else [])]
        p = out / f"sheet_{a}.xlsx"
        if p.exists():
            raise FileExistsError(f"{p} exists: refusing to overwrite an auditor's work")
        write_sheet(rows, p, labels, values, modality, f"{modality.upper()} label audit - auditor {a} - {len(rows)} rows")
        paths.append(p)
    key.drop(columns=[c for c in ("text", "evidence") if c in key]).to_csv(out / "key.csv", index=False)
    auto_audit_summary(key, modality, cfg).to_csv(out / "auto_audit_summary.csv", index=False)
    key[["audit_id", "text"]].to_csv(out / "texts.csv.gz", index=False)
    (out / "README.txt").write_text(
        "key.csv holds the automatic labels (the code's and the automatic auditor's): do NOT open it or share it\n"
        "with the auditors or the cardiologist.\n"
        "Auditors fill sheet_<code>.xlsx using the 'rules' tab and annotation/LABEL_AUDIT.md.\n"
        "Then run: python -m annotation.label_audit score " + modality + " --dir " + str(out) + "\n")
    return paths


# --------------------------------------------------------------------------- scoring
def _load_entries(d: Path, key: pd.DataFrame, labels, values):
    sheets = {}
    for a in sorted((set(key.auditor_1) | set(key.auditor_2)) - {""}):
        p = _find(d, f"sheet_{a}")
        if p is None:
            raise FileNotFoundError(f"sheet for auditor {a} not found in {d}")
        sheets[a] = read_sheet(p, labels, values)
    cardio = {}
    for p in sorted(list(d.glob("adjudication_*.xlsx")) + list(d.glob("adjudication_*.csv"))):
        s = read_sheet(p, labels, values)
        for aid, r in s.iterrows():
            if any(r[c] is not None for c in labels) or any(r[c] is not None for c in values):
                cardio[aid] = r
    return sheets, cardio


def resolve(key: pd.DataFrame, sheets: dict, cardio: dict, labels, values, tol: float) -> pd.DataFrame:
    """One row per audit row x column: automatic value, data scientist entries, cardiologist, reference, status."""
    rows, unfilled = [], []
    for _, k in key.iterrows():
        aid = k.audit_id
        e1 = sheets[k.auditor_1].loc[aid] if k.auditor_1 else None
        e2 = sheets[k.auditor_2].loc[aid] if k.auditor_2 else None
        cr = cardio.get(aid)
        for col in labels + values:
            is_val = col in values
            auto = _auto(k[f"auto_{col}"])
            if not is_val and auto is None:
                auto = NM                               # label left unset by the code (masked in training)
            aa = k.get(f"aa_{col}")
            if e1 is None:                              # settled by the automatic auditor (triage), not read by a person
                c = cr[col] if cr is not None else None
                use_c = cr is not None and (c is not None or is_val)
                rows.append({"audit_id": aid, "column": col, "kind": "value" if is_val else "label", "picked": k.picked,
                             "auto": auto, "auto_auditor": aa, "ds1": None, "ds2": None,
                             "cardiologist": c if cr is not None else None, "reference": c if use_c else aa,
                             "status": "cardiologist" if use_c else "auto auditor", "reasons": "",
                             "spot_check": bool(k.spot_check), "ds_verify": False})
                continue
            d1 = e1[col]
            d2 = e2[col] if e2 is not None else None
            if not is_val and (d1 is None or (e2 is not None and d2 is None)):
                unfilled.append(f"{aid}:{col}")
            t = tol if is_val else None
            reasons = []
            if d1 == U or d2 == U:
                reasons.append("uncertain")
            if e2 is not None and not same(d1, d2, t):
                reasons.append("auditors differ")
            if auto is not None and auto != NM and not same(auto, d1, t):
                reasons.append("automatic differs")     # (unset automatic label vs a stated one: counted, not adjudicated)
            if is_val and auto is None and d1 is not None:
                reasons.append("automatic missed a value")
            c = cr[col] if cr is not None else None
            if cr is not None and (c is not None or is_val):
                ref, status = c, "cardiologist"
            elif reasons:
                ref, status = None, "needs cardiologist"
            else:
                ref, status = d1, "data scientists"
            rows.append({"audit_id": aid, "column": col, "kind": "value" if is_val else "label", "picked": k.picked,
                         "auto": auto, "auto_auditor": aa, "ds1": d1, "ds2": d2, "cardiologist": c if cr is not None else None,
                         "reference": ref, "status": status, "reasons": "; ".join(reasons),
                         "spot_check": bool(k.spot_check), "ds_verify": bool(k.get("ds_verify", False))})
    if unfilled:
        raise ValueError(f"{len(unfilled)} label cells not filled, e.g. {unfilled[:8]}")
    return pd.DataFrame(rows)


def _auto_checks(r: pd.DataFrame, done: pd.Series, tol: Optional[float]) -> Dict[str, float]:
    """How far the automatic auditor can be trusted: on the auto-settled rows the data scientists still read
    (blinded verification), and on auto-settled rows the cardiologist spot-checked."""
    if "auto_auditor" not in r or r.auto_auditor.isna().all():
        return {}
    ver = r[r.ds_verify & done]
    spot = r[r.spot_check & r.cardiologist.notna() & r.ds1.isna()]
    agree = [same(a, b, tol) for a, b in zip(ver.auto_auditor, ver.reference)]
    sp = [same(a, b, tol) for a, b in zip(spot.auto_auditor, spot.cardiologist)]
    read = r[done & r.ds1.notna()]
    allr = [same(a, b, tol) for a, b in zip(read.auto_auditor, read.reference)]
    return {"auto_settled": int((r.status == "auto auditor").sum() + len(spot)), "auto_verify_n": len(ver),
            "auto_vs_reference": float(np.mean(agree)) if agree else float("nan"),
            "auto_vs_cardiologist": float(np.mean(sp)) if sp else float("nan"), "auto_spot_n": len(sp),
            # every row people read (assist: all rows): how the auditor did; information, not a pass rule
            "auto_vs_reference_all": float(np.mean(allr)) if allr else float("nan"), "auto_read_n": len(allr)}


def _auto_fails(row: dict, cfg: dict) -> List[str]:
    f = []
    if row.get("auto_settled") and not row.get("auto_verify_n"):
        f.append("automatic auditor settled rows but none were verified")
    if row.get("auto_verify_n") and not row["auto_vs_reference"] >= cfg["auto_audit"]["verify_min"]:
        f.append(f"automatic auditor {row['auto_vs_reference']:.3f} on the verification rows: re-run with --auto assist")
    if row.get("auto_spot_n") and not row["auto_vs_cardiologist"] >= cfg["pass"]["ds_vs_cardiologist_min"]:
        f.append(f"automatic auditor vs cardiologist {row['auto_vs_cardiologist']:.2f}")
    return f


def metrics(res: pd.DataFrame, labels, values, cfg: dict) -> pd.DataFrame:
    p = cfg["pass"]
    out = []
    for col in labels:
        r = res[res.column == col]
        own = r.picked.str.split(";").map(lambda w: "random" in w or f"topup:{col}" in w)
        rand = r.picked.str.split(";").map(lambda w: "random" in w)
        done = r.status != "needs cardiologist"
        # U as the reference counts AGAINST the automatic label: training on an uncertain report is an error
        pos = r[own & done & (r.auto == 1.0)]
        neg = r[rand & done & (r.auto == 0.0)]
        kp, kn = int((pos.reference == 1.0).sum()), int((neg.reference == 0.0).sum())
        ppv = kp / len(pos) if len(pos) else float("nan")
        npv = kn / len(neg) if len(neg) else float("nan")
        both = r[r.ds2.notna()]
        kap = cohen_kappa(both.ds1.tolist(), both.ds2.tolist()) if len(both) else float("nan")
        spot = r[r.spot_check & r.cardiologist.notna() & r.ds1.notna()]
        dsc = float(np.mean([same(a, b) for a, b in zip(spot.ds1, spot.cardiologist)])) if len(spot) else float("nan")
        hard = r[r.picked.str.contains(f"hard:{col}") & done & r.auto.isin([0.0, 1.0])]
        missed = r[rand & done & (r.auto == NM) & r.reference.isin([0.0, 1.0])]
        row = {"column": col, "kind": "label", "rows": len(r), "pending_cardiologist": int((~done).sum()),
               "ppv": ppv, "ppv_n": len(pos), "ppv_ci_low": wilson(kp, len(pos))[0], "ppv_ci_high": wilson(kp, len(pos))[1],
               "npv": npv, "npv_n": len(neg), "npv_ci_low": wilson(kn, len(neg))[0], "npv_ci_high": wilson(kn, len(neg))[1],
               "uncertain_reference": int((r.reference == U).sum()),
               "labelled_but_not_mentioned": int(((r.reference == NM) & r.auto.isin([0.0, 1.0])).sum()),
               "hard_accuracy": float(np.mean([same(a, b) for a, b in zip(hard.auto, hard.reference)])) if len(hard) else float("nan"),
               "hard_n": len(hard), "masked_but_stated": len(missed),
               "kappa_ds1_ds2": kap, "kappa_n": len(both), "ds_vs_cardiologist": dsc, "spot_n": len(spot),
               **_auto_checks(r, done, None)}
        fails = []
        if row["pending_cardiologist"]:
            fails.append("adjudication pending")
        for m_ in ("ppv", "npv"):
            if row[f"{m_}_n"] == 0:
                fails.append(f"no rows for {m_.upper()}")
            elif not (row[m_] >= p[f"{m_}_min"] and row[f"{m_}_ci_low"] >= p["ci_lower_min"]):
                fails.append(f"{m_.upper()} {row[m_]:.3f} (CI low {row[f'{m_}_ci_low']:.3f})")
        if row["kappa_n"] and not kap >= p["kappa_min"]:
            fails.append(f"kappa {kap:.2f}")
        if row["spot_n"] and not dsc >= p["ds_vs_cardiologist_min"]:
            fails.append(f"data scientists vs cardiologist {dsc:.2f}")
        fails += _auto_fails(row, cfg)
        row["result"] = "PASS" if not fails else "FAIL: " + "; ".join(fails)
        out.append(row)
    tol = p["value_tolerance"]
    for col in values:
        r = res[res.column == col]
        done = r.status != "needs cardiologist"
        stated = r[done & r.auto.notna()]
        match = int(sum(same(a, b, tol) for a, b in zip(stated.auto, stated.reference)))
        mis = r[done & r.picked.str.contains(f"missing:{col}") & r.auto.isna()]
        n_missed = int(sum(isinstance(v, float) for v in mis.reference))
        row = {"column": col, "kind": "value", "rows": len(r), "pending_cardiologist": int((~done).sum()),
               "value_match": match / len(stated) if len(stated) else float("nan"), "value_n": len(stated),
               "value_miss_rate": n_missed / len(mis) if len(mis) else float("nan"), "miss_n": len(mis),
               **_auto_checks(r, done, tol)}
        fails = []
        if row["pending_cardiologist"]:
            fails.append("adjudication pending")
        if row["value_n"] and not row["value_match"] >= p["value_match_min"]:
            fails.append(f"match {row['value_match']:.3f}")
        if row["miss_n"] and not row["value_miss_rate"] <= p["value_miss_max"]:
            fails.append(f"missed {row['value_miss_rate']:.3f}")
        fails += _auto_fails(row, cfg)
        row["result"] = "PASS" if not fails else "FAIL: " + "; ".join(fails)
        out.append(row)
    return pd.DataFrame(out)


def score(d: str, modality: str, cfg: Optional[dict] = None, log=print) -> Dict[str, object]:
    cfg = cfg or load_tasks()["label_audit"]
    dp = Path(d)
    labels, values = _columns(cfg, modality)
    key = pd.read_csv(dp / "key.csv", dtype=str, keep_default_na=False)
    if not all(f"aa_{c}" in key for c in labels + values):
        raise ValueError("this sample has no automatic audit (tier 0), which is required before the data scientists "
                         "and the cardiologist: draw it again with --auto triage or --auto assist")
    for c in labels + values:
        key[f"auto_{c}"] = pd.to_numeric(key[f"auto_{c}"].replace("", np.nan))
        if f"aa_{c}" in key:
            key[f"aa_{c}"] = [(norm_value if c in values else norm_label)(v) for v in key[f"aa_{c}"]]
    for b in ("spot_check", "auto_settled", "ds_verify"):
        key[b] = key[b].astype(str).str.lower() == "true" if b in key else False
    key["cardiologist_round"] = key.cardiologist_round.astype(int)
    texts = pd.read_csv(dp / "texts.csv.gz", dtype=str, keep_default_na=False).set_index("audit_id")["text"]
    sheets, cardio = _load_entries(dp, key, labels, values)
    res = resolve(key, sheets, cardio, labels, values, cfg["pass"]["value_tolerance"])
    pending = set(res.audit_id[res.status == "needs cardiologist"])
    new = sorted(a for a in pending if int(key.set_index("audit_id").at[a, "cardiologist_round"]) == 0)
    sheet = None
    if new:
        rnd = int(key.cardiologist_round.max()) + 1
        rows = set(new)
        if rnd == 1:                                    # blinded spot-check of rows the data scientists settled
            settled = sorted(set(key.audit_id) - pending)
            n = int(round(cfg["cardiologist_spot_check_fraction"] * len(settled)))
            rng = np.random.default_rng(cfg["seed"] + 1)
            spot = set(rng.choice(settled, n, replace=False).tolist()) if n else set()
            key.loc[key.audit_id.isin(spot), "spot_check"] = True
            rows |= spot
        key.loc[key.audit_id.isin(rows), "cardiologist_round"] = rnd
        sel = key[key.audit_id.isin(rows)].sample(frac=1.0, random_state=cfg["seed"] + rnd)
        sel = sel.assign(text=[texts[a] for a in sel.audit_id])
        sheet = dp / f"adjudication_{rnd}.xlsx"
        write_sheet(sel, sheet, labels, values, modality,
                    f"{modality.upper()} label adjudication round {rnd} - cardiologist - {len(sel)} rows (fill every cell you can)")
        key.to_csv(dp / "key.csv", index=False)
        res = resolve(key, sheets, cardio, labels, values, cfg["pass"]["value_tolerance"])
        log(f"{len(new)} rows need the cardiologist (+ spot-check): {sheet}")
    met = metrics(res, labels, values, cfg)
    met.to_csv(dp / "audit_results.csv", index=False)
    wrong = np.array([not same(a, b, cfg["pass"]["value_tolerance"] if k == "value" else None)
                      for a, b, k in zip(res.auto, res.reference, res.kind)], dtype=bool)
    errs = res[(res.status != "needs cardiologist").values & res.auto.notna().values & (res.auto != NM).values & wrong]
    errs = errs.merge(key[["audit_id", cfg[modality]["id"]]], on="audit_id")
    errs.to_csv(dp / "errors.csv", index=False)
    res.to_csv(dp / "resolved.csv", index=False)
    return {"results": met, "resolved": res, "errors": errs, "adjudication_sheet": sheet}


# --------------------------------------------------------------------------- ECG: troponin recomputed
def troponin_rows_sql(modality_cfg: dict, ecg_ids: Iterable[str], days: int = 30) -> str:
    """Raw troponin rows +/- days around each ECG (rendered by the engine: {{cdm}}, {{cs:troponin}})."""
    e = modality_cfg["ecg"]
    c = e["columns"]
    ids = ", ".join("'" + str(i).replace("'", "''") + "'" for i in ecg_ids)
    t = e["table"].replace("{modality}", "{{modality}}")
    return (f"SELECT CAST(x.{c['ecg_id']} AS STRING) AS ecg_id, CAST(x.{c['acquisition_datetime']} AS DATE) AS ecg_date, "
            "p.gender_concept_id, m.measurement_date, m.value_as_number, m.unit_source_value, m.range_high, "
            "m.measurement_concept_id, m.measurement_source_concept_id "
            f"FROM {t} x JOIN {{{{cdm}}}}.person p ON p.person_id = CAST(x.{c['person_id']} AS BIGINT) "
            "JOIN {{cdm}}.measurement m ON m.person_id = p.person_id "
            "AND (m.measurement_concept_id IN {{cs:troponin}} OR m.measurement_source_concept_id IN {{cs:troponin}}) "
            f"WHERE CAST(x.{c['ecg_id']} AS STRING) IN ({ids}) "
            f"AND ABS(DATEDIFF(m.measurement_date, CAST(x.{c['acquisition_datetime']} AS DATE))) <= {int(days)}")


def troponin_concepts_sql() -> str:
    """What the troponin concept set really contains on the platform: for the data scientist and cardiologist to read."""
    return ("SELECT m.measurement_concept_id, c.concept_name, c.concept_code, m.unit_source_value, COUNT(*) AS n, "
            "PERCENTILE_APPROX(m.value_as_number, 0.5) AS median_value, MIN(m.range_high) AS min_range_high, "
            "MAX(m.range_high) AS max_range_high "
            "FROM {{cdm}}.measurement m LEFT JOIN {{cdm}}.concept c ON c.concept_id = m.measurement_concept_id "
            "WHERE m.measurement_concept_id IN {{cs:troponin}} OR m.measurement_source_concept_id IN {{cs:troponin}} "
            "GROUP BY 1, 2, 3, 4 ORDER BY n DESC")


def recompute_troponin(ecgs: pd.DataFrame, lab_rows: pd.DataFrame, ecg_cfg: dict) -> pd.DataFrame:
    """Independent (pandas) version of labels.sql trop_pos, then the acute_mi / stemi rules, compared row by row."""
    lab = ecg_cfg["labels"]
    t = lab["troponin_url_fallback"]
    lo, hi = lab["window_days"]
    r = lab_rows.copy()
    r["ecg_id"] = r.ecg_id.astype(str)
    r["days"] = (pd.to_datetime(r.measurement_date) - pd.to_datetime(r.ecg_date)).dt.days
    unit = r.unit_source_value.fillna("").str.strip().str.lower()
    fb = np.where(unit.isin(["ng/l", "pg/ml"]),
                  np.select([r.gender_concept_id == 8507, r.gender_concept_id == 8532],
                            [t.get("ng_per_l_male", t["ng_per_l"]), t.get("ng_per_l_female", t["ng_per_l"])], t["ng_per_l"]),
                  np.where(unit.isin(["ng/ml", "ug/l"]), t["ng_per_ml"], np.nan))
    limit = r.range_high.where(r.range_high.notna(), fb)
    r["above"] = (r.value_as_number > limit) & r.days.between(lo, hi)
    pos = r.groupby("ecg_id").above.any()
    e = ecgs.copy()
    e["ecg_id"] = e.ecg_id.astype(str)
    e["trop_pos_recomputed"] = e.ecg_id.map(pos).fillna(False).astype(int)
    mi = (e.mi_dx == 1) & (e.trop_pos_recomputed == 1)
    e["acute_mi_recomputed"] = np.where(mi, 1.0, np.where(e.mi_any_near == 0, 0.0, np.nan))
    e["stemi_recomputed"] = np.where(mi & (e.stemi_dx == 1), 1.0,
                                     np.where((e.mi_any_near == 0) | (mi & (e.nstemi_dx == 1) & (e.stemi_dx == 0)), 0.0, np.nan))
    for a, b in (("trop_pos", "trop_pos_recomputed"), ("acute_mi", "acute_mi_recomputed"), ("stemi", "stemi_recomputed")):
        e[f"{a}_match"] = [same(_auto(x), _auto(y)) for x, y in zip(e[a], e[b])]
    units = r.assign(unit=unit).groupby("unit").size().rename("rows").reset_index()
    e.attrs["units"] = units
    return e


def troponin_sample(m: pd.DataFrame, cfg: dict, n_each: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"])
    parts = []
    for v in (1.0, 0.0):
        g = m[m.acute_mi == v]
        parts.append(g.iloc[rng.choice(len(g), min(n_each, len(g)), replace=False)] if len(g) else g)
    return pd.concat(parts)


# --------------------------------------------------------------------------- ECG: rhythm cross-check for AF
def rr_irregularity(sig: np.ndarray, fs: float, leads: Optional[List[str]] = None) -> Dict[str, float]:
    """R peaks on lead II (or the first lead) -> RR coefficient of variation. Simple and transparent, not a diagnosis."""
    from scipy.signal import butter, filtfilt, find_peaks
    li = leads.index("II") if leads and "II" in leads else (1 if sig.shape[0] > 1 else 0)
    x = np.asarray(sig[li], float)
    b, a = butter(2, [5 / (fs / 2), min(25 / (fs / 2), 0.99)], btype="band")
    y = filtfilt(b, a, x)
    e = np.convolve(np.diff(y) ** 2, np.ones(int(0.08 * fs)) / int(0.08 * fs), mode="same")
    pk, _ = find_peaks(e, distance=int(0.28 * fs), height=0.3 * np.percentile(e, 99))
    rr = np.diff(pk) / fs
    if len(rr) < 4:
        return {"beats": float(len(pk)), "rr_cv": float("nan"), "heart_rate": float("nan")}
    return {"beats": float(len(pk)), "rr_cv": float(np.std(rr) / np.mean(rr)), "heart_rate": float(60 / np.mean(rr))}


def af_rhythm_flags(m: pd.DataFrame, cfg: dict, reader=None) -> pd.DataFrame:
    from ecg_training.io import read_any
    reader = reader or read_any
    c = cfg["ecg"]
    rows = []
    for _, r in m[m.waveform_uri.notna() & m.af.notna()].iterrows():
        try:
            sig, fs, leads, _ = reader(r.waveform_uri)
            q = rr_irregularity(sig, fs, leads)
        except Exception as e:  # noqa: BLE001
            q = {"beats": float("nan"), "rr_cv": float("nan"), "heart_rate": float("nan"), "error": str(e)[:120]}
        flag = ""
        if r.af == 1 and q["rr_cv"] < c["rr_regular_cv"]:
            flag = "AF label but regular RR"
        elif r.af == 0 and q["rr_cv"] > c["rr_irregular_cv"]:
            flag = "no AF label but irregular RR"
        rows.append({"ecg_id": str(r.ecg_id), "af": r.af, **q, "flag": flag})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- MRI outlines
def outline_measures(labels: Dict[str, np.ndarray]) -> Dict[str, float]:
    from mri_training.runtime.quantify import LV, MVO, MYO, RV, SCAR
    lge = labels["lge"]
    myo = np.isin(lge, (MYO, SCAR, MVO))
    scar = np.isin(lge, (SCAR, MVO))
    r = {"lv_slices": int(sum(myo[i].any() for i in range(lge.shape[0]))),
         "infarct_pct": 100.0 * scar.sum() / myo.sum() if myo.sum() else float("nan"),
         "mvo_pct": 100.0 * (lge == MVO).sum() / myo.sum() if myo.sum() else float("nan"),
         "rv_drawn": bool((lge == RV).any())}
    if "cine" in labels:
        ed, es = (labels["cine"][0] == LV).sum(), (labels["cine"][1] == LV).sum()
        r["lvef"] = 100.0 * (ed - es) / ed if ed else float("nan")
    else:
        r["lvef"] = float("nan")
    return r


def outline_flags(meas: Dict[str, float], report: Optional[pd.Series], oc: dict) -> List[str]:
    f = []
    if meas["lv_slices"] < oc["min_lv_slices"]:
        f.append(f"LV on {meas['lv_slices']} slices")
    if not math.isnan(meas["lvef"]) and not (oc["lvef_range"][0] <= meas["lvef"] <= oc["lvef_range"][1]):
        f.append(f"LVEF {meas['lvef']:.0f}% outside {oc['lvef_range']}")
    if not math.isnan(meas["infarct_pct"]) and meas["infarct_pct"] > oc["infarct_pct_max"]:
        f.append(f"infarct {meas['infarct_pct']:.0f}% of LV")
    if not meas["rv_drawn"]:
        f.append("no RV")
    if report is not None:
        ef = _auto(report.get("report_lvef"))
        if ef is not None and not math.isnan(meas["lvef"]) and abs(meas["lvef"] - ef) > oc["lvef_report_diff_max"]:
            f.append(f"LVEF {meas['lvef']:.0f}% vs report {ef:.0f}%")
        size = _auto(report.get("report_infarct_size"))
        if size is not None and not math.isnan(meas["infarct_pct"]) and abs(meas["infarct_pct"] - size) > oc["infarct_size_report_diff_max"]:
            f.append(f"infarct {meas['infarct_pct']:.0f}% vs report {size:.0f}%")
        inf = _auto(report.get("infarct"))
        has_scar = meas["infarct_pct"] > 0.5 if not math.isnan(meas["infarct_pct"]) else False
        if inf == 1.0 and not has_scar:
            f.append("report says infarct, no scar drawn")
        if inf == 0.0 and _auto(report.get("nonischemic_lge")) == 0.0 and has_scar:
            f.append("report says no LGE, scar drawn")
        mvo = _auto(report.get("report_mvo"))
        if mvo == 1.0 and not meas["mvo_pct"] > 0:
            f.append("report says MVO, none drawn")
        if mvo == 0.0 and meas["mvo_pct"] > 0:
            f.append("report says no MVO, MVO drawn")
    return f


GRADE_COLUMNS = ["grade", "ring_follows_wall", "no_blood_in_myocardium", "scar_bright_and_in_wall",
                 "mvo_dark_core_in_scar", "base_apex_ok", "rv_ok", "ed_es_phases_ok", "note"]


def outlines(root: str, out: str, graders: List[str], manifest: Optional[pd.DataFrame] = None,
             cfg: Optional[dict] = None, allow_unchanged: bool = True) -> pd.DataFrame:
    """Automatic checks on every exported/annotated study folder + grading sheets for the data scientists."""
    from openpyxl import Workbook
    from openpyxl.worksheet.datavalidation import DataValidation

    from .collect import collect_mri_folder
    cfg = cfg or load_tasks()["label_audit"]
    oc = cfg["outlines"]
    rep = (manifest.assign(study_id=manifest.study_id.astype(str)).drop_duplicates("study_id").set_index("study_id")
           if manifest is not None else None)
    rows = []
    for meta in sorted(Path(root).glob("*/*/meta.json")):
        folder = meta.parent
        labels, err, warn = collect_mri_folder(folder, allow_unchanged=allow_unchanged)
        r = {"study_id": folder.name, "source": folder.parent.name, "folder": str(folder)}
        if labels is None:
            r.update(auto_status="rejected", flags="; ".join(err + warn))
        else:
            meas = outline_measures(labels)
            fl = warn + outline_flags(meas, rep.loc[folder.name] if rep is not None and folder.name in rep.index else None, oc)
            r.update(meas, auto_status="flagged" if fl else "ok", flags="; ".join(fl))
        rows.append(r)
    t = pd.DataFrame(rows)
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"] + 7)
    t = t.iloc[rng.permutation(len(t))].reset_index(drop=True)
    t["grader_1"] = [graders[k % len(graders)] for k in range(len(t))]
    t["grader_2"] = ""
    if len(graders) >= 2 and len(t):
        n2 = int(round(oc["double_grade_fraction"] * len(t)))
        for k in rng.choice(len(t), n2, replace=False).tolist():
            t.at[k, "grader_2"] = graders[(graders.index(t.at[k, "grader_1"]) + 1) % len(graders)]
    t.to_csv(outp / "outline_qc.csv", index=False)
    for g in graders:
        p = outp / f"grades_{g}.xlsx"
        if p.exists():
            raise FileExistsError(f"{p} exists: refusing to overwrite a grader's work")
        sel = t[(t.grader_1 == g) | (t.grader_2 == g)]
        wb = Workbook()
        ws = wb.active
        ws.title = "grades"
        ws.append(["study_id", "source", "folder", "automatic_flags"] + GRADE_COLUMNS)
        for _, r in sel.iterrows():
            ws.append([r["study_id"], r["source"], r["folder"], r["flags"]] + [None] * len(GRADE_COLUMNS))
        if len(sel):
            dv = DataValidation(type="list", formula1='"A,B,C"', allow_blank=True)
            yn = DataValidation(type="list", formula1='"Y,N,NA"', allow_blank=True)
            ws.add_data_validation(dv)
            ws.add_data_validation(yn)
            dv.add(f"E2:E{ws.max_row}")
            yn.add(f"F2:L{ws.max_row}")
        ws.column_dimensions["C"].width = 60
        ws.column_dimensions["D"].width = 60
        ws.freeze_panes = "E2"
        wb.save(p)
    return t


def _read_grades(p: Path) -> pd.DataFrame:
    df = pd.read_excel(p, sheet_name="grades", dtype=object) if p.suffix == ".xlsx" else pd.read_csv(p, dtype=object)
    df["study_id"] = df.study_id.astype(str)
    df["grade"] = df.grade.map(lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v).strip().upper())
    bad = df[~df.grade.isin(["A", "B", "C"])]
    return df, bad


def score_outlines(d: str, cfg: Optional[dict] = None, log=print) -> Dict[str, object]:
    """Grades -> what happens to each outline; writes the cardiologist's review list (all C, grader
    disagreements, every automatic flag graded A, and a blinded spot-check of A)."""
    cfg = cfg or load_tasks()["label_audit"]
    oc = cfg["outlines"]
    dp = Path(d)
    t = pd.read_csv(dp / "outline_qc.csv", dtype={"study_id": str, "grader_2": str}, keep_default_na=False)
    g = {}
    for p in list(dp.glob("grades_*.xlsx")) + list(dp.glob("grades_*.csv")):
        df, bad = _read_grades(p)
        if len(bad):
            raise ValueError(f"{p.name}: {len(bad)} studies without grade A/B/C, e.g. {bad.study_id.head(5).tolist()}")
        g[p.stem[len("grades_"):]] = df.set_index("study_id")
    t["grade_1"] = [g[r.grader_1].at[r.study_id, "grade"] for _, r in t.iterrows()]
    t["grade_2"] = [g[r.grader_2].at[r.study_id, "grade"] if r.grader_2 else "" for _, r in t.iterrows()]
    both = t[t.grader_2 != ""]
    kap = cohen_kappa(both.grade_1.tolist(), both.grade_2.tolist()) if len(both) else float("nan")
    worst = t.apply(lambda r: max(r.grade_1, r.grade_2 or "A"), axis=1)
    t["grade"] = worst
    rng = np.random.default_rng(cfg["seed"] + 9)
    a_ids = t.study_id[(t.grade == "A") & (t.auto_status == "ok")].tolist()
    n = int(round(oc["cardiologist_spot_check_A"] * len(a_ids)))
    spot = set(rng.choice(a_ids, n, replace=False).tolist()) if n else set()
    why = []
    for _, r in t.iterrows():
        w = []
        if r.grade == "C":
            w.append("graded C")
        if r.grade_2 and r.grade_1 != r.grade_2:
            w.append("graders differ")
        if r.grade == "A" and r.auto_status != "ok":
            w.append("graded A despite automatic flags")
        if r.study_id in spot:
            w.append("spot-check")
        why.append("; ".join(w))
    t["cardiologist_review"] = why
    t["action"] = np.select([t.auto_status == "rejected", t.grade == "C", t.grade == "B"],
                            ["redraw (failed automatic checks)", "redraw / full correction by reader", "minor correction by reader"],
                            "usable as is (after any cardiologist review)")
    t.to_csv(dp / "outline_grades.csv", index=False)
    review = t[t.cardiologist_review != ""][["study_id", "source", "folder", "flags"]].sample(frac=1.0, random_state=cfg["seed"])
    review = review.assign(**{c: "" for c in ["cardiologist_grade", "note"]})
    review.to_csv(dp / "cardiologist_review.csv", index=False)
    summ = {"studies": len(t), "A": int((t.grade == "A").sum()), "B": int((t.grade == "B").sum()),
            "C": int((t.grade == "C").sum()), "auto_rejected": int((t.auto_status == "rejected").sum()),
            "kappa_graders": kap, "double_graded": len(both), "cardiologist_review": len(review)}
    (dp / "outline_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    log(json.dumps(summ, default=float))
    return {"grades": t, "review": review, "summary": summ}


# --------------------------------------------------------------------------- CLI
def _read_table(p: str) -> pd.DataFrame:
    return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("modality", choices=["mri", "ecg", "echo"])
    s.add_argument("--manifest", required=True)
    s.add_argument("--auditors", required=True, help="data scientist codes, comma-separated (no names)")
    s.add_argument("--out", required=True)
    s.add_argument("--extra", help="CSV with an id column (e.g. rr_flags.csv rows with a flag) to add as 'flagged'")
    s.add_argument("--auto", choices=AUTO_MODES, default="triage",
                   help="the automatic audit (tier 0) always runs first. triage (default) = it settles rows where it "
                        "agrees with the code, a blinded share still verified by the data scientists; assist = people "
                        "read every row with its evidence sentences (the fallback when triage FAILs for a label)")
    sc = sub.add_parser("score")
    sc.add_argument("modality", choices=["mri", "ecg", "echo"])
    sc.add_argument("--dir", required=True)
    tr = sub.add_parser("troponin")
    tr.add_argument("--manifest", required=True)
    tr.add_argument("--out", required=True)
    tr.add_argument("--n", type=int, default=100, help="ECGs per acute_mi value (1 and 0)")
    af = sub.add_parser("af-rhythm")
    af.add_argument("--manifest", required=True)
    af.add_argument("--out", required=True)
    ol = sub.add_parser("outlines")
    ol.add_argument("--root", required=True, help="annotation.export --out folder (pre-drawn or corrected)")
    ol.add_argument("--graders", required=True)
    ol.add_argument("--out", required=True)
    ol.add_argument("--manifest", help="MRI manifest with the report labels (for the report comparison)")
    so = sub.add_parser("score-outlines")
    so.add_argument("--dir", required=True)
    a = ap.parse_args(argv)
    cfg = load_tasks()["label_audit"]

    if a.cmd == "sample":
        extra = []
        if a.extra:
            x = pd.read_csv(a.extra, dtype=str)
            extra = x[x.get("flag", pd.Series(["x"] * len(x))).fillna("") != ""][cfg[a.modality]["id"]].tolist()
        key = sample(_read_table(a.manifest), a.modality, [c.strip() for c in a.auditors.split(",") if c.strip()], cfg, extra,
                     auto=a.auto)
        paths = write_auditor_sheets(key, Path(a.out), a.modality, cfg)
        print(key.picked.str.split(";").explode().str.split(":").str[0].value_counts().to_string())
        summ = auto_audit_summary(key, a.modality, cfg)
        print("tier 0, automatic auditor vs the code:")
        print(summ[["column", "rows", "agreement", "auditor_uncertain", "auditor_differs"]].to_string(index=False))
        n_ds = int((key.auditor_1 != "").sum())
        print(f"{len(key)} rows; data scientists read {n_ds} ({int(key.ds_verify.sum())} of them verify the automatic "
              f"auditor); {len(key) - n_ds} settled automatically; sheets: {', '.join(p.name for p in paths)}")
    elif a.cmd == "score":
        r = score(a.dir, a.modality, cfg)
        cols = [c for c in ["column", "ppv", "ppv_n", "ppv_ci_low", "npv", "npv_n", "npv_ci_low", "kappa_ds1_ds2",
                            "value_match", "value_miss_rate", "auto_vs_reference", "auto_vs_reference_all",
                            "pending_cardiologist", "result"]
                if c in r["results"]]
        print(r["results"][cols].to_string(index=False))
    elif a.cmd == "troponin":
        from cardioai_mcp.config import load_study
        from cardioai_mcp.engine import make_engine
        from ecg_training.dataset import load_config
        study = load_study()
        eng = make_engine("bigquery", study)
        m = _read_table(a.manifest)
        smp = troponin_sample(m, cfg, a.n)
        rows = eng.query(eng.render(troponin_rows_sql(study["modality"], smp.ecg_id.astype(str)), {}))
        res = recompute_troponin(smp, rows, load_config())
        outp = Path(a.out)
        outp.mkdir(parents=True, exist_ok=True)
        res.drop(columns=[c for c in ("read_text", "waveform_uri") if c in res]).to_csv(outp / "troponin_check.csv", index=False)
        res.attrs["units"].to_csv(outp / "troponin_units.csv", index=False)
        eng.query(eng.render(troponin_concepts_sql(), {})).to_csv(outp / "troponin_concepts.csv", index=False)
        for c in ("trop_pos", "acute_mi", "stemi"):
            ok = res[f"{c}_match"].mean()
            print(f"{c}: {ok:.3f} match  {'PASS' if ok >= cfg['pass']['troponin_match_min'] else 'FAIL'}")
        print(f"units: {res.attrs['units'].to_dict('records')}; concept list: {outp / 'troponin_concepts.csv'} (cardiologist reads it)")
    elif a.cmd == "af-rhythm":
        fl = af_rhythm_flags(_read_table(a.manifest), cfg)
        fl.to_csv(a.out, index=False)
        print(fl.flag.replace("", "no flag").value_counts().to_string())
    elif a.cmd == "outlines":
        t = outlines(a.root, a.out, [g.strip() for g in a.graders.split(",") if g.strip()],
                     _read_table(a.manifest) if a.manifest else None, cfg)
        print(t.auto_status.value_counts().to_string())
    elif a.cmd == "score-outlines":
        score_outlines(a.dir, cfg)


if __name__ == "__main__":
    main()
