"""ECG 4: clinical review of the acute MI and STEMI training labels (proposed on the sign-off sheet 2026-11-17).

The acute_mi and stemi labels come from diagnosis codes and troponin (ecg_training/labels.sql); the label audit only
recomputes them, which proves the code follows the signed rule, not that the labels are clinically right. Here the
cardiologist reads each ECG's timeline and approves or rejects the labels, in a web browser (no Excel):

  1. sheet : 150 ECGs from the training manifest (fixed seed): 50 acute_mi = 1, 50 stemi = 1, 25 acute_mi = 0,
             25 masked (acute_mi not used), plus 10 extra ECGs with one planted error (a flipped label). Each ECG
             gets a one-page timeline: ECG date, over-read text, every MI diagnosis code within 30 days (code,
             name, day from the ECG, type 2 marked) and every troponin value (T or I, value, unit, limit, day, above
             or not). No names or record numbers. Writes ecg_mi_review.html (+ key.csv, timelines.csv).
  2. the cardiologist opens ecg_mi_review.html in the workspace web browser: Approve / Reject (+ note) per ECG,
     then Finish and sign -> review_decisions.json (usually ~/Downloads).
  3. score : PASS when >= 95% are approved in each labelled group (acute MI positive, STEMI positive, negative), at
             least 8 of the 10 planted errors are rejected, and the sign-off is approved. Masked ECGs: the share
             approved is reported (how often masking was right), no pass rule. Rejected ECGs are listed with notes.

    python -m annotation.ecg_mi_review sheet --manifest /workspace/ecg/manifest.parquet --out ~/cardioai_exports/ecg_mi_review
    python -m annotation.ecg_mi_review score --dir ~/cardioai_exports/ecg_mi_review --decisions ~/Downloads/review_decisions.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .review_page import file_id, read_decisions, write_page

GROUPS = {"acute_mi_positive": 50, "stemi_positive": 50, "negative": 25, "masked": 25}
DEFAULTS = {"seed": 20261117, "n_planted": 10, "approve_pass": 0.95, "planted_caught_pass": 8, "days": 30}
CONCEPT_SETS = {"mi_acute", "mi_any", "mi_type2", "stemi", "nstemi", "troponin", "troponin_t"}


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _word(v) -> str:
    return "not used (masked)" if _nan(v) else ("yes" if float(v) == 1.0 else "no")


# --------------------------------------------------------------------------- data
def conditions_sql(modality_cfg: dict, ecg_ids: Iterable[str], days: int = 30) -> str:
    """MI diagnosis rows +/- days around each ECG, with the source code and what each code counts for."""
    e = modality_cfg["ecg"]
    c = e["columns"]
    ids = ", ".join("'" + str(i).replace("'", "''") + "'" for i in ecg_ids)
    t = e["table"].replace("{modality}", "{{modality}}").replace("{work}", "{{work}}")

    def inset(s):
        return (f"CASE WHEN co.condition_concept_id IN {{{{cs:{s}}}}} OR co.condition_source_concept_id IN {{{{cs:{s}}}}} "
                f"THEN 1 ELSE 0 END AS is_{s}")
    return (f"SELECT CAST(x.{c['ecg_id']} AS STRING) AS ecg_id, CAST(x.{c['acquisition_datetime']} AS DATE) AS ecg_date, "
            "co.condition_start_date, sc.vocabulary_id, sc.concept_code, sc.concept_name, "
            f"{inset('mi_acute')}, {inset('stemi')}, {inset('nstemi')}, "
            "CASE WHEN co.condition_source_concept_id IN {{cs:mi_type2}} THEN 1 ELSE 0 END AS is_type2 "
            f"FROM {t} x JOIN {{{{cdm}}}}.condition_occurrence co ON co.person_id = CAST(x.{c['person_id']} AS BIGINT) "
            "LEFT JOIN {{vocab}}.concept sc ON sc.concept_id = co.condition_source_concept_id "
            "WHERE (co.condition_concept_id IN {{cs:mi_any}} OR co.condition_source_concept_id IN {{cs:mi_any}}) "
            f"AND CAST(x.{c['ecg_id']} AS STRING) IN ({ids}) "
            f"AND ABS(DATEDIFF(co.condition_start_date, CAST(x.{c['acquisition_datetime']} AS DATE))) <= {int(days)}")


def _limit(r: pd.Series, lab: dict) -> Optional[float]:
    """The limit the label rule used: the row's own range_high, else (troponin T only) Mayo's sex-specific limit."""
    if not _nan(r.get("range_high")):
        return float(r["range_high"])
    if int(r.get("is_tnt", 0) or 0) != 1:
        return None
    t = lab["troponin_url_fallback"]
    unit = str(r.get("unit_source_value") or "").strip().lower()
    if unit in ("ng/l", "pg/ml"):
        g = r.get("gender_concept_id")
        return float(t.get("ng_per_l_male", t["ng_per_l"]) if g == 8507 else
                     t.get("ng_per_l_female", t["ng_per_l"]) if g == 8532 else t["ng_per_l"])
    if unit in ("ng/ml", "ug/l"):
        return float(t["ng_per_ml"])
    return None


def timeline(ecg: pd.Series, conds: pd.DataFrame, trops: pd.DataFrame, lab: dict) -> Dict[str, object]:
    """The one-page timeline text for one ECG, and the lines to highlight (inside the rule window)."""
    lo, hi = lab["window_days"]
    day0 = pd.Timestamp(ecg["ecg_date"])
    lines = [f"ECG date: {day0.date()}  (day 0)", "", "Over-read:", f"  {str(ecg.get('read_text') or '(none)').strip()}", "",
             f"MI diagnosis codes, day -30 to +30 (the label rule uses day {lo} to +{hi}):"]
    hl = []
    if len(conds):
        c = conds.assign(day=(pd.to_datetime(conds.condition_start_date) - day0).dt.days).sort_values("day")
        for _, r in c.iterrows():
            tags = [t for t, f in (("TYPE 2: not counted", r.is_type2), ("acute MI", r.is_mi_acute and not r.is_type2),
                                   ("STEMI", r.is_stemi), ("NSTEMI", r.is_nstemi)) if f]
            line = (f"  day {int(r.day):+d}  {r.vocabulary_id or ''} {r.concept_code or ''}  {r.concept_name or ''}"
                    f"  [{', '.join(tags) or 'old / other MI'}]")
            lines.append(line)
            if lo <= r.day <= hi:
                hl.append(line.strip())
    else:
        lines.append("  none")
    lines += ["", f"Troponin, day -30 to +30 (above the limit counts on day {lo} to +{hi}):"]
    if len(trops):
        t = trops.assign(day=(pd.to_datetime(trops.measurement_date) - day0).dt.days).sort_values("day")
        for _, r in t.iterrows():
            lim = _limit(r, lab)
            kind = "troponin T" if int(r.get("is_tnt", 0) or 0) == 1 else "troponin I / other"
            above = (lim is not None and not _nan(r.value_as_number) and float(r.value_as_number) > lim)
            src = "lab limit" if not _nan(r.get("range_high")) else "Mayo TnT limit"
            line = (f"  day {int(r.day):+d}  {kind}  {r.value_as_number:g} {r.unit_source_value or ''}  "
                    + (f"(limit {lim:g}, {src})  {'ABOVE' if above else 'not above'}" if lim is not None else
                       "(no limit: not judged)"))
            lines.append(line)
            if above and lo <= r.day <= hi:
                hl.append(line.strip())
    else:
        lines.append("  none")
    return {"text": "\n".join(lines), "highlight": hl}


def draw(m: pd.DataFrame, cfg: dict = DEFAULTS) -> pd.DataFrame:
    """The review sample, groups in GROUPS plus n_planted extra labelled ECGs. One ECG per patient. Seeded."""
    rng = np.random.default_rng(int(cfg["seed"]))
    m = m.sample(frac=1.0, random_state=int(cfg["seed"])).drop_duplicates("person_id")
    pools = {"acute_mi_positive": m[(m.acute_mi == 1) & (m.stemi != 1)], "stemi_positive": m[m.stemi == 1],
             "negative": m[m.acute_mi == 0], "masked": m[m.acute_mi.isna()]}
    parts, used = [], set()
    for g, n in GROUPS.items():
        p = pools[g][~pools[g].ecg_id.astype(str).isin(used)]
        p = p.head(n).assign(group=g)
        used |= set(p.ecg_id.astype(str))
        parts.append(p)
    extra = m[m.acute_mi.isin([0, 1]) & ~m.ecg_id.astype(str).isin(used)].head(int(cfg["n_planted"])).assign(group="planted")
    s = pd.concat(parts + [extra]).sample(frac=1.0, random_state=int(cfg["seed"]) + 1).reset_index(drop=True)
    s["review_id"] = [f"E{i + 1:03d}" for i in range(len(s))]
    # planted error: flip acute MI (and STEMI with it when acute MI goes to no), or flip STEMI on an acute MI ECG
    s["shown_acute_mi"], s["shown_stemi"], s["planted"] = s.acute_mi, s.stemi, ""
    for i in s.index[s.group == "planted"]:
        if s.at[i, "acute_mi"] == 1 and rng.random() < 0.5 and not _nan(s.at[i, "stemi"]):
            s.at[i, "shown_stemi"], s.at[i, "planted"] = 1.0 - float(s.at[i, "stemi"]), "stemi"
        else:
            s.at[i, "shown_acute_mi"] = 1.0 - float(s.at[i, "acute_mi"])
            s.at[i, "shown_stemi"] = 0.0 if s.at[i, "shown_acute_mi"] == 0 else s.at[i, "stemi"]
            s.at[i, "planted"] = "acute_mi"
    return s


def build(m: pd.DataFrame, conds: pd.DataFrame, trops: pd.DataFrame, ecg_cfg: dict, out: str,
          cfg: dict = DEFAULTS) -> Dict:
    s = draw(m, cfg)
    lab = ecg_cfg["labels"]
    rows, tl = [], []
    conds = conds.assign(ecg_id=conds.ecg_id.astype(str)) if len(conds) else conds
    trops = trops.assign(ecg_id=trops.ecg_id.astype(str)) if len(trops) else trops
    for _, r in s.iterrows():
        eid = str(r.ecg_id)
        c = conds[conds.ecg_id == eid] if len(conds) else conds
        t = trops[trops.ecg_id == eid] if len(trops) else trops
        x = timeline(r, c, t, lab)
        tl.append({"review_id": r.review_id, "timeline": x["text"]})
        rows.append({"review_id": r.review_id, "report": x["text"], "evidence": x["highlight"],
                     "values": [["Acute MI label", _word(r.shown_acute_mi)], ["STEMI label", _word(r.shown_stemi)]],
                     "check": ""})
    d = Path(out).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    key = s[["review_id", "ecg_id", "person_id", "group", "acute_mi", "stemi", "shown_acute_mi", "shown_stemi", "planted"]]
    key.to_csv(d / "key.csv", index=False)
    pd.DataFrame(tl).to_csv(d / "timelines.csv", index=False)
    rule = (f"Rule (signed ECG 1-2): acute MI = an acute MI code (not type 2) AND troponin above its limit, both day "
            f"{lab['window_days'][0]} to +{lab['window_days'][1]}; no MI code within 30 days = no; otherwise not used. "
            "STEMI = acute MI AND a STEMI code; NSTEMI only = no; unspecified or before Oct 2015 = not used. ")
    write_page(d / "ecg_mi_review.html", "ECG acute MI / STEMI label review (ECG 4)", "ECG 4", file_id(d / "key.csv"), rows,
               mono=True, help_text=rule + "Approve when both labels are right for this ECG; reject and say why when either "
               "is wrong. Some ECGs carry a planted error: reject those like any other error. Your answers are kept "
               "in this browser as you go.")
    meta = {**cfg, "groups": s.group.value_counts().to_dict(), "ecgs": int(len(s)), "protocol": "ECG 4"}
    (d / "review_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def score(d: str, decisions: Optional[str] = None, cfg: dict = DEFAULTS) -> Dict:
    d = Path(d).expanduser()
    key = pd.read_csv(d / "key.csv", dtype={"planted": str}).fillna({"planted": ""})
    path = decisions or str(d / "review_decisions.json")
    rec = read_decisions(Path(path), file_id(d / "key.csv"))
    dec, sign = rec["decisions"], rec["signoff"]
    key["decision"] = key.review_id.map(lambda r: dec.get(r, {}).get("decision", ""))
    key["note"] = key.review_id.map(lambda r: dec.get(r, {}).get("note", ""))
    problems = [f"{r.review_id}: no decision" for _, r in key.iterrows() if r.decision not in ("approve", "reject")]
    problems += [f"{r.review_id}: rejected without a note" for _, r in key.iterrows() if r.decision == "reject" and not r.note]
    if not sign["cardiologist"] or not sign["date"] or sign["decision"].lower() not in ("approve", "reject"):
        problems.append("sign-off: name, date and decision are needed")
    if problems:
        raise SystemExit("review not complete:\n  " + "\n  ".join(problems[:30]))
    rows = []
    for g in GROUPS:
        k = key[key.group == g]
        rate = float((k.decision == "approve").mean()) if len(k) else float("nan")
        ok = g == "masked" or (len(k) and rate >= cfg["approve_pass"])
        rows.append({"check": f"{g.replace('_', ' ')}: approved", "value": f"{int((k.decision == 'approve').sum())} / {len(k)}"
                     + (f" ({rate:.0%})" if len(k) else ""), "goal": "for information" if g == "masked" else f">= {cfg['approve_pass']:.0%}",
                     "result": "" if g == "masked" else ("PASS" if ok else "FAIL: fix the rule or concept set (see rejected_ecgs.csv), re-sign, rebuild, review a fresh sample")})
    pl = key[key.group == "planted"]
    caught = int((pl.decision == "reject").sum())
    rows.append({"check": "planted errors rejected", "value": f"{caught} / {len(pl)}", "goal": f">= {cfg['planted_caught_pass']}",
                 "result": "PASS" if caught >= min(int(cfg["planted_caught_pass"]), len(pl)) else "REVIEW AGAIN: too many planted errors approved"})
    rows.append({"check": "cardiologist sign-off", "value": f"{sign['cardiologist']}, {sign['date']}: {sign['decision']}",
                 "goal": "approve", "result": "PASS" if sign["decision"].lower() == "approve" else "NOT SIGNED: see comments"})
    res = pd.DataFrame(rows)
    passed = bool(res.result.isin(["PASS", ""]).all())
    res.to_csv(d / "review_results.csv", index=False)
    rej = key[(key.group != "planted") & (key.decision == "reject")][["review_id", "ecg_id", "group", "acute_mi", "stemi", "note"]]
    rej.to_csv(d / "rejected_ecgs.csv", index=False)
    (d / "signoff.json").write_text(json.dumps({"protocol": "ECG 4", "decisions_from": path, "sign_off": sign, "passed": passed,
                                                "scored_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                                "results": rows, "rejected": int(len(rej))}, indent=1))
    return {"results": res, "rejected": rej, "passed": passed}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("--manifest", required=True, help="ECG training manifest (parquet) from ecg_training.dataset")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    sc = sub.add_parser("score")
    sc.add_argument("--dir", required=True)
    sc.add_argument("--decisions", default=None, help="review_decisions.json saved by the page (e.g. ~/Downloads/...)")
    a = ap.parse_args(argv)
    if a.cmd == "sheet":
        from cardioai_mcp.concepts import resolve_concept_sets
        from cardioai_mcp.config import load_study
        from cardioai_mcp.engine import make_engine
        from ecg_training.dataset import load_config

        from .label_audit import troponin_rows_sql
        study = load_study()
        eng = make_engine("bigquery", study)
        resolve_concept_sets(eng, study["concept_sets"], only=CONCEPT_SETS)
        m = pd.read_parquet(Path(a.manifest).expanduser())
        cfg = {**DEFAULTS, "seed": a.seed}
        ids = draw(m, cfg).ecg_id.astype(str).tolist()
        conds = eng.query(eng.render(conditions_sql(study["modality"], ids, cfg["days"]), {}))
        trops = eng.query(eng.render(troponin_rows_sql(study["modality"], ids, cfg["days"]), {}))
        meta = build(m, conds, trops, load_config(), a.out, cfg)
        print(f"{meta['ecgs']} ECGs {meta['groups']}")
        print(f"  open in the workspace web browser: {Path(a.out).expanduser() / 'ecg_mi_review.html'}")
    else:
        r = score(a.dir, a.decisions)
        print(r["results"].to_string(index=False))
        print("\nSIGNED AND PASSED (signoff.json)" if r["passed"] else "\nNot passed: see review_results.csv and rejected_ecgs.csv")


if __name__ == "__main__":
    main()
