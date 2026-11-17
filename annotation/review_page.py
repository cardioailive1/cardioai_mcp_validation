"""Approve / reject review in a web browser, for cardiologists who do not work in Excel.

write_page() builds one self-contained HTML page (no network, no packages, opened from the workspace file system):
one item at a time with its source text, the values to check, Approve / Reject (a note is required to reject),
a grid to jump between items, progress kept in the browser, and a sign-off screen that saves a decisions file
(JSON). read_decisions() reads that file back and refuses one saved from a different review (sheet_id).

Used by the MRI reference review (reference_review), the ECG acute MI / STEMI review (ecg_mi_review) and the
label-audit adjudication (label_audit). Everything stays in the MCP workspace.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

TEMPLATE = Path(__file__).parent / "templates" / "review_app.html"


def file_id(path: Path) -> str:
    """A short hash of a file the review is drawn from (ties a decisions file to its review)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def write_page(path: Path, title: str, protocol: str, sheet_id: str, rows: List[dict], mono: bool = False,
               help_text: Optional[str] = None) -> Path:
    """rows: {review_id, report, evidence: [sentences to highlight], values: [[name, shown value]], check: ''
    | 'CHECK - ...: ...' | other text, check_label?, check_ok?}."""
    data = {"title": title, "protocol": protocol, "sheet_id": sheet_id, "rows": rows, "mono": mono}
    if help_text:
        data["help"] = help_text
    blob = json.dumps(data).replace("</", "<\\/")                   # cannot close the <script> block
    Path(path).write_text(TEMPLATE.read_text().replace("__DATA__", blob), encoding="utf-8")
    return Path(path)


def read_decisions(path: Path, sheet_id: str) -> Dict[str, object]:
    """{decisions: {review_id: {decision, note, wrong: [value names ticked as wrong]}}, signoff: {cardiologist, date, decision, comments}}."""
    rec = json.loads(Path(path).expanduser().read_text())
    if rec.get("sheet_id") != sheet_id:
        raise SystemExit(f"{path} was saved from a different review sheet (sheet_id {rec.get('sheet_id')} != {sheet_id})")
    dec = {x["review_id"]: {"decision": str(x.get("decision", "")).lower(), "note": str(x.get("note", "") or "").strip(),
                            "wrong": list(x.get("wrong") or [])}
           for x in rec.get("decisions", [])}
    s = rec.get("signoff") or {}
    return {"decisions": dec, "signoff": {k: str(s.get(k, "") or "").strip() for k in ("cardiologist", "date", "decision", "comments")}}
