"""Approve / reject review inside a VS Code (Jupyter) notebook: for workspaces where the web browser is blocked.

Every review page the repo writes (reference_review: review_app.html, ecg_mi_review: ecg_mi_review.html,
label_audit: adjudication_<n>.html) carries its items as JSON. This module reads that JSON and shows the same review
as notebook widgets: one item at a time with its text (key sentences highlighted), the values to check, Approve /
Reject (tick what is wrong + a note), Previous / Next, progress, and a sign-off. Every decision is saved at once to
the decisions file next to the page, so closing the notebook loses nothing; the score commands read that file.

In a notebook cell (the kernel's working folder = the repo):

    from annotation.notebook_review import review
    review("~/cardioai_exports/ref_review/review_app.html")

Needs ipywidgets (`%pip install ipywidgets` if the import fails). Everything stays in the workspace.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

_DATA = re.compile(r'<script id="data" type="application/json">(.*?)</script>', re.S)


def load_page(page: str) -> dict:
    text = Path(page).expanduser().read_text(encoding="utf-8")
    m = _DATA.search(text)
    if not m:
        raise ValueError(f"{page} is not a review page written by this repo")
    return json.loads(m.group(1))


def decisions_path(page: str) -> Path:
    """Where the score commands look: review_decisions.json, or adjudication_<n>_decisions.json for the label audit."""
    p = Path(page).expanduser()
    return p.with_name(f"{p.stem}_decisions.json") if p.stem.startswith("adjudication_") else p.with_name("review_decisions.json")


class ReviewState:
    """The decisions and sign-off, saved to the decisions file after every change (same format as the web page)."""

    def __init__(self, data: dict, path: Path):
        self.data, self.path, self.rows = data, Path(path), data["rows"]
        self.decisions: Dict[str, dict] = {}
        self.signoff = {"cardiologist": "", "date": dt.date.today().isoformat(), "decision": "", "comments": ""}
        if self.path.exists():
            old = json.loads(self.path.read_text())
            if old.get("sheet_id") == data["sheet_id"]:
                self.decisions = {d["review_id"]: d for d in old.get("decisions", [])}
                self.signoff.update({k: v for k, v in (old.get("signoff") or {}).items() if v})
        self.i = next((j for j, r in enumerate(self.rows) if r["review_id"] not in self.decisions), 0)

    def done(self) -> int:
        return sum(r["review_id"] in self.decisions for r in self.rows)

    def decide(self, decision: str, note: str = "", wrong: Optional[List[str]] = None) -> None:
        if decision == "reject" and not note.strip():
            raise ValueError("a note is needed to reject")
        r = self.rows[self.i]
        self.decisions[r["review_id"]] = {"review_id": r["review_id"], "decision": decision, "note": note.strip(),
                                          "wrong": list(wrong or [])}
        self.save()
        nxt = [j for j in range(self.i + 1, len(self.rows)) if self.rows[j]["review_id"] not in self.decisions]
        if nxt:
            self.i = nxt[0]

    def sign(self, cardiologist: str, date: str, decision: str, comments: str = "") -> List[str]:
        self.signoff = {"cardiologist": cardiologist.strip(), "date": str(date), "decision": decision, "comments": comments.strip()}
        problems = []
        if self.done() < len(self.rows):
            problems.append(f"{len(self.rows) - self.done()} items have no decision yet")
        if not (self.signoff["cardiologist"] and self.signoff["date"] and decision in ("approve", "reject")):
            problems.append("name, date and decision are needed")
        if decision == "reject" and not self.signoff["comments"]:
            problems.append("say why in the comments")
        self.save()
        return problems

    def save(self) -> None:
        out = {"sheet_id": self.data["sheet_id"], "protocol": self.data.get("protocol"),
               "saved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "signoff": self.signoff,
               "decisions": [self.decisions[r["review_id"]] for r in self.rows if r["review_id"] in self.decisions]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=1))
        tmp.replace(self.path)


def _highlight(text: str, evidence: List[str]) -> str:
    out = html.escape(text)
    for s in evidence or []:
        s = html.escape(s.strip())
        if len(s) >= 4 and s in out:
            out = out.replace(s, f"<mark>{s}</mark>", 1)
    return out


def review(page: str, decisions: Optional[str] = None):
    """Show the review for one page in the notebook. Returns the widget (and keeps the state on it as .state)."""
    import ipywidgets as w
    from IPython.display import display

    data = load_page(page)
    st = ReviewState(data, Path(decisions).expanduser() if decisions else decisions_path(page))
    mono = "font-family:'Courier New',monospace;font-size:14px;" if data.get("mono") else "font-size:15px;"
    head = w.HTML()
    text = w.HTML(layout=w.Layout(width="60%", max_height="520px", overflow="auto", border="1px solid #d9dde4",
                                  padding="8px"))
    values = w.HTML()
    check = w.HTML()
    status = w.HTML()
    approve = w.Button(description="Approve", button_style="success", layout=w.Layout(width="48%", height="44px"))
    reject = w.Button(description="Reject", button_style="danger", layout=w.Layout(width="48%", height="44px"))
    ticks = w.VBox()
    note = w.Textarea(placeholder="Say which value is wrong and why (required to reject)", layout=w.Layout(width="100%", height="70px"))
    save_reject = w.Button(description="Save rejection", button_style="danger", layout=w.Layout(width="100%"))
    reject_box = w.VBox([w.HTML("<b>Which values are wrong?</b>"), ticks, note, save_reject], layout=w.Layout(display="none"))
    prev, nxt = w.Button(description="< Previous"), w.Button(description="Next >")
    jump = w.Dropdown(description="Go to", options=[r["review_id"] for r in st.rows], layout=w.Layout(width="200px"))
    progress = w.IntProgress(min=0, max=len(st.rows))
    help_ = w.HTML(f"<p style='color:#5d6676;font-size:13px'>{html.escape(data.get('help') or 'Approve when every value matches the report. Reject when any value is wrong, and say what. Some items carry a planted error: reject those like any other error.')}</p>")

    name = w.Text(description="Name", value=st.signoff["cardiologist"], layout=w.Layout(width="420px"))
    date = w.Text(description="Date", value=st.signoff["date"], layout=w.Layout(width="260px"))
    dec = w.Dropdown(description="Decision", options=[("Choose...", ""), ("Approve: the reviewed values stand", "approve"),
                                                      ("Reject (say why)", "reject")], value=st.signoff["decision"] or "")
    comments = w.Textarea(description="Comments", value=st.signoff["comments"], layout=w.Layout(width="600px"))
    sign_btn = w.Button(description="Sign", button_style="primary")
    sign_msg = w.HTML()
    sign_box = w.VBox([w.HTML("<h3>Sign-off</h3>"), name, date, dec, comments, sign_btn, sign_msg])

    def render():
        r = st.rows[st.i]
        d = st.decisions.get(r["review_id"])
        head.value = (f"<h3 style='margin:4px 0'>{html.escape(data['title'])}</h3>"
                      f"<b>{html.escape(r['review_id'])}</b> ({st.i + 1} of {len(st.rows)}) &nbsp; {st.done()} reviewed")
        text.value = f"<div style='white-space:pre-wrap;{mono}line-height:1.5'>{_highlight(r['report'], r.get('evidence'))}</div>"
        values.value = ("<table style='font-size:15px;width:100%'>" + "".join(
            f"<tr><td style='padding:4px 8px'>{html.escape(k)}</td><td style='padding:4px 8px;font-weight:bold'>{html.escape(v)}</td></tr>"
            for k, v in r["values"]) + "</table>")
        c = r.get("check") or ""
        if c.startswith("CHECK"):
            check.value = (f"<div style='background:#f9eeda;color:#8a5a00;padding:6px 8px;font-weight:bold'>"
                           f"{html.escape(r.get('check_label') or 'A second, independent reading differs: ')}"
                           f"{html.escape(re.sub(r'^CHECK - [^:]*: ', '', c))}</div>")
        else:
            check.value = "" if not c else "<div style='background:#e5f4e4;color:#1f7a3f;padding:6px 8px'>Second reading agrees</div>"
        status.value = ("" if not d else f"<b style='color:{'#1f7a3f' if d['decision'] == 'approve' else '#a8261c'}'>"
                        f"{'Approved' if d['decision'] == 'approve' else 'Rejected: ' + html.escape(d['note'])}</b>")
        ticks.children = [w.Checkbox(description=f"{k} (shown: {v})", value=bool(d and k in d.get("wrong", [])), indent=False)
                          for k, v in r["values"]]
        note.value = d["note"] if d and d["decision"] == "reject" else ""
        reject_box.layout.display = "none"
        progress.value = st.done()
        prev.disabled, nxt.disabled = st.i == 0, st.i == len(st.rows) - 1
        jump.unobserve(on_jump, "value")
        jump.value = r["review_id"]
        jump.observe(on_jump, "value")

    def on_approve(_):
        st.decide("approve")
        render()

    def on_reject(_):
        reject_box.layout.display = "flex"

    def on_save_reject(_):
        if not note.value.strip():
            note.placeholder = "A note is required to reject"
            return
        st.decide("reject", note.value, [t.description.split(" (shown:")[0] for t in ticks.children if t.value])
        render()

    def move(k):
        st.i = max(0, min(len(st.rows) - 1, st.i + k))
        render()

    def on_jump(ch):
        st.i = [r["review_id"] for r in st.rows].index(ch["new"])
        render()

    def on_sign(_):
        p = st.sign(name.value, date.value, dec.value, comments.value)
        sign_msg.value = ("<b style='color:#a8261c'>" + "; ".join(p) + "</b>") if p else (
            f"<b style='color:#1f7a3f'>Signed and saved to {html.escape(str(st.path))}. Thank you.</b>")

    approve.on_click(on_approve)
    reject.on_click(on_reject)
    save_reject.on_click(on_save_reject)
    prev.on_click(lambda _: move(-1))
    nxt.on_click(lambda _: move(1))
    sign_btn.on_click(on_sign)
    right = w.VBox([values, check, w.HBox([approve, reject]), reject_box, status, w.HBox([prev, nxt, jump]), help_],
                   layout=w.Layout(width="40%", padding="0 0 0 12px"))
    ui = w.VBox([head, progress, w.HBox([text, right]), w.HTML("<hr>"), sign_box])
    ui.state = st
    jump.observe(on_jump, "value")
    render()
    display(ui)
    return ui
