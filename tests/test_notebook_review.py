"""Review inside a VS Code notebook (the workspace browser is blocked): the widget reads any review page, saves every
decision to the decisions file the score commands read, and the result scores like the browser page's."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
pytest.importorskip("openpyxl")

from annotation import notebook_review as N  # noqa: E402
from annotation import reference_review as R  # noqa: E402
from test_reference_review import _cohort, _truth_row  # noqa: E402


def _careful(st, kinds):
    for _ in range(len(st.rows)):
        r = st.rows[st.i]
        truth = _truth_row(kinds[r["review_id"]])
        wrong = [k for k, v in r["values"] if truth[k] != v]
        st.decide("reject" if wrong else "approve", ", ".join(wrong), wrong)


def _kinds(tmp_path, c):
    import pandas as pd
    key = pd.read_csv(tmp_path / "key.csv")
    return dict(zip(key.review_id, c.set_index("person_id").loc[key.person_id, "kind"]))


def test_state_saves_resumes_and_scores(tmp_path):
    c = _cohort()
    R.build(c, str(tmp_path))
    page = tmp_path / "review_app.html"
    st = N.ReviewState(N.load_page(str(page)), N.decisions_path(str(page)))
    assert st.path == tmp_path / "review_decisions.json"
    st.decide("approve")
    st2 = N.ReviewState(N.load_page(str(page)), st.path)          # notebook closed and opened again
    assert st2.done() == 1 and st2.i == 1
    with pytest.raises(ValueError, match="note"):
        st2.decide("reject", "")
    st2.i = 0
    _careful(st2, _kinds(tmp_path, c))
    assert st2.sign("Tamanna Nahar", "2026-11-17", "approve") == []
    r = R.score(str(tmp_path))                                    # picks review_decisions.json from the folder
    assert r["passed"], r["results"].to_string()


def test_widget_buttons_drive_the_same_state(tmp_path):
    pytest.importorskip("ipywidgets")
    c = _cohort()
    R.build(c, str(tmp_path))
    ui = N.review(str(tmp_path / "review_app.html"))
    st = ui.state
    buttons = {b.description: b for b in _walk(ui) if hasattr(b, "on_click") and getattr(b, "description", "")}
    first = st.rows[st.i]["review_id"]
    buttons["Approve"].click()
    assert st.decisions[first]["decision"] == "approve" and st.i == 1
    buttons["Reject"].click()
    note = [x for x in _walk(ui) if type(x).__name__ == "Textarea" and "required" in x.placeholder][0]
    note.value = "MVO present, not absent"
    [x for x in _walk(ui) if type(x).__name__ == "Checkbox"][2].value = True
    buttons["Save rejection"].click()
    d = json.loads((tmp_path / "review_decisions.json").read_text())["decisions"]
    assert d[1]["decision"] == "reject" and d[1]["wrong"] == ["MVO"]


def _walk(x):
    yield x
    for c in getattr(x, "children", ()):
        yield from _walk(c)
