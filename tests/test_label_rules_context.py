"""Report-text label rules: negation, hedges, history, resolved findings (ECG AF, CMR, echo wall motion).

Wording here is deliberately different from the training program's practice sets, whose keys stay with the
program lead.
"""
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardioai_mcp.report_context import Context, sentences  # noqa: E402


def nan(v):
    return isinstance(v, float) and math.isnan(v)


def test_context_statuses():
    c = Context()
    af = r"atrial (fibrillation|flutter)"
    assert c.label("Atrial fibrillation, history of hypertension.", af) == 1.0          # history applies forward only
    assert c.label("Known history of atrial flutter. Sinus rhythm today.", af) == 0.0
    assert nan(c.label("Probable atrial flutter, repeat tracing advised.", af))
    assert c.label("Atrial flutter, no longer evident; sinus rhythm.", af) == 0.0
    assert c.label("Atrial fibrillation persists, previously seen on the 2020 tracing.", af) == 1.0
    assert c.label("No ST changes. Atrial fibrillation with RVR.", af) == 1.0            # negation is local
    assert c.label("Sinus rhythm with PVCs.", af) is None
    assert sentences("LVEF 50.5%. No LGE; small effusion.") == ["LVEF 50.5%.", "No LGE;", "small effusion."]


def _af(texts):
    from ecg_training.dataset import load_config, make_labels
    d = pd.DataFrame({"read_text": texts, "confirmed": 1, "mi_dx": 0, "trop_pos": 0, "mi_any_near": 0,
                      "stemi_dx": 0, "nstemi_dx": 0})
    return list(make_labels(d, load_config()).af)


def test_ecg_af_rules():
    v = _af(["Atrial fibrillation with a rapid rate.",
             "Sinus rhythm; the atrial flutter on the earlier ECG is no longer present.",
             "Atrial flutter vs sinus tachycardia with PACs.",
             "Possible atrial fibrillation, artefact limits interpretation.",
             "Atrial fibrillation not identified on this recording. Sinus rhythm.",
             "Atrial fibrillation with RVR, not seen on the previous tracing.",
             "Sinus bradycardia."])
    assert v[0] == 1 and v[1] == 0 and nan(v[2]) and nan(v[3]) and v[4] == 0 and v[5] == 1 and v[6] == 0


def test_mri_rules():
    from mri_training.dataset import load_config, report_labels
    cfg = load_config()
    r = lambda t: report_labels(t, cfg)  # noqa: E731
    a = r("Subendocardial LGE of the basal inferolateral wall. LVEF 48%.")
    assert a["infarct"] == 1 and a["nonischemic_lge"] == 0
    b = r("Transmural scar in the circumflex distribution. LVEF 44%.")
    assert b["infarct"] == 1
    c = r("Circumferential subendocardial enhancement with difficult myocardial nulling, typical of amyloid. LVEF 55%.")
    assert c["infarct"] == 0 and c["nonischemic_lge"] == 1
    d = r("Transmural anterior LGE consistent with infarction. The previously seen apical LV thrombus has resolved.")
    assert d["lv_thrombus"] == 0
    e = r("Transmural anterior LGE consistent with infarction. Apical LV thrombus is not excluded.")
    assert nan(e["lv_thrombus"])
    f = r("Transmural septal LGE consistent with infarction. Questionable microvascular obstruction at the core.")
    assert nan(f["report_mvo"])
    g = r("Clinical history of myocarditis in 2018. No late gadolinium enhancement on this study. LVEF 61%.")
    assert g["nonischemic_lge"] == 0 and g["infarct"] == 0
    h = r("Subepicardial lateral enhancement versus artefact; ischaemic LGE not seen. LVEF 57%.")
    assert nan(h["nonischemic_lge"])
    i = r("Thinned dyskinetic apical segment after anterior infarction (transmural LGE). LVEF 41%.")
    assert nan(i["lv_aneurysm"]) and i["infarct"] == 1
    j = r("Transmural inferior LGE consistent with infarct. Large area of microvascular obstruction. No LV thrombus.")
    assert j["report_mvo"] == 1 and j["lv_thrombus"] == 0


def test_echo_rules():
    from echo_wm_training.labels import parse_report
    a = parse_report("Previous study (2021) showed anterior akinesis. Today there is normal wall motion. LVEF 58%.")
    assert a["wma_any"] == 0 and a["terr_lad"] == 0
    b = parse_report("Inferolateral hypokinesis seen previously has since resolved. LVEF 60%.")
    assert b["wma_any"] == 0 and b["terr_lcx"] == 0
    c = parse_report("Inferior akinesis, unchanged compared with the prior echo. LVEF 45%.")
    assert c["wma_any"] == 1 and c["terr_rca"] == 1
    d = parse_report("Akinetic apex containing an apical thrombus. LVEF 33%.")
    assert d["terr_lad"] == 1 and d["terr_rca"] == 0 and d["terr_lcx"] == 0
    e = parse_report("Anterior wall hypokinesis, best seen in the apical 2-chamber view. LVEF 44%.")
    assert e["terr_lad"] == 1 and e["terr_rca"] == 0 and e["terr_lcx"] == 0
    f = parse_report("The apical segments are hypokinetic. LVEF 48%.")
    assert f["terr_lad"] == 1 and f["terr_rca"] == 1 and f["terr_lcx"] == 1
