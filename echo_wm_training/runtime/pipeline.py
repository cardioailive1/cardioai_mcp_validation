"""Study pipeline: apical cines -> segment scores, coronary territories, global function, flags.

    analysis = analyse(model, [(video_a4c, "A4C"), (video_a2c, "A2C"), (video_a3c, None)], device)
    result   = interpret(analysis, thresholds, red_flag_rules)

A stated view is trusted unless the model is confident it is a different view (then the model's view is used
and a note says so). A clip without a stated view takes the model's view when P >= MIN_VIEW_PROB, otherwise it
is not used for segments. Clips of other views (parasternal, subcostal, Doppler) are ignored.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .model import view_index
from .preprocess import EchoInputError, Prepared, Video, prepare
from .segments import TERRITORIES, VIEW_CLASSES, VIEWS, summarise

MIN_VIEW_PROB = 0.7          # model view accepted for an unlabelled clip
OVERRULE_VIEW_PROB = 0.9     # model view replaces a stated view


@torch.no_grad()
def analyse(model, clips: Sequence[Tuple[Video, Optional[str]]], device: str = "cpu") -> Dict[str, object]:
    model.eval()
    notes: List[str] = []
    prepared: List[Tuple[Prepared, Optional[str]]] = []
    for k, (v, stated) in enumerate(clips):
        try:
            prepared.append((prepare(v), stated or v.view))
        except EchoInputError as e:
            notes.append(f"clip {k + 1}: not used ({e})")
    if not prepared:
        raise EchoInputError("no usable apical cine: " + "; ".join(notes))
    xs, owner = [], []
    for k, (p, _) in enumerate(prepared):
        xs.append(p.clips)
        owner += [k] * len(p.clips)
        notes += [f"clip {k + 1}: {n}" for n in p.notes]
    x = torch.from_numpy(np.concatenate(xs)).to(device)
    out = model(x)
    vp = out["view"].softmax(-1).cpu().numpy()
    owner = np.asarray(owner)
    views_used: List[Optional[str]] = []
    clip_views = []
    for k, (p, stated) in enumerate(prepared):
        pv = vp[owner == k].mean(0)
        best = VIEW_CLASSES[int(pv.argmax())]
        if stated in VIEWS and not (best != stated and pv.max() >= OVERRULE_VIEW_PROB):
            use = stated
        elif stated in VIEWS:
            use = best if best in VIEWS else None
            notes.append(f"clip {k + 1}: stated {stated} but the model reads {best} (P {pv.max():.2f}); using {use or 'none'}")
        else:
            use = best if (best in VIEWS and pv.max() >= MIN_VIEW_PROB) else None
            if use is None:
                notes.append(f"clip {k + 1}: view not recognised as apical 4/2/3-chamber (best {best}, P {pv.max():.2f}); not used")
        views_used.append(use)
        clip_views.append({"clip": k + 1, "stated_view": stated, "model_view": best, "model_view_p": round(float(pv.max()), 3),
                           "used_as": use, "clips": int((owner == k).sum())})
    per_clip = [views_used[k] for k in owner]
    if not any(v in VIEWS for v in per_clip):
        raise EchoInputError("no apical 4-, 2- or 3-chamber cine recognised: " + "; ".join(notes))
    st = model.study_outputs(out, view_index(per_clip).to(x.device))
    seg_prob = st["seg"].softmax(-1).cpu().numpy()
    seen = st["seen"].cpu().numpy()
    s = summarise(seg_prob, seen)
    missing = [v for v in VIEWS if v not in views_used]
    if missing:
        notes.append(f"views not provided or not recognised: {', '.join(missing)}; their segments are not assessed")
    p_abn = 1.0 - seg_prob[seen, 0]
    probs = {"wma_any": float(p_abn.max()) if len(p_abn) else None,
             "global_hypokinesis": float(torch.sigmoid(st["ghypo"]).item())}
    for t in TERRITORIES:
        probs[f"terr_{t.lower()}"] = s["territories"][t]["p_abnormal"]
    findings = {"lvef_pct": round(float(st["lvef"].item()), 1), "wmsi": s["wmsi"],
                "assessed_segments": s["assessed_segments"], "n_abnormal_segments": s["n_abnormal_segments"],
                "max_segment_score": max([v["score"] for v in s["segments"].values()], default=None),
                "segments": s["segments"], "territories": s["territories"], "views": clip_views}
    return {"findings": findings, "probabilities": probs, "notes": notes,
            "attention": out["attn"].cpu().numpy(), "clip_views": per_clip}


def _rule_hits(findings: Dict[str, object], probs: Dict[str, Optional[float]], flags: Dict[str, bool],
               rules: Dict[str, Dict]) -> List[Dict[str, object]]:
    out = []
    for name, r in (rules or {}).items():
        if "flag" in r:
            v = sum(bool(flags.get(f)) for f in ([r["flag"]] if isinstance(r["flag"], str) else r["flag"]))
            hit = v >= int(r.get("min", 1))
        elif "prob" in r:
            v = probs.get(r["prob"])
            hit = v is not None and v >= float(r["min"])
        else:
            v = findings.get(r["value"])
            hit = v is not None and (("min" not in r or v >= float(r["min"])) and ("max" not in r or v <= float(r["max"])))
        if hit:
            out.append({"flag": name, "value": v, "text": r["text"]})
    return out


def interpret(analysis: Dict[str, object], thresholds: Dict[str, float], rules: Dict[str, Dict]) -> Dict[str, object]:
    """Locked thresholds -> flags; flags and findings -> red flags; territory summary for the clinician."""
    probs = analysis["probabilities"]
    flags = {h: (probs.get(h) is not None and probs[h] >= float(t)) for h, t in thresholds.items()}
    terr = [t for t in TERRITORIES if flags.get(f"terr_{t.lower()}")]
    f = dict(analysis["findings"])
    f["territories_flagged"] = terr
    f["n_territories_flagged"] = len(terr)
    pattern = ("global" if flags.get("global_hypokinesis") and len(terr) >= 2 else
               "regional" if terr else "global" if flags.get("global_hypokinesis") else
               "abnormal, territory unclear" if flags.get("wma_any") else "normal")
    f["pattern"] = pattern
    return {"flags": flags, "findings": f, "red_flags": _rule_hits(f, probs, flags, rules), "pattern": pattern}
