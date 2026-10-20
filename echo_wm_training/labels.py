"""Echo report text -> wall-motion labels (the AUTOMATIC labels; audited with annotation/label_audit.py).

    from echo_wm_training.labels import parse_report
    parse_report("Hypokinesis of the basal and mid inferior and inferolateral walls. LVEF 45%.")
    -> {'seg_1': 0.0, ..., 'seg_4': 1.0, 'seg_5': 1.0, ..., 'global_hypokinesis': 0.0, 'wma_any': 1.0,
        'terr_lad': 0.0, 'terr_rca': 1.0, 'terr_lcx': 1.0, 'wmsi': 1.24, 'report_lvef': 45.0, ...}

Rules (the rule sheet in annotation/LABEL_AUDIT.md must say the same):
  * clauses (split at . ; and new lines) are read separately;
  * severity words: hypokinesis/-tic (score 2), akinesis/-tic (3), dyskinesis/-tic or aneurysm(al) (4);
    "normal(ly) contracting / normal wall motion" (1);
  * walls: anterior, anteroseptal, inferoseptal, septal (= both septal walls), inferior, inferolateral
    (= posterior, posterolateral), anterolateral, lateral (= both lateral walls), apex; levels: basal, mid,
    apical (and ranges "basal to mid", "entire"); no level = the whole wall; "apical segments" without a wall
    = all apical segments and the apex; "LAD / RCA / circumflex territory" = that territory's segments;
  * a wall takes the severity before it ("hypokinesis of the inferior wall") or, when the clause starts with
    the walls, the severity after it ("the apex is akinetic");
  * global / diffuse / generalised hypokinesis: global_hypokinesis = 1 and every segment not described
    otherwise = hypokinetic;
  * "no (other) regional wall motion abnormalities" / "normal wall motion": the remaining segments = normal;
  * when abnormal segments are listed, segments not mentioned are normal (unmentioned_normal = True,
    stated in the model card);
  * hedged clauses (possible, questionable, may represent, cannot exclude, suggestive, equivocal) leave the
    segments they mention UNKNOWN (masked), never normal;
  * a clause about another study ("previous echo showed ...", "on the 2019 study") is skipped unless it says
    the finding is current ("persists", "unchanged", "new"); a resolved finding ("has resolved", "no longer
    seen") is normal now, and counts as a wall-motion statement;
  * "apical" naming a view, clot or mass ("apical thrombus", "apical 4-chamber view") is not a segment level;
  * no wall-motion statement at all: every segment unknown (masked).
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Set, Tuple

from .runtime.segments import N_SEG, SEGMENTS, TERRITORIES, territory_labels

_CLAUSE = re.compile(r"[.;\n]+|(?<=\D)\.(?=\s)")
_SEV = [(re.compile(r"\bhypo-?kine(?:sis|tic|sia)\b|\bhypocontractil\w*"), 2),
        (re.compile(r"\ba-?kine(?:sis|tic|sia)\b"), 3),
        (re.compile(r"\bdys-?kine(?:sis|tic|sia)\b|\baneurysm(?:al)?\b"), 4),
        (re.compile(r"\bnormal(?:ly)? contract\w*|\bcontract\w* normally\b|\bnormokine\w*"), 1)]
_WALLS = [("AS", r"antero-?sept(?:al|um)"), ("IS", r"infero-?sept(?:al|um)"), ("AL", r"antero-?lateral"),
          ("IL", r"infero-?lateral|postero-?lateral|posterior|infero-?posterior"), ("SEPT", r"sept(?:al|um)"),
          ("LAT", r"lateral"), ("A", r"anterior"), ("I", r"inferior"), ("APEX", r"apex|apical cap"),
          ("LAD", r"(?:lad|left anterior descending)(?: artery)? (?:territory|distribution)"),
          ("RCA", r"(?:rca|right coronary)(?: artery)? (?:territory|distribution)"),
          ("LCX", r"(?:lcx|cx|circumflex|left circumflex)(?: artery)? (?:territory|distribution)")]
_WALL_RE = re.compile(r"\b(" + "|".join(f"(?P<{k}>{p})" for k, p in _WALLS) + r")\b")
_LEVEL_RE = re.compile(r"\b(?P<range>bas(?:e|al)[- ]?(?:to|through|-)[- ]?(?:mid|apical|apex)|mid[- ]?(?:to|-)[- ]?(?:apical|apex)|"
                       r"entire|whole|all segments)\b|\b(?P<B>bas(?:e|al|ilar))\b|\b(?P<M>mid(?:-?ventricular|-?cavity)?)\b|"
                       r"\b(?P<AP>apical|distal)\b")
_NEG_BEFORE = re.compile(r"\b(no|without|not|absence of|resolved)\W+(?:\w+\W+){0,2}$")
_HEDGE = re.compile(r"\b(possible|possibly|probable|questionabl[ey]|equivocal|may (?:be|represent)|"
                    r"can ?not (?:be )?(?:exclude|excluded|ruled out)|cannot rule out|suggestive|suggests?|r/o|rule out|\?)")
# a clause about another study (skipped unless it also says the finding is current), and resolved findings (normal now)
_PRIOR = re.compile(r"\b(?:previous|prior|old|earlier|outside|referring|last)\s+(?:echo\w*|stud(?:y|ies)|exam\w*|tte|tee|scan)\b|"
                    r"\b(?:on|in|from)\s+(?:the\s+)?(?:19|20)\d\d\s+(?:echo\w*|study|exam\w*)\b|\bpreviously\s+(?:seen|reported|described|noted)\b")
_CURRENT = re.compile(r"\b(?:today|now|currently|current study|this study|this exam\w*|new(?:ly)?|persist\w*|still|again|unchanged|"
                      r"remains?|no change|not significantly changed)\b")
_RESOLVED = re.compile(r"\b(?:has|have|had)?\s*(?:since\s+)?resolved\b|\bno longer (?:seen|present|evident|visible)\b|"
                       r"\b(?:has|have) normali[sz]ed\b|\bnow normal\b")
# a level word that names a view, a clot or a mass, not segments ("apical thrombus", "apical 4-chamber view")
_LEVEL_NOT_SEG = re.compile(r"^\W+(?:[a-z]+\W+)?(?:thromb|clot|mass|trabecul|view|window|imag|foreshort|"
                            r"(?:two|three|four|2|3|4|long)[- ]?(?:chamber|axis))")
_GLOBAL = re.compile(r"\b(global(?:ly)?|diffuse(?:ly)?|generali[sz]ed)\b[\w\s,-]{0,40}?\bhypo-?kine")
_NORMAL_WM = re.compile(r"\bno (?:other |further )?(?:definite |significant )?(?:regional |segmental )?wall[- ]motion abnormalit|"
                        r"\bnormal (?:regional |segmental |left ventricular )?wall[- ]motion|\bwall[- ]motion (?:is|was|appears) normal|"
                        r"\bno (?:other )?(?:regional|segmental) (?:wall[- ]motion )?(?:abnormalit|dysfunction)")
_LVEF = re.compile(r"\b(?:lvef|(?:lv )?ejection fraction|\bef\b)[^0-9%]{0,30}?(\d{1,2}(?:\.\d)?)\s*(?:(?:-|–|to)\s*(\d{1,2}(?:\.\d)?)\s*)?%")
_WMSI = re.compile(r"wall[- ]motion score index[^0-9]{0,20}(\d(?:\.\d{1,2})?)")

_SEG = {  # (level, wall) -> segments
    "B": {"A": [1], "AS": [2], "IS": [3], "I": [4], "IL": [5], "AL": [6], "SEPT": [2, 3], "LAT": [5, 6]},
    "M": {"A": [7], "AS": [8], "IS": [9], "I": [10], "IL": [11], "AL": [12], "SEPT": [8, 9], "LAT": [11, 12]},
    "AP": {"A": [13], "AS": [14], "IS": [14], "I": [15], "IL": [16], "AL": [16], "SEPT": [14], "LAT": [16]},
}
ALL_LEVELS = ("B", "M", "AP")
SEG_COLUMNS = [f"seg_{s}" for s in SEGMENTS]


def _segments(wall: str, levels: Set[str]) -> List[int]:
    if wall == "APEX":
        return [17]
    if wall in ("LAD", "RCA", "LCX"):
        return list(TERRITORIES["LCx" if wall == "LCX" else wall])
    out: List[int] = []
    for lv in (levels or set(ALL_LEVELS)):
        out += _SEG[lv].get(wall, [])
    return out


def _tokens(clause: str) -> List[Tuple[int, str, object]]:
    toks: List[Tuple[int, str, object]] = []
    for rx, score in _SEV:
        for m in rx.finditer(clause):
            if score > 1 and _NEG_BEFORE.search(clause[:m.start()]):
                continue
            toks.append((m.start(), "sev", score))
    for m in _WALL_RE.finditer(clause):
        toks.append((m.start(), "wall", next(k for k, _ in _WALLS if m.group(k))))
    for m in _LEVEL_RE.finditer(clause):
        if _LEVEL_NOT_SEG.match(clause[m.end():m.end() + 40]):
            continue
        if m.group("range"):
            r = m.group("range")
            lv = {"B", "M", "AP"} if re.search(r"apical|apex|entire|whole|all", r) and r.startswith(("bas", "entire", "whole", "all")) \
                else {"M", "AP"} if r.startswith("mid") else {"B", "M"}
            toks.append((m.start(), "level", lv))
        else:
            toks.append((m.start(), "level", {k for k in ("B", "M", "AP") if m.group(k)}))
    return sorted(toks, key=lambda t: t[0])


def _read_clause(clause: str) -> Tuple[Dict[int, int], Set[int]]:
    """-> (segment -> score, segments mentioned in a hedged clause)."""
    toks = _tokens(clause)
    sevs = [(p, s) for p, k, s in toks if k == "sev"]
    if not sevs:
        return {}, set()
    first_sev = sevs[0][0]
    first_wall = next((p for p, k, _ in toks if k == "wall"), None)
    sev_first = first_wall is None or first_sev < first_wall
    groups: List[Tuple[int, List[int]]] = []          # (position, segments)
    levels: Set[str] = set()
    last_wall = False
    pending_levels_pos = None
    for p, k, v in toks:
        if k == "level":
            if last_wall:
                levels = set()
            levels |= v
            pending_levels_pos = p
            last_wall = False
        elif k == "wall":
            groups.append((p, _segments(v, set(levels))))
            last_wall = True
            pending_levels_pos = None
        else:                                          # severity
            if pending_levels_pos is not None and not last_wall and levels:
                # "apical segments are akinetic", "the basal segments are hypokinetic": a level with no wall
                segs = [s for lv in levels for w in ("A", "AS", "IS", "I", "IL", "AL") for s in _SEG[lv][w]]
                groups.append((pending_levels_pos, sorted(set(segs + ([17] if "AP" in levels else [])))))
            levels, last_wall, pending_levels_pos = set(), False, None
    if pending_levels_pos is not None and levels and sev_first:
        segs = [s for lv in levels for w in ("A", "AS", "IS", "I", "IL", "AL") for s in _SEG[lv][w]]
        groups.append((pending_levels_pos, sorted(set(segs + ([17] if "AP" in levels else [])))))
    out: Dict[int, int] = {}
    for p, segs in groups:
        before = [s for q, s in sevs if q < p]
        after = [s for q, s in sevs if q > p]
        score = (before[-1] if before else after[0]) if sev_first else (after[0] if after else before[-1])
        for s in segs:
            out[s] = max(out.get(s, 0), score)
    hedged = set(out) if _HEDGE.search(clause) else set()
    if _RESOLVED.search(clause) and not hedged:
        out = {s: 1 for s in out}                       # "inferior hypokinesis has resolved": normal now
    return ({s: v for s, v in out.items() if s not in hedged}, hedged)


def parse_report(text: Optional[str], unmentioned_normal: bool = True) -> Dict[str, object]:
    nan = float("nan")
    out: Dict[str, object] = {c: nan for c in SEG_COLUMNS}
    out.update({"wm_described": False, "global_hypokinesis": nan, "wm_uncertain": False, "report_lvef": nan, "wmsi_stated": nan})
    if not isinstance(text, str) or not text.strip():
        out.update({k: nan for k in ("wma_any", "terr_lad", "terr_rca", "terr_lcx", "wmsi")})
        return out
    t = text.lower()
    scores: Dict[int, int] = {}
    hedged: Set[int] = set()
    kept: List[str] = []
    resolved = False
    for clause in _CLAUSE.split(t):
        if _PRIOR.search(clause) and not _CURRENT.search(clause) and not _RESOLVED.search(clause):
            continue                                    # "previous echo (2019) showed inferior akinesis"
        kept.append(clause)
        sc, hd = _read_clause(clause)
        resolved |= bool(sc) and all(v == 1 for v in sc.values()) and bool(_RESOLVED.search(clause))
        for s, v in sc.items():
            scores[s] = max(scores.get(s, 0), v)
        hedged |= hd
    k = " . ".join(kept)
    glob = bool(_GLOBAL.search(k)) and not re.search(r"\bno (?:global|diffuse)\b", k)
    normal_stmt = bool(_NORMAL_WM.search(k)) or resolved
    described = bool(scores or glob or normal_stmt or hedged)
    final: Dict[int, Optional[int]] = {}
    for s in range(1, N_SEG + 1):
        if s in scores:
            final[s] = scores[s]
        elif s in hedged:
            final[s] = None
        elif glob:
            final[s] = 2
        elif normal_stmt or (unmentioned_normal and any(v > 1 for v in scores.values())):
            final[s] = 1
        else:
            final[s] = None
    if glob:
        for s in list(final):
            if final[s] == 1 and s not in scores:
                final[s] = 2
    for s, v in final.items():
        out[f"seg_{s}"] = float(v - 1) if v is not None else nan
    out["wm_described"] = described
    out["wm_uncertain"] = bool(hedged)
    out["global_hypokinesis"] = (1.0 if glob else 0.0) if described else nan
    out.update(territory_labels(final) if described else {k: nan for k in ("wma_any", "terr_lad", "terr_rca", "terr_lcx", "wmsi")})
    m = None
    for m in _LVEF.finditer(t):
        before = t[:m.start()]
        if not re.search(r"\b(prior|previous|previously)\b", before[max(before.rfind("."), before.rfind(";"), 0):]):
            break
    else:
        m = None
    if m:
        a = float(m.group(1))
        b = float(m.group(2)) if m.group(2) else a
        if 5 <= a <= 90 and 5 <= b <= 90:
            out["report_lvef"] = (a + b) / 2
    w = _WMSI.search(t)
    if w:
        out["wmsi_stated"] = float(w.group(1))
    return out


def label_columns() -> List[str]:
    return SEG_COLUMNS + ["global_hypokinesis", "wma_any", "terr_lad", "terr_rca", "terr_lcx", "wmsi", "report_lvef"]


def is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)
