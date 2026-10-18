"""Automatic auditor: an independent, second reading of report text, used before the data scientists.

It does NOT reuse the training rules (mri_training/config.yaml, ecg_training/config.yaml). It reads each
sentence, finds the label's terms (tasks.yaml label_audit.auto_terms), and decides with NegEx-style
triggers whether each mention is affirmed, negated or uncertain (or, for AF, historical):

    present (affirmed)          -> 1
    only uncertain mentions     -> U
    negated / absent (per rule) -> 0
    not mentioned               -> NM (or 0 where silence means absent, as in the rule sheet)

Because it is built differently from the training rules, agreement between the two is evidence; their
shared blind spots are what the data-scientist verification sample exists to catch.
Runs in the workspace with the standard library only: no model, no network, no text leaves MCP.

    from annotation.auto_audit import AutoAuditor
    aa = AutoAuditor("mri"); aa.read(report_text)   # {'infarct': 1.0, ..., 'report_lvef': 42.0}
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .select import load_tasks

U, NM = "U", "NM"

_SENT = re.compile(r"(?<=[.;!?])\s+|\n+|(?<=:)\s+(?=[A-Z])")
_SKIP = re.compile(r"(?i)^\s*(indication|history|clinical (history|information|question)|reason for (exam|study|referral)|"
                   r"question|comparison|technique)\b")


def sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT.split(text or "") if s and s.strip()]


class AutoAuditor:
    def __init__(self, modality: str, cfg: Optional[dict] = None):
        cfg = cfg or load_tasks()["label_audit"]
        t = cfg["auto_terms"]
        self.modality = modality
        self.m = t[modality]
        g = t["triggers"]
        self.pre_neg = re.compile(g["pre_negation"], re.I)
        self.post_neg = re.compile(g["post_negation"], re.I)
        self.pre_unc = re.compile(g["pre_uncertain"], re.I)
        self.post_unc = re.compile(g["post_uncertain"], re.I)
        self.hist = re.compile(g["historical"], re.I)
        self.term = re.compile(g["terminators"], re.I)
        self.pre_words, self.post_words = int(g.get("pre_window_words", 8)), int(g.get("post_window_words", 6))
        self.concepts = {k: re.compile(v, re.I) for k, v in self.m["concepts"].items()}
        self.exclude = {k: re.compile(v, re.I) for k, v in (self.m.get("exclude") or {}).items()}

    # ---------------------------------------------------------------- one mention
    def _window(self, s: str, start: int, end: int):
        before = s[:start]
        cut = list(self.term.finditer(before))
        if cut:
            before = before[cut[-1].end():]
        before = " ".join(before.split()[-self.pre_words:])
        after = s[end:]
        m = self.term.search(after)
        if m:
            after = after[:m.start()]
        after = " ".join(after.split()[:self.post_words])
        return before, after

    def status(self, s: str, start: int, end: int) -> str:
        """'affirmed' / 'negated' / 'uncertain' / 'historical' for the mention s[start:end]."""
        before, after = self._window(s, start, end)
        if self.post_unc.search(after) or self.pre_unc.search(before) or s.rstrip().endswith("?"):
            return "uncertain"                      # 'cannot be excluded' must win over 'excluded'
        if self.pre_neg.search(before) or self.post_neg.search(after):
            return "negated"
        if self.hist.search(before):
            return "historical"
        return "affirmed"

    def mentions(self, text: str, concept: str) -> List[str]:
        pat, ex = self.concepts[concept], self.exclude.get(concept)
        out = []
        for s in sentences(text):
            if _SKIP.search(s):
                continue
            if ex is not None and ex.search(s):
                continue
            st = [self.status(s, m.start(), m.end()) for m in pat.finditer(s)]
            # one hedged mention makes the whole sentence hedged for this label
            # ("subendocardial enhancement, possibly representing a small infarct")
            out += ["uncertain"] if "uncertain" in st else st
        return out

    def evidence(self, text: str) -> str:
        """Sentences mentioning any label's terms (what the reader should look at; not the answer)."""
        pats = list(self.concepts.values()) + [re.compile(v, re.I) for v in (self.m.get("values") or {}).values()]
        keep = [s for s in sentences(text) if any(p.search(s) for p in pats)]
        return "\n".join(dict.fromkeys(keep))[:4000]

    # ---------------------------------------------------------------- labels
    @staticmethod
    def _simple(st: List[str], silent: object) -> object:
        if "affirmed" in st:
            return 1.0
        if "uncertain" in st:
            return U
        if "negated" in st or "historical" in st:
            return 0.0
        return silent

    def read(self, text: Optional[str]) -> Dict[str, object]:
        text = text if isinstance(text, str) else ""
        if self.modality == "mri":
            return self._read_mri(text)
        if self.modality == "echo":
            return self._read_echo(text)
        return self._read_ecg(text)

    def _read_mri(self, text: str) -> Dict[str, object]:
        lge, isch, non = (self.mentions(text, c) for c in ("lge", "ischemic", "nonischemic"))
        lge_absent = "negated" in lge and "affirmed" not in lge
        lge_seen = any(x in ("affirmed", "uncertain") for x in lge + isch + non)

        def pattern(own, other):
            if "affirmed" in own:
                return 1.0
            if "uncertain" in own:
                return U
            if lge_absent or "affirmed" in other:
                return 0.0
            if lge_seen:
                return U                             # enhancement described, pattern not named
            return 0.0 if "negated" in own else NM

        r: Dict[str, object] = {"infarct": pattern(isch, non), "nonischemic_lge": pattern(non, isch)}
        for lab in ("lv_thrombus", "lv_aneurysm", "report_mvo"):
            r[lab] = self._simple(self.mentions(text, lab), 0.0)      # rule sheet: not mentioned = absent
        r["report_lvef"] = self._value(text, "report_lvef", lo=5, hi=90)
        r["report_infarct_size"] = self._value(text, "report_infarct_size", lo=0, hi=100)
        return r

    def _read_ecg(self, text: str) -> Dict[str, object]:
        st = self.mentions(text, "af")
        return {"af": self._simple(st, 0.0)}

    def _read_echo(self, text: str) -> Dict[str, object]:
        """Wall motion: any abnormality, global hypokinesis, and the coronary territories of the walls named in
        sentences that affirm an abnormality (its own wall-to-territory words, not the training parser's segments)."""
        wma, glob = self.concepts["wma"], self.concepts["global"]
        terr = {t: re.compile(p, re.I) for t, p in self.m["territory_walls"].items()}
        st_any: List[str] = []
        found: Dict[str, List[str]] = {t: [] for t in terr}
        gl: List[str] = []
        for sent in sentences(text):
            if _SKIP.search(sent):
                continue
            st = [self.status(sent, m.start(), m.end()) for m in wma.finditer(sent)]
            st += ["negated" for _ in self.concepts["wm_normal"].finditer(sent)]     # "normal wall motion" = no abnormality
            if not st:
                continue
            st = ["uncertain"] if "uncertain" in st else st
            st_any += st
            s0 = "affirmed" if "affirmed" in st else st[0]
            if glob.search(sent) and s0 in ("affirmed", "uncertain"):
                gl.append(s0)
            for t, rx in terr.items():
                if rx.search(sent) and s0 in ("affirmed", "uncertain"):
                    found[t].append(s0)
        described = bool(st_any)
        r: Dict[str, object] = {"wma_any": self._simple(st_any, NM)}
        g = 1.0 if "affirmed" in gl else U if "uncertain" in gl else (0.0 if described else NM)
        r["global_hypokinesis"] = g
        for t in terr:
            r[t] = 1.0 if (g == 1.0 or "affirmed" in found[t]) else U if "uncertain" in found[t] else (0.0 if described else NM)
        r["report_lvef"] = self._value(text, "report_lvef", lo=5, hi=90)
        return r

    def _value(self, text: str, name: str, lo: float, hi: float) -> object:
        kw = re.compile(self.m["values"][name], re.I)
        ex = re.compile(self.m["value_exclude"][name], re.I) if self.m.get("value_exclude", {}).get(name) else None
        found, ranged = [], False
        for s in sentences(text):
            if _SKIP.search(s) or not kw.search(s) or (ex is not None and ex.search(s)):
                continue
            tail = s[kw.search(s).end():]
            if re.match(r"[^0-9%]{0,30}\d{1,2}(?:\.\d)?\s*(?:-|–|to)\s*\d{1,2}(?:\.\d)?\s*%", tail):
                ranged = True
                continue
            m = re.match(r"[^0-9%]{0,30}?(\d{1,2}(?:\.\d)?)\s*%", tail)
            if m and lo <= float(m.group(1)) <= hi:
                found.append(float(m.group(1)))
        vals = sorted(set(found))
        if ranged or len(vals) > 1:
            return U
        return vals[0] if vals else None
