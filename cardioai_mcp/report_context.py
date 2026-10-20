"""Sentence context for report-text label rules: is a mention affirmed, negated, uncertain, historical or resolved?

Used by the ECG (AF from the confirmed over-read) and MRI (CMR report) labelling code. The echo wall-motion parser
has its own clause reader (echo_wm_training/labels.py). This is NOT the automatic auditor (annotation/auto_audit.py):
the auditor stays an independent second reader with its own vocabulary.

    from cardioai_mcp.report_context import Context
    ctx = Context()                                   # or Context(cfg["labels"].get("context"))
    ctx.label("Sinus rhythm. Atrial flutter on the prior tracing has resolved.", r"atrial (fibrillation|flutter)")
    -> 0.0   (1.0 affirmed somewhere; NaN only uncertain; 0.0 mentioned but negated / historical / resolved;
              None not mentioned)

Rules (each mention is judged inside its own sentence, from the words just before and after it: by default 6 words
before and 8 after; sentences split at . ; ! ? and new lines):
  * uncertain beats negated: "cannot be excluded", "possible", "versus", "differential includes" -> uncertain;
  * negated: a negation shortly BEFORE the mention ("no", "without", "negative for", "rule out") or AFTER it
    ("not seen", "no longer present", "has resolved");
  * historical: "history of" / "h/o" BEFORE the mention, or the finding placed on another study AFTER it ("seen on
    the prior tracing", "was reported on the referring echo"). "Prior infarct" is NOT historical: old scar is
    still there;
  * current beats historical and negated: "persists", "still present", "new since", "not present on the prior
    tracing" (= new);
  * a window stops at ; : and "but", "however", "although", "except".
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional

DEFAULT: Dict[str, object] = {
    "pre_negation": r"\b(no|not|without|negative for|absence of|free of|rule out|r/o|ruled out)\b",
    "post_negation": r"\b(not (seen|present|identified|demonstrated|evident|visuali[sz]ed|detected)|"
                     r"no longer (present|seen|visible|evident|identified)|(has|have|had) (since )?resolved|resolved|"
                     r"is absent|are absent|ruled out|has normali[sz]ed)\b",
    "pre_uncertain": r"(\b(possible|possibly|probable|probably|consider|questionabl[ey]|suspect(ed)?|suspicious (for|of)|"
                     r"suggestive of|equivocal|may (represent|be)|might (represent|be)|could (represent|be)|"
                     r"thought to (represent|be)|cannot (be )?(excluded?|ruled out)|can ?not exclude|cannot rule out|"
                     r"likely .{0,30}versus)\b|\?)",
    "post_uncertain": r"(\b(cannot be (excluded|ruled out)|can ?not be (excluded|ruled out)|not (be )?excluded|"
                      r"is (possible|suspected|questioned)|not certain|uncertain)\b|\?)",
    "anywhere_uncertain": r"\b(versus|vs\.?|differential (diagnosis )?(includes?|of)|indeterminate)\b",
    # before the mention: "history of AF", "prior echo showed inferior akinesis"
    "pre_historical": r"\b(history of|h/o|hx of|known history of|previously (seen|reported|described|noted)|"
                      r"(prior|previous|referring|outside) (tracing|ecg|ekg|echo|study|cmr|mri) (showed|demonstrated|revealed))\b",
    # after the mention: "thrombus ... was reported on the referring echo", "flutter seen on the prior tracing"
    "post_historical": r"\b(was (previously )?(reported|seen|described|noted)|previously (seen|reported|described|noted)|"
                       r"(on|in|from) (the |an? )?(prior|previous|old|earlier|outside|referring) "
                       r"(tracing|ecg|ekg|study|studies|echo|echocardiogram|exam|examination|imaging|cmr|mri|scan))\b",
    # the finding is current despite a comparison: "persists", "still present", "new since", "not present on the
    # prior tracing" (= new). Checked after uncertainty, before history and negation.
    "current": r"\b(persist\w*|ongoing|still (present|seen|evident)|again (seen|noted|demonstrated)|continues|"
               r"new(ly)? (since|compared|when compared|from|relative)|not (present|seen|evident) (on|in) (the |an? )?"
               r"(prior|previous|old|earlier|outside|referring))\b",
    "terminators": r"[;:]|\b(but|however|although|except)\b",
    "pre_window_words": 6,
    "post_window_words": 8,
}

_SENT = re.compile(r"(?<=[.!?;])\s+|\n+")


def sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT.split(text or "") if s and s.strip()]


class Context:
    def __init__(self, cfg: Optional[dict] = None):
        c = {**DEFAULT, **(cfg or {})}
        self.pre_neg, self.post_neg = re.compile(c["pre_negation"], re.I), re.compile(c["post_negation"], re.I)
        self.pre_unc, self.post_unc = re.compile(c["pre_uncertain"], re.I), re.compile(c["post_uncertain"], re.I)
        self.any_unc = re.compile(c["anywhere_uncertain"], re.I)
        self.hist_pre, self.hist_post = re.compile(c["pre_historical"], re.I), re.compile(c["post_historical"], re.I)
        self.current = re.compile(c["current"], re.I)
        self.term = re.compile(c["terminators"], re.I)
        self.pre_w, self.post_w = int(c["pre_window_words"]), int(c["post_window_words"])

    def _before(self, s: str, start: int) -> str:
        w = s[:start]
        cut = list(self.term.finditer(w))
        if cut:
            w = w[cut[-1].end():]
        return " ".join(w.split()[-self.pre_w:])

    def _after(self, s: str, end: int) -> str:
        w = s[end:]
        m = self.term.search(w)
        if m:
            w = w[:m.start()]
        return " ".join(w.split()[:self.post_w])

    def status(self, sentence: str, start: int, end: int) -> str:
        pre, post = self._before(sentence, start), self._after(sentence, end)
        near = f"{pre} {sentence[start:end]} {post}"
        if self.any_unc.search(near) or self.pre_unc.search(pre) or self.post_unc.search(post):
            return "uncertain"
        if self.current.search(post) or re.search(r"\bnew(ly)?\b", pre, re.I):
            return "affirmed"
        if self.hist_pre.search(pre) or self.hist_post.search(post):
            return "historical"
        if self.pre_neg.search(pre) or self.post_neg.search(post):
            return "negated"
        return "affirmed"

    def statuses(self, text: str, pattern, skip=None) -> List[str]:
        """Status of every mention of `pattern` (str or compiled); `skip(sentence)` -> True ignores that sentence."""
        rx = re.compile(pattern, re.I) if isinstance(pattern, str) else pattern
        out: List[str] = []
        for s in sentences(text):
            if skip is not None and skip(s):
                continue
            out += [self.status(s, m.start(), m.end()) for m in rx.finditer(s)]
        return out

    @staticmethod
    def combine(st: Iterable[str]) -> Optional[float]:
        st = list(st)
        if not st:
            return None
        if "affirmed" in st:
            return 1.0
        if "uncertain" in st:
            return math.nan
        return 0.0

    def label(self, text: str, pattern, skip=None) -> Optional[float]:
        return self.combine(self.statuses(text, pattern, skip))
