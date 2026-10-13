"""Load and validate YAML configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT, "config")


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class Element:
    name: str
    domain: str
    description: str
    required: bool
    expected_completeness: float
    src: dict
    fill: str | None = None
    transform: str | None = None
    aux: bool = False          # helper column, not one of the deck's data elements
    role: str | None = None    # 'reference' | 'input' | 'outcome'

    @property
    def is_series(self) -> bool:
        return self.src.get("type") in ("measurement_series", "event_series")


@dataclass
class Cohort:
    key: str
    raw: dict
    elements: list[Element] = field(default_factory=list)

    def __getattr__(self, item: str) -> Any:  # convenience access to raw keys
        try:
            return self.raw[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    @property
    def deck_elements(self) -> list[Element]:
        return [e for e in self.elements if not e.aux]


def load_study(config_dir: str = CONFIG_DIR) -> dict:
    study = _load(os.path.join(config_dir, "study.yaml"))
    study["concept_sets"] = _load(os.path.join(config_dir, "concept_sets.yaml"))
    study["modality"] = _load(os.path.join(config_dir, "modality_tables.yaml"))
    return study


def active_plan(config_dir: str = CONFIG_DIR, plan: str | None = None) -> tuple[str, dict, dict]:
    """(plan key, plan spec, margins) from study.yaml; env CARDIOAI_PLAN overrides."""
    st = _load(os.path.join(config_dir, "study.yaml"))
    key = plan or os.environ.get("CARDIOAI_PLAN") or st.get("plan", "B")
    return key, (st.get("plans") or {}).get(key, {"rule": "goal", "sizes": {}}), st.get("margins", {})


def load_cohort(key: str, config_dir: str = CONFIG_DIR, plan: str | None = None) -> Cohort:
    raw = _load(os.path.join(config_dir, "cohorts", f"{key}.yaml"))
    pkey, pspec, margins = active_plan(config_dir, plan)
    raw["deck_target_n"] = raw.get("target_n")
    size = (pspec.get("sizes") or {}).get(key)
    if size:  # plan overrides deck N and sampling quotas
        raw["target_n"] = size["target_n"]
        raw.setdefault("sampling", {})["quotas"] = size["quotas"]
    raw["plan"] = {"key": pkey, "rule": pspec.get("rule", "goal"), "margins": margins}
    elements = []
    for e in raw.get("elements", []):
        elements.append(
            Element(
                name=e["name"],
                domain=e.get("domain", "aux"),
                description=e.get("description", ""),
                required=bool(e.get("required", False)),
                expected_completeness=float(e.get("expected_completeness", 0) or 0),
                src=e["src"],
                fill=e.get("fill"),
                transform=e.get("transform"),
                aux=bool(e.get("aux", False)),
                role=e.get("role"),
            )
        )
    c = Cohort(key=key, raw=raw, elements=elements)
    validate_cohort(c)
    return c


def validate_cohort(c: Cohort) -> None:
    names = [e.name for e in c.elements]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"{c.key}: duplicate element names {sorted(dupes)}")
    exp = c.raw.get("expected", {})
    deck = c.deck_elements
    if exp:
        if len(deck) != exp["total_elements"]:
            raise ValueError(f"{c.key}: {len(deck)} elements, deck says {exp['total_elements']}")
        n_req = sum(e.required for e in deck)
        if n_req != exp["required"]:
            raise ValueError(f"{c.key}: {n_req} required, deck says {exp['required']}")
        avg = 100 * sum(e.expected_completeness for e in deck) / len(deck)
        if abs(avg - exp["avg_completeness"]) > 0.05 + 1e-9:  # deck reports to 0.1; 85.95 -> 86.0 must pass on every float platform
            raise ValueError(f"{c.key}: avg completeness {avg:.2f} != deck {exp['avg_completeness']}")
    for e in c.elements:
        for key in ("pattern", "title_regex"):
            pat = e.src.get(key)
            if pat and "\\" in pat:
                raise ValueError(
                    f"{c.key}.{e.name}: regex must not contain backslashes (Spark and DuckDB "
                    "escape them differently) — use [0-9], [ ] etc."
                )
