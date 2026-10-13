"""High-level entry points used by the notebook and the CLI."""
from __future__ import annotations

import os

import pandas as pd

from .builder import CohortBuilder
from .concepts import concept_set_report, resolve_concept_sets
from .config import load_cohort, load_study
from .engine import Engine
from .qc import completeness, export, site_summary, table1


def build_all(engine: Engine, cohorts: list[str], study: dict | None = None, export_dir: str | None = None) -> dict:
    study = study or engine.study
    export_dir = export_dir or study.get("export_dir", "exports")
    resolve_concept_sets(engine, study["concept_sets"])
    summary, detail = concept_set_report(engine)
    export_dir and os.makedirs(export_dir, exist_ok=True)
    summary.to_csv(os.path.join(export_dir, "concept_sets_summary.csv"), index=False)
    detail.to_csv(os.path.join(export_dir, "concept_sets_detail_for_review.csv"), index=False)
    out = {}
    for key in cohorts:
        c = load_cohort(key)
        b = CohortBuilder(engine, study, c)
        att = b.build(resolve_concepts=False)
        analytic = engine.query(f"SELECT * FROM {key}_analytic")
        comp = completeness(c, analytic)
        sc = study.get("small_cell", 11)
        export(c, "attrition", att, export_dir)
        export(c, "completeness", comp, export_dir)
        export(c, "sites", site_summary(analytic, sc), export_dir)
        export(c, "table1", table1(c, analytic, sc), export_dir)
        out[key] = dict(cohort=c, attrition=att, completeness=comp, analytic=analytic)
    return out


def model_input_map(cohorts: list[str]) -> pd.DataFrame:
    """One row per trained-model input: which cohort element (or expression over elements) supplies it."""
    rows = []
    for key in cohorts:
        c = load_cohort(key)
        mi = c.raw.get("model_inputs")
        if not mi:
            continue
        by_name = {e.name: e for e in c.elements}
        for feat, spec in mi["columns"].items():
            el = by_name.get(str(spec).split()[0]) if spec else None
            rows.append(dict(cohort=key, model=mi.get("model", ""), model_input=feat, source=spec,
                             element_is_deck=bool(el and not el.aux), extractor=el.src["type"] if el else "expression",
                             concept_set=(el.src.get("set", "") if el else ""), window_days=str(el.src.get("window", "")) if el else ""))
    return pd.DataFrame(rows)


def data_dictionary(cohorts: list[str]) -> pd.DataFrame:
    """One row per deck data element with its OMOP source specification."""
    rows = []
    for key in cohorts:
        c = load_cohort(key)
        for e in c.deck_elements:
            s = e.src
            t = s["type"]
            if t == "modality":
                fb = s.get("fallback", {})
                src = f"MCP {s['modality']} table .{s['column']} (fallback: {fb.get('type')} {fb.get('set', fb.get('note', ''))})"
            elif t == "derived":
                src = "derived: " + s["expr"][:160]
            elif t == "person":
                src = f"PERSON.{s['field']}"
            elif t == "index_col":
                src = f"index definition column {s['col']}"
            else:
                tables = s.get("tables") or {"measurement": ["MEASUREMENT"], "measurement_series": ["MEASUREMENT"],
                                             "observation_value": ["OBSERVATION"], "death": ["DEATH"],
                                             "visit_count": ["VISIT_OCCURRENCE"], "new_drug_count": ["DRUG_EXPOSURE"]}.get(t, ["CONDITION/OBSERVATION"])
                src = f"{t} on {', '.join(x.upper() for x in tables)}; concept set '{s.get('set', s.get('cause_set', ''))}'"
            rows.append(dict(cohort=key, study_id=c.study_id, domain=e.domain, element=e.name, description=e.description,
                             status="REQ" if e.required else "OPT", expected_completeness=e.expected_completeness,
                             extractor=t, window_days=str(s.get("window", "")), omop_source=src,
                             fill=e.fill or "", transform=(e.transform or "")[:120], role=e.role or "input"))
    return pd.DataFrame(rows)
