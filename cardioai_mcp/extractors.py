"""SQL generators for data elements (Spark SQL).

Every extractor returns a SELECT producing ``person_id, index_date, value``
(plus ``value_days`` for outcomes) for each row of the cohort index view
``{idx}``.  Windows are in days relative to index_date:
``DATEDIFF(event_date, index_date) BETWEEN lo AND hi``.

Missing-data semantics (important for completeness QC):
* history flags  -> NULL when the person has < min_prior_observation_days of
  observation before index (absence of a code is not evidence of absence)
* outcome flags  -> NULL when follow-up is shorter than the window and no event
  / death occurred (censored); 0/1 otherwise
* measurements   -> NULL when no plausible value inside the window
"""
from __future__ import annotations

import re

from .config import Element

# OMOP event tables: (table, concept col, source concept col, date col)
EVENT_TABLES = {
    "condition": ("condition_occurrence", "condition_concept_id", "condition_source_concept_id", "condition_start_date"),
    "observation": ("observation", "observation_concept_id", "observation_source_concept_id", "observation_date"),
    "procedure": ("procedure_occurrence", "procedure_concept_id", "procedure_source_concept_id", "procedure_date"),
    "drug": ("drug_exposure", "drug_concept_id", "drug_source_concept_id", "drug_exposure_start_date"),
    "measurement": ("measurement", "measurement_concept_id", "measurement_source_concept_id", "measurement_date"),
    "device": ("device_exposure", "device_concept_id", "device_source_concept_id", "device_exposure_start_date"),
}

DEFAULT_TABLES = {
    "Condition": ["condition", "observation"],
    "Observation": ["observation", "condition"],
    "Procedure": ["procedure"],
    "Drug": ["drug"],
    "Measurement": ["measurement"],
}


def events_sql(cs: str, tables: list[str]) -> str:
    """Union of events (person_id, event_date, visit_occurrence_id) in concept set ``cs``."""
    parts = []
    for t in tables:
        tbl, ccol, scol, dcol = EVENT_TABLES[t]
        parts.append(
            f"SELECT person_id, CAST({dcol} AS DATE) AS event_date, visit_occurrence_id "
            f"FROM {{{{cdm}}}}.{tbl} WHERE {ccol} IN {{{{cs:{cs}}}}} OR {scol} IN {{{{cs:{cs}}}}}"
        )
    return " UNION ALL ".join(parts)


def _win(col: str, w: list[int]) -> str:
    return f"DATEDIFF({col}, i.index_date) BETWEEN {int(w[0])} AND {int(w[1])}"


def _value_expr(el: Element, study: dict) -> str:
    conv = study.get("unit_conversions", {}).get(el.src.get("convert") or el.name)
    if not conv:
        return "m.value_as_number"
    whens = []
    for c in conv:
        likes = " OR ".join(f"LOWER(m.unit_source_value) LIKE '{u}'" for u in c.get("unit_source_like", []))
        cond = f"m.unit_concept_id = {c['unit_concept_id']}" + (f" OR {likes}" if likes else "")
        whens.append(f"WHEN {cond} THEN m.value_as_number * {c['factor']}")
    return "CASE " + " ".join(whens) + " ELSE m.value_as_number END"


# --------------------------------------------------------------------------- extractors
def x_index_col(el, idx, study):
    return f"SELECT i.person_id, i.index_date, i.{el.src['col']} AS value FROM {idx} i"


def x_person(el, idx, study):
    field = el.src["field"]
    if field == "age":
        expr = "YEAR(i.index_date) - p.year_of_birth"
    elif field == "gender":
        expr = "CASE WHEN p.gender_concept_id = 8507 THEN 'Male' WHEN p.gender_concept_id = 8532 THEN 'Female' ELSE NULL END"
    elif field == "race":
        # race + Hispanic ethnicity combined (deck: "Race / ethnicity")
        expr = (
            "CASE WHEN p.ethnicity_concept_id = 38003563 THEN 'Hispanic' "
            "WHEN rc.concept_name IS NULL OR p.race_concept_id = 0 THEN NULL ELSE rc.concept_name END"
        )
    else:
        raise ValueError(field)
    return (
        f"SELECT i.person_id, i.index_date, {expr} AS value FROM {idx} i "
        f"JOIN {{{{cdm}}}}.person p ON p.person_id = i.person_id "
        f"LEFT JOIN {{{{vocab}}}}.concept rc ON rc.concept_id = p.race_concept_id"
    )


def x_measurement(el, idx, study):
    s = el.src
    w = s.get("window", [-365, 0])
    lo, hi = s.get("range", [None, None])
    v = _value_expr(el, study)
    pick = s.get("pick", "nearest")
    rng = ""
    if lo is not None:
        rng += f" AND ({v}) >= {lo}"
    if hi is not None:
        rng += f" AND ({v}) <= {hi}"
    base = (
        f"FROM {idx} i JOIN {{{{cdm}}}}.measurement m ON m.person_id = i.person_id "
        f"WHERE (m.measurement_concept_id IN {{{{cs:{s['set']}}}}} OR m.measurement_source_concept_id IN {{{{cs:{s['set']}}}}}) "
        f"AND {_win('m.measurement_date', w)} AND m.value_as_number IS NOT NULL{rng}"
    )
    if pick in ("max", "min"):
        return f"SELECT i.person_id, i.index_date, {pick.upper()}({v}) AS value {base} GROUP BY i.person_id, i.index_date"
    order = {
        "nearest": "ABS(DATEDIFF(m.measurement_date, i.index_date)), m.measurement_date DESC",
        "latest": "m.measurement_date DESC",
        "earliest": "m.measurement_date ASC",
    }[pick]
    return (
        f"SELECT person_id, index_date, value FROM (SELECT i.person_id, i.index_date, {v} AS value, "
        f"ROW_NUMBER() OVER (PARTITION BY i.person_id, i.index_date ORDER BY {order}) AS rn {base}) t WHERE rn = 1"
    )


def x_flag(el, idx, study):
    """History / exposure flag; NULL if insufficient prior observation."""
    s = el.src
    w = s.get("window", [-36500, 0])
    tables = s.get("tables") or DEFAULT_TABLES[s.get("domain", "Condition")]
    need_obs = s.get("require_lookback", w[0] < 0)
    obs = "CASE WHEN o.prior_obs_days < {{min_prior_observation_days}} THEN NULL ELSE 0 END" if need_obs else "0"
    return f"""
    SELECT i.person_id, i.index_date,
           CASE WHEN e.person_id IS NOT NULL THEN 1 ELSE {obs} END AS value
    FROM {idx} i
    JOIN {idx}_obs o ON o.person_id = i.person_id AND o.index_date = i.index_date
    LEFT JOIN (
        SELECT DISTINCT i.person_id, i.index_date
        FROM {idx} i JOIN ({events_sql(s['set'], tables)}) ev ON ev.person_id = i.person_id
        WHERE {_win('ev.event_date', w)}
    ) e ON e.person_id = i.person_id AND e.index_date = i.index_date
    """


def x_days_since(el, idx, study):
    """Days from most recent event (<= index) to index; NULL if never."""
    s = el.src
    w = s.get("window", [-36500, 0])
    tables = s.get("tables") or DEFAULT_TABLES[s.get("domain", "Condition")]
    return f"""
    SELECT i.person_id, i.index_date, MIN(DATEDIFF(i.index_date, ev.event_date)) AS value
    FROM {idx} i JOIN ({events_sql(s['set'], tables)}) ev ON ev.person_id = i.person_id
    WHERE {_win('ev.event_date', w)}
    GROUP BY i.person_id, i.index_date
    """


def _visit_filter(s: dict) -> tuple[str, str]:
    vc = s.get("visit_concepts")
    if not vc:
        return "", ""
    join = "JOIN {{cdm}}.visit_occurrence v ON v.visit_occurrence_id = ev.visit_occurrence_id"
    return join, f" AND v.visit_concept_id IN ({', '.join(str(x) for x in vc)})"


def x_outcome(el, idx, study):
    """Binary outcome in (lo, hi] days after index + time-to-event in value_days."""
    s = el.src
    w = s.get("window", [1, 365])
    tables = s.get("tables") or DEFAULT_TABLES[s.get("domain", "Condition")]
    vjoin, vfilt = _visit_filter(s)
    return f"""
    SELECT i.person_id, i.index_date,
           CASE WHEN e.first_day IS NOT NULL THEN 1
                WHEN o.followup_days >= {w[1]} THEN 0
                WHEN o.death_day IS NOT NULL AND o.death_day <= {w[1]} THEN 0
                ELSE NULL END AS value,
           COALESCE(e.first_day, LEAST(o.followup_days, {w[1]}, COALESCE(o.death_day, {w[1]}))) AS value_days
    FROM {idx} i
    JOIN {idx}_obs o ON o.person_id = i.person_id AND o.index_date = i.index_date
    LEFT JOIN (
        SELECT i.person_id, i.index_date, MIN(DATEDIFF(ev.event_date, i.index_date)) AS first_day
        FROM {idx} i JOIN ({events_sql(s['set'], tables)}) ev ON ev.person_id = i.person_id
        {vjoin}
        WHERE {_win('ev.event_date', w)}{vfilt}
        GROUP BY i.person_id, i.index_date
    ) e ON e.person_id = i.person_id AND e.index_date = i.index_date
    """


def x_death(el, idx, study):
    s = el.src
    w = s.get("window", [1, 365])
    cause = f" AND (d.cause_concept_id IN {{{{cs:{s['cause_set']}}}}} OR d.cause_source_concept_id IN {{{{cs:{s['cause_set']}}}}})" if s.get("cause_set") else ""
    return f"""
    SELECT i.person_id, i.index_date,
           CASE WHEN dd.person_id IS NOT NULL THEN 1
                WHEN o.followup_days >= {w[1]} OR o.death_day IS NOT NULL THEN 0
                ELSE NULL END AS value,
           COALESCE(dd.day, LEAST(o.followup_days, {w[1]})) AS value_days
    FROM {idx} i
    JOIN {idx}_obs o ON o.person_id = i.person_id AND o.index_date = i.index_date
    LEFT JOIN (
        SELECT i.person_id, i.index_date, DATEDIFF(CAST(d.death_date AS DATE), i.index_date) AS day
        FROM {idx} i JOIN {{{{cdm}}}}.death d ON d.person_id = i.person_id
        WHERE {_win('CAST(d.death_date AS DATE)', w)}{cause}
    ) dd ON dd.person_id = i.person_id AND dd.index_date = i.index_date
    """


def x_visit_count(el, idx, study):
    """Number of distinct visits of given type(s) in window, optionally with a dx in set."""
    s = el.src
    w = s.get("window", [1, 180])
    vc = ", ".join(str(x) for x in s["visit_concepts"])
    dx = ""
    if s.get("set"):
        dx = (
            f" AND EXISTS (SELECT 1 FROM {{{{cdm}}}}.condition_occurrence co WHERE co.visit_occurrence_id = v.visit_occurrence_id "
            f"AND (co.condition_concept_id IN {{{{cs:{s['set']}}}}} OR co.condition_source_concept_id IN {{{{cs:{s['set']}}}}}))"
        )
    return f"""
    SELECT i.person_id, i.index_date,
           CASE WHEN o.followup_days < {w[1]} AND o.death_day IS NULL THEN NULL ELSE COALESCE(c.n, 0) END AS value
    FROM {idx} i
    JOIN {idx}_obs o ON o.person_id = i.person_id AND o.index_date = i.index_date
    LEFT JOIN (
        SELECT i.person_id, i.index_date, COUNT(DISTINCT v.visit_occurrence_id) AS n
        FROM {idx} i JOIN {{{{cdm}}}}.visit_occurrence v ON v.person_id = i.person_id
        WHERE v.visit_concept_id IN ({vc}) AND {_win('CAST(v.visit_start_date AS DATE)', w)}{dx}
        GROUP BY i.person_id, i.index_date
    ) c ON c.person_id = i.person_id AND c.index_date = i.index_date
    """


def x_new_drug_count(el, idx, study):
    """Medication changes: count of new drug-ingredient starts in window not exposed in prior 180 d."""
    s = el.src
    w = s.get("window", [1, 180])
    return f"""
    SELECT i.person_id, i.index_date,
           CASE WHEN o.followup_days < {w[1]} AND o.death_day IS NULL THEN NULL ELSE COALESCE(c.n, 0) END AS value
    FROM {idx} i
    JOIN {idx}_obs o ON o.person_id = i.person_id AND o.index_date = i.index_date
    LEFT JOIN (
        SELECT i.person_id, i.index_date, COUNT(DISTINCT de.drug_concept_id) AS n
        FROM {idx} i JOIN {{{{cdm}}}}.drug_exposure de ON de.person_id = i.person_id
        WHERE de.drug_concept_id IN {{{{cs:{s['set']}}}}} AND {_win('CAST(de.drug_exposure_start_date AS DATE)', w)}
          AND NOT EXISTS (
            SELECT 1 FROM {{{{cdm}}}}.drug_exposure p
            WHERE p.person_id = de.person_id AND p.drug_concept_id = de.drug_concept_id
              AND DATEDIFF(CAST(p.drug_exposure_start_date AS DATE), i.index_date) BETWEEN -180 AND 0)
        GROUP BY i.person_id, i.index_date
    ) c ON c.person_id = i.person_id AND c.index_date = i.index_date
    """


def x_observation_value(el, idx, study):
    s = el.src
    w = s.get("window", [-365, 0])
    return f"""
    SELECT person_id, index_date, value FROM (
        SELECT i.person_id, i.index_date,
               COALESCE(vc.concept_name, ob.value_as_string, CAST(ob.value_as_number AS STRING)) AS value,
               ROW_NUMBER() OVER (PARTITION BY i.person_id, i.index_date
                                  ORDER BY ABS(DATEDIFF(ob.observation_date, i.index_date)), ob.observation_date DESC) AS rn
        FROM {idx} i JOIN {{{{cdm}}}}.observation ob ON ob.person_id = i.person_id
        LEFT JOIN {{{{vocab}}}}.concept vc ON vc.concept_id = ob.value_as_concept_id AND ob.value_as_concept_id <> 0
        WHERE (ob.observation_concept_id IN {{{{cs:{s['set']}}}}} OR ob.observation_source_concept_id IN {{{{cs:{s['set']}}}}})
          AND {_win('ob.observation_date', w)}
    ) t WHERE rn = 1 AND value IS NOT NULL
    """


def _note_title(s: dict, study: dict) -> str:
    key = s.get("note")
    pat = study["modality"]["notes"][key] if key else s.get("title_regex")
    return f" AND n.note_title RLIKE '{pat}'" if pat else ""


def x_note_regex(el, idx, study):
    """NLP fall-back: regexp_extract from the note nearest to index in window."""
    s = el.src
    w = s.get("window", [0, 0])
    pat = s["pattern"].replace("'", "''")
    grp = int(s.get("group", 1))
    cast = s.get("cast")
    val = f"REGEXP_EXTRACT(n.note_text, '{pat}', {grp})"
    if s.get("lower", True) and cast != "double":
        val = f"LOWER({val})"
    outer = "CAST(value AS DOUBLE)" if cast == "double" else "value"
    return f"""
    SELECT person_id, index_date, {outer} AS value FROM (
        SELECT i.person_id, i.index_date, {val} AS value,
               ROW_NUMBER() OVER (PARTITION BY i.person_id, i.index_date
                                  ORDER BY ABS(DATEDIFF(n.note_date, i.index_date)), n.note_date DESC) AS rn
        FROM {idx} i JOIN {{{{cdm}}}}.note n ON n.person_id = i.person_id
        WHERE {_win('n.note_date', w)}{_note_title(s, study)} AND n.note_text RLIKE '{pat}'
    ) t WHERE rn = 1 AND value IS NOT NULL AND value <> ''
    """


def x_note_text(el, idx, study):
    """Aux: (truncated) text of the nearest matching note — feeds derived NLP elements."""
    s = el.src
    w = s.get("window", [0, 0])
    n = int(s.get("max_chars", 4000))
    return f"""
    SELECT person_id, index_date, value FROM (
        SELECT i.person_id, i.index_date, LOWER(SUBSTRING(n.note_text, 1, {n})) AS value,
               ROW_NUMBER() OVER (PARTITION BY i.person_id, i.index_date
                                  ORDER BY ABS(DATEDIFF(n.note_date, i.index_date)), n.note_date DESC) AS rn
        FROM {idx} i JOIN {{{{cdm}}}}.note n ON n.person_id = i.person_id
        WHERE {_win('n.note_date', w)}{_note_title(s, study)}
    ) t WHERE rn = 1
    """


def _schema_macro(name: str) -> str:
    """'{modality}.x' / '{work}.x' in modality_tables.yaml -> the engine's '{{modality}}' / '{{work}}' macros."""
    return re.sub(r"(?<!\{)\{(\w+)\}(?!\})", r"{{\1}}", name)


def x_modality(el, idx, study):
    """Column from MCP modality table (ECG / echo) nearest to index; falls back when disabled."""
    s = el.src
    mod = study["modality"][s["modality"]]
    if not mod.get("enabled"):
        fb = s.get("fallback")
        if fb:
            return build_element_sql(Element(el.name, el.domain, "", False, 0, fb), idx, study)
        return f"SELECT i.person_id, i.index_date, CAST(NULL AS STRING) AS value FROM {idx} i"
    table_key = s.get("table", "table" if "table" in mod else "study_table")
    cols = mod["columns"]
    dt = cols.get("acquisition_datetime") or cols.get("study_datetime")
    w = s.get("window", [0, 0])
    return f"""
    SELECT person_id, index_date, value FROM (
        SELECT i.person_id, i.index_date, t.{cols[s['column']]} AS value,
               ROW_NUMBER() OVER (PARTITION BY i.person_id, i.index_date
                                  ORDER BY ABS(DATEDIFF(CAST(t.{dt} AS DATE), i.index_date))) AS rn
        FROM {idx} i JOIN {_schema_macro(mod[table_key])} t ON t.{cols['person_id']} = i.person_id
        WHERE {_win(f'CAST(t.{dt} AS DATE)', w)}
    ) x WHERE rn = 1
    """


def x_null(el, idx, study):
    return f"SELECT i.person_id, i.index_date, CAST(NULL AS STRING) AS value FROM {idx} i"


# ---- series (long format) ------------------------------------------------------
def series_measurement_sql(el, idx, study):
    s = el.src
    w = s.get("window", [-30, 180])
    lo, hi = s.get("range", [None, None])
    v = _value_expr(el, study)
    rng = (f" AND ({v}) >= {lo}" if lo is not None else "") + (f" AND ({v}) <= {hi}" if hi is not None else "")
    return f"""
    SELECT i.person_id, i.index_date, '{el.name}' AS element,
           CAST(COALESCE(m.measurement_datetime, m.measurement_date) AS TIMESTAMP) AS event_time,
           DATEDIFF(m.measurement_date, i.index_date) AS day, {v} AS value_num
    FROM {idx} i JOIN {{{{cdm}}}}.measurement m ON m.person_id = i.person_id
    WHERE (m.measurement_concept_id IN {{{{cs:{s['set']}}}}} OR m.measurement_source_concept_id IN {{{{cs:{s['set']}}}}})
      AND {_win('m.measurement_date', w)} AND m.value_as_number IS NOT NULL{rng}
    """


def series_event_sql(el, idx, study):
    s = el.src
    w = s.get("window", [1, 180])
    tables = s.get("tables") or DEFAULT_TABLES[s.get("domain", "Condition")]
    vjoin, vfilt = _visit_filter(s)
    return f"""
    SELECT DISTINCT i.person_id, i.index_date, '{el.name}' AS element,
           CAST(ev.event_date AS TIMESTAMP) AS event_time,
           DATEDIFF(ev.event_date, i.index_date) AS day, CAST(NULL AS DOUBLE) AS value_num
    FROM {idx} i JOIN ({events_sql(s['set'], tables)}) ev ON ev.person_id = i.person_id
    {vjoin}
    WHERE {_win('ev.event_date', w)}{vfilt}
    """


EXTRACTORS = {
    "index_col": x_index_col,
    "person": x_person,
    "measurement": x_measurement,
    "flag": x_flag,
    "days_since": x_days_since,
    "outcome": x_outcome,
    "death": x_death,
    "visit_count": x_visit_count,
    "new_drug_count": x_new_drug_count,
    "observation_value": x_observation_value,
    "note_regex": x_note_regex,
    "note_text": x_note_text,
    "modality": x_modality,
    "empty": x_null,
}


def build_element_sql(el: Element, idx: str, study: dict) -> str | None:
    t = el.src["type"]
    if t == "derived":
        return None  # evaluated in the final SELECT
    if t == "sql":
        return el.src["sql"].replace("{idx}", idx)
    if t == "measurement_series":
        return series_measurement_sql(el, idx, study)
    if t == "event_series":
        return series_event_sql(el, idx, study)
    return EXTRACTORS[t](el, idx, study)
