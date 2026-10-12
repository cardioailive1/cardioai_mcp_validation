"""Cohort build pipeline.

    candidates  -> exclusions (attrition) -> one index per person -> site
    -> case-enriched / stratified sampling (with sampling weights)
    -> data-element extraction (wide table + long series table) -> QC

Views created (prefix = cohort key, e.g. ``ecg``):
    {k}_candidates, {k}_eligible, {k}_index, {k}_index_obs,
    {k}_analytic (wide, 1 row per patient), {k}_series (long), {k}_attrition
"""
from __future__ import annotations

import pandas as pd

from .concepts import resolve_concept_sets
from .config import Cohort
from .engine import Engine
from .extractors import build_element_sql


class CohortBuilder:
    def __init__(self, engine: Engine, study: dict, cohort: Cohort):
        self.e = engine
        self.study = study
        self.c = cohort
        self.k = cohort.key
        self.attrition: list[dict] = []

    # ------------------------------------------------------------------ steps
    def _count(self, view: str) -> int:
        return int(self.e.query(f"SELECT COUNT(*) AS n FROM {view}")["n"].iloc[0])

    def _persons(self, view: str) -> int:
        return int(self.e.query(f"SELECT COUNT(DISTINCT person_id) AS n FROM {view}")["n"].iloc[0])

    def _cohort_macros(self) -> dict:
        m = {k: v for k, v in self.c.raw.get("calendar", {}).items()}
        return m

    def candidates(self) -> None:
        sql = self.e.render(self.c.raw["index"]["sql"], self._cohort_macros())
        self.e.create_view(f"{self.k}_candidates", sql)
        self.attrition.append({"step": "Candidates meeting index definition (" + self.c.raw["index"]["description"] + ")",
                               "rows": self._count(f"{self.k}_candidates"),
                               "persons": self._persons(f"{self.k}_candidates")})

    def exclusions(self) -> None:
        prev = f"{self.k}_candidates"
        for n, ex in enumerate(self.c.raw.get("exclusions", []), start=1):
            view = f"{self.k}_ex{n}"
            sql = f"SELECT c.* FROM {prev} c WHERE NOT ({ex['where']})"
            self.e.create_view(view, self.e.render(sql, self._cohort_macros()))
            self.attrition.append({"step": f"Exclude: {ex['name']}", "rows": self._count(view), "persons": self._persons(view)})
            prev = view
        # one index per person
        rule = self.c.raw["index"].get("one_per_person", "earliest")
        if rule == "earliest":
            order = "c.index_date ASC"
        elif rule == "random":
            order = "{{hash:c.person_id || CAST(c.index_date AS STRING)}}"
        else:  # custom ORDER BY expression over c.* (e.g. prefer labelled events, then earliest)
            order = rule
        sql = f"""
        SELECT * FROM (SELECT c.*, ROW_NUMBER() OVER (PARTITION BY c.person_id ORDER BY {order}) AS _rn FROM {prev} c) t
        WHERE _rn = 1
        """
        self.e.create_view(f"{self.k}_eligible", sql)
        label = self.c.raw["index"].get("one_per_person_label", rule)
        self.attrition.append({"step": f"One index per patient ({label})", "rows": self._count(f"{self.k}_eligible"),
                               "persons": self._persons(f"{self.k}_eligible")})

    def assign_sites(self) -> None:
        site_map = self.study.get("site_map") or {}
        mapped = ""
        if site_map:
            whens = " ".join(f"WHEN e.care_site_id = {cid} THEN '{name}'" for cid, name in site_map.items())
            mapped = f"CASE {whens} ELSE NULL END, "
        sql = f"""
        SELECT e.*, COALESCE({mapped} CONCAT('STATE-', COALESCE(l.state, 'UNK'), '-CS', CAST(COALESCE(e.care_site_id, 0) AS STRING))) AS site
        FROM {self.k}_eligible e
        LEFT JOIN {{{{cdm}}}}.care_site cs ON cs.care_site_id = e.care_site_id
        LEFT JOIN {{{{cdm}}}}.location l ON l.location_id = cs.location_id
        """
        self.e.create_view(f"{self.k}_eligible_sites", sql)

    def sample(self) -> None:
        """Stratified sampling with quotas; stores inverse-probability sampling weights."""
        sp = self.c.raw.get("sampling", {"method": "random"})
        target = int(self.c.raw["target_n"])
        src = f"{self.k}_eligible_sites"
        strat = sp.get("strata_expr", "'all'")
        quotas = sp.get("quotas", {})
        quota_case = " ".join(f"WHEN stratum = '{s}' THEN {q}" for s, q in quotas.items())
        default_q = sp.get("default_quota", target)
        sql = f"""
        WITH s AS (SELECT e.*, {strat} AS stratum FROM {src} e),
        r AS (
            SELECT s.*, ROW_NUMBER() OVER (PARTITION BY stratum ORDER BY {{{{hash:s.person_id}}}}) AS _srn,
                   COUNT(*) OVER (PARTITION BY stratum) AS stratum_pool
            FROM s
        ),
        q AS (SELECT r.*, {"CASE " + quota_case + f" ELSE {default_q} END" if quota_case else str(default_q)} AS quota FROM r),
        -- quota rows first; if a stratum pool is short, backfill from remaining eligible rows (deterministic)
        ranked AS (
            SELECT q.*, ROW_NUMBER() OVER (ORDER BY CASE WHEN _srn <= quota THEN 0 ELSE 1 END, {{{{hash:q.person_id || 'fill'}}}}) AS _trn
            FROM q
        ),
        chosen AS (SELECT * FROM ranked WHERE _trn <= {target})
        SELECT chosen.*, CAST(stratum_pool AS DOUBLE) / COUNT(*) OVER (PARTITION BY stratum) AS sampling_weight
        FROM chosen
        """
        self.e.create_view(f"{self.k}_index", sql)
        alloc = self.e.query(
            f"SELECT stratum, MAX(stratum_pool) AS pool, MAX(quota) AS quota, COUNT(*) AS sampled FROM {self.k}_index GROUP BY stratum ORDER BY stratum"
        )
        self.allocation = alloc
        short = alloc[alloc.sampled < alloc.quota]
        note = "" if short.empty else " — SHORTFALL: " + ", ".join(f"{r.stratum} {r.sampled}/{r.quota}" for r in short.itertuples())
        self.attrition.append({"step": f"Sampled to target N={target} ({sp.get('method', 'random')}){note}",
                               "rows": self._count(f"{self.k}_index"), "persons": self._persons(f"{self.k}_index")})

    def observation_windows(self) -> None:
        sql = f"""
        SELECT i.person_id, i.index_date,
               MAX(CASE WHEN op.observation_period_start_date <= i.index_date AND op.observation_period_end_date >= i.index_date
                        THEN DATEDIFF(i.index_date, op.observation_period_start_date) END) AS prior_obs_days,
               MAX(CASE WHEN op.observation_period_start_date <= i.index_date AND op.observation_period_end_date >= i.index_date
                        THEN DATEDIFF(LEAST(op.observation_period_end_date, CAST('{{{{data_cutoff}}}}' AS DATE)), i.index_date) END) AS followup_days,
               MIN(DATEDIFF(CAST(d.death_date AS DATE), i.index_date)) AS death_day
        FROM {self.k}_index i
        LEFT JOIN {{{{cdm}}}}.observation_period op ON op.person_id = i.person_id
        LEFT JOIN {{{{cdm}}}}.death d ON d.person_id = i.person_id
        GROUP BY i.person_id, i.index_date
        """
        self.e.create_view(f"{self.k}_index_obs", sql)

    def extract(self) -> None:
        idx = f"{self.k}_index"
        joins, cols, series = [], [], []
        for n, el in enumerate(self.c.elements):
            if el.is_series:
                series.append(build_element_sql(el, idx, self.study))
                view = f"{self.k}_el_{n}"
                hi = int(el.src.get("window", [0, 180])[1])
                if el.src["type"] == "event_series":
                    # no event = 0 when follow-up covers the window (or patient died); else unknown
                    zero = f"CASE WHEN o.followup_days < {hi} AND o.death_day IS NULL THEN NULL ELSE 0 END" if hi > 0 else "0"
                else:
                    zero = "NULL"  # no measurement = missing
                self.e.create_view(view, f"""
                    SELECT i.person_id, i.index_date, COALESCE(s.n, {zero}) AS value
                    FROM {idx} i JOIN {idx}_obs o ON o.person_id = i.person_id AND o.index_date = i.index_date
                    LEFT JOIN (SELECT person_id, index_date, COUNT(*) AS n FROM ({series[-1]}) x GROUP BY person_id, index_date) s
                      ON s.person_id = i.person_id AND s.index_date = i.index_date""")
            else:
                sql = build_element_sql(el, idx, self.study)
                if sql is None:
                    continue
                view = f"{self.k}_el_{n}"
                self.e.create_view(view, sql)
            alias = f"x{n}"
            joins.append(f"LEFT JOIN {view} {alias} ON {alias}.person_id = i.person_id AND {alias}.index_date = i.index_date")
            if el.src["type"] in ("outcome", "death"):
                cols.append(f"{alias}.value_days AS {el.name}_days")
            cols.append(f"{alias}.value AS {el.name}")
        # stage 1: raw element values
        keep = self.c.raw.get("index", {}).get("keep_columns", [])
        base_cols = ["i.person_id", "i.index_date", "i.site", "i.stratum", "i.sampling_weight"] + [f"i.{k}" for k in keep]
        self.e.create_view(
            f"{self.k}_raw",
            f"SELECT {', '.join(base_cols + cols)} FROM {idx} i " + " ".join(joins),
        )
        # stage 2: fills / transforms / derived (in YAML order, so later ones can use earlier)
        colnames = [c.split(".")[-1].split(" AS ")[-1] for c in base_cols + cols]
        current = f"{self.k}_raw"
        for n, el in enumerate(self.c.elements):
            expr = None
            if el.src["type"] == "derived":
                expr = el.src["expr"]
                if el.transform:
                    expr = el.transform.replace("{v}", f"({expr})")
            elif el.fill or el.transform:
                expr = el.name
                if el.fill:
                    expr = f"COALESCE({el.name}, {el.fill})"
                if el.transform:
                    expr = el.transform.replace("{v}", f"({expr})")
            if expr is None:
                continue
            view = f"{self.k}_d{n}"
            if el.src["type"] == "derived":
                select = [f"r.{c}" for c in colnames] + [f"{expr} AS {el.name}"]
                colnames.append(el.name)
            else:  # replace column in place (no SELECT * EXCEPT — not in OSS Spark SQL)
                select = [f"{expr} AS {c}" if c == el.name else f"r.{c}" for c in colnames]
            self.e.create_view(view, f"SELECT {', '.join(select)} FROM {current} r")
            current = view
        self.e.create_view(f"{self.k}_analytic", f"SELECT * FROM {current}")
        if series:
            self.e.create_view(f"{self.k}_series", " UNION ALL ".join(series))

    # ------------------------------------------------------------------ main
    def build(self, resolve_concepts: bool = True) -> pd.DataFrame:
        if resolve_concepts:
            resolve_concept_sets(self.e, self.study["concept_sets"])
        self.candidates()
        self.exclusions()
        self.assign_sites()
        self.sample()
        self.observation_windows()
        self.extract()
        att = pd.DataFrame(self.attrition)
        return att
