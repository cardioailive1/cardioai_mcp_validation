"""Resolve YAML concept sets into the ``cs_members`` view — one set-based query.

cs_members(set_name, concept_id, is_standard, needs_review)

All code patterns of all sets go into one small inline table; matching against
CONCEPT is done with equi-joins (exact codes, and fixed-length prefixes for
'I21%'-style patterns), so the 7M+ row vocabulary is scanned a handful of times
rather than once per set.  Matched source concepts (ICD10CM, ...) are kept so
extractors can match *_source_concept_id too; standard concepts reached via
'Maps to' (+ descendants via CONCEPT_ANCESTOR when the set asks) are added.
"""
from __future__ import annotations

from .engine import Engine


def _seed_rows(concept_sets: dict) -> tuple[list[tuple], list[tuple]]:
    rows, flags = [], []
    for name, spec in concept_sets.items():
        flags.append((name, 1 if spec.get("descendants") else 0))
        dom = spec.get("domain", "")
        for inc in spec.get("include", []):
            vocab, review = inc["vocab"], 1 if inc.get("review") else 0
            for code in inc.get("codes", []):
                if "%" not in code:
                    rows.append((name, vocab, code, "eq", len(code), review, dom))
                elif code.endswith("%") and "%" not in code[:-1]:
                    rows.append((name, vocab, code[:-1], "prefix", len(code) - 1, review, dom))
                else:
                    rows.append((name, vocab, code, "like", 0, review, dom))
            for pat in inc.get("name_like", []):
                rows.append((name, vocab, pat.lower(), "name", 0, 1, dom))
            for cid in inc.get("concept_ids", []):
                rows.append((name, vocab, str(cid), "id", 0, review, dom))
    return rows, flags


def _values(rows: list[tuple], cols: list[str], alias: str) -> str:
    def lit(v):
        return str(v) if isinstance(v, int) else "'" + str(v).replace("'", "''") + "'"

    body = ", ".join("(" + ", ".join(lit(v) for v in r) + ")" for r in rows)
    return f"SELECT * FROM VALUES {body} AS {alias}({', '.join(cols)})"


def concept_sets_sql(concept_sets: dict) -> tuple[str, str, str]:
    rows, flags = _seed_rows(concept_sets)
    patt = _values(rows, ["set_name", "vocab", "code", "mode", "len", "review", "dom"], "p")
    flag = _values(flags, ["set_name", "desc_flag"], "f")
    lens = sorted({r[4] for r in rows if r[3] == "prefix"})
    prefix_parts = [
        f"SELECT p.set_name, c.concept_id, c.standard_concept, p.review FROM cs_patterns p "
        f"JOIN {{{{vocab}}}}.concept c ON c.vocabulary_id = p.vocab AND SUBSTRING(c.concept_code, 1, {n}) = p.code "
        f"WHERE p.mode = 'prefix' AND p.len = {n}"
        for n in lens
    ]
    seed = " UNION ALL ".join(
        [
            "SELECT p.set_name, c.concept_id, c.standard_concept, p.review FROM cs_patterns p "
            "JOIN {{vocab}}.concept c ON c.vocabulary_id = p.vocab AND c.concept_code = p.code WHERE p.mode = 'eq'",
            *prefix_parts,
            "SELECT p.set_name, c.concept_id, c.standard_concept, p.review FROM cs_patterns p "
            "JOIN {{vocab}}.concept c ON c.vocabulary_id = p.vocab AND c.concept_code LIKE p.code WHERE p.mode = 'like'",
            "SELECT p.set_name, c.concept_id, c.standard_concept, p.review FROM cs_patterns p "
            "JOIN {{vocab}}.concept c ON c.vocabulary_id = p.vocab AND LOWER(c.concept_name) LIKE p.code "
            "AND (p.dom = '' OR c.domain_id = p.dom) WHERE p.mode = 'name'",
            "SELECT p.set_name, c.concept_id, c.standard_concept, p.review FROM cs_patterns p "
            "JOIN {{vocab}}.concept c ON CAST(c.concept_id AS STRING) = p.code WHERE p.mode = 'id'",
        ]
    )
    members = f"""
    WITH seed AS ({seed}),
    mapped AS (
        SELECT s.set_name, s.concept_id, s.review FROM seed s WHERE s.standard_concept IN ('S', 'C')
        UNION ALL
        SELECT s.set_name, cr.concept_id_2 AS concept_id, s.review
        FROM seed s JOIN {{{{vocab}}}}.concept_relationship cr
          ON cr.concept_id_1 = s.concept_id AND cr.relationship_id = 'Maps to'
         AND (cr.invalid_reason IS NULL OR cr.invalid_reason = '')
    ),
    expanded AS (
        SELECT set_name, concept_id, review FROM mapped
        UNION ALL
        SELECT m.set_name, ca.descendant_concept_id AS concept_id, m.review
        FROM mapped m JOIN cs_flags f ON f.set_name = m.set_name AND f.desc_flag = 1
        JOIN {{{{vocab}}}}.concept_ancestor ca ON ca.ancestor_concept_id = m.concept_id
    ),
    allc AS (
        SELECT set_name, concept_id, 1 AS is_standard, review FROM expanded
        UNION ALL
        SELECT set_name, concept_id, 0 AS is_standard, review FROM seed
        WHERE standard_concept IS NULL OR standard_concept NOT IN ('S', 'C')
    )
    SELECT set_name, concept_id, MAX(is_standard) AS is_standard, MAX(review) AS needs_review
    FROM allc GROUP BY set_name, concept_id
    """
    return patt, flag, members


def resolve_concept_sets(engine: Engine, concept_sets: dict, only: set[str] | None = None) -> None:
    sets = {k: v for k, v in concept_sets.items() if only is None or k in only}
    patt, flag, members = concept_sets_sql(sets)
    engine.create_view("cs_patterns", patt)
    engine.create_view("cs_flags", flag)
    engine.create_view("cs_members", members)


def concept_set_report(engine: Engine):
    """Counts per set and the concepts that need clinical review (export for sign-off)."""
    summary = engine.query(
        """
        SELECT m.set_name, COUNT(*) AS n_concepts, SUM(m.is_standard) AS n_standard,
               SUM(m.needs_review) AS n_needs_review
        FROM cs_members m GROUP BY m.set_name ORDER BY m.set_name
        """
    )
    detail = engine.query(
        """
        SELECT m.set_name, c.concept_id, c.concept_name, c.vocabulary_id, c.concept_code,
               c.domain_id, m.is_standard, m.needs_review
        FROM cs_members m JOIN {{vocab}}.concept c ON c.concept_id = m.concept_id
        ORDER BY m.set_name, c.vocabulary_id, c.concept_code
        """
    )
    return summary, detail
