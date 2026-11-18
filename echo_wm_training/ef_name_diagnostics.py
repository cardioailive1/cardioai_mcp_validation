"""Read-only EF result-name diversity diagnostic for flagged Mayo TTE groups.

This analyzes *names*, not clinical equivalence. No SQL writes or exports.
Per-exam detail contains identifiers and must remain inside MCP.
"""
from __future__ import annotations

import re

import pandas as pd

from .exam_comparison import _clean_string
from .mcp_echo import parse_lvef


def diagnose_within_method_names(measurement_rows: pd.DataFrame,
                                 candidate_exams: pd.DataFrame,
                                 results_cfg: dict,
                                 *, include_both: bool = False
                                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify within-method conflicts as same exact name, different names, both.

    Uses the same plausible-EF parser, method regex precedence, and group keys
    as the unified candidate builder. Distinct names are compared using their
    individual numeric values (rather than only per-name medians). Thus
    'different_names' means >=2 distinct result names in a common method
    family have values differing by more than the configured threshold;
    it does NOT imply those names describe different clinical methods.

    A matched ACC/ORD candidate can contain two groups; each is analyzed
    separately. The candidate classification is the union of those groups,
    never a cross-identifier comparison.
    """
    details_columns = [
        "person_id", "index_date", "group_key", "partner_key", "record_source",
        "candidate_category", "exact_name_conflict", "different_names_conflict",
        "name_conflict_class", "method_families_with_conflict",
    ]
    summary_columns = ["record_source", "name_conflict_class", "groups"]
    allowed = {"within_method"}
    if include_both:
        allowed.add("both")
    targets = candidate_exams.loc[
        candidate_exams["lvef_disagreement_category"].isin(allowed)
    ].copy()
    if targets.empty:
        return pd.DataFrame(columns=details_columns), pd.DataFrame(columns=summary_columns)
    if measurement_rows.empty:
        raise ValueError("No measurement rows supplied for flagged examinations")
    required = {"person_id", "index_date", "echo_test_fpk",
                "accession_number", "order_proc_id", "metric",
                "result_name", "result_label", "result_value"}
    missing = required - set(measurement_rows.columns)
    if missing:
        raise ValueError(f"Missing measurement fields: {sorted(missing)}")
    data = measurement_rows.loc[measurement_rows.metric == "lvef"].copy()
    data["index_date"] = pd.to_datetime(data.index_date).dt.date
    acc = _clean_string(data.accession_number)
    order = _clean_string(data.order_proc_id)
    fpk = _clean_string(data.echo_test_fpk)
    data["group_key"] = "TEST:" + fpk
    data.loc[order.notna(), "group_key"] = "ORD:" + order[order.notna()]
    data.loc[acc.notna(), "group_key"] = "ACC:" + acc[acc.notna()]
    data["number"] = data.result_value.map(parse_lvef)
    data = data[data.number.notna()].copy()
    rules = [(rule["method"], re.compile(rule["regex"], re.I))
             for rule in results_cfg["method_priority"]]
    def classify(label: str) -> str:
        return next((name for name, regex in rules if regex.search(str(label))), "other")
    data["method"] = data.result_label.map(classify)
    data["result_name"] = data.result_name.fillna("").astype(str).str.strip()
    threshold = float(results_cfg.get("lvef_disagreement_threshold_pp", 10.0))
    lookup = {key: group for key, group in data.groupby(
        ["person_id", "index_date", "group_key"], dropna=False
    )}
    output = []
    for candidate in targets.itertuples(index=False):
        keys = [candidate.group_key]
        if pd.notna(candidate.partner_key) and candidate.partner_key:
            keys.append(candidate.partner_key)
        same_conflict, different_conflict = False, False
        conflict_methods = set()
        for key in keys:
            group = lookup.get((candidate.person_id, candidate.index_date, key))
            if group is None:
                continue
            for method, family in group.groupby("method", dropna=False):
                exact = family.groupby("result_name")["number"].agg(["min", "max"])
                same = bool(((exact["max"] - exact["min"]) > threshold).any())
                # Compare raw values across distinct exact result names.
                names = exact.index.tolist()
                cross = any(
                    max(float(exact.loc[names[i], "max"]) - float(exact.loc[names[j], "min"]),
                        float(exact.loc[names[j], "max"]) - float(exact.loc[names[i], "min"])) > threshold
                    for i in range(len(names)) for j in range(i + 1, len(names))
                )
                same_conflict |= same
                different_conflict |= cross
                if same or cross:
                    conflict_methods.add(str(method))
        classification = (
            "both" if same_conflict and different_conflict else
            "same_name_only" if same_conflict else
            "different_names_only" if different_conflict else
            "neither"
        )
        output.append(dict(
            person_id=candidate.person_id, index_date=candidate.index_date,
            group_key=candidate.group_key, partner_key=candidate.partner_key,
            record_source=candidate.record_source,
            candidate_category=candidate.lvef_disagreement_category,
            exact_name_conflict=same_conflict,
            different_names_conflict=different_conflict,
            name_conflict_class=classification,
            method_families_with_conflict=len(conflict_methods),
        ))
    details = pd.DataFrame(output, columns=details_columns)
    summary = (details.groupby(["record_source", "name_conflict_class"])
               .size().rename("groups").reset_index())
    return details, summary[summary_columns]
