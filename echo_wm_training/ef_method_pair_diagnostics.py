"""Read-only pairwise EF variant disagreement analysis within eligible TTE groups.

A candidate may have multiple discordant method pairs. Pair counts must NOT
be summed to infer the number of independent examination candidates.
Detailed patient/group rows must remain in the restricted MCP environment.
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd

from .exam_comparison import _clean_string
from .mcp_echo import classify_lvef_method, parse_lvef


def diagnose_ef_method_pairs(
    measurement_rows: pd.DataFrame,
    candidate_exams: pd.DataFrame,
    results_cfg: dict,
    *,
    include_categories: tuple[str, ...] = ("between_method",),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return private pair details and two aggregate summaries, with no writes.

    Pair differences compare the *medians* for each diagnostic method variant
    within an original ACC or ORD identifier group. A provisional ACC+ORD pair
    is NOT itself used as evidence of clinical exam identity, and measurements
    from opposite sides are never cross-compared.

    All variant pairs are retained, including those <= review threshold, so
    their contribution can be interpreted accurately. The flagged subset uses
    the configured strict > threshold. Candidate counting is distinct per pair;
    one candidate can contribute to several method pairs.
    """
    detail_cols = [
        "person_id", "index_date", "candidate_key", "source_group_key",
        "record_source", "method_a", "method_b", "ef_a", "ef_b",
        "difference_pp", "over_threshold",
    ]
    summary_cols = [
        "method_a", "method_b", "candidate_groups", "source_groups",
        "median_abs_difference_pp", "p90_abs_difference_pp",
        "max_abs_difference_pp",
    ]
    coverage_cols = ["record_source", "candidates", "source_groups", "flagged_pairs"]
    targets = candidate_exams.loc[
        candidate_exams["lvef_disagreement_category"].isin(include_categories)
    ].copy()
    if targets.empty:
        return (pd.DataFrame(columns=detail_cols),
                pd.DataFrame(columns=summary_cols),
                pd.DataFrame(columns=coverage_cols))
    required = {
        "person_id", "index_date", "echo_test_fpk",
        "accession_number", "order_proc_id", "metric",
        "result_label", "result_value",
    }
    missing = required - set(measurement_rows.columns)
    if missing:
        raise ValueError(f"Missing measurement fields: {sorted(missing)}")
    targets["index_date"] = pd.to_datetime(targets.index_date).dt.date
    ef = measurement_rows.loc[measurement_rows.metric == "lvef"].copy()
    ef["index_date"] = pd.to_datetime(ef.index_date).dt.date
    acc = _clean_string(ef.accession_number)
    order = _clean_string(ef.order_proc_id)
    fpk = _clean_string(ef.echo_test_fpk)
    ef["group_key"] = "TEST:" + fpk
    ef.loc[order.notna(), "group_key"] = "ORD:" + order[order.notna()]
    ef.loc[acc.notna(), "group_key"] = "ACC:" + acc[acc.notna()]
    ef["number"] = ef.result_value.map(parse_lvef)
    ef = ef[ef.number.notna()].copy()
    ef["variant"] = ef.result_label.map(
        lambda label: classify_lvef_method(label, results_cfg)[2]
    )
    # One median per variant per original identifier group, and never
    # conflate measurements from separate same-day ECHO_TEST_FPK records.
    medians = {
        k: g.groupby("variant")["number"].median().sort_index()
        for k, g in ef.groupby(["person_id", "index_date", "group_key"])
    }
    threshold = float(results_cfg.get("lvef_disagreement_threshold_pp", 10.0))
    pairs = []
    for candidate in targets.itertuples(index=False):
        keys = [candidate.group_key]
        if pd.notna(candidate.partner_key) and bool(candidate.partner_key):
            keys.append(candidate.partner_key)
        for key in dict.fromkeys(keys):
            values = medians.get((candidate.person_id, candidate.index_date, key))
            if values is None:
                continue
            for a, b in combinations(values.index.tolist(), 2):
                va, vb = float(values[a]), float(values[b])
                diff = abs(va - vb)
                pairs.append(dict(
                    person_id=candidate.person_id, index_date=candidate.index_date,
                    candidate_key=candidate.group_key, source_group_key=key,
                    record_source=candidate.record_source,
                    method_a=a, method_b=b, ef_a=va, ef_b=vb,
                    difference_pp=diff, over_threshold=diff > threshold,
                ))
    details = pd.DataFrame(pairs, columns=detail_cols)
    flagged = details.loc[details.over_threshold].copy()
    if flagged.empty:
        summary = pd.DataFrame(columns=summary_cols)
    else:
        # Distinct candidates within each method pair, even when an ACC/ORD
        # candidate contributes multiple source groups of the same method pair.
        def pair_statistics(g: pd.DataFrame) -> pd.Series:
            return pd.Series({
                "candidate_groups": len(g[["person_id", "index_date", "candidate_key"]].drop_duplicates()),
                "source_groups": len(g[["person_id", "index_date", "source_group_key"]].drop_duplicates()),
                "median_abs_difference_pp": float(g.difference_pp.median()),
                "p90_abs_difference_pp": float(g.difference_pp.quantile(0.9)),
                "max_abs_difference_pp": float(g.difference_pp.max()),
            })
        summary_rows = []
        for (a, b), group in flagged.groupby(["method_a", "method_b"]):
            summary_rows.append({"method_a": a, "method_b": b,
                                 **pair_statistics(group).to_dict()})
        summary = (pd.DataFrame(summary_rows, columns=summary_cols)
                   .sort_values(["candidate_groups", "median_abs_difference_pp"],
                                ascending=[False, False]))
    coverage = []
    for source, sub in targets.groupby("record_source"):
        current = flagged[flagged.record_source == source]
        coverage.append({
            "record_source": source,
            "candidates": len(sub),
            "source_groups": len(current[["person_id", "index_date", "source_group_key"]].drop_duplicates()),
            "flagged_pairs": len(current),
        })
    return details, summary, pd.DataFrame(coverage, columns=coverage_cols)
