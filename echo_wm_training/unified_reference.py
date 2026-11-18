"""Read-only unified Mayo Echo candidate coverage vs the active LVEF baseline.

This is a diagnostic only, never a clinical reference-selection policy.
ACC/ORD records retain provenance, only mutual unique provisional pairs are
coalesced, and patients with multiple candidate examinations are flagged.
No tables are modified; patient-level DataFrames must stay inside MCP.
"""
from __future__ import annotations

import pandas as pd

from .exam_comparison import _measurement_groups, tte_measurement_rows_sql
from .exam_resolution import resolve_tte_groups, tte_record_rows_sql
from .mcp_echo import _cfg


def baseline_lvef_rows_sql(*, start: str, end: str) -> str:
    """Fetch only saved cohort patients' existing EF baseline (including day +1)."""
    from datetime import date
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last:
        raise ValueError("start must be on or before end")
    return f"""
    SELECT a.person_id, CAST(a.index_date AS DATE) AS index_date,
           CAST(l.lvef_date AS DATE) AS lvef_date, l.lvef AS baseline_lvef
    FROM {{{{work}}}}.echo_analytic a
    JOIN {{{{work}}}}.echo_lvef l ON l.person_id = a.person_id
    WHERE CAST(a.index_date AS DATE) BETWEEN CAST('{start}' AS DATE) AND CAST('{end}' AS DATE)
      AND DATEDIFF(CAST(l.lvef_date AS DATE), CAST(a.index_date AS DATE)) BETWEEN 0 AND 1
    """


def unified_exam_candidates(tte_rows: pd.DataFrame, result_rows: pd.DataFrame,
                            results_cfg: dict, *, max_seconds: int = 60) -> pd.DataFrame:
    """One row per provisional/residual exam group; never silently merge conflicts."""
    cols = ["person_id", "index_date", "group_key", "partner_key", "group_status",
            "record_source", "lvef", "lvedv", "lvef_source", "lvedv_source",
            "lvef_conflict", "lvedv_conflict", "lvef_method_disagreement",
            "candidate_exams", "candidate_status"]
    if tte_rows.empty:
        return pd.DataFrame(columns=cols)
    groups = resolve_tte_groups(tte_rows, max_seconds=max_seconds)
    values = _measurement_groups(result_rows, results_cfg)
    lookup = {(r.person_id, r.index_date, r.group_key): r
              for r in values.itertuples(index=False)}
    entries = []
    for group in groups.itertuples(index=False):
        # Each pair is represented once, by its accession side.
        if group.status == "provisional_mutual_match" and group.key_type == "ORD":
            continue
        partner = (group.partner_key if group.status == "provisional_mutual_match"
                   else None)
        own = lookup.get((group.person_id, group.index_date, group.group_key))
        other = lookup.get((group.person_id, group.index_date, partner)) if partner else None
        def choose(metric):
            x = getattr(own, metric) if own is not None else None
            y = getattr(other, metric) if other is not None else None
            has_x, has_y = pd.notna(x), pd.notna(y)
            if has_x and has_y:
                if float(x) != float(y):
                    return None, "CONFLICT", True
                return float(x), "BOTH", False
            source = group.key_type if has_x else ("ORD" if group.key_type == "ACC" else "ACC") if has_y else None
            return (float(x) if has_x else float(y) if has_y else None), source, False
        ef, ef_source, ef_conflict = choose("lvef")
        edv, edv_source, edv_conflict = choose("lvedv")
        ef_method_disagreement = any(
            bool(x.lvef_method_disagreement)
            for x in (own, other) if x is not None
        )
        # A discrepant method set cannot be silently promoted to a reference.
        if ef_method_disagreement:
            ef = None
        entries.append(dict(
            person_id=group.person_id, index_date=group.index_date,
            group_key=group.group_key, partner_key=partner,
            group_status=group.status,
            record_source=("ACC+ORD" if partner else group.key_type),
            lvef=ef, lvedv=edv, lvef_source=ef_source,
            lvedv_source=edv_source, lvef_conflict=ef_conflict,
            lvedv_conflict=edv_conflict,
            lvef_method_disagreement=ef_method_disagreement,
        ))
    result = pd.DataFrame(entries)
    result["candidate_exams"] = result.groupby(
        ["person_id", "index_date"])["group_key"].transform("size")
    # Any competing or absent identifier blocks automatic uniqueness.
    result["candidate_status"] = "ambiguous_multiple_exams"
    result.loc[result.candidate_exams.eq(1), "candidate_status"] = "single_exam_candidate"
    result.loc[result.group_status.isin(
        ["ambiguous_competing_match", "missing_group_identifier"]
    ), "candidate_status"] = "unresolved_identifier"
    result.loc[result.lvef_conflict | result.lvedv_conflict,
               "candidate_status"] = "conflicting_measurements"
    result.loc[result.lvef_method_disagreement,
               "candidate_status"] = "unresolved_method_disagreement"
    return result[cols]


def compare_to_baseline(candidates: pd.DataFrame, baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-index comparisons (MCP only) and aggregate diagnostic summary."""
    detail_cols = ["person_id", "index_date", "candidate_status", "candidate_exams",
                   "lvef", "lvedv", "baseline_lvef", "lvef_abs_difference"]
    summary_cols = ["scope", "patients", "lvef_available", "lvedv_available",
                    "baseline_available", "both_lvef", "equal_lvef", "changed_lvef",
                    "median_lvef_abs_difference", "p90_lvef_abs_difference"]
    if candidates.empty:
        return pd.DataFrame(columns=detail_cols), pd.DataFrame(columns=summary_cols)
    c = candidates.copy()
    c["index_date"] = pd.to_datetime(c.index_date).dt.date
    base = baseline.copy()
    if not base.empty:
        base["index_date"] = pd.to_datetime(base.index_date).dt.date
        base["lvef_date"] = pd.to_datetime(base.lvef_date).dt.date
        base["offset"] = (pd.to_datetime(base.lvef_date) -
                          pd.to_datetime(base.index_date)).dt.days
        base = (base.sort_values(["person_id", "index_date", "offset", "lvef_date"])
                .drop_duplicates(["person_id", "index_date"]))
        base = base[["person_id", "index_date", "baseline_lvef"]]
    else:
        base = pd.DataFrame(columns=["person_id", "index_date", "baseline_lvef"])
    c = c.merge(base, on=["person_id", "index_date"], how="left", validate="many_to_one")
    c["lvef_abs_difference"] = (pd.to_numeric(c.lvef, errors="coerce") -
                                pd.to_numeric(c.baseline_lvef, errors="coerce")).abs()
    # One row per patient/index-date for metrics, including ambiguous cases.
    c["selected"] = c.candidate_exams.eq(1) & c.candidate_status.eq("single_exam_candidate")
    detail = c[c.selected].copy()
    all_indexes = c[["person_id", "index_date", "baseline_lvef"]].drop_duplicates(
        ["person_id", "index_date"])
    scopes = [("all_index_dates", all_indexes), ("single_exam_candidates", detail)]
    aggregates = []
    for name, frame in scopes:
        if name == "all_index_dates":
            aggregates.append(dict(scope=name, patients=len(frame),
                                   lvef_available=None, lvedv_available=None,
                                   baseline_available=int(frame.baseline_lvef.notna().sum()),
                                   both_lvef=None, equal_lvef=None, changed_lvef=None,
                                   median_lvef_abs_difference=None, p90_lvef_abs_difference=None))
            continue
        diffs = frame.lvef_abs_difference.dropna()
        aggregates.append(dict(scope=name, patients=len(frame),
                               lvef_available=int(frame.lvef.notna().sum()),
                               lvedv_available=int(frame.lvedv.notna().sum()),
                               baseline_available=int(frame.baseline_lvef.notna().sum()),
                               both_lvef=len(diffs), equal_lvef=int((diffs == 0).sum()),
                               changed_lvef=int((diffs > 0).sum()),
                               median_lvef_abs_difference=float(diffs.median()) if len(diffs) else None,
                               p90_lvef_abs_difference=float(diffs.quantile(.90)) if len(diffs) else None))
    return detail[detail_cols], pd.DataFrame(aggregates, columns=summary_cols)


def build_unified_echo_diagnostics(engine, modality_cfg: dict, *, start: str, end: str,
                                   max_seconds: int = 60, log=print):
    """Returns (all candidate groups, eligible comparison rows, aggregate metrics)."""
    cfg = _cfg(modality_cfg)
    tests = engine.query(engine.render(tte_record_rows_sql(
        modality_cfg, start=start, end=end), {}))
    results = engine.query(engine.render(tte_measurement_rows_sql(
        modality_cfg, start=start, end=end), {}))
    base = engine.query(engine.render(baseline_lvef_rows_sql(start=start, end=end), {}))
    candidates = unified_exam_candidates(tests, results, cfg["results"],
                                          max_seconds=max_seconds)
    detail, summary = compare_to_baseline(candidates, base)
    if not candidates.empty:
        log(f"Exam candidate statuses: {candidates.candidate_status.value_counts().to_dict()}")
        log(f"Candidate measurement sources: {candidates.record_source.value_counts().to_dict()}")
        log(f"Candidate LVEF: {int(candidates.lvef.notna().sum())}; "
            f"LVEDV: {int(candidates.lvedv.notna().sum())}")
    log("Read-only comparison to saved echo_lvef; provisional matching is NOT clinical validation")
    if not summary.empty:
        log(summary.to_string(index=False))
    return candidates, detail, summary
