"""Read-only examination topology for the 22 ambiguous LVEDV patient-days.

This counts candidate exam groups and ACC/ORD matching statuses rather than
assuming two source records are two clinical examinations. No automatic
merging, clinical adjudication, cohort changes, or raw identifier export.
"""
from __future__ import annotations

import pandas as pd

from .cohort_type_availability import KEYS
from .exam_resolution import resolve_tte_groups, tte_record_rows_sql
from .source_candidate_gap import run_source_candidate_gap_audit
from .unified_reference import build_unified_echo_diagnostics


def classify_ambiguous_exams(
    excluded: pd.DataFrame,
    candidates: pd.DataFrame,
    tte_rows: pd.DataFrame,
    *,
    max_seconds: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (PRIVATE per-patient-day, summary, status counts).

    'Provisional groups' collapse only mutually unique ACC/ORD pairs
    within max_seconds. Unmatched groups stay separate, even if a reviewer
    might later decide two groups belong to one clinical examination.
    """
    required = set(KEYS + ["reason"])
    if required - set(excluded):
        raise ValueError(f"Excluded cases missing {sorted(required - set(excluded))}")
    selected = excluded.loc[
        excluded["reason"].eq("multiple_exam_candidates"), KEYS
    ].drop_duplicates().copy()
    selected["index_date"] = pd.to_datetime(selected.index_date).dt.date
    if selected.empty:
        cols = KEYS + ["raw_source_records", "identifier_groups",
                       "provisional_groups", "candidate_groups",
                       "candidate_groups_with_lvedv", "provisional_pairs",
                       "unmatched_acc", "unmatched_ord", "missing_identifier",
                       "competing_matches", "review_category"]
        return (pd.DataFrame(columns=cols),
                pd.DataFrame(columns=["review_category", "patient_days"]),
                pd.DataFrame(columns=["group_status", "groups"]))

    c = candidates.copy()
    need = set(KEYS + ["candidate_status", "lvedv", "group_key"])
    if need - set(c):
        raise ValueError(f"Candidate rows missing {sorted(need - set(c))}")
    c["index_date"] = pd.to_datetime(c.index_date).dt.date
    c = c.merge(selected, on=KEYS, how="inner")
    t = tte_rows.copy()
    if set(KEYS + ["echo_test_fpk"]) - set(t):
        raise ValueError("TTE record rows missing person/date/test key")
    t["index_date"] = pd.to_datetime(t.index_date).dt.date
    t = t.merge(selected, on=KEYS, how="inner")
    t = t.drop_duplicates(KEYS + ["echo_test_fpk"])
    groups = resolve_tte_groups(t, max_seconds=max_seconds)

    data = []
    for item in selected.itertuples(index=False):
        pid, day = item.person_id, item.index_date
        cg = c.loc[(c.person_id == pid) & (c.index_date == day)]
        gg = groups.loc[(groups.person_id == pid) & (groups.index_date == day)]
        tt = t.loc[(t.person_id == pid) & (t.index_date == day)]
        vc = gg.status.value_counts()
        pairs = int(((gg.status == "provisional_mutual_match") &
                     (gg.key_type == "ACC")).sum())
        # Each matched pair consists of two identifier groups but is
        # represented by one provisional group; other groups stay separate.
        logical = len(gg) - pairs
        # A one-group result still does NOT establish clinical identity.
        if int(vc.get("ambiguous_competing_match", 0)):
            category = "competing_acc_ord_matches"
        elif int(vc.get("missing_group_identifier", 0)):
            category = "missing_group_identifier"
        elif logical > 1 and pairs:
            category = "multiple_groups_after_provisional_pairing"
        elif logical > 1:
            category = "multiple_unmatched_identifier_groups"
        elif logical == 1:
            category = "single_provisional_group_review"
        else:
            category = "no_resolved_groups_review"
        data.append({
            "person_id": pid, "index_date": day,
            "raw_source_records": tt.echo_test_fpk.nunique(),
            "identifier_groups": len(gg),
            "provisional_groups": logical,
            "candidate_groups": len(cg),
            "candidate_groups_with_lvedv": int(cg.lvedv.notna().sum()),
            "provisional_pairs": pairs,
            "unmatched_acc": int(vc.get("unmatched_acc", 0)),
            "unmatched_ord": int(vc.get("unmatched_ord", 0)),
            "missing_identifier": int(vc.get("missing_group_identifier", 0)),
            "competing_matches": int(vc.get("ambiguous_competing_match", 0)),
            "review_category": category,
        })
    private = pd.DataFrame(data)
    if private.duplicated(KEYS).any() or len(private) != len(selected):
        raise ValueError("Audit did not preserve unique patient-days")
    summary = (private.review_category.value_counts()
               .rename_axis("review_category")
               .reset_index(name="patient_days"))
    status_counts = (
        groups.status.value_counts().rename_axis("group_status")
        .reset_index(name="groups")
    )
    return private, summary, status_counts


def run_ambiguous_lvedv_audit(engine, study: dict):
    """Recreate excluded cases, candidate groups, and raw TTE test records.

    PRIVATE detail contains identifiers. Summary/status tables are aggregates
    but still require MCP disclosure review before external distribution.
    """
    start, end = str(study["index_start"]), str(study["index_end"])
    excluded, _, _ = run_source_candidate_gap_audit(engine, study)
    candidates, _, _ = build_unified_echo_diagnostics(
        engine, study["modality"], start=start, end=end,
        max_seconds=60, log=lambda _: None
    )
    sql = tte_record_rows_sql(study["modality"], start=start, end=end)
    raw_tests = engine.query(engine.render(sql, {}))
    return classify_ambiguous_exams(excluded, candidates, raw_tests)
