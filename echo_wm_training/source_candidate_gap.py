"""Read-only root-cause diagnostics for cohort LVEDV source/candidate gap.

Investigates patient/index-days with a numeric mapped LVEDV in at least one
conventional index-day FACT_ECHO_TEST record, but no eligible provisional
candidate with LVEDV. Neither the saved cohort nor extraction policy changes.

Retains patient identifiers only in private in-memory output. The public
report aggregates mutually exclusive reason categories. These describe
mechanical status / provenance, NOT clinical eligibility decisions.
"""
from __future__ import annotations

import pandas as pd

from .cohort_type_availability import (
    KEYS, cohort_type_rows_sql, summarize_cohort_types,
)
from .unified_reference import build_unified_echo_diagnostics


def diagnose_source_not_candidate(
    cohort: pd.DataFrame,
    source_rows: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (private patient-day diagnostics, aggregate reasons, status counts).

    Root-cause reason priority:
      - Mayo-only result name, not included in current candidate SQL.
      - Recognized measurement but candidate explicitly conflicts on LVEDV.
      - Other candidate conflict on LVEF or multiple measurements.
      - Missing/ambiguous identifier in candidate grouping.
      - Multiple candidate exam groups.
      - No eligible candidate matching the source patient-day.
      - Recognized source value missing at candidate stage (investigate).

    Candidate group values may be non-null even when their status is excluded.
    This is expected and must not be treated as accepted coverage.

    Reason is assigned per patient-day, never per ACC/ORD record. Additional
    flags remain in private output for cases with multiple simultaneous
    causes. No clinical promotion is recommended here.
    """
    disposition, _, _ = summarize_cohort_types(
        cohort, source_rows, candidates, small_cell=1
    )
    target = disposition.loc[
        disposition.coverage_class.eq("source_lvedv_not_candidate"),
        KEYS
    ].copy()

    source_required = {*KEYS, "echo_test_fpk", "has_mapped_lvedv", "has_mayo_lvedv", "has_priority_lvedv"}
    missing = source_required - set(source_rows.columns)
    if missing:
        raise ValueError(f"Source rows missing columns: {sorted(missing)}")
    rows = source_rows.copy()
    rows["index_date"] = pd.to_datetime(rows["index_date"]).dt.date
    rows["has_mapped_lvedv"] = rows["has_mapped_lvedv"].fillna(False).astype(bool)
    rows["has_mayo_lvedv"] = rows["has_mayo_lvedv"].fillna(False).astype(bool)
    rows = rows.merge(target, on=KEYS, how="inner")
    rows = rows.drop_duplicates(KEYS + ["echo_test_fpk"])

    # Distinguish the additional Mayo name from existing LVEDV_PRIORITY
    # names, even if both occur in the same ECHO_TEST_FPK record.
    rows["has_priority_lvedv"] = rows["has_priority_lvedv"].fillna(False).astype(bool)
    source_days = (rows.groupby(KEYS, as_index=False)
                   .agg(has_priority_lvedv=("has_priority_lvedv", "any"),
                        has_mayo_lvedv=("has_mayo_lvedv", "any"),
                        source_tests=("echo_test_fpk", "nunique")))

    candidate_required = {*KEYS, "candidate_status", "lvedv"}
    missing_candidate = candidate_required - set(candidates.columns)
    if missing_candidate:
        raise ValueError(f"Candidate rows missing columns: {sorted(missing_candidate)}")
    c = candidates.copy()
    c["index_date"] = pd.to_datetime(c["index_date"]).dt.date
    c = c.merge(target, on=KEYS, how="inner")
    optional = ("lvedv_conflict", "lvef_conflict", "lvef_method_disagreement")
    for col in optional:
        if col not in c.columns:
            c[col] = False
        c[col] = c[col].fillna(False).astype(bool)

    flag_labels = [
        "conflicting_measurements", "unresolved_identifier",
        "ambiguous_multiple_exams", "unresolved_method_disagreement",
        "single_exam_candidate",
    ]
    for label in flag_labels:
        c["status_" + label] = c.candidate_status.eq(label)

    agg_fields = {
        "candidate_groups": ("candidate_status", "size"),
        "candidate_has_lvedv": ("lvedv", lambda x: bool(x.notna().any())),
        "lvedv_conflict": ("lvedv_conflict", "any"),
        "lvef_conflict": ("lvef_conflict", "any"),
        "ef_method_disagreement": ("lvef_method_disagreement", "any"),
    }
    agg_fields.update({
        "status_" + label: ("status_" + label, "any") for label in flag_labels
    })
    if not c.empty:
        candidate_days = c.groupby(KEYS, as_index=False).agg(**agg_fields)
    else:
        candidate_days = pd.DataFrame(columns=KEYS + list(agg_fields))

    private = (target.merge(source_days, on=KEYS, how="left", validate="one_to_one")
              .merge(candidate_days, on=KEYS, how="left", validate="one_to_one"))
    private["candidate_groups"] = private.candidate_groups.fillna(0).astype(int)
    private["candidate_has_lvedv"] = private.candidate_has_lvedv.fillna(False).astype(bool)
    # A numeric LVEDV can survive group extraction while its candidate_status
    # prevents provisional acceptance (e.g. multiple examinations or an
    # unresolved identifier). That is precisely an exclusion to investigate,
    # not a consistency failure. Preserve the flag in private details.
    for col in ("has_priority_lvedv", "has_mayo_lvedv", "lvedv_conflict",
                "lvef_conflict", "ef_method_disagreement",
                *(("status_" + label) for label in flag_labels)):
        private[col] = private[col].fillna(False).astype(bool)

    def reason(r):
        if r.has_mayo_lvedv and not r.has_priority_lvedv:
            return "mayo_result_name_only"
        if r.lvedv_conflict:
            return "lvedv_conflict"
        if r.lvef_conflict:
            return "lvef_conflict"
        if r["status_unresolved_identifier"]:
            return "unresolved_identifier"
        if r["status_ambiguous_multiple_exams"]:
            return "multiple_exam_candidates"
        if r["status_unresolved_method_disagreement"] or r.ef_method_disagreement:
            return "ef_method_disagreement"
        if r.candidate_groups == 0:
            return "no_candidate_group"
        return "priority_lvedv_not_carried_to_candidate"

    private["reason"] = private.apply(reason, axis=1)
    reasons = (private.reason.value_counts().rename_axis("reason")
               .reset_index(name="patient_days"))
    status_counts = (
        c.candidate_status.value_counts().rename_axis("candidate_status")
         .reset_index(name="candidate_groups")
        if not c.empty else
        pd.DataFrame(columns=["candidate_status", "candidate_groups"])
    )
    return private, reasons, status_counts


def run_source_candidate_gap_audit(engine, study: dict):
    """Perform read-only analysis with existing SQL and matcher.

    Returned private_detail includes identifiers and must remain inside MCP.
    Counts may require small-cell / complementary suppression to share.
    """
    start, end = str(study["index_start"]), str(study["index_end"])
    cohort_sql = (
        "SELECT person_id, DATE(index_date) AS index_date "
        "FROM {{work}}.echo_analytic "
        f"WHERE DATE(index_date) BETWEEN DATE('{start}') AND DATE('{end}')"
    )
    cohort = engine.query(engine.render(cohort_sql, {}))
    candidates, _, _ = build_unified_echo_diagnostics(
        engine, study["modality"], start=start, end=end,
        max_seconds=60, log=lambda _: None,
    )
    source_sql = engine.render(
        cohort_type_rows_sql(study["modality"], start=start, end=end), {}
    )
    source_rows = engine.client.query(
        source_sql, location=engine.location
    ).to_dataframe()
    private, reasons, statuses = diagnose_source_not_candidate(
        cohort, source_rows, candidates
    )
    return private, reasons, statuses
