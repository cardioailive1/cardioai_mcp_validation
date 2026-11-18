"""Disclosure-controlled diagnosis of conservative echo extraction preview.

Consumes the PRIVATE result of run_echo_extraction_preview. Exposes only
aggregate categories, never patient keys, volume measurements, or note text.
Threshold = study['small_cell']; site/cumulative disclosure review is still
required before sharing outside MCP.

The counts classify the *preview implementation* -- they do not establish
availability of a clinically accepted reference or justify threshold changes.
"""
from __future__ import annotations

import pandas as pd


_METRICS = ("lv_end_diastolic_volume", "lv_end_systolic_volume")
_REASON_TO_GROUP = {
    "unique_consistent_ml": "selected_consistent_absolute_ml",
    "no_named_measurement": "no_recognized_source_result",
    "no_numeric_in_range": "no_recognized_source_result",
    "unknown_or_non_absolute_unit": "held_for_unit_or_method_review",
    "disagreeing_measurements_review": "held_for_unit_or_method_review",
}
_EXAM_TO_GROUP = {
    "single_identifier_group": "selected_single_identifier",
    "provisional_pair_requires_review": "provisional_acc_ord_pair_requires_review",
    "multiple_exams": "other_exam_or_identifier_review",
    "unresolved_identifier": "other_exam_or_identifier_review",
    "no_conventional_tte": "other_exam_or_identifier_review",
}
_VOLUME_REASON_ORDER = [
    "selected_consistent_absolute_ml",
    "no_recognized_source_result",
    "held_for_unit_or_method_review",
]
_EXAM_REASON_ORDER = [
    "selected_single_identifier",
    "provisional_acc_ord_pair_requires_review",
    "other_exam_or_identifier_review",
]


def _safe_categorical_counts(
    counts: pd.Series, categories: list[str], *, small_cell: int
) -> list[str]:
    """Complementary suppression: if any positive category is small, redact ALL.

    Suppressing only the small category would disclose its count by subtraction
    from the known cohort/selected-examination denominator.
    """
    if small_cell < 1:
        raise ValueError("small_cell must be positive")
    raw = [int(counts.get(c, 0)) for c in categories]
    if any(0 < count < small_cell for count in raw):
        return ["suppressed"] * len(categories)
    return [str(count) for count in raw]


def summarize_preview_barriers(
    private: pd.DataFrame, *, small_cell: int = 11
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(exam topology, selected-group volume barriers), both aggregates.

    The volume reason denominator is the count of *single identifier group*
    patient-days only. It excludes exams withheld for identifier ambiguity.
    All numeric cells must be disclosure reviewed in combination with any
    separately published cohort/report audit.
    """
    required = {"exam_status", *[m + "_status" for m in _METRICS]}
    if not required.issubset(private):
        raise ValueError(f"Missing preview fields: {sorted(required - set(private))}")
    if private.empty:
        raise ValueError("No cohort patients to summarize")
    if small_cell < 1:
        raise ValueError("small_cell must be positive")
    statuses = private["exam_status"].map(_EXAM_TO_GROUP)
    if statuses.isna().any():
        raise ValueError("Unknown examination preview status")

    ec = statuses.value_counts()
    exam = pd.DataFrame({
        "review_category": _EXAM_REASON_ORDER,
        "patient_days": _safe_categorical_counts(
            ec, _EXAM_REASON_ORDER, small_cell=small_cell),
    })
    selected = private.loc[statuses.eq("selected_single_identifier")]
    volume = []
    for metric in _METRICS:
        raw = selected[metric + "_status"]
        reason = raw.map(_REASON_TO_GROUP)
        if reason.isna().any():
            raise ValueError(f"Unknown volume preview reason for {metric}")
        rc = reason.value_counts()
        values = _safe_categorical_counts(
            rc, _VOLUME_REASON_ORDER, small_cell=small_cell)
        for key, n in zip(_VOLUME_REASON_ORDER, values):
            volume.append({
                "metric": metric, "review_category": key,
                "selected_group_patient_days": n,
            })
    return exam, pd.DataFrame(volume)


def within_mcp_reason_counts(private: pd.DataFrame) -> pd.DataFrame:
    """PRIVATE output: detailed reasons. Do not print/export this DataFrame.

    Useful to an authorized internal reviewer when aggregate suppression
    means we cannot break down a category for external sharing.
    """
    frames = []
    for metric in _METRICS:
        source = metric + "_status"
        frames.append(
            private.groupby(["exam_status", source], dropna=False)
            .size().rename("patient_days").reset_index()
            .rename(columns={source: "volume_status"})
            .assign(metric=metric)
        )
    return pd.concat(frames, ignore_index=True)


def run_preview_barrier_audit(eng, study: dict):
    """Read-only fresh-session diagnostic; returns private detail and safe summaries.

    Only print the 2nd and 3rd returned DataFrames. Detailed reasons and
    patient-level source information are available inside MCP via the first.
    """
    from .cohort_extraction_preview import run_echo_extraction_preview
    private, _, _ = run_echo_extraction_preview(eng, study)
    exam, volume = summarize_preview_barriers(
        private, small_cell=int(study.get("small_cell", 11)))
    return private, exam, volume
