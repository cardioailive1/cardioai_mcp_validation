"""Read-only feasibility of combining mutually unique ACC/ORD *candidate* pairs.

A 60-second reciprocal match is NOT an approved clinical exam-identity join.
This module does not promote the pair, produce clinical reference values,
or modify the saved echo_analytic cohort. It asks how many of the 718
withheld provisional-pair patient-days would have explicit-mL values under
a hypothetical paired-exam interpretation.

Source values and test/patient identifiers remain inside MCP. Only
suppressed aggregate counts are returned. Verify cumulative disclosure
with MCP policy before externally sharing aggregates.
"""
from __future__ import annotations

import pandas as pd

from .cohort_extraction_preview import (
    _KEYS, _group_keys, echo_volume_rows_sql, preview_echo_extraction,
)
from .exam_resolution import resolve_tte_groups, tte_record_rows_sql
from .explicit_ml_feasibility import _method_feasibility

_METRICS = ("lv_end_diastolic_volume", "lv_end_systolic_volume")
_TOPOLOGY = ("sole_mutual_pair", "pair_with_additional_identifier_groups")
_MEASURE_STATUSES = (
    "explicit_ml_preferred_tier_candidate",
    "conflicting_preferred_tier",
    "no_explicit_ml",
)


def _classify_pair_only(groups: pd.DataFrame) -> str:
    """Only two reciprocal ACC/ORD groups qualify for hypothetical union."""
    paired = groups.loc[groups.status.eq("provisional_mutual_match")]
    if paired.empty:
        raise ValueError("Preview marked pair but resolver found no pair")
    if len(groups) != 2 or len(paired) != 2:
        return "pair_with_additional_identifier_groups"
    acc = paired.loc[paired.key_type.eq("ACC")]
    order = paired.loc[paired.key_type.eq("ORD")]
    if len(acc) != 1 or len(order) != 1:
        return "pair_with_additional_identifier_groups"
    a, o = acc.iloc[0], order.iloc[0]
    if a.partner_key != o.group_key or o.partner_key != a.group_key:
        raise ValueError("Provisional pairing is not reciprocal")
    return "sole_mutual_pair"


def _counts_with_primary_complement_suppression(
    data: pd.Series, categories: tuple[str, ...], *, small_cell: int
) -> list[str]:
    counts = data.value_counts()
    values = [int(counts.get(c, 0)) for c in categories]
    if any(0 < v < small_cell for v in values):
        return ["suppressed"] * len(categories)
    return [str(v) for v in values]


def summarize_provisional_pair_feasibility(
    private: pd.DataFrame,
    tests: pd.DataFrame,
    volumes: pd.DataFrame,
    *,
    small_cell: int = 11,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pair topology, explicit-mL method feasibility), aggregates only.

    The metric denominator is *sole, reciprocal, mutually matched ACC/ORD
    pairs*, not all 718 paired patient-days and not all 1,500 cohort days.
    Even when candidate source values exist, no examination is approved.
    """
    if small_cell < 1:
        raise ValueError("small_cell must be positive")
    if not set(_KEYS + ["exam_status"]).issubset(private):
        raise ValueError("Preview lacks cohort identity or exam_status")
    if private.duplicated(_KEYS).any():
        raise ValueError("Preview has duplicate patient/index-days")
    test_keys = _KEYS + ["echo_test_fpk"]
    required_t = set(test_keys + [
        "exam_datetime", "accession_number", "order_proc_id"
    ])
    if not required_t.issubset(tests):
        raise ValueError("TTE records missing examination-resolution columns")
    if not set(test_keys + [
        "result_name", "raw_value", "reported_unit"
    ]).issubset(volumes):
        raise ValueError("Volume records missing unit or result fields")

    selected = private.loc[
        private.exam_status.eq("provisional_pair_requires_review"), _KEYS
    ].copy()
    selected.index_date = pd.to_datetime(selected.index_date).dt.date
    t = tests.copy()
    if not t.empty:
        t.index_date = pd.to_datetime(t.index_date).dt.date
        t.echo_test_fpk = t.echo_test_fpk.astype("string").str.strip()
        t = t.merge(selected, on=_KEYS, how="inner", validate="many_to_one")
        t = t.drop_duplicates(test_keys)
    resolver = resolve_tte_groups(t)
    mapped = _group_keys(t)
    source = volumes.copy()
    if not source.empty:
        source.index_date = pd.to_datetime(source.index_date).dt.date
        source.echo_test_fpk = source.echo_test_fpk.astype("string").str.strip()
        source = source.merge(
            mapped[test_keys + ["group_key"]],
            on=test_keys, how="inner", validate="many_to_one",
        )
    by_group = {key: d for key, d in resolver.groupby(_KEYS, dropna=False)}
    by_source = {key: d for key, d in source.groupby(_KEYS, dropna=False)}
    empty_src = pd.DataFrame(columns=["result_name", "raw_value", "reported_unit"])

    topologies = []
    status_by_metric = {m: [] for m in _METRICS}
    for row in selected.itertuples(index=False):
        key = (row.person_id, row.index_date)
        group = by_group.get(key, pd.DataFrame(columns=resolver.columns))
        topology = _classify_pair_only(group)
        topologies.append(topology)
        if topology != "sole_mutual_pair":
            continue
        # Only a sole two-group pair reaches this point; source results
        # cannot be drawn from another index-day exam.
        source_day = by_source.get(key, empty_src)
        for metric in _METRICS:
            outcome = _method_feasibility(source_day, metric)
            status_by_metric[metric].append(
                "no_explicit_ml" if outcome == "no_explicit_ml" else
                "conflicting_preferred_tier" if outcome == "conflicting_preferred_tier" else
                "explicit_ml_preferred_tier_candidate"
            )
    if len(topologies) != len(selected):
        raise ValueError("Pair cohort lost index-day identity")
    topo = pd.DataFrame({
        "pair_topology": _TOPOLOGY,
        "patient_days": _counts_with_primary_complement_suppression(
            pd.Series(topologies, dtype="string"), _TOPOLOGY,
            small_cell=small_cell,
        ),
    })
    metric_rows = []
    count_only = sum(x == "sole_mutual_pair" for x in topologies)
    for metric, outcomes in status_by_metric.items():
        if len(outcomes) != count_only:
            raise ValueError("Volume count did not reconcile to sole-pair count")
        vals = _counts_with_primary_complement_suppression(
            pd.Series(outcomes, dtype="string"), _MEASURE_STATUSES,
            small_cell=small_cell,
        )
        for category, value in zip(_MEASURE_STATUSES, vals):
            metric_rows.append({
                "metric": metric, "candidate_category": category,
                "patient_days": value,
            })
    return topo, pd.DataFrame(metric_rows)


def run_provisional_pair_feasibility(eng, study: dict):
    """Read-only MCP runner; returns only two disclosure-guarded summaries."""
    start, end = str(study["index_start"]), str(study["index_end"])
    t = eng.query(tte_record_rows_sql(study["modality"], start=start, end=end))
    v = eng.query(echo_volume_rows_sql(study["modality"], start=start, end=end))
    analytic = eng.read_table("echo_analytic")
    no_notes = pd.DataFrame(columns=_KEYS + ["echo_test_fpk", "echo_note"])
    private = preview_echo_extraction(analytic, t, v, no_notes)
    return summarize_provisional_pair_feasibility(
        private, t, v, small_cell=int(study.get("small_cell", 11)),
    )
