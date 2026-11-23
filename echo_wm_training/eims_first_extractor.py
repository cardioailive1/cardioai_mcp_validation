"""Staged EIMS-first LVEDV/LVESV extractor for the saved Mayo echo cohort.

This is an implemented clinical *candidate* extractor, not an approved
clinical-reference or production cohort migration. All volume values are
sourced from EIMS ROW_SOURCE_ID=12315 on the same ECHO_TEST_FPK, with
explicit mL unit metadata and a unique preferred-tier value. Never merge
EIMS and Epic examinations on person/date or infer units for Epic MC CV.

The provisional (biplane, 3D, single-plane) method hierarchy is carried
forward strictly as a WHAT-IF until a clinical advisor approves it.
A second same-day source-system TTE is marked for exam-identity review;
multiple EIMS test FPKs are always withheld. No data are written.

WARNING: The private candidate frame includes patient IDs, test FPKs and
numeric clinical values. Never print, export, or share it outside MCP.
Only the disclosure-gated cohort completeness summary may be considered
for release after cumulative disclosure review.
"""
from __future__ import annotations

import pandas as pd

from cardioai_mcp.config import load_cohort

from .cohort_extraction_preview import (
    _KEYS, _RANGE, compare_echo_completeness,
    suppress_completeness_comparison,
)
from .eims_volume_eligibility import (
    _EIMS, _METRICS, _TEST_KEYS, _clean_source, eims_test_source_sql,
)
from .explicit_ml_feasibility import (
    _METHODS, _TIER_ORDER, _UNIT_OK, _method_feasibility,
)
from .source_system_bridge import source_system_bridge_rows_sql


def _candidate_volume(rows: pd.DataFrame, metric: str) -> tuple[float | None, str, str | None]:
    """Return candidate numeric mL and method tier, or a review status.

    This intentionally shares method screening with _method_feasibility.
    A conflicting value in the preferred available method tier is withheld,
    not replaced by a lower-priority method or arbitrarily averaged.
    """
    status = _method_feasibility(rows, metric)
    if status == "no_explicit_ml":
        return None, status, None
    if status == "conflicting_preferred_tier":
        return None, status, None

    tier = status.removesuffix("_candidate")
    if tier not in _TIER_ORDER:
        raise ValueError(f"Unexpected provisional method status: {status}")
    d = rows.loc[rows.result_name.isin(_METHODS[metric][tier])].copy()
    d["number"] = pd.to_numeric(d.raw_value, errors="coerce")
    low, high = _RANGE[metric]
    units = d.reported_unit.fillna("").astype(str).str.strip().str.lower()
    d = d.loc[d.number.between(low, high) & units.isin(_UNIT_OK)]
    values = d.number.drop_duplicates()
    if len(values) != 1:
        raise ValueError("Method status/value disagreement: fail closed")
    return float(values.iloc[0]), status, tier


def stage_eims_first_volumes(
    cohort_days: pd.DataFrame,
    tests: pd.DataFrame,
    source_results: pd.DataFrame,
) -> pd.DataFrame:
    """Return one PRIVATE staged candidate row per saved patient/index-day.

    No values are promoted. 'single_eims_other_source_review' explicitly
    flags another same-day source's TTE, even if this EIMS FPK is unique.
    """
    needed_cohort = set(_KEYS)
    needed_test = set(_TEST_KEYS + ["test_row_source_id"])
    needed_result = set(_TEST_KEYS + [
        "test_row_source_id", "result_row_source_id",
        "result_name", "raw_value", "reported_unit",
    ])
    if not needed_cohort.issubset(cohort_days):
        raise ValueError("Missing saved-cohort person/index-day keys")
    if not needed_test.issubset(tests):
        raise ValueError("Missing EIMS test FPK/source fields")
    if not needed_result.issubset(source_results):
        raise ValueError("Missing EIMS test/result provenance or volume fields")

    c = cohort_days[_KEYS].copy()
    c.index_date = pd.to_datetime(c.index_date).dt.date
    if c[_KEYS].isna().any().any() or c.duplicated(_KEYS).any():
        raise ValueError("Cohort keys must be nonnull and unique")
    t = tests[_TEST_KEYS + ["test_row_source_id"]].copy()
    t.index_date = pd.to_datetime(t.index_date).dt.date
    t.echo_test_fpk = t.echo_test_fpk.astype("string").str.strip()
    if t[_TEST_KEYS].isna().any().any() or t.echo_test_fpk.eq("").any():
        raise ValueError("Missing source test FPK or cohort keys")
    t.test_row_source_id = t.test_row_source_id.map(_clean_source)
    if t.drop_duplicates().duplicated(_TEST_KEYS).any():
        raise ValueError("Conflicting source provenance for one source test")
    t = t.drop_duplicates(_TEST_KEYS)
    t = t.merge(c, on=_KEYS, how="inner", validate="many_to_one")

    v = source_results[_TEST_KEYS + [
        "test_row_source_id", "result_row_source_id",
        "result_name", "raw_value", "reported_unit"
    ]].copy()
    v.index_date = pd.to_datetime(v.index_date).dt.date
    v.echo_test_fpk = v.echo_test_fpk.astype("string").str.strip()
    if v[_TEST_KEYS].isna().any().any() or v.echo_test_fpk.eq("").any():
        raise ValueError("Missing result FPK or cohort keys")
    v.test_row_source_id = v.test_row_source_id.map(_clean_source)
    v.result_row_source_id = v.result_row_source_id.map(_clean_source)
    v = v.merge(
        t, on=_TEST_KEYS, how="inner",
        suffixes=("_result", "_test"), validate="many_to_one",
    )
    # Source metadata must not silently contradict the keyed test.
    bad = v.loc[
        v.test_row_source_id_result.isna()
        | v.test_row_source_id_test.isna()
        | v.test_row_source_id_result.ne(v.test_row_source_id_test).fillna(True)
    ]
    if not bad.empty:
        raise ValueError("Inconsistent source test/result ROW_SOURCE_ID")
    v = v.loc[
        v.test_row_source_id_test.eq(_EIMS)
        & v.result_row_source_id.eq(_EIMS)
    ]
    t_by_day = {key: frame for key, frame in t.groupby(_KEYS, dropna=False)}
    v_by_fpk = {
        key: frame for key, frame in v.groupby(_TEST_KEYS, dropna=False)
    }
    empty = pd.DataFrame(columns=["result_name", "raw_value", "reported_unit"])

    records = []
    for day in c.itertuples(index=False):
        key = (day.person_id, day.index_date)
        group = t_by_day.get(key)
        selected_fpk = None
        if group is None or group.empty:
            exam_status = "no_conventional_tte"
        else:
            eims = group.loc[group.test_row_source_id.eq(_EIMS)]
            if eims.empty:
                exam_status = "no_eims_tte"
            elif eims.echo_test_fpk.nunique() != 1:
                exam_status = "multiple_eims_tests_review"
            else:
                selected_fpk = str(eims.echo_test_fpk.iloc[0])
                exam_status = (
                    "single_eims_only_candidate" if len(group) == 1
                    else "single_eims_other_source_review"
                )
        record = dict(
            person_id=day.person_id,
            index_date=day.index_date,
            eims_echo_test_fpk=selected_fpk,
            exam_status=exam_status,
            source_system="EIMS",
            extraction_state="staged_not_approved",
        )
        for metric in _METRICS:
            value, method_status, method = (
                _candidate_volume(
                    v_by_fpk.get((*key, selected_fpk), empty), metric
                ) if selected_fpk is not None
                else (None, "exam_not_selected", None)
            )
            record[metric] = value
            record[metric + "_status"] = method_status
            record[metric + "_method_tier"] = method
            record[metric + "_requires_exam_review"] = (
                exam_status == "single_eims_other_source_review"
            )
        records.append(record)
    output = pd.DataFrame(records)
    if len(output) != len(c) or output.duplicated(_KEYS).any():
        raise ValueError("Staged EIMS candidates must retain cohort cardinality")
    return output


def run_eims_first_extraction_preview(eng, study: dict):
    """MCP-only read-only run: (PRIVATE candidates, disclosure-gated report).

    Preserve the first result INSIDE Mayo Cloud PC. Never print, display,
    save, or export the private DataFrame; it includes patient identifiers,
    EIMS examination keys and measurement values.
    """
    start, end = str(study["index_start"]), str(study["index_end"])
    analytic = eng.read_table("echo_analytic")
    tests = eng.query(eims_test_source_sql(
        study["modality"], start=start, end=end,
    ))
    results = eng.query(source_system_bridge_rows_sql(
        study["modality"], start=start, end=end,
    ))
    private = stage_eims_first_volumes(analytic, tests, results)
    original, _not_written = compare_echo_completeness(
        load_cohort("echo"), analytic, private,
    )
    safe = suppress_completeness_comparison(
        original,
        cohort_size=len(analytic),
        small_cell=int(study.get("small_cell", 11)),
    )
    return private, safe
