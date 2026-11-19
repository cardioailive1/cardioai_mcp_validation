"""Read-only source encounter/order/accession signals for MC CV/mL test records.

Mayo confirmed ENCOUNTER_NUMBER on FACT_ECHO_TEST and FACT_ECHO_RESULTS,
alongside ORDER_PROC_ID; FACT_ECHO_TEST also has ACCESSION_NUMBER.

The output is *exam-linkage feasibility*, not proof of identical clinical
exams or inferred units. Joining only on patient/date or encounter does not
establish examination identity. No values or identifiers leave MCP.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cohort_extraction_preview import _KEYS, _RANGE, echo_volume_rows_sql
from .exam_resolution import tte_record_rows_sql
from .mcp_echo import _cfg, _t
from .unitless_scale_validation import (
    _COMPARATORS, _EXAM_KEYS, _ML, _PERIODS, _UNITLESS_NAMES, _period
)

_CATEGORIES = (
    "no_explicit_ml_other_test",
    "multiple_possible_test_pairs",
    "unique_shared_order_or_accession",
    "unique_shared_encounter_only",
    "unique_without_shared_identifiers",
)


def encounter_test_rows_sql(modality_cfg: dict, *, start: str, end: str) -> str:
    """Existing cohort TTE tests, adding encounter only from FACT_ECHO_TEST.

    Join on the same ECHO_TEST_FPK already used by tte_record_rows_sql.
    Keep patient-level rows inside the MCP Cloud PC; do not export them.
    """
    cfg = _cfg(modality_cfg)
    e = cfg["echo_test_columns"]
    return f"""
WITH tte AS (
    {tte_record_rows_sql(modality_cfg, start=start, end=end)}
)
SELECT DISTINCT tte.person_id, tte.index_date, tte.echo_test_fpk,
       tte.exam_datetime, tte.accession_number, tte.order_proc_id,
       CAST(test.ENCOUNTER_NUMBER AS STRING) AS encounter_number
FROM tte
JOIN {_t(cfg, "echo_test_table")} test
  ON CAST(test.{e['test_key']} AS STRING) = tte.echo_test_fpk
"""


def _nonblank(value) -> str | None:
    if pd.isna(value):
        return None
    normalized = str(value).strip()
    return normalized or None


def _classify_pair(
    mc_keys: set[str],
    ml_keys: set[str],
    test_lookup: dict[str, dict[str, str | None]],
) -> str:
    possible = [(a, b) for a in mc_keys for b in ml_keys if a != b]
    if not possible:
        return "no_explicit_ml_other_test"
    if len(possible) != 1:
        return "multiple_possible_test_pairs"
    a, b = possible[0]
    ma, mb = test_lookup[a], test_lookup[b]
    shared_order = (
        ma["order_proc_id"] is not None
        and ma["order_proc_id"] == mb["order_proc_id"]
    )
    shared_accession = (
        ma["accession_number"] is not None
        and ma["accession_number"] == mb["accession_number"]
    )
    if shared_order or shared_accession:
        return "unique_shared_order_or_accession"
    shared_encounter = (
        ma["encounter_number"] is not None
        and ma["encounter_number"] == mb["encounter_number"]
    )
    if shared_encounter:
        return "unique_shared_encounter_only"
    return "unique_without_shared_identifiers"


def summarize_encounter_volume_linkage(
    tests: pd.DataFrame,
    volumes: pd.DataFrame,
    *,
    small_cell: int = 11,
) -> pd.DataFrame:
    """Mutually exclusive classifications per unitless metric/patient-day.

    All counts for a metric/period suppressed if any positive category cell
    is smaller than the threshold, so categories cannot be subtracted from
    their known cohort denominator. Further cumulative disclosure review
    is still required before publishing beside prior analyses.
    """
    if small_cell < 1:
        raise ValueError("small_cell must be positive")
    needed = set(_EXAM_KEYS + [
        "encounter_number", "order_proc_id", "accession_number"
    ])
    if not needed.issubset(tests):
        raise ValueError("Test rows missing Mayo source linkage identifiers")
    if not set(_EXAM_KEYS + [
        "result_name", "raw_value", "reported_unit"
    ]).issubset(volumes):
        raise ValueError("Volume rows missing source test/result fields")

    t = tests.copy()
    t.index_date = pd.to_datetime(t.index_date).dt.date
    t.echo_test_fpk = t.echo_test_fpk.astype("string").str.strip()
    if t.echo_test_fpk.isna().any() or t.echo_test_fpk.eq("").any():
        raise ValueError("Missing ECHO_TEST_FPK")
    for column in ("encounter_number", "order_proc_id", "accession_number"):
        t[column] = t[column].map(_nonblank)
    # Fail closed on conflicting source identity attributes for same test.
    cols = _EXAM_KEYS + [
        "encounter_number", "order_proc_id", "accession_number"
    ]
    if t[cols].drop_duplicates().duplicated(_EXAM_KEYS).any():
        raise ValueError("Conflicting test identifiers for one ECHO_TEST_FPK")
    t = t.drop_duplicates(_EXAM_KEYS)
    v = volumes.copy()
    v.index_date = pd.to_datetime(v.index_date).dt.date
    v.echo_test_fpk = v.echo_test_fpk.astype("string").str.strip()
    v = v.merge(t[_EXAM_KEYS], on=_EXAM_KEYS, how="inner", validate="many_to_one")
    named = {
        name: (metric, "mc_cv")
        for metric, name in _UNITLESS_NAMES.items()
    }
    for metric, methods in _COMPARATORS.items():
        for name in methods.values():
            named[name] = (metric, "explicit_ml")
    v = v.loc[v.result_name.isin(named)].copy()
    v["metric"] = v.result_name.map(lambda name: named[name][0])
    v["family"] = v.result_name.map(lambda name: named[name][1])
    v["number"] = pd.to_numeric(v.raw_value, errors="coerce")
    v = v.loc[np.isfinite(v.number) & v.number.gt(0)].copy()
    v["unit"] = v.reported_unit.fillna("").astype(str).str.strip().str.lower()

    valid = []
    for key, subset in v.groupby(
        _EXAM_KEYS + ["metric", "result_name", "family"],
        dropna=False
    ):
        pid, day, fpk, metric, result_name, family = key
        if subset.number.nunique() != 1:
            continue
        if family == "mc_cv":
            accepted = subset.unit.eq("").all()
        else:
            lo, hi = _RANGE[metric]
            accepted = (
                subset.unit.isin(_ML).all()
                and subset.number.between(lo, hi).all()
            )
        if accepted:
            valid.append(dict(person_id=pid, index_date=day,
                              echo_test_fpk=fpk, metric=metric, family=family))
    d = pd.DataFrame(valid, columns=_EXAM_KEYS + ["metric", "family"])
    lookup = {
        tuple(getattr(record, k) for k in _EXAM_KEYS): {
            name: getattr(record, name)
            for name in ("encounter_number", "order_proc_id", "accession_number")
        }
        for record in t.itertuples(index=False)
    }
    cases = []
    for key, frame in d.groupby(_KEYS + ["metric"], dropna=False):
        pid, day, metric = key
        mc = set(frame.loc[frame.family.eq("mc_cv"), "echo_test_fpk"])
        if not mc:
            continue
        ml = set(frame.loc[frame.family.eq("explicit_ml"), "echo_test_fpk"])
        selected_lookup = {
            fpk: lookup[(pid, day, fpk)] for fpk in mc | ml
        }
        cases.append({
            "metric": metric,
            "period": _period(day),
            "category": _classify_pair(mc, ml, selected_lookup),
        })
    tagged = pd.DataFrame(cases, columns=["metric", "period", "category"])
    rows = []
    for metric in _UNITLESS_NAMES:
        for period in _PERIODS:
            subset = tagged.loc[
                tagged.metric.eq(metric) & tagged.period.eq(period)
            ]
            count = subset.category.value_counts()
            values = [int(count.get(c, 0)) for c in _CATEGORIES]
            if sum(values) != len(subset):
                raise ValueError("Linkage classifications failed reconciliation")
            unsafe = any(0 < n < small_cell for n in values)
            for category, n in zip(_CATEGORIES, values):
                rows.append({
                    "metric": metric, "period": period,
                    "linkage_category": category,
                    "patient_days": "suppressed" if unsafe else str(n),
                })
    return pd.DataFrame(rows)


def run_encounter_volume_linkage(eng, study: dict) -> pd.DataFrame:
    """Read only against cohort and FACT_ECHO_TEST/RESULTS inside MCP."""
    start, end = str(study["index_start"]), str(study["index_end"])
    tests = eng.query(encounter_test_rows_sql(
        study["modality"], start=start, end=end
    ))
    volumes = eng.query(echo_volume_rows_sql(
        study["modality"], start=start, end=end
    ))
    return summarize_encounter_volume_linkage(
        tests, volumes, small_cell=int(study.get("small_cell", 11))
    )
