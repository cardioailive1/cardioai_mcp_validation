"""Read-only next-day LVEDV inventory, separate from index-day candidate selection.

Day +1 is a *coverage diagnostic*, not evidence that the next-day TTE is
the index examination. Do not write, promote, or auto-assign these values.
Keep individual measurements and identifiers in the restricted MCP workspace.
"""
from __future__ import annotations

import pandas as pd

from .exam_resolution import tte_record_rows_sql
from .exam_results import LVEDV_PRIORITY
from .mcp_echo import _cfg, _macro

ADDITIONAL_LVEDV_NAME = "MC CV LV END DIASTOLIC VOLUME"


def next_day_lvedv_rows_sql(modality_cfg: dict, *, start: str, end: str) -> str:
    """Retrieve all conventional TTE test records on index date + 1.

    Include both the four previously mapped volume results and the candidate
    unrecognized accession-side result. Preserve result name and ECHO_TEST_FPK.
    Every TTE test row is retained even if no volume result exists.
    """
    tte = tte_record_rows_sql(modality_cfg, start=start, end=end, day_offset=1)
    c = _cfg(modality_cfg)
    result = c["results"]
    rc, ty = result["columns"], result["result_type"]
    names = (*LVEDV_PRIORITY, ADDITIONAL_LVEDV_NAME)
    quoted = ", ".join("'" + name.replace("'", "''") + "'" for name in names)
    return f"""
    WITH tte AS ({tte}),
    volume_rows AS (
        SELECT CAST(r.{rc['test_key']} AS STRING) AS echo_test_fpk,
               CAST(ty.{ty['name']} AS STRING) AS result_name,
               CAST(r.{rc['value']} AS STRING) AS value_raw
        FROM {_macro(result['table'])} r
        JOIN {_macro(ty['table'])} ty
          ON ty.{ty['key']} = r.{rc['result_type_key']}
        WHERE ty.{ty['name']} IN ({quoted})
    )
    SELECT tte.*, v.result_name, v.value_raw
    FROM tte
    LEFT JOIN volume_rows v
      ON v.echo_test_fpk = tte.echo_test_fpk
    """


def summarize_next_day_lvedv(
    next_day_rows: pd.DataFrame,
    cohort: pd.DataFrame,
    candidates: pd.DataFrame,
    recovered: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return (private per-index evidence, aggregate metrics) without writes.

    Conservative candidate: exactly one distinct next-day ECHO_TEST_FPK and
    a numeric LVEDV within 20..500 mL, with no conflict in *identical* result
    names. Multiple method variants remain for clinical review and are NOT
    automatically reconciled. This does not establish exam-date eligibility.
    """
    keys = ["person_id", "index_date"]
    cohort_keys = cohort[keys].copy()
    cohort_keys["index_date"] = pd.to_datetime(cohort_keys["index_date"]).dt.date
    if cohort_keys.duplicated(keys).any():
        raise ValueError("Duplicate cohort person/index-date keys")
    c = candidates.copy()
    c["index_date"] = pd.to_datetime(c["index_date"]).dt.date
    keep_status = c.candidate_status.isin(
        ["single_exam_candidate", "unresolved_method_disagreement"]
    )
    baseline = c.loc[keep_status & c.lvedv.notna(), keys].drop_duplicates()
    if recovered is not None and not recovered.empty:
        r = recovered.copy()
        r["index_date"] = pd.to_datetime(r["index_date"]).dt.date
        r = r.merge(c.loc[c.candidate_status == "single_exam_candidate", keys]
                    .drop_duplicates(), on=keys, how="inner")
        baseline = pd.concat([baseline, r[keys]], ignore_index=True).drop_duplicates()
    baseline = baseline.merge(cohort_keys, on=keys, how="inner")

    columns = keys + ["echo_test_fpk", "result_name", "value_raw"]
    if next_day_rows.empty:
        df = pd.DataFrame(columns=columns)
    else:
        missing = set(columns) - set(next_day_rows.columns)
        if missing:
            raise ValueError(f"Missing next-day fields: {sorted(missing)}")
        df = next_day_rows[columns].copy()
    df["index_date"] = pd.to_datetime(df["index_date"]).dt.date
    df = df.merge(cohort_keys, on=keys, how="inner")
    df["volume"] = pd.to_numeric(df.value_raw, errors="coerce")
    df["valid_volume"] = df.volume.between(20, 500)
    info = []
    for (pid, date), group in df.groupby(keys):
        values = group[group.valid_volume]
        count_tests = group.echo_test_fpk.nunique()
        # Multiple values under the *same* result name in one test are
        # suspicious and must not be assigned automatically.
        repeated_name_conflict = any(
            sub.volume.nunique() > 1
            for _, sub in values.groupby(["echo_test_fpk", "result_name"])
        )
        info.append({
            "person_id": pid, "index_date": date,
            "next_day_test_records": count_tests,
            "next_day_valid_volume": not values.empty,
            "next_day_single_test": count_tests == 1,
            "next_day_repeated_name_conflict": repeated_name_conflict,
            "next_day_distinct_methods": values.result_name.nunique(),
        })
    evidence = cohort_keys.merge(pd.DataFrame(info, columns=keys + [
        "next_day_test_records", "next_day_valid_volume",
        "next_day_single_test", "next_day_repeated_name_conflict",
        "next_day_distinct_methods",
    ]), on=keys, how="left")
    evidence = evidence.merge(baseline.assign(has_same_day=True),
                              on=keys, how="left")
    existing = evidence.has_same_day.fillna(False).astype(bool)
    has_volume = evidence.next_day_valid_volume.fillna(False).astype(bool)
    single = evidence.next_day_single_test.fillna(False).astype(bool)
    conflict = evidence.next_day_repeated_name_conflict.fillna(False).astype(bool)
    new = ~existing
    strict = new & has_volume & single & ~conflict
    metrics = {
        "cohort_patient_days": len(evidence),
        "same_day_potential_covered": int(existing.sum()),
        "same_day_potential_missing": int(new.sum()),
        "next_day_tte_present_while_missing": int((new & evidence.next_day_test_records.notna()).sum()),
        "next_day_any_valid_lvedv_while_missing": int((new & has_volume).sum()),
        "next_day_single_test_valid_no_same_name_conflict": int(strict.sum()),
        "next_day_multi_test_with_volume_while_missing": int((new & has_volume & ~single).sum()),
        "next_day_name_conflict_while_missing": int((new & has_volume & conflict).sum()),
        "potential_total_if_next_day_approved": int(existing.sum() + strict.sum()),
        "target_90pct_patients": int(len(evidence) * .9 + .999999),
    }
    return evidence, metrics
