"""Staged, read-only Mayo echo measurement/report extraction preview.

This is a DRY RUN against the saved 1,500-row echo_analytic cohort.
It never writes BigQuery tables, changes configured thresholds, or replaces
existing reference values. Patient IDs, numeric values and ECHO_NOTE text
remain in PRIVATE in-memory DataFrames inside MCP; only aggregate reports
may be shared, subject to MCP disclosure review.

Reuses resolve_tte_groups() and its ACC/ORD mutual uniqueness rule.
For this conservative preview, even a provisional mutual ACC/ORD match is
held for clinical exam-identity review rather than automatically selected.
If >1 logical exam remains, or same-measurement values conflict, no value is
promoted. Later clinical approval is needed to define multi-method priority.
"""
from __future__ import annotations

import pandas as pd

from cardioai_mcp.config import load_cohort
from cardioai_mcp.qc import completeness

from .exam_resolution import resolve_tte_groups, tte_record_rows_sql
from .mcp_echo import _cfg, _macro, _t


# Exact, verified names from Mayo source type discovery. No indexed volumes.
# Do not treat volume, area, or LA dimension as interchangeable.
_VOLUME_NAMES = {
    "lv_end_diastolic_volume": (
        "LV_vol_BP_ed_2D_by_MOD",
        "LV_vol_4C_ed_2D_by_MOD",
        "LV_vol_2C_ed_2D_by_MOD",
        "LV_vol_ed_3D",
        "MC CV LV END DIASTOLIC VOLUME",
    ),
    "lv_end_systolic_volume": (
        "LV_vol_BP_es_2D_by_MOD",
        "LV_vol_4C_es_2D_by_MOD",
        "LV_vol_2C_es_2D_by_MOD",
        "LV_vol_es_3D",
        "MC CV LV END SYSTOLIC VOLUME",
    ),
}
_RANGE = {
    "lv_end_diastolic_volume": (20.0, 500.0),
    "lv_end_systolic_volume": (5.0, 450.0),
}
_KEYS = ["person_id", "index_date"]


def _quoted(name: str) -> str:
    return "'" + name.replace("'", "''") + "'"


def echo_volume_rows_sql(modality_cfg: dict, *, start: str, end: str) -> str:
    """Selected-cohort conventional-TTE source volume rows; Spark SQL."""
    tte = tte_record_rows_sql(modality_cfg, start=start, end=end)
    c = _cfg(modality_cfg)
    r = c["results"]
    rc, ty = r["columns"], r["result_type"]
    names = ", ".join(_quoted(n) for names in _VOLUME_NAMES.values() for n in names)
    return f"""
WITH tte AS ({tte})
SELECT tte.person_id, tte.index_date, tte.echo_test_fpk,
       CAST(ty.{ty['name']} AS STRING) AS result_name,
       CAST(x.{rc['value']} AS STRING) AS raw_value,
       CAST(x.ECHO_RESULTS_VALUE_UNIT_OF_MEASURE AS STRING) AS reported_unit
FROM tte
JOIN {_macro(r['table'])} x
  ON CAST(x.{rc['test_key']} AS STRING) = tte.echo_test_fpk
JOIN {_macro(ty['table'])} ty
  ON ty.{ty['key']} = x.{rc['result_type_key']}
WHERE ty.{ty['name']} IN ({names})
"""


def echo_source_note_rows_sql(modality_cfg: dict, *, start: str, end: str) -> str:
    """Same source test's ECHO_NOTE, not an unverified patient/day note."""
    tte = tte_record_rows_sql(modality_cfg, start=start, end=end)
    c = _cfg(modality_cfg)
    e = c["echo_test_columns"]
    return f"""
WITH tte AS ({tte})
SELECT tte.person_id, tte.index_date, tte.echo_test_fpk,
       CAST(t.{e['note']} AS STRING) AS echo_note
FROM tte
JOIN {_t(c, 'echo_test_table')} t
  ON CAST(t.{e['test_key']} AS STRING) = tte.echo_test_fpk
"""


def _group_keys(tests: pd.DataFrame) -> pd.DataFrame:
    """Apply the resolver's exact identifier priority at record level."""
    t = tests.copy()
    for col in ("echo_test_fpk", "accession_number", "order_proc_id"):
        t[col] = t[col].astype("string").str.strip().replace("", pd.NA)
    t["group_key"] = "TEST:" + t.echo_test_fpk
    t.loc[t.order_proc_id.notna(), "group_key"] = (
        "ORD:" + t.loc[t.order_proc_id.notna(), "order_proc_id"]
    )
    t.loc[t.accession_number.notna(), "group_key"] = (
        "ACC:" + t.loc[t.accession_number.notna(), "accession_number"]
    )
    return t


def _select_source_group(group: pd.DataFrame) -> tuple[str, set[str]]:
    """Return the sole non-provisional group, else a review disposition."""
    if group.empty:
        return "no_conventional_tte", set()
    if group.status.isin(["missing_group_identifier", "ambiguous_competing_match"]).any():
        return "unresolved_identifier", set()
    # Every paired ACC/ORD candidate remains provisional until independently
    # shown to be one examination, even when it is mutually unique by time.
    if group.status.eq("provisional_mutual_match").any():
        return "provisional_pair_requires_review", set()
    if len(group) != 1:
        return "multiple_exams", set()
    one = group.iloc[0]
    if one.status not in ("unmatched_acc", "unmatched_ord"):
        return "unresolved_identifier", set()
    return "single_identifier_group", {str(one.group_key)}


def _read_volume(rows: pd.DataFrame, element: str) -> tuple[float | None, str]:
    """Never average discordant values or assume absent unit metadata is mL."""
    sub = rows.loc[rows.result_name.isin(_VOLUME_NAMES[element])].copy()
    if sub.empty:
        return None, "no_named_measurement"
    sub["number"] = pd.to_numeric(sub.raw_value, errors="coerce")
    lo, hi = _RANGE[element]
    sub = sub[sub.number.between(lo, hi)]
    if sub.empty:
        return None, "no_numeric_in_range"
    units = sub.reported_unit.fillna("").astype(str).str.strip().str.lower()
    # A recognized result name does not override conflicting/unknown units.
    if not units.isin(("ml", "milliliter", "milliliters")).all():
        return None, "unknown_or_non_absolute_unit"
    values = sub.number.drop_duplicates()
    if len(values) != 1:
        return None, "disagreeing_measurements_review"
    return float(values.iloc[0]), "unique_consistent_ml"


def _read_report(rows: pd.DataFrame) -> tuple[str | None, str]:
    """Report is selected only within the sole established identifier group.

    An examination may have multiple copies of identical text; distinct
    source notes remain a review issue rather than first-row-wins.
    """
    if rows.empty:
        return None, "no_source_report"
    notes = rows.echo_note.dropna().astype(str).str.strip()
    notes = notes[notes.ne("")].drop_duplicates()
    if notes.empty:
        return None, "no_source_report"
    if len(notes) > 1:
        return None, "conflicting_source_reports"
    return str(notes.iloc[0]), "unique_source_report"


def preview_echo_extraction(
    cohort_days: pd.DataFrame,
    tests: pd.DataFrame,
    volume_rows: pd.DataFrame,
    note_rows: pd.DataFrame,
) -> pd.DataFrame:
    """PRIVATE per-patient/index-day preview; never print or export this frame."""
    if not set(_KEYS).issubset(cohort_days):
        raise ValueError("Cohort missing person/date keys")
    cohort = cohort_days[_KEYS].copy()
    cohort.index_date = pd.to_datetime(cohort.index_date).dt.date
    if cohort.duplicated(_KEYS).any():
        raise ValueError("Duplicate cohort person/index-day")
    required = {*_KEYS, "echo_test_fpk", "exam_datetime",
                "accession_number", "order_proc_id"}
    if required - set(tests):
        raise ValueError(f"Tests missing {sorted(required - set(tests))}")
    t = tests.copy()
    if not t.empty:
        t.index_date = pd.to_datetime(t.index_date).dt.date
        t = t.merge(cohort, on=_KEYS, how="inner", validate="many_to_one")
        t = t.drop_duplicates(_KEYS + ["echo_test_fpk"])
    resolved = resolve_tte_groups(t)
    keyed = _group_keys(t) if not t.empty else t.assign(group_key=pd.Series(dtype=str))
    mapping = keyed[_KEYS + ["echo_test_fpk", "group_key"]]
    def attach(frame: pd.DataFrame, required_col: str) -> pd.DataFrame:
        if not set(_KEYS + ["echo_test_fpk", required_col]).issubset(frame):
            raise ValueError(f"Source data missing {required_col} or exam keys")
        f = frame.copy()
        if f.empty:
            return f.assign(group_key=pd.Series(dtype=str))
        f.index_date = pd.to_datetime(f.index_date).dt.date
        f["echo_test_fpk"] = f.echo_test_fpk.astype("string").str.strip()
        return f.merge(mapping, on=_KEYS + ["echo_test_fpk"],
                       how="inner", validate="many_to_one")
    volume = attach(volume_rows, "result_name")
    notes = attach(note_rows, "echo_note")
    result = []
    for c in cohort.itertuples(index=False):
        pid, day = c.person_id, c.index_date
        g = resolved.loc[(resolved.person_id == pid) &
                         (resolved.index_date == day)]
        status, allowed = _select_source_group(g)
        row = {"person_id": pid, "index_date": day, "exam_status": status}
        if allowed:
            v = volume.loc[(volume.person_id == pid) &
                           (volume.index_date == day) &
                           (volume.group_key.isin(allowed))]
            n = notes.loc[(notes.person_id == pid) &
                          (notes.index_date == day) &
                          (notes.group_key.isin(allowed))]
            for metric in _VOLUME_NAMES:
                row[metric], row[metric + "_status"] = _read_volume(v, metric)
            row["echo_report"], row["echo_report_status"] = _read_report(n)
        else:
            for metric in _VOLUME_NAMES:
                row[metric], row[metric + "_status"] = None, "exam_not_selected"
            row["echo_report"], row["echo_report_status"] = None, "exam_not_selected"
        result.append(row)
    return pd.DataFrame(result)


def compare_echo_completeness(
    cohort_cfg, analytic: pd.DataFrame, private_preview: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All 27 fields: before vs DRY-RUN volume-only proposed changes.

    Report text may be selected privately but is NOT used to guess clinical
    wall-motion, quality or indication labels; those remain unchanged in the
    before/after table until sign-off and tested parsing are available.
    """
    keys = _KEYS
    if analytic.duplicated(keys).any() or private_preview.duplicated(keys).any():
        raise ValueError("Cohort and preview require unique person/index-days")
    a = analytic.copy()
    a.index_date = pd.to_datetime(a.index_date).dt.date
    p = private_preview.copy()
    p.index_date = pd.to_datetime(p.index_date).dt.date
    p = p[keys + list(_VOLUME_NAMES)]
    a = a.merge(p, on=keys, how="left", suffixes=("", "_preview"),
                validate="one_to_one")
    if len(a) != len(analytic):
        raise ValueError("Preview changed cohort size")
    original = completeness(cohort_cfg, analytic).rename(
        columns={"observed": "before", "flag": "before_flag"})
    for name in _VOLUME_NAMES:
        a[name] = a[name].combine_first(a[name + "_preview"])
    after = completeness(cohort_cfg, a).rename(
        columns={"observed": "after", "flag": "after_flag"})
    report = original[["element", "domain", "required", "expected",
                       "before", "before_flag"]].merge(
        after[["element", "after", "after_flag"]],
        on="element", validate="one_to_one")
    report["change_pp"] = ((report.after - report.before) * 100).round(2)
    report["gap_to_target_pp"] = ((report.after - report.expected) * 100).round(2)
    return report, a


def run_echo_extraction_preview(eng, study: dict):
    """MCP read-only runner: (private, 27-field comparison, safe summary).

    Never print/export private or the second DataFrame from
    compare_echo_completeness(); they contain patient-level information.
    """
    start, end = str(study["index_start"]), str(study["index_end"])
    tests = eng.query(tte_record_rows_sql(study["modality"], start=start, end=end))
    volume = eng.query(echo_volume_rows_sql(study["modality"], start=start, end=end))
    notes = eng.query(echo_source_note_rows_sql(study["modality"], start=start, end=end))
    analytic = eng.read_table("echo_analytic")
    private = preview_echo_extraction(analytic, tests, volume, notes)
    cohort = load_cohort("echo")
    report, _ = compare_echo_completeness(cohort, analytic, private)
    statuses = private.exam_status.value_counts().rename_axis(
        "exam_status").reset_index(name="patient_days")
    # Aggregates require MCP disclosure review, especially status small cells.
    return private, report, statuses
