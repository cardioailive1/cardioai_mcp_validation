"""Read-only, provisional matching of accession and order-only Mayo TTE groups.

A <=60-second timestamp match is a *candidate*, not proof of one clinical exam.
No BigQuery tables are written. All identifiers remain inside the MCP workspace.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .mcp_echo import _cfg, _macro, _t


def tte_record_rows_sql(modality_cfg: dict, *, start: str, end: str) -> str:
    """Retrieve all TTE-classified tests for saved Echo cohort index dates.

    Unlike the measurement candidate query, this does not require an LVEF or
    LVEDV result, so measurement absence cannot create artificial uniqueness.
    """
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last:
        raise ValueError("start must be on or before end")
    c = _cfg(modality_cfg)
    e, d, pm, tt = (c["echo_test_columns"], c["dim_patient"],
                    c.get("person_map") or {}, c["test_type"])
    if not pm.get("table") or not tt.get("table"):
        raise ValueError("Echo person_map and test_type are required")
    return f"""
    SELECT DISTINCT a.person_id,
           CAST(a.index_date AS DATE) AS index_date,
           CAST(t.{e['test_key']} AS STRING) AS echo_test_fpk,
           CAST(t.{e['datetime']} AS DATETIME) AS exam_datetime,
           CAST(t.{e['link']} AS STRING) AS accession_number,
           CAST(t.ORDER_PROC_ID AS STRING) AS order_proc_id,
           CAST(tt.{tt['description']} AS STRING) AS test_description
    FROM {{{{work}}}}.echo_analytic a
    JOIN {_macro(pm['table'])} pm
      ON pm.{pm['person_id_column']} = a.person_id
    JOIN {_macro(d['table'])} d
      ON CAST(d.{d['clinic_number']} AS STRING) = CAST(pm.{pm['clinic_number_column']} AS STRING)
    JOIN {_t(c, 'echo_test_table')} t
      ON t.{e['patient_dk']} = d.{d['patient_dk']}
    JOIN {_macro(tt['table'])} tt
      ON tt.{tt['key']} = t.{e['performed_test_dk']}
    WHERE CAST(a.index_date AS DATE) BETWEEN CAST('{start}' AS DATE) AND CAST('{end}' AS DATE)
      AND CAST(t.{e['datetime']} AS DATE) = CAST(a.index_date AS DATE)
      AND LOWER(CAST(tt.{tt['description']} AS STRING)) RLIKE 'tte|transthoracic|echocardiogram'
      AND NOT (LOWER(CAST(tt.{tt['description']} AS STRING)) RLIKE
            'tee|transesophageal|stress|strain|outside|review|fetal')
    """


def resolve_tte_groups(rows: pd.DataFrame, *, max_seconds: int = 60) -> pd.DataFrame:
    """Return a record-level grouping and conservative match disposition.

    Independently consolidate records with the same patient/date and identifier.
    For an ACC group and an order-only ORD group, calculate their minimum
    timestamp distance. Link ONLY when each group has exactly one candidate
    counterpart inside the threshold (mutual uniqueness). A second plausible
    counterpart is ambiguous, even if one is closer. No transitive merging.
    """
    if max_seconds < 0:
        raise ValueError("max_seconds must not be negative")
    cols = ["person_id", "index_date", "exam_date", "group_key",
            "key_type", "test_records", "status", "partner_key",
            "closest_seconds"]
    if rows.empty:
        return pd.DataFrame(columns=cols)
    req = {"person_id", "index_date", "echo_test_fpk", "exam_datetime",
           "accession_number", "order_proc_id"}
    if req - set(rows):
        raise ValueError(f"Missing fields: {sorted(req - set(rows))}")

    df = rows.copy()
    df["index_date"] = pd.to_datetime(df.index_date).dt.date
    df["exam_datetime"] = pd.to_datetime(df.exam_datetime)
    df["exam_date"] = df.exam_datetime.dt.date
    for field in ("accession_number", "order_proc_id", "echo_test_fpk"):
        df[field] = df[field].astype("string").str.strip().replace("", pd.NA)
    if df["person_id"].isna().any() or df["exam_datetime"].isna().any() or df["echo_test_fpk"].isna().any():
        raise ValueError("Every TTE record must have patient, examination timestamp, and ECHO_TEST_FPK")
    # Avoid accidental cross-date matching, even when identifiers recur.
    df["key_type"] = "TEST"
    df.loc[df.order_proc_id.notna(), "key_type"] = "ORD"
    df.loc[df.accession_number.notna(), "key_type"] = "ACC"
    df["group_key"] = "TEST:" + df.echo_test_fpk
    df.loc[df.order_proc_id.notna(), "group_key"] = "ORD:" + df.loc[df.order_proc_id.notna(), "order_proc_id"]
    df.loc[df.accession_number.notna(), "group_key"] = "ACC:" + df.loc[df.accession_number.notna(), "accession_number"]

    output = []
    for (person_id, index_date, exam_date), patient in df.groupby(
        ["person_id", "index_date", "exam_date"], dropna=False
    ):
        groups = {}
        for key, g in patient.groupby("group_key"):
            groups[str(key)] = {
                "type": g.key_type.iloc[0],
                "times": tuple(g.exam_datetime.drop_duplicates()),
                "records": g.echo_test_fpk.nunique(),
            }
        accs = [k for k, v in groups.items() if v["type"] == "ACC"]
        orders = [k for k, v in groups.items() if v["type"] == "ORD"]
        possible = {key: {} for key in accs + orders}
        for a in accs:
            for o in orders:
                seconds = min(abs((ta - to).total_seconds())
                              for ta in groups[a]["times"] for to in groups[o]["times"])
                if seconds <= max_seconds:
                    possible[a][o] = seconds
                    possible[o][a] = seconds

        for key, group in groups.items():
            partner = None
            delta = None
            if group["type"] == "TEST":
                status = "missing_group_identifier"
            elif len(possible[key]) > 1:
                status = "ambiguous_competing_match"
            elif len(possible[key]) == 1:
                candidate, delta = next(iter(possible[key].items()))
                if len(possible[candidate]) == 1:
                    status = "provisional_mutual_match"
                    partner = candidate
                else:
                    status = "ambiguous_competing_match"
                    delta = None
            else:
                status = "unmatched_" + group["type"].lower()
            output.append({
                "person_id": person_id,
                "index_date": index_date,
                "exam_date": exam_date,
                "group_key": key,
                "key_type": group["type"],
                "test_records": group["records"],
                "status": status,
                "partner_key": partner,
                "closest_seconds": delta,
            })
    return pd.DataFrame(output, columns=cols)


def build_tte_match_diagnostics(engine, modality_cfg: dict, *, start: str, end: str,
                                max_seconds: int = 60, log=print) -> pd.DataFrame:
    """Run read-only cohort-scoped diagnostics; return identifiers only within MCP."""
    sql = tte_record_rows_sql(modality_cfg, start=start, end=end)
    rows = engine.query(engine.render(sql, {}))
    out = resolve_tte_groups(rows, max_seconds=max_seconds)
    log(f"TTE matching: {len(rows)} test rows, {len(out)} groups")
    if not out.empty:
        log(f"Group statuses: {out.status.value_counts().to_dict()}")
        pairs = int((out.status == "provisional_mutual_match").sum()) // 2
        log(f"Provisional mutual ACC/ORD pairs: {pairs}; not clinically confirmed")
        log(f"Patients with ambiguous competing matches: "
            f"{out.loc[out.status == 'ambiguous_competing_match', 'person_id'].nunique()}")
    return out
