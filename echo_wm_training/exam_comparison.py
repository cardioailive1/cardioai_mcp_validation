"""Read-only comparison of measurements on provisional ACC/ORD TTE pairs.

All row-level identifiers stay in the MCP workspace. This module never writes
BigQuery tables, approves clinical linkage, or changes cohort reference values.
"""
from __future__ import annotations

import re

import pandas as pd

from .exam_resolution import resolve_tte_groups, tte_record_rows_sql
from .exam_results import LVEDV_PRIORITY
from .mcp_echo import _cfg, _macro, _rx, parse_lvef


def tte_measurement_rows_sql(modality_cfg: dict, *, start: str, end: str) -> str:
    """Fetch TTE-classified records and their LVEF/LVEDV measurements only."""
    tte = tte_record_rows_sql(modality_cfg, start=start, end=end)
    c = _cfg(modality_cfg)
    r = c["results"]
    rc, ty = r["columns"], r["result_type"]
    label = (f"CONCAT(COALESCE(CAST(ty.{ty['name']} AS STRING), ''), ' | ', "
             f"COALESCE(CAST(ty.{ty['description']} AS STRING), ''))")
    # The accession-side Mayo result catalog names EF explicitly. Keep this
    # alongside the configured LVEF regex, rather than depending on the
    # description remaining unchanged.
    ef = (f"(ty.{ty['name']} = 'MC CV ECHO EJECTION FRACTION' OR "
          f"({label} RLIKE {_rx(r['lvef_regex'])} "
          f"AND NOT ({label} RLIKE {_rx(r['exclude_regex'])})))")
    volumes = ", ".join(f"'{v}'" for v in LVEDV_PRIORITY)
    return f"""
    WITH tte AS ({tte})
    SELECT tte.*,
           CAST(ty.{ty['name']} AS STRING) AS result_name,
           {label} AS result_label,
           CAST(x.{rc['value']} AS STRING) AS result_value,
           CASE WHEN ty.{ty['name']} IN ({volumes}) THEN 'lvedv'
                ELSE 'lvef' END AS metric
    FROM tte
    LEFT JOIN {_macro(r['table'])} x
      ON CAST(x.{rc['test_key']} AS STRING) = tte.echo_test_fpk
    LEFT JOIN {_macro(ty['table'])} ty
      ON ty.{ty['key']} = x.{rc['result_type_key']}
    WHERE ty.{ty['name']} IN ({volumes}) OR {ef}
    """


def _clean_string(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace("", pd.NA)


def _measurement_groups(rows: pd.DataFrame, results_cfg: dict) -> pd.DataFrame:
    """Values per patient/index-date/identifier group without cross-group mixing."""
    cols = ["person_id", "index_date", "group_key", "lvef", "lvedv"]
    if rows.empty:
        return pd.DataFrame(columns=cols)
    d = rows.copy()
    d["index_date"] = pd.to_datetime(d.index_date).dt.date
    acc = _clean_string(d.accession_number)
    order = _clean_string(d.order_proc_id)
    fpk = _clean_string(d.echo_test_fpk)
    d["group_key"] = "TEST:" + fpk
    d.loc[order.notna(), "group_key"] = "ORD:" + order[order.notna()]
    d.loc[acc.notna(), "group_key"] = "ACC:" + acc[acc.notna()]
    methods = [(x["method"], re.compile(x["regex"], re.I))
               for x in results_cfg["method_priority"]]
    rank = {name: idx for idx, name in enumerate(LVEDV_PRIORITY)}
    out = []
    for (pid, dt, key), g in d.groupby(["person_id", "index_date", "group_key"]):
        ef = g[g.metric == "lvef"].copy()
        ef["number"] = ef.result_value.map(parse_lvef)
        ef = ef[ef.number.notna()]
        if not ef.empty:
            def ef_rank(label):
                return next((i for i, (_, rx) in enumerate(methods)
                             if rx.search(str(label))), len(methods))
            ef["rank"] = ef.result_label.map(ef_rank)
            ef = ef[ef["rank"] == ef["rank"].min()]
            lvef = float(ef.number.median())
        else:
            lvef = None
        edv = g[g.metric == "lvedv"].copy()
        edv["number"] = pd.to_numeric(edv.result_value, errors="coerce")
        edv = edv[edv.number.between(20, 500) & edv.result_name.isin(rank)]
        if not edv.empty:
            edv["rank"] = edv.result_name.map(rank)
            edv = edv[edv["rank"] == edv["rank"].min()]
            lvedv = float(edv.number.median())
        else:
            lvedv = None
        out.append(dict(person_id=pid, index_date=dt, group_key=key,
                        lvef=lvef, lvedv=lvedv))
    return pd.DataFrame(out, columns=cols)


def compare_provisional_pairs(tte_rows: pd.DataFrame, measurement_rows: pd.DataFrame,
                              results_cfg: dict, *, max_seconds: int = 60) -> pd.DataFrame:
    """Return ONE row per provisional ACC/ORD pair, for analysis within MCP.

    The returned table contains patient/group identifiers; do not export it.
    Only provisional mutual matches are compared; ambiguous groups excluded.
    """
    cols = ["person_id", "index_date", "acc_group", "ord_group",
            "closest_seconds", "acc_lvef", "ord_lvef", "acc_lvedv", "ord_lvedv",
            "lvef_abs_difference", "lvedv_abs_difference"]
    if tte_rows.empty:
        return pd.DataFrame(columns=cols)
    resolved = resolve_tte_groups(tte_rows, max_seconds=max_seconds)
    selected = resolved[(resolved.status == "provisional_mutual_match") &
                        (resolved.key_type == "ACC")].copy()
    if selected.empty:
        return pd.DataFrame(columns=cols)
    measurements = _measurement_groups(measurement_rows, results_cfg)
    lookup = {(r.person_id, r.index_date, r.group_key): r
              for r in measurements.itertuples(index=False)}
    out = []
    for r in selected.itertuples(index=False):
        acc = lookup.get((r.person_id, r.index_date, r.group_key))
        order = lookup.get((r.person_id, r.index_date, r.partner_key))
        aef, oef = (acc.lvef if acc else None), (order.lvef if order else None)
        av, ov = (acc.lvedv if acc else None), (order.lvedv if order else None)
        out.append(dict(person_id=r.person_id, index_date=r.index_date,
                        acc_group=r.group_key, ord_group=r.partner_key,
                        closest_seconds=r.closest_seconds,
                        acc_lvef=aef, ord_lvef=oef,
                        acc_lvedv=av, ord_lvedv=ov,
                        lvef_abs_difference=(abs(aef-oef) if pd.notna(aef) and pd.notna(oef) else None),
                        lvedv_abs_difference=(abs(av-ov) if pd.notna(av) and pd.notna(ov) else None)))
    return pd.DataFrame(out, columns=cols)


def comparison_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate coverage and disagreement; no identifiers in the summary."""
    cols = ["metric", "pairs", "both", "acc_only", "ord_only", "neither",
            "median_abs_difference", "p90_abs_difference"]
    if pairs.empty:
        return pd.DataFrame(columns=cols)
    out = []
    for metric in ("lvef", "lvedv"):
        acc, order = pairs["acc_"+metric].notna(), pairs["ord_"+metric].notna()
        diffs = pairs[metric+"_abs_difference"].dropna()
        out.append(dict(metric=metric, pairs=len(pairs),
                        both=int((acc & order).sum()),
                        acc_only=int((acc & ~order).sum()),
                        ord_only=int((~acc & order).sum()),
                        neither=int((~acc & ~order).sum()),
                        median_abs_difference=float(diffs.median()) if len(diffs) else None,
                        p90_abs_difference=float(diffs.quantile(.9)) if len(diffs) else None))
    return pd.DataFrame(out, columns=cols)


def build_tte_measurement_comparison(engine, modality_cfg: dict, *, start: str, end: str,
                                     max_seconds: int = 60, log=print):
    """Return (pair_details, aggregate_summary) in MCP; no writes."""
    tte_sql = tte_record_rows_sql(modality_cfg, start=start, end=end)
    tests = engine.query(engine.render(tte_sql, {}))
    result_sql = tte_measurement_rows_sql(modality_cfg, start=start, end=end)
    rows = engine.query(engine.render(result_sql, {}))
    pairs = compare_provisional_pairs(tests, rows, _cfg(modality_cfg)["results"],
                                      max_seconds=max_seconds)
    summary = comparison_summary(pairs)
    log(f"Provisional ACC/ORD pairs compared: {len(pairs)}")
    if not summary.empty:
        log(summary.to_string(index=False))
    return pairs, summary
