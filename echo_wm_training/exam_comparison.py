"""Read-only comparison of measurements on provisional ACC/ORD TTE pairs.

All row-level identifiers stay in the MCP workspace. This module never writes
BigQuery tables, approves clinical linkage, or changes cohort reference values.
"""
from __future__ import annotations

import re

import pandas as pd

from .exam_resolution import resolve_tte_groups, tte_record_rows_sql
from .exam_results import LVEDV_PRIORITY
from .mcp_echo import _cfg, _macro, _rx, parse_lvef, classify_lvef_method


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
    cols = ["person_id", "index_date", "group_key", "lvef", "lvedv",
            "lvef_spread_pp", "lvef_method_disagreement",
            "lvef_within_method_spread_pp", "lvef_between_method_spread_pp",
            "lvef_disagreement_category"]
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
    rank = {name: idx for idx, name in enumerate(LVEDV_PRIORITY)}
    out = []
    for (pid, dt, key), g in d.groupby(["person_id", "index_date", "group_key"]):
        ef = g[g.metric == "lvef"].copy()
        ef["number"] = ef.result_value.map(parse_lvef)
        ef = ef[ef.number.notna()]
        # Compare all plausible stored EF values within the same identifier group,
        # before applying a method priority. Never auto-resolve a large conflict.
        ef_spread = float(ef.number.max() - ef.number.min()) if len(ef) else None
        threshold = float(results_cfg.get("lvef_disagreement_threshold_pp", 10))
        ef_disagreement = ef_spread is not None and ef_spread > threshold
        within_spread = None
        between_spread = None
        disagreement_category = "none"
        if not ef.empty:
            # Classify ALL plausible measurements before selecting a preferred
            # method, so a priority change cannot hide conflicting measurements.
            classified = ef.result_label.map(
                lambda label: classify_lvef_method(label, results_cfg)
            )
            ef["rank"] = classified.map(lambda value: value[0])
            ef["method_group"] = classified.map(lambda value: value[2])
            spread_by_method = ef.groupby("method_group")["number"].agg(
                lambda values: float(values.max() - values.min())
            )
            within_spread = float(spread_by_method.max())
            if ef["method_group"].nunique() > 1:
                # Across method families, summarize each family with its median
                # to avoid comparing every outlying repeated measurement.
                method_medians = ef.groupby("method_group")["number"].median()
                between_spread = float(method_medians.max() - method_medians.min())
            within_flag = within_spread > threshold
            between_flag = between_spread is not None and between_spread > threshold
            if ef_disagreement:
                if within_flag and between_flag:
                    disagreement_category = "both"
                elif within_flag:
                    disagreement_category = "within_method"
                elif between_flag:
                    disagreement_category = "between_method"
                else:
                    # Full range can exceed threshold even though no family
                    # median or within-family range individually exceeds it.
                    disagreement_category = "mixed_spread"
            best = ef[ef["rank"] == ef["rank"].min()]
            lvef = float(best.number.median())
        else:
            lvef = None
        edv = g[g.metric == "lvedv"].copy()
        # EF-only diagnostics may have no LVEDV rows or result_name column.
        # The name is only required when actual LVEDV measurements exist.
        if not edv.empty:
            if "result_name" not in edv.columns:
                raise ValueError("LVEDV rows require result_name")
            edv["number"] = pd.to_numeric(edv.result_value, errors="coerce")
            edv = edv[edv.number.between(20, 500) & edv.result_name.isin(rank)]
        if not edv.empty:
            edv["rank"] = edv.result_name.map(rank)
            edv = edv[edv["rank"] == edv["rank"].min()]
            lvedv = float(edv.number.median())
        else:
            lvedv = None
        out.append(dict(person_id=pid, index_date=dt, group_key=key,
                        lvef=lvef, lvedv=lvedv,
                        lvef_spread_pp=ef_spread,
                        lvef_method_disagreement=bool(ef_disagreement),
                        lvef_within_method_spread_pp=within_spread,
                        lvef_between_method_spread_pp=between_spread,
                        lvef_disagreement_category=disagreement_category))
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
