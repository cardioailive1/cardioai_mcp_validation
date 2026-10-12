"""Data-quality checks and disclosure-controlled exports (data stays behind glass)."""
from __future__ import annotations

import os

import pandas as pd

from .config import Cohort


def completeness(cohort: Cohort, analytic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for e in cohort.deck_elements:
        obs = float(analytic[e.name].notna().mean()) if e.name in analytic else 0.0
        rows.append(dict(element=e.name, domain=e.domain, required=e.required, expected=e.expected_completeness,
                         observed=obs, gap=obs - e.expected_completeness,
                         flag="BELOW TARGET" if obs + 1e-9 < e.expected_completeness else ""))
    return pd.DataFrame(rows)


def site_summary(analytic: pd.DataFrame, small_cell: int = 11) -> pd.DataFrame:
    s = analytic.groupby("site").size().rename("n").reset_index()
    s["n"] = s["n"].where(s["n"] >= small_cell, other=pd.NA)  # suppress small cells
    return s


def suppress(df: pd.DataFrame, count_cols: list[str], small_cell: int = 11) -> pd.DataFrame:
    out = df.copy()
    for c in count_cols:
        out[c] = out[c].where((out[c].isna()) | (out[c] >= small_cell) | (out[c] == 0), other=pd.NA)
    return out


def table1(cohort: Cohort, analytic: pd.DataFrame, small_cell: int = 11) -> pd.DataFrame:
    """Aggregate baseline characteristics (n, %, median [IQR]) — safe to export."""
    rows = []
    for e in cohort.deck_elements:
        if e.name not in analytic or e.is_series or e.domain == "outcome":
            continue
        s = analytic[e.name].dropna()
        if s.empty:
            continue
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 2:
            q = s.quantile([0.25, 0.5, 0.75])
            rows.append(dict(variable=e.name, level="median [IQR]", value=f"{q[0.5]:.1f} [{q[0.25]:.1f}-{q[0.75]:.1f}]", n=len(s)))
        else:
            for lvl, cnt in s.astype(str).value_counts().items():
                rows.append(dict(variable=e.name, level=lvl, value=f"{100 * cnt / len(s):.1f}%", n=int(cnt)))
    t = pd.DataFrame(rows)
    if not t.empty:
        mask = t["n"] < small_cell
        t.loc[mask, ["value", "n"]] = ["<suppressed>", pd.NA]
    return t


def export(cohort: Cohort, name: str, df: pd.DataFrame, export_dir: str) -> str:
    os.makedirs(export_dir, exist_ok=True)
    path = os.path.join(export_dir, f"{cohort.key}_{name}.csv")
    df.to_csv(path, index=False)
    return path
