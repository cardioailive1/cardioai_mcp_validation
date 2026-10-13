"""End-to-end test: synthetic OMOP -> 5 cohorts -> simulated agent outputs -> endpoint table.

    python -m pytest -q tests/        (or)   python tests/test_end_to_end.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cardioai_mcp import COHORTS  # noqa: E402
from cardioai_mcp.config import load_cohort, load_study  # noqa: E402
from cardioai_mcp.engine import DuckDBEngine  # noqa: E402
from cardioai_mcp.evaluate import evaluate  # noqa: E402
from cardioai_mcp.pipeline import build_all, data_dictionary  # noqa: E402
from cardioai_mcp.risk_scores import pce_one  # noqa: E402
from tests.synthetic_omop import build  # noqa: E402

RNG = np.random.default_rng(0)


def _flip(s: pd.Series, p: float, choices) -> pd.Series:
    s = s.copy()
    m = RNG.random(len(s)) < p
    s[m] = RNG.choice(list(choices), m.sum())
    return s


def simulate_predictions(key: str, a: pd.DataFrame) -> pd.DataFrame:
    p = pd.DataFrame({"person_id": a.person_id})
    if key == "echo":
        p["agent_lvef"] = a.lvef_expert + RNG.normal(0, 3.5, len(a))
        p["agent_ef_category"] = np.where(p.agent_lvef <= 40, "reduced", np.where(p.agent_lvef < 50, "mildly_reduced", "normal"))
        p["agent_wall_motion"] = _flip(a.wall_motion.fillna("normal"), 0.05, ["normal", "abnormal"])
        p["agent_lv_size"] = _flip(a.ref_lv_size.fillna("normal"), 0.05, ["normal", "dilated"])
        p["runtime_s"] = RNG.gamma(9, 5, len(a))
    elif key == "ecg":
        p["agent_superclass"] = _flip(a.ref_superclass.fillna("NORM"), 0.03, ["NORM", "MI", "STTC", "CD", "HYP"])
        p["agent_stemi"] = _flip(a.ref_stemi.fillna(0).astype(int), 0.03, [0, 1])
        p["agent_rhythm"] = _flip(a.rhythm_classification.fillna("SR"), 0.03, ["SR", "AFIB", "VT", "OTHER"])
        p["agent_bbb"] = _flip(a.bundle_branch_block.fillna("NONE"), 0.03, ["NONE", "LBBB", "RBBB"])
        p["runtime_s"] = RNG.gamma(4, 0.5, len(a))
    elif key == "cad":
        y = a.cad_present.fillna(0).astype(int)
        p["agent_cad_prob"] = np.clip(0.75 * y + RNG.normal(0.12, 0.15, len(a)), 0, 1)
        p["agent_cad"] = (p.agent_cad_prob >= 0.5).astype(int)
        p["agent_multivessel"] = _flip(a.ref_multivessel.fillna(0).astype(int), 0.06, [0, 1])
        p["agent_risk_category"] = _flip(a.ref_risk_category.fillna("low"), 0.06, ["low", "intermediate", "high"])
    elif key == "risk":
        y = a.ref_heart_disease.fillna(0).astype(int)
        p["agent_hd_prob"] = np.clip(0.2 + 0.55 * y + RNG.normal(0, 0.12, len(a)), 0.001, 0.999)
        p["agent_hd"] = (p.agent_hd_prob >= 0.5).astype(int)
    elif key == "longitudinal":
        p["risk_mi_30d"] = np.clip(0.8 * a.ref_mi_30d.fillna(0) + RNG.normal(0.1, 0.1, len(a)), 0, 1)
        p["risk_event_30d"] = np.clip(0.6 * a.ref_event_30d.fillna(0) + RNG.normal(0.2, 0.15, len(a)), 0, 1)
        p["pred_readmit_30d"] = _flip(a.ref_readmit_30d.fillna(0).astype(int), 0.1, [0, 1])
        p["pred_risk_90d"] = _flip(a.ref_event_90d.fillna(0).astype(int), 0.1, [0, 1])
    return p


def simulate_daily(a: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    ev = series[series.element == "hf_exacerbation_dates"][["person_id", "day"]]
    ac = series[series.element == "acute_cardiac_dates"][["person_id", "day"]]
    rows = []
    evd = ev.groupby("person_id").day.apply(list).to_dict()
    acd = ac.groupby("person_id").day.apply(list).to_dict()
    for pid in a.person_id:
        for d in range(0, 180):
            near = any(0 < e - d <= 7 for e in evd.get(pid, []))
            near1 = any(0 < e - d <= 1 for e in acd.get(pid, []))
            rows.append((pid, d, int(near and RNG.random() < 0.9 or RNG.random() < 0.02), int(near1 or RNG.random() < 0.01)))
    return pd.DataFrame(rows, columns=["person_id", "day", "alert_hf_7d", "alert_acute_24h"])


@pytest.fixture(scope="module")
def built():
    study = load_study()
    study["schemas"].update(cdm="main", vocab="main", modality="main", work="main")
    eng = DuckDBEngine(build(6000), study)
    return eng, build_all(eng, COHORTS, study, os.path.join(ROOT, "exports", "synthetic_test"))


def test_configs_match_deck():
    for k in COHORTS:
        load_cohort(k)  # raises if element counts / REQ counts / avg completeness differ from the deck
    dd = data_dictionary(COHORTS)
    assert len(dd) == 27 + 30 + 41 + 41 + 40


def test_pce_reference_values():
    assert round(100 * pce_one(55, "Female", "White", 213, 50, 120, False, False, False), 1) == 2.1
    assert round(100 * pce_one(55, "Female", "Black or African American", 213, 50, 120, False, False, False), 1) == 3.0
    assert abs(100 * pce_one(55, "Male", "White", 213, 50, 120, False, False, False) - 5.3) < 0.1
    assert round(100 * pce_one(55, "Male", "Black or African American", 213, 50, 120, False, False, False), 1) == 6.1


def test_cohorts_build(built):
    eng, res = built
    for k, r in res.items():
        c = r["cohort"]
        a = r["analytic"]
        eligible = int(r["attrition"].iloc[-2]["persons"])  # after 'one index per patient'
        assert len(a) == min(c.target_n, eligible), f"{k}: {len(a)} rows != min(target {c.target_n}, eligible {eligible})"
        assert a.person_id.is_unique
        for e in c.deck_elements:
            assert e.name in a.columns, f"{k}.{e.name} missing"


def test_endpoints_evaluate(built):
    eng, res = built
    for k, r in res.items():
        a = r["analytic"]
        preds = simulate_predictions(k, a)
        series = daily = None
        if k == "longitudinal":
            series = eng.query("SELECT * FROM longitudinal_series")
            daily = simulate_daily(a, series)
        ep = evaluate(r["cohort"], a, preds, series=series, daily=daily, n_boot=200)
        errs = ep[ep.note.astype(str).str.startswith("ERROR")]
        assert errs.empty, errs.to_string()
        assert ep.estimate.notna().sum() >= len(ep) - 1, ep.to_string()
        ep.to_csv(os.path.join(ROOT, "exports", "synthetic_test", f"{k}_endpoints_SIMULATED.csv"), index=False)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-x"]))
