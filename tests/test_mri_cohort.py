"""Phase 2 MRI cohort (config/cohorts/mri.yaml) builds on synthetic OMOP, its report-derived reference
agrees with the training label rules (mri_training.dataset.report_labels), and endpoints evaluate."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cardioai_mcp.config import load_cohort, load_study  # noqa: E402
from cardioai_mcp.engine import DuckDBEngine  # noqa: E402
from cardioai_mcp.evaluate import evaluate  # noqa: E402
from cardioai_mcp.pipeline import build_all  # noqa: E402
from tests.synthetic_omop import build  # noqa: E402

REPORTS = [
    "Cardiac MRI: subendocardial late gadolinium enhancement in the LAD territory consistent with acute myocardial infarction. "
    "Infarct size 18% of LV mass. Microvascular obstruction present. LVEF 38%. No LV thrombus.",
    "Cardiac MRI: transmural enhancement of the apex with an apical thrombus. Ejection fraction 31%. Scar burden 24%.",
    "Cardiac MRI: patchy subepicardial late gadolinium enhancement of the inferolateral wall consistent with myocarditis. LVEF 55%.",
    "Cardiac MRI: no late gadolinium enhancement to suggest myocardial scar or infarction. LVEF 62%. No MVO.",
    "Cardiac MRI: lge in the rca territory, consistent with prior infarction, with an inferior aneurysm. LVEF 44.5%.",
]


def _seed_cmr(con):
    """Contrast CMR procedures (some after an acute MI) and their signed reports."""
    cpt = con.execute("SELECT concept_id FROM concept WHERE concept_code = '75561' LIMIT 1").fetchone()[0]
    con.execute("""
      CREATE TEMP TABLE cmr_seed AS
      SELECT p.person_id, CAST(DATE '2021-03-01' + CAST((p.person_id * 37) % 900 AS INTEGER) AS DATE) AS d,
             (p.person_id % 5) AS kind
      FROM person p WHERE p.person_id % 2 = 0
      UNION ALL
      SELECT co.person_id, CAST(co.condition_start_date + INTERVAL 6 DAY AS DATE), 0
      FROM condition_occurrence co
      WHERE co.condition_start_date >= DATE '2021-01-01'
        AND co.condition_concept_id IN (SELECT concept_id FROM cs_members WHERE set_name IN ('stemi', 'mi_acute'))""")
    con.execute(f"""INSERT INTO procedure_occurrence
      SELECT 900000000 + ROW_NUMBER() OVER (), person_id, {cpt}, {cpt}, d, NULL FROM cmr_seed""")
    cases = " ".join(f"WHEN {i} THEN '{r}'" for i, r in enumerate(REPORTS))
    con.execute(f"""INSERT INTO note
      SELECT 900000000 + ROW_NUMBER() OVER (), person_id, d, 'Cardiac MRI report', CASE kind {cases} END, NULL FROM cmr_seed""")


def test_mri_cohort_builds_and_evaluates(tmp_path):
    from cardioai_mcp.concepts import resolve_concept_sets
    from mri_training.dataset import load_config, report_labels
    study = load_study()
    study["schemas"].update(cdm="main", vocab="main", modality="main", work="main")
    con = build(3000)
    eng = DuckDBEngine(con, study)
    resolve_concept_sets(eng, study["concept_sets"], only={"stemi", "mi_acute"})
    _seed_cmr(con)
    res = build_all(eng, ["mri"], study, str(tmp_path))["mri"]
    a, c = res["analytic"], res["cohort"]
    assert len(a) > 300 and a.person_id.is_unique and (pd.to_datetime(a.index_date) >= "2021-01-01").all()
    for e in c.deck_elements:
        assert e.name in a.columns, e.name
    assert a.recent_stemi.sum() > 0 or a.recent_acute_mi.sum() > 0
    # the cohort's SQL reference = the training label rules, report by report
    cfg = load_config()
    for txt, grp in a.groupby("cmr_report"):
        lab = report_labels(txt, cfg)
        row = grp.iloc[0]
        for ref, key in (("ref_infarct", "infarct"), ("ref_nonischemic_lge", "nonischemic_lge"),
                         ("ref_lv_thrombus", "lv_thrombus"), ("ref_lv_aneurysm", "lv_aneurysm"), ("ref_mvo", "report_mvo"),
                         ("lvef_report", "report_lvef"), ("infarct_size_report", "report_infarct_size")):
            v, w = row[ref], lab[key]
            assert (pd.isna(v) and np.isnan(w)) or float(v) == w, (txt[:60], ref, v, w)
    # endpoints evaluate with simulated agent output
    rng = np.random.default_rng(0)
    p = pd.DataFrame({"person_id": a.person_id})
    for head, ref in (("infarct", "ref_infarct"), ("nonischemic", "ref_nonischemic_lge"), ("thrombus", "ref_lv_thrombus"),
                      ("aneurysm", "ref_lv_aneurysm")):
        y = a[ref].fillna(0).astype(float).to_numpy()
        p[f"agent_{head}_prob"] = np.clip(0.8 * y + rng.normal(0.1, 0.1, len(a)), 0, 1)
        p[f"agent_{head}"] = (p[f"agent_{head}_prob"] >= 0.5).astype(int)
    p["agent_lvef"] = a.lvef_report.astype(float) + rng.normal(0, 3, len(a))
    p["agent_infarct_size_pct"] = a.infarct_size_report.astype(float).fillna(0) + rng.normal(0, 2, len(a)).clip(-2, None)
    p["agent_mvo"] = a.ref_mvo.fillna(0).astype(int)
    p["runtime_s"] = rng.gamma(4, 5, len(a))
    ep = evaluate(c, a, p, n_boot=100)
    assert ep[ep.note.astype(str).str.startswith("ERROR")].empty, ep.to_string()
    est = ep.set_index("endpoint").estimate
    assert {"infarct_sensitivity", "infarct_specificity", "lvef_mae"} <= set(ep[ep.tier == "primary"].endpoint)
    assert est["infarct_sensitivity"] > 0.8 and est["lvef_mae"] < 5


def test_mri_cohort_is_outside_the_deck():
    from cardioai_mcp import COHORTS, EXTRA_COHORTS
    assert "mri" in EXTRA_COHORTS and "mri" not in COHORTS
    c = load_cohort("mri")
    assert c.raw.get("expected") is None and c.calendar["index_start_mri"] == load_config_test_start()


def load_config_test_start():
    from mri_training.dataset import load_config
    return load_config()["splits"]["test_start"]
