"""Synthetic ACC/ORD sole-pair sensitivity tests; no clinical data."""
import pandas as pd

from echo_wm_training.cohort_extraction_preview import preview_echo_extraction
from echo_wm_training.provisional_pair_feasibility import (
    _classify_pair_only, summarize_provisional_pair_feasibility,
)


def _synthetic():
    day = "2020-03-02"
    tests = []
    vols = []
    def add(pid, fpk, minute, acc=None, order=None):
        tests.append(dict(person_id=pid, index_date=day, echo_test_fpk=fpk,
                          exam_datetime=f"{day} {minute}", accession_number=acc,
                          order_proc_id=order, test_description="Adult TTE"))
    def vol(pid, fpk, name, value, unit="mL"):
        vols.append(dict(person_id=pid, index_date=day, echo_test_fpk=fpk,
                         result_name=name, raw_value=value, reported_unit=unit))
    # 1: sole pair, biplane volume on accession side, ESV only order side.
    add(1, "acc1", "10:00:00", acc="A1")
    add(1, "ord1", "10:00:15", order="O1")
    vol(1, "acc1", "LV_vol_BP_ed_2D_by_MOD", "110")
    vol(1, "ord1", "LV_vol_BP_es_2D_by_MOD", "49")
    # 2: pair plus a genuinely different candidate exam: must NOT combine.
    add(2, "acc2", "10:00:00", acc="A2")
    add(2, "ord2", "10:00:11", order="O2")
    add(2, "second2", "13:00:00", acc="A2SECOND")
    vol(2, "acc2", "LV_vol_BP_ed_2D_by_MOD", "140")
    # 3: two mutual pairs are NOT a single exam.
    add(3, "acc3", "10:00:00", acc="A3")
    add(3, "ord3", "10:00:10", order="O3")
    add(3, "acc3b", "13:00:00", acc="A3b")
    add(3, "ord3b", "13:00:14", order="O3b")
    # 4: single unmatched group: outside 718 withheld pair days.
    add(4, "acc4", "10:00:00", acc="A4")
    # 5: sole pair but conflicting biplane in both sources.
    add(5, "acc5", "10:00:00", acc="A5")
    add(5, "ord5", "10:00:10", order="O5")
    vol(5, "acc5", "LV_vol_BP_ed_2D_by_MOD", "120")
    vol(5, "ord5", "LV_vol_BP_ed_2D_by_MOD", "125")
    # 6: sole pair with only unitless MC CV source value.
    add(6, "acc6", "10:00:00", acc="A6")
    add(6, "ord6", "10:00:10", order="O6")
    vol(6, "ord6", "MC CV LV END DIASTOLIC VOLUME", "150", None)
    # 7: sole pair with explicit mL single-plane on accession.
    add(7, "acc7", "10:00:00", acc="A7")
    add(7, "ord7", "10:00:10", order="O7")
    vol(7, "acc7", "LV_vol_4C_ed_2D_by_MOD", "133")
    cohort = pd.DataFrame([
        dict(person_id=i, index_date=day) for i in range(1, 8)
    ])
    t = pd.DataFrame(tests)
    v = pd.DataFrame(vols)
    notes = pd.DataFrame(columns=[
        "person_id", "index_date", "echo_test_fpk", "echo_note"
    ])
    private = preview_echo_extraction(cohort, t, v, notes)
    return private, t, v


def test_pair_only_topology_does_not_promote_or_mix_separate_exams():
    private, tests, volume = _synthetic()
    assert private.loc[private.person_id.eq(1), "exam_status"].iloc[0] == (
        "provisional_pair_requires_review"
    )
    assert pd.isna(private.loc[private.person_id.eq(1),
                              "lv_end_diastolic_volume"].iloc[0])
    topology, methods = summarize_provisional_pair_feasibility(
        private, tests, volume, small_cell=1
    )
    assert topology.patient_days.tolist() == ["4", "2"]
    edv = methods.loc[methods.metric.eq("lv_end_diastolic_volume")]
    assert edv.patient_days.tolist() == ["2", "1", "1"]
    esv = methods.loc[methods.metric.eq("lv_end_systolic_volume")]
    assert esv.patient_days.tolist() == ["1", "0", "3"]
    assert "person_id" not in topology and "person_id" not in methods


def test_small_cells_suppress_all_topology_and_method_categories():
    private, tests, volume = _synthetic()
    topology, methods = summarize_provisional_pair_feasibility(
        private, tests, volume, small_cell=11
    )
    assert (topology.patient_days == "suppressed").all()
    assert (methods.patient_days == "suppressed").all()


def test_invalid_nonreciprocal_pair_is_rejected():
    bad = pd.DataFrame([
        dict(status="provisional_mutual_match",
             key_type="ACC", group_key="ACC:A", partner_key="ORD:O"),
        dict(status="provisional_mutual_match",
             key_type="ORD", group_key="ORD:O", partner_key="ACC:WRONG"),
    ])
    try:
        _classify_pair_only(bad)
        assert False, "Expected nonreciprocal pair to be rejected"
    except ValueError as exc:
        assert "not reciprocal" in str(exc)
