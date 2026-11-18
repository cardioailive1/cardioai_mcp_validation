"""Synthetic tests for distinguishing exact-name and same-family EF conflicts."""
import pandas as pd

from cardioai_mcp.config import load_study
from echo_wm_training.ef_name_diagnostics import diagnose_within_method_names


def _case(pid, values, group="ACC"):
    key = "ACC:" + str(pid) if group == "ACC" else "ORD:" + str(pid)
    candidate = dict(
        person_id=pid, index_date=pd.Timestamp("2020-01-02"),
        group_key=key, partner_key=None, record_source=group,
        lvef_disagreement_category="within_method",
    )
    measurements = [
        dict(person_id=pid, index_date="2020-01-02",
             echo_test_fpk=str(pid),
             accession_number=str(pid) if group == "ACC" else None,
             order_proc_id=str(pid) if group == "ORD" else None,
             metric="lvef", result_name=name,
             result_label=name, result_value=str(value))
        for name, value in values
    ]
    return candidate, measurements


def test_ef_name_diagnostic_classifies_same_and_different_names():
    cfg = load_study()["modality"]["echo"]["mcp"]["results"]
    scenarios = [
        (1, [("LVEF visual", 40), ("LVEF visual", 60)], "same_name_only"),
        (2, [("LVEF visual A", 40), ("LVEF visual B", 60)], "different_names_only"),
        (3, [("LVEF visual A", 40), ("LVEF visual A", 60),
             ("LVEF visual B", 50)], "both"),
        (4, [("LVEF visual A", 40), ("LVEF biplane", 60)], "neither"),
    ]
    candidates, readings = [], []
    for pid, values, expected in scenarios:
        candidate, rows = _case(pid, values)
        candidates.append(candidate)
        readings.extend(rows)
    details, summary = diagnose_within_method_names(
        pd.DataFrame(readings), pd.DataFrame(candidates), cfg
    )
    assert len(details) == 4
    for pid, _, expected in scenarios:
        assert details.loc[details.person_id == pid, "name_conflict_class"].iloc[0] == expected
    assert summary.groups.sum() == 4


def test_matched_groups_not_compared_across_identifiers():
    cfg = load_study()["modality"]["echo"]["mcp"]["results"]
    candidate = pd.DataFrame([dict(
        person_id=8, index_date="2020-01-02",
        group_key="ACC:8", partner_key="ORD:8", record_source="ACC+ORD",
        lvef_disagreement_category="within_method",
    )])
    acc, arows = _case(8, [("LVEF visual", 20), ("LVEF visual", 40)])
    _, orows = _case(8, [("LVEF visual", 80)], group="ORD")
    details, summary = diagnose_within_method_names(
        pd.DataFrame(arows + orows), candidate, cfg
    )
    assert details.name_conflict_class.iloc[0] == "same_name_only"
    assert summary.groups.sum() == 1


def test_diagnostics_do_not_reclassify_unflagged_exams():
    cfg = load_study()["modality"]["echo"]["mcp"]["results"]
    candidate, rows = _case(9, [("LVEF visual", 40), ("LVEF visual", 60)])
    candidate["lvef_disagreement_category"] = "between_method"
    detail, summary = diagnose_within_method_names(
        pd.DataFrame(rows), pd.DataFrame([candidate]), cfg
    )
    assert detail.empty and summary.empty
