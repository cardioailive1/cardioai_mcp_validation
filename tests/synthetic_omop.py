"""Synthetic OMOP CDM v5.4 instance (DuckDB) for offline testing.

Creates a mini vocabulary derived from config/concept_sets.yaml (source codes ->
'Maps to' standard concepts, ATC -> RxNorm ancestors, LOINC/CPT standard) and
~N synthetic patients whose records exercise every cohort definition:
echo reports, ECG reads, cath / CCTA / stress reports, HF trajectories, MACE.
The data are random and clinically meaningless — they test SQL mechanics only.
"""
from __future__ import annotations

import datetime as dt
import random

import duckdb
import pandas as pd
import yaml

DDL = """
CREATE TABLE concept (concept_id BIGINT, concept_name VARCHAR, domain_id VARCHAR, vocabulary_id VARCHAR,
  concept_class_id VARCHAR, standard_concept VARCHAR, concept_code VARCHAR, invalid_reason VARCHAR);
CREATE TABLE concept_relationship (concept_id_1 BIGINT, concept_id_2 BIGINT, relationship_id VARCHAR, invalid_reason VARCHAR);
CREATE TABLE concept_ancestor (ancestor_concept_id BIGINT, descendant_concept_id BIGINT, min_levels_of_separation INT, max_levels_of_separation INT);
CREATE TABLE location (location_id BIGINT, state VARCHAR);
CREATE TABLE care_site (care_site_id BIGINT, care_site_name VARCHAR, location_id BIGINT);
CREATE TABLE person (person_id BIGINT, gender_concept_id BIGINT, year_of_birth INT, race_concept_id BIGINT,
  ethnicity_concept_id BIGINT, care_site_id BIGINT);
CREATE TABLE observation_period (person_id BIGINT, observation_period_start_date DATE, observation_period_end_date DATE);
CREATE TABLE death (person_id BIGINT, death_date DATE, cause_concept_id BIGINT, cause_source_concept_id BIGINT);
CREATE TABLE visit_occurrence (visit_occurrence_id BIGINT, person_id BIGINT, visit_concept_id BIGINT,
  visit_start_date DATE, visit_end_date DATE, care_site_id BIGINT);
CREATE TABLE condition_occurrence (condition_occurrence_id BIGINT, person_id BIGINT, condition_concept_id BIGINT,
  condition_source_concept_id BIGINT, condition_start_date DATE, visit_occurrence_id BIGINT);
CREATE TABLE observation (observation_id BIGINT, person_id BIGINT, observation_concept_id BIGINT, observation_source_concept_id BIGINT,
  observation_date DATE, value_as_number DOUBLE, value_as_string VARCHAR, value_as_concept_id BIGINT, visit_occurrence_id BIGINT);
CREATE TABLE procedure_occurrence (procedure_occurrence_id BIGINT, person_id BIGINT, procedure_concept_id BIGINT,
  procedure_source_concept_id BIGINT, procedure_date DATE, visit_occurrence_id BIGINT);
CREATE TABLE drug_exposure (drug_exposure_id BIGINT, person_id BIGINT, drug_concept_id BIGINT, drug_source_concept_id BIGINT,
  drug_exposure_start_date DATE, visit_occurrence_id BIGINT);
CREATE TABLE device_exposure (device_exposure_id BIGINT, person_id BIGINT, device_concept_id BIGINT, device_source_concept_id BIGINT,
  device_exposure_start_date DATE, visit_occurrence_id BIGINT);
CREATE TABLE measurement (measurement_id BIGINT, person_id BIGINT, measurement_concept_id BIGINT, measurement_source_concept_id BIGINT,
  measurement_date DATE, measurement_datetime TIMESTAMP, value_as_number DOUBLE, unit_concept_id BIGINT, unit_source_value VARCHAR,
  visit_occurrence_id BIGINT);
CREATE TABLE note (note_id BIGINT, person_id BIGINT, note_date DATE, note_title VARCHAR, note_text VARCHAR, visit_occurrence_id BIGINT);
"""

NAME_HINT = {
    "lv_edv": "Left ventricular end diastolic volume by US", "lv_esv": "Left ventricular end systolic volume by US",
    "la_diam": "Left atrium Diameter by US", "la_vol": "Left atrium Volume by US",
    "lvef": "Left ventricular Ejection fraction by US", "pack_years": "Tobacco pack year",
    "family_history_ihd": "Family history of ischemic heart disease",
}


def _concrete(code: str) -> str:
    if "%" not in code:
        return code
    return code.replace("%", "0") if "." in code else code.replace("%", ".0", 1).replace("%", "0")


class Vocab:
    def __init__(self, concept_sets: dict):
        self.rows, self.rel, self.anc = [], [], []
        self.next_id = 1_000_000
        self.sets: dict[str, list[tuple[int, int]]] = {}  # set -> [(standard_id, source_id)]
        for name, spec in concept_sets.items():
            dom = spec.get("domain", "Condition")
            pairs = []
            for inc in spec.get("include", []):
                vocab = inc["vocab"]
                codes = inc.get("codes") or [f"NL-{name}"]
                for code in codes:
                    cc = _concrete(code)
                    label = NAME_HINT.get(name, f"{name} {cc}")
                    if vocab in ("ICD10CM", "ICD9CM", "ICD10", "HCPCS"):
                        src = self._add(f"{label}", dom, vocab, None, cc)
                        std = self._add(f"{label} (SNOMED)", dom, "SNOMED", "S", f"S{src}")
                        self.rel.append((src, std, "Maps to", None))
                        pairs.append((std, src))
                    elif vocab == "ATC":
                        cls = self._add(f"ATC {cc}", "Drug", "ATC", "C", cc)
                        ing = self._add(f"{name} ingredient {cc}", "Drug", "RxNorm", "S", f"R{cls}")
                        self.anc.append((cls, ing, 1, 1))
                        pairs.append((ing, ing))
                    else:  # LOINC, CPT4, SNOMED name searches
                        if "name_like" in inc:
                            cc = f"NL{self.next_id}"
                        std = self._add(label, dom, vocab, "S", cc)
                        pairs.append((std, std))
            self.sets[name] = pairs

    def _add(self, name, dom, vocab, std, code):
        for r in self.rows:  # reuse identical codes across sets
            if r[3] == vocab and r[6] == code:
                return r[0]
        cid = self.next_id
        self.next_id += 1
        self.rows.append((cid, name, dom, vocab, "x", std, code, None))
        return cid

    def pick(self, name, rnd):
        return rnd.choice(self.sets[name])


def build(n_persons: int = 6000, seed: int = 7, config_dir: str | None = None) -> duckdb.DuckDBPyConnection:
    import os

    config_dir = config_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    cs = yaml.safe_load(open(os.path.join(config_dir, "concept_sets.yaml")))
    V = Vocab(cs)
    # value concepts for smoking status / race names
    V.rows += [(8527, "White", "Race", "Race", "x", "S", "5", None), (8516, "Black or African American", "Race", "Race", "x", "S", "3", None),
               (8515, "Asian", "Race", "Race", "x", "S", "2", None), (45879404, "Never smoker", "Meas Value", "LOINC", "x", "S", "LA18978-9", None),
               (45883458, "Former smoker", "Meas Value", "LOINC", "x", "S", "LA15920-4", None),
               (45881517, "Current every day smoker", "Meas Value", "LOINC", "x", "S", "LA18976-3", None)]
    rnd = random.Random(seed)
    con = duckdb.connect()
    con.execute(DDL)

    T = {k: [] for k in ["person", "op", "death", "visit", "cond", "obs", "proc", "drug", "meas", "note", "loc", "cs"]}
    ids = {"visit": 0, "row": 0}
    states = ["MN", "MN", "MN", "AZ", "FL", "WI", "IA", "MN", "AZ", "FL", "WI", "MN", "GA", "MN"]
    for i, s in enumerate(states, start=1):
        T["loc"].append((i, s))
        T["cs"].append((100 + i, f"site {i}", i))

    def nid():
        ids["row"] += 1
        return ids["row"]

    def visit(pid, d, vtype, cs_id, los=0):
        ids["visit"] += 1
        T["visit"].append((ids["visit"], pid, vtype, d, d + dt.timedelta(days=los), cs_id))
        return ids["visit"]

    def cond(pid, setname, d, vid=None):
        std, src = V.pick(setname, rnd)
        dom = cs[setname].get("domain", "Condition")
        if dom == "Observation":
            T["obs"].append((nid(), pid, std, src, d, None, None, None, vid))
        else:
            T["cond"].append((nid(), pid, std, src, d, vid))

    def proc(pid, setname, d, vid=None):
        std, src = V.pick(setname, rnd)
        T["proc"].append((nid(), pid, std, src, d, vid))

    def drug(pid, setname, d, vid=None):
        std, src = V.pick(setname, rnd)
        T["drug"].append((nid(), pid, std, src, d, vid))

    def meas(pid, setname, d, val, vid=None, unit=None, unit_src=None):
        std, src = V.pick(setname, rnd)
        T["meas"].append((nid(), pid, std, src, d, dt.datetime.combine(d, dt.time(rnd.randint(0, 23))), val, unit, unit_src, vid))

    def note(pid, d, title, text, vid=None):
        T["note"].append((nid(), pid, d, title, text, vid))

    D0, D1 = dt.date(2008, 1, 1), dt.date(2025, 6, 30)

    def rdate(a=dt.date(2012, 1, 1), b=dt.date(2023, 12, 31)):
        return a + dt.timedelta(days=rnd.randint(0, (b - a).days))

    for pid in range(1, n_persons + 1):
        sex = rnd.choice([8507, 8532])
        yob = rnd.randint(1930, 1995)
        race = rnd.choices([8527, 8516, 8515, 0], [70, 12, 8, 10])[0]
        eth = 38003563 if rnd.random() < 0.08 else 38003564
        cs_id = 100 + rnd.randint(1, len(states))
        T["person"].append((pid, sex, yob, race, eth, cs_id))
        start = D0 + dt.timedelta(days=rnd.randint(0, 900))
        end = D1 - dt.timedelta(days=rnd.randint(0, 2500)) if rnd.random() < 0.3 else D1
        died = rnd.random() < 0.08
        if died:
            ddate = end
            std, src = V.pick("cv_death_causes", rnd) if rnd.random() < 0.5 else (0, 0)
            T["death"].append((pid, ddate, std, src))
        T["op"].append((pid, start, end))

        def inwin(d):
            return start <= d <= end

        # chronic conditions
        for s, p in [("hypertension", .45), ("diabetes", .2), ("heart_failure", .15), ("cad", .2), ("atrial_fibrillation", .12),
                     ("ckd", .1), ("copd", .08), ("hyperlipidemia", .35), ("valvular_disease", .08), ("mi_any", .06),
                     ("family_history_ihd", .1), ("smoking_current", .12), ("smoking_former", .15), ("pacemaker_any", .03)]:
            if rnd.random() < p:
                d = rdate(start, end)
                if s == "pacemaker_any":
                    cond(pid, "pacemaker_icd_status", d)
                else:
                    cond(pid, s, d)
        if rnd.random() < 0.5:
            vals = [45879404, 45883458, 45881517]
            T["obs"].append((nid(), pid, V.sets["smoking_status_obs"][0][0], V.sets["smoking_status_obs"][0][0], rdate(start, end),
                             None, None, rnd.choice(vals), None))
        if sex == 8532 and rnd.random() < 0.03:
            cond(pid, "pregnancy", rdate(start, end))
        # drugs
        for s, p in [("statins", .35), ("antihypertensives", .45), ("aspirin", .25), ("cardiac_meds", .2), ("loop_diuretics", .1)]:
            if rnd.random() < p:
                for _ in range(rnd.randint(1, 4)):
                    drug(pid, s, rdate(start, end))
        # routine vitals + labs (quarterly-ish)
        d = start
        height = rnd.gauss(170 if sex == 8507 else 162, 8)
        wt = rnd.gauss(85, 18)
        while d < end:
            vid = visit(pid, d, 9202, cs_id)
            if rnd.random() < 0.15:
                meas(pid, "height", d, height / 2.54, vid, 9330, "in")
            else:
                meas(pid, "height", d, height, vid, 8582, "cm")
            w = wt + rnd.gauss(0, 2)
            if rnd.random() < 0.2:
                meas(pid, "weight", d, w / 0.45359237, vid, 8739, "lb")
            else:
                meas(pid, "weight", d, w, vid, 9529, "kg")
            if rnd.random() < 0.6:
                meas(pid, "bmi", d, w / (height / 100) ** 2, vid)
            meas(pid, "sbp", d, rnd.gauss(132, 18), vid)
            meas(pid, "dbp", d, rnd.gauss(78, 10), vid)
            meas(pid, "heart_rate", d, rnd.gauss(74, 12), vid)
            if rnd.random() < .3:
                meas(pid, "spo2", d, min(100, rnd.gauss(96, 2)), vid)
            if rnd.random() < .5:
                for s, mu, sd in [("total_chol", 190, 40), ("ldl", 110, 35), ("hdl", 50, 14), ("triglycerides", 150, 70),
                                  ("glucose_fasting", 105, 25), ("creatinine", 1.0, .3), ("egfr", 75, 20), ("hba1c", 6.0, 1.0),
                                  ("potassium", 4.2, .4), ("sodium", 139, 3), ("magnesium", 2.0, .2), ("hscrp", 3, 2),
                                  ("bnp", 150, 200), ("ntprobnp", 900, 1200), ("troponin", 10, 30)]:
                    if rnd.random() < .7:
                        meas(pid, s, d, abs(rnd.gauss(mu, sd)) + 0.01, vid)
            d += dt.timedelta(days=rnd.randint(40, 200))
        # echo
        if rnd.random() < 0.35:
            d = rdate()
            if inwin(d):
                vid = visit(pid, d, rnd.choice([9202, 9201, 9203]), cs_id)
                proc(pid, "tte", d, vid)
                ef = max(10, min(80, rnd.gauss(55, 14) if rnd.random() < .7 else rnd.uniform(10, 50)))
                if rnd.random() < .97:
                    meas(pid, "lvef", d, round(ef), vid)
                edv = rnd.gauss(120, 35)
                meas(pid, "lv_edv", d, edv, vid)
                meas(pid, "lv_esv", d, edv * (1 - ef / 100), vid)
                meas(pid, "la_diam", d, rnd.gauss(39, 6), vid)  # mm -> transformed to cm
                if rnd.random() < .7:
                    meas(pid, "la_vol", d, rnd.gauss(55, 15), vid)
                wm = "normal regional wall motion" if ef > 50 else "regional wall motion abnormalities with inferior hypokinesis"
                rv = rnd.choice(["normal", "mildly reduced", "low normal"])
                dd = rnd.choice(["grade I diastolic dysfunction", "normal diastolic function", "grade II diastolic dysfunction"])
                q = rnd.choice(["Technically difficult study.", "", ""])
                text = (f"Indication: {rnd.choice(['dyspnea', 'heart failure', 'murmur', 'chest pain'])}. {q} "
                        f"LVEF {round(ef)}%. There is {wm}. Right ventricular systolic function is {rv}. {dd}. "
                        f"{rnd.choice(['mild', 'moderate', 'trace'])} mitral regurgitation. {rnd.choice(['no', 'mild', 'severe'])} aortic stenosis.")
                note(pid, d, "Echocardiogram Report", text, vid)
        # ECG
        if rnd.random() < 0.5:
            for _ in range(rnd.randint(1, 3)):
                d = rdate()
                if not inwin(d):
                    continue
                kind = rnd.choices(["nsr", "af", "stemi", "lbbb", "rbbb", "vt", "lvh", "stt"], [40, 15, 8, 6, 6, 3, 6, 10])[0]
                vt = 9203 if kind in ("stemi", "vt") or rnd.random() < .3 else 9202
                vid = visit(pid, d, vt, cs_id, 3 if vt == 9203 else 0)
                proc(pid, "ecg_12lead", d, vid)
                hr = {"af": 110, "vt": 170}.get(kind, 72) + rnd.gauss(0, 8)
                meas(pid, "ecg_hr", d, hr, vid)
                meas(pid, "ecg_pr", d, rnd.gauss(160, 20), vid)
                meas(pid, "ecg_qrs", d, rnd.gauss(140 if "bbb" in kind else 92, 10), vid)
                meas(pid, "ecg_qt", d, rnd.gauss(400, 30), vid)
                meas(pid, "ecg_qtc", d, rnd.gauss(440, 25), vid)
                meas(pid, "ecg_qrs_axis", d, rnd.gauss(30, 40), vid)
                read = {"nsr": "Normal sinus rhythm. Normal ECG.", "af": "Atrial fibrillation with rapid ventricular response. Nonspecific T wave abnormality.",
                        "stemi": "Sinus rhythm. ST elevation in II, III, aVF consistent with acute inferior injury. Reciprocal ST depression.",
                        "lbbb": "Sinus rhythm. Left bundle branch block.", "rbbb": "Sinus rhythm. Right bundle branch block.",
                        "vt": "Ventricular tachycardia.", "lvh": "Sinus rhythm. Left ventricular hypertrophy. Left atrial enlargement.",
                        "stt": "Sinus rhythm. ST depression and T wave inversion, consider ischemia."}[kind]
                note(pid, d, "ECG 12-lead interpretation", "Indication: " + rnd.choice(["chest pain", "palpitations", "preop"]) + ". " + read, vid)
                if kind == "stemi":
                    cond(pid, "stemi", d, vid)
                    if rnd.random() < .7:
                        proc(pid, "pci", d, vid)
                if kind == "af":
                    cond(pid, "atrial_fibrillation", d, vid)
                if kind == "vt":
                    cond(pid, "ventricular_tachycardia", d, vid)
                if "bbb" in kind:
                    cond(pid, "bundle_branch_block", d, vid)
                if rnd.random() < .5:
                    meas(pid, "troponin", d, abs(rnd.gauss(500 if kind == "stemi" else 8, 20)), vid)
                    meas(pid, "potassium", d, rnd.gauss(4.2, .4), vid)
        # chest-pain outpatient presentation (risk cohort) and CAD work-up
        if rnd.random() < 0.3:
            d = rdate(dt.date(2012, 1, 1), dt.date(2020, 12, 31))
            if inwin(d):
                vid = visit(pid, d, 9202, cs_id)
                cond(pid, rnd.choice(["chest_pain", "angina"]), d, vid)
                for s, mu, sd in [("total_chol", 200, 40), ("hdl", 48, 12), ("ldl", 120, 30), ("triglycerides", 160, 60),
                                  ("glucose_fasting", 100, 20), ("creatinine", 1.0, .25)]:
                    meas(pid, s, d - dt.timedelta(days=rnd.randint(0, 200)), abs(rnd.gauss(mu, sd)), vid)
                meas(pid, "sbp", d, rnd.gauss(135, 18), vid)
                proc(pid, "ecg_12lead", d, vid)
                note(pid, d, "ECG report", rnd.choice(["Normal sinus rhythm.", "Sinus rhythm. ST depression in V4-V6.", "Sinus rhythm. T wave inversion."]), vid)
                if rnd.random() < .4:
                    d2 = d + dt.timedelta(days=rnd.randint(3, 40))
                    proc(pid, "exercise_ett", d2, vid)
                    note(pid, d2, "Exercise stress test", f"Achieved {rnd.randint(4, 13)}.{rnd.randint(0, 9)} METs. Maximum heart rate {rnd.randint(110, 175)} bpm. "
                         + rnd.choice(["Test negative for ischemia.", "Positive test: 1.5 mm horizontal ST depression; exercise-induced angina."]), vid)
                if rnd.random() < .45:
                    d3 = d + dt.timedelta(days=rnd.randint(10, 90))
                    vid3 = visit(pid, d3, 9201, cs_id, 1)
                    if rnd.random() < .8:
                        proc(pid, "coronary_angiography", d3, vid3)
                        st = [rnd.choice([0, 20, 30, 50, 70, 90, 100]) for _ in range(4)]
                        txt = (f"Indication: typical angina on exertion for 3 months. Left main {st[0]}% stenosis. LAD {st[1]}% stenosis. "
                               f"Circumflex {st[2]}% stenosis. RCA {st[3]}% stenosis. SYNTAX score {rnd.randint(0, 40)}.")
                        if max(st) < 50 and rnd.random() < .5:
                            txt = "Indication: atypical chest pain. Angiographically normal coronary arteries."
                        note(pid, d3, "Cardiac Cath Report", txt, vid3)
                    else:
                        proc(pid, "ccta", d3, vid3)
                        note(pid, d3, "Coronary CT angiography", f"CAD-RADS {rnd.randint(0, 5)}. Calcium score {rnd.randint(0, 900)}.", vid3)
                # outcomes after presentation
                for s, p in [("mi_acute", .05), ("stroke", .03), ("heart_failure", .05), ("unstable_angina", .04)]:
                    if rnd.random() < p:
                        dd = d + dt.timedelta(days=rnd.randint(40, 1700))
                        if inwin(dd):
                            vo = visit(pid, dd, 9201, cs_id, 3)
                            cond(pid, s, dd, vo)
                if rnd.random() < .05:
                    dd = d + dt.timedelta(days=rnd.randint(40, 1700))
                    if inwin(dd):
                        proc(pid, rnd.choice(["pci", "cabg"]), dd, visit(pid, dd, 9201, cs_id, 2))
        # HF trajectories (longitudinal cohort)
        if rnd.random() < 0.15:
            d = rdate()
            if inwin(d):
                vid = visit(pid, d, 9201, cs_id, rnd.randint(2, 6))
                cond(pid, "heart_failure", d, vid)
                note(pid, d, "Cardiology progress note", f"Heart failure follow-up. NYHA class {rnd.choice(['I', 'II', 'III', 'IV'])}. Dyspnea on exertion.", vid)
                meas(pid, "bnp", d, abs(rnd.gauss(600, 400)), vid)
                meas(pid, "lvef", d - dt.timedelta(days=30), rnd.uniform(15, 60), vid)
                for k in range(rnd.randint(4, 15)):
                    dd = d + dt.timedelta(days=rnd.randint(1, 180))
                    v2 = visit(pid, dd, 9202, cs_id)
                    meas(pid, "heart_rate", dd, rnd.gauss(78, 10), v2)
                    meas(pid, "sbp", dd, rnd.gauss(120, 15), v2)
                    meas(pid, "dbp", dd, rnd.gauss(72, 9), v2)
                    meas(pid, "weight", dd, wt + rnd.gauss(0, 2.5), v2, 9529, "kg")
                    if rnd.random() < .3:
                        meas(pid, rnd.choice(["bnp", "creatinine", "potassium", "sodium", "troponin"]), dd, abs(rnd.gauss(5, 3)), v2)
                    if rnd.random() < .2:
                        cond(pid, rnd.choice(["dyspnea", "edema", "chest_pain"]), dd, v2)
                for k in range(rnd.choice([0, 0, 1, 1, 2, 3])):
                    dd = d + dt.timedelta(days=rnd.randint(5, 180))
                    v3 = visit(pid, dd, rnd.choice([9201, 9203]), cs_id, 2)
                    cond(pid, rnd.choice(["heart_failure", "hf_acute"]), dd, v3)
                    if rnd.random() < .5:
                        drug(pid, "loop_diuretics", dd, v3)
                if rnd.random() < .1:
                    dd = d + dt.timedelta(days=rnd.randint(5, 180))
                    cond(pid, "mi_acute", dd, visit(pid, dd, 9201, cs_id, 3))
                if rnd.random() < .2:
                    dd = d + dt.timedelta(days=rnd.randint(5, 180))
                    proc(pid, "cardiac_procedure_any", dd, visit(pid, dd, 9201, cs_id, 1))

    con.executemany("INSERT INTO concept VALUES (?,?,?,?,?,?,?,?)", V.rows)
    con.executemany("INSERT INTO concept_relationship VALUES (?,?,?,?)", V.rel)
    con.executemany("INSERT INTO concept_ancestor VALUES (?,?,?,?)", V.anc)
    for tbl, key in [("location", "loc"), ("care_site", "cs"), ("person", "person"), ("observation_period", "op"), ("death", "death"),
                     ("visit_occurrence", "visit"), ("condition_occurrence", "cond"), ("observation", "obs"),
                     ("procedure_occurrence", "proc"), ("drug_exposure", "drug"), ("measurement", "meas"), ("note", "note")]:
        rows = T[key]
        if not rows:
            continue
        cols = [r[1] for r in con.execute(f"PRAGMA table_info('{tbl}')").fetchall()]
        df = pd.DataFrame(rows, columns=cols)
        con.register("tmp_df", df)
        con.execute(f"INSERT INTO {tbl} SELECT * FROM tmp_df")
        con.unregister("tmp_df")
    return con


if __name__ == "__main__":
    c = build(2000)
    for t in ["person", "measurement", "note", "condition_occurrence", "procedure_occurrence", "concept"]:
        print(t, c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
