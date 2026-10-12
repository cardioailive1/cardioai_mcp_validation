"""Comparator risk scores.

ACC/AHA 2013 Pooled Cohort Equations (10-year hard ASCVD risk), Goff et al.,
Circulation 2014;129:S49-S73.  Valid for ages 40-79, TC 130-320, HDL 20-100,
SBP 90-200 mg/dL / mmHg; inputs are clipped to those ranges.  Race groups other
than African American use the White equations (guideline convention).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# coefficient order: ln_age, ln_age2, ln_tc, ln_age*ln_tc, ln_hdl, ln_age*ln_hdl,
#                    ln_tsbp, ln_age*ln_tsbp, ln_usbp, ln_age*ln_usbp, smoker, ln_age*smoker, diabetes
PCE = {
    ("Female", "white"): dict(c=[-29.799, 4.884, 13.540, -3.114, -13.578, 3.149, 2.019, 0, 1.957, 0, 7.574, -1.665, 0.661], s0=0.9665, mean=-29.18),
    ("Female", "aa"):    dict(c=[17.114, 0, 0.940, 0, -18.920, 4.475, 29.291, -6.432, 27.820, -6.087, 0.691, 0, 0.874], s0=0.9533, mean=86.61),
    ("Male", "white"):   dict(c=[12.344, 0, 11.853, -2.664, -7.990, 1.769, 1.797, 0, 1.764, 0, 7.837, -1.795, 0.658], s0=0.9144, mean=61.18),
    ("Male", "aa"):      dict(c=[2.469, 0, 0.302, 0, -0.307, 0, 1.916, 0, 1.809, 0, 0.549, 0, 0.645], s0=0.8954, mean=19.54),
}


def pce_one(age, sex, race, tc, hdl, sbp, bp_treated, smoker, diabetes) -> float:
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (age, tc, hdl, sbp)) or sex not in ("Male", "Female"):
        return math.nan
    grp = "aa" if isinstance(race, str) and "black" in race.lower() else "white"
    k = PCE[(sex, grp)]
    age = min(max(age, 40), 79)
    tc = min(max(tc, 130), 320)
    hdl = min(max(hdl, 20), 100)
    sbp = min(max(sbp, 90), 200)
    la, lt, lh, ls = math.log(age), math.log(tc), math.log(hdl), math.log(sbp)
    t, u = (ls, 0.0) if bp_treated else (0.0, ls)
    x = [la, la * la, lt, la * lt, lh, la * lh, t, la * t, u, la * u, float(smoker), la * float(smoker), float(diabetes)]
    s = sum(ci * xi for ci, xi in zip(k["c"], x))
    return 1 - k["s0"] ** math.exp(s - k["mean"])


def pce(df: pd.DataFrame) -> pd.Series:
    """Vectorised over the risk cohort analytic table."""
    return df.apply(
        lambda r: pce_one(r.get("age"), r.get("gender"), r.get("race"), r.get("total_cholesterol"), r.get("hdl_cholesterol"),
                          r.get("systolic_bp"), bool(r.get("antihypertensive_meds") == 1), r.get("smoking_status") == "current",
                          bool(r.get("diabetes_diagnosis") == 1)),
        axis=1,
    ).astype(float)
