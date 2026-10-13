"""Analytic-table rows -> the exact vectors the trained models receive.

The validation must feed the models the same way production will.  So this
module only renames cohort columns into the app's request fields
(app/validated_agents/contracts.py) and then calls the SAME encoder code the
app serves with (cardioai_mcp/encoders = app/validated_agents/preprocessing,
byte-identical, checked by fingerprint).  Missing-data rules match
`model_inputs.missing_rule` in config/cohorts/*.yaml and the app's contracts.

    from cardioai_mcp.agent_inputs import encode_cohort
    enc = encode_cohort("risk", analytic_df)   # person_id, scorable, reason, vector, imputed
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from . import encoders as enc

ENCODERS_FINGERPRINT = enc.fingerprint()


def _v(row: Dict[str, Any], key: str) -> Any:
    """Value or None (NaN / NaT / missing column -> None)."""
    x = row.get(key)
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return x


def _flag(row, key) -> bool:
    """History / exam / ECG flags: not documented -> 0, as in the training data."""
    x = _v(row, key)
    return bool(x) and x not in (0, "0", "N", "n", False)


def _num(row, key) -> Optional[float]:
    x = _v(row, key)
    return None if x is None else float(x)


# --------------------------------------------------------------------------- CAD
CAD_REQUIRED = {"age": "age", "height_cm": "height", "weight_kg": "weight", "systolic_bp": "mi_bp", "pulse_rate": "mi_pr",
                "fasting_glucose": "fasting_glucose", "creatinine": "creatinine", "triglycerides": "triglycerides",
                "ldl_cholesterol": "ldl_cholesterol", "hdl_cholesterol": "hdl_cholesterol"}
CAD_OPTIONAL = {"bun": "mi_bun", "esr": "mi_esr", "hemoglobin": "mi_hb", "potassium": "mi_k", "sodium": "mi_na", "wbc": "mi_wbc",
                "lymphocyte_pct": "mi_lymph", "neutrophil_pct": "mi_neut", "platelets": "mi_plt", "ef_tte": "mi_ef_tte"}
CAD_FLAGS = {"diabetes": "diabetes", "hypertension": "hypertension", "family_history_cad": "family_history_cad",
             "dyslipidemia": "hyperlipidemia", "chronic_renal_failure": "mi_crf", "cerebrovascular_accident": "mi_cva",
             "airway_disease": "mi_airway_disease", "thyroid_disease": "mi_thyroid_disease",
             "congestive_heart_failure": "mi_chf", "edema": "mi_edema", "weak_peripheral_pulse": "mi_weak_peripheral_pulse",
             "lung_rales": "mi_lung_rales", "systolic_murmur": "mi_systolic_murmur", "diastolic_murmur": "mi_diastolic_murmur",
             "exertional_chest_pain": "exertional_symptoms", "low_threshold_angina": "mi_lowth_angina", "dyspnea": "mi_dyspnea",
             "ecg_q_wave": "mi_q_wave", "ecg_st_elevation": "mi_st_elevation", "ecg_st_depression": "mi_st_depression",
             "ecg_t_inversion": "mi_t_inversion", "ecg_lvh": "mi_lvh", "ecg_poor_r_progression": "mi_poor_r_progression"}
VHD = {"N": "none", "mild": "mild", "Moderate": "moderate", "Severe": "severe"}
MAX_OPTIONAL_MISSING = 5


def cad_request(row: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Analytic row -> CadInputs fields (app contract), or (None, reason) if not scorable."""
    f: Dict[str, Any] = {}
    missing = [k for k, col in CAD_REQUIRED.items() if _v(row, col) is None]
    if _v(row, "gender") not in ("Male", "Female"):
        missing.append("sex")
    if missing:
        return None, "missing required input: " + ", ".join(missing)
    for k, col in CAD_REQUIRED.items():
        f[k] = _num(row, col)
    f["age"] = int(f["age"])
    f["sex"] = _v(row, "gender")
    f["bmi"] = _num(row, "bmi") or f["weight_kg"] / (f["height_cm"] / 100.0) ** 2
    for k, col in CAD_OPTIONAL.items():
        f[k] = _num(row, col)
    n_opt = sum(f[k] is None for k in CAD_OPTIONAL)
    if n_opt > MAX_OPTIONAL_MISSING:
        return None, f"{n_opt} of 10 optional labs/echo missing (limit {MAX_OPTIONAL_MISSING})"
    for k, col in CAD_FLAGS.items():
        f[k] = _flag(row, col)
    smoke = _v(row, "smoking_status")
    f["smoking_status"] = smoke if smoke in ("current", "former") else "never"   # training: unknown -> both 0
    f["chest_pain_type"] = _v(row, "chest_pain_type") or "unspecified"
    f["function_class"] = int(_v(row, "mi_function_class") or 0)
    f["region_rwma"] = int(_v(row, "mi_region_rwma") or 0)
    f["valvular_heart_disease"] = VHD.get(_v(row, "mi_vhd"), "none")
    return f, ""


# --------------------------------------------------------------------------- Risk (Cleveland)
_CP = {v: k for k, v in enc.CP.items()}
_RESTECG = {v: k for k, v in enc.RESTECG.items()}
_SLOPE = {v: k for k, v in enc.SLOPE.items()}
_THAL = {v: k for k, v in enc.THAL.items()}


def risk_request(row: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Analytic row -> RiskInputs fields (app contract). cl_* columns hold Cleveland codes."""
    need = {"age": "age", "resting_systolic_bp": "systolic_bp", "total_cholesterol": "total_cholesterol",
            "fasting_glucose": "fasting_glucose", "cl_cp": "cl_cp"}
    missing = [k for k, col in need.items() if _v(row, col) is None]
    if _v(row, "gender") not in ("Male", "Female"):
        missing.append("sex")
    if missing:
        return None, "missing required input: " + ", ".join(missing)
    code = lambda col: None if _v(row, col) is None else int(_v(row, col))  # noqa: E731
    exang = _v(row, "exercise_induced_angina")
    ca = code("cl_ca")
    f = {"age": int(_v(row, "age")), "sex": _v(row, "gender"), "chest_pain_type": _CP[code("cl_cp")],
         "resting_systolic_bp": _num(row, "systolic_bp"), "total_cholesterol": _num(row, "total_cholesterol"),
         "fasting_glucose": _num(row, "fasting_glucose"),
         "resting_ecg": _RESTECG.get(code("cl_restecg")),
         "max_heart_rate_achieved": _num(row, "max_heart_rate_achieved"),
         "exercise_induced_angina": None if exang is None else bool(exang),
         "st_depression_exercise": _num(row, "st_depression_exercise"),
         "st_slope": _SLOPE.get(code("cl_slope")),
         "fluoroscopy_vessels": None if ca in (None, 4) else ca,
         "thallium": _THAL.get(code("cl_thal"))}
    return f, ""


# --------------------------------------------------------------------------- Longitudinal (EHR vitals)
LONG_COLS = {"bp_systolic": "mi_bp_systolic", "bp_diastolic": "mi_bp_diastolic", "heart_rate": "mi_heart_rate",
             "respiratory_rate": "mi_respiratory_rate", "temperature": "mi_temperature",
             "oxygen_saturation": "mi_oxygen_saturation", "med_adherence": "mi_med_adherence",
             "symptom_severity": "mi_symptom_severity"}


def longitudinal_request(row: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    f = {k: _num(row, col) for k, col in LONG_COLS.items()}
    vitals = [k for k in list(LONG_COLS)[:6] if f[k] is not None]
    if len(vitals) < 4:
        return None, f"only {len(vitals)} of 6 vitals (need 4)"
    return f, ""


# --------------------------------------------------------------------------- encode
REQUEST: Dict[str, Callable] = {"cad": cad_request, "risk": risk_request, "longitudinal": longitudinal_request}
ENCODE: Dict[str, Callable] = {
    "cad": lambda f: enc.zalizadeh_vector(f),
    "risk": lambda f: enc.cleveland_vector(enc.cleveland_codes(f)),
    "longitudinal": lambda f: enc.ehr_vitals_vector(f),
}


def encode_row(cohort: str, row: Dict[str, Any]) -> Dict[str, Any]:
    f, reason = REQUEST[cohort](row)
    out = {"person_id": row.get("person_id"), "scorable": False, "reason": reason, "vector": None, "imputed": []}
    if f is None:
        return out
    try:
        vec, imputed = ENCODE[cohort](f)
    except enc.PreprocessingUnavailable as e:
        out["reason"] = f"model not scorable yet: {str(e)[:120]}"
        return out
    except enc.PreprocessingError as e:
        out["reason"] = str(e)
        return out
    if any(not math.isfinite(x) for x in vec):
        out["reason"] = "non-finite value after encoding"
        return out
    out.update(scorable=True, vector=vec, imputed=imputed)
    return out


def encode_cohort(cohort: str, analytic: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: person_id, scorable, reason, vector (list of floats), imputed (list of columns)."""
    rows: List[Dict[str, Any]] = [encode_row(cohort, r) for r in analytic.to_dict("records")]
    df = pd.DataFrame(rows, columns=["person_id", "scorable", "reason", "vector", "imputed"])
    df.attrs["encoders_fingerprint"] = ENCODERS_FINGERPRINT
    return df
