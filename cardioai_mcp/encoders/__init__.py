"""Exact training-time preprocessing for the tabular agents.

Each JSON file in this folder holds the parameters the deployed model was trained
with (column order, scaler mean/scale, one-hot categories) and where they came
from.  The functions here turn harmonised contract inputs into the vector the
model was trained on.  They are pure Python so the API image needs no
scikit-learn; tests check them against the original sklearn pipelines.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).parent


def fingerprint(folder: Optional[Path] = None) -> str:
    """SHA-256 over this package's code and parameter files (name + bytes, sorted).

    This file is kept byte-identical in two places: the app
    (app/validated_agents/preprocessing) and the validation toolkit
    (cardioai_mcp/encoders).  registry.yaml records the fingerprint the Mayo
    validation ran with; the app's tests fail if its copy no longer matches.
    """
    folder = Path(folder or HERE)
    h = hashlib.sha256()
    for f in sorted(folder.iterdir(), key=lambda x: x.name):
        if f.is_file() and (f.name == "__init__.py" or f.suffix == ".json"):
            h.update(f.name.encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


class PreprocessingError(ValueError):
    """Input cannot be encoded the way the model was trained (-> HTTP 422)."""


class PreprocessingUnavailable(RuntimeError):
    """The parameters needed to encode inputs are not known yet (-> HTTP 503)."""


@lru_cache(maxsize=None)
def load(name: str) -> Dict[str, Any]:
    return json.loads((HERE / name).read_text())


def _scale(values: List[float], mean: List[float], scale: List[float]) -> List[float]:
    return [(v - m) / (s if s else 1.0) for v, m, s in zip(values, mean, scale)]


# --------------------------------------------------------------------------- CAD (Z-Alizadeh Sani)
def zalizadeh_raw(f: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Contract fields -> the 59 training columns (before scaling). None = missing (imputed to the training mean)."""
    b = lambda k: 1.0 if f.get(k) else 0.0  # noqa: E731
    cp = f.get("chest_pain_type")
    smoke = f.get("smoking_status")
    rwma = int(f.get("region_rwma") or 0)
    vhd = f.get("valvular_heart_disease", "none")
    row = {
        "Age": f["age"], "Weight": f["weight_kg"], "Length": f["height_cm"], "Sex": 1.0 if f["sex"] == "Male" else 0.0,
        "BMI": f["bmi"], "DM": b("diabetes"), "HTN": b("hypertension"),
        "Current Smoker": 1.0 if smoke == "current" else 0.0, "EX-Smoker": 1.0 if smoke == "former" else 0.0,
        "FH": b("family_history_cad"), "Obesity": 1.0 if f["bmi"] >= 25 else 0.0,
        "CRF": b("chronic_renal_failure"), "CVA": b("cerebrovascular_accident"), "Airway disease": b("airway_disease"),
        "Thyroid Disease": b("thyroid_disease"), "CHF": b("congestive_heart_failure"), "DLP": b("dyslipidemia"),
        "BP": f["systolic_bp"], "PR": f["pulse_rate"], "Edema": b("edema"),
        "Weak Peripheral Pulse": b("weak_peripheral_pulse"), "Lung rales": b("lung_rales"),
        "Systolic Murmur": b("systolic_murmur"), "Diastolic Murmur": b("diastolic_murmur"),
        "Typical Chest Pain": 1.0 if cp == "typical" else 0.0, "Dyspnea": b("dyspnea"),
        "Function Class": float(f["function_class"]),
        "Atypical": 1.0 if cp == "atypical" else 0.0, "Nonanginal": 1.0 if cp == "non_anginal" else 0.0,
        "Exertional CP": b("exertional_chest_pain"), "LowTH Ang": b("low_threshold_angina"),
        "Q Wave": b("ecg_q_wave"), "St Elevation": b("ecg_st_elevation"), "St Depression": b("ecg_st_depression"),
        "Tinversion": b("ecg_t_inversion"), "LVH": b("ecg_lvh"), "Poor R Progression": b("ecg_poor_r_progression"),
        "FBS": f["fasting_glucose"], "CR": f["creatinine"], "TG": f["triglycerides"], "LDL": f["ldl_cholesterol"],
        "HDL": f["hdl_cholesterol"], "BUN": f.get("bun"), "ESR": f.get("esr"), "HB": f.get("hemoglobin"),
        "K": f.get("potassium"), "Na": f.get("sodium"), "WBC": f.get("wbc"), "Lymph": f.get("lymphocyte_pct"),
        "Neut": f.get("neutrophil_pct"), "PLT": f.get("platelets"), "EF-TTE": f.get("ef_tte"),
        "Region RWMA_1": 1.0 if rwma == 1 else 0.0, "Region RWMA_2": 1.0 if rwma == 2 else 0.0,
        "Region RWMA_3": 1.0 if rwma == 3 else 0.0, "Region RWMA_4": 1.0 if rwma == 4 else 0.0,
        "VHD_N": 1.0 if vhd == "none" else 0.0, "VHD_Severe": 1.0 if vhd == "severe" else 0.0,
        "VHD_mild": 1.0 if vhd == "mild" else 0.0,
    }
    missing = set(row) - set(params["columns"]) | set(params["columns"]) - set(row)
    if missing:
        raise PreprocessingError(f"CAD column mismatch with the locked preprocessing: {sorted(missing)}")
    return row


def zalizadeh_vector(f: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> tuple[List[float], List[str]]:
    params = params or load("cad_zalizadeh_mlp.json")
    row = zalizadeh_raw(f, params)
    imputed = [c for c in params["columns"] if row[c] is None]
    values = [params["mean"][i] if row[c] is None else float(row[c]) for i, c in enumerate(params["columns"])]
    return _scale(values, params["mean"], params["scale"]), imputed


# --------------------------------------------------------------------------- Risk (UCI Cleveland)
CP = {"asymptomatic": 0, "atypical": 1, "non_anginal": 2, "typical": 3}
RESTECG = {"lvh": 0, "normal": 1, "st_t_abnormality": 2}
SLOPE = {"downsloping": 0, "flat": 1, "upsloping": 2}
THAL = {"fixed_defect": 1, "normal": 2, "reversible_defect": 3}


def cleveland_codes(f: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Clinical contract fields -> the 13 Cleveland columns, in the training file's coding.

    Missing values follow the pre-specified rule shared with the validation toolkit
    (config/cohorts/risk.yaml, model_inputs.missing_rule)."""
    if f["chest_pain_type"] not in CP:
        raise PreprocessingError("chest_pain_type must be typical, atypical, non_anginal or asymptomatic for the Cleveland model")
    exang = f.get("exercise_induced_angina")
    return {
        "age": f["age"], "sex": 1 if f["sex"] == "Male" else 0, "cp": CP[f["chest_pain_type"]],
        "trestbps": f["resting_systolic_bp"], "chol": f["total_cholesterol"], "fbs": 1 if f["fasting_glucose"] > 120 else 0,
        "restecg": RESTECG.get(f.get("resting_ecg")),                 # None -> unknown category (all-zero one-hot)
        "thalach": f.get("max_heart_rate_achieved"),                  # None -> training mean
        "exang": None if exang is None else int(bool(exang)),         # None -> unknown category
        "oldpeak": f.get("st_depression_exercise"),                   # None -> training mean
        "slope": SLOPE.get(f.get("st_slope")),                        # None -> unknown category
        "ca": 4 if f.get("fluoroscopy_vessels") is None else int(f["fluoroscopy_vessels"]),  # 4 = training 'missing' code
        "thal": THAL.get(f.get("thallium"), 0),                       # 0 = training 'missing' code
    }


def cleveland_vector(codes: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> tuple[List[float], List[str]]:
    """ColumnTransformer(StandardScaler(num), OneHotEncoder(handle_unknown='ignore')(cat)) — 30 values."""
    params = params or load("risk_cleveland_mlp.json")
    num, cat = params["numeric"], params["categorical"]
    imputed = [c for c in num["columns"] if codes.get(c) is None] + [c for c in cat["columns"] if codes.get(c) is None]
    vals = [m if codes.get(c) is None else float(codes[c]) for c, m in zip(num["columns"], num["mean"])]
    out = _scale(vals, num["mean"], num["scale"])
    for c, cats in zip(cat["columns"], cat["categories"]):
        v = codes.get(c)
        out += [1.0 if v is not None and int(v) == k else 0.0 for k in cats]
    if len(out) != len(params["feature_names_out"]):
        raise PreprocessingError("Cleveland vector length does not match the locked preprocessor")
    return out, imputed


# --------------------------------------------------------------------------- Longitudinal (EHR vitals)
def ehr_vitals_vector(f: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> tuple[List[float], List[str]]:
    params = params or load("longitudinal_ehr_vitals_mlp.json")
    norm = params.get("raw_unit_normalization")
    if not norm:
        raise PreprocessingUnavailable(
            "Longitudinal EHR vitals model: the raw-unit normalization of its training data is unknown "
            "(training file was pre-standardized). " + params.get("blocker", ""))
    imputed, z = [], []
    for c in params["columns"]:
        v = f.get(c)
        if v is None:
            imputed.append(c)
            z.append(0.0)  # training mean on the standardized scale
        else:
            z.append((float(v) - norm[c]["mean"]) / norm[c]["sd"])
    return _scale(z, params["mean"], params["scale"]), imputed
