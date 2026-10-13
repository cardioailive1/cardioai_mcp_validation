"""The encoders are the app's (app/validated_agents/preprocessing), byte for byte.

Golden vectors come from the original training pipelines (EHR_branch): the CAD
training script's encoding + scaler and the Cleveland preprocessor.joblib.
"""
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cardioai_mcp import encoders as enc  # noqa: E402

GOLDEN = json.loads((ROOT / "tests" / "fixtures" / "golden_vectors.json").read_text())


def test_encoders_match_the_locked_fingerprint():
    lock = yaml.safe_load((ROOT / "config" / "study.yaml").read_text())["model_lock"]["encoders_fingerprint"]
    assert enc.fingerprint() == lock, (
        "cardioai_mcp/encoders changed. It must stay byte-identical to the app's "
        "app/validated_agents/preprocessing; update both and the fingerprint in config/study.yaml "
        "and in the app's registry.yaml together (change control).")


@pytest.mark.parametrize("i", range(len(GOLDEN["cad"])))
def test_cad_vector_matches_training_pipeline(i):
    vec, imputed = enc.zalizadeh_vector(GOLDEN["cad"][i]["inputs"])
    assert vec == pytest.approx(GOLDEN["cad"][i]["expected"], abs=1e-9) and not imputed


@pytest.mark.parametrize("i", range(len(GOLDEN["risk"])))
def test_cleveland_vector_matches_preprocessor_joblib(i):
    vec, _ = enc.cleveland_vector(enc.cleveland_codes(GOLDEN["risk"][i]["inputs"]))
    assert vec == pytest.approx(GOLDEN["risk"][i]["expected"], abs=1e-9)


def test_longitudinal_refuses_until_normalization_known():
    with pytest.raises(enc.PreprocessingUnavailable):
        enc.ehr_vitals_vector({"bp_systolic": 120})


if os.environ.get("CARDIOAI_APP_REPO"):  # optional cross-repo check, e.g. CARDIOAI_APP_REPO=../cardioaimedconnect
    def test_same_bytes_as_app_copy():
        app = Path(os.environ["CARDIOAI_APP_REPO"]) / "app" / "validated_agents" / "preprocessing"
        assert enc.fingerprint(app) == enc.fingerprint()
