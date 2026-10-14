"""Vertex AI serving container for the CardioAI cardiac MRI model (MobileNetV3-UNet).

Runs the SAME runtime/ code as training and the validation study (preprocess, model, quantify,
pipeline). At startup it refuses to serve unless:
  * runtime.fingerprint() == model_card.json "runtime_fingerprint"
  * sha256(weights.pt)     == model_card.json "weights_sha256"
  * weights load with strict=True
  * thresholds.json covers exactly the model card's study heads

A CMR study is far larger than Vertex AI's online request limit (1.5 MB), so a study is normally
passed by reference:
  {"instances": [{"study_uri": "gs://<bucket>/<path>.zip"}]}       zip of the study's DICOM files
or, for small pre-selected series (tests, LGE-only uploads), inline:
  {"instances": [{"study_zip_b64": "<base64 zip>"}]}
study_uri must start with one of ALLOWED_STUDY_PREFIXES (comma-separated env var); no prefix set
= references refused.

Response per instance: findings (infarct size, MVO, LV mass, transmurality per AHA segment, scar
depth, LV volumes, LVEF), probabilities, flags (locked thresholds), red_flags (locked clinical
rules), notes (series used, what could not be measured), plus fingerprint, weights hash, model
name, preprocess version; or {"error": ..., "error_type": "input" | "processing"}.
"""
from __future__ import annotations

import base64
import io as _io
import binascii
import hashlib
import json
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))       # repo layout: <root>/mri_training/serve/predictor.py
sys.path.insert(0, "/app")                       # container layout: /app/mri_training/...

from mri_training import runtime  # noqa: E402
from mri_training.io import read_study  # noqa: E402
from mri_training.runtime.model import load_model  # noqa: E402
from mri_training.runtime.pipeline import analyse, interpret  # noqa: E402
from mri_training.runtime.preprocess import PREPROCESS_VERSION, MRIInputError  # noqa: E402

log = logging.getLogger("mri-predictor")
BUNDLE_FILES = ("weights.pt", "thresholds.json", "model_card.json")
MAX_STUDY_BYTES = int(os.environ.get("MAX_STUDY_BYTES", 1_000_000_000))


class BundleError(RuntimeError):
    pass


def _gcs_download(uri: str, dst: Path) -> Path:
    from google.cloud import storage
    bucket, _, name = uri[5:].partition("/")
    blob = storage.Client().bucket(bucket).blob(name)
    if not blob.exists():
        raise FileNotFoundError(uri)
    blob.download_to_filename(str(dst))
    return dst


def fetch_bundle(uri: str) -> Path:
    """Local folder, or gs://bucket/prefix (Vertex AIP_STORAGE_URI) downloaded to a temp folder."""
    if not uri.startswith("gs://"):
        return Path(uri)
    dst = Path(tempfile.mkdtemp(prefix="mri_bundle_"))
    for name in BUNDLE_FILES:
        try:
            _gcs_download(f"{uri.rstrip('/')}/{name}", dst / name)
        except FileNotFoundError as e:
            raise BundleError(f"missing {name} under {uri}") from e
    return dst


class MRIPredictor:
    def __init__(self, bundle_dir: Path, device: Optional[str] = None, allowed_prefixes: Optional[List[str]] = None):
        bundle_dir = Path(bundle_dir)
        for name in BUNDLE_FILES:
            if not (bundle_dir / name).exists():
                raise BundleError(f"bundle is missing {name}")
        self.card = json.loads((bundle_dir / "model_card.json").read_text())
        self.fingerprint = runtime.fingerprint()
        if self.card.get("runtime_fingerprint") != self.fingerprint:
            raise BundleError(f"runtime code {self.fingerprint[:12]} does not match the model card "
                              f"{str(self.card.get('runtime_fingerprint'))[:12]}: preprocessing, architecture or "
                              "quantification differs from the validated model")
        self.weights_sha256 = hashlib.sha256((bundle_dir / "weights.pt").read_bytes()).hexdigest()
        if self.card.get("weights_sha256") != self.weights_sha256:
            raise BundleError("weights.pt does not match the model card hash")
        self.heads: List[str] = list(self.card["study_heads"])
        self.thresholds: Dict[str, float] = json.loads((bundle_dir / "thresholds.json").read_text())
        if set(self.thresholds) != set(self.heads):
            raise BundleError("thresholds.json heads differ from the model card")
        self.rules: Dict[str, Dict] = self.card.get("red_flags") or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(str(bundle_dir / "weights.pt"), {**self.card["arch"], "study_heads": self.heads}, self.device)
        env = os.environ.get("ALLOWED_STUDY_PREFIXES", "")
        self.allowed = [p for p in (allowed_prefixes if allowed_prefixes is not None else env.split(",")) if p]
        log.info("MRI model ready: %s, runtime %s, weights %s", self.card.get("model_name"), self.fingerprint[:12], self.weights_sha256[:12])

    def _study_bytes(self, inst: Dict[str, Any]) -> bytes:
        if "study_zip_b64" in inst:
            try:
                data = base64.b64decode(inst["study_zip_b64"], validate=True)
            except (binascii.Error, ValueError, TypeError) as e:
                raise MRIInputError(f"study_zip_b64 is not valid base64: {e}") from e
        elif "study_uri" in inst:
            uri = str(inst["study_uri"])
            if not any(uri.startswith(p) for p in self.allowed):
                raise MRIInputError("study_uri is not under an allowed prefix (ALLOWED_STUDY_PREFIXES)")
            if uri.startswith("gs://"):
                with tempfile.TemporaryDirectory() as d:
                    data = _gcs_download(uri, Path(d) / "study.zip").read_bytes()
            else:
                data = Path(uri).read_bytes()
        else:
            raise MRIInputError("instance needs 'study_uri' (gs:// zip of the study's DICOM) or 'study_zip_b64'")
        if len(data) > MAX_STUDY_BYTES:
            raise MRIInputError(f"study is larger than {MAX_STUDY_BYTES} bytes")
        if not zipfile.is_zipfile(_io.BytesIO(data)):
            raise MRIInputError("study must be a zip of DICOM files")
        return data

    def _one(self, inst: Dict[str, Any]) -> Dict[str, Any]:
        try:
            lge, cine = read_study(self._study_bytes(inst))
            a = analyse(self.model, lge, cine, self.device)
        except (MRIInputError, FileNotFoundError, zipfile.BadZipFile) as e:
            return {"error": str(e), "error_type": "input"}
        verdict = interpret(a, self.thresholds, self.rules)
        f = dict(a["findings"])
        f["segments"] = {str(k): v for k, v in f["segments"].items()}
        return {"findings": f, "probabilities": a["probabilities"], "flags": verdict["flags"],
                "red_flags": verdict["red_flags"], "notes": a["notes"], "thresholds": self.thresholds,
                "runtime_fingerprint": self.fingerprint, "weights_sha256": self.weights_sha256,
                "model_name": self.card.get("model_name"), "preprocess_version": PREPROCESS_VERSION}

    def predict(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for inst in instances:
            try:
                out.append(self._one(inst))
            except Exception as e:  # noqa: BLE001
                log.exception("prediction failed")
                out.append({"error": str(e), "error_type": "processing"})
        return out


def create_app(predictor: MRIPredictor):
    from flask import Flask, jsonify, request
    app = Flask(__name__)

    @app.get(os.environ.get("AIP_HEALTH_ROUTE", "/health"))
    def health():
        return jsonify({"status": "healthy", "runtime_fingerprint": predictor.fingerprint,
                        "weights_sha256": predictor.weights_sha256}), 200

    @app.post(os.environ.get("AIP_PREDICT_ROUTE", "/predict"))
    def predict():
        body = request.get_json(silent=True) or {}
        if "instances" not in body:
            return jsonify({"error": "missing 'instances'"}), 400
        return jsonify({"predictions": predictor.predict(body["instances"])}), 200

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uri = os.environ.get("AIP_STORAGE_URI") or os.environ.get("MODEL_DIR", "/app/model")
    app = create_app(MRIPredictor(fetch_bundle(uri)))
    app.run(host="0.0.0.0", port=int(os.environ.get("AIP_HTTP_PORT", 8080)))
