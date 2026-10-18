"""Vertex AI serving container for the CardioAI echo wall-motion model (MobileNetV3-WM).

Runs the SAME runtime/ code as training and the validation study (preprocess, model, segments, pipeline).
At startup it refuses to serve unless:
  * runtime.fingerprint() == model_card.json "runtime_fingerprint"
  * sha256(weights.pt)     == model_card.json "weights_sha256"
  * weights load with strict=True
  * thresholds.json covers exactly the model card's heads

Echo cines are larger than Vertex AI's online request limit (1.5 MB) unless pre-cropped, so clips are normally
passed by reference:
  {"instances": [{"clips": [{"uri": "gs://<bucket>/<study>/a4c.dcm", "view": "A4C"},
                            {"uri": "gs://<bucket>/<study>/a2c.dcm", "view": "A2C"},
                            {"uri": "gs://<bucket>/<study>/a3c.dcm"}]}]}            view optional: the model reads it
  {"instances": [{"study_uri": "gs://<bucket>/<path>.zip"}]}                      every 2-D cine in the zip, views read by the model
or inline for small clips: {"clips": [{"video_b64": "<base64 DICOM / AVI / MP4>", "view": "A4C"}]}.
URIs must start with one of ALLOWED_STUDY_PREFIXES (comma-separated env var); no prefix set = references refused.

Response per instance: findings (segments 1-17 with scores, territories, WMSI, LVEF estimate, views used),
probabilities, flags (locked thresholds), red_flags (locked rules), pattern, notes, plus runtime fingerprint, weights
hash, model name, preprocess version; or {"error": ..., "error_type": "input" | "processing"}.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import sys
import tempfile
import zipfile
import io as _io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))       # repo layout: <root>/echo_wm_training/serve/predictor.py
sys.path.insert(0, "/app")                       # container layout: /app/echo_wm_training/...

from echo_wm_training import runtime  # noqa: E402
from echo_wm_training.io import read_zip_clips  # noqa: E402
from echo_wm_training.runtime.model import load_model  # noqa: E402
from echo_wm_training.runtime.pipeline import analyse, interpret  # noqa: E402
from echo_wm_training.runtime.preprocess import PREPROCESS_VERSION, EchoInputError, Video, decode  # noqa: E402

log = logging.getLogger("echo-wm-predictor")
BUNDLE_FILES = ("weights.pt", "thresholds.json", "model_card.json")
MAX_BYTES = int(os.environ.get("MAX_STUDY_BYTES", 2_000_000_000))
VIEWS_OK = {"A4C", "A2C", "A3C"}


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
    if not uri.startswith("gs://"):
        return Path(uri)
    dst = Path(tempfile.mkdtemp(prefix="echo_wm_bundle_"))
    for name in BUNDLE_FILES:
        try:
            _gcs_download(f"{uri.rstrip('/')}/{name}", dst / name)
        except FileNotFoundError as e:
            raise BundleError(f"missing {name} under {uri}") from e
    return dst


class EchoWMPredictor:
    def __init__(self, bundle_dir: Path, device: Optional[str] = None, allowed_prefixes: Optional[List[str]] = None):
        bundle_dir = Path(bundle_dir)
        for name in BUNDLE_FILES:
            if not (bundle_dir / name).exists():
                raise BundleError(f"bundle is missing {name}")
        self.card = json.loads((bundle_dir / "model_card.json").read_text())
        self.fingerprint = runtime.fingerprint()
        if self.card.get("runtime_fingerprint") != self.fingerprint:
            raise BundleError(f"runtime code {self.fingerprint[:12]} does not match the model card "
                              f"{str(self.card.get('runtime_fingerprint'))[:12]}: preprocessing, architecture or segment "
                              "model differs from the validated model")
        self.weights_sha256 = hashlib.sha256((bundle_dir / "weights.pt").read_bytes()).hexdigest()
        if self.card.get("weights_sha256") != self.weights_sha256:
            raise BundleError("weights.pt does not match the model card hash")
        self.heads: List[str] = list(self.card["heads"])
        self.thresholds: Dict[str, float] = json.loads((bundle_dir / "thresholds.json").read_text())
        if set(self.thresholds) != set(self.heads):
            raise BundleError("thresholds.json heads differ from the model card")
        self.rules: Dict[str, Dict] = self.card.get("red_flags") or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(str(bundle_dir / "weights.pt"), self.card["arch"], self.device)
        env = os.environ.get("ALLOWED_STUDY_PREFIXES", "")
        self.allowed = [p for p in (allowed_prefixes if allowed_prefixes is not None else env.split(",")) if p]
        log.info("echo wall-motion model ready: %s, runtime %s, weights %s", self.card.get("model_name"),
                 self.fingerprint[:12], self.weights_sha256[:12])

    def _read(self, uri: str) -> bytes:
        if not any(uri.startswith(p) for p in self.allowed):
            raise EchoInputError("uri is not under an allowed prefix (ALLOWED_STUDY_PREFIXES)")
        if uri.startswith("gs://"):
            with tempfile.TemporaryDirectory() as d:
                data = _gcs_download(uri, Path(d) / "blob").read_bytes()
        else:
            data = Path(uri).read_bytes()
        if len(data) > MAX_BYTES:
            raise EchoInputError(f"file is larger than {MAX_BYTES} bytes")
        return data

    def _clips(self, inst: Dict[str, Any]) -> List[Tuple[Video, Optional[str]]]:
        if "study_uri" in inst:
            data = self._read(str(inst["study_uri"]))
            if not zipfile.is_zipfile(_io.BytesIO(data)):
                raise EchoInputError("study_uri must be a zip of the study's DICOM files")
            return [(v, None) for v in read_zip_clips(data)]
        if not isinstance(inst.get("clips"), list) or not inst["clips"]:
            raise EchoInputError("instance needs 'clips' (list of {uri | video_b64, view}) or 'study_uri'")
        out = []
        for c in inst["clips"]:
            view = c.get("view")
            if view is not None and view not in VIEWS_OK:
                raise EchoInputError(f"view must be one of {sorted(VIEWS_OK)} (or omitted), got {view!r}")
            if "video_b64" in c:
                try:
                    data = base64.b64decode(c["video_b64"], validate=True)
                except (binascii.Error, ValueError, TypeError) as e:
                    raise EchoInputError(f"video_b64 is not valid base64: {e}") from e
            elif "uri" in c:
                data = self._read(str(c["uri"]))
            else:
                raise EchoInputError("each clip needs 'uri' or 'video_b64'")
            out.append((decode(data), view))
        return out

    def _one(self, inst: Dict[str, Any]) -> Dict[str, Any]:
        try:
            a = analyse(self.model, self._clips(inst), self.device)
        except (EchoInputError, FileNotFoundError, zipfile.BadZipFile) as e:
            return {"error": str(e), "error_type": "input"}
        verdict = interpret(a, self.thresholds, self.rules)
        return {"findings": verdict["findings"], "probabilities": a["probabilities"], "flags": verdict["flags"],
                "red_flags": verdict["red_flags"], "pattern": verdict["pattern"], "notes": a["notes"],
                "thresholds": self.thresholds, "runtime_fingerprint": self.fingerprint, "weights_sha256": self.weights_sha256,
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


def create_app(predictor: EchoWMPredictor):
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
    app = create_app(EchoWMPredictor(fetch_bundle(uri)))
    app.run(host="0.0.0.0", port=int(os.environ.get("AIP_HTTP_PORT", 8080)))
