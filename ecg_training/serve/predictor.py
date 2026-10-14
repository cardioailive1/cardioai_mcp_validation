"""Vertex AI serving container for the CardioAI ECG model (MobileNetV3-1D).

Runs the SAME runtime/ code as training (preprocess + model). At startup it refuses to
serve unless:
  * runtime.fingerprint() == model_card.json "runtime_fingerprint"  (same preprocessing + architecture)
  * sha256(weights.pt)     == model_card.json "weights_sha256"        (the exported weights, unaltered)
  * weights load with strict=True                                      (no silent partial load)

Request  {"instances": [{"signal": [[...], ...] (n_leads x N), "fs": 500, "leads": ["I", ...], "units": "mV"}]}
Response {"predictions": [{"probabilities": {...}, "flags": {...}, "thresholds": {...},
          "runtime_fingerprint": ..., "weights_sha256": ..., "model_name": ..., "preprocess_version": ...}]}
         or {"error": ..., "error_type": "input" | "processing"} per instance.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))       # repo layout: <root>/ecg_training/serve/predictor.py
sys.path.insert(0, "/app")                       # container layout: /app/ecg_training/...

from ecg_training import runtime  # noqa: E402
from ecg_training.runtime.model import load_model  # noqa: E402
from ecg_training.runtime.preprocess import PREPROCESS_VERSION, ECGInputError, preprocess  # noqa: E402

log = logging.getLogger("ecg-predictor")
BUNDLE_FILES = ("weights.pt", "thresholds.json", "model_card.json")


class BundleError(RuntimeError):
    pass


def fetch_bundle(uri: str) -> Path:
    """Local folder, or gs://bucket/prefix (Vertex AIP_STORAGE_URI) downloaded to a temp folder."""
    if not uri.startswith("gs://"):
        return Path(uri)
    from google.cloud import storage
    bucket, _, prefix = uri[5:].partition("/")
    dst = Path(tempfile.mkdtemp(prefix="ecg_bundle_"))
    client = storage.Client()
    for name in BUNDLE_FILES:
        blob = client.bucket(bucket).blob(f"{prefix.rstrip('/')}/{name}" if prefix else name)
        if not blob.exists():
            raise BundleError(f"missing {name} under {uri}")
        blob.download_to_filename(str(dst / name))
    return dst


class ECGPredictor:
    def __init__(self, bundle_dir: Path, device: Optional[str] = None):
        bundle_dir = Path(bundle_dir)
        for name in BUNDLE_FILES:
            if not (bundle_dir / name).exists():
                raise BundleError(f"bundle is missing {name}")
        self.card = json.loads((bundle_dir / "model_card.json").read_text())
        self.fingerprint = runtime.fingerprint()
        if self.card.get("runtime_fingerprint") != self.fingerprint:
            raise BundleError(f"runtime code {self.fingerprint[:12]} does not match the model card "
                              f"{str(self.card.get('runtime_fingerprint'))[:12]}: preprocessing or architecture differs from training")
        self.weights_sha256 = hashlib.sha256((bundle_dir / "weights.pt").read_bytes()).hexdigest()
        if self.card.get("weights_sha256") != self.weights_sha256:
            raise BundleError("weights.pt does not match the model card hash")
        self.heads: List[str] = list(self.card["heads"])
        self.thresholds: Dict[str, float] = json.loads((bundle_dir / "thresholds.json").read_text())
        if set(self.thresholds) != set(self.heads):
            raise BundleError("thresholds.json heads differ from the model card")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(str(bundle_dir / "weights.pt"), {**self.card["arch"], "heads": self.heads}, self.device)
        log.info("ECG model ready: %s, runtime %s, weights %s", self.card.get("model_name"), self.fingerprint[:12], self.weights_sha256[:12])

    def _one(self, inst: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if "signal" not in inst or "fs" not in inst:
                raise ECGInputError("instance needs 'signal' (n_leads x N) and 'fs'")
            x = preprocess(np.asarray(inst["signal"], dtype=np.float64), float(inst["fs"]),
                           inst.get("leads"), inst.get("units", "mV"))
        except (ECGInputError, ValueError, TypeError) as e:
            return {"error": str(e), "error_type": "input"}
        with torch.no_grad():
            p = torch.sigmoid(self.model(torch.from_numpy(x)[None].to(self.device)))[0].cpu().numpy()
        probs = {h: float(v) for h, v in zip(self.heads, p)}
        return {"probabilities": probs, "flags": {h: probs[h] >= self.thresholds[h] for h in self.heads},
                "thresholds": self.thresholds, "runtime_fingerprint": self.fingerprint,
                "weights_sha256": self.weights_sha256, "model_name": self.card.get("model_name"),
                "preprocess_version": PREPROCESS_VERSION}

    def predict(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for inst in instances:
            try:
                out.append(self._one(inst))
            except Exception as e:  # noqa: BLE001
                log.exception("prediction failed")
                out.append({"error": str(e), "error_type": "processing"})
        return out


def create_app(predictor: ECGPredictor):
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
    app = create_app(ECGPredictor(fetch_bundle(uri)))
    app.run(host="0.0.0.0", port=int(os.environ.get("AIP_HTTP_PORT", 8080)))
