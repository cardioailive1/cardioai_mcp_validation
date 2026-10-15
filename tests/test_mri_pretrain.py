"""Self-supervised Mayo-only encoder pretraining (training period only) and starting training from it."""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
torch = pytest.importorskip("torch")

import synthetic_cmr as S  # noqa: E402
from mri_training.dataset import load_config  # noqa: E402
from mri_training.pretrain import ENCODER_PREFIXES, nt_xent, pretrain, view  # noqa: E402
from mri_training.runtime.model import build_model  # noqa: E402
from mri_training.train import Example, load_init, train  # noqa: E402


def _cfg():
    cfg = copy.deepcopy(load_config())
    cfg["arch"]["width"] = 0.25
    cfg["pretrain"].update(epochs=3, batch_size=16)
    return cfg


def _loader(r, _c):
    lge, cine, _ = S.study(seed=int(r.seed), n_slices=5, phases=4)
    return Example(lge=lge, cine=cine)


def _manifest(n=4, date="2018-06-01"):
    return pd.DataFrame([dict(study_id=f"s{i}", person_id=i, split="train", seed=i, study_date=date) for i in range(n)])


def test_mayo_only_defaults():
    cfg = load_config()
    assert cfg["use_public_datasets"] is False
    assert cfg["splits"]["tune_start"] == "2020-01-01" and cfg["splits"]["test_start"] == "2021-01-01"   # as the ECG model
    with pytest.raises(ValueError, match="Mayo-only"):
        train(pd.DataFrame([{"source": "emidec", "split": "train", "image_uri": "x", "label_uri": "y"}]), cfg, "/tmp/x")


def test_views_and_loss():
    rng = np.random.default_rng(0)
    img = np.random.default_rng(1).random((192, 192)).astype(np.float32)
    a, b = view(img, rng, _cfg()["pretrain"]["augment"]), view(img, rng, _cfg()["pretrain"]["augment"])
    assert a.shape == (192, 192) and 0 <= a.min() and a.max() <= 1 and not np.allclose(a, b)
    z = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
    assert nt_xent(z, z, 0.2) < nt_xent(z, torch.nn.functional.normalize(torch.randn(8, 16), dim=1), 0.2)   # matched views score lower


def test_pretrain_writes_encoder_and_training_starts_from_it(tmp_path):
    cfg = _cfg()
    card = pretrain(_manifest(), cfg, str(tmp_path), loader=_loader, device="cpu", log=lambda s: None)
    assert card["uses_outside_data"] is False and card["data"]["studies"] == 4 and card["data"]["cine_slices"] > 0
    assert len(card["history"]) == 3 and all(np.isfinite(h["loss"]) for h in card["history"])
    enc = torch.load(tmp_path / "encoder.pt", weights_only=True)
    assert enc and all(k.startswith(ENCODER_PREFIXES) for k in enc)
    m = build_model({**cfg["arch"], "study_heads": cfg["study_heads"]})
    info = load_init(m, str(tmp_path), lambda s: None)
    assert info["scope"] == "encoder" and info["data"]["before"] == "2020-01-01"
    st = m.state_dict()
    assert all(torch.equal(st[k], v) for k, v in enc.items())                          # encoder loaded exactly
    m2 = build_model({**cfg["arch"], "width": 0.5, "study_heads": cfg["study_heads"]})
    with pytest.raises(RuntimeError):
        load_init(m2, str(tmp_path), lambda s: None)                                   # other architecture refused


def test_pretraining_refuses_tuning_or_test_period_studies(tmp_path):
    with pytest.raises(AssertionError, match="leak"):
        pretrain(_manifest(date="2020-06-01"), _cfg(), str(tmp_path), loader=_loader, device="cpu", log=lambda s: None)
