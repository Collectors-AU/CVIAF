"""Task 3: the real backbone keeps the synthetic detector's interface and format.

The whole module skips without torch, so the numpy-only suite is unaffected; run it with
the torch interpreter (see TASK3_NOTES.md) to actually exercise the backbone.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cviaf.lab.detector import DetectorConfig                      # noqa: E402
from cviaf.lab.real_backbone import (RealBackboneConfig, RealBackboneDetector,   # noqa: E402
                                     export_parity, is_real_backbone_artifact)
from cviaf.lab.train import ModelArtifact                          # noqa: E402

SMOKE_DIR = "runs/real_cifar_smoke"
SYNTHETIC_MANIFEST = "runs/clean_null/clean_none_fixed_s100/manifest.json"


def make(seed: int = 0, pretrained: bool = False) -> RealBackboneDetector:
    rb = RealBackboneConfig(pretrained=pretrained, seed=seed)
    return RealBackboneDetector(cfg=rb.detector_config(), rb=rb)


def test_interface_matches_synthetic_and_is_deterministic():
    a, b = make(0), make(0)
    img = np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)
    feats = a.features(img)
    assert feats.shape == (a.cfg.grid, a.cfg.grid, 64), feats.shape
    assert a.cfg.grid == DetectorConfig().grid and a.cfg.cell == DetectorConfig().cell
    # same seed -> identical weights (determinism) and identical features
    assert a.digest() == b.digest()
    assert np.array_equal(feats, b.features(img))
    out = a.predict(img)
    assert set(out) == {"boxes", "scores", "labels", "cells", "obj", "cls"}
    assert out["obj"].shape == (a.cfg.grid, a.cfg.grid)


def test_save_load_roundtrip_preserves_identity(tmp_path):
    m = make(1)
    rng = np.random.default_rng(1)
    m.Wh = rng.normal(0, .1, m.Wh.shape).astype(np.float32)   # pretend it was trained
    m.feat_std = (rng.random(m.feat_std.shape).astype(np.float32) + 0.5)
    path = str(tmp_path / "weights.npz")
    digest = m.save(path)
    back = RealBackboneDetector.load(path)
    assert back.digest() == digest
    img = rng.random((64, 64, 3)).astype(np.float32)
    assert np.allclose(m.features(img), back.features(img), atol=1e-5)


def test_onnx_export_parity_gate(tmp_path):
    pytest.importorskip("onnxruntime")
    m = make(2)
    path = str(tmp_path / "features.onnx")
    m.export_onnx(path)
    imgs = np.random.default_rng(2).random((2, 64, 64, 3)).astype(np.float32)
    rec = export_parity(m, path, imgs)
    assert rec["pass"], rec
    assert rec["max_feature_delta"] <= rec["tolerance"]
    assert rec["channel_argmax_agreement"] == 1.0
    assert rec["digest_unchanged"] is True


def test_model_artifact_dispatches_to_real_backbone(tmp_path):
    m = make(3)
    d = tmp_path / "realcifar_clean_s3"
    d.mkdir()
    m.save(str(d / "weights.npz"))
    (d / "manifest.json").write_text(json.dumps({"model_id": "realcifar_clean_s3"}))
    assert is_real_backbone_artifact(str(d / "weights.npz")) is True
    art = ModelArtifact.load(str(d))
    assert isinstance(art.model, RealBackboneDetector)
    assert art.model.digest() == m.digest()


def test_real_manifest_has_the_synthetic_manifest_keys():
    """Cross-check after the smoke run: same top-level manifest contract."""
    if not os.path.isfile(SYNTHETIC_MANIFEST) or not os.path.isdir(SMOKE_DIR):
        pytest.skip("run scripts/train_real_backbone.py --smoke first")
    synthetic = set(json.load(open(SYNTHETIC_MANIFEST)))
    dirs = [os.path.join(SMOKE_DIR, d) for d in sorted(os.listdir(SMOKE_DIR))
            if os.path.isfile(os.path.join(SMOKE_DIR, d, "manifest.json"))]
    assert dirs, "smoke run produced no artifacts"
    for d in dirs:
        real = json.load(open(os.path.join(d, "manifest.json")))
        missing = synthetic - set(real)
        assert not missing, f"{d} manifest is missing synthetic keys: {sorted(missing)}"
        assert real["artifact"]["weights_digest"]
        assert real["dataset_digests"]["train"]
