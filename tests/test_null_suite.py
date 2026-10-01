import os

import numpy as np
import pytest

from cviaf.lab.null_suite import (REAL_BACKBONE_GUIDANCE, fft_energy, pair_metric,
                                 pvalues, require_synthetic_corpus, run)


def test_identical_scores_are_chance_and_not_rejected_at_five_percent():
    x = np.arange(20, dtype=float)
    m = pair_metric(x, x)
    assert m['auroc'] == .5
    assert m['tpr_at_5fpr'] == .05


def test_fft_only_depends_on_images():
    images = np.ones((2, 64, 64, 3), dtype=float)
    stamped = images.copy()
    stamped[:, :8, :8, :] = 0
    assert np.all(fft_energy(stamped) > fft_energy(images))


# --------------------------------------------------------------------------- #
# the corpus guard: a real-backbone corpus must be refused, not crashed on
# --------------------------------------------------------------------------- #

def _synthetic_manifest(model_id="m0"):
    return {"model_id": model_id, "spec": {
        "model_id": model_id, "kind": "clean",
        "scene": {"terrain": "desert", "season": "summer", "illumination": 1.0,
                  "gamma": 1.0, "sensor_noise": 0.02, "sensor_blur": 0,
                  "objects_per_image": [1, 3], "seed": 7},
        "attack": {"kind": "oga", "rate": 0.25, "trigger_size": 3,
                   "target_class": 1, "seed": 5},
        "detector": {"img_size": 64, "c1": 16, "c2": 64, "hidden": 48,
                     "n_classes": 3, "epochs": 2, "lr": 0.02, "batch": 8,
                     "pos_weight": 1.0, "ignore_radius": 1, "weight_decay": 0.0,
                     "seed": 5},
        "n_train": 32, "n_eval": 24, "n_cal": 32,
        "contributors": ["lab_alpha"], "contributor_mode": "round_robin"}}


def _real_backbone_manifest(model_id="realcifar_clean_s0"):
    return {"model_id": model_id, "spec": {
        "kind": "clean", "model_id": model_id,
        "backbone": {"backbone": "resnet18_stem1", "pretrained": True, "seed": 0},
        "dataset": {"source": "cifar-10", "seed": 1000},
        "detector": {"img_size": 64, "seed": 0}}}


def test_synthetic_registry_passes_the_guard():
    require_synthetic_corpus([{"dir": "x", "manifest": _synthetic_manifest()}])


def test_real_backbone_registry_is_refused_with_guidance():
    """The bug: build_splits(None) -> AttributeError: 'NoneType' has no n_train."""
    registry = [{"dir": "a", "manifest": _real_backbone_manifest("rb0")},
                {"dir": "b", "manifest": _synthetic_manifest("m1")}]
    with pytest.raises(ValueError) as exc:
        require_synthetic_corpus(registry)
    message = str(exc.value)
    assert "rb0" in message and "no synthetic train spec" in message
    assert "battery_model_attacks" in message      # names the substitute instrument
    assert "fpr_tpr" in message
    assert REAL_BACKBONE_GUIDANCE in message


def test_run_on_the_real_corpus_raises_valueerror_not_attributeerror():
    """End-to-end: the sweep that produced runs/real_cifar/null_suite.log."""
    corpus = os.path.join(os.path.dirname(__file__), "..", "runs", "real_cifar")
    if not os.path.isdir(corpus):
        pytest.skip("runs/real_cifar not present")
    with pytest.raises(ValueError, match="no synthetic train spec"):
        run(corpus, seeds=(0,), attacks=("oga",), n_eval=1, n_cal=20)


def test_stamped_null_calibration_fuses_with_and_without_ftc():
    cells = ('clean_unstamped', 'clean_stamped', 'backdoored_unstamped',
             'backdoored_stamped', 'peer_clean_stamped')
    data = {c: {s: np.arange(20, dtype=float) for s in ('ctc', 'refdiv', 'ftc', 'fft')}
            for c in cells}
    p = pvalues(data, data)
    assert len(p['clean_stamped']['with_ftc']) == 20
    assert np.all(p['clean_stamped']['with_ftc'] >= p['clean_stamped']['without_ftc'])
