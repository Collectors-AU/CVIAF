"""Regression tests for three confirmed defects, one test per fix.

Each test is written as *the test that would have caught the bug*, so a future
refactor that reintroduces it fails here rather than in a downstream measurement.

  1. `build_feature_extractor` drew the random-init backbone from the unseeded
     GLOBAL torch RNG, so two constructions of the same declared seed produced
     different weights and different digests. (Caught in the Task 3 lane; fixed
     in 19f1039. This is the direct regression test that commit did not add.)
  2. `compare.py` carried its own stale attack-kind tuple, so weight-space tamper
     arms were scored as NEGATIVES and the model axis reported "n=0 attacked".
     (Fixed in 3267a1f.)
  3. `cviaf/model_integrity` drew trigger patterns and noise probes from the
     legacy global `np.random` stream with no seed, so the same model produced
     different findings on every run.
  4. The `cviaf demo` mock model seeded its weights from `hash(bytes)`, which
     Python salts per process (PYTHONHASHSEED), so two runs of the demo on the
     same inputs disagreed on every logit while every declared seed matched.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# 1. real-backbone feature extractor determinism
# --------------------------------------------------------------------------- #

def test_feature_extractor_is_deterministic_for_a_declared_seed():
    """Two constructions of the same declared seed must be bit-identical.

    The bug: `resnet18(weights=None)` initialises through the global torch RNG, so
    with `pretrained=False` the extractor differed on every construction. With
    `pretrained=True` the checkpoint masked it, which is why the lane's training
    and parity runs looked fine while this was broken.
    """
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from cviaf.lab.real_backbone import build_feature_extractor, RealBackboneConfig

    cfg = RealBackboneConfig(pretrained=False, seed=0)
    a = build_feature_extractor(cfg)
    # Interleave unrelated global-RNG traffic: a fix that only happens to work
    # because nothing else drew from the stream is not a fix.
    np.random.seed(1234)
    b = build_feature_extractor(cfg)
    assert a.state_dict().keys() == b.state_dict().keys()
    for name, tensor in a.state_dict().items():
        assert np.array_equal(tensor.detach().numpy(),
                              b.state_dict()[name].detach().numpy()), name


def test_real_backbone_digest_is_stable_across_constructions():
    """The end-to-end symptom: same-seed models must share one digest."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from cviaf.lab.real_backbone import RealBackboneConfig, RealBackboneDetector

    rb = RealBackboneConfig(pretrained=False, seed=3)
    a = RealBackboneDetector(cfg=rb.detector_config(), rb=rb)
    b = RealBackboneDetector(cfg=rb.detector_config(), rb=rb)
    assert a.digest() == b.digest()

    # And a different declared seed must actually change the model, or the test
    # above would pass on a constant.
    rb_other = RealBackboneConfig(pretrained=False, seed=4)
    c = RealBackboneDetector(cfg=rb_other.detector_config(), rb=rb_other)
    assert c.digest() != a.digest()


def test_seeding_the_extractor_does_not_disturb_the_head_stream():
    """Seeding must not silently re-seed the caller's numpy/torch stream.

    `make_head` seeds torch explicitly for its own draw, so head init must stay
    a pure function of its seed argument even after the extractor fix.
    """
    torch = pytest.importorskip("torch")
    from cviaf.lab.real_backbone import build_feature_extractor, RealBackboneConfig
    from scripts.train_real_backbone import make_head  # noqa: E402

    cfg = RealBackboneConfig(pretrained=False, seed=0)
    build_feature_extractor(cfg)
    h1 = make_head(64, 48, 3, seed=0)
    build_feature_extractor(cfg)
    h2 = make_head(64, 48, 3, seed=0)
    assert all(np.array_equal(p1.detach().numpy(), p2.detach().numpy())
               for p1, p2 in zip(h1.parameters(), h2.parameters()))


# --------------------------------------------------------------------------- #
# 2. compare.py ground truth for weight-space arms
# --------------------------------------------------------------------------- #

def test_weight_space_arms_are_positives_on_the_model_axis():
    """`weight_tamper` / `substitution` must count as attacked, not as negatives.

    The bug: compare.py defined its own tuple `("oga","oda","rma","gma")`, so a
    corpus of 16 weight-space arms reported `n_positive=0, n_negative=24` and the
    baseline TPR was `None` — a ground-truth error, not a detection result.
    """
    from cviaf.lab.compare import MODEL_ATTACK_KINDS
    from cviaf.lab.poison import MODEL_ATTACK_KINDS as CANONICAL

    for kind in CANONICAL:
        assert kind in MODEL_ATTACK_KINDS, kind
    # The trigger-based kinds must survive the union — a fix that swapped one
    # stale tuple for another would drop these.
    for kind in ("oga", "oda", "rma", "gma"):
        assert kind in MODEL_ATTACK_KINDS, kind


def test_weight_space_kinds_are_not_data_attacks():
    """A weight-space tamper carries no per-sample poison, so it must not be
    scored on the data axis (the confusion that made the first battery run
    report a data TPR of 0 for arms whose datasets are clean by construction)."""
    from cviaf.lab.compare import DATA_ATTACK_KINDS
    from cviaf.lab.poison import MODEL_ATTACK_KINDS

    for kind in MODEL_ATTACK_KINDS:
        assert kind not in DATA_ATTACK_KINDS, kind


def test_arm_manifest_reports_its_own_kind_and_divergence():
    """The corpus contract the battery depends on: an arm manifest declares a
    weight-space kind and a measured behaviour divergence. This is the contract a
    hand-built arm silently broke (by saving a non-real-backbone npz)."""
    from cviaf.lab.poison import MODEL_ATTACK_KINDS
    from cviaf.lab.real_backbone import is_real_backbone_artifact

    corpus = "runs/real_cifar"
    if not os.path.isdir(corpus):
        pytest.skip("real_cifar corpus not present")
    checked = 0
    for name in sorted(os.listdir(corpus)):
        mpath = os.path.join(corpus, name, "manifest.json")
        if not os.path.isfile(mpath):
            continue
        with open(mpath) as fh:
            m = json.load(fh)
        kind = m["ground_truth"]["kind"]
        if kind not in MODEL_ATTACK_KINDS:
            continue
        assert m["quality_flags"]["is_model_attack"] is True, name
        bd = m["metrics"]["behaviour_divergence"]
        assert "f1_relative_drop" in bd, name
        assert bd["f1_relative_drop"] >= 0.0, name
        assert m["ground_truth"]["kind"] == kind
        assert is_real_backbone_artifact(os.path.join(corpus, name, "weights.npz")), name
        checked += 1
    assert checked > 0, "no weight-space arms found to validate"


# --------------------------------------------------------------------------- #
# 3. model-integrity RNG reproducibility
# --------------------------------------------------------------------------- #

def test_model_integrity_assessment_is_reproducible():
    """The same model + same seed must give the same findings.

    The bug: the trigger loop and the noise probe drew from the global
    `np.random` stream, so two assessments of one model returned different
    trigger statistics — a published finding could not be re-derived.
    """
    from cviaf.model_integrity import ModelIntegrityAssessor

    rng = np.random.default_rng(7)
    shape = (3, 8, 8)

    def predict_fn(batch):
        # Deterministic pseudo-model: peaky logits so the checks have signal.
        flat = batch.reshape(len(batch), -1)
        return flat @ np.linspace(-1, 1, flat.shape[1]).astype(np.float32)

    clean = rng.random((4, *shape)).astype(np.float32)

    def run(seed):
        a = ModelIntegrityAssessor(access_level="white-box", seed=seed).assess(
            predict_fn=predict_fn, input_shape=shape, num_classes=3,
            clean_images=clean)
        return [(f.title, f.severity) for f in a["findings"]]

    # pollute the global legacy stream between the two runs; a fix that still
    # depends on it will diverge here.
    np.random.seed(99)
    first = run(0)
    np.random.seed(12345)
    second = run(0)
    assert first == second


def test_model_integrity_does_not_consume_the_global_rng():
    """The bug's signature: the assessment *advanced* the legacy global stream.

    Checking the finding lists only proves the outputs agree; checking that the
    global stream is untouched proves the module no longer reads it at all, which
    is what protects every other consumer of `np.random` in the same process.
    """
    from cviaf.model_integrity import ModelIntegrityAssessor

    rng = np.random.default_rng(11)
    shape = (3, 8, 8)

    def predict_fn(batch):
        flat = batch.reshape(len(batch), -1)
        return flat @ np.linspace(-1, 1, flat.shape[1]).astype(np.float32)

    clean = rng.random((4, *shape)).astype(np.float32)

    np.random.seed(4321)
    before = np.random.get_state()[1].copy()
    ModelIntegrityAssessor(access_level="white-box", seed=0).assess(
        predict_fn=predict_fn, input_shape=shape, num_classes=3, clean_images=clean)
    after = np.random.get_state()[1].copy()
    assert np.array_equal(before, after), (
        "assessment advanced the global legacy RNG stream; some check is still "
        "drawing from np.random instead of its own seeded generator")


def test_model_integrity_source_has_no_global_rng_calls():
    """Static guard: a new check must not reintroduce `np.random.*` here."""
    import re
    path = os.path.join(os.path.dirname(__file__), "..", "cviaf", "model_integrity",
                        "__init__.py")
    with open(os.path.abspath(path), encoding="utf-8") as fh:
        src = fh.read()
    offenders = [ln for ln, line in enumerate(src.splitlines(), 1)
                 if re.search(r"\bnp\.random\.(?!default_rng|get_state|seed)\w+", line)]
    assert offenders == [], f"global legacy RNG calls at lines {offenders}"


# --------------------------------------------------------------------------- #
# 4. the demo mock model's seed must not come from hash()
# --------------------------------------------------------------------------- #

_HASH_SEED_PROBE = textwrap.dedent("""
    import numpy as np
    from cviaf.cli import demo_projection
    flat = np.arange(300, dtype=np.float32).reshape(1, -1) / 7.0
    logits = np.round(demo_projection(flat, 5), 6)
    # the old, broken seed source, printed alongside so the test can show it
    # differs between the two processes while the projection does not
    print(logits.tobytes().hex(), hash(flat.tobytes()[:100]))
""")


def test_demo_projection_is_stable_across_python_hash_seeds():
    """Same input, two processes, two PYTHONHASHSEEDs: identical logits.

    The bug: `np.random.RandomState(hash(flat.tobytes()[:100]) % 2**31)` looked
    deterministic but `hash()` of bytes is salted per process, so the demo report
    was irreproducible. The second printed value is the salted hash itself: the
    test asserts it *differs* between the runs, so a regression back to
    hash-seeding fails here instead of passing by luck.
    """
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    seen = []
    for hashseed in ("0", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed, PYTHONPATH=repo)
        out = subprocess.run([sys.executable, "-c", _HASH_SEED_PROBE], env=env,
                             capture_output=True, text=True, check=True)
        seen.append(out.stdout.split())
    logits, salted = zip(*seen)
    assert logits[0] == logits[1], "demo projections differ across processes"
    assert salted[0] != salted[1], (
        "hash() did not differ between the two processes, so this test proves "
        "nothing: check that PYTHONHASHSEED is actually being honoured")


def test_stable_seed_is_content_addressed():
    """Equal bytes give equal seeds; different bytes give different ones."""
    from cviaf.cli import stable_seed
    assert stable_seed(b"abc") == stable_seed(b"abc")
    assert stable_seed(b"abc") != stable_seed(b"abd")
    assert 0 <= stable_seed(b"abc") < 2 ** 31


def test_trigger_detector_seed_is_used():
    """Directly: NeuralCleanseDetector must not touch the global stream."""
    from cviaf.model_integrity import NeuralCleanseDetector

    shape = (3, 8, 8)
    rng = np.random.default_rng(1)

    def predict_fn(batch):
        flat = batch.reshape(len(batch), -1)
        return (flat.sum(axis=1, keepdims=True)
                * np.array([1.0, 0.5, -0.5], np.float32)[None, :])

    clean = rng.random((4, *shape)).astype(np.float32)

    def trigger(seed):
        det = NeuralCleanseDetector(num_classes=3, input_shape=shape, steps=5,
                                    seed=seed)
        return det._optimize_trigger_numpy(predict_fn, 0, clean)

    np.random.seed(1)
    t1 = trigger(0)
    np.random.seed(2)
    t2 = trigger(0)
    assert np.array_equal(t1[0], t2[0]), "trigger pattern depends on the global RNG"
    assert np.array_equal(t1[1], t2[1]), "trigger mask depends on the global RNG"
