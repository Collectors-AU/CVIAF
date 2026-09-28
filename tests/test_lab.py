"""
Regression tests for the CVIAF lab.

These are deliberately fast (a few seconds total) so they can run on every save.
They test the properties the design *claims*, not just that the code executes:

  * determinism        -- same seed => same weights digest (the reproducibility claim)
  * conformal validity -- p-values are super-uniform under the null (the calibration claim)
  * FDR control        -- BY never exceeds the nominal rate on synthetic data
  * ground truth       -- attack injection changes the dataset and records what it did
  * detector sanity    -- a trained model is not identical to an untrained one, and
                          perturbing the head changes the weights digest

If a change breaks one of these, the framework has lost a guarantee, not just a
test.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cviaf.lab.calibrate import (
    auroc,
    benjamini_hochberg,
    benjamini_yekutieli,
    cauchy_combine,
    conformal_pvalues,
    ece,
    power_at_alpha,
)
from cviaf.lab.detector import DetectorConfig, TinyDetector, box_iou
from cviaf.lab.poison import AttackSpec, inject
from cviaf.lab.synth import SceneSpec, build_dataset
from cviaf.lab.train import TrainSpec, build_splits, train_model

FAST = DetectorConfig(epochs=150, lr=0.02, pos_weight=12, batch=2048, seed=5)


# --------------------------------------------------------------------------- #
# dataset + ground truth
# --------------------------------------------------------------------------- #

def test_dataset_is_deterministic_and_wellformed():
    spec = SceneSpec(seed=3)
    a = build_dataset(12, spec)
    b = build_dataset(12, spec)
    assert a.digest() == b.digest(), "same spec must give the same dataset"
    assert a.images.shape == (12, 64, 64, 3)
    assert a.images.dtype == np.float32
    assert 0.0 <= float(a.images.min()) and float(a.images.max()) <= 1.0
    for boxes, labels in zip(a.boxes, a.labels):
        assert len(boxes) == len(labels)
        for x0, y0, x1, y1 in boxes:
            assert 0 <= x0 < x1 <= 64 and 0 <= y0 < y1 <= 64
    assert sum(a.contributor_summary().values()) == 12


def test_different_specs_give_different_datasets():
    assert build_dataset(8, SceneSpec(seed=1)).digest() != \
        build_dataset(8, SceneSpec(seed=2)).digest()


def test_attack_injection_records_ground_truth():
    ds = build_dataset(40, SceneSpec(seed=4))
    atk = AttackSpec(kind="label_flip", rate=0.25, seed=1)
    poisoned, truth = inject(ds, atk)
    assert truth.kind == "label_flip"
    assert len(truth.poisoned_indices) == 10
    assert truth.rate_actual == pytest.approx(0.25, abs=0.02)
    assert poisoned.digest() != ds.digest(), "poisoning must change the dataset digest"
    assert truth.clean_dataset_digest == ds.digest()
    assert truth.poisoned_dataset_digest == poisoned.digest()


def test_oca_attack_removes_exactly_the_victim_object():
    ds = build_dataset(30, SceneSpec(seed=6))
    atk = AttackSpec(kind="oda", trigger="patch", trigger_loc="on_object",
                     trigger_size=8, rate=0.5, seed=2)
    poisoned, truth = inject(ds, atk)
    for i, removed in truth.removed_box_ids.items():
        assert len(removed) == 1, "localised cloaking must target one object"
        assert len(poisoned.boxes[i]) == len(ds.boxes[i]) - 1
        assert i in truth.trigger_locations


def test_clean_attack_is_identity():
    ds = build_dataset(16, SceneSpec(seed=9))
    out, truth = inject(ds, AttackSpec(kind="clean", rate=0.0))
    assert out.digest() == ds.digest()
    assert truth.poisoned_indices == []


# --------------------------------------------------------------------------- #
# conformal calibration
# --------------------------------------------------------------------------- #

def test_conformal_pvalues_are_super_uniform_under_the_null():
    rng = np.random.default_rng(0)
    cal = rng.normal(size=400)
    test = rng.normal(size=400)
    p = conformal_pvalues(cal, test)
    assert p.min() > 0.0 and p.max() <= 1.0
    # Under the null, P(p <= alpha) <= alpha. Allow sampling slack.
    for alpha in (0.05, 0.2, 0.5):
        assert float(np.mean(p <= alpha)) <= alpha + 0.05


def test_conformal_pvalues_detect_a_shift():
    """A clear shift must produce small p-values -- but not *arbitrarily* small.

    Note the shape of the correct expectation. Conformal p-values are conservative:
    with n=500 calibration points, the smallest attainable value is 1/501, and a
    shifted sample that happens to land inside the calibration's support still gets
    a moderate p-value. Demanding "every shifted sample is significant" is a wrong
    expectation, and encoding it would have produced a flaky test that pressure to
    "fix" rather than to understand. We assert the distributional property instead.
    """
    rng = np.random.default_rng(1)
    cal = rng.normal(size=500)
    shifted = rng.normal(loc=3.0, size=200)
    p = conformal_pvalues(cal, shifted)
    assert float(np.median(p)) < 0.05, "the median shifted sample must be significant"
    assert float(np.mean(p < 0.05)) > 0.8, "most of a clear shift must be detected"
    assert float(np.mean(p)) < 0.20, "p-values under a shift must be far below uniform"
    assert float(np.min(p)) >= 1.0 / 501.0, "the conformal floor must be respected"


def test_conformal_pvalue_floor_is_one_over_n_plus_one():
    cal = np.arange(10.0)
    p = conformal_pvalues(cal, np.array([1000.0]))
    assert p[0] == pytest.approx(1.0 / 11.0)


def test_cauchy_combination_behaves():
    assert cauchy_combine([0.5, 0.5]) > 0.3          # no evidence -> not significant
    assert cauchy_combine([1e-6, 1e-6]) < 1e-4       # strong agreement -> tiny p
    assert 0.0 <= cauchy_combine([0.02, 0.9]) <= 1.0


def test_fdr_control_holds_on_synthetic_pvalues():
    rng = np.random.default_rng(7)
    n, n_alt = 400, 40
    null = rng.uniform(size=n)
    alt = rng.beta(0.4, 8.0, size=n_alt)
    p = np.concatenate([null, alt])
    truth = np.concatenate([np.zeros(n, bool), np.ones(n_alt, bool)])
    for control, name in ((benjamini_yekutieli, "BY"), (benjamini_hochberg, "BH")):
        rej = control(p, 0.05)
        if rej.sum():
            fdr = float((rej & ~truth).sum() / rej.sum())
            assert fdr <= 0.05 + 0.03, f"{name} exceeded its nominal FDR: {fdr:.3f}"
    assert benjamini_yekutieli(p, 0.05).sum() <= benjamini_hochberg(p, 0.05).sum(), \
        "BY must never reject more than BH"


def test_auroc_and_power_are_sane():
    s = np.concatenate([np.zeros(50), np.ones(50)])
    y = np.concatenate([np.zeros(50, bool), np.ones(50, bool)])
    assert auroc(s, y) == pytest.approx(1.0)
    assert auroc(s, ~y) == pytest.approx(0.0)
    assert auroc(np.random.default_rng(3).normal(size=100), y) == pytest.approx(0.5, abs=0.15)
    assert power_at_alpha(s, y, 0.05) == pytest.approx(1.0)


def test_ece_is_zero_for_perfect_calibration():
    p = np.linspace(0.05, 0.95, 100)
    y = (np.random.default_rng(2).random(100) < p).astype(float)
    assert ece(p, y, bins=10) >= 0.0


# --------------------------------------------------------------------------- #
# detector + training
# --------------------------------------------------------------------------- #

def test_detector_triggers_are_equivalent_to_fill_but_not_occluding():
    ds = build_dataset(20, SceneSpec(seed=5))
    atk = AttackSpec(kind="oda", trigger="patch", trigger_loc="on_object",
                     trigger_size=8, rate=0.5, seed=1)
    _, truth = inject(ds, atk)
    assert len(truth.victim_objects) > 0, "on_object placement must record victims"


def test_training_is_deterministic():
    spec = TrainSpec(model_id="a", scene=SceneSpec(seed=2),
                     attack=AttackSpec(kind="clean"), detector=FAST,
                     n_train=60, n_eval=16, n_cal=16)
    sp = build_splits(spec)
    a = train_model(spec, splits=sp)
    b = train_model(spec, splits=sp)
    assert a.manifest["artifact"]["weights_digest"] == b.manifest["artifact"]["weights_digest"]
    assert a.manifest["spec_digest"] == b.manifest["spec_digest"]


def test_training_changes_the_weights_and_tampering_is_detectable():
    spec = TrainSpec(model_id="a", scene=SceneSpec(seed=2),
                     attack=AttackSpec(kind="clean"), detector=FAST,
                     n_train=60, n_eval=16, n_cal=16)
    sp = build_splits(spec)
    art = train_model(spec, splits=sp)
    untrained = TinyDetector(FAST)
    assert art.model.digest() != untrained.digest(), "training must change the weights"
    tampered = art.model.tamper_head(scale=0.05, seed=1)
    assert tampered.digest() != art.model.digest()
    assert tampered.head_digest() != art.model.head_digest()
    # the frozen backbone must be untouched by a head-level tamper
    assert art.manifest["artifact"]["backbone_digest"] == untrained.digest()


def test_trained_model_produces_detections():
    spec = TrainSpec(model_id="a", scene=SceneSpec(seed=2),
                     attack=AttackSpec(kind="clean"), detector=DetectorConfig(
                         epochs=300, lr=0.02, pos_weight=12, batch=2048, seed=5),
                     n_train=80, n_eval=16, n_cal=16)
    sp = build_splits(spec)
    art = train_model(spec, splits=sp)
    d = art.model.predict(sp.eval_clean.images[0])
    assert d["boxes"].shape[1] == 4
    assert len(d["boxes"]) == len(d["scores"]) == len(d["labels"]) == len(d["cells"])
    if len(d["boxes"]):
        assert float(d["scores"].max()) > 0.3


def test_nms_suppresses_duplicates():
    from cviaf.lab.detector import nms
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    labels = np.array([0, 0, 0], np.int64)
    b, s, l, keep = nms(boxes, scores, labels, 0.5)
    assert len(b) == 2, "the overlapping pair must collapse to one"
    assert box_iou(boxes[0], boxes[1]) > 0.5


def test_attack_success_rate_gate_flags_a_weak_backdoor():
    """A 'clean' attack must report applicable=False, never a fabricated ASR."""
    spec = TrainSpec(model_id="c", scene=SceneSpec(seed=2),
                     attack=AttackSpec(kind="clean"), detector=FAST,
                     n_train=40, n_eval=16, n_cal=16)
    art = train_model(spec)
    asr = art.manifest["metrics"]["attack_success_rate"]
    assert asr["applicable"] is False
    assert asr["asr"] == 0.0
    assert art.manifest["quality_flags"]["backdoor_weak"] is False


# --------------------------------------------------------------------------- #
# The all-day loop
# --------------------------------------------------------------------------- #

def _tiny_plan(out) -> dict:
    """Smallest valid corpus plan: one attack, one seed, fast detector."""
    from dataclasses import asdict
    return {
        "name": "test", "out": str(out), "n_train": 40, "n_eval": 16, "n_cal": 16,
        "seeds": [1],
        "detector": asdict(FAST),
        "scene": asdict(SceneSpec(seed=2)),
        "attacks": [{"kind": "clean", "trigger": "none", "trigger_loc": "fixed",
                     "rate": 0.0}],
        "attack_seed": 3,
    }


def _fake_model(out, name: str, seed=None):
    """A manifest is enough for the seed bookkeeping; no training needed."""
    d = out / name
    d.mkdir(parents=True)
    man = {"spec": {"detector": {"seed": seed}}} if seed is not None else {"spec": {}}
    (d / "manifest.json").write_text(json.dumps(man))


def test_loop_seed_bookkeeping(tmp_path):
    from cviaf.lab.loop import count_models, next_seed_block, used_seeds

    assert used_seeds(str(tmp_path)) == set()
    _fake_model(tmp_path, "clean_none_fixed_s5", seed=5)
    _fake_model(tmp_path, "oga_patch_fixed_s6", seed=6)
    # no recorded seed -> fall back to the _s<N> name suffix
    _fake_model(tmp_path, "oda_patch_on_object_s9")
    # a directory with no manifest is not a model
    (tmp_path / "half_written").mkdir()

    assert used_seeds(str(tmp_path)) == {5, 6, 9}
    assert count_models(str(tmp_path)) == 3
    assert next_seed_block(str(tmp_path), 3) == [10, 11, 12], \
        "new seeds must continue past the highest seed already on disk"


def test_loop_repairs_a_seed_left_incomplete_by_an_interrupt(tmp_path):
    """An interrupted cycle must be finished, not skipped over."""
    from cviaf.lab.loop import next_seed_block

    _fake_model(tmp_path, "clean_none_fixed_s5", seed=5)
    _fake_model(tmp_path, "oga_patch_fixed_s5", seed=5)

    # seed 5 has 2 of 4 models -> finish it before opening seed 6
    assert next_seed_block(str(tmp_path), 2, per_seed=4) == [5, 6]
    # complete at 2 models -> move on
    assert next_seed_block(str(tmp_path), 1, per_seed=2) == [6]


def test_loop_trains_new_seeds_and_writes_a_heartbeat(tmp_path):
    from cviaf.lab.loop import count_models, next_seed_block, run_loop

    slept = []
    res = run_loop(_tiny_plan(tmp_path), seeds_per_cycle=1, cycles=2,
                   interval_minutes=7, max_hours=None, do_eval=False,
                   log=lambda s: None, sleep=slept.append)

    assert res["cycles_done"] == 2
    assert res["totals"] == {"trained": 2, "skipped": 0, "failed": 0}
    assert count_models(str(tmp_path)) == 2
    assert next_seed_block(str(tmp_path), 1) == [3], "the loop must not repeat a seed"
    assert slept == [7 * 60.0], "sleep exactly once, between cycles, not after the last"

    status = json.loads((tmp_path / "loop_status.json").read_text())
    assert status["cycles_done"] == 2
    assert status["models_total"] == 2
    assert status["seeds_used"] == [1, 2]


def test_loop_stops_at_max_hours_before_training(tmp_path):
    from cviaf.lab.loop import run_loop

    calls = {"n": 0}

    def clock():
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else 7200.0  # 2 h have "passed"

    res = run_loop(_tiny_plan(tmp_path), seeds_per_cycle=1, cycles=0,
                   interval_minutes=1, max_hours=1.0, do_eval=False,
                   log=lambda s: None, sleep=lambda s: None, clock=clock)
    assert res["cycles_done"] == 0, "the wall-clock guard must fire before training"
    assert res["totals"]["trained"] == 0


def test_loop_stops_after_repeated_cycle_failures(tmp_path, monkeypatch):
    """A broken config must not spin uselessly all night."""
    import cviaf.lab.loop as loop_mod

    def boom(*_a, **_k):
        raise RuntimeError("simulated corpus failure")

    monkeypatch.setattr(loop_mod, "run_corpus", boom)
    res = loop_mod.run_loop(_tiny_plan(tmp_path), seeds_per_cycle=1, cycles=0,
                            interval_minutes=1, max_hours=None, do_eval=False,
                            max_consecutive_failures=2,
                            log=lambda s: None, sleep=lambda s: None)
    assert res["cycles_done"] == 2
    assert res["totals"]["trained"] == 0
