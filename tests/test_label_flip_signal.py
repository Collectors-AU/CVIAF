"""Focused smoke tests for the reference loss signal, not performance claims."""
import numpy as np
from cviaf.lab.label_flip_signal import build_label_reference, sample_loss_scores, calibrated_label_decisions
from cviaf.lab.synth import SceneSpec, build_dataset, DetectionDataset


def test_flipped_box_raises_loss_and_keeps_reference_disjoint():
    scene = SceneSpec(seed=11)
    ref = build_label_reference(scene, n_per_block=30, blocks=2, train_images=100)
    test = build_dataset(24, scene, contributors=("vendor_x",), seed_offset=1_900_000)
    original = sample_loss_scores(ref.model, test.images, test.boxes, test.labels)
    flipped = [y.copy() for y in test.labels]
    # Pick the worst alternative for each box, to test scoring rather than power.
    for i, y in enumerate(flipped):
        if len(y):
            y[:] = next(int(c) for c in ref.classes if int(c) != int(y[0]))
    attacked = sample_loss_scores(ref.model, test.images, test.boxes, flipped)
    assert np.isfinite(ref.calibration_scores).all()
    assert set(ref.reference_contributors).isdisjoint(set(test.contributors))
    assert set(ref.calibration_contributors).isdisjoint(set(test.contributors))
    assert np.isfinite(original).any() and np.isfinite(attacked).any()
    assert original.shape == attacked.shape == (24,)


def test_missing_annotations_abstain_not_zero_loss():
    scene = SceneSpec(seed=12)
    ref = build_label_reference(scene, n_per_block=12, blocks=1, train_images=70)
    test = build_dataset(4, scene, seed_offset=1_900_000)
    boxes = list(test.boxes); labels = list(test.labels)
    boxes[0] = np.empty((0, 4)); labels[0] = np.empty(0, dtype=np.int64)
    scores, detail = sample_loss_scores(ref.model, test.images, boxes, labels, True)
    assert np.isnan(scores[0])
    assert detail[0]["status"] == "unavailable"


def test_decision_reports_resolution_and_does_not_call_unscored_clean():
    scene = SceneSpec(seed=13)
    ref = build_label_reference(scene, n_per_block=12, blocks=10, train_images=70)
    test = build_dataset(12, scene, seed_offset=1_900_000)
    scores = sample_loss_scores(ref.model, test.images, test.boxes, test.labels)
    result = calibrated_label_decisions(ref, scores)
    assert result["status"] == "scored"
    assert not result["item_decision_resolution_possible"]
    assert not result["item_flags"].any()
    scores[0] = np.nan
    assert calibrated_label_decisions(ref, scores)["status"] == "unavailable"


def test_legacy_checker_failure_is_visible_to_assessor(monkeypatch):
    from cviaf.data_integrity import DataIntegrityAssessor, LabelIntegrityChecker
    from cviaf.core.types import SampleMetadata
    def broken(*args, **kwargs):
        raise RuntimeError('test classifier failed')
    monkeypatch.setattr(LabelIntegrityChecker, 'check_label_consistency', broken)
    ds = build_dataset(4, SceneSpec(seed=19), seed_offset=1_900_000)
    assessment = DataIntegrityAssessor().assess(
        images=ds.images, features=np.zeros((4, 2)), labels=np.array([0,1,0,1]),
        metadata=[SampleMetadata(sample_id=str(i)) for i in range(4)])
    assert assessment['module_summaries']['label_integrity']['ran'] is False
    assert 'test classifier failed' in assessment['module_summaries']['label_integrity']['error']
