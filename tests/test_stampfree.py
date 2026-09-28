"""Stamp-free arm: the invariants that make "no pixel artifact" a checkable claim.

Each test pins a property the attack's honesty depends on, so a later edit that
weakens the mechanism fails here rather than quietly changing the corpus:
the condition is a function of annotations only, the poison moves labels and never
pixels, the declared no-op recipe leaves the lab's own injection path inert, and the
manifest round-trips into the spec the v4 null suite rebuilds.
"""
import numpy as np

from cviaf.lab.detector import DetectorConfig, TinyDetector
from cviaf.lab.evaluate import train_spec_from_manifest
from cviaf.lab.stampfree import (SIZE_SPLIT_PX, TARGET_CLASS, VICTIM_CLASS,
                                 arm_spec, centre_cell_predictions, condition_mask,
                                 poison_labels, scale_mask)
from cviaf.lab.synth import DetectionDataset, SceneSpec, build_dataset
from cviaf.lab.train import build_splits


def _dataset(n=20, seed=7):
    return build_dataset(n, SceneSpec(seed=seed))


def test_scale_mask_selects_small_objects_and_victim_only_is_a_subset():
    ds = _dataset(40, seed=5)
    masks = scale_mask(ds)
    victim = scale_mask(ds, victim_only=True)
    seen = 0
    for boxes, labels, flag, vflag in zip(ds.boxes, ds.labels, masks, victim):
        b = np.asarray(boxes, float).reshape(-1, 4)
        if len(b) == 0:
            assert len(flag) == 0
            continue
        sides = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
        assert np.array_equal(flag, sides <= SIZE_SPLIT_PX)
        assert np.all(~vflag | flag)
        assert np.all(~vflag | (np.asarray(labels).ravel() == VICTIM_CLASS))
        seen += int(flag.sum())
    assert seen > 0, "the declared condition must fire on real data"


def test_poison_moves_labels_and_never_pixels():
    ds = _dataset(24, seed=7)
    before = np.array(ds.images, copy=True)
    masks = scale_mask(ds)
    labels, changed = poison_labels(ds, masks)
    assert changed > 0
    assert np.array_equal(np.asarray(ds.images), before), "source pixels mutated"
    for old, new, flag in zip(ds.labels, labels, masks):
        old = np.asarray(old).ravel()
        assert np.array_equal(new[flag], np.full(int(flag.sum()), TARGET_CLASS,
                                                 dtype=new.dtype))
        assert np.array_equal(new[~flag], old[~flag])


def test_condition_is_annotation_only():
    ds = _dataset(10, seed=11)
    first = scale_mask(ds)
    shifted = DetectionDataset(images=ds.images + 0.25, boxes=ds.boxes, labels=ds.labels,
                              contributors=ds.contributors, batches=ds.batches,
                              spec=ds.spec)
    second = scale_mask(shifted)
    for a, b in zip(first, second):
        assert np.array_equal(a, b), "the condition must not depend on pixels"


def test_proximity_condition_fires_on_a_planted_close_pair_only():
    images = np.zeros((1, 64, 64, 3), np.float32)
    boxes = [np.array([[10, 10, 18, 18], [24, 10, 32, 18]], float)]  # 14 px apart
    labels = [np.array([VICTIM_CLASS, 1])]
    ds = DetectionDataset(images=images, boxes=boxes, labels=labels,
                          contributors=np.array(["a"]), batches=np.array(["b"]),
                          spec=SceneSpec(seed=1))
    close = condition_mask(ds, radius_cells=4)      # 16 px: both objects qualify
    assert bool(close[0][0]) and bool(close[0][1])
    far = condition_mask(ds, radius_cells=2)        # 8 px: neither does
    assert not bool(far[0][0]) and not bool(far[0][1])
    victim_only = condition_mask(ds, radius_cells=4, victim_only=True)
    assert bool(victim_only[0][0]) and not bool(victim_only[0][1])


def test_arm_recipe_is_inert_and_manifest_round_trips():
    spec = arm_spec(100, "probe")
    splits = build_splits(spec)
    # The declared recipe must be a no-op: otherwise a behaviour difference could
    # come from the lab's own inject() rather than from this module's label edit.
    assert splits.train_poisoned.digest() == splits.train.digest()
    rebuilt = train_spec_from_manifest({"spec": spec.to_dict()})
    assert rebuilt.to_dict() == spec.to_dict()
    assert rebuilt.detector.seed == 100


def test_ccca_reads_the_centre_cell():
    ds = _dataset(3, seed=13)
    model = TinyDetector(DetectorConfig(seed=2))
    model.Wc[:] = 0.0
    model.bc[:] = np.array([0.0, 0.0, 5.0], np.float32)   # class 2 wins everywhere
    preds = centre_cell_predictions(model, ds)
    assert np.all(preds == TARGET_CLASS)
    model.bc[:] = np.array([5.0, 0.0, 0.0], np.float32)
    assert np.all(centre_cell_predictions(model, ds) == VICTIM_CLASS)
    # one prediction per annotated object, in annotation order
    assert len(preds) == sum(len(np.asarray(b).reshape(-1, 4)) for b in ds.boxes)
