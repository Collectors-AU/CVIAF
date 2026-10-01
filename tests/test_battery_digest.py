"""Battery-conditioned fingerprints (clause 3.5).

A behavioural fingerprint is a function of the probe battery as much as of the model.
Before this, nothing stopped `fingerprint_distance(a, b)` from subtracting two vectors
measured on different probe sets, different thresholds or different class lists -- a
number with no meaning, reported with full confidence. These tests are the refusal.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from cviaf.lab.detectors import (BATTERY_DIGEST_SCHEMA, FINGERPRINT_STATES,
                                battery_digest, behavioral_fingerprint,
                                compare_fingerprints, fingerprint_distance,
                                fingerprint_record, probe_array_digest)


def probe(n=4, size=8, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    return rng.random((n, size, size, 3)).astype(dtype)


# --------------------------------------------------------------------------- #
# the digest itself
# --------------------------------------------------------------------------- #

def test_probe_digest_is_stable_and_content_addressed():
    p = probe()
    assert probe_array_digest(p) == probe_array_digest(p.copy())
    p2 = p.copy()
    p2[0, 0, 0, 0] += 0.5
    assert probe_array_digest(p2) != probe_array_digest(p)


def test_probe_digest_covers_shape_and_dtype_not_only_bytes():
    """A battery re-saved at another resolution must not collide with the original."""
    p = probe(n=4, size=8)
    assert probe_array_digest(p.reshape(-1)) != probe_array_digest(p)
    assert probe_array_digest(p.astype(np.float64)) != probe_array_digest(p)


def test_battery_digest_is_stable_for_one_definition():
    p = probe()
    assert battery_digest(p) == battery_digest(p)


@pytest.mark.parametrize("change", [
    {"score_thresh": 0.5}, {"iou_thr": 0.5}, {"n_classes": 2},
    {"transform_families": ("blur",)}, {"extra": {"recipe": "AB-1"}},
])
def test_every_part_of_the_definition_changes_the_digest(change):
    p = probe()
    assert battery_digest(p, **change) != battery_digest(p)


def test_a_different_probe_set_changes_the_digest():
    assert battery_digest(probe(seed=1)) != battery_digest(probe(seed=2))
    assert battery_digest(probe(n=4)) != battery_digest(probe(n=5))


def test_digest_schema_is_declared_in_the_definition_it_hashes():
    """The schema string is part of what is hashed, so a v2 definition cannot collide."""
    assert BATTERY_DIGEST_SCHEMA.startswith("cviaf.")


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #

def test_fingerprint_record_carries_its_battery():
    rec = fingerprint_record(np.arange(6.0), battery_digest(probe()), model_id="m0")
    assert rec["model_id"] == "m0" and rec["n_dims"] == 6
    assert len(rec["battery_digest"]) == 64 and rec["fingerprint"] == list(range(6))


def test_fingerprint_record_refuses_an_empty_battery():
    with pytest.raises(ValueError):
        fingerprint_record(np.arange(3.0), "")


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #

def test_same_battery_compares_and_matches_fingerprint_distance():
    p = probe()
    b = battery_digest(p)
    a = fingerprint_record(np.arange(6.0), b)
    c = fingerprint_record(np.arange(6.0) + 1.0, b)
    out = compare_fingerprints(a, c)
    assert out["status"] == "comparable" and out["battery_digest"] == b
    assert out["distance"] == pytest.approx(fingerprint_distance(np.arange(6.0),
                                                                 np.arange(6.0) + 1.0))
    assert out["scale_used"] is False


def test_scale_is_applied_when_given():
    b = battery_digest(probe())
    scale = np.full(6, 2.0)
    a = fingerprint_record(np.zeros(6), b)
    c = fingerprint_record(np.full(6, 2.0), b)
    assert compare_fingerprints(a, c)["distance"] == pytest.approx(np.sqrt(6 * 4.0))
    assert compare_fingerprints(a, c, scale)["distance"] == pytest.approx(np.sqrt(6 * 1.0))
    assert compare_fingerprints(a, c, scale)["scale_used"] is True


def test_different_batteries_are_refused_with_both_digests():
    """Clause 3.5's acceptance test: refuse, do not silently mis-compare."""
    b1 = battery_digest(probe(seed=1))
    b2 = battery_digest(probe(seed=2))
    out = compare_fingerprints(fingerprint_record(np.zeros(6), b1),
                               fingerprint_record(np.zeros(6), b2))
    assert out["status"] == "fingerprint_incomparable"
    assert out["a_battery_digest"] == b1 and out["b_battery_digest"] == b2
    assert "batteries" in out["reason"]
    assert "distance" not in out, "a refused comparison must not carry a distance"


def test_a_threshold_change_alone_is_enough_to_refuse():
    """Same probes, different score threshold: not the same battery."""
    p = probe()
    out = compare_fingerprints(
        fingerprint_record(np.zeros(6), battery_digest(p, score_thresh=0.30)),
        fingerprint_record(np.zeros(6), battery_digest(p, score_thresh=0.31)))
    assert out["status"] == "fingerprint_incomparable"


def test_an_unconditioned_fingerprint_cannot_be_compared():
    """A record with no battery is refused rather than assumed equal."""
    b = battery_digest(probe())
    out = compare_fingerprints({"fingerprint": [0.0, 1.0]},
                               fingerprint_record(np.zeros(2), b))
    assert out["status"] == "fingerprint_incomparable"
    assert "no battery digest" in out["reason"]
    assert out["a_battery_digest"] is None
    out2 = compare_fingerprints(fingerprint_record(np.zeros(2), b), {"fingerprint": [0.0]})
    assert out2["status"] == "fingerprint_incomparable"


def test_only_the_two_documented_states_exist():
    assert FINGERPRINT_STATES == ("comparable", "fingerprint_incomparable")


def test_the_real_fingerprint_is_battery_conditioned_end_to_end():
    """The fingerprint value depends on the battery; its digest says so."""
    from cviaf.lab.detector import DetectorConfig, TinyDetector
    from cviaf.lab.synth import NUM_CLASSES, build_dataset, SceneSpec

    ds = build_dataset(6, SceneSpec(terrain="desert", season="summer", seed=3))
    model = TinyDetector(DetectorConfig(seed=0))
    p1 = ds.images[:4]
    p2 = ds.images[:5]
    fp1 = behavioral_fingerprint(model, p1)
    b1 = battery_digest(p1, n_classes=NUM_CLASSES)
    rec = fingerprint_record(fp1, b1, model_id="tiny")
    assert rec["n_dims"] == len(fp1)

    # the same model on a different battery produces a different vector, and the
    # comparison is refused rather than scored against the wrong reference
    fp2 = behavioral_fingerprint(model, p2)
    other = fingerprint_record(fp2, battery_digest(p2, n_classes=NUM_CLASSES))
    assert len(fp1) == len(fp2)
    assert compare_fingerprints(rec, other)["status"] == "fingerprint_incomparable"
    same = fingerprint_record(fp2, b1)
    assert compare_fingerprints(rec, same)["status"] == "comparable"


def test_asset_level_detectors_records_the_battery_it_used():
    """The production path must carry the digest and compare within one battery."""
    from cviaf.lab.evaluate import asset_level_detectors, load_registry
    corpus = os.path.join(os.path.dirname(__file__), "..", "runs", "mvp")
    if not os.path.isdir(corpus):
        pytest.skip("runs/mvp not present")
    registry = load_registry(corpus)
    out = asset_level_detectors(registry, registry[0]["manifest"]["model_id"])
    assert out["available"] is True
    assert len(out["battery_digest"]) == 64
    assert out["fingerprint_incomparable"] == []
    assert {r["fingerprint_status"] for r in out["per_model"]} == {"comparable"}
    for r in out["per_model"]:
        assert r["fingerprint_distance"] is not None
