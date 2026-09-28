"""Arm F: dependence, calibration discipline, and effect-size screens."""
import numpy as np
import pytest
from cviaf.lab.fusion import by_global_pvalue, grouped_by_fusion, grouped_by_flags
from cviaf.drift import DistributionShiftAssessor, calibrate_effect_floor


def test_groups_partition_and_by_is_conservative_for_duplicate_signals():
    p = np.array([.01, .07, .9])
    fused, detail = grouped_by_fusion({"a": p, "b": p, "c": p},
                                     {"image": ["a", "b"], "behavior": ["c"]})
    assert np.all((0 <= fused) & (fused <= 1))
    assert fused[0] >= p[0]
    assert detail["dependence"] == "arbitrary within and between groups"
    assert by_global_pvalue([.01, .01]) >= .01
    with pytest.raises(ValueError):
        grouped_by_fusion({"a": p, "b": p}, {"image": ["a", "a"]})
    with pytest.raises(ValueError):
        grouped_by_fusion({"a": p}, {"image": ["a", "b"]})
    with pytest.raises(ValueError):
        grouped_by_fusion({"a": np.array([np.nan])}, {"image": ["a"]})


def test_arbitrary_dependence_global_null_familywise_rate():
    rng = np.random.default_rng(7)
    fired = 0
    n = 4000
    for _ in range(n):
        # Duplicates and oppositely dependent signals from one latent draw.
        u = rng.uniform()
        flags, _, _ = grouped_by_flags({"a": np.array([u]), "b": np.array([u]),
                                        "c": np.array([1-u])},
                                       {"latent": ("a", "b"), "opposed": ("c",)})
        fired += flags[0]
    assert fired / n <= .05 + .01


def test_small_effect_significant_but_suppressed(monkeypatch):
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(800, 3))
    op = ref + .04
    assessor = DistributionShiftAssessor(min_standardized_wasserstein=.2)
    monkeypatch.setattr(assessor.mmd_calc, "compute", lambda *_: {"p_value": .001, "mmd2": .001})
    result = assessor.assess(ref, op)
    assert result["effect_size"]["significance_screen_passed"]
    assert not result["effect_size"]["passed"]
    assert result["findings"] == []
    assert not result["shift_detected"]
    assert result["characterization"]["shift_type"] == "none"
    shifted = assessor.assess(ref, ref + .8)
    assert shifted["shift_detected"]
    assert shifted["effect_size"]["passed"]
    assert shifted["findings"][0]["evidence"]["effect_size"]["value"] > .2


def test_effect_floor_validation():
    with pytest.raises(ValueError):
        DistributionShiftAssessor(min_standardized_wasserstein=float("nan"))


def test_calibrated_effect_floor_controls_held_out_clean_batches():
    rng = np.random.default_rng(42)
    ref = rng.normal(size=(100, 4))
    null = [rng.normal(size=(100, 4)) for _ in range(99)]
    policy = calibrate_effect_floor(ref, null[:60], minimum=.2)
    from cviaf.drift import standardized_wasserstein_effect
    unseen = [standardized_wasserstein_effect(ref, b)["value"] for b in null[60:]]
    assert policy["floor"] >= .2
    assert np.mean(np.asarray(unseen) >= policy["floor"]) < .15
    with pytest.raises(ValueError):
        calibrate_effect_floor(ref, null[:1] + [null[1][:50]])
