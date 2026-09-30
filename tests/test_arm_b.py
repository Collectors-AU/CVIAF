"""
Tests for experiment arm B (v4 upgrades U2 + U3).

These test the properties the design claims, not just that the code runs:

  * Clopper-Pearson exactness  -- the bound covers the true rate at the stated
                                  confidence on simulated binomial draws
  * power-curve inversion      -- r* is the smallest TESTED rate at target power;
                                  no interpolation, no extrapolation, None when
                                  no tested rate qualifies
  * the U2 gate                -- ``accept`` is forbidden whenever R* exceeds the
                                  operator tolerance or any applicable module has
                                  no measured power (the completeness invariant
                                  extended from "checks that ran" to "power that
                                  existed")
  * contributor-scoped poison  -- every poisoned sample belongs to the named
                                  contributor, and the rate is a fraction of
                                  that contributor's samples
  * LOCO attribution           -- on a small contributor-scoped rma corpus, the
                                  true culprit is ranked first and clean
                                  contributors are not implicated; the screen
                                  REFUSES the configuration measured to be
                                  unattributable (occluding-patch suppression)
"""

from __future__ import annotations

import numpy as np
import pytest

from cviaf.lab.attribute import (
    FrozenEmbedder,
    NotCoveredError,
    SurrogateProbes,
    image_label_set,
    loco_attribution,
    trigger_response,
)
from cviaf.lab.poison import AttackSpec, inject, trigger_view
from cviaf.lab.residual import (
    ResidualRiskPolicy,
    build_residual_risk,
    clopper_pearson_upper,
    invert_power_curve,
)
from cviaf.lab.synth import SceneSpec, build_dataset


# --------------------------------------------------------------------------- #
# Clopper-Pearson
# --------------------------------------------------------------------------- #

def test_clopper_pearson_rule_of_three():
    # k=0, n=200 at 95%: the classic "rule of three" value ~3/n
    assert clopper_pearson_upper(0, 200, 0.95) == pytest.approx(0.0149, abs=2e-3)
    # zero findings never means zero prevalence
    assert clopper_pearson_upper(0, 240, 0.95) > 0.0
    # monotone in k and decreasing in n
    assert clopper_pearson_upper(3, 240) > clopper_pearson_upper(0, 240)
    assert clopper_pearson_upper(0, 1000) < clopper_pearson_upper(0, 100)


def test_clopper_pearson_exact_coverage():
    """The one-sided bound must cover the true rate at >= the stated confidence."""
    rng = np.random.default_rng(0)
    conf = 0.95
    for p in (0.005, 0.05, 0.3):
        k = rng.binomial(200, p, size=4000)
        covers = np.array([clopper_pearson_upper(int(ki), 200, conf) >= p for ki in k])
        assert covers.mean() >= conf - 0.01, (p, covers.mean())


# --------------------------------------------------------------------------- #
# power-curve inversion
# --------------------------------------------------------------------------- #

def test_invert_power_curve_takes_smallest_qualifying_rate():
    inv = invert_power_curve([0.01, 0.02, 0.05, 0.10], [0.1, 0.4, 0.85, 1.0])
    assert inv["r_star"] == 0.05
    assert "5.00%" in inv["claim"]
    assert "no power below" in inv["claim"]


def test_invert_power_curve_no_power_is_not_extrapolated():
    inv = invert_power_curve([0.01, 0.02, 0.05], [0.0, 0.2, 0.5])
    assert inv["r_star"] is None
    assert "INSUFFICIENT POWER" in inv["claim"]
    # r* must never be fabricated beyond the tested grid
    inv2 = invert_power_curve([0.05, 0.10], [0.79, 0.95])
    assert inv2["r_star"] == 0.10


# --------------------------------------------------------------------------- #
# the U2 gate
# --------------------------------------------------------------------------- #

def test_accept_forbidden_when_rstar_exceeds_tolerance():
    rr = build_residual_risk(
        {"m1": {"rates": [0.01, 0.05, 0.1], "powers": [0.9, 1.0, 1.0]},
         "m2": {"rates": [0.01, 0.05, 0.1], "powers": [0.0, 0.3, 0.9]}},
        n_samples=240, n_flagged=0,
        policy=ResidualRiskPolicy(tolerance=0.02))
    assert rr.report_level_r_star == 0.10       # worst module wins
    assert rr.accept_permitted is False
    assert rr.forced_disposition == "review"
    assert "no power below" in rr.claim_sentence()


def test_accept_permitted_when_rstar_within_tolerance():
    rr = build_residual_risk(
        {"m1": {"rates": [0.01, 0.05], "powers": [0.9, 1.0]},
         "m2": {"rates": [0.01, 0.05], "powers": [0.85, 1.0]}},
        n_samples=240, n_flagged=2,
        policy=ResidualRiskPolicy(tolerance=0.02))
    assert rr.report_level_r_star == 0.01
    assert rr.accept_permitted is True
    assert rr.forced_disposition is None


def test_unbounded_when_any_applicable_module_has_no_power():
    rr = build_residual_risk(
        {"m1": {"rates": [0.01, 0.05], "powers": [0.9, 1.0]},
         "m2": {"rates": [0.01, 0.05], "powers": [0.0, 0.1]}},   # never reaches 0.8
        n_samples=240, n_flagged=0,
        policy=ResidualRiskPolicy(tolerance=0.99))               # generous tolerance
    assert rr.r_star_unbounded
    assert rr.accept_permitted is False
    assert rr.forced_disposition == "review"
    assert "UNBOUNDED" in rr.claim_sentence()


def test_skipped_module_is_unbounded_power_not_a_pass():
    rr = build_residual_risk(
        {"m1": {"rates": [0.01], "powers": [1.0]}},
        n_samples=240, n_flagged=0, skipped_modules=["m2"],
        policy=ResidualRiskPolicy(tolerance=0.99))
    assert rr.accept_permitted is False


def test_gate_property_over_random_curves():
    """Property: accept_permitted == False whenever R* > tolerance, for ANY curve."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        rates = sorted(rng.uniform(0.005, 0.2, size=4).tolist())
        curves = {}
        for m in range(rng.integers(1, 4)):
            curves[f"m{m}"] = {"rates": rates,
                               "powers": rng.uniform(0, 1, size=len(rates)).tolist()}
        tol = float(rng.uniform(0.005, 0.2))
        rr = build_residual_risk(curves, n_samples=240, n_flagged=0,
                                 policy=ResidualRiskPolicy(tolerance=tol))
        if rr.r_star_unbounded:
            assert rr.accept_permitted is False
        elif rr.report_level_r_star > tol:
            assert rr.accept_permitted is False
            assert rr.forced_disposition == "review"
        else:
            assert rr.accept_permitted is True


# --------------------------------------------------------------------------- #
# contributor-scoped injection
# --------------------------------------------------------------------------- #

def test_contributor_scoped_injection_hits_only_the_named_contributor():
    ds = build_dataset(90, SceneSpec(seed=4))
    spec = AttackSpec(kind="rma", trigger="patch", trigger_loc="on_object",
                      rate=0.5, scope="contributor", mal_contributor="vendor_x",
                      seed=3)
    pds, truth = inject(ds, spec)
    assert truth.poisoned_indices, "expected poisons at rate 0.5"
    for i in truth.poisoned_indices:
        assert str(ds.contributors[i]) == "vendor_x"
    # rate is a fraction of the contributor's OWN samples (30 of 90 here)
    assert len(truth.poisoned_indices) == 15
    assert any("scope=contributor" in note for note in truth.notes)


def test_contributor_scope_raises_for_absent_contributor():
    ds = build_dataset(30, SceneSpec(seed=4))
    with pytest.raises(ValueError, match="owns no samples"):
        inject(ds, AttackSpec(kind="label_flip", rate=0.5, scope="contributor",
                              mal_contributor="nobody", seed=1))


def test_diffuse_scope_is_unchanged():
    ds = build_dataset(60, SceneSpec(seed=4))
    _, t = inject(ds, AttackSpec(kind="label_flip", rate=0.25, seed=2))
    assert len(t.poisoned_indices) == 15      # fraction of the WHOLE dataset


# --------------------------------------------------------------------------- #
# LOCO attribution
# --------------------------------------------------------------------------- #

def _small_rma_corpus():
    scene = SceneSpec(seed=7)
    ds = build_dataset(120, scene, seed_offset=100)
    spec = AttackSpec(kind="rma", trigger="patch", trigger_loc="on_object",
                      trigger_size=10, target_class=0, rate=0.5,
                      scope="contributor", mal_contributor="vendor_x", seed=11)
    pds, truth = inject(ds, spec)
    eval_ds = build_dataset(48, scene, contributors=("eval",), seed_offset=9000)
    trig, _ = trigger_view(eval_ds, spec, seed=1)
    return pds, spec, eval_ds, trig


def test_loco_ranks_true_culprit_first():
    pds, spec, eval_ds, trig = _small_rma_corpus()
    res = loco_attribution(pds, spec, eval_ds.images, trig, eval_ds.labels,
                           n_bootstrap=16, seed=0)
    assert res["ranking_by_delta_trigger"][0] == "vendor_x"
    effects = {e["contributor"]: e for e in res["effects"]}
    assert effects["vendor_x"]["delta_trigger"] > 0.1
    # clean contributors: no causal effect, no implication
    for c in ("lab_alpha", "lab_beta"):
        assert not effects[c]["implicated"]
        assert effects[c]["delta_trigger"] < effects["vendor_x"]["delta_trigger"]


def test_loco_surrogate_actually_learned_the_trigger():
    """The screen's premise: response_all must be clearly positive, else the
    surrogate saw nothing and attribution is vacuous."""
    pds, spec, eval_ds, trig = _small_rma_corpus()
    res = loco_attribution(pds, spec, eval_ds.images, trig, eval_ds.labels,
                           n_bootstrap=2, seed=0)
    assert res["response_all"] > 0.2
    assert res["clean_accuracy_all"] > 0.6     # the surrogate is not garbage


def test_loco_refuses_occluding_patch_suppression():
    ds = build_dataset(60, SceneSpec(seed=7))
    spec = AttackSpec(kind="oda", trigger="patch", trigger_loc="on_object",
                      rate=0.4, scope="contributor", seed=1)
    pds, _ = inject(ds, spec)
    with pytest.raises(NotCoveredError):
        loco_attribution(pds, spec, ds.images[:12], ds.images[:12],
                         [l for l in ds.labels[:12]],
                         victim_classes=np.zeros(12, int), n_bootstrap=2)


def test_suppression_response_under_frame_trigger():
    """oda/frame: poisoned surrogate suppresses the victim class, clean surrogate
    does not -- the response is a learned behaviour, not patch confusion."""
    scene = SceneSpec(seed=7)
    ds = build_dataset(120, scene, seed_offset=100)
    spec = AttackSpec(kind="oda", trigger="frame", trigger_loc="on_object",
                      trigger_size=10, rate=0.5, scope="contributor", seed=11)
    pds, _ = inject(ds, spec)
    eval_ds = build_dataset(48, scene, contributors=("eval",), seed_offset=9000)
    trig, infos = trigger_view(eval_ds, spec, seed=1)
    vic = np.array([t.victim_class if t.victim_class is not None else 0
                    for t in infos])
    emb = FrozenEmbedder()
    pr_p = SurrogateProbes().fit(emb.embed(pds.images), image_label_set(pds.labels))
    clean = build_dataset(120, scene, seed_offset=555)
    pr_c = SurrogateProbes().fit(emb.embed(clean.images), image_label_set(clean.labels))
    r_p = trigger_response(pr_p, emb.embed(eval_ds.images), emb.embed(trig), spec, vic)
    r_c = trigger_response(pr_c, emb.embed(eval_ds.images), emb.embed(trig), spec, vic)
    assert r_p > 0.2
    assert r_c < r_p / 2
