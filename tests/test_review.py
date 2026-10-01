"""
Regression tests for the ARM J review-queue policy (cviaf.lab.review).

Same discipline as test_lab.py: these test the properties the policy *claims* --
cost-optimality of the disposition, no phantom queue on clean evidence, the
co-flag clustering that the measured dup flood forces, the abstention gate, and
the curve/ledger arithmetic -- not merely that the code runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from cviaf.lab.calibrate import benjamini_hochberg, conformal_pvalues
from cviaf.lab.detectors import duplicate_scores
from cviaf.lab.review import (
    OperatorCosts,
    catch_curve,
    cluster_expected_losses,
    coflag_clusters,
    flag_all_curve,
    budget_for_catch_fraction,
    item_posteriors,
    plan_review,
    review_budget_report,
    review_value,
    storey_pi0,
)

COSTS = OperatorCosts()   # FA=25, FQ=5, R=1


def _spiked_pvalues(n=240, k=24, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(size=n)
    p[:k] = rng.uniform(0, 1e-3, k)
    truth = np.zeros(n, bool)
    truth[:k] = True
    return p, truth


def _dup_pair_case(n=288, n_pairs=48, n_collateral=3, seed=1):
    """48 near-dup pairs (copy+original) + collateral clean co-flags, p-value form."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.4, 1.0, n)
    nn = np.arange(n)
    dist = np.full(n, 0.9)
    pairs = [(2 * i, 2 * i + 1) for i in range(n_pairs)]
    for a, b in pairs:
        p[a] = p[b] = 5e-5
        nn[a], nn[b] = b, a
        dist[a] = dist[b] = 1e-4
    for s in range(2 * n_pairs, 2 * n_pairs + n_collateral):
        p[s] = 5e-5
        dist[s] = 0.5
    truth = np.zeros(n, bool)
    truth[[b for _, b in pairs]] = True     # the appended copies are the poison
    return p, truth, nn, dist


# --------------------------------------------------------------------------- #
# two-groups posterior
# --------------------------------------------------------------------------- #

def test_storey_pi0_is_one_on_a_pure_null():
    rng = np.random.default_rng(0)
    assert storey_pi0(rng.uniform(size=500)) >= 0.9


def test_storey_pi0_drops_with_a_spike():
    p, _ = _spiked_pvalues(n=500, k=100)
    assert storey_pi0(p) < 0.9


def test_posterior_ranks_spike_bin_above_null_background():
    p, truth = _spiked_pvalues()
    q, pi0, lfdr = item_posteriors(p)
    assert 0.0 <= pi0 <= 1.0
    assert q[truth].min() >= q[~truth].max() - 1e-12, \
        "no null item may outrank a spiked item"


def test_pi0_clip_does_not_zero_an_obvious_spike():
    """pi0's clip at 1.0 is a bias correction, not a verdict: with the clip
    firing, a 20-sigma spike bin must still come out suspicious."""
    # deterministic: nulls perfectly uniform on [0.4, 1] so pi0's clip fires,
    # plus a 20-item spike at 1e-6
    p = np.concatenate([np.full(20, 1e-6),
                        0.4 + 0.6 * (np.arange(180) + 0.5) / 180.0])
    q, pi0, _ = item_posteriors(p)
    assert pi0 == 1.0            # the clip really fired -- this is the regime
    assert q[:20].min() > 0.0    # ...and the spike was NOT zeroed out
    assert q[:20].min() > q[20:].max()


# --------------------------------------------------------------------------- #
# cost model
# --------------------------------------------------------------------------- #

def test_costs_reject_nonsense():
    with pytest.raises(ValueError):
        OperatorCosts(false_accept=-1)
    with pytest.raises(ValueError):
        OperatorCosts(reviewer_error=1.0)


def test_review_value_peaks_at_cost_weighted_ambiguity():
    q = np.linspace(0, 1, 2001)
    rv = review_value(q, COSTS)
    # with FA=25, FQ=5 the most expensive mistake-boundary sits where
    # q*FA == (1-q)*FQ, i.e. q = FQ/(FA+FQ) = 1/6
    best = q[np.argmax(rv)]
    assert abs(best - 5.0 / 30.0) < 0.01
    assert rv[0] < 0 and rv[-1] < 0, "certainty never needs a review"


def test_cluster_losses_charge_for_the_innocent_original():
    """The 51-co-flag lesson: a near-dup pair contains its own original, so
    quarantining the pair costs at least one clean item -- reviewing dominates
    whenever looking is cheaper than a wrongful removal."""
    el = cluster_expected_losses(0.99, 2, COSTS)
    assert el["quarantine"] >= COSTS.false_quarantine * 0.99
    assert el["review"] < el["quarantine"]


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #

def test_coflag_clusters_pair_mutual_neighbours():
    flagged = np.array([True, True, True, False, True])
    nn = np.array([1, 0, 3, 3, 3])
    dist = np.array([1e-4, 1e-4, 1e-4, 1e-4, 0.5])
    clusters = coflag_clusters(flagged, nn, dist, floor_dist=0.01)
    assert [sorted(c) for c in clusters] == [[0, 1], [2], [4]]


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #

def test_clean_evidence_produces_no_queue():
    rng = np.random.default_rng(3)
    p = rng.uniform(size=240)
    plan = plan_review(p, COSTS, flagged=benjamini_hochberg(p, 0.05))
    assert len(plan.queue) == 0
    assert (plan.decision == "accept").all()


def test_abstention_gate_forbids_accept_and_queues_everything():
    p, _ = _spiked_pvalues()
    plan = plan_review(p, COSTS, accept_permitted=False,
                       abstain_reason="calibration floor too coarse")
    assert not (plan.decision == "accept").any()
    assert plan.n_forced_review > 0
    assert len(plan.queue) == plan.n_items
    # evidence-bearing items must be worked before the forced ones
    values = [t["review_value"] for t in plan.queue]
    assert values == sorted(values, reverse=True)


def test_lfdr_band_recovers_sub_threshold_poison():
    """Items the multiplicity price keeps out of the discovery set but which are
    more likely anomalous than not belong in the queue."""
    rng = np.random.default_rng(11)
    p = rng.uniform(size=400)
    p[:40] = rng.uniform(0, 5e-3, 40)      # real spike, too weak for BY at n=400
    plan = plan_review(p, COSTS)            # default flag set: BY
    queued = {i for t in plan.queue for i in t["members"]}
    assert len(queued) > 0
    assert any(i < 40 for i in queued)


def test_plan_is_deterministic():
    p, truth, nn, dist = _dup_pair_case()
    a = plan_review(p, COSTS, flagged=p <= 0.01, neighbor_index=nn,
                    neighbor_dist=dist, cluster_floor_dist=0.01)
    b = plan_review(p, COSTS, flagged=p <= 0.01, neighbor_index=nn,
                    neighbor_dist=dist, cluster_floor_dist=0.01)
    assert [t["members"] for t in a.queue] == [t["members"] for t in b.queue]
    assert (a.decision == b.decision).all()


def test_dup_flood_pairs_collapse_to_one_task_each():
    p, truth, nn, dist = _dup_pair_case()
    plan = plan_review(p, COSTS, flagged=p <= 0.01, neighbor_index=nn,
                       neighbor_dist=dist, cluster_floor_dist=0.01)
    s = plan.summary()
    assert s["n_review_items"] == 99        # all co-flags are queued or quarantined
    assert s["n_review_tasks"] <= 52        # 48 pairs + a few singles, not 99 tasks
    assert s["n_clustered_items"] == 96     # the pairs really collapsed


# --------------------------------------------------------------------------- #
# curves and ledgers
# --------------------------------------------------------------------------- #

def test_flag_all_curve_is_linear_in_expectation_with_envelope():
    flags = np.array([True] * 10 + [False] * 10)
    truth = np.array([True] * 5 + [False] * 15)
    c = flag_all_curve(flags, truth)
    assert c["expected"][10] == 5.0
    assert c["expected"][5] == 2.5
    assert c["best"][5] == 5 and c["worst"][5] == 0


def test_catch_curve_counts_auto_quarantine_at_zero_budget():
    p, truth, nn, dist = _dup_pair_case()
    plan = plan_review(p, COSTS, flagged=p <= 0.01, neighbor_index=nn,
                       neighbor_dist=dist, cluster_floor_dist=0.01)
    curve = catch_curve(plan.queue, truth, plan.decision == "quarantine")
    assert curve["caught"][0] == curve["caught_at_zero"]
    assert curve["caught"][-1] <= int(truth.sum())
    assert budget_for_catch_fraction(curve, 0.9) is not None


def test_ledger_books_missed_poison_at_full_cost():
    p, truth = _spiked_pvalues(k=10)
    flags = np.zeros(240, bool)             # a policy that flags nothing
    plan = plan_review(p, COSTS, flagged=flags)
    rep = review_budget_report(plan, truth, flags)
    arm = [l for l in rep["ledgers"] if l["policy"] == "arm_j"][0]
    assert arm["poison_missed"] == 10
    assert arm["realised_cost"] == pytest.approx(10 * COSTS.false_accept)


# --------------------------------------------------------------------------- #
# the benchmark, end to end on real signals
# --------------------------------------------------------------------------- #

def test_ranked_queue_beats_flag_all_on_a_synthetic_dup_flood():
    """End-to-end through duplicate_scores and conformal calibration: plant a
    near-dup flood in random imagery and require the ARM J queue to resolve the
    same poison in strictly fewer review actions than the flag-all pile."""
    rng = np.random.default_rng(42)
    n_clean, n_flood = 240, 48
    base = rng.uniform(0, 1, (n_clean, 24, 24, 3)).astype(np.float32)
    src = rng.choice(n_clean, size=n_flood, replace=False)
    copies = np.clip(base[src] + rng.normal(0, 0.004, base[src].shape), 0, 1).astype(np.float32)
    images = np.concatenate([base, copies], axis=0)
    truth = np.zeros(n_clean + n_flood, bool)
    truth[n_clean:] = True

    dist, nn = duplicate_scores(images, return_neighbors=True)
    ref = duplicate_scores(rng.uniform(0, 1, (400, 24, 24, 3)).astype(np.float32))
    fused = conformal_pvalues(ref, dist, higher_is_more_anomalous=False)
    flags = benjamini_hochberg(fused, 0.05)

    plan = plan_review(fused, COSTS, flagged=flags, neighbor_index=nn,
                       neighbor_dist=dist, cluster_floor_dist=0.01)
    rep = review_budget_report(plan, truth, flags)
    fa = rep["flag_all_curve"]
    arm = rep["arm_j_curve"]
    assert fa["n_flagged"] > 0
    # same catch, fewer actions -- the whole point of the arm
    assert arm["caught"][-1] >= fa["n_poisoned_flagged"]
    assert len(plan.queue) < fa["n_flagged"]
    # and the costed ledger must not be worse than reviewing the whole pile
    led = {l["policy"]: l for l in rep["ledgers"]}
    assert led["arm_j"]["realised_cost"] <= led["flag_all_review"]["realised_cost"]
