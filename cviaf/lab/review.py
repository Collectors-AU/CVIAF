"""
Cost-sensitive abstention and review-queue policy (experiment ARM J).

Why this module exists
----------------------
The data axis today ends in a flag set: an FDR-controlled rejection mask an
analyst is expected to work through. On the measured dup-flood case
(``dup_flood_none_fixed_s5`` in ``runs/mvp``) that mask is 99 items for 48 true
poison -- the 48 poisoned copies plus 51 co-flagged innocent originals, because a
nearest-neighbour duplicate statistic is symmetric: the original is as close to
its copy as the copy is to the original. Flag-all therefore hands a human a pile
whose precision is 0.48, in no order, with no statement of what the pile costs or
what skipping it costs. Two further gaps sit next to that one:

  * there is no disposition between "flag everything" and "accept" -- no
    cost-sensitive abstention, so when the evidence cannot carry a decision the
    system still forces one;
  * nothing ranks the pile, so the first item an analyst opens is drawn from the
    pile at random instead of being the item whose review buys the most.

What the policy is
------------------
Every item gets a posterior probability of being anomalous, estimated from the
same fused conformal p-values the flag set already uses, via a two-groups model
(Storey's pi0 + a histogram local-FDR). The disposition of each item is then the
argmin over a declared, editable loss matrix:

  ``accept``      expected loss ``q * L_false_accept``
  ``review``      cost ``L_review`` (+ residual reviewer error)
  ``quarantine``  expected loss ``(1 - q) * L_false_quarantine``

Items whose cheapest safe action is a human look enter the **review queue**,
ranked by review value -- the expected loss reduction the review buys
(``min(E_accept, E_quarantine) - E_review``), i.e. expected information gain
priced in the operator's own currency. Items near-certain to be poison are
auto-quarantined, items near-certain clean are auto-accepted, and neither spends
analyst time.

Two structural additions fall out of the lab's own measurements:

  * **Co-flag clustering.** In a duplicate flood the poisoned copy and the clean
    original are the same image to every signal we have -- no item-level
    statistic can separate them, and pretending otherwise is exactly how the
    51-item false pile is produced. What resolves the pair is reviewing it *as a
    pair* with its provenance (which copy arrived from which contributor, in
    which batch). The queue therefore operates on co-flag clusters: one review
    task per near-duplicate clique, displaying the members together. The
    review-action count -- not the item count -- is what an analyst spends, and
    the budget curves below are drawn against that unit.
  * **Abstention gate (accept forbidden).** When the calibration floor makes an
    item-level decision impossible, or a declared signal has no measured power at
    the declared prevalence, ``accept`` is removed from the action set. Items
    that would have been accepted are forced into the queue and the reason is
    recorded. A low-power "clean" is never emitted as a clean.

Everything is measured, nothing asserted: ``review_budget_report`` scores the
ranked queue against the flag-all pile (as an *unordered* pile -- its expected
catch at budget B is linear in B, with the best/worst envelope reported) and
books the realised operator cost of each policy against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# operator loss model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OperatorCosts:
    """The operator's declared loss matrix, in one currency (analyst-minutes,
    rupees, whatever DGIS prices in -- the policy only needs the ratios).

    Defaults encode the A4 posture: waving a poisoned item through is the
    catastrophic error (25), removing a good item is costly but recoverable (5),
    and one review action is the unit of analyst time (1). ``reviewer_error`` is
    the probability a human review fails to resolve the item; it keeps the
    review action honest instead of free.
    """

    false_accept: float = 25.0
    false_quarantine: float = 5.0
    review: float = 1.0
    reviewer_error: float = 0.0

    def __post_init__(self) -> None:
        for name in ("false_accept", "false_quarantine", "review"):
            v = float(getattr(self, name))
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"OperatorCosts.{name} must be a non-negative finite number")
        if not (0.0 <= float(self.reviewer_error) < 1.0):
            raise ValueError("OperatorCosts.reviewer_error must be in [0, 1)")

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# two-groups posterior
# --------------------------------------------------------------------------- #

def storey_pi0(p_values: np.ndarray, lam: float = 0.5) -> float:
    """Storey's estimate of the null proportion among p-values.

    ``pi0 = #{p > lam} / ((1 - lam) * n)``, clipped to [0, 1]. ``lam = 0.5``
    trades a little bias for a lot of variance, which is the right side when the
    alternative hypothesis is sparse -- exactly the poison regime.
    """
    p = np.asarray(p_values, np.float64).ravel()
    p = p[np.isfinite(p)]
    n = p.size
    if n == 0:
        return 1.0
    est = float(np.mean(p > lam)) / (1.0 - lam)
    return float(min(1.0, max(0.0, est)))


def item_posteriors(
    p_values: np.ndarray,
    bins: int = 20,
    lam: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """P(item is anomalous | its p-value) under a two-groups model.

    The fused conformal p-values mix a uniform null component (proportion pi0)
    with an alternative component concentrated near 0. The local false-discovery
    rate is ``lfdr(p) = pi0 * f0(p) / f(p)`` with ``f0 = 1``; ``f`` is estimated
    from a histogram of the observed p-values, which is deliberately crude --
    the posterior is a ranking and decision quantity, not a published
    probability, and the histogram estimate is monotone-stable in the regime we
    use it. Returns ``(q, pi0)`` with ``q = 1 - lfdr`` clipped to [0, 1].

    An all-null vector returns q == 0 everywhere (pi0 == 1 forces it), so a
    clean asset is never assigned phantom suspicion.
    """
    p = np.asarray(p_values, np.float64).ravel()
    if p.size == 0:
        return np.zeros(0), 1.0, np.ones(0)
    p = np.clip(np.where(np.isfinite(p), p, 1.0), 0.0, 1.0)
    # Conformal p-values are DISCRETE (ties on shared scores), and a histogram
    # over raw ties reads every tie-bump as alternative density -- on a clean
    # asset that is a phantom queue. Jitter by half the observed granularity
    # (deterministically seeded) so ties spread across neighbouring bins; this
    # is the randomised-p-value remedy for discrete tests, and it leaves the
    # low-p spike region -- where granularity is finest -- untouched.
    uniq = np.unique(p)
    diffs = np.diff(uniq)
    grain = float(diffs[diffs > 0].min()) if (diffs > 0).any() else 1.0
    eps = min(0.5 * grain, 0.01)
    if eps > 0:
        rng = np.random.default_rng(20260928)
        p = np.clip(p + rng.uniform(-eps, eps, p.size), 1e-12, 1.0)
    pi0 = storey_pi0(p, lam)
    # NOTE: pi0 clipping at 1.0 is a bias correction on the TAIL count, not
    # evidence that no alternative exists -- with n ~ 240 items the clip fires
    # often, and a hard zero-out here would accept 20-sigma spikes. The local
    # density estimate below still separates a concentrated spike bin from the
    # uniform background; where the data are genuinely null, lfdr saturates at
    # 1 and q comes out 0 anyway.
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _ = np.histogram(p, bins=edges)
    width = 1.0 / bins
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    # density estimate per bin; an empty bin cannot occur for an observed item
    f_hat = counts[idx] / (p.size * width)
    lfdr = np.clip(pi0 / f_hat, 0.0, 1.0)
    q = np.clip(1.0 - lfdr, 0.0, 1.0)
    return q, pi0, lfdr


# --------------------------------------------------------------------------- #
# cost-sensitive decision
# --------------------------------------------------------------------------- #

def expected_losses(q: np.ndarray, costs: OperatorCosts) -> Dict[str, np.ndarray]:
    """Expected loss of each action for items with anomaly posterior ``q``."""
    q = np.asarray(q, np.float64)
    return {
        "accept": q * costs.false_accept,
        "review": np.full_like(q, costs.review) + costs.reviewer_error * q * costs.false_accept,
        "quarantine": (1.0 - q) * costs.false_quarantine,
    }


def review_value(q: np.ndarray, costs: OperatorCosts) -> np.ndarray:
    """Expected loss reduction a review buys: the value of the information.

    ``min(E_accept, E_quarantine) - E_review``. Positive exactly where review is
    the cost-optimal action; maximal where the evidence is most ambiguous
    *weighted by what the mistake costs* -- which is the operator-costed form of
    expected information gain.
    """
    el = expected_losses(q, costs)
    return np.minimum(el["accept"], el["quarantine"]) - el["review"]


def cluster_expected_losses(
    q_cluster: float,
    size: int,
    costs: OperatorCosts,
) -> Dict[str, float]:
    """Expected loss of each action on one co-flag cluster.

    The cluster-native correction that the 51-co-flag measurement forces: a
    near-duplicate cluster contains its own innocent original with near
    certainty (the duplicate statistic is symmetric -- the original co-flags
    BECAUSE the copy exists), so quarantining a ``size``-s cluster costs at
    least ``size - 1`` clean-item losses when the cluster is a poison pair and
    ``size`` when it is benign. Reviewing the cluster costs one action and
    resolves which member is which via provenance. The consequence is
    measurable: for pairs, review dominates quarantine whenever
    ``false_quarantine > review``, i.e. whenever removing a good item costs
    more than looking at it -- which is the declared posture.
    """
    s = max(int(size), 1)
    poison_members = (s - 1) if s > 1 else 1     # the original is clean
    return {
        "accept": q_cluster * poison_members * costs.false_accept,
        "quarantine": (q_cluster * poison_members
                       + (1.0 - q_cluster) * s) * costs.false_quarantine,
        "review": costs.review + costs.reviewer_error * q_cluster * poison_members
                  * costs.false_accept,
    }


def cluster_review_value(q_cluster: float, size: int, costs: OperatorCosts) -> float:
    """Expected loss reduction from reviewing the cluster -- EIG in the loss currency."""
    el = cluster_expected_losses(q_cluster, size, costs)
    return min(el["accept"], el["quarantine"]) - el["review"]


# --------------------------------------------------------------------------- #
# co-flag clustering
# --------------------------------------------------------------------------- #

def coflag_clusters(
    flagged: np.ndarray,
    neighbor_index: np.ndarray,
    neighbor_dist: np.ndarray,
    floor_dist: float,
) -> List[List[int]]:
    """Group flagged items joined by near-duplicate edges below ``floor_dist``.

    Union-find over the flagged subgraph of the duplicate nearest-neighbour
    graph (edges added in both directions -- a pair is a pair whether the flood
    appended the copy after the original or before). Each cluster is one review
    task: the analyst sees the members side by side with their contributor and
    batch provenance, which is the evidence that separates a copy from its
    original when no item-level statistic can. Singletons are clusters too --
    the queue treats every item uniformly.
    """
    flagged = np.asarray(flagged, bool).ravel()
    nn = np.asarray(neighbor_index, np.int64).ravel()
    dist = np.asarray(neighbor_dist, np.float64).ravel()
    n = flagged.size
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        if not flagged[i]:
            continue
        j = int(nn[i])
        if 0 <= j < n and flagged[j] and np.isfinite(dist[i]) and dist[i] <= floor_dist:
            union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        if flagged[i]:
            groups.setdefault(find(i), []).append(i)
    return sorted((sorted(m) for m in groups.values()), key=lambda m: m)


# --------------------------------------------------------------------------- #
# the review plan
# --------------------------------------------------------------------------- #

@dataclass
class ReviewPlan:
    """The cost-sensitive disposition of one asset's items.

    ``queue`` is the ranked list of review tasks (clusters). ``decision[i]`` is
    the item's auto-disposition -- ``accept`` / ``review`` / ``quarantine`` --
    and an item in the queue always has decision ``review``. ``forced_review``
    counts items pushed into the queue by the abstention gate rather than by
    cost-optimality; the distinction matters because a forced queue entry is a
    power statement, not an evidence statement.
    """

    n_items: int
    pi0: float
    posterior: np.ndarray
    decision: np.ndarray            # object array of "accept"/"review"/"quarantine"
    expected_loss: np.ndarray       # per-item expected loss under the chosen action
    accept_permitted: bool
    abstain_reason: str
    queue: List[Dict[str, Any]]     # ranked clusters
    n_forced_review: int
    costs: OperatorCosts

    def summary(self) -> Dict[str, Any]:
        decisions = self.decision
        return {
            "n_items": int(self.n_items),
            "pi0": round(float(self.pi0), 4),
            "accept_permitted": bool(self.accept_permitted),
            "abstain_reason": self.abstain_reason,
            "n_accept": int(np.sum(decisions == "accept")),
            "n_quarantine": int(np.sum(decisions == "quarantine")),
            "n_review_items": int(np.sum(decisions == "review")),
            "n_review_tasks": int(len(self.queue)),
            "n_forced_review": int(self.n_forced_review),
            "n_clustered_items": int(sum(len(c["members"]) for c in self.queue
                                         if len(c["members"]) > 1)),
            "expected_total_loss": round(float(self.expected_loss.sum()), 3),
            "costs": self.costs.to_dict(),
        }


def plan_review(
    fused_p: np.ndarray,
    costs: OperatorCosts = OperatorCosts(),
    per_signal_p: Optional[Dict[str, np.ndarray]] = None,
    accept_permitted: bool = True,
    abstain_reason: str = "",
    flagged: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    lfdr_max: float = 0.5,
    lfdr_band_p_cap: float = 0.25,
    neighbor_index: Optional[np.ndarray] = None,
    neighbor_dist: Optional[np.ndarray] = None,
    cluster_floor_dist: float = 0.0,
    posterior_bins: int = 20,
) -> ReviewPlan:
    """Build the cost-sensitive disposition and ranked review queue for one asset.

    ``fused_p``           per-item fused conformal p-values (the same numbers the
                          FDR flag set is computed from -- the policy adds a
                          decision layer, not new evidence).
    ``flagged``           the calibrated discovery set the queue is built on.
                          Defaults to Benjamini-Yekutieli at ``alpha`` (valid
                          under arbitrary dependence); the comparison harness
                          passes the same mask it already measures. Membership of
                          the queue is a *calibrated* claim; the costs decide
                          what happens to a member, never whether the evidence
                          was real.
    ``accept_permitted``  the abstention gate. False when the calibration floor
                          or the measured signal power cannot carry an accept:
                          every unflagged item is then forced into the queue
                          (recorded in ``n_forced_review``) and the asset must be
                          worked by a human.
    ``neighbor_index`` / ``neighbor_dist`` / ``cluster_floor_dist``
                          the duplicate signal's nearest-neighbour graph and the
                          distance below which two flagged items are one task.
    """
    from cviaf.lab.calibrate import benjamini_yekutieli

    p = np.asarray(fused_p, np.float64).ravel()
    n = p.size
    q, pi0, lfdr = item_posteriors(p, bins=posterior_bins)
    if flagged is None:
        flagged = benjamini_yekutieli(p, alpha)
    flagged = np.asarray(flagged, bool).ravel()

    # ---- queue membership: three routes in, all recorded
    #   1. the calibrated discovery set (``flagged``) -- the FDR claim;
    #   2. the local-FDR band: items MORE LIKELY anomalous than not
    #      (``lfdr <= lfdr_max``) that the multiplicity price kept out of the
    #      discovery set -- this is where sub-threshold poison is recovered;
    #   3. the abstention gate (handled below).
    # Costs never decide membership: multiplying a loss by a phantom posterior
    # is how a clean asset ends up with a work queue.
    # The band only operates on the LOW-p region: the alternative component of
    # a two-groups mixture concentrates near 0, so a high-p item whose bin has a
    # density bump is a tie artefact, not a suspect.
    member_mask = np.asarray(
        flagged | ((lfdr <= lfdr_max) & (p <= lfdr_band_p_cap)), bool)

    # ---- cluster the membership set on the duplicate nearest-neighbour graph
    if neighbor_index is not None and neighbor_dist is not None and member_mask.any():
        clusters = coflag_clusters(member_mask, neighbor_index, neighbor_dist,
                                   floor_dist=cluster_floor_dist)
    else:
        clusters = [[int(i)] for i in np.where(member_mask)[0]]

    # ---- decide per cluster; items inherit their cluster's disposition
    decision = np.empty(n, dtype=object)
    decision[:] = "accept" if accept_permitted else "review"
    n_forced = 0 if accept_permitted else int(np.sum(~flagged))
    tasks: List[Dict[str, Any]] = []
    for cl in clusters:
        m = np.asarray(cl, np.int64)
        q_c = float(np.max(q[m])) if m.size else 0.0
        el = cluster_expected_losses(q_c, len(cl), costs)
        # review wins exact ties: the informative action is preferred
        act = min(("review", "accept", "quarantine"), key=lambda a: el[a])
        decision[m] = act
        if act != "review":
            continue
        rep = int(m[np.argmin(p[m])])
        tasks.append({
            "members": cl,
            "representative": rep,
            "size": len(cl),
            "posterior_max": q_c,
            "review_value": cluster_review_value(q_c, len(cl), costs),
            "expected_losses": {k: round(v, 4) for k, v in el.items()},
            "signal_p": ({name: [float(pp[i]) for i in cl]
                          for name, pp in per_signal_p.items()}
                         if per_signal_p else {}),
        })

    # The abstention gate's forced reviews are not in any co-flag cluster
    # (clusters cover the flagged discovery set only); queue them as singleton
    # tasks. Their review value is usually negative -- the evidence says clean
    # and the POWER says nothing -- so they sink below every evidence-driven
    # task in the ranking, which is the honest order to work them in.
    if not accept_permitted:
        for i in np.where(~member_mask)[0]:
            tasks.append({
                "members": [int(i)], "representative": int(i), "size": 1,
                "posterior_max": float(q[i]),
                "review_value": cluster_review_value(float(q[i]), 1, costs),
                "expected_losses": {k: round(v, 4) for k, v in
                                    cluster_expected_losses(float(q[i]), 1, costs).items()},
                "forced": True,
                "signal_p": ({name: [float(pp[i])] for name, pp in per_signal_p.items()}
                             if per_signal_p else {}),
            })

    # Rank: review value first (expected information gain priced in the loss
    # currency), then posterior, then cluster size (a bigger cluster resolves
    # more items per action), then the representative's fused p-value, then its
    # index -- total determinism, and within a tied posterior bin the most
    # extreme evidence is worked first.
    tasks.sort(key=lambda t: (-t["review_value"], -t["posterior_max"],
                              -t["size"], float(p[t["representative"]]),
                              t["representative"]))
    for rank, t in enumerate(tasks):
        t["rank"] = rank

    el_items = expected_losses(q, costs)
    chosen_loss = np.array([el_items[d][i] if accept_permitted or member_mask[i]
                            else el_items["review"][i]
                            for i, d in enumerate(decision)], np.float64)

    return ReviewPlan(
        n_items=n, pi0=pi0, posterior=q, decision=decision,
        expected_loss=chosen_loss, accept_permitted=accept_permitted,
        abstain_reason=abstain_reason, queue=tasks,
        n_forced_review=n_forced, costs=costs,
    )


# --------------------------------------------------------------------------- #
# measurement: review-budget curves and cost ledgers
# --------------------------------------------------------------------------- #

def catch_curve(
    queue: Sequence[Dict[str, Any]],
    poisoned: np.ndarray,
    auto_quarantined: np.ndarray,
) -> Dict[str, Any]:
    """Poison resolved vs review actions taken, for the ARM J queue.

    Budget unit: one review *action* (a cluster task; the analyst sees the
    members together). Auto-quarantined poison is resolved at budget 0 -- the
    policy caught it without spending analyst time. Reviewing a cluster resolves
    every member, because the pair is examined as a pair.
    """
    poisoned = np.asarray(poisoned, bool).ravel()
    auto_q = np.asarray(auto_quarantined, bool).ravel()
    caught0 = int(np.sum(auto_q & poisoned))
    budgets = [0]
    caught = [caught0]
    run = caught0
    for t in queue:
        run += int(np.sum(poisoned[np.asarray(t["members"], np.int64)]))
        budgets.append(budgets[-1] + 1)
        caught.append(run)
    return {"budgets": budgets, "caught": caught,
            "caught_at_zero": caught0, "n_poisoned": int(poisoned.sum())}


def flag_all_curve(
    flagged: np.ndarray,
    poisoned: np.ndarray,
) -> Dict[str, Any]:
    """The current behaviour, scored honestly: a flag set with NO ranking.

    An unordered pile catches poison in expectation at rate precision per item
    reviewed; the best and worst orderings bracket what luck could do. None of
    the three is a policy anyone can execute -- they are what 'no ranking'
    means, stated as curves.
    """
    flagged = np.asarray(flagged, bool).ravel()
    poisoned = np.asarray(poisoned, bool).ravel()
    n_flag = int(flagged.sum())
    n_pois_flag = int(np.sum(flagged & poisoned))
    budgets = list(range(n_flag + 1))
    expected = [round(b * n_pois_flag / n_flag, 4) if n_flag else 0 for b in budgets]
    best = [min(b, n_pois_flag) for b in budgets]
    worst = [max(0, b - (n_flag - n_pois_flag)) for b in budgets]
    return {"budgets": budgets, "expected": expected, "best": best, "worst": worst,
            "n_flagged": n_flag, "n_poisoned_flagged": n_pois_flag,
            "n_poisoned": int(poisoned.sum()),
            "precision": round(n_pois_flag / n_flag, 4) if n_flag else None}


def budget_for_catch_fraction(curve: Dict[str, Any], fraction: float = 0.9) -> Optional[int]:
    """Smallest budget whose caught count reaches ``fraction`` of all poison."""
    total = curve.get("n_poisoned", 0)
    if total <= 0:
        return None
    key = "caught" if "caught" in curve else "expected"
    target = fraction * total
    for b, c in zip(curve["budgets"], curve[key]):
        if c >= target - 1e-9:
            return int(b)
    return None


def policy_ledger(
    name: str,
    truth_poisoned: np.ndarray,
    n_review_actions: int,
    auto_quarantined: np.ndarray,
    costs: OperatorCosts,
) -> Dict[str, Any]:
    """Realised operator cost of one policy on one asset, against ground truth.

    The ledger books every error class, not just the convenient ones: poison
    auto-accepted is a false accept at full cost wherever it sits, clean items
    quarantined cost their recoverable loss, and review actions cost their time.
    """
    poisoned = np.asarray(truth_poisoned, bool).ravel()
    auto_q = np.asarray(auto_quarantined, bool).ravel()
    caught = auto_q & poisoned
    quar_clean = auto_q & ~poisoned
    missed = poisoned & ~auto_q   # poison the policy did not resolve
    total = (n_review_actions * costs.review
             + int(quar_clean.sum()) * costs.false_quarantine
             + int(missed.sum()) * costs.false_accept)
    return {
        "policy": name,
        "n_review_actions": int(n_review_actions),
        "poison_caught": int(caught.sum()),
        "clean_quarantined": int(quar_clean.sum()),
        "poison_missed": int(missed.sum()),
        "realised_cost": round(float(total), 3),
        "costs": costs.to_dict(),
    }


def review_budget_report(
    plan: ReviewPlan,
    truth_poisoned: np.ndarray,
    flag_all_mask: np.ndarray,
) -> Dict[str, Any]:
    """The ARM J measurement for one asset: ranked queue vs the flag-all pile.

    Three policies on identical evidence:

    ``flag_all_review``   today's behaviour: the FDR flag set as an unordered
                          pile, every flag reviewed.
    ``flag_all_quarantine``  the cheap escape: quarantine every flag, review
                          nothing. Books its clean-item cost.
    ``arm_j``             the cost-sensitive queue with co-flag clustering.
    """
    poisoned = np.asarray(truth_poisoned, bool).ravel()
    costs = plan.costs
    auto_q = plan.decision == "quarantine"

    arm = catch_curve(plan.queue, poisoned, auto_q)
    fa = flag_all_curve(np.asarray(flag_all_mask, bool), poisoned)

    flag_mask = np.asarray(flag_all_mask, bool).ravel()

    # flag-all + review-everything: every flag costs an action; every flagged
    # poison is resolved; poison the flags never reached is missed at full cost.
    fa_review = policy_ledger(
        "flag_all_review", poisoned,
        n_review_actions=int(flag_mask.sum()),
        auto_quarantined=flag_mask & poisoned, costs=costs)
    # flag-all + quarantine-everything: no actions, flagged poison resolved,
    # flagged clean booked at the recoverable loss.
    fa_quar = policy_ledger("flag_all_quarantine", poisoned, 0, flag_mask, costs)
    # arm_j: reviewed poison is resolved (reviews are paid for), auto-quarantined
    # poison is resolved at budget 0, auto-accepted poison is the false accept.
    reviewed_poison = sum(
        int(np.sum(poisoned[np.asarray(t["members"], np.int64)])) for t in plan.queue)
    arm_ledger = policy_ledger("arm_j", poisoned, len(plan.queue), auto_q, costs=costs)
    arm_ledger["poison_caught"] = int(np.sum(auto_q & poisoned)) + reviewed_poison
    arm_ledger["poison_missed"] = int(np.sum(poisoned & (plan.decision == "accept")))
    arm_ledger["realised_cost"] = round(
        len(plan.queue) * costs.review
        + int(np.sum(auto_q & ~poisoned)) * costs.false_quarantine
        + arm_ledger["poison_missed"] * costs.false_accept, 3)
    ledgers = [fa_review, fa_quar, arm_ledger]

    return {
        "arm_j_curve": arm,
        "flag_all_curve": fa,
        "budget_at_90pct": {
            "arm_j": budget_for_catch_fraction(arm, 0.9),
            "flag_all_expected": budget_for_catch_fraction(fa, 0.9),
        },
        "ledgers": ledgers,
    }
