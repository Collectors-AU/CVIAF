"""Bounded, seeded evasion search over ACTUAL detector scores.

Each AttackFamily produces a score vector and a declared nonnegative budget for
one replayable attack parameter. The evaluator never relabels clean as poison;
threshold is selected on held-out clean scores and frozen before attack search.
Search samples independent replicate indices; it does not claim a population
minimum outside the finite supplied grid. All score-producing adapters must
retain the attack-success gate, since otherwise a failed implant looks like evasion.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import time

import numpy as np


@dataclass(frozen=True)
class Candidate:
    family: str
    parameter: float
    budget: float


def fixed_threshold(clean_scores, alpha=.05):
    """Strict > cut at the order statistic, conservative on ties."""
    clean = np.asarray(clean_scores, float)
    clean = clean[np.isfinite(clean)]
    if not len(clean) or not 0 < alpha < 1:
        raise ValueError('empty clean calibration or invalid alpha')
    index = max(0, int(math.ceil((1-alpha)*len(clean))) - 1)
    return float(np.sort(clean)[index])


def empirical_tpr(scores, threshold):
    s = np.asarray(scores, float)
    if not len(s):
        raise ValueError('empty attacked set')
    return float(np.mean(np.isfinite(s) & (s > threshold)))


def score_trial(clean, attacked, candidate, rng, count):
    """Explicit score-stream threat model: replace attack alerts with normal traffic.

    dilution: malicious requests camouflaged by selecting independent clean
    outputs; budget = proportion of replaced attacked requests. Not a model
    backdoor and cannot prove robustness against one.
    suppression: attacker caps an output anomaly score; budget = fraction of
    scored attacks whose scores can be overwritten. Requires output control.
    """
    a = np.asarray(attacked, float)
    c = np.asarray(clean, float)
    idx = rng.integers(len(a), size=count)
    out = a[idx].copy()
    mask = rng.random(count) < candidate.parameter
    if candidate.family == 'dilution':
        out[mask] = c[rng.integers(len(c), size=int(mask.sum()))]
    elif candidate.family == 'suppression':
        out[mask] = float(np.nanmin(c))
    else:
        raise ValueError('unsupported score-stream family')
    return out


def successive_halving(clean_by_detector, attacked_by_detector, candidates,
                       seed=7, alpha=.05, target=.5, initial_count=32, rounds=3,
                       retain=.5, trial=score_trial, attack_qualified=False):
    """Successive halving allocates larger independent test samples to survivors.

    Ranks by lower TPR then lower budget. Also runs a full-budget validation of
    EVERY candidate, to avoid claiming a minimum from censored halving rounds.
    This second pass is mandatory for reporting B*; its cost is logged.
    """
    if not 0 < retain < 1 or initial_count < 2 or rounds < 1:
        raise ValueError('invalid search allocation')
    started = time.monotonic()
    grid = list(candidates)
    if not grid or any(not 0 <= x.parameter <= 1 or x.budget < 0 for x in grid):
        raise ValueError('invalid candidates')
    detectors = sorted(set(clean_by_detector) & set(attacked_by_detector))
    if not detectors:
        raise ValueError('no shared detector')
    rows = {}
    for detector in detectors:
        clean = np.asarray(clean_by_detector[detector], float)
        attacked = np.asarray(attacked_by_detector[detector], float)
        cut = fixed_threshold(clean, alpha)
        base_tpr = empirical_tpr(attacked, cut)
        survivors = grid[:]
        evaluations = []
        for r in range(rounds):
            count = initial_count * 2**r
            ranked = []
            for ci, c in enumerate(survivors):
                rng = np.random.default_rng(seed + 100000*r + grid.index(c))
                scores = trial(clean, attacked, c, rng, count)
                tpr = empirical_tpr(scores, cut)
                evaluations.append({'stage': r, 'family': c.family,
                                    'parameter': c.parameter, 'budget': c.budget,
                                    'n': count, 'tpr': tpr})
                ranked.append((tpr, c.budget, ci, c))
            ranked.sort(key=lambda x: x[:3])
            survivors = [v[3] for v in ranked[:max(1, math.ceil(len(ranked)*retain))]]
        # Independent validation on all grid points, so minimum is not inferred
        # from the set of survivors. Freeze calibration threshold throughout.
        validation_count = initial_count * 2**rounds * 8
        curve = []
        for i, c in enumerate(grid):
            rng = np.random.default_rng(seed + 999999 + i)
            tpr = empirical_tpr(trial(clean, attacked, c, rng, validation_count), cut)
            curve.append({'family': c.family, 'parameter': c.parameter,
                          'budget': c.budget, 'tpr': tpr, 'n': validation_count})
        winners = [x for x in curve if x['tpr'] < target]
        best = (min(winners, key=lambda x: (x['budget'], x['tpr']))
                if winners and base_tpr >= target and attack_qualified else None)
        rows[detector] = {'threshold': cut, 'baseline_tpr': base_tpr,
                          'b_star_grid': best, 'curve': curve, 'halving_log': evaluations,
                          'status': ('attack_unqualified' if not attack_qualified else
                                     'already_below_target' if base_tpr < target else
                                     'evasion_found' if best else 'not_found_in_grid'),
                          'valid_only_for': 'finite score-stream attack grid and declared budgets'}
    config = {'seed': seed, 'alpha': alpha, 'target': target,
              'initial_count': initial_count, 'rounds': rounds,
              'grid': [vars(c) for c in grid]}
    return {'config': config, 'search_digest': hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()).hexdigest(),
            'detectors': rows, 'elapsed_seconds': time.monotonic()-started}
