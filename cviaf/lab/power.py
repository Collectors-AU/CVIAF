"""Power analysis for recall claims: how many attacked arms does a TPR gain need?

Why this module exists
----------------------
The lane's recall benchmark is 48 tamper arms. The 95% interval on "7 of 48
detected" is [0.07, 0.28] -- 21 points wide. And the next compute phase is about to
train an attack corpus on lab machines, at a size nobody has justified. Without
this arithmetic the corpus is sized by habit: someone picks "a few hundred arms",
the resulting interval is still too wide to separate two detectors, and the money is
spent measuring a difference the design could not have seen.

The instrument is an exact binomial one-sample test against a *fixed* baseline
recall. That is the comparison the lab actually runs: the new corpus's recall is
compared with the battery's 0.1458 (7/48). Exact, not normal-approximated, because
the whole point is small n and small p, exactly where the normal approximation is
wrong in the direction that flatters the design.

Three numbers are reported for every cell, because they answer three different
questions and only one of them is usually quoted:

* ``arms_for_gain(p0, delta)`` -- what the next corpus must contain to have an 80%
  chance of *statistically* seeing a gain of ``delta``. The sizing number.
* ``mde(p0, n)`` -- the smallest gain that n arms can detect at 80% power. The
  "what will I actually learn" number, in the other direction.
* ``interval_floor(p0, n)`` -- the smallest gain whose Wilson interval still
  excludes the baseline. This one requires NO power assumption, so it is the
  *smallest* of the three: it says what a difference would have to be to be
  distinguishable in n arms with certainty rather than in 80% of draws. It is the
  honest answer to "can I already see the effect I claim", and any claimed effect
  below it is inside the noise of its own report.

Conventions inherited from the harness: every number is a statement about a
denominator (arms), alpha is one-sided by default (a gain is the direction anyone
funds), and the two-sided variant is reported beside it rather than instead of it.

Direction matters and is not symmetric. With p0 = 0.1458 a *loss* of 5 points is
detectable with fewer arms than a *gain* of 5 points, because the exact rejection
region around a small proportion is left-skewed. A design sized on the gain
therefore does not silently under-power the loss, but a design sized on a symmetric
assumption would.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from cviaf.lab.fpr_tpr import wilson_interval

# The battery's measured recall, and its denominator: every cell is planned against
# a baseline that itself came from 7/48, which is a reminder that the baseline
# carries its own interval [0.07, 0.28].
BATTERY_TPR = 7.0 / 48.0
BATTERY_DENOMINATOR = 48
DEFAULT_ARM_COUNTS = (12, 24, 48, 96, 192, 384, 768, 1536, 3072)
DEFAULT_BASELINES = (0.05, 0.1458, 0.30, 0.50)


# --------------------------------------------------------------------------- #
# exact binomial arithmetic
# --------------------------------------------------------------------------- #
# A naive tail sum per candidate count is O(n^2) per test and made the first
# version of this module take minutes on the sizes the question is about. The
# mass function is built once per (n, p) and turned into cumulative arrays, so a
# region boundary is a searchsorted and a tail is an array lookup.
_PMF_CACHE: Dict[tuple, np.ndarray] = {}


def _pmf(n: int, p: float) -> np.ndarray:
    """pmf of Binomial(n, p) for k = 0..n, normalised to sum to one."""
    key = (int(n), float(p))
    cached = _PMF_CACHE.get(key)
    if cached is not None:
        return cached
    if not 0.0 < p < 1.0:
        arr = np.zeros(n + 1)
        arr[0 if p <= 0.0 else n] = 1.0
        _PMF_CACHE[key] = arr
        return arr
    log_p, log_q = math.log(p), math.log1p(-p)
    log_n_fact = math.lgamma(n + 1)
    arr = np.empty(n + 1)
    for k in range(n + 1):
        arr[k] = math.exp(log_n_fact - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                          + k * log_p + (n - k) * log_q)
    total = arr.sum()
    if total > 0:
        arr /= total          # correct the O(1e-13) drift of many exp/log terms
    if len(_PMF_CACHE) > 64:
        _PMF_CACHE.clear()    # bounded memory: this is a CLI, not a service
    _PMF_CACHE[key] = arr
    return arr


def _tail_ge(n: int, p: float, k: int) -> float:
    """P(X >= k) under Binomial(n, p), exactly (from the cached pmf)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return float(_pmf(n, p)[k:].sum())


def _tail_le(n: int, p: float, k: int) -> float:
    """P(X <= k) under Binomial(n, p), exactly."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return float(_pmf(n, p)[:k + 1].sum())


def critical_region(n: int, p0: float, alpha: float, sided: int = 1,
                    direction: str = "gain") -> Dict[str, Any]:
    """The exact rejection region for a binomial test of p1 vs a FIXED p0.

    ``direction='gain'`` puts the one-sided test in the upper tail (a detector that
    catches more, which is the claim anyone funds). ``direction='loss'`` puts it in
    the lower tail. That is not cosmetic: a one-sided UPWARD test has exactly zero
    power against a decrease, so sizing a corpus with a gain-shaped test and then
    reading a loss off it is reading a number the design cannot produce.
    ``sided=2`` splits alpha between both tails -- the right shape when either
    direction counts, and it costs arm count.
    """
    if sided not in (1, 2):
        raise ValueError("sided must be 1 or 2")
    if direction not in ("gain", "loss"):
        raise ValueError("direction must be 'gain' or 'loss'")
    pmf = _pmf(n, p0)
    threshold = alpha if sided == 1 else alpha / 2.0
    k_hi: Optional[int] = None
    k_lo: Optional[int] = None
    if sided == 2 or direction == "gain":
        suffix = np.cumsum(pmf[::-1])[::-1]    # suffix[k] = P(X >= k), decreasing
        hit = np.nonzero(suffix <= threshold)[0]
        k_hi = int(hit[0]) if hit.size else n + 1
    if sided == 2 or direction == "loss":
        prefix = np.cumsum(pmf)                # prefix[k] = P(X <= k), increasing
        hit_lo = np.nonzero(prefix <= threshold)[0]
        k_lo = int(hit_lo[-1]) if hit_lo.size else -1
    if sided == 1:
        note = (f"rejects when the count reaches {k_hi} of {n} ({k_hi / n:.4f} recall)"
                if direction == "gain" else
                f"rejects when the count falls to {k_lo} of {n} ({k_lo / n:.4f} recall)")
    else:
        note = f"rejects when the count is <= {k_lo} or >= {k_hi} of {n}"
    return {"sided": sided, "k_hi": k_hi, "k_lo": k_lo, "alpha": alpha, "p0": p0,
            "n": n, "direction": direction, "note": note}


def exact_power(n: int, p0: float, p1: float, alpha: float = 0.05, sided: int = 1,
                direction: str = "gain") -> float:
    """P(reject) when the true recall is p1 and the test is run against p0."""
    region = critical_region(n, p0, alpha, sided, direction)
    pmf = _pmf(n, p1)
    hits = 0.0
    k_hi = region["k_hi"]
    if k_hi is not None and k_hi <= n:
        hits += float(pmf[k_hi:].sum())
    k_lo = region["k_lo"]
    if k_lo is not None and k_lo >= 0:
        hits += float(pmf[:k_lo + 1].sum())
    return min(1.0, max(0.0, hits))


def mde(n: int, p0: float, alpha: float = 0.05, power: float = 0.8, sided: int = 1,
        direction: str = "gain", steps: int = 60) -> float:
    """Smallest recall change n arms can detect with the requested power.

    Bisection on the recall, not on a z-score: the exact test's rejection region
    moves in integer counts, so the power is a step function of p1 and the answer
    should be the boundary of the step, not a formula's suggestion of it.
    """
    if direction not in ("gain", "loss"):
        raise ValueError("direction must be 'gain' or 'loss'")
    if p0 <= 0.0 or p0 >= 1.0:
        raise ValueError("baseline recall must be strictly inside (0, 1)")
    lo, hi = 0.0, (1.0 - p0 if direction == "gain" else p0)
    if exact_power(n, p0, p0 + (hi if direction == "gain" else -hi), alpha,
                   sided, direction) < power:
        return float("nan")     # not achievable at any recall; n is too small
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        p1 = p0 + (mid if direction == "gain" else -mid)
        if exact_power(n, p0, p1, alpha, sided, direction) >= power:
            hi = mid
        else:
            lo = mid
    return round(hi, 4)


def arms_needed(p0: float, delta: float, alpha: float = 0.05, power: float = 0.8,
                sided: int = 1, direction: str = "gain", n_max: int = 20000
                ) -> Optional[int]:
    """Smallest arm count whose exact test detects ``delta`` with the requested power."""
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    p1 = p0 + (delta if direction == "gain" else -delta)
    if not 0.0 < p1 < 1.0:
        return None
    if exact_power(n_max, p0, p1, alpha, sided, direction) < power:
        return None             # not achievable even at n_max arms
    lo, hi = 1, n_max
    while lo < hi:
        mid = (lo + hi) // 2
        if exact_power(mid, p0, p1, alpha, sided, direction) >= power:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


def interval_floor(n: int, p0: float, alpha: float = 0.05, direction: str = "gain",
                   steps: int = 40) -> float:
    """Smallest change whose Wilson interval at n arms excludes the baseline.

    No power assumption: if the observed recall is p0 + delta on n arms, this is the
    smallest delta whose interval still says "not the baseline". Below it, a reported
    difference is inside the noise of its own report.
    """
    def clears(delta: float) -> bool:
        p1 = p0 + (delta if direction == "gain" else -delta)
        lo, hi = wilson_interval(round(p1 * n), n)
        return lo > p0 if direction == "gain" else hi < p0

    lo, hi = 0.0, (1.0 - p0 if direction == "gain" else p0)
    if not clears(hi):
        return float("nan")
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        if clears(mid):
            hi = mid
        else:
            lo = mid
    return round(hi, 4)


def mcnemar_arms(delta: float, discordance: float, alpha: float = 0.05,
                 power: float = 0.8, sided: int = 2) -> Dict[str, Any]:
    """Arms needed to separate TWO RULES on the SAME arms (paired design).

    Comparing a fused rule against a single signal is not a two-sample problem: the
    same arms are scored by both rules, so what decides whether the difference is
    visible is the *discordance* -- the share of arms the two rules disagree about.
    An unpaired calculation answers a question nobody asked (and overstates the
    corpus by a factor that grows as the rules agree).

    Normal-approximation form, ``n ~ (z_{1-a/s} + z_{1-b})^2 * psi / delta^2`` for a
    two-sided exact McNemar test, reported with the approximation named rather than
    hidden: at these counts the exact version differs by a few percent either way,
    and the point of the table is the order of magnitude of the corpus.
    """
    if delta <= 0.0 or not 0.0 < discordance < 1.0:
        raise ValueError("delta > 0 and 0 < discordance < 1 are required")
    z_alpha = 1.959963984540054 if sided == 2 else 1.6448536269514722
    z_beta = 0.8416212335729143
    n = ((z_alpha + z_beta) ** 2) * discordance / (delta ** 2)
    return {"n_arms": int(math.ceil(n)), "delta": delta,
            "discordance": discordance, "alpha": alpha, "power": power,
            "sided": sided, "method": "normal approximation to exact McNemar",
            "note": ("discordance is the share of arms the two rules classify "
                     "differently; it is NOT known for recall until a corpus with "
                     "attacked arms exists, so the table is reported over a range")}


def power_table(arm_counts: Sequence[int] = DEFAULT_ARM_COUNTS,
                baselines: Sequence[float] = DEFAULT_BASELINES,
                alpha: float = 0.05, power: float = 0.8) -> Dict[str, Any]:
    """The sizing table: arms x baseline -> detectable recall change."""
    rows: List[Dict[str, Any]] = []
    for n in arm_counts:
        row: Dict[str, Any] = {"arms": int(n)}
        for p0 in baselines:
            key = f"{p0:g}"
            row[key] = {
                "mde_gain_onesided": mde(n, p0, alpha, power, 1, "gain"),
                "mde_gain_twosided": mde(n, p0, alpha, power, 2, "gain"),
                "mde_loss_onesided": mde(n, p0, alpha, power, 1, "loss"),
                "interval_floor_gain": interval_floor(n, p0, alpha, "gain"),
                "interval_floor_loss": interval_floor(n, p0, alpha, "loss"),
            }
        rows.append(row)
    return {"schema": "cviaf.power.v1",
            "alpha": alpha, "power": power,
            "arm_counts": [int(n) for n in arm_counts],
            "baselines": list(baselines),
            "method": ("exact binomial test of the observed recall against a FIXED "
                       "baseline; the rejection region is an integer count, and the "
                       "interval floor is the Wilson interval at the same n"),
            "caveat": (f"the baseline is estimated from {BATTERY_DENOMINATOR} arms "
                       f"({BATTERY_TPR:.4f}); treating it as known makes these numbers "
                       f"optimistic. If the corpus must also re-estimate the baseline, "
                       f"double the arm count."),
            "table": rows}


def delta_table(delta: float = 0.05, baselines: Sequence[float] = DEFAULT_BASELINES,
                alpha: float = 0.05, power: float = 0.8,
                n_max: int = 20000) -> Dict[str, Any]:
    """Arms needed for one declared change -- the '5-point gain' question."""
    out: Dict[str, Any] = {"delta": delta, "alpha": alpha, "power": power,
                           "n_max_searched": n_max, "per_baseline": {}}
    for p0 in baselines:
        loss = arms_needed(p0, delta, alpha, power, 1, "loss", n_max)
        if loss is None and p0 - delta <= 0.0:
            # Not "unachievable": undefined. A loss of delta from a baseline of p0
            # needs p0 - delta > 0 to be a recall at all, and saying "not
            # achievable" would blame the design for an arithmetic impossibility.
            loss = (f"undefined: a {delta:g} loss from a {p0:g} baseline is not a "
                    f"recall")
        out["per_baseline"][f"{p0:g}"] = {
            "gain_onesided": arms_needed(p0, delta, alpha, power, 1, "gain", n_max),
            "gain_twosided": arms_needed(p0, delta, alpha, power, 2, "gain", n_max),
            "loss_onesided": loss,
        }
        n_gain = out["per_baseline"][f"{p0:g}"]["gain_onesided"]
        out["per_baseline"][f"{p0:g}"]["gain_onesided_reject_at"] = (
            None if n_gain is None else
            int(min(n_gain, math.ceil((p0 + delta) * n_gain))))
    out["none_found_is_an_answer"] = (
        "null means no arm count up to n_max_searched reaches the requested power: "
        "the change is smaller than any design in this range can resolve")
    return out


def paired_table(deltas: Sequence[float] = (0.02, 0.05, 0.10),
                 discordances: Sequence[float] = (0.05, 0.10, 0.20, 0.40),
                 alpha: float = 0.05, power: float = 0.8) -> Dict[str, Any]:
    return {"method": "paired (McNemar) — two rules scored on the same arms",
            "deltas": list(deltas), "discordances": list(discordances),
            "cells": {f"{d:g}": {f"{psi:g}": mcnemar_arms(d, psi, alpha, power, 2)
                                 for psi in discordances} for d in deltas}}


def render(report: Dict[str, Any]) -> str:
    lines = [f"power  alpha={report['alpha']:.3g}  target power={report['power']:.2f}  "
             f"method=exact binomial vs a fixed baseline", ""]
    lines.append("arms x baseline recall -> smallest detectable GAIN (one-sided), "
                 "and the Wilson interval floor")
    hdr = f"{'arms':>6s} " + " ".join(f"{('p0=' + f'{p0:g}'):>26s}"
                                      for p0 in report["baselines"])
    lines += [hdr, "-" * len(hdr)]
    for row in report["table"]:
        cells = []
        for p0 in report["baselines"]:
            cell = row[f"{p0:g}"]
            cells.append(f"{cell['mde_gain_onesided']:>11.4f} "
                         f"{cell['interval_floor_gain']:>11.4f}")
        lines.append(f"{row['arms']:>6d} " + " ".join(cells))
    lines.append("  (left column per baseline = exact-test MDE at 80% power, gain-"
                 "shaped upper-tail test; right = interval floor, which needs no "
                 "power assumption)")
    lines.append("  a one-sided UPWARD test has zero power against a decrease, so a "
                 "loss is sized with its own lower-tail table rather than read off "
                 "this one")
    lines += ["", f"per baseline: the same table twice-sided and for a LOSS"]
    for row in report["table"]:
        for p0 in report["baselines"]:
            cell = row[f"{p0:g}"]
            lines.append(f"  arms={row['arms']:>5d} p0={p0:<7g} "
                         f"gain(1s)={cell['mde_gain_onesided']:.4f} "
                         f"gain(2s)={cell['mde_gain_twosided']:.4f} "
                         f"loss(1s)={cell['mde_loss_onesided']:.4f}")
    lines += ["", report["caveat"]]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="arm-count power analysis for recall claims")
    ap.add_argument("--out", default=None, help="write the power report JSON here")
    ap.add_argument("--markdown", default=None, help="write the tables as markdown")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.8)
    ap.add_argument("--delta", type=float, default=0.05,
                    help="declared change to size the corpus for")
    ap.add_argument("--arms", type=int, nargs="+", default=list(DEFAULT_ARM_COUNTS))
    ap.add_argument("--baselines", type=float, nargs="+", default=list(DEFAULT_BASELINES))
    args = ap.parse_args(argv)

    report = power_table(args.arms, args.baselines, args.alpha, args.power)
    report["declared_delta"] = delta_table(args.delta, args.baselines, args.alpha,
                                           args.power)
    report["paired"] = paired_table(alpha=args.alpha, power=args.power)
    print(render(report))
    print()
    print(render_delta(report["declared_delta"]))
    print()
    print(render_paired(report["paired"]))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, default=str)
        print(f"\nwritten to {args.out}")
    return 0


def render_delta(report: Dict[str, Any]) -> str:
    lines = [f"arms needed for a declared change of {report['delta']:g} "
             f"(exact, power {report['power']:.2f})",
             f"{'baseline':>10s} {'gain 1-sided':>13s} {'gain 2-sided':>13s} "
             f"{'loss 1-sided':>13s}"]
    for p0, cell in report["per_baseline"].items():
        def fmt(v):
            return "not achievable at n_max" if v is None else str(v)
        lines.append(f"{p0:>10s} {fmt(cell['gain_onesided']):>13s} "
                     f"{fmt(cell['gain_twosided']):>13s} "
                     f"{fmt(cell['loss_onesided']):>13s}")
    return "\n".join(lines)


def render_paired(report: Dict[str, Any]) -> str:
    lines = [f"paired arms needed: {report['method']}",
             f"{'delta':>7s} " + " ".join(f"{('psi=' + f'{p:g}'):>9s}"
                                          for p in report["discordances"])]
    for d, cells in report["cells"].items():
        lines.append(f"{d:>7s} " + " ".join(
            f"{cells[f'{p:g}']['n_arms']:>9d}" for p in report["discordances"]))
    lines.append("  psi = share of arms the two rules classify differently; unknown "
                 "for recall until an attacked corpus exists")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
