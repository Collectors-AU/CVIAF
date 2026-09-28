"""
Calibration and decision mathematics.

This module is the reason the framework can say "calibrated confidence" without
it being a slogan. Every threshold in the system comes from here, and every
number it produces has a stated meaning:

  ``conformal_pvalues``  p-values valid under exchangeability of calibration and
                         test items, distribution-free and finite-sample
  ``cauchy_combine``     fusion of many p-values, valid under ARBITRARY dependence
  ``benjamini_yekutieli``FDR control valid under ARBITRARY dependence (conservative)
  ``power_at_alpha``     measured detection power at a target false-positive rate
  ``detection_floor``    the smallest effect size at which we have power -- the
                         number that lets the report say "we cannot exclude X"
                         instead of falsely saying "clean"

Three deliberate choices worth defending:

1. **Conformal, not a tuned threshold.** A threshold picked by grid search on one
   benchmark does not transfer to another model family, sensor, or poison rate, and
   cannot be audited. A conformal p-value carries a validity statement instead.

2. **Cauchy combination, not a weighted average.** Our detectors are strongly
   correlated (spectral and FFT both look at high-frequency structure; per-box
   Mahalanobis and OOD scoring overlap). Averaging p-values double-counts that
   shared evidence. The Cauchy combination test is explicitly robust to arbitrary
   dependence, which removes the need to estimate a covariance matrix from an
   underpowered calibration split -- a matrix we would get wrong.

3. **Benjamini-Yekutieli by default, Benjamini-Hochberg on request.** BH assumes
   positive dependence (PRDS), which we cannot prove for our detector families. BY
   is valid under arbitrary dependence at the cost of a log(n) factor. We take the
   conservative one by default and *record the choice*, because a false discovery
   in a defence pipeline is not a cosmetic problem.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# conformal
# --------------------------------------------------------------------------- #

def conformal_pvalues(
    cal_scores: np.ndarray, test_scores: np.ndarray, higher_is_more_anomalous: bool = True
) -> np.ndarray:
    """Distribution-free conformal p-values.

    ``p(x) = (1 + #{i : r_i >= r(x)}) / (n + 1)``

    Validity requires the calibration items and the test item to be *exchangeable*.
    That assumption is not free, and the design must protect it: calibration
    splits are drawn **contributor-disjoint** from test items (see
    ``poison.make_clean_holdout``), because contributor-level clustering breaks
    exchangeability and silently makes these p-values optimistic.

    Returns p in (0, 1]. Small p means anomalous. Under the null, p is
    super-uniform: P(p <= alpha) <= alpha.
    """
    cal = np.asarray(cal_scores, dtype=np.float64).ravel()
    tst = np.asarray(test_scores, dtype=np.float64).ravel()
    n = cal.size
    if n == 0:
        raise ValueError("calibration set is empty; conformal p-values are undefined")
    if not higher_is_more_anomalous:
        cal = -cal
        tst = -tst
    # rank of each test score among calibration scores (ties count as >=, which is
    # the conservative direction)
    order = np.sort(cal)
    ge = n - np.searchsorted(order, tst, side="left")
    return (1.0 + ge) / (n + 1.0)


def conformal_threshold(
    cal_scores: np.ndarray, alpha: float = 0.05, higher_is_more_anomalous: bool = True
) -> float:
    """Score threshold whose false-positive rate on exchangeable clean data is <= alpha."""
    cal = np.asarray(cal_scores, dtype=np.float64).ravel()
    n = cal.size
    if n == 0:
        raise ValueError("calibration set is empty")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    s = np.sort(cal)
    return float(s[k - 1] if higher_is_more_anomalous else s[n - k])


# --------------------------------------------------------------------------- #
# fusion
# --------------------------------------------------------------------------- #

def cauchy_combine(p_values: Sequence[float] | np.ndarray,
                   weights: Optional[Sequence[float]] = None) -> float:
    """Cauchy combination test. Valid under ARBITRARY dependence between p-values.

    ``T = sum_d w_d * tan((0.5 - p_d) * pi)``, ``p = 0.5 - arctan(T) / pi``.

    Why not Fisher's method: Fisher assumes independence. Our detectors share
    evidence, so an independence-assuming combination overstates significance --
    exactly the failure mode that makes a framework confidently wrong.
    """
    p = np.clip(np.asarray(p_values, dtype=np.float64).ravel(), 1e-12, 1 - 1e-12)
    if p.size == 0:
        return 1.0
    w = np.ones_like(p) if weights is None else np.asarray(weights, np.float64).ravel()
    w = w / w.sum()
    t = float(np.sum(w * np.tan((0.5 - p) * np.pi)))
    return float(0.5 - np.arctan(t) / np.pi)


def combine_matrix(p_matrix: np.ndarray,
                   weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Combine a (n_items, n_detectors) p-value matrix row-wise."""
    P = np.asarray(p_matrix, dtype=np.float64)
    if P.ndim == 1:
        P = P[:, None]
    return np.array([cauchy_combine(P[i], None if weights is None else weights)
                     for i in range(P.shape[0])])


# --------------------------------------------------------------------------- #
# multiple testing
# --------------------------------------------------------------------------- #

def benjamini_yekutieli(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """BY rejection mask: valid under ARBITRARY dependence. Default choice."""
    p = np.asarray(p_values, dtype=np.float64).ravel()
    n = p.size
    if n == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    c_n = float(np.sum(1.0 / np.arange(1, n + 1)))
    thresh = (np.arange(1, n + 1) / (n * c_n)) * alpha
    passed = p[order] <= thresh
    k = int(np.max(np.where(passed)[0]) + 1) if passed.any() else 0
    mask = np.zeros(n, bool)
    if k > 0:
        mask[order[:k]] = True
    return mask


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """BH rejection mask. Valid only under positive dependence (PRDS)."""
    p = np.asarray(p_values, dtype=np.float64).ravel()
    n = p.size
    if n == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    thresh = (np.arange(1, n + 1) / n) * alpha
    passed = p[order] <= thresh
    k = int(np.max(np.where(passed)[0]) + 1) if passed.any() else 0
    mask = np.zeros(n, bool)
    if k > 0:
        mask[order[:k]] = True
    return mask


def q_values(p_values: np.ndarray) -> np.ndarray:
    """BH-adjusted q-values (the smallest FDR at which each item is rejected)."""
    p = np.asarray(p_values, dtype=np.float64).ravel()
    n = p.size
    if n == 0:
        return np.zeros(0)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


# --------------------------------------------------------------------------- #
# detection metrics
# --------------------------------------------------------------------------- #

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC. ``labels`` True = positive (e.g. backdoored)."""
    s = np.asarray(scores, np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, np.float64)
    ranks[order] = np.arange(1, s.size + 1)
    # average ranks for ties
    us = np.unique(s)
    for v in us:
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    tpr = tp / max(int(y.sum()), 1)
    fpr = fp / max(int((~y).sum()), 1)
    return np.concatenate([[0.0], fpr]), np.concatenate([[0.0], tpr])


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, max_fpr: float = 0.05) -> float:
    fpr, tpr = roc_curve(scores, labels)
    ok = fpr <= max_fpr + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def power_at_alpha(scores: np.ndarray, labels: np.ndarray, alpha: float = 0.05) -> float:
    """Sensitivity at a false-positive rate of ``alpha`` (the FPR@95-type metric)."""
    return tpr_at_fpr(scores, labels, max_fpr=alpha)


def f1_at_threshold(scores: np.ndarray, labels: np.ndarray, thr: float) -> float:
    s = np.asarray(scores).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    pred = s >= thr
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return float(2 * prec * rec / max(prec + rec, 1e-9))


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error. Detector confidences are known to be miscalibrated."""
    p = np.clip(np.asarray(probs, np.float64).ravel(), 0, 1)
    y = np.asarray(labels).ravel().astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    n = p.size
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if not m.any():
            continue
        total += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(total)


# --------------------------------------------------------------------------- #
# power curves and detection floors
# --------------------------------------------------------------------------- #

def detection_floor(
    rates: Sequence[float],
    scores_by_rate: Dict[float, Tuple[np.ndarray, np.ndarray]],
    target_power: float = 0.80,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """Smallest effect size at which measured power reaches ``target_power``.

    This is the number that lets the report say, honestly, *"we have no power
    below r, so we cannot call this asset clean."* It is the direct consequence of
    the published impossibility result for universal backdoor detection: when a
    general guarantee is unavailable, the honest deliverable is a measured
    boundary. Every module reports one.
    """
    curve: List[Dict[str, float]] = []
    floor: Optional[float] = None
    for r in sorted(rates):
        if r not in scores_by_rate:
            continue
        s, y = scores_by_rate[r]
        pw = power_at_alpha(np.asarray(s), np.asarray(y), alpha=alpha)
        curve.append({"rate": float(r), "power": float(pw),
                      "auroc": float(auroc(np.asarray(s), np.asarray(y)))})
        if floor is None and pw >= target_power:
            floor = float(r)
    return {"alpha": float(alpha), "target_power": float(target_power),
            "detection_floor": floor, "power_curve": curve,
            "claim": _floor_claim(floor, target_power)}


def _floor_claim(floor: Optional[float], target_power: float) -> str:
    if floor is None:
        return ("No tested effect size reached the target power. The assessment "
                "cannot exclude the attack at any tested rate; report INSUFFICIENT POWER.")
    pct = floor * 100.0
    return (f"Detected with power >= {target_power:.2f} at an effect size of "
            f"{pct:.2f}% or larger. Effect sizes below {pct:.2f}% are NOT excluded "
            f"by this run and must not be reported as 'clean'.")


def reliability_diagram(probs: np.ndarray, labels: np.ndarray, bins: int = 10):
    p = np.clip(np.asarray(probs, np.float64).ravel(), 0, 1)
    y = np.asarray(labels).ravel().astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if m.any():
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                        "mean_confidence": float(p[m].mean()), "empirical_acc": float(y[m].mean())})
    return out
