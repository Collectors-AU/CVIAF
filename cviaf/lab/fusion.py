"""Dependence-robust, grouped detector fusion for a family of item hypotheses.

The BY global-null p-value (minimum BY-adjusted ordered p) is super-uniform
under arbitrary dependence when its input p-values are valid. It is applied
within prespecified detector families and again across families; BY is applied
across items separately. Groups must be declared before looking at test scores.
"""
from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np
from cviaf.lab.calibrate import benjamini_yekutieli


def by_global_pvalue(p_values: Sequence[float]) -> float:
    p = np.asarray(p_values, dtype=float).ravel()
    if not p.size or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be finite in [0, 1] and nonempty")
    n = p.size
    harmonic = np.sum(1.0 / np.arange(1, n + 1))
    return float(min(1.0, np.min(np.sort(p) * n * harmonic /
                                  np.arange(1, n + 1))))


def grouped_by_fusion(pvalues: Mapping[str, np.ndarray],
                      groups: Mapping[str, Sequence[str]]) -> tuple[np.ndarray, dict]:
    """Return fused item p-values and auditable family-level evidence.

    Every available detector must occur exactly once. No family may be empty.
    Matrices are expected to have identical item order; a missing detector must
    be omitted from both inputs rather than replaced with a fabricated p-value.
    """
    if not pvalues or not groups:
        raise ValueError("at least one calibrated detector and group are required")
    seen = [name for names in groups.values() for name in names]
    if len(seen) != len(set(seen)) or set(seen) != set(pvalues) or any(not g for g in groups.values()):
        raise ValueError("groups must partition the available detectors exactly once")
    cols = {name: np.asarray(pvalues[name], dtype=float).ravel() for name in seen}
    sizes = {col.size for col in cols.values()}
    if len(sizes) != 1 or not sizes.pop():
        raise ValueError("detector p-value arrays must have the same nonzero length")
    if any(not np.all(np.isfinite(c)) or np.any((c < 0) | (c > 1)) for c in cols.values()):
        raise ValueError("detector p-values must be finite and in [0, 1]")
    family = {group: np.array([by_global_pvalue(row) for row in
                               np.stack([cols[name] for name in names], axis=1)])
              for group, names in groups.items()}
    fused = np.array([by_global_pvalue(row) for row in np.stack(list(family.values()), axis=1)])
    return fused, {"method": "BY_global_null_within_and_between_groups",
                   "dependence": "arbitrary within and between groups",
                   "groups": {g: list(names) for g, names in groups.items()},
                   "group_min_p": {g: float(v.min()) for g, v in family.items()}}


def grouped_by_flags(pvalues: Mapping[str, np.ndarray],
                     groups: Mapping[str, Sequence[str]], alpha: float = .05):
    """Apply BY across item hypotheses to the hierarchical family p-values."""
    fused, evidence = grouped_by_fusion(pvalues, groups)
    return benjamini_yekutieli(fused, alpha), fused, evidence
