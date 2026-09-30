"""Reference-trained annotation consistency at the dataset decision level.

This check uses pixels and boxes, never the attack truth or generator's class
colours. Its clean reference must represent the same sensor/domain as the asset.
An absent reference is an abstention, not a clean bill of health.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import numpy as np


def box_features(ds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interior RGB medians for annotated boxes; empty/invalid boxes are skipped."""
    features, labels, sample_ids = [], [], []
    for i, (boxes, labs) in enumerate(zip(ds.boxes, ds.labels)):
        for box, lab in zip(boxes, labs):
            x0, y0, x1, y1 = [float(v) for v in box]
            # Inset avoids box-edge background and antialiasing.
            xa = max(0, int(np.ceil(x0 + .20 * (x1-x0))))
            xb = min(ds.images.shape[2], int(np.floor(x1 - .20 * (x1-x0))))
            ya = max(0, int(np.ceil(y0 + .20 * (y1-y0))))
            yb = min(ds.images.shape[1], int(np.floor(y1 - .20 * (y1-y0))))
            if xa >= xb or ya >= yb:
                continue
            patch = ds.images[i, ya:yb, xa:xb]
            features.append(np.median(patch.reshape(-1, 3), axis=0))
            labels.append(int(lab)); sample_ids.append(i)
    if not features:
        return np.empty((0, 3)), np.empty(0, int), np.empty(0, int)
    return np.asarray(features), np.asarray(labels), np.asarray(sample_ids)


@dataclass
class LabelConsistencyGate:
    """Clean-only fit and asset-level conformal count test.

    Fit uses independent clean reference assets for class centroids and null
    mismatch-count assets. A mismatched box is not automatically an attack:
    domain shift, ambiguous boxes and wrong reference labels can cause it.
    """
    centroids: np.ndarray
    null_counts: np.ndarray
    n_reference: int
    reference_spec: dict
    alpha: float = .05

    @classmethod
    def fit(cls, reference_assets: list, calibration_assets: list, alpha: float = .05):
        if not reference_assets:
            raise ValueError('independent clean reference and clean calibration assets required')
        f, y = [], []
        for asset in reference_assets:
            a, b, _ = box_features(asset)
            f.append(a); y.append(b)
        f = np.concatenate(f); y = np.concatenate(y)
        classes = np.unique(y)
        if classes.size < 2 or not np.array_equal(classes, np.arange(classes[-1]+1)):
            raise ValueError('clean reference must contain contiguous class IDs and at least two classes')
        centroids = np.stack([np.median(f[y == k], axis=0) for k in classes])
        reference_spec = {k: v for k, v in reference_assets[0].spec.items()
                          if k not in ("seed", "seed_offset", "n", "contributors", "contributor_mode")}
        for ds in reference_assets[1:] + calibration_assets:
            spec = {k: v for k, v in ds.spec.items() if k in reference_spec}
            if spec != reference_spec:
                raise ValueError("reference/calibration domain specs differ")
        gate = cls(centroids, np.empty(0, int), len(reference_assets), reference_spec, alpha)
        gate.null_counts = np.asarray([gate.mismatch_count(ds) for ds in calibration_assets])
        if len(gate.null_counts) < int(np.ceil(1 / (alpha / 2))) - 1:
            raise ValueError("too few clean calibration assets for Bonferroni-split alpha/2")
        return gate

    def mismatch_count(self, ds) -> int:
        f, labels, ids = box_features(ds)
        if len(labels) == 0:
            return 0
        valid = (labels >= 0) & (labels < len(self.centroids))
        d = np.sum((f[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2)
        predicted = np.argmin(d, axis=1)
        return int(np.unique(ids[(~valid) | (predicted != labels)]).size)

    def assess(self, ds, alpha: float | None = None) -> Dict[str, Any]:
        spec = {k: v for k, v in ds.spec.items() if k in self.reference_spec}
        if not self.reference_spec or spec != self.reference_spec:
            return {"flagged": False, "abstained": True, "reason":
                    "reference domain mismatch or missing domain metadata"}
        count = self.mismatch_count(ds)
        alpha = self.alpha if alpha is None else alpha
        p = (1 + int(np.sum(self.null_counts >= count))) / (len(self.null_counts) + 1)
        return {'flagged': bool(p <= alpha), 'abstained': False, 'p_value': float(p),
                'mismatch_images': count, 'n_images': len(ds),
                'alpha': alpha, 'n_calibration_assets': len(self.null_counts),
                'calibration_max': int(np.max(self.null_counts)),
                'reference': 'independent clean boxes, matching sensor/domain'}
