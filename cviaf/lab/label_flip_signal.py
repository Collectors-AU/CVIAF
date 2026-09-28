"""Reference-trained box-label disagreement signal for CVIAF lab.

Requires trusted, annotated reference images. Never train this model on the
submission being audited. This is a data-assurance reference, not a test of
the submitted detector and not proof of malicious intent.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cviaf.lab.synth import DetectionDataset, SceneSpec, build_dataset


@dataclass
class LabelReference:
    model: Any
    classes: np.ndarray
    train_digest: str
    calibration_digest: str
    calibration_scores: np.ndarray
    calibration_blocks: int
    n_per_block: int
    reference_contributors: Tuple[str, ...]
    calibration_contributors: Tuple[str, ...]


def _box_feature(image: np.ndarray, box: np.ndarray, side: int = 8) -> np.ndarray:
    """Deterministic ROI RGB grid. Coordinates are xyxy in pixel units."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in box]
    if not np.all(np.isfinite([x0, y0, x1, y1])) or x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid box")
    xs = np.clip(np.floor(x0 + (np.arange(side) + .5) * (x1 - x0) / side).astype(int), 0, w-1)
    ys = np.clip(np.floor(y0 + (np.arange(side) + .5) * (y1 - y0) / side).astype(int), 0, h-1)
    roi = np.asarray(image[ys[:, None], xs[None, :], :3], dtype=np.float64)
    if roi.shape != (side, side, 3) or not np.all(np.isfinite(roi)):
        raise ValueError("Invalid pixels")
    return roi.reshape(-1)


def _training_matrix(ds: DetectionDataset) -> Tuple[np.ndarray, np.ndarray]:
    x: List[np.ndarray] = []
    y: List[int] = []
    for image, boxes, labels in zip(ds.images, ds.boxes, ds.labels):
        if len(boxes) != len(labels):
            raise ValueError("Annotation count mismatch")
        for box, label in zip(boxes, labels):
            x.append(_box_feature(image, box))
            y.append(int(label))
    if not x or len(set(y)) < 2:
        raise ValueError("Trusted reference needs boxes in at least two classes")
    return np.stack(x), np.asarray(y, dtype=np.int64)


def sample_loss_scores(
    model: Any, images: np.ndarray, boxes: Sequence[np.ndarray],
    labels: Sequence[np.ndarray], return_evidence: bool = False,
):
    """Highest box NLL per image; large means the given class is implausible.

    Do not silently score empty/invalid images as clean. No annotation yields NaN;
    caller must mark it unavailable and set its decision p to one. Evidence is
    evidence for review, not a corrected label.
    """
    if len(images) != len(boxes) or len(images) != len(labels):
        raise ValueError("Images/boxes/labels lengths differ")
    classes = {int(c): k for k, c in enumerate(model.classes_)}
    scores = np.full(len(images), np.nan, dtype=np.float64)
    details: List[Dict[str, Any]] = []
    for i, (image, bb, yy) in enumerate(zip(images, boxes, labels)):
        if not len(bb) or len(bb) != len(yy):
            details.append({"status": "unavailable", "reason": "empty or mismatched annotations"})
            continue
        try:
            xx = np.stack([_box_feature(image, box) for box in bb])
            proba = model.predict_proba(xx)
        except ValueError as exc:
            details.append({"status": "unavailable", "reason": str(exc)})
            continue
        losses = [float(-np.log(max(proba[j, classes[int(label)]], 1e-12)))
                  if int(label) in classes else float("nan")
                  for j, label in enumerate(yy)]
        if not np.all(np.isfinite(losses)):
            details.append({"status": "unavailable", "reason": "label absent from trusted reference"})
            continue
        j = int(np.argmax(losses))
        scores[i] = losses[j]
        details.append({"status": "scored", "worst_box_index": j,
                        "given_label": int(yy[j]),
                        "suggested_label": int(model.classes_[np.argmax(proba[j])]),
                        "given_label_probability": float(np.exp(-losses[j])),
                        "n_boxes": len(bb), "loss": scores[i]})
    return (scores, details) if return_evidence else scores


def build_label_reference(
    scene: SceneSpec, n_per_block: int, blocks: int,
    train_images: int = 1200, seed_base: int = 1_300_000,
) -> LabelReference:
    """Lab-only trusted clean training and separate clean calibration blocks.

    In production replace the synthetic datasets with a vetted, same-domain,
    contributor-disjoint train/cal set. Never turn the submitted annotations into
    the clean training or calibration labels. Avoid reference/test duplicate leakage.
    """
    if n_per_block <= 0 or blocks <= 0 or train_images <= 0:
        raise ValueError("Positive image and block counts required")
    train_names = ("trusted_label_train_1", "trusted_label_train_2", "trusted_label_train_3")
    cal_names = ("trusted_label_cal_1", "trusted_label_cal_2", "trusted_label_cal_3")
    train = build_dataset(train_images, scene, contributors=train_names,
                          seed_offset=seed_base)
    x, y = _training_matrix(train)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400,
                           solver="lbfgs", random_state=0))
    model.fit(x, y)
    pool: List[np.ndarray] = []
    digests: List[str] = []
    for b in range(blocks):
        ds = build_dataset(n_per_block, scene, contributors=cal_names,
                           seed_offset=seed_base + 100_000 + b * 9973)
        pool.append(sample_loss_scores(model, ds.images, ds.boxes, ds.labels))
        digests.append(ds.digest())
    pooled = np.concatenate(pool)
    if not np.all(np.isfinite(pooled)):
        raise ValueError("Unscorable clean calibration item: cannot publish null pool")
    digest = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
    return LabelReference(model=model, classes=np.asarray(model.classes_),
                          train_digest=train.digest(), calibration_digest=digest,
                          calibration_scores=pooled, calibration_blocks=blocks,
                          n_per_block=n_per_block,
                          reference_contributors=train_names,
                          calibration_contributors=cal_names)


def calibrated_label_decisions(reference: LabelReference, scores: np.ndarray,
                               alpha: float = .05) -> Dict[str, Any]:
    """Conservative block-calibrated inference with arbitrary-dependence BY.

    Conditional on a *fixed* independently trained model, clean calibration
    blocks and the clean candidate batch must be exchangeable as WHOLE BLOCKS.
    Compare the mean of the top 10% test scores with the same statistic
    across calibration blocks for the asset p. Each image score is compared
    with calibration block maxima, giving a conservative per-item p. This avoids falsely treating adjacent or
    same-source samples as independent calibration items. If block exchangeability
    cannot be justified (different terrain, sensor, source mix, size), abstain.
    """
    from cviaf.lab.calibrate import conformal_pvalues, benjamini_yekutieli
    arr = np.asarray(scores, dtype=np.float64).ravel()
    if arr.size != reference.n_per_block:
        raise ValueError("Candidate size must match each clean calibration block")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    maxima = reference.calibration_scores.reshape(reference.calibration_blocks,
                                                   reference.n_per_block).max(axis=1)
    good = np.isfinite(arr)
    if not good.any():
        return {"status": "unavailable", "reason": "no scorable annotations"}
    item_p = np.ones(arr.size, dtype=np.float64)
    item_p[good] = conformal_pvalues(maxima, arr[good])
    # Predeclared robust batch statistic: mean of top 10% losses. A single
    # naturally difficult clean box cannot by itself dominate the asset decision.
    k = max(1, int(np.ceil(.10 * reference.n_per_block)))
    blocks = reference.calibration_scores.reshape(reference.calibration_blocks,
                                                  reference.n_per_block)
    cal_asset = np.mean(np.sort(blocks, axis=1)[:, -k:], axis=1)
    if good.sum() != arr.size:
        return {"status": "unavailable", "reason": "incomplete candidate annotations",
                "n_unscorable": int((~good).sum())}
    asset_statistic = float(np.mean(np.sort(arr)[-k:]))
    asset_p = float(conformal_pvalues(cal_asset, np.array([asset_statistic]))[0])
    flags = benjamini_yekutieli(item_p, alpha)
    n = arr.size
    harmonic = float(np.sum(1 / np.arange(1, n + 1)))
    first_threshold = alpha / (n * harmonic)
    return {"status": "scored", "item_p": item_p, "item_flags": flags,
            "asset_p": asset_p, "asset_flag": asset_p <= alpha,
            "asset_statistic": asset_statistic, "asset_statistic_name": "mean_top_10_percent_box_loss",
            "n_scorable": int(good.sum()), "n_unscorable": int((~good).sum()),
            "conformal_floor": 1 / (reference.calibration_blocks + 1),
            "by_first_threshold": first_threshold,
            "item_decision_resolution_possible":
                1 / (reference.calibration_blocks + 1) <= first_threshold,
            "calibration_unit": "same-sized independent clean blocks; block maxima",
            "fdr_method": "BY; per-item p conservative under block exchangeability"}


def summarize_label_patterns(details: Sequence[Dict[str, Any]],
                             contributors: Sequence[str],
                             pvalues: np.ndarray) -> Dict[str, Any]:
    """Descriptive contributor x given->suggested table for analyst review.

    Uses p-values as rankings. Counts are *not* calibrated posterior probabilities
    and must not be reported as source guilt or FDR-controlled discoveries.
    """
    if len(details) != len(contributors) or len(details) != len(pvalues):
        raise ValueError("Evidence lengths differ")
    out: Dict[str, Any] = {}
    for i, (row, contributor) in enumerate(zip(details, contributors)):
        key = str(contributor)
        src = out.setdefault(key, {"n_images": 0, "n_scorable": 0,
                                   "directions": {}, "top_review_indices": []})
        src["n_images"] += 1
        if row.get("status") != "scored":
            continue
        src["n_scorable"] += 1
        a, b = int(row["given_label"]), int(row["suggested_label"])
        if a != b:
            direction = f"{a}->{b}"
            src["directions"][direction] = src["directions"].get(direction, 0) + 1
            src["top_review_indices"].append((float(pvalues[i]), i))
    for src in out.values():
        src["top_review_indices"] = [i for _, i in sorted(src["top_review_indices"])[:20]]
    return out
