"""Drift harness: the metric definitions, fixed as data, plus a runnable skeleton.

``runs/drift_cells`` already builds 11 cells -- one reference domain, five declared
shifts, and a no-shift resample control for each -- and ``cviaf/lab/driftbench.py``
measures attribution on scenarios it generates in memory. What was missing is the
piece between them: a harness that consumes the *built* cells, states exactly what
each drift metric is, and refuses to call something drift when the number it is
reading cannot tell a declared axis from ordinary resampling noise.

So the metric definitions live here as data, not as prose in a report:

    name, unit, direction, formula, minimum sample size, what it can support and
    what it cannot.

and every metric gets its p-value from one shared permutation scheme with one seed,
so two metrics' p-values in the same report are comparable. The decision rule is
fixed and narrow:

    drift            p <= alpha AND value > control_value
    no_drift         p >  alpha AND value <= control_value
    under_determined otherwise -- including "the metric fires, but it fires harder
                     on the no-shift control, so it is not reading the declared axes"

The control value is the same metric computed on the cell's declared no-shift
resample against the same reference. Without that contrast an MMD p-value on
n=240 measures "these are two different samples", which for image data is always
true.

This is a skeleton and says so in its output. It fixes the definitions and the
contrast; it does not yet have a calibrated null across many seeds, so it makes no
false-positive claim, and it never attributes a shift to manipulation -- a poisoning
shift and a seasonal shift are both "a shift", and separating them needs the
attribution arbiter (``driftbench``) plus a manipulation signal this harness does
not receive. Every cell therefore declares ``under_determined`` attribution.

CLI
---
    python -m cviaf.lab.drift_harness --corpus runs/drift_cells \\
        [--json out.json] [--max-images 160] [--permutations 200] [--alpha 0.05]

Exit: 0 report written, 2 corpus unusable, 4 nothing measurable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

CELL_SCHEMA = "cviaf.drift-cells/1"
HARNESS_SCHEMA = "cviaf.drift-harness.v1"
DEFAULT_SEED = 20260928

# --------------------------------------------------------------------------- #
# metric definitions, as data
# --------------------------------------------------------------------------- #

METRICS: Dict[str, Dict[str, Any]] = {
    "mmd2_rbf": {
        "unit": "squared distance in RBF reproducing-kernel Hilbert space",
        "direction": "higher_is_more_shift",
        "formula": "unbiased MMD^2 with RBF kernel, bandwidth = median pairwise "
                   "distance of the pooled sample",
        "minimum_n": 50,
        "p_value": "label-permutation over the pooled reference and operational "
                   "features, shared scheme",
        "supports": ["a difference in distribution between two image sets"],
        "cannot": ["the cause of the difference", "which feature moved",
                   "a threshold that generalises across corpora: MMD^2 has no units"],
    },
    "wasserstein_1_mean": {
        "unit": "mean pixel-statistic units, averaged over feature dimensions",
        "direction": "higher_is_more_shift",
        "formula": "mean over feature dimensions of the 1-Wasserstein distance "
                   "between sorted samples (quantile interpolation)",
        "minimum_n": 30,
        "p_value": "same permutation scheme",
        "supports": ["how far the marginal distributions moved, in feature units"],
        "cannot": ["correlation or joint-structure changes",
                   "compare across feature extractors"],
    },
    "ks_max": {
        "unit": "supremum of |F_ref - F_op| over one feature dimension",
        "direction": "higher_is_more_shift",
        "formula": "max over feature dimensions of the two-sample Kolmogorov-Smirnov "
                   "statistic",
        "minimum_n": 30,
        "p_value": "same permutation scheme",
        "supports": ["that some single dimension moved, and by how much"],
        "cannot": ["that the moved dimension is the declared axis -- brightness, "
                   "blur and terrain all move pixel statistics"],
    },
    "feature_mean_z": {
        "unit": "pooled standard deviations",
        "direction": "higher_is_more_shift",
        "formula": "max over feature dimensions of |mean_ref - mean_op| divided by "
                   "the pooled standard deviation",
        "minimum_n": 30,
        "p_value": "same permutation scheme",
        "supports": ["a scale-free mean shift per dimension"],
        "cannot": ["variance or shape changes", "multi-modal or small shifts"],
    },
}

DECISIONS = ("drift", "no_drift", "under_determined", "insufficient_n")

NULL_DEFINITION = (
    "noise_floor is the larger of two nulls measured with the SAME metric: the "
    "control_value (the cell's declared no-shift resample against the same reference, "
    "role no_shift_control) and the self_null (the reference split in half against "
    "itself, which is the metric's own noise scale at this sample size). A shift is "
    "called drift only when it clears alpha AND exceeds the noise floor; otherwise the "
    "decision is under_determined. The second null matters: on rendered scenes, resample "
    "noise and the declared axis are often within a fraction of a percent of each other, "
    "and against the control alone a difference of that size looks like a finding."
)


# --------------------------------------------------------------------------- #
# cell manifests
# --------------------------------------------------------------------------- #

CELL_REQUIRED = ("schema", "cell_id", "role", "scene_spec", "declared_axes_moved",
                 "n_images", "seed_offset", "contributors", "contributor_counts",
                 "image_shape", "digests", "object_count", "class_counts")
AXES = ("terrain", "season", "illumination", "gamma", "sensor_noise", "sensor_blur",
        "objects_per_image", "seed")


def validate_drift_cell(cell: Dict[str, Any], cell_ids: Optional[Sequence[str]] = None,
                        reference: Optional[Dict[str, Any]] = None,
                        pair: Optional[Dict[str, Any]] = None) -> List[str]:
    """Problems with one ``cviaf.drift-cells/1`` descriptor. Never raises.

    A control cell is a resample of its PAIRED cell's scene, so it is compared to
    that pair, not to the reference domain; a shifted cell is compared to the
    reference domain. Checking both against the former was the first thing this
    validator got wrong, and it flagged every legitimate control in the corpus.
    """
    problems: List[str] = []
    if not isinstance(cell, dict):
        return ["cell is not a JSON object"]
    name = cell.get("cell_id") if isinstance(cell.get("cell_id"), str) else "<unnamed>"
    for key in CELL_REQUIRED:
        if key not in cell:
            problems.append(f"{name}: missing required key {key!r}")
    if cell.get("schema") != CELL_SCHEMA:
        problems.append(f"{name}: schema {cell.get('schema')!r} is not {CELL_SCHEMA!r}")
    if cell.get("role") not in ("reference", "shifted", "no_shift_control"):
        problems.append(f"{name}: unknown role {cell.get('role')!r}")

    scene = cell.get("scene_spec")
    moved = cell.get("declared_axes_moved")
    if moved is not None and not isinstance(moved, dict):
        problems.append(f"{name}: declared_axes_moved must be an object")
    elif isinstance(moved, dict):
        for axis, value in moved.items():
            if axis not in AXES:
                problems.append(f"{name}: declared axis {axis!r} is not a scene axis "
                                f"({sorted(AXES)})")
                continue
            if not isinstance(scene, dict) or scene.get(axis) != value:
                problems.append(
                    f"{name}: declared_axes_moved[{axis!r}] = {value!r} but "
                    f"scene_spec says {scene.get(axis) if isinstance(scene, dict) else None!r}")
        if cell.get("role") == "shifted" and not moved:
            problems.append(f"{name}: role is 'shifted' but no axis is declared as moved")
        if cell.get("role") == "no_shift_control" and moved:
            problems.append(f"{name}: role is 'no_shift_control' but axes are declared "
                            f"moved ({sorted(moved)}); a control that moves is not a control")
        if isinstance(scene, dict) and isinstance(moved, dict):
            if cell.get("role") == "shifted" and reference is not None:
                ref_scene = reference.get("scene_spec") or {}
                for axis in AXES:
                    if axis in moved:
                        continue
                    if axis in ref_scene and scene.get(axis) != ref_scene[axis]:
                        problems.append(
                            f"{name}: scene_spec[{axis!r}] = {scene.get(axis)!r} differs "
                            f"from the reference domain ({ref_scene[axis]!r}) but is not "
                            f"declared in declared_axes_moved")
            elif cell.get("role") == "no_shift_control":
                if not isinstance(pair, dict):
                    problems.append(f"{name}: a control must name the cell it resamples "
                                    f"in paired_cell")
                else:
                    pair_scene = pair.get("scene_spec") or {}
                    for axis in AXES:
                        if axis in pair_scene and scene.get(axis) != pair_scene[axis]:
                            problems.append(
                                f"{name}: scene_spec[{axis!r}] = {scene.get(axis)!r} "
                                f"differs from its paired cell {pair.get('cell_id')!r} "
                                f"({pair_scene[axis]!r}); a control that does not "
                                f"resample its pair's scene is not a control")

    counts = cell.get("contributor_counts")
    if isinstance(counts, dict):
        total = sum(v for v in counts.values() if isinstance(v, (int, float)))
        if cell.get("n_images") is not None and total != cell.get("n_images"):
            problems.append(f"{name}: contributor_counts sum to {total}, n_images is "
                            f"{cell.get('n_images')}")
        contributors = cell.get("contributors")
        if isinstance(contributors, list) and sorted(counts) != sorted(contributors):
            problems.append(f"{name}: contributor_counts keys {sorted(counts)} do not "
                            f"match contributors {sorted(contributors)}")

    shape = cell.get("image_shape")
    if isinstance(shape, (list, tuple)):
        if len(shape) != 4:
            problems.append(f"{name}: image_shape must be [n, h, w, c]")
        elif cell.get("n_images") is not None and shape[0] != cell.get("n_images"):
            problems.append(f"{name}: image_shape[0] = {shape[0]} but n_images is "
                            f"{cell.get('n_images')}")

    digests = cell.get("digests")
    if isinstance(digests, dict):
        for key in ("dataset", "pixels_uint8_sha256"):
            value = digests.get(key)
            if not isinstance(value, str) or len(value) != 64:
                problems.append(f"{name}: digests[{key!r}] must be a 64-char sha256")
    elif digests is not None:
        problems.append(f"{name}: digests must be an object")

    pair = cell.get("paired_cell")
    if pair is not None and cell_ids is not None and pair not in cell_ids:
        problems.append(f"{name}: paired_cell {pair!r} is not among the cells")
    return problems


def load_cells(corpus: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read and validate ``index.json``. Returns (cells, problems)."""
    path = os.path.join(corpus, "index.json")
    if not os.path.isfile(path):
        return [], [f"{path}: no cell index"]
    with open(path, encoding="utf-8") as fh:
        index = json.load(fh)
    cells = index.get("cells")
    if not isinstance(cells, list) or not cells:
        return [], [f"{path}: index has no cells"]
    ids = [c.get("cell_id") for c in cells if isinstance(c, dict)]
    by_id = {c.get("cell_id"): c for c in cells if isinstance(c, dict)}
    reference = next((c for c in cells if isinstance(c, dict)
                      and c.get("role") == "reference"), None)
    problems: List[str] = []
    if reference is None:
        problems.append(f"{path}: no cell has role 'reference'")
    for cell in cells:
        pair = by_id.get(cell.get("paired_cell")) if isinstance(cell, dict) else None
        problems += validate_drift_cell(cell, cell_ids=ids, reference=reference, pair=pair)
    return cells, problems


def cell_features(corpus: str, cell: Dict[str, Any], max_images: int = 160,
                  method: str = "pixel_stats", target_dim: int = 64) -> np.ndarray:
    """Load a cell's stored images and reduce them to feature vectors."""
    from cviaf.utils import extract_features_from_images
    path = os.path.join(corpus, cell["cell_id"], "images_uint8.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path}: cell images are missing")
    with np.load(path) as z:
        key = "images" if "images" in z.files else z.files[0]
        images = np.asarray(z[key])
    images = images[:max_images].astype(np.float32) / 255.0
    return np.asarray(extract_features_from_images(images, method=method,
                                                   target_dim=target_dim), dtype=np.float64)


# --------------------------------------------------------------------------- #
# metrics and the shared permutation scheme
# --------------------------------------------------------------------------- #

def _median_bandwidth(pooled: np.ndarray) -> float:
    n = len(pooled)
    idx = np.linspace(0, n - 1, min(n, 200)).astype(int)
    sub = pooled[idx]
    d = np.sqrt(np.maximum(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1), 0.0))
    med = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    return med if med > 0 else 1.0


def _mmd2(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    def k(A, B):
        return np.exp(-((A[:, None, :] - B[None, :, :]) ** 2).sum(-1) / (2 * sigma ** 2))
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return 0.0
    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    return float((Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
                 + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
                 - 2 * Kxy.sum() / (n * m))


def _wasserstein_1(a: np.ndarray, b: np.ndarray) -> float:
    """Exact 1-Wasserstein between two 1-D samples via quantile interpolation."""
    qs = np.linspace(0, 1, 101)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def _ks_2samp(a: np.ndarray, b: np.ndarray) -> float:
    grid = np.sort(np.unique(np.concatenate([a, b])))
    fa = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def _ks_max(X: np.ndarray, Y: np.ndarray) -> float:
    return max(_ks_2samp(X[:, j], Y[:, j]) for j in range(X.shape[1]))


def _feature_mean_z(X: np.ndarray, Y: np.ndarray) -> float:
    mx, my = X.mean(0), Y.mean(0)
    vx, vy = X.var(0, ddof=1), Y.var(0, ddof=1)
    pooled = np.sqrt(np.maximum((vx + vy) / 2.0, 1e-12))
    return float(np.max(np.abs(mx - my) / pooled))


def _statistics(X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None) -> Dict[str, float]:
    sigma = sigma if sigma is not None else _median_bandwidth(np.vstack([X, Y]))
    wass = float(np.mean([_wasserstein_1(X[:, j], Y[:, j]) for j in range(X.shape[1])]))
    return {"mmd2_rbf": _mmd2(X, Y, sigma), "wasserstein_1_mean": wass,
            "ks_max": _ks_max(X, Y), "feature_mean_z": _feature_mean_z(X, Y)}


def permutation_pvalues(X: np.ndarray, Y: np.ndarray, observed: Dict[str, float],
                        n_permutations: int = 200, seed: int = DEFAULT_SEED,
                        ) -> Dict[str, float]:
    """One permutation scheme, one seed, for every metric in the same report.

    The rng is seeded from the seed AND the two sample sizes, so two contrasts of
    different sizes cannot share a stream by accident, and re-running a report
    reproduces its p-values exactly.
    """
    rng = np.random.default_rng([int(seed), len(X), len(Y), X.shape[1]])
    pooled = np.vstack([X, Y])
    n = len(X)
    sigma = _median_bandwidth(pooled)
    counts = {name: 0 for name in observed}
    for _ in range(n_permutations):
        perm = rng.permutation(len(pooled))
        stats = _statistics(pooled[perm[:n]], pooled[perm[n:]], sigma=sigma)
        for name in observed:
            if stats[name] >= observed[name]:
                counts[name] += 1
    return {name: (1 + counts[name]) / (1 + n_permutations) for name in observed}


def split_half(reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic half-split of the reference, for the metric's own noise scale."""
    n = len(reference)
    return reference[: n // 2], reference[n // 2: 2 * (n // 2)]


def evaluate_contrast(reference: np.ndarray, operational: np.ndarray,
                      control: Optional[np.ndarray] = None, alpha: float = 0.05,
                      n_permutations: int = 200, seed: int = DEFAULT_SEED,
                      metrics: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Per-metric value, p-value and decision for one cell against the reference.

    Two nulls are measured with the same metric: the cell's declared no-shift resample
    (``control``) and the reference against itself in halves (``self_null``, the
    metric's noise scale at this n). The larger is the noise floor a shift has to
    clear. Without the control the decision is ``under_determined`` whatever the
    p-value says, because on image data a distribution test is always significant.
    """
    result: Dict[str, Any] = {"n_reference": int(len(reference)),
                              "n_operational": int(len(operational)),
                              "n_control": int(len(control)) if control is not None else 0,
                              "metrics": {}}
    observed = _statistics(reference, operational)
    pvalues = permutation_pvalues(reference, operational, observed,
                                  n_permutations=n_permutations, seed=seed)
    control_values: Dict[str, float] = {}
    if control is not None and len(control) >= 2:
        control_values = _statistics(reference, control)
    a, b = split_half(reference)
    self_null = _statistics(a, b) if len(a) >= 2 and len(b) >= 2 else {}

    for name, definition in METRICS.items():
        if metrics is not None and name not in metrics:
            continue
        value = observed[name]
        p = pvalues[name]
        floor_parts = [v for v in (control_values.get(name), self_null.get(name))
                       if v is not None]
        floor = max(floor_parts) if floor_parts else None
        entry: Dict[str, Any] = {
            "value": value, "p_value": p, "direction": definition["direction"],
            "control_value": control_values.get(name),
            "self_null_value": self_null.get(name), "noise_floor": floor,
            "unit": definition["unit"], "minimum_n": definition["minimum_n"],
        }
        min_n = definition["minimum_n"]
        if min(len(reference), len(operational)) < min_n:
            entry["decision"] = "insufficient_n"
            entry["reason"] = (f"smallest sample has {min(len(reference), len(operational))} "
                               f"items, metric requires {min_n}")
        elif control_values.get(name) is None:
            entry["decision"] = "under_determined"
            entry["reason"] = ("no no-shift control for this cell, so a significant "
                               "difference cannot be attributed to the declared axis")
        elif p <= alpha and value > floor:
            margin = (value - floor) / floor if floor else float("inf")
            entry["margin_over_floor"] = margin
            entry["decision"] = "drift"
            entry["reason"] = (f"p={p:.4g} <= {alpha} and value {value:.4g} exceeds the "
                               f"noise floor {floor:.4g} (control "
                               f"{control_values[name]:.4g}, self-null "
                               f"{self_null.get(name, float('nan')):.4g}) by "
                               f"{margin * 100:.2f}%")
        elif p > alpha and value <= floor:
            entry["decision"] = "no_drift"
            entry["reason"] = (f"p={p:.4g} > {alpha} and value {value:.4g} does not "
                               f"exceed the noise floor {floor:.4g}")
        else:
            entry["decision"] = "under_determined"
            entry["reason"] = (
                f"contradictory evidence: p={p:.4g} (alpha {alpha}) with value "
                f"{value:.4g} against a noise floor of {floor:.4g} (control "
                f"{control_values[name]:.4g}, self-null {self_null.get(name, float('nan')):.4g}) "
                f"-- the metric fires, but it fires at least as hard on a no-shift "
                f"sample of the same data, so it is not reading the declared axis")
        if value == 1.0 and name in ("ks_max",):
            entry["saturated"] = True
            entry["saturation_note"] = ("the statistic is at its maximum for every "
                                        "contrast in this corpus, including the "
                                        "no-shift ones, so it cannot discriminate: "
                                        "two different image sets are always perfectly "
                                        "separable by this dimension-wise statistic")
        result["metrics"][name] = entry
    return result


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def run(corpus: str, alpha: float = 0.05, max_images: int = 160,
        n_permutations: int = 200, seed: int = DEFAULT_SEED,
        metrics: Sequence[str] = tuple(METRICS)) -> Dict[str, Any]:
    """Evaluate every non-reference cell against the reference domain."""
    cells, problems = load_cells(corpus)
    if problems:
        raise ValueError("drift cell manifest problems: " + "; ".join(problems[:5]))
    by_id = {c["cell_id"]: c for c in cells}
    reference_cell = next(c for c in cells if c["role"] == "reference")
    ref_features = cell_features(corpus, reference_cell, max_images=max_images)
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise ValueError(f"unknown metric(s) {unknown}; known: {sorted(METRICS)}")

    per_cell: Dict[str, Any] = {}
    for cell in cells:
        if cell["role"] == "reference":
            continue
        features = cell_features(corpus, cell, max_images=max_images)
        control_id = cell.get("paired_cell")
        control_cell = by_id.get(control_id) if control_id else None
        control = (cell_features(corpus, control_cell, max_images=max_images)
                   if control_cell is not None and control_cell["role"] == "no_shift_control"
                   else None)
        evaluated = evaluate_contrast(ref_features, features, control=control, alpha=alpha,
                                      n_permutations=n_permutations, seed=seed)
        evaluated["role"] = cell["role"]
        evaluated["declared_axes_moved"] = cell.get("declared_axes_moved")
        evaluated["control_cell"] = control_id if control is not None else None
        evaluated["metrics"] = {k: v for k, v in evaluated["metrics"].items() if k in metrics}
        per_cell[cell["cell_id"]] = evaluated

    summary = {name: {d: 0 for d in DECISIONS} for name in metrics}
    for cell in per_cell.values():
        for name, entry in cell["metrics"].items():
            summary[name][entry["decision"]] += 1
    diagnostics = {}
    for name in metrics:
        values = [c["metrics"][name]["value"] for c in per_cell.values()
                  if name in c["metrics"]]
        margins = [c["metrics"][name]["margin_over_floor"] for c in per_cell.values()
                   if c["metrics"][name].get("margin_over_floor") is not None]
        diagnostics[name] = {
            "n_cells": len(values),
            "saturated": bool(values) and all(v == 1.0 for v in values),
            "max_margin_over_floor": max(margins) if margins else None,
            "median_margin_over_floor": (
                float(np.median(margins)) if margins else None),
        }

    return {
        "schema": HARNESS_SCHEMA,
        "status": "skeleton",
        "corpus": corpus,
        "alpha": alpha,
        "seed": seed,
        "n_permutations": n_permutations,
        "max_images_per_cell": max_images,
        "reference_cell": reference_cell["cell_id"],
        "metric_definitions": {m: METRICS[m] for m in metrics},
        "null_definition": NULL_DEFINITION,
        "cells": [{"cell_id": c["cell_id"], "role": c["role"],
                   "digest": (c.get("digests") or {}).get("dataset"),
                   "paired_cell": c.get("paired_cell")} for c in cells],
        "per_cell": per_cell,
        "summary": summary,
        "metric_diagnostics": diagnostics,
        "attribution": {
            "verdict": "under_determined",
            "why": ("this harness distinguishes a declared drift from resampling noise; "
                    "it receives no manipulation signal, and a poisoning shift and a "
                    "seasonal shift are both 'a shift'. Attribution needs the arbiter "
                    "in driftbench (natural calibration + contributor concentration) or "
                    "a manipulation battery supplied alongside the cells."),
        },
        "limitations": [
            "skeleton: the metric values are real, but the null is one resample per cell, "
            "so this is not a calibrated false-positive rate.",
            "pixel-statistic features only; a drift in detector behaviour with unchanged "
            "pixel statistics is invisible here.",
            "each cell has one seed_offset, so cell-to-cell differences include scene "
            "sampling noise that a multi-seed null would separate.",
            "no attribution: every cell reports under_determined, by design.",
        ],
    }


def render(report: Dict[str, Any]) -> str:
    lines = [f"drift harness ({report['status']}) alpha={report['alpha']} "
             f"reference={report['reference_cell']} "
             f"permutations={report['n_permutations']}", ""]
    lines.append(f"{'cell':34s} {'metric':22s} {'value':>10s} {'control':>10s} "
                 f"{'p':>7s}  decision")
    lines.append("-" * 92)
    for cell_id, cell in report["per_cell"].items():
        for name, entry in cell["metrics"].items():
            control = entry.get("control_value")
            lines.append(f"{cell_id:34s} {name:22s} {entry['value']:10.4g} "
                         f"{(f'{control:10.4g}' if control is not None else '         -')} "
                         f"{entry['p_value']:7.4f}  {entry['decision']}")
    lines.append("")
    for name, counts in report["summary"].items():
        lines.append(f"{name:22s} " + "  ".join(f"{d}={counts[d]}" for d in DECISIONS
                                                if counts[d]))
    lines.append("")
    lines.append(f"attribution: {report['attribution']['verdict']} "
                 f"-- {report['attribution']['why']}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cviaf lab drift-harness",
                                 description="drift metrics with fixed definitions")
    ap.add_argument("--corpus", default="runs/drift_cells")
    ap.add_argument("--json", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max-images", type=int, default=160)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--metric", action="append", choices=sorted(METRICS))
    args = ap.parse_args(argv)

    if not os.path.isdir(args.corpus):
        print(f"{args.corpus}: no such corpus")
        return 2
    try:
        report = run(args.corpus, alpha=args.alpha, max_images=args.max_images,
                     n_permutations=args.permutations, seed=args.seed,
                     metrics=args.metric or tuple(METRICS))
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 2
    measured = sum(1 for c in report["per_cell"].values()
                   for e in c["metrics"].values() if e["decision"] != "insufficient_n")
    print(render(report))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, allow_nan=False)
        print(f"\nreport written to {args.json}")
    return 0 if measured else 4


if __name__ == "__main__":
    raise SystemExit(main())
