"""
Distribution Shift and Anomaly Assessment Module.

Detects material deviation from a declared reference distribution,
including changes caused by terrain, season, sensor, illumination,
or acquisition conditions.

Implements:
  - Mahalanobis distance from reference distribution
  - Maximum Mean Discrepancy (MMD) kernel test
  - Maximum Softmax Probability (MSP) analysis
  - Kolmogorov-Smirnov tests per feature dimension
  - Shift type characterization (covariate vs concept vs prior)
  - Natural drift vs adversarial manipulation distinction
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA

from cviaf.core.types import Finding, Severity, Disposition, AttackClass
from cviaf.drift.attribution import (
    NaturalDriftCalibration,
    run_attribution,
    VERDICT_SUSPICIOUS,
    VERDICT_NATURAL,
    VERDICT_UNDERDETERMINED,
    VERDICT_UNAVAILABLE,
)


class MahalanobisDetector:
    """
    Mahalanobis distance-based OOD and shift detector.
    
    Computes the Mahalanobis distance of operational samples
    from the reference distribution, using class-conditional
    or global statistics.
    """

    def __init__(self):
        self.ref_mean: Optional[np.ndarray] = None
        self.ref_cov_inv: Optional[np.ndarray] = None
        self.ref_cov: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, reference_features: np.ndarray) -> None:
        """Fit the reference distribution from training features."""
        self.ref_mean = np.mean(reference_features, axis=0)
        cov = np.cov(reference_features, rowvar=False)
        # Regularize for numerical stability
        reg = np.eye(cov.shape[0]) * 1e-6
        self.ref_cov = cov + reg
        try:
            self.ref_cov_inv = np.linalg.inv(self.ref_cov)
        except np.linalg.LinAlgError:
            self.ref_cov_inv = np.linalg.pinv(self.ref_cov)
        self._fitted = True

    def score(self, features: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distances for each sample.
        Returns array of distances.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() with reference features first.")

        delta = features - self.ref_mean
        # d = sqrt(delta @ Sigma_inv @ delta^T) per sample
        left = delta @ self.ref_cov_inv
        distances = np.sqrt(np.sum(left * delta, axis=1))
        return distances


class MMDCalculator:
    """
    Maximum Mean Discrepancy (MMD) with RBF kernel.
    
    Non-parametric two-sample test comparing distributions
    in a reproducing kernel Hilbert space.
    """

    def __init__(self, kernel_bandwidth: float = None, seed: Optional[int] = 0):
        self.bandwidth = kernel_bandwidth
        # Seed for the permutation test. Default is a fixed seed so the p-value
        # (and every verdict derived from it) is reproducible run to run.
        # Pass seed=None only to restore the legacy global-RNG behaviour.
        self.seed = seed

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
        """Compute RBF kernel matrix."""
        # ||x - y||^2
        XX = np.sum(X ** 2, axis=1, keepdims=True)
        YY = np.sum(Y ** 2, axis=1, keepdims=True)
        distances = XX + YY.T - 2 * X @ Y.T
        return np.exp(-distances / (2 * sigma ** 2))

    def _median_heuristic(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute bandwidth using the median heuristic."""
        combined = np.vstack([X[:100], Y[:100]])  # Subsample for speed
        dists = np.sqrt(np.sum(
            (combined[:, np.newaxis] - combined[np.newaxis, :]) ** 2, axis=2
        ))
        return float(np.median(dists[dists > 0]))

    def compute(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
        """
        Compute MMD^2 between two sets of samples.
        
        Returns dict with mmd2 value and p-value from permutation test.
        """
        n = len(X)
        m = len(Y)

        if n < 2 or m < 2:
            return {"mmd2": 0.0, "p_value": 1.0}

        sigma = self.bandwidth or self._median_heuristic(X, Y)

        K_XX = self._rbf_kernel(X, X, sigma)
        K_YY = self._rbf_kernel(Y, Y, sigma)
        K_XY = self._rbf_kernel(X, Y, sigma)

        # Unbiased MMD^2 estimator
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_XX = (np.sum(K_XX) - np.sum(diag_X)) / (n * (n - 1))
        sum_YY = (np.sum(K_YY) - np.sum(diag_Y)) / (m * (m - 1))
        sum_XY = np.sum(K_XY) / (n * m)

        mmd2 = sum_XX + sum_YY - 2 * sum_XY

        # Permutation test for p-value. Uses an explicit Generator instead of
        # the global RNG: with the global RNG the same clean control could
        # flip-flop between no-finding and a borderline finding across runs.
        rng = np.random.default_rng(self.seed) if self.seed is not None else np.random
        combined = np.vstack([X, Y])
        num_permutations = 100
        null_mmd2 = []

        for _ in range(num_permutations):
            perm = rng.permutation(n + m)
            X_perm = combined[perm[:n]]
            Y_perm = combined[perm[n:]]

            K_XX_p = self._rbf_kernel(X_perm, X_perm, sigma)
            K_YY_p = self._rbf_kernel(Y_perm, Y_perm, sigma)
            K_XY_p = self._rbf_kernel(X_perm, Y_perm, sigma)

            s_XX = (np.sum(K_XX_p) - np.sum(np.diagonal(K_XX_p))) / (n * (n - 1))
            s_YY = (np.sum(K_YY_p) - np.sum(np.diagonal(K_YY_p))) / (m * (m - 1))
            s_XY = np.sum(K_XY_p) / (n * m)

            null_mmd2.append(s_XX + s_YY - 2 * s_XY)

        p_value = float(np.mean(np.array(null_mmd2) >= mmd2))

        return {
            "mmd2": float(mmd2),
            "p_value": p_value,
            "kernel_bandwidth": float(sigma),
        }


class KSTestRunner:
    """
    Kolmogorov-Smirnov test per feature dimension.
    Identifies which feature dimensions show the most shift.
    """

    def test(self, reference: np.ndarray, operational: np.ndarray,
             top_k: int = 10) -> Dict[str, Any]:
        """
        Run KS test on each feature dimension.
        
        Returns the top-k dimensions with highest KS statistic
        and overall rejection summary.
        """
        from scipy import stats

        n_features = min(reference.shape[1], operational.shape[1])
        results = []

        for dim in range(n_features):
            stat, p_value = stats.ks_2samp(reference[:, dim], operational[:, dim])
            results.append({
                "dimension": dim,
                "ks_statistic": float(stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
            })

        # Sort by KS statistic descending
        results.sort(key=lambda x: x["ks_statistic"], reverse=True)

        num_significant = sum(1 for r in results if r["significant"])
        # Apply Bonferroni correction
        bonferroni_threshold = 0.05 / n_features
        num_bonferroni_sig = sum(
            1 for r in results if r["p_value"] < bonferroni_threshold
        )

        return {
            "top_shifted_dimensions": results[:top_k],
            "ks_statistic_array": [r["ks_statistic"] for r in
                                   sorted(results, key=lambda x: x["dimension"])],
            "num_dimensions": n_features,
            "num_significant_raw": num_significant,
            "num_significant_bonferroni": num_bonferroni_sig,
            "fraction_shifted": float(num_significant / n_features) if n_features > 0 else 0,
        }


class ShiftCharacterizer:
    """
    Characterizes the type of distribution shift:
      - Covariate shift: P(X) changes but P(Y|X) stays same
      - Concept drift: P(Y|X) changes
      - Prior probability shift: P(Y) changes
    
    Also distinguishes natural operational drift from adversarial manipulation.
    """

    def characterize(
        self,
        ref_features: np.ndarray,
        op_features: np.ndarray,
        ref_labels: np.ndarray = None,
        op_labels: np.ndarray = None,
        ref_predictions: np.ndarray = None,
        op_predictions: np.ndarray = None,
        maha_distances: np.ndarray = None,
        mmd_result: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """
        Characterize the observed shift using multiple signals.
        """
        result = {
            "shift_type": "none",
            "shift_magnitude": 0.0,
            "natural_vs_adversarial": "unknown",
            "confidence": 0.0,
            "reasoning": [],
        }

        signals = []

        # Signal 1: Feature distribution change (covariate shift indicator)
        if mmd_result and mmd_result.get("p_value", 1.0) < 0.05:
            signals.append("covariate")
            result["reasoning"].append(
                f"MMD test rejects null hypothesis (p={mmd_result['p_value']:.4f}), "
                f"indicating feature distribution change."
            )

        # Signal 2: Label distribution change (prior shift indicator)
        if ref_labels is not None and op_labels is not None:
            ref_class_dist = self._class_distribution(ref_labels)
            op_class_dist = self._class_distribution(op_labels)
            kl_div = self._kl_divergence(ref_class_dist, op_class_dist)

            if kl_div > 0.1:
                signals.append("prior")
                result["reasoning"].append(
                    f"Label distribution KL divergence is {kl_div:.4f}, "
                    f"indicating prior probability shift."
                )

        # Signal 3: Prediction accuracy change (concept drift indicator)
        if ref_predictions is not None and op_predictions is not None:
            if ref_labels is not None and op_labels is not None:
                ref_acc = np.mean(ref_predictions == ref_labels)
                op_acc = np.mean(op_predictions == op_labels)
                acc_drop = ref_acc - op_acc

                if acc_drop > 0.1 and "covariate" not in signals:
                    signals.append("concept")
                    result["reasoning"].append(
                        f"Accuracy dropped from {ref_acc:.3f} to {op_acc:.3f} "
                        f"without proportional feature shift, suggesting concept drift."
                    )

        # Determine shift type
        if not signals:
            result["shift_type"] = "none"
            result["confidence"] = 0.9
        elif "concept" in signals:
            result["shift_type"] = "concept_drift"
            result["shift_magnitude"] = 0.8
            result["confidence"] = 0.7
        elif "covariate" in signals and "prior" in signals:
            result["shift_type"] = "covariate_and_prior"
            result["shift_magnitude"] = 0.6
            result["confidence"] = 0.65
        elif "covariate" in signals:
            result["shift_type"] = "covariate_shift"
            result["shift_magnitude"] = 0.5
            result["confidence"] = 0.7
        elif "prior" in signals:
            result["shift_type"] = "prior_probability_shift"
            result["shift_magnitude"] = 0.4
            result["confidence"] = 0.6

        # Natural-vs-manipulated attribution is NOT decided here. The previous
        # implementation asserted "probable_natural_drift" from a fall-through
        # else-branch on uncalibrated Mahalanobis moments, which mislabelled a
        # genuinely manipulated contribution. Attribution is now owned by the
        # calibrated arbiter in cviaf.drift.attribution, invoked from
        # DistributionShiftAssessor.assess when a shift is detected.
        result["natural_vs_adversarial"] = "not_assessed"
        result["reasoning"].append(
            "Natural-vs-manipulated attribution is delegated to the calibrated "
            "attribution arbiter (see the assessment's 'attribution' block)."
        )

        return result

    def _class_distribution(self, labels: np.ndarray) -> np.ndarray:
        """Compute normalized class distribution."""
        classes = np.unique(labels)
        dist = np.zeros(int(np.max(classes)) + 1)
        for c in classes:
            dist[int(c)] = np.sum(labels == c)
        return dist / (np.sum(dist) + 1e-10)

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence D(p || q) with smoothing."""
        min_len = min(len(p), len(q))
        p = p[:min_len] + 1e-10
        q = q[:min_len] + 1e-10
        p = p / np.sum(p)
        q = q / np.sum(q)
        return float(np.sum(p * np.log(p / q)))

    def _skewness(self, data: np.ndarray) -> float:
        """Compute skewness."""
        n = len(data)
        if n < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))


def standardized_wasserstein_effect(reference: np.ndarray, operational: np.ndarray) -> Dict[str, Any]:
    """Median per-active-dimension Wasserstein distance, in reference SD units.

    A constant feature is excluded rather than allowed to dominate via division
    by epsilon. The screening tests still see it.
    """
    from scipy.stats import wasserstein_distance
    ref = np.asarray(reference, dtype=float)
    op = np.asarray(operational, dtype=float)
    if (ref.ndim != 2 or op.ndim != 2 or ref.shape[1] != op.shape[1]
            or len(ref) < 2 or len(op) < 2 or not np.isfinite(ref).all()
            or not np.isfinite(op).all()):
        raise ValueError("reference and operational features must be finite matrices of equal width, with >=2 rows")
    ref_scale = np.std(ref, axis=0, ddof=1)
    active = np.isfinite(ref_scale) & (ref_scale > 1e-10)
    dimensions = np.flatnonzero(active)
    effects = [wasserstein_distance(ref[:, i], op[:, i]) / ref_scale[i]
               for i in dimensions]
    return {"value": float(np.median(effects)) if effects else 0.0,
            "active_dimensions": int(active.sum()), "total_dimensions": int(ref.shape[1]),
            "reference_n": int(len(ref)), "operational_n": int(len(op))}


def calibrate_effect_floor(reference: np.ndarray, clean_batches: List[np.ndarray],
                           alpha: float = .05, minimum: float = .2) -> Dict[str, Any]:
    """Held-out clean-batch quantile for the effect gate, at the same batch size.

    Never reuse these calibration batches in the evaluation. This per-round
    calibration does not control the lifetime false alarm rate of a monitor.
    """
    if not 0 < alpha < 1 or not clean_batches:
        raise ValueError("alpha must be in (0,1) and clean batches nonempty")
    sizes = {len(b) for b in clean_batches}
    if len(sizes) != 1:
        raise ValueError("clean calibration batches must have matching sizes")
    null = np.sort([standardized_wasserstein_effect(reference, b)["value"]
                    for b in clean_batches])
    rank = int(np.ceil((len(null) + 1) * (1 - alpha)))
    # If n is too small, there is no finite threshold at this alpha.
    floor = float(max(minimum, null[rank-1])) if rank <= len(null) else float("inf")
    return {"floor": floor, "clean_batches": len(null), "batch_size": sizes.pop(),
            "alpha": alpha, "minimum": minimum,
            "calibration": "held_out_clean_batch_quantile"}



class DistributionShiftAssessor:
    """
    Orchestrates distribution shift detection and characterization.
    """

    def __init__(self, seed: Optional[int] = 0, min_standardized_wasserstein: Optional[float] = None):
        if min_standardized_wasserstein is not None and (not np.isfinite(min_standardized_wasserstein) or min_standardized_wasserstein < 0):
            raise ValueError("effect-size floor must be finite and nonnegative")
        self.min_standardized_wasserstein = min_standardized_wasserstein
        self.seed = seed
        self.maha_detector = MahalanobisDetector()
        self.mmd_calc = MMDCalculator(seed=seed)
        self.characterizer = ShiftCharacterizer()

    def assess(
        self,
        reference_features: np.ndarray,
        operational_features: np.ndarray,
        reference_logits: np.ndarray = None,
        operational_logits: np.ndarray = None,
        reference_labels: np.ndarray = None,
        operational_labels: np.ndarray = None,
        reference_images: np.ndarray = None,
        operational_images: np.ndarray = None,
        operational_contributors: list = None,
        attribution_calibration: "NaturalDriftCalibration" = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive distribution shift assessment.
        """
        findings = []

        # 1. Fit reference distribution and compute Mahalanobis distances
        self.maha_detector.fit(reference_features)
        maha_distances = self.maha_detector.score(operational_features)
        mean_maha = float(np.mean(maha_distances))
        std_maha = float(np.std(maha_distances))

        # 2. MMD test (subsample for large datasets)
        max_samples = 500
        ref_sub = reference_features[:max_samples]
        op_sub = operational_features[:max_samples]

        # Reduce dimensionality if too high (MMD is O(n^2) in features too)
        if ref_sub.shape[1] > 50:
            # Cap components at min(n_samples, n_features); PCA with more
            # components than samples raises on svd_solver='full'.
            n_comp = min(50, ref_sub.shape[0], ref_sub.shape[1])
            pca = PCA(n_components=n_comp)
            ref_sub = pca.fit_transform(ref_sub)
            op_sub = pca.transform(op_sub)

        mmd_result = self.mmd_calc.compute(ref_sub, op_sub)

        # 3. KS tests (try scipy, fall back if unavailable)
        ks_result = None
        try:
            ks_runner = KSTestRunner()
            ks_sub_ref = reference_features[:1000]
            ks_sub_op = operational_features[:1000]
            if ks_sub_ref.shape[1] > 100:
                n_comp = min(100, ks_sub_ref.shape[0], ks_sub_ref.shape[1])
                pca2 = PCA(n_components=n_comp)
                ks_sub_ref = pca2.fit_transform(ks_sub_ref)
                ks_sub_op = pca2.transform(ks_sub_op)
            ks_result = ks_runner.test(ks_sub_ref, ks_sub_op)
        except ImportError:
            ks_result = {"note": "scipy not available for KS tests"}

        # The screen answers whether the distributions differ; this floor answers
        # whether they differ enough to merit an operational alert. Use the same
        # reference units for every dimension, including when batch size changes.
        effect_result = standardized_wasserstein_effect(reference_features, operational_features)
        effect_size = effect_result["value"]
        effect_floor = self.min_standardized_wasserstein

        # 4. MSP analysis
        msp_result = None
        if operational_logits is not None:
            max_logits = np.max(operational_logits, axis=-1, keepdims=True)
            exp_logits = np.exp(operational_logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            max_probs = np.max(probs, axis=-1)
            mean_msp = float(np.mean(max_probs))
            std_msp = float(np.std(max_probs))

            msp_result = {
                "mean_max_softmax_prob": mean_msp,
                "std_max_softmax_prob": std_msp,
                "low_confidence_fraction": float(np.mean(max_probs < 0.5)),
            }

        # 5. Predictions for characterization
        ref_preds = None
        op_preds = None
        if reference_logits is not None:
            ref_preds = np.argmax(reference_logits, axis=-1)
        if operational_logits is not None:
            op_preds = np.argmax(operational_logits, axis=-1)

        # 6. Characterize shift
        char_result = self.characterizer.characterize(
            ref_features=reference_features,
            op_features=operational_features,
            ref_labels=reference_labels,
            op_labels=operational_labels,
            ref_predictions=ref_preds,
            op_predictions=op_preds,
            maha_distances=maha_distances,
            mmd_result=mmd_result,
        )

        # 7. Generate findings
        significance_screen = bool(mmd_result["p_value"] < 0.05 or mean_maha > 15.0)
        # No independently calibrated floor is supplied by default. Keep the
        # legacy significance screen and expose effect evidence, without claiming
        # a production false-alarm guarantee from the provisional 0.2 threshold.
        effect_passed = bool(effect_floor is None or effect_size >= effect_floor)
        shift_detected = significance_screen and effect_passed
        effect_evidence = {
            "metric": "median_per_dimension_wasserstein_over_reference_std",
            "value": effect_size, "floor": effect_floor,
            "passed": effect_passed, "significance_screen_passed": significance_screen,
            **effect_result,
        }
        if not shift_detected and effect_floor is not None:
            char_result["shift_type"] = "none"
            char_result["natural_vs_adversarial"] = "unknown"
            char_result["reasoning"].append("No material shift passed both screens.")

        attribution = None
        if shift_detected:
            # --- natural-vs-manipulated attribution (calibrated; honesty valve) ---
            if attribution_calibration is not None:
                ref_maha = self.maha_detector.score(reference_features)
                ks_stats = None
                if ks_result and "ks_statistic_array" in ks_result:
                    ks_stats = np.asarray(ks_result["ks_statistic_array"], float)
                else:
                    ks_stats = np.zeros(reference_features.shape[1])
                attribution = run_attribution(
                    reference_features=reference_features,
                    operational_features=operational_features,
                    maha_ref=ref_maha,
                    maha_op=maha_distances,
                    ks_stats=ks_stats,
                    calibration=attribution_calibration,
                    reference_images=reference_images,
                    operational_images=operational_images,
                    operational_contributors=operational_contributors,
                    reference_logits=reference_logits,
                    operational_logits=operational_logits,
                )
                verdict = attribution["natural_vs_adversarial"]
            else:
                verdict = VERDICT_UNAVAILABLE
                attribution = {
                    "natural_vs_adversarial": verdict,
                    "votes": [],
                    "reasoning": [
                        "No natural-drift calibration was supplied, so attribution "
                        "cannot run. Build one with `python -m cviaf.lab driftbench`."
                    ],
                    "resolution_suggestions": [
                        "run driftbench against a natural-drift battery and pass "
                        "the resulting calibration"
                    ],
                }
            char_result["natural_vs_adversarial"] = verdict
            char_result["attribution"] = attribution
            char_result.pop("confidence", None)

            if verdict == VERDICT_SUSPICIOUS:
                severity = Severity.CRITICAL
                disposition = Disposition.QUARANTINE
                attack_class = AttackClass.ADVERSARIAL_DISTRIBUTION_SHIFT.value
            elif verdict == VERDICT_UNDERDETERMINED:
                # Escalate to HIGH only when there is both magnitude and at
                # least one manipulation-side signal; a borderline-detected
                # shift with no fired signal stays a MEDIUM review item.
                n_shape = (attribution or {}).get("n_shape_votes_fired", 0)
                contrib_fired = (attribution or {}).get(
                    "contributor_concentration_fired", False)
                severity = (Severity.HIGH
                            if (mean_maha > 15.0 and (n_shape >= 1 or contrib_fired))
                            else Severity.MEDIUM)
                disposition = Disposition.REVIEW
                attack_class = AttackClass.ADVERSARIAL_DISTRIBUTION_SHIFT.value
            elif verdict == VERDICT_UNAVAILABLE:
                severity = Severity.HIGH if mean_maha > 30 else Severity.MEDIUM
                disposition = Disposition.REVIEW
                attack_class = AttackClass.COVARIATE_SHIFT.value
            else:  # probable_natural_drift
                severity = Severity.HIGH if mean_maha > 30 else Severity.MEDIUM
                disposition = Disposition.REVIEW
                attack_class = AttackClass.COVARIATE_SHIFT.value

            # Confidence is the measured reliability of this verdict class on
            # the calibration battery, not a hand-set constant. When the
            # battery has not measured it, say so instead of inventing a number.
            measured = (attribution_calibration.measured.get(verdict, {})
                        if attribution_calibration is not None else {})
            if "reliability" in measured:
                confidence = float(measured["reliability"])
            else:
                confidence = 0.5
                attribution.setdefault("reasoning", []).append(
                    "Verdict reliability unmeasured on the calibration battery; "
                    "confidence set to 0.5 (unknown)."
                )

            findings.append(Finding(
                module="distribution_shift",
                attack_class=attack_class,
                severity=severity.value,
                confidence=confidence,
                title=f"Distribution shift detected: {char_result['shift_type']}",
                description=(
                    f"Significant distribution shift detected between reference "
                    f"and operational data. MMD p-value={mmd_result['p_value']:.4f}, "
                    f"mean Mahalanobis distance={mean_maha:.2f}, "
                    f"standardized Wasserstein median={effect_size:.3f} "
                    f"(policy floor={effect_floor if effect_floor is not None else 'uncalibrated'}). "
                    f"Shift type: {char_result['shift_type']}. "
                    f"Assessment: {char_result['natural_vs_adversarial']}."
                ),
                evidence={
                    "mahalanobis_mean": mean_maha,
                    "mahalanobis_std": std_maha,
                    "effect_size": effect_evidence,
                    "mmd2": mmd_result["mmd2"],
                    "mmd_p_value": mmd_result["p_value"],
                    "shift_type": char_result["shift_type"],
                    "natural_vs_adversarial": char_result["natural_vs_adversarial"],
                    "attribution_votes": (attribution or {}).get("votes", []),
                    "attribution_reasoning": (attribution or {}).get("reasoning", []),
                    "resolution_suggestions": (attribution or {}).get(
                        "resolution_suggestions", []),
                },
                affected_assets=["operational_dataset"],
                disposition=disposition.value,
                remediation="Investigate the source of shift. If adversarial, quarantine affected samples.",
            ))

        return {
            "shift_detected": shift_detected,
            "effect_size": effect_evidence,
            "mahalanobis": {
                "mean_distance": mean_maha,
                "std_distance": std_maha,
                "per_sample_distances": maha_distances.tolist()[:100],  # Cap for serialization
            },
            "mmd": mmd_result,
            "ks_test": ks_result,
            "msp": msp_result,
            "characterization": char_result,
            "attribution": attribution,
            "findings": [f.to_dict() for f in findings],
            "overall_severity": findings[0].severity if findings else Severity.LOW.value,
            "overall_disposition": findings[0].disposition if findings else Disposition.ACCEPT.value,
        }


DRIFT_CALIBRATION_SCHEMA = "cviaf-drift-calibration/1.0"


class DriftCalibrationError(ValueError):
    pass


def load_drift_calibration(path: str) -> Dict[str, Any]:
    """Load a runtime-generated assessment record; do not confuse with arbiter calibration."""
    import json
    try:
        with open(path) as fh:
            obj = json.load(fh)
        if obj.get("schema") != DRIFT_CALIBRATION_SCHEMA or not isinstance(obj.get("assessment"), dict):
            raise ValueError("wrong schema or missing assessment")
        return obj
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise DriftCalibrationError(f"invalid drift assessment record {path}: {exc}") from exc
