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

    def __init__(self, kernel_bandwidth: float = None):
        self.bandwidth = kernel_bandwidth

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

        # Permutation test for p-value
        combined = np.vstack([X, Y])
        num_permutations = 100
        null_mmd2 = []

        for _ in range(num_permutations):
            perm = np.random.permutation(n + m)
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

        # Natural vs adversarial distinction
        if maha_distances is not None and len(maha_distances) > 0:
            # Adversarial manipulation tends to produce:
            # 1. Bimodal Mahalanobis distance distribution
            # 2. Sudden, uniform shift rather than gradual
            # 3. Concentrated effect on specific classes
            
            maha_mean = np.mean(maha_distances)
            maha_std = np.std(maha_distances)
            maha_skew = self._skewness(maha_distances)
            extreme_fraction = np.mean(maha_distances > maha_mean + 3 * maha_std)

            if extreme_fraction > 0.1 and abs(maha_skew) > 2.0:
                result["natural_vs_adversarial"] = "suspicious_manipulation"
                result["reasoning"].append(
                    f"Mahalanobis distances show {extreme_fraction*100:.1f}% "
                    f"extreme outliers with skewness {maha_skew:.2f}, "
                    f"inconsistent with natural drift patterns."
                )
            elif maha_mean > 20 and maha_std < maha_mean * 0.2:
                result["natural_vs_adversarial"] = "suspicious_uniform_shift"
                result["reasoning"].append(
                    f"Uniform high Mahalanobis distance (mean={maha_mean:.1f}, "
                    f"std={maha_std:.1f}) suggests coordinated manipulation "
                    f"rather than natural environmental change."
                )
            else:
                result["natural_vs_adversarial"] = "probable_natural_drift"
                result["reasoning"].append(
                    "Shift characteristics are consistent with natural "
                    "operational drift (gradual, non-concentrated)."
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


class DistributionShiftAssessor:
    """
    Orchestrates distribution shift detection and characterization.
    """

    def __init__(self):
        self.maha_detector = MahalanobisDetector()
        self.mmd_calc = MMDCalculator()
        self.characterizer = ShiftCharacterizer()

    def assess(
        self,
        reference_features: np.ndarray,
        operational_features: np.ndarray,
        reference_logits: np.ndarray = None,
        operational_logits: np.ndarray = None,
        reference_labels: np.ndarray = None,
        operational_labels: np.ndarray = None,
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
        shift_detected = mmd_result["p_value"] < 0.05 or mean_maha > 15.0

        if shift_detected:
            if char_result.get("natural_vs_adversarial") == "suspicious_manipulation":
                severity = Severity.CRITICAL
                disposition = Disposition.QUARANTINE
                attack_class = AttackClass.ADVERSARIAL_DISTRIBUTION_SHIFT.value
            elif char_result.get("natural_vs_adversarial") == "suspicious_uniform_shift":
                severity = Severity.HIGH
                disposition = Disposition.REVIEW
                attack_class = AttackClass.ADVERSARIAL_DISTRIBUTION_SHIFT.value
            elif mean_maha > 30:
                severity = Severity.HIGH
                disposition = Disposition.REVIEW
                attack_class = AttackClass.COVARIATE_SHIFT.value
            else:
                severity = Severity.MEDIUM
                disposition = Disposition.REVIEW
                attack_class = AttackClass.COVARIATE_SHIFT.value

            confidence = min(0.95, 0.4 + (1.0 - mmd_result["p_value"]) * 0.3 + min(0.3, mean_maha / 100))

            findings.append(Finding(
                module="distribution_shift",
                attack_class=attack_class,
                severity=severity.value,
                confidence=confidence,
                title=f"Distribution shift detected: {char_result['shift_type']}",
                description=(
                    f"Significant distribution shift detected between reference "
                    f"and operational data. MMD p-value={mmd_result['p_value']:.4f}, "
                    f"mean Mahalanobis distance={mean_maha:.2f}. "
                    f"Shift type: {char_result['shift_type']}. "
                    f"Assessment: {char_result['natural_vs_adversarial']}."
                ),
                evidence={
                    "mahalanobis_mean": mean_maha,
                    "mahalanobis_std": std_maha,
                    "mmd2": mmd_result["mmd2"],
                    "mmd_p_value": mmd_result["p_value"],
                    "shift_type": char_result["shift_type"],
                    "natural_vs_adversarial": char_result["natural_vs_adversarial"],
                },
                affected_assets=["operational_dataset"],
                disposition=disposition.value,
                remediation="Investigate the source of shift. If adversarial, quarantine affected samples.",
            ))

        return {
            "shift_detected": shift_detected,
            "mahalanobis": {
                "mean_distance": mean_maha,
                "std_distance": std_maha,
                "per_sample_distances": maha_distances.tolist()[:100],  # Cap for serialization
            },
            "mmd": mmd_result,
            "ks_test": ks_result,
            "msp": msp_result,
            "characterization": char_result,
            "findings": [f.to_dict() for f in findings],
            "overall_severity": findings[0].severity if findings else Severity.LOW.value,
            "overall_disposition": findings[0].disposition if findings else Disposition.ACCEPT.value,
        }
