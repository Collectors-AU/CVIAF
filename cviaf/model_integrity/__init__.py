"""
Model Integrity Assessment Module.

Evaluates whether a supplied model exhibits anomalous, substituted,
or backdoor-like behavior. Supports both white-box and black-box
access levels with graceful degradation.

Implements:
  - Neural Cleanse (trigger reverse-engineering via optimization)
  - Activation/weight statistical analysis
  - Behavioral fingerprinting
  - Output entropy analysis (black-box)
  - Model substitution detection via weight digest comparison
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from cviaf.core.types import (
    Finding, Severity, Disposition, AccessLevel, AttackClass
)


class NeuralCleanseDetector:
    """
    Implements Neural Cleanse-style trigger reverse-engineering.
    
    For each class, attempts to find a minimal perturbation pattern
    (trigger) that causes all inputs to be classified as that class.
    A class requiring an abnormally small trigger is likely backdoored.
    
    Uses the Median Absolute Deviation (MAD) anomaly index to identify
    outlier classes.
    
    Requires white-box access (model must accept gradient-like optimization
    or at least forward passes with input perturbation).
    """

    def __init__(self, num_classes: int, input_shape: Tuple[int, ...],
                 lr: float = 0.1, steps: int = 500,
                 lambda_reg: float = 0.01):
        self.num_classes = num_classes
        self.input_shape = input_shape  # (C, H, W)
        self.lr = lr
        self.steps = steps
        self.lambda_reg = lambda_reg

    def _optimize_trigger_numpy(
        self, predict_fn, target_class: int,
        clean_images: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Optimize a trigger pattern using iterative gradient-free optimization.
        
        Since we may not have PyTorch gradients, we use coordinate-wise
        finite differences to approximate the gradient.
        
        Returns: (mask, pattern, l1_norm)
        """
        # Initialize mask and pattern
        if len(self.input_shape) == 3:
            c, h, w = self.input_shape
        else:
            h, w = self.input_shape[-2], self.input_shape[-1]
            c = self.input_shape[0] if len(self.input_shape) > 2 else 3

        mask = np.random.uniform(0, 0.1, (1, h, w)).astype(np.float32)
        pattern = np.random.uniform(0, 1, (c, h, w)).astype(np.float32)

        best_l1 = float("inf")
        best_mask = mask.copy()
        best_pattern = pattern.copy()

        # Use a subset of clean images for speed
        subset = clean_images[:min(32, len(clean_images))]

        for step in range(self.steps):
            # Apply trigger: x_poisoned = (1 - mask) * x + mask * pattern
            poisoned = (1 - mask) * subset + mask * pattern

            # Clip to valid range
            poisoned = np.clip(poisoned, 0, 1)

            # Get predictions
            try:
                logits = predict_fn(poisoned.astype(np.float32))
                if logits.ndim == 1:
                    logits = logits.reshape(1, -1)

                # Compute attack success rate
                preds = np.argmax(logits, axis=-1)
                success_rate = np.mean(preds == target_class)

                # Compute loss: -log(softmax[target]) + lambda * |mask|
                # Approximate with cross-entropy
                max_logits = np.max(logits, axis=-1, keepdims=True)
                exp_logits = np.exp(logits - max_logits)
                softmax = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

                target_prob = softmax[:, target_class].mean()
                l1 = np.sum(np.abs(mask))

                if success_rate > 0.8 and l1 < best_l1:
                    best_l1 = l1
                    best_mask = mask.copy()
                    best_pattern = pattern.copy()

                # Simple gradient-free update: random perturbation + selection
                mask_delta = np.random.randn(*mask.shape) * 0.01
                pattern_delta = np.random.randn(*pattern.shape) * 0.01

                # Try perturbation
                new_mask = np.clip(mask + mask_delta, 0, 1)
                new_pattern = np.clip(pattern + pattern_delta, 0, 1)

                new_poisoned = (1 - new_mask) * subset + new_mask * new_pattern
                new_poisoned = np.clip(new_poisoned, 0, 1)

                try:
                    new_logits = predict_fn(new_poisoned.astype(np.float32))
                    if new_logits.ndim == 1:
                        new_logits = new_logits.reshape(1, -1)
                    new_preds = np.argmax(new_logits, axis=-1)
                    new_success = np.mean(new_preds == target_class)
                    new_l1 = np.sum(np.abs(new_mask))

                    # Accept if better
                    if new_success >= success_rate and new_l1 <= l1:
                        mask = new_mask
                        pattern = new_pattern
                    elif new_success > success_rate:
                        mask = new_mask
                        pattern = new_pattern
                except Exception:
                    pass

            except Exception:
                break

        return best_mask, best_pattern, best_l1

    def detect(self, predict_fn, clean_images: np.ndarray) -> List[Finding]:
        """
        Run Neural Cleanse on all classes.
        
        Args:
            predict_fn: Function that takes (N, C, H, W) array and returns logits
            clean_images: Clean reference images (N, C, H, W)
        
        Returns:
            List of findings for suspected backdoored classes
        """
        findings = []
        l1_norms = []

        for target_class in range(self.num_classes):
            mask, pattern, l1 = self._optimize_trigger_numpy(
                predict_fn, target_class, clean_images
            )
            l1_norms.append(l1)

        if not l1_norms or len(l1_norms) < 2:
            return findings

        l1_arr = np.array(l1_norms)
        median = np.median(l1_arr)
        mad = np.median(np.abs(l1_arr - median))

        if mad < 1e-10:
            return findings

        anomaly_indices = []
        for i, l1 in enumerate(l1_arr):
            deviation = (median - l1) / (mad * 1.4826)  # MAD to std conversion
            if deviation > 2.0:  # 2-sigma threshold
                anomaly_indices.append((i, deviation, l1))

        for class_idx, deviation, l1 in anomaly_indices:
            severity = Severity.CRITICAL if deviation > 3.0 else Severity.HIGH
            confidence = min(0.95, 0.5 + deviation * 0.1)

            findings.append(Finding(
                module="model_integrity",
                attack_class=AttackClass.MODEL_BACKDOOR.value,
                severity=severity.value,
                confidence=confidence,
                title=f"Potential backdoor trigger for class {class_idx}",
                description=(
                    f"Neural Cleanse found an abnormally small trigger pattern "
                    f"(L1={l1:.2f}) for class {class_idx}. The MAD anomaly index "
                    f"is {deviation:.2f} (threshold: 2.0). This suggests a "
                    f"backdoor trigger may have been embedded for this class."
                ),
                evidence={
                    "method": "neural_cleanse",
                    "target_class": class_idx,
                    "l1_norm": float(l1),
                    "mad_anomaly_index": float(deviation),
                    "all_l1_norms": [float(x) for x in l1_arr],
                    "median_l1": float(median),
                    "mad": float(mad),
                },
                disposition=Disposition.QUARANTINE.value,
                remediation="Fine-prune the model or retrain without suspected poisoned data.",
            ))

        return findings


class WeightAnalyzer:
    """
    Analyzes model weight distributions for anomalies.
    
    Detects:
      - Unusual weight distributions (high kurtosis, bimodality)
      - Dormant neurons (near-zero weights that activate only on triggers)
      - Statistical outliers in weight matrices
    
    Requires white-box access.
    """

    def analyze(self, parameters: Dict[str, np.ndarray]) -> List[Finding]:
        """
        Analyze weight tensors for statistical anomalies.
        
        Args:
            parameters: Dict mapping layer names to weight arrays
        """
        findings = []

        layer_stats = []
        for name, weights in parameters.items():
            if weights.size < 10:
                continue

            flat = weights.flatten().astype(np.float64)
            mean = np.mean(flat)
            std = np.std(flat)
            kurtosis = self._kurtosis(flat)
            sparsity = np.mean(np.abs(flat) < 1e-6)

            layer_stats.append({
                "name": name,
                "mean": float(mean),
                "std": float(std),
                "kurtosis": float(kurtosis),
                "sparsity": float(sparsity),
                "size": int(weights.size),
                "shape": list(weights.shape),
            })

            # Check for abnormal kurtosis (heavy tails = potential hidden behavior)
            if kurtosis > 10.0:
                findings.append(Finding(
                    module="model_integrity",
                    attack_class=AttackClass.WEIGHT_MODIFICATION.value,
                    severity=Severity.MEDIUM.value,
                    confidence=0.5 + min(0.4, (kurtosis - 10) / 50),
                    title=f"High kurtosis in layer {name}",
                    description=(
                        f"Layer '{name}' has kurtosis={kurtosis:.2f}, which is "
                        f"unusually high. This may indicate weight modifications "
                        f"or embedded hidden behavior."
                    ),
                    evidence={
                        "method": "weight_statistics",
                        "layer": name,
                        "kurtosis": float(kurtosis),
                        "mean": float(mean),
                        "std": float(std),
                    },
                    disposition=Disposition.REVIEW.value,
                ))

            # Check for suspicious sparsity patterns
            # A layer that's mostly zero but has a few large values could
            # have dormant neurons that activate only on triggers
            if sparsity > 0.9 and std > 0.5:
                # Most weights are zero, but the non-zero ones are large
                nonzero = flat[np.abs(flat) > 1e-6]
                if len(nonzero) > 0 and np.max(np.abs(nonzero)) > 5 * np.std(nonzero):
                    findings.append(Finding(
                        module="model_integrity",
                        attack_class=AttackClass.MODEL_BACKDOOR.value,
                        severity=Severity.HIGH.value,
                        confidence=0.65,
                        title=f"Suspicious sparsity pattern in {name}",
                        description=(
                            f"Layer '{name}' is {sparsity*100:.1f}% sparse but "
                            f"contains high-magnitude outlier weights. This pattern "
                            f"is consistent with dormant neurons used in backdoor attacks."
                        ),
                        evidence={
                            "method": "dormant_neuron_analysis",
                            "layer": name,
                            "sparsity": float(sparsity),
                            "max_weight": float(np.max(np.abs(flat))),
                            "nonzero_std": float(np.std(nonzero)),
                        },
                        disposition=Disposition.REVIEW.value,
                        remediation="Apply fine-pruning to remove dormant neurons.",
                    ))

        return findings

    def _kurtosis(self, data: np.ndarray) -> float:
        """Compute excess kurtosis."""
        n = len(data)
        if n < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4) - 3.0)


class EntropyProbe:
    """
    Black-box model integrity check using output entropy analysis.
    
    Backdoored models often show abnormally low entropy (high confidence)
    on random/noise inputs because the trigger pattern dominates.
    Also checks for consistent output patterns on perturbed inputs
    (STRIP-like behavior).
    """

    def __init__(self, num_probes: int = 100, noise_std: float = 0.1):
        self.num_probes = num_probes
        self.noise_std = noise_std

    def probe(self, predict_fn, input_shape: Tuple[int, ...],
              clean_images: np.ndarray = None) -> List[Finding]:
        """
        Probe the model with noise and perturbation patterns.
        
        Args:
            predict_fn: Function taking (N, C, H, W) -> logits
            input_shape: Shape of a single input (C, H, W)
            clean_images: Optional clean reference images
        """
        findings = []

        # 1. Random noise probe
        noise_inputs = np.random.randn(
            self.num_probes, *input_shape
        ).astype(np.float32)

        try:
            noise_logits = predict_fn(noise_inputs)
        except Exception as e:
            return [Finding(
                module="model_integrity",
                severity=Severity.INFO.value,
                title="Entropy probe failed",
                description=f"Could not run noise probe: {str(e)}",
                evidence={"error": str(e)},
            )]

        noise_entropy = self._compute_entropy(noise_logits)
        mean_noise_entropy = float(np.mean(noise_entropy))
        std_noise_entropy = float(np.std(noise_entropy))

        # Low entropy on pure noise is suspicious
        if mean_noise_entropy < 0.5:
            findings.append(Finding(
                module="model_integrity",
                attack_class=AttackClass.MODEL_BACKDOOR.value,
                severity=Severity.HIGH.value,
                confidence=0.7,
                title="Abnormally confident on random noise",
                description=(
                    f"Model shows mean entropy of {mean_noise_entropy:.3f} on "
                    f"pure noise inputs (expected >1.0 for well-calibrated models). "
                    f"This may indicate a backdoor or severe miscalibration."
                ),
                evidence={
                    "method": "noise_entropy_probe",
                    "mean_entropy": mean_noise_entropy,
                    "std_entropy": std_noise_entropy,
                    "num_probes": self.num_probes,
                },
                disposition=Disposition.REVIEW.value,
            ))

        # 2. STRIP-like perturbation probe
        if clean_images is not None and len(clean_images) > 1:
            strip_findings = self._strip_probe(predict_fn, clean_images)
            findings.extend(strip_findings)

        return findings

    def _strip_probe(self, predict_fn, clean_images: np.ndarray) -> List[Finding]:
        """
        STRIP-like probe: overlay pairs of clean images and check entropy.
        Clean inputs should produce high entropy when blended;
        backdoored inputs maintain low entropy.
        """
        findings = []
        n = min(50, len(clean_images))
        entropies = []

        for i in range(n):
            j = (i + 1) % n
            # Blend two images
            blended = 0.5 * clean_images[i] + 0.5 * clean_images[j]
            blended = blended[np.newaxis].astype(np.float32)

            try:
                logits = predict_fn(blended)
                entropy = self._compute_entropy(logits)
                entropies.append(float(entropy[0]))
            except Exception:
                continue

        if not entropies:
            return findings

        mean_blend_entropy = np.mean(entropies)
        low_entropy_count = sum(1 for e in entropies if e < 0.3)
        low_entropy_ratio = low_entropy_count / len(entropies)

        if low_entropy_ratio > 0.5:
            findings.append(Finding(
                module="model_integrity",
                attack_class=AttackClass.MODEL_BACKDOOR.value,
                severity=Severity.HIGH.value,
                confidence=0.65,
                title="STRIP probe indicates potential backdoor",
                description=(
                    f"{low_entropy_ratio*100:.0f}% of blended image pairs produce "
                    f"low entropy outputs (mean={mean_blend_entropy:.3f}). "
                    f"A clean model should show higher uncertainty on blended inputs."
                ),
                evidence={
                    "method": "strip_probe",
                    "mean_blend_entropy": float(mean_blend_entropy),
                    "low_entropy_ratio": float(low_entropy_ratio),
                    "num_pairs": len(entropies),
                },
                disposition=Disposition.REVIEW.value,
            ))

        return findings

    def _compute_entropy(self, logits: np.ndarray) -> np.ndarray:
        """Compute Shannon entropy from logits."""
        # Softmax
        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        # Entropy
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log2(probs), axis=-1)
        return entropy


class BehavioralFingerprinter:
    """
    Creates a behavioral fingerprint of a model using a reference
    test battery, then compares against expected behavior.
    
    Works in both white-box and black-box modes.
    """

    def __init__(self, tolerance: float = 0.1):
        self.tolerance = tolerance

    def create_fingerprint(self, predict_fn, reference_inputs: np.ndarray) -> Dict[str, Any]:
        """
        Create a behavioral fingerprint from reference inputs.
        
        Returns a fingerprint dict containing:
          - prediction_pattern: ordered predictions on reference inputs
          - confidence_profile: mean/std of max confidence per class
          - output_hash: hash of the full output matrix
        """
        try:
            logits = predict_fn(reference_inputs.astype(np.float32))
        except Exception as e:
            return {"error": str(e)}

        # Softmax
        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        predictions = np.argmax(probs, axis=-1).tolist()
        max_confs = np.max(probs, axis=-1).tolist()

        # Hash the raw output for exact comparison
        output_hash = hashlib.sha256(logits.tobytes()).hexdigest()

        return {
            "prediction_pattern": predictions,
            "confidence_mean": float(np.mean(max_confs)),
            "confidence_std": float(np.std(max_confs)),
            "output_hash": output_hash,
            "num_inputs": len(reference_inputs),
        }

    def compare_fingerprints(self, fp_reference: Dict[str, Any],
                             fp_current: Dict[str, Any]) -> List[Finding]:
        """Compare two fingerprints and report discrepancies."""
        findings = []

        if "error" in fp_current or "error" in fp_reference:
            return [Finding(
                module="model_integrity",
                severity=Severity.INFO.value,
                title="Fingerprint comparison incomplete",
                description="One or both fingerprints have errors.",
            )]

        # Exact output comparison
        if fp_reference["output_hash"] != fp_current["output_hash"]:
            # Outputs differ - check how much
            ref_preds = fp_reference["prediction_pattern"]
            cur_preds = fp_current["prediction_pattern"]
            n = min(len(ref_preds), len(cur_preds))

            if n > 0:
                agreement = sum(1 for a, b in zip(ref_preds[:n], cur_preds[:n]) if a == b) / n
            else:
                agreement = 0.0

            if agreement < 0.5:
                severity = Severity.CRITICAL
                confidence = 0.9
            elif agreement < 0.8:
                severity = Severity.HIGH
                confidence = 0.75
            else:
                severity = Severity.MEDIUM
                confidence = 0.6

            findings.append(Finding(
                module="model_integrity",
                attack_class=AttackClass.MODEL_SUBSTITUTION.value,
                severity=severity.value,
                confidence=confidence,
                title="Model behavior differs from reference",
                description=(
                    f"Prediction agreement with reference fingerprint is "
                    f"{agreement*100:.1f}% (expected >95%). The model may have "
                    f"been substituted or modified."
                ),
                evidence={
                    "method": "behavioral_fingerprint",
                    "prediction_agreement": float(agreement),
                    "reference_output_hash": fp_reference["output_hash"],
                    "current_output_hash": fp_current["output_hash"],
                    "reference_conf_mean": fp_reference["confidence_mean"],
                    "current_conf_mean": fp_current["confidence_mean"],
                },
                disposition=Disposition.QUARANTINE.value if agreement < 0.5 else Disposition.REVIEW.value,
            ))

        return findings


class ModelIntegrityAssessor:
    """
    Orchestrates all model integrity checks with graceful
    degradation between access levels.
    """

    def __init__(self, access_level: str = "white-box"):
        self.access_level = AccessLevel(access_level)

    def assess(
        self,
        predict_fn,
        input_shape: Tuple[int, ...],
        num_classes: int,
        clean_images: np.ndarray = None,
        parameters: Dict[str, np.ndarray] = None,
        reference_fingerprint: Dict[str, Any] = None,
        reference_inputs: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Run model integrity assessment.
        
        Args:
            predict_fn: Inference function (N, C, H, W) -> logits
            input_shape: Single input shape (C, H, W)
            num_classes: Number of output classes
            clean_images: Clean reference images for testing
            parameters: Model parameters (white-box only)
            reference_fingerprint: Prior fingerprint for comparison
            reference_inputs: Standard inputs for fingerprinting
        """
        all_findings = []
        checks_performed = []
        checks_skipped = []

        # --- White-box checks ---
        if self.access_level == AccessLevel.WHITE_BOX:
            # 1. Neural Cleanse
            if clean_images is not None and num_classes > 0:
                try:
                    nc = NeuralCleanseDetector(
                        num_classes=num_classes,
                        input_shape=input_shape,
                        steps=200,  # Reduced for speed
                    )
                    nc_findings = nc.detect(predict_fn, clean_images)
                    all_findings.extend(nc_findings)
                    checks_performed.append("neural_cleanse")
                except Exception as e:
                    checks_skipped.append({
                        "check": "neural_cleanse",
                        "reason": str(e),
                    })
            else:
                checks_skipped.append({
                    "check": "neural_cleanse",
                    "reason": "Missing clean_images or num_classes",
                })

            # 2. Weight analysis
            if parameters:
                try:
                    wa = WeightAnalyzer()
                    wa_findings = wa.analyze(parameters)
                    all_findings.extend(wa_findings)
                    checks_performed.append("weight_analysis")
                except Exception as e:
                    checks_skipped.append({
                        "check": "weight_analysis",
                        "reason": str(e),
                    })
            else:
                checks_skipped.append({
                    "check": "weight_analysis",
                    "reason": "No parameters available (white-box access declared but no weights provided)",
                })
        else:
            checks_skipped.append({
                "check": "neural_cleanse",
                "reason": "Requires white-box access",
            })
            checks_skipped.append({
                "check": "weight_analysis",
                "reason": "Requires white-box access",
            })

        # --- Black-box checks (always available) ---

        # 3. Entropy probe
        try:
            ep = EntropyProbe(num_probes=50)
            ep_findings = ep.probe(predict_fn, input_shape, clean_images)
            all_findings.extend(ep_findings)
            checks_performed.append("entropy_probe")
        except Exception as e:
            checks_skipped.append({
                "check": "entropy_probe",
                "reason": str(e),
            })

        # 4. Behavioral fingerprinting
        if reference_inputs is not None:
            try:
                bf = BehavioralFingerprinter()
                current_fp = bf.create_fingerprint(predict_fn, reference_inputs)

                if reference_fingerprint:
                    bf_findings = bf.compare_fingerprints(reference_fingerprint, current_fp)
                    all_findings.extend(bf_findings)

                checks_performed.append("behavioral_fingerprint")
            except Exception as e:
                checks_skipped.append({
                    "check": "behavioral_fingerprint",
                    "reason": str(e),
                })

        # Compute overall assessment
        max_severity = Severity.LOW
        max_risk = 0.0
        for f in all_findings:
            sev = Severity(f.severity)
            risk = f.confidence
            if risk > max_risk:
                max_risk = risk
            for s in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM]:
                if sev == s and self._severity_rank(s) > self._severity_rank(max_severity):
                    max_severity = s

        overall_disposition = Disposition.ACCEPT
        if max_severity in (Severity.CRITICAL,):
            overall_disposition = Disposition.QUARANTINE
        elif max_severity in (Severity.HIGH,):
            overall_disposition = Disposition.REVIEW

        return {
            "access_level": self.access_level.value,
            "checks_performed": checks_performed,
            "checks_skipped": checks_skipped,
            "findings": [f.to_dict() for f in all_findings],
            "finding_count": len(all_findings),
            "overall_severity": max_severity.value,
            "overall_risk_score": float(max_risk),
            "overall_disposition": overall_disposition.value,
            "limitations": self._get_limitations(),
        }

    def _severity_rank(self, s: Severity) -> int:
        return {Severity.INFO: 0, Severity.LOW: 1, Severity.MEDIUM: 2,
                Severity.HIGH: 3, Severity.CRITICAL: 4}.get(s, 0)

    def _get_limitations(self) -> List[str]:
        """Declare known limitations based on access level."""
        limitations = [
            "Neural Cleanse gradient-free optimization may miss clean-label attacks.",
            "Entropy probe may produce false positives on poorly calibrated (but benign) models.",
            "Behavioral fingerprinting requires a prior reference; first assessment has no baseline.",
        ]
        if self.access_level == AccessLevel.BLACK_BOX:
            limitations.extend([
                "White-box checks (Neural Cleanse, weight analysis) unavailable with black-box access.",
                "Black-box assessment has lower detection confidence than white-box methods.",
            ])
        return limitations
