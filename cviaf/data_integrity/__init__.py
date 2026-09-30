"""
Training Data Integrity module for the CV Integrity Assurance Framework.

Detects five classes of training data attacks:
  1. Trigger/backdoor injection (BadNets patches, blended triggers)
  2. Label flipping and systematic mislabeling
  3. Near-duplicate flooding
  4. Out-of-distribution sample insertion
  5. Orchestrated assessment combining all detectors

Uses only numpy and scikit-learn. No torch, PIL, or cv2.
"""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from cviaf.core.types import (
    AttackClass,
    Disposition,
    Finding,
    SampleMetadata,
    Severity,
    normalize_metadata,
)

__all__ = [
    "TriggerDetector",
    "LabelIntegrityChecker",
    "DuplicateDetector",
    "OODDetector",
    "DataIntegrityAssessor",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODULE = "data_integrity"


def _finding_id() -> str:
    return str(uuid.uuid4())[:8]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_nhwc(images: np.ndarray) -> np.ndarray:
    """Normalize image arrays to (N, H, W, C) layout.

    Accepts (N, H, W, C) or (N, C, H, W).  Returns a view or copy in
    NHWC order so downstream analysis is layout-agnostic.
    """
    if images.ndim != 4:
        raise ValueError(
            f"Expected 4-D image array (N, H, W, C) or (N, C, H, W), got shape {images.shape}"
        )
    # Heuristic: if axis-1 is small (1, 3, or 4) and axis-3 is large,
    # treat it as NCHW.
    if images.shape[1] in (1, 3, 4) and images.shape[3] not in (1, 3, 4):
        return np.transpose(images, (0, 2, 3, 1))
    return images


def _sample_ids(metadata: List[SampleMetadata]) -> List[str]:
    return [m.sample_id or m.file_path or str(i) for i, m in enumerate(metadata)]


# -----------------------------------------------------------------------
# 1. TriggerDetector
# -----------------------------------------------------------------------


class TriggerDetector:
    """Detect trigger-pattern backdoor injection in image datasets.

    Covers two attack surfaces:
    * **Patch triggers** (BadNets-style): small static pixel patches stamped
      onto poisoned images.  Detected via per-pixel variance analysis and
      FFT-based high-frequency screening.
    * **Spectral signatures**: statistical fingerprints left in learned feature
      representations by poisoned samples.  Detected via SVD correlation
      analysis per class.
    """

    # -- configuration knobs ------------------------------------------------
    LOW_VARIANCE_PERCENTILE: float = 1.0  # bottom 1 % of pixel variance
    FFT_ENERGY_THRESHOLD: float = 3.0  # z-score above mean HF energy
    SPECTRAL_STD_MULTIPLIER: float = 2.0  # outlier if > 2 sigma on top SV
    MIN_SAMPLES_PER_CLASS: int = 10

    def detect_patch_triggers(
        self,
        images: np.ndarray,
        metadata: List[SampleMetadata],
    ) -> List[Finding]:
        """Scan images for static-patch and blended triggers.

        Args:
            images: Array of shape (N, H, W, C) or (N, C, H, W), dtype
                float or uint8.
            metadata: Per-sample metadata aligned with *images*.

        Returns:
            List of ``Finding`` objects, one per suspicious sample or
            cluster of samples.
        """
        if images.size == 0 or len(metadata) == 0:
            return []

        imgs = _ensure_nhwc(images).astype(np.float64)
        n, h, w, c = imgs.shape
        if n < 3:
            return []  # need enough samples for variance analysis

        ids = _sample_ids(metadata)
        findings: List[Finding] = []

        # -- 1a. Per-pixel variance across the dataset ----------------------
        # A static patch produces near-zero variance at its pixel locations
        # across all poisoned images while being high elsewhere.
        pixel_var = np.var(imgs, axis=0)  # (H, W, C)
        mean_var = np.mean(pixel_var, axis=-1)  # (H, W), averaged over channels

        low_var_thresh = np.percentile(mean_var, self.LOW_VARIANCE_PERCENTILE)
        low_var_mask = mean_var <= max(low_var_thresh, 1e-9)  # binary mask
        low_var_pixels = int(np.sum(low_var_mask))

        # Only flag if the low-variance region is compact (plausible patch)
        # and covers a meaningful rectangle.
        patch_flagged_indices: List[int] = []
        patch_region: Optional[Dict[str, Any]] = None

        if low_var_pixels > 0:
            rows, cols = np.where(low_var_mask)
            r_min, r_max = int(rows.min()), int(rows.max())
            c_min, c_max = int(cols.min()), int(cols.max())
            region_area = (r_max - r_min + 1) * (c_max - c_min + 1)
            density = low_var_pixels / max(region_area, 1)
            # A real patch is compact (density > 0.5) and small relative to
            # the whole image.
            relative_size = region_area / (h * w)
            if density > 0.5 and 1e-4 < relative_size < 0.15:
                patch_region = {
                    "row_range": [r_min, r_max],
                    "col_range": [c_min, c_max],
                    "pixel_count": low_var_pixels,
                    "density": round(float(density), 4),
                    "relative_size": round(float(relative_size), 6),
                }
                # Identify which samples actually carry the patch: they
                # should have high structural similarity in that region.
                region_crops = imgs[:, r_min:r_max + 1, c_min:c_max + 1, :]
                flat_crops = region_crops.reshape(n, -1)
                # Pairwise cosine similarity of the crop region
                if flat_crops.shape[1] > 0:
                    norms = np.linalg.norm(flat_crops, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1.0, norms)
                    normed = flat_crops / norms
                    sim_matrix = normed @ normed.T
                    # Samples whose mean similarity to others is very high
                    # (>0.98) in the patch region are suspicious.
                    mean_sim = np.mean(sim_matrix, axis=1)
                    suspicious = np.where(mean_sim > 0.98)[0]
                    patch_flagged_indices = suspicious.tolist()

        if patch_flagged_indices and patch_region is not None:
            affected = [ids[i] for i in patch_flagged_indices]
            contributors = Counter(
                metadata[i].contributor for i in patch_flagged_indices
            )
            severity = Severity.CRITICAL if len(affected) > 10 else Severity.HIGH
            findings.append(Finding(
                finding_id=_finding_id(),
                module=_MODULE,
                attack_class=AttackClass.TRIGGER_INJECTION.value,
                severity=severity.value,
                confidence=round(float(np.mean(
                    [np.mean(cosine_similarity(
                        imgs[i, patch_region["row_range"][0]:patch_region["row_range"][1] + 1,
                             patch_region["col_range"][0]:patch_region["col_range"][1] + 1, :].reshape(1, -1),
                        imgs[patch_flagged_indices[0],
                             patch_region["row_range"][0]:patch_region["row_range"][1] + 1,
                             patch_region["col_range"][0]:patch_region["col_range"][1] + 1, :].reshape(1, -1),
                    )) for i in patch_flagged_indices]
                )), 4),
                title="Static patch trigger detected",
                description=(
                    f"A low-variance pixel region consistent with a BadNets-style "
                    f"static patch was found in {len(affected)} samples.  The "
                    f"region spans rows {patch_region['row_range']} and columns "
                    f"{patch_region['col_range']} with density "
                    f"{patch_region['density']:.2f}."
                ),
                evidence={
                    "patch_region": patch_region,
                    "flagged_count": len(affected),
                    "contributors": dict(contributors),
                },
                affected_assets=affected,
                disposition=Disposition.QUARANTINE.value,
                remediation=(
                    "Remove or re-inspect flagged samples. Verify contributor "
                    "accounts. Retrain without flagged data and compare accuracy."
                ),
                timestamp=_now(),
            ))

        # -- 1b. FFT high-frequency analysis (blended triggers) -------------
        # Blended triggers add a periodic pattern across the image, which
        # shows up as energy spikes in the high-frequency FFT bins.
        hf_energies = np.zeros(n, dtype=np.float64)
        for i in range(n):
            gray = np.mean(imgs[i], axis=-1)  # (H, W)
            fft_mag = np.abs(np.fft.fft2(gray))
            fft_shifted = np.fft.fftshift(fft_mag)
            cy, cx = h // 2, w // 2
            # Mask out the low-frequency center (inner 25 % of each axis)
            r_y, r_x = max(h // 8, 1), max(w // 8, 1)
            center_mask = np.zeros((h, w), dtype=bool)
            center_mask[cy - r_y:cy + r_y, cx - r_x:cx + r_x] = True
            hf_energy = np.sum(fft_shifted[~center_mask])
            total_energy = np.sum(fft_shifted) + 1e-12
            hf_energies[i] = hf_energy / total_energy

        mean_hf = np.mean(hf_energies)
        std_hf = np.std(hf_energies) + 1e-12
        z_scores = (hf_energies - mean_hf) / std_hf
        fft_flagged = np.where(z_scores > self.FFT_ENERGY_THRESHOLD)[0]

        if len(fft_flagged) > 0:
            affected = [ids[i] for i in fft_flagged]
            contributors = Counter(
                metadata[i].contributor for i in fft_flagged
            )
            findings.append(Finding(
                finding_id=_finding_id(),
                module=_MODULE,
                attack_class=AttackClass.TRIGGER_INJECTION.value,
                severity=Severity.HIGH.value,
                confidence=round(float(np.mean(
                    np.clip(z_scores[fft_flagged] / 10.0, 0.5, 1.0)
                )), 4),
                title="High-frequency blended trigger anomaly",
                description=(
                    f"{len(fft_flagged)} sample(s) have abnormally high energy "
                    f"in high-frequency FFT bins (z-score > "
                    f"{self.FFT_ENERGY_THRESHOLD}), consistent with blended "
                    f"trigger injection."
                ),
                evidence={
                    "z_scores": {ids[i]: round(float(z_scores[i]), 4) for i in fft_flagged},
                    "mean_hf_ratio": round(float(mean_hf), 6),
                    "std_hf_ratio": round(float(std_hf), 6),
                    "contributors": dict(contributors),
                },
                affected_assets=affected,
                disposition=Disposition.QUARANTINE.value,
                remediation=(
                    "Inspect flagged images visually and in the frequency "
                    "domain. Remove confirmed poisoned samples and audit the "
                    "contributing source."
                ),
                timestamp=_now(),
            ))

        return findings

    def detect_spectral_signatures(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        metadata: List[SampleMetadata],
    ) -> List[Finding]:
        """Detect backdoor spectral signatures in feature space via SVD.

        Spectral Signatures (Tran et al., 2018): poisoned samples correlate
        unusually strongly with the top right singular vector of the
        per-class centered feature matrix.

        Args:
            features: (N, D) feature representations.
            labels: (N,) integer class labels.
            metadata: Per-sample metadata.

        Returns:
            Findings for classes that contain statistical outliers.
        """
        if features.size == 0 or labels.size == 0:
            return []

        n, d = features.shape
        ids = _sample_ids(metadata)
        unique_labels = np.unique(labels)
        findings: List[Finding] = []

        for lbl in unique_labels:
            mask = labels == lbl
            class_idx = np.where(mask)[0]
            if len(class_idx) < self.MIN_SAMPLES_PER_CLASS:
                continue

            class_feats = features[class_idx].astype(np.float64)
            centered = class_feats - np.mean(class_feats, axis=0)

            # Truncated SVD: only need the top singular vector.
            try:
                _, s, vt = np.linalg.svd(centered, full_matrices=False)
            except np.linalg.LinAlgError:
                continue

            top_v = vt[0]  # (D,)
            correlations = centered @ top_v  # projection scores

            mu = np.mean(correlations)
            sigma = np.std(correlations) + 1e-12
            outlier_scores = np.abs(correlations - mu) / sigma

            outlier_mask = outlier_scores > self.SPECTRAL_STD_MULTIPLIER
            outlier_idx = class_idx[outlier_mask]

            if len(outlier_idx) == 0:
                continue

            affected = [ids[i] for i in outlier_idx]
            contributors = Counter(
                metadata[i].contributor for i in outlier_idx
            )
            # Higher ratio of outliers in a class -> higher severity
            outlier_ratio = len(outlier_idx) / len(class_idx)
            if outlier_ratio > 0.15:
                severity = Severity.CRITICAL
            elif outlier_ratio > 0.05:
                severity = Severity.HIGH
            else:
                severity = Severity.MEDIUM

            findings.append(Finding(
                finding_id=_finding_id(),
                module=_MODULE,
                attack_class=AttackClass.TRIGGER_INJECTION.value,
                severity=severity.value,
                confidence=round(float(np.mean(np.clip(
                    outlier_scores[outlier_mask] / 5.0, 0.5, 1.0
                ))), 4),
                title=f"Spectral signature anomaly in class {lbl}",
                description=(
                    f"{len(outlier_idx)} of {len(class_idx)} samples in "
                    f"class {lbl} show high correlation with the top singular "
                    f"vector ({outlier_ratio:.1%} of the class), indicating "
                    f"a potential backdoor spectral signature."
                ),
                evidence={
                    "class_label": int(lbl),
                    "class_size": int(len(class_idx)),
                    "outlier_count": int(len(outlier_idx)),
                    "outlier_ratio": round(float(outlier_ratio), 4),
                    "top_singular_value": round(float(s[0]), 4),
                    "singular_value_ratio": round(float(s[0] / (s[1] + 1e-12)), 4),
                    "max_outlier_score": round(float(np.max(outlier_scores[outlier_mask])), 4),
                    "contributors": dict(contributors),
                },
                affected_assets=affected,
                disposition=Disposition.QUARANTINE.value,
                remediation=(
                    "Apply spectral signature defense: remove the top "
                    "singular-vector-correlated samples and retrain. Audit "
                    "the contributing data sources."
                ),
                timestamp=_now(),
            ))

        return findings


# -----------------------------------------------------------------------
# 2. LabelIntegrityChecker
# -----------------------------------------------------------------------


class LabelIntegrityChecker:
    """Detect label flipping and systematic mislabeling.

    Uses a confident-learning approach (similar to Cleanlab): train a
    simple classifier, obtain cross-validated probability estimates, and
    flag samples whose predicted label disagrees with the given label at
    high confidence.  Systematic patterns (e.g., all class A->B flips
    from one contributor) are surfaced separately.
    """

    CONFIDENCE_THRESHOLD: float = 0.7
    MIN_SAMPLES: int = 20
    CV_FOLDS: int = 5

    def check_label_consistency(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        metadata: List[SampleMetadata],
    ) -> List[Finding]:
        """Run confident-learning label audit.

        Args:
            features: (N, D) feature matrix.
            labels: (N,) integer class labels.
            metadata: Per-sample metadata.

        Returns:
            Findings for individual label errors and systematic patterns.
        """
        if features.size == 0 or labels.size == 0 or len(metadata) == 0:
            raise ValueError("label integrity unavailable: empty features, labels, or metadata")

        n = features.shape[0]
        ids = _sample_ids(metadata)
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)

        if n < self.MIN_SAMPLES or n_classes < 2:
            raise ValueError("label integrity unavailable: insufficient samples or classes")

        # Ensure labels are contiguous 0..K-1 for the classifier
        label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        mapped_labels = np.array([label_map[l] for l in labels])

        # Fit the scaler inside each CV fold: never leak validation features.

        # Choose classifier: KNN for small / medium datasets, logistic
        # regression for larger ones.
        if n < 5000:
            clf = KNeighborsClassifier(
                n_neighbors=min(10, n // n_classes),
                weights="distance",
            )
        else:
            clf = LogisticRegression(
                max_iter=500,
                solver="lbfgs",
                multi_class="multinomial",
                C=1.0,
            )

        # Cross-validated predicted probabilities
        cv_folds = min(self.CV_FOLDS, min(Counter(mapped_labels).values()))
        if cv_folds < 2:
            raise ValueError("label integrity unavailable: fewer than two examples in a class")

        try:
            proba = cross_val_predict(
                make_pipeline(StandardScaler(), clf), features, mapped_labels,
                cv=cv_folds, method="predict_proba"
            )
        except Exception as exc:
            raise RuntimeError("label integrity cross-validation failed") from exc

        predicted_labels = np.argmax(proba, axis=1)
        predicted_conf = np.max(proba, axis=1)

        # Find mismatches where the classifier is confident
        mismatch = predicted_labels != mapped_labels
        confident = predicted_conf >= self.CONFIDENCE_THRESHOLD
        flagged_mask = mismatch & confident
        flagged_idx = np.where(flagged_mask)[0]

        findings: List[Finding] = []

        if len(flagged_idx) == 0:
            return findings

        # Build a "noisy label confusion matrix": given -> predicted
        confusion: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for i in flagged_idx:
            given = int(mapped_labels[i])
            pred = int(predicted_labels[i])
            confusion[(given, pred)].append(int(i))

        # -- Individual label-error finding --
        affected = [ids[i] for i in flagged_idx]
        contributors = Counter(metadata[i].contributor for i in flagged_idx)
        error_rate = len(flagged_idx) / n

        severity = Severity.HIGH if error_rate > 0.05 else Severity.MEDIUM

        # Reverse map labels for reporting
        rev_map = {v: int(k) for k, v in label_map.items()}
        confusion_report = {
            f"{rev_map[g]}->{rev_map[p]}": len(idxs)
            for (g, p), idxs in confusion.items()
        }

        findings.append(Finding(
            finding_id=_finding_id(),
            module=_MODULE,
            attack_class=AttackClass.LABEL_FLIPPING.value,
            severity=severity.value,
            confidence=round(float(np.mean(predicted_conf[flagged_idx])), 4),
            title="Label inconsistencies detected via confident learning",
            description=(
                f"{len(flagged_idx)} of {n} samples ({error_rate:.1%}) have "
                f"labels that disagree with cross-validated predictions at "
                f">= {self.CONFIDENCE_THRESHOLD} confidence."
            ),
            evidence={
                "flagged_count": int(len(flagged_idx)),
                "total_samples": n,
                "error_rate": round(float(error_rate), 4),
                "confusion_matrix": confusion_report,
                "contributors": dict(contributors),
                "mean_confidence": round(float(np.mean(predicted_conf[flagged_idx])), 4),
            },
            affected_assets=affected,
            disposition=Disposition.REVIEW.value,
            remediation=(
                "Re-label flagged samples using a trusted annotator. "
                "Cross-reference with original source imagery."
            ),
            timestamp=_now(),
        ))

        # -- Systematic pattern detection per contributor --
        contributor_flips: Dict[str, Dict[Tuple[int, int], int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for i in flagged_idx:
            c = metadata[i].contributor
            g = int(mapped_labels[i])
            p = int(predicted_labels[i])
            contributor_flips[c][(g, p)] += 1

        for contrib, flips in contributor_flips.items():
            total_from_contrib = sum(
                1 for m in metadata if m.contributor == contrib
            )
            total_flips = sum(flips.values())
            flip_rate = total_flips / max(total_from_contrib, 1)

            if flip_rate < 0.10 or total_flips < 3:
                continue

            # Find the dominant flip direction
            dominant_pair, dominant_count = max(flips.items(), key=lambda x: x[1])

            findings.append(Finding(
                finding_id=_finding_id(),
                module=_MODULE,
                attack_class=AttackClass.SYSTEMATIC_MISLABEL.value,
                severity=Severity.CRITICAL.value if flip_rate > 0.3 else Severity.HIGH.value,
                confidence=round(float(min(flip_rate * 2, 1.0)), 4),
                title=f"Systematic mislabeling from contributor '{contrib}'",
                description=(
                    f"Contributor '{contrib}' has {total_flips} label "
                    f"inconsistencies out of {total_from_contrib} samples "
                    f"({flip_rate:.1%}). Dominant flip: class "
                    f"{rev_map[dominant_pair[0]]} -> {rev_map[dominant_pair[1]]} "
                    f"({dominant_count} times)."
                ),
                evidence={
                    "contributor": contrib,
                    "total_samples_from_contributor": total_from_contrib,
                    "total_flips": total_flips,
                    "flip_rate": round(float(flip_rate), 4),
                    "flip_directions": {
                        f"{rev_map[g]}->{rev_map[p]}": cnt
                        for (g, p), cnt in flips.items()
                    },
                    "dominant_flip": {
                        "from": rev_map[dominant_pair[0]],
                        "to": rev_map[dominant_pair[1]],
                        "count": dominant_count,
                    },
                },
                affected_assets=[
                    ids[i] for i in flagged_idx
                    if metadata[i].contributor == contrib
                ],
                disposition=Disposition.QUARANTINE.value,
                remediation=(
                    f"Quarantine all data from contributor '{contrib}'. "
                    f"Investigate whether the systematic pattern is malicious "
                    f"or the result of a labeling tool misconfiguration."
                ),
                timestamp=_now(),
            ))

        return findings


# -----------------------------------------------------------------------
# 3. DuplicateDetector
# -----------------------------------------------------------------------


class DuplicateDetector:
    """Detect exact and near-duplicate flooding in training data.

    Exact duplicates are grouped by SHA-256 file hash.  Near-duplicates
    are identified via cosine similarity on feature vectors.  Contributor
    aggregation flags sources that contribute disproportionately many
    duplicates.
    """

    # If a contributor owns more than this fraction of a duplicate
    # cluster, flag as suspicious flooding.
    CONTRIBUTOR_FLOOD_THRESHOLD: float = 0.80
    MIN_CLUSTER_SIZE: int = 3

    def detect_duplicates(
        self,
        image_hashes: List[str],
        features: np.ndarray,
        metadata: List[SampleMetadata],
        threshold: float = 0.95,
    ) -> List[Finding]:
        """Detect exact and near-duplicate clusters.

        Args:
            image_hashes: Per-sample SHA-256 hex strings.
            features: (N, D) feature vectors.
            metadata: Per-sample metadata.
            threshold: Cosine similarity threshold for near-duplicates.

        Returns:
            Findings for duplicate clusters and suspicious flooding.
        """
        if len(image_hashes) == 0 or features.size == 0 or len(metadata) == 0:
            return []

        n = features.shape[0]
        ids = _sample_ids(metadata)
        findings: List[Finding] = []

        # -- Exact duplicates by hash ------------------------------------
        hash_groups: Dict[str, List[int]] = defaultdict(list)
        for i, h in enumerate(image_hashes):
            hash_groups[h].append(i)

        exact_clusters = {
            h: idxs for h, idxs in hash_groups.items() if len(idxs) > 1
        }
        total_exact_dupes = sum(len(v) for v in exact_clusters.values())

        if exact_clusters:
            cluster_details = []
            for h, idxs in exact_clusters.items():
                contribs = Counter(metadata[i].contributor for i in idxs)
                cluster_details.append({
                    "hash": h[:16] + "...",
                    "size": len(idxs),
                    "contributors": dict(contribs),
                    "sample_ids": [ids[i] for i in idxs[:10]],  # cap for readability
                })
            findings.append(Finding(
                finding_id=_finding_id(),
                module=_MODULE,
                attack_class=AttackClass.DUPLICATE_FLOODING.value,
                severity=Severity.MEDIUM.value,
                confidence=1.0,
                title="Exact duplicate images found",
                description=(
                    f"{total_exact_dupes} images fall into "
                    f"{len(exact_clusters)} exact-duplicate clusters "
                    f"(identical SHA-256 hashes)."
                ),
                evidence={
                    "exact_cluster_count": len(exact_clusters),
                    "total_exact_duplicates": total_exact_dupes,
                    "clusters": cluster_details[:20],
                },
                affected_assets=[
                    ids[i]
                    for idxs in exact_clusters.values()
                    for i in idxs
                ],
                disposition=Disposition.REVIEW.value,
                remediation="De-duplicate the dataset. Keep one copy per unique image.",
                timestamp=_now(),
            ))

        # -- Near-duplicates via cosine similarity -------------------------
        # For large N, computing the full N x N matrix is expensive.
        # We batch the computation to keep memory bounded.
        BATCH = 2000
        near_dup_pairs: List[Tuple[int, int, float]] = []

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = features / norms

        for start in range(0, n, BATCH):
            end = min(start + BATCH, n)
            sim_block = normed[start:end] @ normed.T  # (batch, N)
            for local_i in range(end - start):
                global_i = start + local_i
                # Only upper triangle to avoid double-counting
                hits = np.where(sim_block[local_i, global_i + 1:] >= threshold)[0]
                for offset in hits:
                    j = global_i + 1 + offset
                    # Skip if they are already exact duplicates
                    if image_hashes[global_i] == image_hashes[j]:
                        continue
                    near_dup_pairs.append(
                        (global_i, j, float(sim_block[local_i, global_i + 1 + offset]))
                    )

        # Union-find to build clusters from pairs
        if near_dup_pairs:
            parent: Dict[int, int] = {}

            def find(x: int) -> int:
                while parent.get(x, x) != x:
                    parent[x] = parent.get(parent[x], parent[x])
                    x = parent[x]
                return x

            def union(a: int, b: int) -> None:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            for i, j, _ in near_dup_pairs:
                parent.setdefault(i, i)
                parent.setdefault(j, j)
                union(i, j)

            clusters_map: Dict[int, List[int]] = defaultdict(list)
            for node in parent:
                clusters_map[find(node)].append(node)

            near_clusters = {
                root: members
                for root, members in clusters_map.items()
                if len(members) >= self.MIN_CLUSTER_SIZE
            }

            if near_clusters:
                total_near = sum(len(v) for v in near_clusters.values())
                cluster_details = []
                for root, members in list(near_clusters.items())[:20]:
                    contribs = Counter(metadata[i].contributor for i in members)
                    cluster_details.append({
                        "cluster_root": ids[root],
                        "size": len(members),
                        "contributors": dict(contribs),
                        "sample_ids": [ids[i] for i in members[:10]],
                    })
                findings.append(Finding(
                    finding_id=_finding_id(),
                    module=_MODULE,
                    attack_class=AttackClass.DUPLICATE_FLOODING.value,
                    severity=Severity.MEDIUM.value,
                    confidence=round(float(np.mean([s for _, _, s in near_dup_pairs])), 4),
                    title="Near-duplicate image clusters detected",
                    description=(
                        f"{total_near} images form {len(near_clusters)} "
                        f"near-duplicate clusters (cosine sim >= {threshold})."
                    ),
                    evidence={
                        "near_cluster_count": len(near_clusters),
                        "total_near_duplicates": total_near,
                        "similarity_threshold": threshold,
                        "clusters": cluster_details,
                    },
                    affected_assets=[
                        ids[i]
                        for members in near_clusters.values()
                        for i in members
                    ],
                    disposition=Disposition.REVIEW.value,
                    remediation=(
                        "Review near-duplicate clusters.  Keep the "
                        "highest-quality version and remove others to avoid "
                        "biasing the model."
                    ),
                    timestamp=_now(),
                ))

            # -- Contributor flooding check --------------------------------
            all_clusters = {**exact_clusters, **near_clusters}
            contributor_dup_count: Dict[str, int] = defaultdict(int)
            contributor_total: Dict[str, int] = Counter(
                m.contributor for m in metadata
            )

            for cluster_members in all_clusters.values():
                contribs = Counter(
                    metadata[i].contributor for i in cluster_members
                )
                for contrib, cnt in contribs.items():
                    ratio = cnt / len(cluster_members)
                    if ratio >= self.CONTRIBUTOR_FLOOD_THRESHOLD:
                        contributor_dup_count[contrib] += cnt

            for contrib, dup_cnt in contributor_dup_count.items():
                total = contributor_total.get(contrib, 1)
                dup_rate = dup_cnt / total
                if dup_rate < 0.15 or dup_cnt < 5:
                    continue
                findings.append(Finding(
                    finding_id=_finding_id(),
                    module=_MODULE,
                    attack_class=AttackClass.DUPLICATE_FLOODING.value,
                    severity=Severity.HIGH.value,
                    confidence=round(float(min(dup_rate * 1.5, 1.0)), 4),
                    title=f"Suspected duplicate flooding by contributor '{contrib}'",
                    description=(
                        f"Contributor '{contrib}' dominates {dup_cnt} duplicate "
                        f"samples out of {total} total contributions "
                        f"({dup_rate:.1%}), suggesting intentional flooding."
                    ),
                    evidence={
                        "contributor": contrib,
                        "duplicate_count": dup_cnt,
                        "total_from_contributor": total,
                        "duplicate_rate": round(float(dup_rate), 4),
                    },
                    affected_assets=[
                        ids[i] for i in range(n)
                        if metadata[i].contributor == contrib
                    ],
                    disposition=Disposition.QUARANTINE.value,
                    remediation=(
                        f"Audit all submissions from '{contrib}'. Rate-limit "
                        f"future contributions and require diversity checks."
                    ),
                    timestamp=_now(),
                ))

        return findings


# -----------------------------------------------------------------------
# 4. OODDetector
# -----------------------------------------------------------------------


class OODDetector:
    """Detect out-of-distribution samples in training data.

    Uses two complementary approaches:
    * **Mahalanobis distance** from the reference distribution's fitted
      covariance.
    * **Isolation Forest** as a secondary non-parametric detector.

    Samples flagged by both methods receive a higher confidence score.
    """

    MIN_REFERENCE: int = 10

    def detect_ood(
        self,
        features: np.ndarray,
        reference_features: np.ndarray,
        metadata: List[SampleMetadata],
        contamination: float = 0.05,
    ) -> List[Finding]:
        """Score samples against a reference distribution.

        Args:
            features: (N, D) features of the samples to evaluate.
            reference_features: (M, D) features from a trusted reference
                set that defines the in-distribution.
            metadata: Per-sample metadata for *features*.
            contamination: Expected fraction of outliers (passed to
                Isolation Forest).

        Returns:
            Findings for detected OOD samples.
        """
        if (
            features.size == 0
            or reference_features.size == 0
            or len(metadata) == 0
        ):
            return []

        n = features.shape[0]
        m = reference_features.shape[0]
        if m < self.MIN_REFERENCE:
            return []

        ids = _sample_ids(metadata)
        findings: List[Finding] = []

        # Scale using the reference distribution's statistics
        scaler = StandardScaler()
        scaler.fit(reference_features)
        ref_scaled = scaler.transform(reference_features)
        feat_scaled = scaler.transform(features)

        # -- Mahalanobis distance ------------------------------------------
        mahal_scores = np.zeros(n, dtype=np.float64)
        try:
            # Robust covariance if we have enough samples relative to
            # dimensionality; fall back to empirical otherwise.
            d = ref_scaled.shape[1]
            if m > 5 * d:
                cov_estimator = MinCovDet(random_state=42)
            else:
                cov_estimator = EmpiricalCovariance()
            cov_estimator.fit(ref_scaled)
            mahal_scores = cov_estimator.mahalanobis(feat_scaled)
        except Exception:
            # If covariance estimation fails (singular matrix, etc.),
            # fall back to simple Euclidean distance from mean.
            ref_mean = np.mean(ref_scaled, axis=0)
            mahal_scores = np.sum((feat_scaled - ref_mean) ** 2, axis=1)

        # Threshold: samples beyond the 1 - contamination quantile of the
        # reference distribution's own distances.
        try:
            ref_mahal = cov_estimator.mahalanobis(ref_scaled)  # type: ignore[union-attr]
        except Exception:
            ref_mean = np.mean(ref_scaled, axis=0)
            ref_mahal = np.sum((ref_scaled - ref_mean) ** 2, axis=1)

        mahal_thresh = np.percentile(ref_mahal, 100 * (1 - contamination))
        mahal_flagged = set(np.where(mahal_scores > mahal_thresh)[0].tolist())

        # -- Isolation Forest ----------------------------------------------
        iso = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
        )
        # Train on reference, predict on evaluation set
        iso.fit(ref_scaled)
        iso_preds = iso.predict(feat_scaled)  # -1 = outlier, 1 = inlier
        iso_scores = -iso.score_samples(feat_scaled)  # higher = more anomalous
        iso_flagged = set(np.where(iso_preds == -1)[0].tolist())

        # -- Combined scoring ----------------------------------------------
        both = mahal_flagged & iso_flagged
        mahal_only = mahal_flagged - iso_flagged
        iso_only = iso_flagged - mahal_flagged
        all_flagged = mahal_flagged | iso_flagged

        if not all_flagged:
            return findings

        per_sample: Dict[str, Dict[str, Any]] = {}
        for i in sorted(all_flagged):
            in_both = i in both
            per_sample[ids[i]] = {
                "mahalanobis_distance": round(float(mahal_scores[i]), 4),
                "isolation_score": round(float(iso_scores[i]), 4),
                "flagged_by": (
                    "both" if in_both
                    else ("mahalanobis" if i in mahal_flagged else "isolation_forest")
                ),
            }

        # Confidence: flagged by both => high, one method => moderate
        confidence_map = {
            i: 0.9 if i in both else 0.6 for i in all_flagged
        }

        affected = [ids[i] for i in sorted(all_flagged)]
        contributors = Counter(metadata[i].contributor for i in all_flagged)
        mean_conf = float(np.mean(list(confidence_map.values())))

        severity = Severity.HIGH if len(both) > 5 else Severity.MEDIUM

        findings.append(Finding(
            finding_id=_finding_id(),
            module=_MODULE,
            attack_class=AttackClass.OOD_INSERTION.value,
            severity=severity.value,
            confidence=round(mean_conf, 4),
            title="Out-of-distribution samples detected",
            description=(
                f"{len(all_flagged)} sample(s) flagged as OOD: "
                f"{len(both)} flagged by both detectors, "
                f"{len(mahal_only)} by Mahalanobis only, "
                f"{len(iso_only)} by Isolation Forest only."
            ),
            evidence={
                "total_flagged": len(all_flagged),
                "both_detectors": len(both),
                "mahalanobis_only": len(mahal_only),
                "isolation_forest_only": len(iso_only),
                "contamination": contamination,
                "mahalanobis_threshold": round(float(mahal_thresh), 4),
                "per_sample_scores": per_sample,
                "contributors": dict(contributors),
            },
            affected_assets=affected,
            disposition=Disposition.QUARANTINE.value if len(both) > 0 else Disposition.REVIEW.value,
            remediation=(
                "Inspect flagged samples for relevance to the target "
                "domain. Remove confirmed OOD samples. Investigate "
                "contributing sources for intentional insertion."
            ),
            timestamp=_now(),
        ))

        # -- Per-contributor aggregation -----------------------------------
        for contrib, cnt in contributors.items():
            total_from = sum(1 for m in metadata if m.contributor == contrib)
            ood_rate = cnt / max(total_from, 1)
            if ood_rate > 0.20 and cnt >= 3:
                findings.append(Finding(
                    finding_id=_finding_id(),
                    module=_MODULE,
                    attack_class=AttackClass.OOD_INSERTION.value,
                    severity=Severity.HIGH.value,
                    confidence=round(float(min(ood_rate * 1.5, 1.0)), 4),
                    title=f"Elevated OOD rate from contributor '{contrib}'",
                    description=(
                        f"{cnt} of {total_from} samples ({ood_rate:.1%}) from "
                        f"contributor '{contrib}' are flagged as OOD."
                    ),
                    evidence={
                        "contributor": contrib,
                        "ood_count": cnt,
                        "total_from_contributor": total_from,
                        "ood_rate": round(float(ood_rate), 4),
                    },
                    affected_assets=[
                        ids[i] for i in sorted(all_flagged)
                        if metadata[i].contributor == contrib
                    ],
                    disposition=Disposition.QUARANTINE.value,
                    remediation=(
                        f"Audit all data from '{contrib}'. Consider "
                        f"restricting future submissions."
                    ),
                    timestamp=_now(),
                ))

        return findings


# -----------------------------------------------------------------------
# 5. DataIntegrityAssessor (orchestrator)
# -----------------------------------------------------------------------


class DataIntegrityAssessor:
    """Orchestrate all data-integrity checks and aggregate results.

    Runs every applicable detector, collects findings, computes per-source
    risk scores, and returns a structured assessment dictionary.
    """

    def __init__(self) -> None:
        self.trigger_detector = TriggerDetector()
        self.label_checker = LabelIntegrityChecker()
        self.duplicate_detector = DuplicateDetector()
        self.ood_detector = OODDetector()

    def assess(
        self,
        images: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        metadata: List[SampleMetadata],
        image_hashes: List[str] | None = None,
        reference_features: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        """Run all applicable integrity checks and return a combined assessment.

        Args:
            images: (N, H, W, C) or (N, C, H, W) image array.
            features: (N, D) feature representations.
            labels: (N,) integer class labels.
            metadata: Per-sample metadata list.
            image_hashes: Optional per-sample SHA-256 file hashes.  If
                omitted, exact-duplicate detection is skipped.
            reference_features: Optional (M, D) reference features for
                OOD detection.  If omitted, OOD detection is skipped.

        Returns:
            Dictionary with keys ``findings``, ``source_risks``,
            ``summary``, and ``overall_risk``.
        """
        all_findings: List[Finding] = []
        module_summaries: Dict[str, Dict[str, Any]] = {}

        # Normalize metadata (accept dicts / SampleMetadata / mixed) so every
        # downstream .contributor / .source access cannot hit plain dicts.
        metadata = normalize_metadata(metadata)

        # -- Trigger detection --
        if images is not None and images.size > 0:
            try:
                patch_findings = self.trigger_detector.detect_patch_triggers(
                    images, metadata
                )
                all_findings.extend(patch_findings)
                module_summaries["patch_triggers"] = {
                    "ran": True,
                    "finding_count": len(patch_findings),
                }
            except Exception as exc:
                module_summaries["patch_triggers"] = {
                    "ran": False,
                    "error": str(exc),
                }

        if features is not None and features.size > 0 and labels is not None:
            try:
                spectral_findings = self.trigger_detector.detect_spectral_signatures(
                    features, labels, metadata
                )
                all_findings.extend(spectral_findings)
                module_summaries["spectral_signatures"] = {
                    "ran": True,
                    "finding_count": len(spectral_findings),
                }
            except Exception as exc:
                module_summaries["spectral_signatures"] = {
                    "ran": False,
                    "error": str(exc),
                }

        # -- Label integrity --
        if features is not None and features.size > 0 and labels is not None:
            try:
                label_findings = self.label_checker.check_label_consistency(
                    features, labels, metadata
                )
                all_findings.extend(label_findings)
                module_summaries["label_integrity"] = {
                    "ran": True,
                    "finding_count": len(label_findings),
                }
            except Exception as exc:
                module_summaries["label_integrity"] = {
                    "ran": False,
                    "error": str(exc),
                }

        # -- Duplicate detection --
        if image_hashes is not None and features is not None and features.size > 0:
            try:
                dup_findings = self.duplicate_detector.detect_duplicates(
                    image_hashes, features, metadata
                )
                all_findings.extend(dup_findings)
                module_summaries["duplicates"] = {
                    "ran": True,
                    "finding_count": len(dup_findings),
                }
            except Exception as exc:
                module_summaries["duplicates"] = {
                    "ran": False,
                    "error": str(exc),
                }

        # -- OOD detection --
        if reference_features is not None and features is not None and features.size > 0:
            try:
                ood_findings = self.ood_detector.detect_ood(
                    features, reference_features, metadata
                )
                all_findings.extend(ood_findings)
                module_summaries["ood"] = {
                    "ran": True,
                    "finding_count": len(ood_findings),
                }
            except Exception as exc:
                module_summaries["ood"] = {
                    "ran": False,
                    "error": str(exc),
                }

        # -- Aggregate per-source risk scores ------------------------------
        source_risks = self._compute_source_risks(all_findings, metadata)

        # -- Overall risk --------------------------------------------------
        overall_risk = self._compute_overall_risk(all_findings)

        return {
            "findings": [f.to_dict() for f in all_findings],
            "finding_count": len(all_findings),
            "module_summaries": module_summaries,
            "source_risks": source_risks,
            "overall_risk": overall_risk,
            "timestamp": _now(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _severity_weight(severity: str) -> float:
        """Map severity string to a numeric weight for risk scoring."""
        return {
            Severity.CRITICAL.value: 10.0,
            Severity.HIGH.value: 5.0,
            Severity.MEDIUM.value: 2.0,
            Severity.LOW.value: 1.0,
            Severity.INFO.value: 0.0,
        }.get(severity, 0.0)

    def _compute_source_risks(
        self,
        findings: List[Finding],
        metadata: List[SampleMetadata],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute a risk score per contributor/source."""
        contributor_total = Counter(m.contributor for m in metadata)
        source_total = Counter(m.source for m in metadata)

        contributor_findings: Dict[str, List[Finding]] = defaultdict(list)
        source_findings: Dict[str, List[Finding]] = defaultdict(list)

        for f in findings:
            # Match affected assets back to contributors
            affected_set = set(f.affected_assets)
            for i, m in enumerate(metadata):
                sid = m.sample_id or m.file_path or str(i)
                if sid in affected_set:
                    contributor_findings[m.contributor].append(f)
                    if m.source:
                        source_findings[m.source].append(f)

        risks: Dict[str, Dict[str, Any]] = {}

        for contrib, total in contributor_total.items():
            related = contributor_findings.get(contrib, [])
            if not related:
                risks[contrib] = {
                    "type": "contributor",
                    "total_samples": total,
                    "finding_count": 0,
                    "risk_score": 0.0,
                    "risk_level": Severity.INFO.value,
                }
                continue

            weighted = sum(
                self._severity_weight(f.severity) * f.confidence
                for f in related
            )
            # Normalize by total samples from this contributor
            score = round(float(weighted / max(total, 1)), 4)
            if score >= 5.0:
                level = Severity.CRITICAL.value
            elif score >= 2.0:
                level = Severity.HIGH.value
            elif score >= 1.0:
                level = Severity.MEDIUM.value
            elif score > 0:
                level = Severity.LOW.value
            else:
                level = Severity.INFO.value

            attack_classes = list({f.attack_class for f in related})

            risks[contrib] = {
                "type": "contributor",
                "total_samples": total,
                "finding_count": len(related),
                "risk_score": score,
                "risk_level": level,
                "attack_classes": attack_classes,
            }

        for src, total in source_total.items():
            if not src:
                continue
            related = source_findings.get(src, [])
            weighted = sum(
                self._severity_weight(f.severity) * f.confidence
                for f in related
            )
            score = round(float(weighted / max(total, 1)), 4)
            if score >= 5.0:
                level = Severity.CRITICAL.value
            elif score >= 2.0:
                level = Severity.HIGH.value
            elif score >= 1.0:
                level = Severity.MEDIUM.value
            elif score > 0:
                level = Severity.LOW.value
            else:
                level = Severity.INFO.value

            risks[f"source:{src}"] = {
                "type": "source",
                "total_samples": total,
                "finding_count": len(related),
                "risk_score": score,
                "risk_level": level,
            }

        return risks

    @staticmethod
    def _compute_overall_risk(findings: List[Finding]) -> str:
        """Determine overall risk from the most severe finding."""
        if not findings:
            return Severity.INFO.value

        severity_order = [
            Severity.CRITICAL.value,
            Severity.HIGH.value,
            Severity.MEDIUM.value,
            Severity.LOW.value,
            Severity.INFO.value,
        ]
        worst = Severity.INFO.value
        for f in findings:
            if severity_order.index(f.severity) < severity_order.index(worst):
                worst = f.severity
        return worst
