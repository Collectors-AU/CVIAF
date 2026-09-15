"""
Attack Scenario Generators for Testing.

Provides reproducible methods to create representative poisoning,
backdoor, substitution, and tampering scenarios for validating
the assurance framework.

All generators produce:
  - Poisoned/modified data
  - Ground truth labels (which samples are poisoned)
  - Attack parameters for reproducibility
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cviaf.core.types import SampleMetadata


class BadNetsPoisoner:
    """
    BadNets-style trigger injection.
    Pastes a small patch pattern onto a subset of images
    and flips their labels to the target class.
    """

    def __init__(self, target_class: int = 0, poison_ratio: float = 0.1,
                 patch_size: int = 5, patch_position: str = "bottom-right",
                 patch_value: float = 1.0, seed: int = 42):
        self.target_class = target_class
        self.poison_ratio = poison_ratio
        self.patch_size = patch_size
        self.patch_position = patch_position
        self.patch_value = patch_value
        self.rng = np.random.RandomState(seed)

    def poison(self, images: np.ndarray, labels: np.ndarray,
               metadata: List[SampleMetadata] = None
               ) -> Tuple[np.ndarray, np.ndarray, List[bool], Dict[str, Any]]:
        """
        Apply BadNets poisoning.
        
        Args:
            images: (N, H, W, C) or (N, C, H, W) image array
            labels: (N,) label array
            metadata: Optional sample metadata
        
        Returns:
            (poisoned_images, poisoned_labels, poison_mask, attack_info)
        """
        n = len(images)
        num_poison = max(1, int(n * self.poison_ratio))
        poison_indices = self.rng.choice(n, num_poison, replace=False)
        poison_mask = [False] * n

        poisoned_images = images.copy()
        poisoned_labels = labels.copy()

        # Determine image format (NHWC or NCHW)
        if images.ndim == 4:
            if images.shape[1] <= 4:  # NCHW
                _, c, h, w = images.shape
                channel_last = False
            else:  # NHWC
                _, h, w, c = images.shape
                channel_last = True
        else:
            raise ValueError(f"Expected 4D image array, got shape {images.shape}")

        # Compute patch position
        ps = self.patch_size
        if self.patch_position == "bottom-right":
            y_start, x_start = h - ps - 2, w - ps - 2
        elif self.patch_position == "top-left":
            y_start, x_start = 2, 2
        elif self.patch_position == "center":
            y_start, x_start = h // 2 - ps // 2, w // 2 - ps // 2
        else:
            y_start, x_start = h - ps - 2, w - ps - 2

        for idx in poison_indices:
            if channel_last:
                poisoned_images[idx, y_start:y_start+ps, x_start:x_start+ps, :] = self.patch_value
            else:
                poisoned_images[idx, :, y_start:y_start+ps, x_start:x_start+ps] = self.patch_value

            poisoned_labels[idx] = self.target_class
            poison_mask[idx] = True

        attack_info = {
            "attack_type": "badnets",
            "target_class": self.target_class,
            "poison_ratio": self.poison_ratio,
            "num_poisoned": num_poison,
            "patch_size": ps,
            "patch_position": self.patch_position,
            "patch_value": self.patch_value,
            "poisoned_indices": poison_indices.tolist(),
        }

        return poisoned_images, poisoned_labels, poison_mask, attack_info


class BlendedTriggerPoisoner:
    """
    Blended trigger attack: overlays a subtle pattern
    (e.g., hello kitty, random noise pattern) on images
    with a low blend ratio so the trigger is nearly invisible.
    """

    def __init__(self, target_class: int = 0, poison_ratio: float = 0.1,
                 blend_ratio: float = 0.1, pattern: str = "random",
                 seed: int = 42):
        self.target_class = target_class
        self.poison_ratio = poison_ratio
        self.blend_ratio = blend_ratio
        self.pattern_type = pattern
        self.rng = np.random.RandomState(seed)
        self._pattern = None

    def _generate_pattern(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate the trigger pattern."""
        if self.pattern_type == "random":
            self._pattern = self.rng.rand(*shape).astype(np.float32)
        elif self.pattern_type == "checkerboard":
            pattern = np.zeros(shape, dtype=np.float32)
            if len(shape) == 3:
                h, w = shape[0], shape[1]
                for i in range(h):
                    for j in range(w):
                        if (i + j) % 2 == 0:
                            pattern[i, j] = 1.0
            self._pattern = pattern
        elif self.pattern_type == "stripes":
            pattern = np.zeros(shape, dtype=np.float32)
            if len(shape) == 3:
                for i in range(shape[0]):
                    if i % 4 < 2:
                        pattern[i, :] = 1.0
            self._pattern = pattern
        else:
            self._pattern = self.rng.rand(*shape).astype(np.float32)

        return self._pattern

    def poison(self, images: np.ndarray, labels: np.ndarray,
               ) -> Tuple[np.ndarray, np.ndarray, List[bool], Dict[str, Any]]:
        """Apply blended trigger poisoning."""
        n = len(images)
        num_poison = max(1, int(n * self.poison_ratio))
        poison_indices = self.rng.choice(n, num_poison, replace=False)
        poison_mask = [False] * n

        poisoned_images = images.copy()
        poisoned_labels = labels.copy()

        # Generate pattern matching single image shape
        img_shape = images.shape[1:]
        pattern = self._generate_pattern(img_shape)

        for idx in poison_indices:
            poisoned_images[idx] = (
                (1 - self.blend_ratio) * poisoned_images[idx] +
                self.blend_ratio * pattern
            )
            poisoned_labels[idx] = self.target_class
            poison_mask[idx] = True

        attack_info = {
            "attack_type": "blended_trigger",
            "target_class": self.target_class,
            "poison_ratio": self.poison_ratio,
            "num_poisoned": num_poison,
            "blend_ratio": self.blend_ratio,
            "pattern_type": self.pattern_type,
            "poisoned_indices": poison_indices.tolist(),
        }

        return poisoned_images, poisoned_labels, poison_mask, attack_info


class LabelFlipper:
    """
    Label flipping attack: changes labels of a subset of samples
    from one class to another, optionally from a specific contributor.
    """

    def __init__(self, source_class: int = 0, target_class: int = 1,
                 flip_ratio: float = 0.1, contributor_filter: str = None,
                 seed: int = 42):
        self.source_class = source_class
        self.target_class = target_class
        self.flip_ratio = flip_ratio
        self.contributor_filter = contributor_filter
        self.rng = np.random.RandomState(seed)

    def poison(self, labels: np.ndarray,
               metadata: List[SampleMetadata] = None
               ) -> Tuple[np.ndarray, List[bool], Dict[str, Any]]:
        """Apply label flipping."""
        poisoned_labels = labels.copy()
        poison_mask = [False] * len(labels)

        # Find eligible samples
        eligible = []
        for i, label in enumerate(labels):
            if label == self.source_class:
                if self.contributor_filter and metadata:
                    if metadata[i].contributor == self.contributor_filter:
                        eligible.append(i)
                else:
                    eligible.append(i)

        if not eligible:
            return poisoned_labels, poison_mask, {"attack_type": "label_flip", "num_flipped": 0}

        num_flip = max(1, int(len(eligible) * self.flip_ratio))
        flip_indices = self.rng.choice(eligible, min(num_flip, len(eligible)), replace=False)

        for idx in flip_indices:
            poisoned_labels[idx] = self.target_class
            poison_mask[idx] = True

        attack_info = {
            "attack_type": "label_flip",
            "source_class": self.source_class,
            "target_class": self.target_class,
            "flip_ratio": self.flip_ratio,
            "num_flipped": len(flip_indices),
            "contributor_filter": self.contributor_filter,
            "flipped_indices": flip_indices.tolist(),
        }

        return poisoned_labels, poison_mask, attack_info


class DuplicateFlooder:
    """
    Near-duplicate flooding attack: creates near-duplicates of selected
    images with minor perturbations, attributed to a malicious contributor.
    """

    def __init__(self, num_duplicates: int = 50, noise_std: float = 0.01,
                 malicious_contributor: str = "malicious_vendor",
                 seed: int = 42):
        self.num_duplicates = num_duplicates
        self.noise_std = noise_std
        self.malicious_contributor = malicious_contributor
        self.rng = np.random.RandomState(seed)

    def flood(self, images: np.ndarray, labels: np.ndarray,
              metadata: List[SampleMetadata] = None,
              source_indices: List[int] = None
              ) -> Tuple[np.ndarray, np.ndarray, List[SampleMetadata], List[bool], Dict[str, Any]]:
        """
        Create near-duplicate flood.
        
        Returns:
            (augmented_images, augmented_labels, augmented_metadata, is_duplicate_mask, attack_info)
        """
        n_orig = len(images)

        if source_indices is None:
            source_indices = self.rng.choice(n_orig, min(5, n_orig), replace=False).tolist()

        new_images = []
        new_labels = []
        new_metadata = []
        dup_mask = [False] * n_orig

        per_source = max(1, self.num_duplicates // len(source_indices))

        for src_idx in source_indices:
            for j in range(per_source):
                noisy = images[src_idx] + self.rng.randn(*images[src_idx].shape) * self.noise_std
                noisy = np.clip(noisy, 0, 1)
                new_images.append(noisy)
                new_labels.append(labels[src_idx])

                meta = SampleMetadata(
                    sample_id=f"dup_{src_idx}_{j}",
                    contributor=self.malicious_contributor,
                    label=str(labels[src_idx]),
                )
                if metadata:
                    meta.source = metadata[src_idx].source
                new_metadata.append(meta)

        aug_images = np.concatenate([images, np.array(new_images)], axis=0)
        aug_labels = np.concatenate([labels, np.array(new_labels)])
        aug_metadata = (metadata or [SampleMetadata() for _ in range(n_orig)]) + new_metadata
        dup_mask.extend([True] * len(new_images))

        attack_info = {
            "attack_type": "duplicate_flooding",
            "num_duplicates": len(new_images),
            "source_indices": source_indices,
            "noise_std": self.noise_std,
            "malicious_contributor": self.malicious_contributor,
        }

        return aug_images, aug_labels, aug_metadata, dup_mask, attack_info


class OODInjector:
    """
    Out-of-distribution sample injection: adds samples from a
    different distribution to the dataset.
    """

    def __init__(self, num_ood: int = 50, ood_type: str = "gaussian_noise",
                 target_label: int = 0, contributor: str = "untrusted_source",
                 seed: int = 42):
        self.num_ood = num_ood
        self.ood_type = ood_type
        self.target_label = target_label
        self.contributor = contributor
        self.rng = np.random.RandomState(seed)

    def inject(self, images: np.ndarray, labels: np.ndarray,
               metadata: List[SampleMetadata] = None
               ) -> Tuple[np.ndarray, np.ndarray, List[SampleMetadata], List[bool], Dict[str, Any]]:
        """Inject OOD samples."""
        n_orig = len(images)
        img_shape = images.shape[1:]

        if self.ood_type == "gaussian_noise":
            ood_images = self.rng.randn(self.num_ood, *img_shape).astype(np.float32)
            ood_images = np.clip(ood_images * 0.3 + 0.5, 0, 1)
        elif self.ood_type == "uniform_noise":
            ood_images = self.rng.rand(self.num_ood, *img_shape).astype(np.float32)
        elif self.ood_type == "constant":
            ood_images = np.full((self.num_ood, *img_shape), 0.5, dtype=np.float32)
        elif self.ood_type == "texture":
            # Sinusoidal pattern - looks like a texture, very different from natural images
            ood_images = np.zeros((self.num_ood, *img_shape), dtype=np.float32)
            if len(img_shape) >= 2:
                h, w = img_shape[0], img_shape[1]
                for i in range(self.num_ood):
                    freq = self.rng.uniform(5, 20)
                    phase = self.rng.uniform(0, 2 * np.pi)
                    x = np.linspace(0, freq * np.pi, w)
                    y = np.linspace(0, freq * np.pi, h)
                    XX, YY = np.meshgrid(x, y)
                    pattern = 0.5 + 0.5 * np.sin(XX + phase) * np.cos(YY - phase)
                    if len(img_shape) == 3:
                        ood_images[i] = pattern[:, :, np.newaxis]
                    elif len(img_shape) == 4:
                        ood_images[i] = pattern[np.newaxis, :, :]
        else:
            ood_images = self.rng.randn(self.num_ood, *img_shape).astype(np.float32)

        ood_labels = np.full(self.num_ood, self.target_label)
        ood_metadata = [
            SampleMetadata(
                sample_id=f"ood_{i}",
                contributor=self.contributor,
                label=str(self.target_label),
            )
            for i in range(self.num_ood)
        ]

        aug_images = np.concatenate([images, ood_images], axis=0)
        aug_labels = np.concatenate([labels, ood_labels])
        aug_metadata = (metadata or [SampleMetadata() for _ in range(n_orig)]) + ood_metadata
        ood_mask = [False] * n_orig + [True] * self.num_ood

        attack_info = {
            "attack_type": "ood_injection",
            "num_ood": self.num_ood,
            "ood_type": self.ood_type,
            "target_label": self.target_label,
            "contributor": self.contributor,
        }

        return aug_images, aug_labels, aug_metadata, ood_mask, attack_info


class InferenceTamperer:
    """
    Tampers with inference records to test provenance verification.
    """

    @staticmethod
    def tamper_output(seal_dict: Dict[str, Any], new_output_hash: str = None
                      ) -> Dict[str, Any]:
        """Modify the output hash in a seal to simulate output tampering."""
        tampered = deepcopy(seal_dict)
        tampered["output_hash"] = new_output_hash or hashlib.sha256(b"tampered").hexdigest()
        return tampered

    @staticmethod
    def tamper_input(seal_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Modify the image hash to simulate input substitution."""
        tampered = deepcopy(seal_dict)
        tampered["image_hash"] = hashlib.sha256(b"substituted_image").hexdigest()
        return tampered

    @staticmethod
    def replay_seal(seal_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Create a replay of an existing seal (same nonce)."""
        return deepcopy(seal_dict)

    @staticmethod
    def forge_seal(seal_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Forge a seal with a fake signature."""
        forged = deepcopy(seal_dict)
        forged["signature"] = secrets.token_hex(32)
        forged["seal_hash"] = hashlib.sha256(
            (forged["payload_hash"] + forged["signature"]).encode()
        ).hexdigest()
        return forged


class ModelSubstitutor:
    """
    Simulates model substitution by providing a different model's
    outputs while claiming to be the original.
    """

    @staticmethod
    def create_substitute_prediction_fn(num_classes: int, seed: int = 99):
        """
        Create a prediction function that returns different outputs
        than the original model (simulating substitution).
        """
        rng = np.random.RandomState(seed)

        def substitute_predict(inputs: np.ndarray) -> np.ndarray:
            n = inputs.shape[0]
            # Generate random but consistent logits
            key = hashlib.sha256(inputs.tobytes()[:1024]).hexdigest()
            local_rng = np.random.RandomState(int(key[:8], 16) % 2**31)
            return local_rng.randn(n, num_classes).astype(np.float32)

        return substitute_predict


def generate_test_dataset(
    num_samples: int = 500,
    num_classes: int = 10,
    image_shape: Tuple[int, ...] = (32, 32, 3),
    feature_dim: int = 128,
    num_contributors: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate a synthetic test dataset with images, labels,
    features, and metadata. Used for framework testing.
    """
    rng = np.random.RandomState(seed)

    images = rng.rand(num_samples, *image_shape).astype(np.float32)
    labels = rng.randint(0, num_classes, num_samples)

    # Generate class-conditional features (so clustering works)
    class_centers = rng.randn(num_classes, feature_dim) * 3
    features = np.zeros((num_samples, feature_dim), dtype=np.float32)
    for i in range(num_samples):
        features[i] = class_centers[labels[i]] + rng.randn(feature_dim) * 0.5

    contributors = [f"contributor_{j}" for j in range(num_contributors)]
    metadata = []
    for i in range(num_samples):
        meta = SampleMetadata(
            sample_id=f"sample_{i:05d}",
            file_path=f"images/img_{i:05d}.jpg",
            contributor=contributors[i % num_contributors],
            batch_id=f"batch_{i // 100}",
            label=str(labels[i]),
            label_id=int(labels[i]),
        )
        metadata.append(meta)

    # Perceptual hashes (simulated)
    image_hashes = [
        hashlib.sha256(images[i].tobytes()).hexdigest()
        for i in range(num_samples)
    ]

    return {
        "images": images,
        "labels": labels,
        "features": features,
        "metadata": metadata,
        "image_hashes": image_hashes,
        "num_classes": num_classes,
        "num_contributors": num_contributors,
        "class_centers": class_centers,
    }
