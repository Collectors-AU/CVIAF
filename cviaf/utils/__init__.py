"""
Utility functions for the CVIAF framework.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def extract_features_from_images(
    images: np.ndarray,
    method: str = "pixel_stats",
    target_dim: int = 128,
) -> np.ndarray:
    """
    Extract feature vectors from images without requiring a neural network.
    
    Methods:
      - pixel_stats: Statistical features (mean, std, histogram per channel)
      - pca: PCA on flattened pixels
      - random_projection: Random projection for dimensionality reduction
    
    Args:
        images: (N, H, W, C) or (N, C, H, W)
        method: Feature extraction method
        target_dim: Output feature dimensionality
    
    Returns:
        (N, target_dim) feature array
    """
    n = images.shape[0]

    if method == "pixel_stats":
        features_list = []
        for i in range(n):
            img = images[i]
            if img.ndim == 3:
                # Per-channel statistics
                feats = []
                num_channels = img.shape[-1] if img.shape[-1] <= 4 else img.shape[0]
                for c in range(num_channels):
                    if img.shape[-1] <= 4:
                        channel = img[:, :, c].flatten()
                    else:
                        channel = img[c].flatten()

                    feats.extend([
                        np.mean(channel),
                        np.std(channel),
                        np.median(channel),
                        np.min(channel),
                        np.max(channel),
                    ])
                    # Add histogram bins
                    hist, _ = np.histogram(channel, bins=20, range=(0, 1))
                    feats.extend(hist / (len(channel) + 1e-10))

                features_list.append(feats)
            else:
                flat = img.flatten()
                feats = [np.mean(flat), np.std(flat), np.median(flat)]
                hist, _ = np.histogram(flat, bins=target_dim - 3, range=(0, 1))
                feats.extend(hist / (len(flat) + 1e-10))
                features_list.append(feats)

        features = np.array(features_list, dtype=np.float32)

        # Pad or truncate to target_dim
        if features.shape[1] < target_dim:
            padding = np.zeros((n, target_dim - features.shape[1]))
            features = np.hstack([features, padding])
        elif features.shape[1] > target_dim:
            features = features[:, :target_dim]

        return features

    elif method == "random_projection":
        from sklearn.random_projection import GaussianRandomProjection
        flat = images.reshape(n, -1)
        projector = GaussianRandomProjection(n_components=target_dim, random_state=42)
        return projector.fit_transform(flat).astype(np.float32)

    elif method == "pca":
        from sklearn.decomposition import PCA
        flat = images.reshape(n, -1).astype(np.float64)
        pca = PCA(n_components=min(target_dim, n - 1, flat.shape[1]))
        reduced = pca.fit_transform(flat).astype(np.float32)
        if reduced.shape[1] < target_dim:
            padding = np.zeros((n, target_dim - reduced.shape[1]), dtype=np.float32)
            reduced = np.hstack([reduced, padding])
        return reduced

    else:
        raise ValueError(f"Unknown feature extraction method: {method}")


def compute_image_hashes(images: np.ndarray) -> List[str]:
    """Compute SHA-256 hashes for each image in a batch."""
    return [
        hashlib.sha256(images[i].tobytes()).hexdigest()
        for i in range(len(images))
    ]


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Normalize images to [0, 1] range."""
    images = images.astype(np.float32)
    if images.max() > 1.0:
        images = images / 255.0
    return images
