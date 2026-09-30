"""Data-side 64x64 patch-local chroma residual. Raw score, no threshold.

Run on float RGB [0,1] images before per-image histogram compression.
The maximum includes the position search; calibrate the same maximum on clean images.
This flags localized unnatural high-frequency chroma, not backdoor intent.
"""
from __future__ import annotations
import numpy as np


def patch_local_scores(images: np.ndarray, window: int = 10, batch_size: int = 128) -> np.ndarray:
    x = np.asarray(images)
    if x.ndim != 4:
        raise ValueError('images must be NHWC or NCHW')
    if x.shape[1] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        x = x.transpose(0, 2, 3, 1)
    n, h, w, c = x.shape
    if c < 3 or window < 4 or min(h, w) < window:
        raise ValueError('need RGB images and 4 <= window <= min(H,W)')
    out = np.empty(n, np.float64)
    for start in range(0, n, batch_size):
        z = x[start:start + batch_size, :, :, :3].astype(np.float32)
        # uint8 input and float [0,1] are both supported; reject malformed floats.
        if x.dtype.kind in 'ui':
            z /= 255.0
        elif not (np.isfinite(z).all() and z.min() >= 0 and z.max() <= 1):
            raise ValueError('float images must be finite in [0,1]')
        # Two chroma opponent channels; 4-neighbour high-pass rejects smooth scene colour.
        a = z[..., 0] - z[..., 2]
        b = z[..., 1] - .5 * (z[..., 0] + z[..., 2])
        energy = np.zeros(a.shape, np.float32)
        for ch in (a, b):
            r = ch[:, 1:-1, 1:-1]
            hp = r - .25 * (ch[:, :-2, 1:-1] + ch[:, 2:, 1:-1] +
                            ch[:, 1:-1, :-2] + ch[:, 1:-1, 2:])
            energy[:, 1:-1, 1:-1] += hp * hp
        # Interior window excludes its edge, so a normal object's sharp boundary
        # alone cannot make the score large. Integral-image pooling avoids a huge
        # (N,H,W,window,window) temporary. Search the SAME locations on null/test.
        k = window - 2
        inner = energy[:, 1:-1, 1:-1]
        sat = np.pad(inner, ((0, 0), (1, 0), (1, 0)))
        sat = sat.cumsum(axis=1, dtype=np.float64).cumsum(axis=2, dtype=np.float64)
        pooled = (sat[:, k:, k:] - sat[:, :-k, k:] -
                  sat[:, k:, :-k] + sat[:, :-k, :-k]) / (k * k)
        out[start:start + len(z)] = pooled.reshape(len(z), -1).max(axis=1)
    return out
