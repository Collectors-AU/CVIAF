"""
A tiny, fully deterministic object detector in pure numpy.

Architecture (deliberately small; see docs/MVP_MAC.md for why)
--------------------------------------------------------------
  image (64x64x3 float32)
    -> FROZEN random conv bank, 5x5x3x16, stride 2, ReLU      -> 32x32x16
    -> FROZEN random conv bank, 3x3x16x16, stride 2, ReLU     -> 16x16x16
    -> per-cell feature standardisation (running stats, like BatchNorm)
    -> TRAINED linear head, per cell:
         objectness logit      (1)
         class logits          (K)
         box regression (raw)  (4)
    -> decode + per-class NMS

Why freeze the backbone? Three reasons, all deliberate:

1. **Speed.** Caching features for the whole dataset turns head training into a
   few matmuls, so a full model trains in seconds. That is what lets you train
   dozens of clean/backdoored variants on a laptop and get real power curves
   instead of one anecdote.
2. **Attribution.** Every model in the corpus shares the identical frozen
   backbone, so any behavioural difference is caused by the trained head — which
   is exactly where the backdoor lives. That removes a confound from the
   assurance measurements.
3. **Honesty.** A frozen random-feature backbone is a legitimate design point
   (random-feature / extreme-learning-machine models), and it is *stated* rather
   than hidden. The scaling plan replaces it with a real trainable YOLO-class
   backbone behind the same interface; nothing in the assurance engine changes.

This is a real detector: it localises (box regression), classifies (per-cell
softmax), and suppresses (NMS). It is small, not fake.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DETECTOR_VERSION = "lab-detector-v1"


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #

def conv2d_relu(
    x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 1, pad: int = 0
) -> np.ndarray:
    """Stride-``stride`` convolution via stride tricks + einsum.

    x: (H, W, Cin) -> (Ho, Wo, Cout). ``w``: (kh, kw, Cin, Cout).
    """
    if pad > 0:
        x = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
    kh, kw, _cin, _cout = w.shape
    win = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(0, 1))
    # win: (H-kh+1, W-kw+1, Cin, kh, kw)  -- reduced axes go last
    win = win[::stride, ::stride]
    win = np.ascontiguousarray(np.transpose(win, (0, 1, 3, 4, 2)))  # (Ho,Wo,kh,kw,Cin)
    out = np.einsum("ijklm,klmc->ijc", win, w)
    out += b
    return np.maximum(out, 0.0, out=out)  # ReLU in place


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def softplus(z: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def inv_softplus(y: np.ndarray) -> np.ndarray:
    y = np.maximum(y, 1e-4)
    return np.log(np.expm1(np.minimum(y, 30.0)))


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = ua + ub - inter
    return float(inter / denom) if denom > 1e-9 else 0.0


def nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_thr: float = 0.5):
    """Greedy per-class NMS. Returns (boxes, scores, labels, kept_indices)."""
    keep: List[int] = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        order = idx[np.argsort(-scores[idx])]
        while len(order):
            i = int(order[0])
            keep.append(i)
            rest = order[1:]
            if len(rest) == 0:
                break
            ious = np.array([box_iou(boxes[i], boxes[j]) for j in rest])
            order = rest[ious <= iou_thr]
    keep = np.array(sorted(keep), dtype=np.int64)
    return boxes[keep], scores[keep], labels[keep], keep


# --------------------------------------------------------------------------- #
# the detector
# --------------------------------------------------------------------------- #

@dataclass
class DetectorConfig:
    img_size: int = 64
    c1: int = 16
    c2: int = 16
    hidden: int = 48          # width of the trained head's hidden layer
    n_classes: int = 3
    epochs: int = 400
    lr: float = 0.02
    batch: int = 2048   # large batches: tiny batches make the Python loop the bottleneck
    pos_weight: float = 12.0  # weight on positive cells, normalised per batch
    ignore_radius: int = 1    # cells around a positive are excluded from the objectness loss
    weight_decay: float = 1e-5
    seed: int = 0

    @property
    def grid(self) -> int:
        return self.img_size // 4          # two stride-2 stages

    @property
    def cell(self) -> float:
        return float(self.img_size) / float(self.grid)

    def digest(self) -> str:
        payload = json.dumps({**self.__dict__, "version": DETECTOR_VERSION}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class TinyDetector:
    """Frozen random-feature backbone + trained linear detection head."""

    def __init__(self, cfg: Optional[DetectorConfig] = None):
        self.cfg = cfg or DetectorConfig()
        c = self.cfg
        rng = np.random.default_rng(c.seed)

        # Frozen backbone (seeded, therefore reproducible byte-for-byte).
        self.f1 = (rng.normal(0, 0.8, (5, 5, 3, c.c1))).astype(np.float32)
        self.b1 = (rng.normal(0, 0.05, (c.c1,))).astype(np.float32)
        self.f2 = (rng.normal(0, 0.6, (3, 3, c.c1, c.c2))).astype(np.float32)
        self.b2 = (rng.normal(0, 0.05, (c.c2,))).astype(np.float32)

        # Trained head: one hidden layer, then three linear outputs.
        # Random sub-critical init so an untrained model detects nothing.
        g = 1.0 / np.sqrt(c.c2)
        self.Wh = (rng.normal(0, g, (c.c2, c.hidden))).astype(np.float32)
        self.bh = np.zeros((c.hidden,), np.float32)
        self.wo = np.zeros((c.hidden,), np.float32)
        self.bo = np.float32(-2.0)
        self.Wc = np.zeros((c.hidden, c.n_classes), np.float32)
        self.bc = np.zeros((c.n_classes,), np.float32)
        self.Wb = np.zeros((c.hidden, 4), np.float32)
        self.bb = np.zeros((4,), np.float32)

        # Feature standardisation stats (set during fit).
        self.feat_mean = np.zeros((c.c2,), np.float32)
        self.feat_std = np.ones((c.c2,), np.float32)

        self.trained = False
        self.meta: Dict[str, Any] = {}

    # -- parameters -------------------------------------------------------- #

    def _backbone_params(self) -> List[Tuple[str, np.ndarray]]:
        return [("f1", self.f1), ("b1", self.b1), ("f2", self.f2), ("b2", self.b2),
                ("feat_mean", self.feat_mean), ("feat_std", self.feat_std)]

    def _head_params(self) -> List[Tuple[str, np.ndarray]]:
        return [("Wh", self.Wh), ("bh", self.bh),
                ("wo", self.wo), ("bo", np.asarray([self.bo], np.float32)),
                ("Wc", self.Wc), ("bc", self.bc), ("Wb", self.Wb), ("bb", self.bb)]

    def _params(self) -> List[Tuple[str, np.ndarray]]:
        return self._backbone_params() + self._head_params()

    def digest(self) -> str:
        """Content hash of every parameter, in a fixed order. Model identity."""
        h = hashlib.sha256()
        h.update(DETECTOR_VERSION.encode())
        h.update(self.cfg.digest().encode())
        for name, arr in self._params():
            h.update(name.encode())
            h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
        return h.hexdigest()

    def head_digest(self) -> str:
        """Digest of the TRAINED head only -- the part a backdoor actually changes."""
        h = hashlib.sha256()
        for name, arr in self._head_params():
            h.update(name.encode())
            h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
        return h.hexdigest()

    def save(self, path: str) -> str:
        payload = {name: arr for name, arr in self._params()}
        payload["_meta"] = np.frombuffer(
            json.dumps({"cfg": self.cfg.__dict__, "version": DETECTOR_VERSION,
                        "trained": self.trained, "meta": self.meta}).encode(), dtype=np.uint8)
        np.savez_compressed(path, **payload)
        return self.digest()

    @classmethod
    def load(cls, path: str) -> "TinyDetector":
        z = np.load(path, allow_pickle=False)
        blob = json.loads(bytes(z["_meta"]).decode())
        cfg = DetectorConfig(**blob["cfg"])
        m = cls(cfg)
        for name, arr in m._params():
            if name in z:
                val = np.asarray(z[name], dtype=np.float32)
                if name == "bo":
                    m.bo = np.float32(val[0])
                else:
                    setattr(m, name, val)
        m.trained = bool(blob.get("trained", True))
        m.meta = blob.get("meta", {})
        return m

    # -- forward ----------------------------------------------------------- #

    def features(self, image: np.ndarray) -> np.ndarray:
        """(H,W,3) float32 in [0,1] -> (G,G,C2) standardised feature map."""
        x = np.ascontiguousarray(image, dtype=np.float32)
        x = conv2d_relu(x, self.f1, self.b1, stride=2, pad=2)
        x = conv2d_relu(x, self.f2, self.b2, stride=2, pad=1)
        return (x - self.feat_mean[None, None, :]) / self.feat_std[None, None, :]

    def features_raw(self, image: np.ndarray) -> np.ndarray:
        """Un-standardised feature map (used once, to compute the running stats)."""
        x = np.ascontiguousarray(image, dtype=np.float32)
        x = conv2d_relu(x, self.f1, self.b1, stride=2, pad=2)
        return conv2d_relu(x, self.f2, self.b2, stride=2, pad=1)

    def set_feature_stats(self, raw_feats: np.ndarray) -> None:
        self.feat_mean = raw_feats.mean(axis=(0, 1, 2)).astype(np.float32)
        self.feat_std = (raw_feats.std(axis=(0, 1, 2)) + 1e-6).astype(np.float32)

    def head_forward(self, feats: np.ndarray):
        """Standardised feature map -> (obj_logit (G,G), cls_logit (G,G,K), box_raw (G,G,4)).

        The hidden layer is why a frozen random backbone still works: the objectness
        decision is what needs capacity, and it gets it here rather than in the
        (shared, frozen) feature extractor.
        """
        h = np.maximum(feats @ self.Wh + self.bh, 0.0)
        obj = h @ self.wo + self.bo
        cls = h @ self.Wc + self.bc
        box = h @ self.Wb + self.bb
        return obj, cls, box

    def decode_boxes(self, box_raw: np.ndarray) -> np.ndarray:
        """Raw per-cell box params -> (G,G,4) xyxy in pixels."""
        G = box_raw.shape[0]
        cell = self.cfg.cell
        js, is_ = np.meshgrid(np.arange(G, dtype=np.float32), np.arange(G, dtype=np.float32))
        cx = (js + 0.5) * cell + np.tanh(box_raw[..., 0]) * cell
        cy = (is_ + 0.5) * cell + np.tanh(box_raw[..., 1]) * cell
        w = softplus(box_raw[..., 2]) * cell
        h = softplus(box_raw[..., 3]) * cell
        return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)

    def predict(
        self,
        image: np.ndarray,
        score_thresh: float = 0.30,
        iou_thresh: float = 0.50,
        top_k: int = 20,
    ) -> Dict[str, np.ndarray]:
        """Full inference: image -> boxes, scores, labels."""
        feats = self.features(image)
        obj_logit, cls_logit, box_raw = self.head_forward(feats)
        return self._postprocess(obj_logit, cls_logit, box_raw, score_thresh, iou_thresh, top_k)

    def _postprocess(self, obj_logit, cls_logit, box_raw, score_thresh, iou_thresh, top_k):
        obj_p = sigmoid(obj_logit)
        cls_p = softmax(cls_logit, axis=-1)
        best_cls = np.argmax(cls_p, axis=-1)
        best_p = np.take_along_axis(cls_p, best_cls[..., None], axis=-1)[..., 0]
        score = obj_p * best_p
        boxes = self.decode_boxes(box_raw)

        # 3x3 peak extraction (the CenterNet trick). Adjacent cells often fire with
        # slightly offset boxes that do not overlap enough for NMS to merge, which
        # floods the output with duplicates on small objects. Keeping only local
        # maxima of the score map removes them at the source.
        score = self._peaks3x3(score)

        empty = {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros((0,), np.float32),
                 "labels": np.zeros((0,), np.int64), "cells": np.zeros((0, 2), np.int64),
                 "obj": obj_p, "cls": cls_p}
        flat = score.ravel()
        if flat.size == 0:
            return empty
        order = np.argsort(-flat)[: max(top_k * 4, 40)]
        keep = order[flat[order] >= score_thresh]
        if keep.size == 0:
            return empty
        G = score.shape[0]
        bi = keep // G
        bj = keep % G
        bb = boxes[bi, bj]
        ss = flat[keep].astype(np.float32)
        ll = best_cls[bi, bj].astype(np.int64)
        bb, ss, ll, sel = nms(bb, ss, ll, iou_thresh)
        cells = np.stack([bi[sel], bj[sel]], axis=1).astype(np.int64)
        return {"boxes": bb, "scores": ss, "labels": ll, "cells": cells,
                "obj": obj_p, "cls": cls_p}

    @staticmethod
    def _peaks3x3(score: np.ndarray) -> np.ndarray:
        """Zero out every cell that is not the max of its 3x3 neighbourhood."""
        G, _ = score.shape
        mx = score.copy()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                shifted = np.full_like(score, -1.0)
                si0, si1 = max(0, di), min(G, G + di)
                sj0, sj1 = max(0, dj), min(G, G + dj)
                ti0, ti1 = max(0, -di), min(G, G - di)
                tj0, tj1 = max(0, -dj), min(G, G - dj)
                shifted[si0:si1, sj0:sj1] = score[ti0:ti1, tj0:tj1]
                mx = np.maximum(mx, shifted)
        return np.where(score >= mx - 1e-9, score, 0.0)

    # -- targets and training --------------------------------------------- #

    def build_targets(
        self, boxes: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """CenterNet-style assignment: a GT box claims the cell containing its centre.

        Returns (obj (G,G), cls (G,G,K), box_raw_target (G,G,4), pos (G,G) bool).

        Cells within ``cfg.ignore_radius`` of a positive are neither positive nor
        negative: they are excluded from the objectness loss. Without this, a cell
        one step off the true centre is punished for producing a perfectly
        reasonable detection, and the model learns to suppress genuine neighbours.
        """
        G = self.cfg.grid
        cell = self.cfg.cell
        obj = np.zeros((G, G), np.float32)
        cls = np.zeros((G, G, self.cfg.n_classes), np.float32)
        bt = np.zeros((G, G, 4), np.float32)
        pos = np.zeros((G, G), bool)
        ignore = np.zeros((G, G), bool)
        for b, l in zip(boxes, labels):
            if not np.all(np.isfinite(b)):
                continue
            cx = float((b[0] + b[2]) / 2.0)
            cy = float((b[1] + b[3]) / 2.0)
            w = float(max(b[2] - b[0], 0.5))
            h = float(max(b[3] - b[1], 0.5))
            j = int(min(G - 1, max(0, cx // cell)))
            i = int(min(G - 1, max(0, cy // cell)))
            obj[i, j] = 1.0
            cls[i, j, int(l)] = 1.0
            bt[i, j, 0] = np.clip((cx - (j + 0.5) * cell) / cell, -0.95, 0.95)
            bt[i, j, 1] = np.clip((cy - (i + 0.5) * cell) / cell, -0.95, 0.95)
            bt[i, j, 2] = float(inv_softplus(np.array(w / cell)))
            bt[i, j, 3] = float(inv_softplus(np.array(h / cell)))
            pos[i, j] = True
            r = max(0, int(self.cfg.ignore_radius))
            if r:
                i0, i1 = max(0, i - r), min(G, i + r + 1)
                j0, j1 = max(0, j - r), min(G, j + r + 1)
                ignore[i0:i1, j0:j1] = True
        ignore &= ~pos
        return obj, cls, bt, pos, ignore

    def fit(
        self,
        feat_cache: np.ndarray,
        boxes_list: Sequence[np.ndarray],
        labels_list: Sequence[np.ndarray],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Train the head on cached *unstandardised* features.

        ``feat_cache``: (N, G, G, C2) from :meth:`features_raw`.

        Loss balance note: the objectness term is normalised by the SUM of cell
        weights per mini-batch rather than by the cell count. Normalising by the
        cell count is what made an earlier version fire objectness everywhere --
        3 positives in 256 cells get ~1% of the gradient, so negatives are barely
        pushed down. Balanced normalisation gives positives ~20% of the gradient
        without letting negatives vanish.
        """
        c = self.cfg
        n = feat_cache.shape[0]
        cell_feats = feat_cache.reshape(-1, c.c2)

        # Feature standardisation stats (the backbone's "running stats").
        self.feat_mean = cell_feats.mean(axis=0).astype(np.float32)
        self.feat_std = (cell_feats.std(axis=0) + 1e-6).astype(np.float32)
        Z = ((feat_cache - self.feat_mean) / self.feat_std).reshape(n, -1, c.c2)
        M = n * c.grid * c.grid
        G2 = c.grid * c.grid

        # Targets, flattened.
        OBJ = np.zeros((M,), np.float32)
        POS = np.zeros((M,), bool)
        IGN = np.zeros((M,), bool)
        CLS = np.zeros((M, c.n_classes), np.float32)
        BOX = np.zeros((M, 4), np.float32)
        for k in range(n):
            o, cl, bt, ps, ig = self.build_targets(boxes_list[k], labels_list[k])
            sl = slice(k * G2, (k + 1) * G2)
            OBJ[sl] = o.ravel()
            CLS[sl] = cl.reshape(-1, c.n_classes)
            BOX[sl] = bt.reshape(-1, 4)
            POS[sl] = ps.ravel()
            IGN[sl] = ig.ravel()

        Zf = Z.reshape(M, c.c2)
        w_scale = np.where(OBJ > 0, c.pos_weight, 1.0).astype(np.float32)
        w_scale[IGN] = 0.0        # ignore region: excluded from the objectness loss

        rng = np.random.default_rng(c.seed + 12345)
        params = {"Wh": self.Wh, "bh": self.bh, "wo": self.wo,
                  "bo": np.asarray([self.bo], np.float32), "Wc": self.Wc,
                  "bc": self.bc, "Wb": self.Wb, "bb": self.bb}
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v) for k, v in params.items()}
        beta1, beta2, eps, t = 0.9, 0.999, 1e-8, 0

        def sync() -> None:
            self.Wh = params["Wh"]; self.bh = params["bh"]
            self.wo = params["wo"]; self.bo = np.float32(params["bo"][0])
            self.Wc = params["Wc"]; self.bc = params["bc"]
            self.Wb = params["Wb"]; self.bb = params["bb"]

        history: List[float] = []
        for epoch in range(c.epochs):
            perm = rng.permutation(M)
            epoch_loss = 0.0
            for s in range(0, M, c.batch):
                idx = perm[s:s + c.batch]
                zb = Zf[idx]
                y = OBJ[idx]
                ws = w_scale[idx]
                wsum = float(ws.sum()) or 1.0
                pmask = POS[idx]

                # ---- forward pass of the head
                z1 = zb @ params["Wh"] + params["bh"]
                h = np.maximum(z1, 0.0)
                z_obj = h @ params["wo"] + params["bo"][0]
                s_obj = sigmoid(z_obj)

                # ---- objectness gradient (all cells, balanced normalisation)
                dl_obj = ws * (s_obj - y) / wsum
                dh = dl_obj[:, None] * params["wo"][None, :]
                g_wo = h.T @ dl_obj
                g_bo = np.array([dl_obj.sum()], np.float32)
                g_Wc = np.zeros_like(params["Wc"]); g_bc = np.zeros_like(params["bc"])
                g_Wb = np.zeros_like(params["Wb"]); g_bb = np.zeros_like(params["bb"])

                # ---- class + box gradients (positives only)
                npos = int(pmask.sum())
                if npos:
                    hp = h[pmask]
                    z_cls = hp @ params["Wc"] + params["bc"]
                    p = softmax(z_cls, axis=-1)
                    gt = CLS[idx][pmask]
                    dl_cls = (p - gt) / npos
                    g_Wc = hp.T @ dl_cls
                    g_bc = dl_cls.sum(axis=0)
                    dh[pmask] += dl_cls @ params["Wc"].T

                    z_box = hp @ params["Wb"] + params["bb"]
                    dl_box = np.sign(z_box - BOX[idx][pmask]) / npos
                    g_Wb = hp.T @ dl_box
                    g_bb = dl_box.sum(axis=0)
                    dh[pmask] += dl_box @ params["Wb"].T

                    epoch_loss += float(-np.log(np.maximum(
                        np.take_along_axis(p, np.argmax(gt, axis=1)[:, None], axis=1), 1e-9)).sum())

                # ---- backprop into the hidden layer
                dz1 = dh * (z1 > 0)
                g_Wh = zb.T @ dz1
                g_bh = dz1.sum(axis=0)

                grads = {"Wh": g_Wh, "bh": g_bh, "wo": g_wo, "bo": g_bo,
                         "Wc": g_Wc, "bc": g_bc, "Wb": g_Wb, "bb": g_bb}
                t += 1
                for k, gk in grads.items():
                    if c.weight_decay and k[0] in "Ww":
                        gk = gk + c.weight_decay * params[k]
                    m[k] = beta1 * m[k] + (1 - beta1) * gk
                    v[k] = beta2 * v[k] + (1 - beta2) * gk ** 2
                    params[k] = params[k] - c.lr * (m[k] / (1 - beta1 ** t)) / (
                        np.sqrt(v[k] / (1 - beta2 ** t)) + eps)
                sync()

                epoch_loss += float((ws * (-(y * np.log(np.maximum(s_obj, 1e-9)))
                                           - (1 - y) * np.log(np.maximum(1 - s_obj, 1e-9)))).sum()) / wsum
            history.append(epoch_loss / max(M, 1))
            if verbose and (epoch + 1) % max(1, c.epochs // 6) == 0:
                print(f"    epoch {epoch+1:4d}/{c.epochs}  loss {history[-1]:.4f}")

        self.trained = True
        return {"final_loss": history[-1] if history else float("nan"),
                "loss_history": history}

    def copy(self) -> "TinyDetector":
        m = TinyDetector(self.cfg)
        for name, arr in self._params():
            if name == "bo":
                m.bo = np.float32(arr[0])
            else:
                setattr(m, name, np.array(arr, copy=True))
        m.trained = self.trained
        m.meta = dict(self.meta)
        return m

    # ----------------------------------------------------------------------- #
    # weight-modification attacks
    #
    # Four DIFFERENT mechanisms, because "a weight was modified" is not one
    # attack and a detector that only ever meets one of them has not been
    # tested. Each returns a copy; the original is left untouched. Every one is
    # deterministic in its seed so the corpus stays reproducible.
    #
    # Measured note (scripts/tamper_probe.py): the original zero-mean Gaussian
    # perturbation is a WEAK attack at small scale -- it barely moved F1 and on
    # one seed it *improved* it, because adding zero-mean noise to a head is
    # closer to regularisation than to sabotage. It is kept (the framework must
    # handle the literal case of "weights were altered"), but the corpus pairs it
    # with mechanisms that have a direction and therefore a measurable effect.
    # ----------------------------------------------------------------------- #

    def tamper_head(self, scale: float = 0.05, seed: int = 0) -> "TinyDetector":
        """Unstructured: add zero-mean Gaussian noise to the head weights."""
        m = self.copy()
        rng = np.random.default_rng(seed)
        for name in ("wo", "Wc", "Wb"):
            arr = getattr(m, name)
            setattr(m, name, (arr + rng.normal(0, scale, arr.shape)).astype(np.float32))
        return m

    def tamper_bias(self, delta: float = 1.0, target: int = 0) -> "TinyDetector":
        """Targeted: lift one class's logit by ``delta`` ABSOLUTE logit units.

        The realistic insider version of a weight modification -- nobody ships a
        random tensor, they nudge one output so a class over-fires.

        ``delta`` is in logit units, deliberately, and that choice was forced by a
        measurement. The first version scaled the shift by the standard deviation of
        the class-bias vector, which sounds reasonable and is a no-op: that vector is
        small after training, so "2 sigma" moved the logit by ~0.006 and the model's
        behaviour did not change at all (F1 moved by -0.012, i.e. it *improved*).
        Severity has to be expressed in the units that actually act on predictions.
        """
        m = self.copy()
        m.bc = m.bc.copy()
        m.bc[int(target) % len(m.bc)] += np.float32(delta)
        return m

    def tamper_objectness(self, delta: float = 1.0) -> "TinyDetector":
        """Global: move the objectness bias, so the model sees more (or fewer) objects.

        The bluntest and most common real modification -- anything that re-exports a
        checkpoint with a re-calibrated detection threshold. ``delta`` is in logit
        units; the shipped bias is around -2, so +1.0 is a large but not absurd move.
        """
        m = self.copy()
        m.bo = np.float32(m.bo + np.float32(delta))
        return m

    def tamper_gain(self, gain: float = 1.5) -> "TinyDetector":
        """Calibration: scale the class head, as a re-exported / mis-calibrated copy."""
        m = self.copy()
        m.Wc = (m.Wc * np.float32(gain)).astype(np.float32)
        m.bc = (m.bc * np.float32(gain)).astype(np.float32)
        return m

    def tamper_prune(self, frac: float = 0.25, seed: int = 0) -> "TinyDetector":
        """Structural: zero the ``frac`` least-important hidden units, by |Wh| row norm.

        The supply-chain version: the artifact that arrives is a degraded or
        compressed copy of the one that was certified, not a random tensor. It is
        a real capability loss (whole hidden units vanish) and it is visible in the
        weight statistics as a shape change rather than as a magnitude change.
        """
        m = self.copy()
        rng = np.random.default_rng(seed)
        # ``Wh`` is (c2, hidden): the hidden units are its COLUMNS. Taking shape[0]
        # (the input width) pruned frac*c2 units instead of frac*hidden, and crashed
        # outright whenever c2 > hidden (the real-backbone lane, c2=64 > hidden=48).
        h = m.Wh.shape[1]
        k = int(round(float(frac) * h))
        if k <= 0:
            return m
        order = np.argsort(np.linalg.norm(m.Wh, axis=0) + 1e-6 * rng.standard_normal(h))
        drop = order[:k]
        m.Wh = m.Wh.copy(); m.bh = m.bh.copy()
        m.wo = m.wo.copy(); m.Wc = m.Wc.copy(); m.Wb = m.Wb.copy()
        m.Wh[:, drop] = 0.0
        m.bh[drop] = 0.0
        m.wo[drop] = 0.0
        m.Wc[drop] = 0.0
        m.Wb[drop] = 0.0
        return m
