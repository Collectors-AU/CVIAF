"""Real (pretrained) backbone behind the SAME detector interface as the synthetic one.

Task 3. This module is the swap the synthetic lab anticipated: ``TinyDetector``'s own
docstring says the frozen random-feature backbone "behind the same interface" is a
stand-in for a real pretrained one, "nothing in the assurance engine changes".

FROZEN INTERFACE (v1, do not redesign; finish the REMAINING items in TASK3_NOTES.md)

``RealBackboneDetector`` subclasses ``TinyDetector`` and therefore exposes exactly:

  * ``cfg``                : ``DetectorConfig`` - ``grid = img_size // 4``, ``cell = img_size / grid``
  * ``features(image)``    : ``(H, W, 3)`` float32 in [0, 1] -> ``(G, G, C2)`` standardised
  * ``features_raw``       : same but un-standardised (used once to fit ``feat_mean``/``feat_std``)
  * ``head_forward(feats)``: ``(G, G)`` objectness logit, ``(G, G, K)`` class logits, ``(G, G, 4)`` box raw
  * ``predict(image, ...)``: ``{boxes, scores, labels, cells, obj, cls}`` - inherited, unchanged
  * ``digest()`` / ``head_digest()``, ``save(path)`` / ``load(path)``, ``_backbone_params()``

Only ``features*``, ``_backbone_params`` and the (de)serialisation are overridden. The
head, the box decode, the 3x3 peak extraction, NMS and the whole postprocess are
inherited verbatim, so a real-backbone model is scorable by ``detection_quality``,
``attack_success_rate`` and every downstream battery without a code change.

DECISIONS (frozen; see TASK3_NOTES.md for the rationale)

  * backbone: ``resnet18_stem1`` - torchvision ResNet-18 truncated after ``layer1``.
    Stride 4 total (conv1 s2 + maxpool s2), 64 channels, so for ``img_size=64`` it
    yields ``G=16``, exactly ``DetectorConfig.grid``. Torchvision ResNet-50 and wider
    variants were rejected: they change ``G`` and would move the grid under the head.
  * ImageNet weights are loaded by default (``pretrained=True``) and the extractor is
    frozen unless ``finetune`` is set; a download failure degrades to random init with
    a warning rather than crashing (recorded in the manifest under ``backbone.pretrained``).
  * preprocessing: pixels arrive in [0, 1]; ImageNet normalisation is applied inside
    the extractor with torchvision's own constants on the [0, 1] scale.
  * the ONNX artefact is the FEATURE EXTRACTOR only (the numpy head is portable
    already). ``runtime="onnx"`` swaps only ``features`` for an onnxruntime session, so
    the ONNX detector's digest is unchanged and export parity is checkable.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cviaf.lab.detector import DetectorConfig, TinyDetector

REAL_BACKBONE_VERSION = "real-backbone-1"
BACKBONE_NAMES = ("resnet18_stem1",)
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
BB_PREFIX = "bb__"          # backbone state_dict keys inside weights.npz
ONNX_NAME = "features.onnx"


@dataclass
class RealBackboneConfig:
    """Everything that decides what the real backbone is. Digestable like DetectorConfig."""

    backbone: str = "resnet18_stem1"
    pretrained: bool = True
    finetune: bool = False
    runtime: str = "torch"            # torch | onnx
    img_size: int = 64
    n_classes: int = 3
    hidden: int = 48
    seed: int = 0
    normalize: str = "imagenet"

    def digest(self) -> str:
        payload = json.dumps({**asdict(self), "version": REAL_BACKBONE_VERSION}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def detector_config(self) -> DetectorConfig:
        return DetectorConfig(img_size=self.img_size, c1=16, c2=64, hidden=self.hidden,
                              n_classes=self.n_classes, seed=self.seed)


def _torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise RuntimeError(
            "torch is required for the real-backbone detector; see TASK3_NOTES.md for the "
            "interpreter that has it") from exc


def build_feature_extractor(cfg: RealBackboneConfig):
    """ResNet-18 stem+layer1 as a frozen (or finetunable) feature extractor.

    Returns the torch ``nn.Module``; it maps ``(N, 3, H, W)`` float32 -> ``(N, 64, H/4, W/4)``.
    """
    torch = _torch()
    if cfg.backbone not in BACKBONE_NAMES:
        raise ValueError(f"unknown backbone {cfg.backbone!r}; known: {BACKBONE_NAMES}")
    from torchvision.models import resnet18

    # Seed before ANY backbone construction. With pretrained weights the checkpoint
    # overwrites the initialisation, so this is invisible there -- but with
    # pretrained=False the kaiming init draws from the GLOBAL torch RNG, and an
    # unseeded draw makes two constructions of the "same" model digest differently
    # (measured: `make(0)` twice gave different bb__ conv weights and digests).
    # The declared seed is the reproducibility contract; honour it here too.
    torch.manual_seed(int(cfg.seed))

    weights = None
    if cfg.pretrained:
        try:
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.IMAGENET1K_V1
            net = resnet18(weights=weights)
        except Exception as exc:                      # offline / cache miss
            print(f"  [warn] ImageNet weights unavailable ({exc.__class__.__name__}: {exc}); "
                  f"using random init -- record this in the manifest")
            net = resnet18(weights=None)
    else:
        net = resnet18(weights=None)

    extractor = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad_(bool(cfg.finetune))
    return extractor


class OnnxFeatureExtractor:
    """onnxruntime session exposing the same feature map as the torch extractor."""

    def __init__(self, path: str):
        import onnxruntime as ort
        self.path = path
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.channels = int(self.session.get_outputs()[0].shape[1])

    def features_raw(self, batch_nchw: np.ndarray) -> np.ndarray:
        """(N, 3, H, W) float32 -> (N, C2, G, G) float32."""
        out = self.session.run([self.output_name],
                               {self.input_name: np.ascontiguousarray(batch_nchw, np.float32)})
        return np.asarray(out[0], np.float32)


class RealBackboneDetector(TinyDetector):
    """Pretrained backbone + the synthetic lab's trained numpy head, same interface."""

    def __init__(self, cfg: Optional[DetectorConfig] = None,
                 rb: Optional[RealBackboneConfig] = None,
                 extractor=None, onnx_path: Optional[str] = None):
        self.rb = rb or RealBackboneConfig()
        self.cfg = cfg or self.rb.detector_config()
        c = self.cfg
        rng = np.random.default_rng(c.seed)

        # Head: identical shapes/init to TinyDetector, sized to the real feature width.
        self.c2_real = int(self.rb.detector_config().c2)
        g = 1.0 / np.sqrt(self.c2_real)
        self.Wh = (rng.normal(0, g, (self.c2_real, c.hidden))).astype(np.float32)
        self.bh = np.zeros((c.hidden,), np.float32)
        self.wo = np.zeros((c.hidden,), np.float32)
        self.bo = np.float32(-2.0)
        self.Wc = np.zeros((c.hidden, c.n_classes), np.float32)
        self.bc = np.zeros((c.n_classes,), np.float32)
        self.Wb = np.zeros((c.hidden, 4), np.float32)
        self.bb = np.zeros((4,), np.float32)

        self.feat_mean = np.zeros((self.c2_real,), np.float32)
        self.feat_std = np.ones((self.c2_real,), np.float32)

        self.trained = False
        self.meta: Dict[str, Any] = {}
        # backbone parameter arrays kept when there is no live torch extractor (ONNX
        # runtime swap or a load that had no torch): identity must survive the swap.
        self._bb_arrays: Dict[str, np.ndarray] = {}
        self.pretrained_loaded = bool(self.rb.pretrained)
        self._extractor = extractor if extractor is not None else (
            None if onnx_path else build_feature_extractor(self.rb))
        self._onnx = OnnxFeatureExtractor(onnx_path) if onnx_path else None

    # -- backbone plumbing ------------------------------------------------- #

    @property
    def feature_channels(self) -> int:
        return int(self._onnx.channels) if self._onnx is not None else self.c2_real

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        """(H, W, 3) float32 [0, 1] -> (1, 3, img_size, img_size) float32, ImageNet-scaled."""
        x = np.asarray(image, np.float32)
        if x.ndim != 3:
            raise ValueError(f"expected (H, W, 3), got {x.shape}")
        s = self.cfg.img_size
        if x.shape[0] != s or x.shape[1] != s:
            idx = (np.arange(s) * (x.shape[0] / s)).astype(np.int64)
            idy = (np.arange(s) * (x.shape[1] / s)).astype(np.int64)
            x = x[idx][:, idy]
        mean = np.asarray(DEFAULT_MEAN, np.float32)
        std = np.asarray(DEFAULT_STD, np.float32)
        x = (x - mean[None, None, :]) / std[None, None, :]
        return np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None], np.float32)

    def features_raw(self, image: np.ndarray) -> np.ndarray:
        """(H, W, 3) float32 -> (G, G, C2) un-standardised feature map."""
        batch = self._prepare(image)
        if self._onnx is not None:
            out = self._onnx.features_raw(batch)[0]              # (C2, G, G)
        else:
            torch = _torch()
            with torch.no_grad():
                out = self._extractor(torch.from_numpy(batch)).numpy()[0]
        return np.ascontiguousarray(np.transpose(out, (1, 2, 0)), np.float32)

    def features(self, image: np.ndarray) -> np.ndarray:
        x = self.features_raw(image)
        return (x - self.feat_mean[None, None, :]) / self.feat_std[None, None, :]

    # -- parameters / serialisation ---------------------------------------- #

    def _backbone_params(self) -> List[Tuple[str, np.ndarray]]:
        out: List[Tuple[str, np.ndarray]] = [("feat_mean", self.feat_mean),
                                             ("feat_std", self.feat_std)]
        if self._extractor is not None:
            for name, tensor in self._extractor.state_dict().items():
                out.append((f"{BB_PREFIX}{name}", tensor.detach().cpu().numpy().astype(np.float32)))
        else:
            # insertion order, not sorted: the digest depends on parameter order, and
            # this order is the state_dict order recorded in `bb_order` at save time
            out.extend(self._bb_arrays.items())
        return out

    def save(self, path: str) -> str:
        payload: Dict[str, np.ndarray] = {name: arr for name, arr in self._params()}
        payload["_meta"] = np.frombuffer(json.dumps({
            "kind": "real_backbone",
            "version": REAL_BACKBONE_VERSION,
            "cfg": self.cfg.__dict__,
            "rb": asdict(self.rb),
            "trained": self.trained,
            "meta": self.meta,
            "onnx": os.path.basename(path).replace("weights.npz", ONNX_NAME)
                    if self._onnx is not None or self.rb.runtime == "onnx" else None,
            # parameter order is part of the digest, so it is recorded rather than
            # inferred from the npz member order
            "bb_order": [n for n, _ in self._backbone_params() if n.startswith(BB_PREFIX)],
        }).encode(), dtype=np.uint8)
        np.savez_compressed(path, **payload)
        return self.digest()

    @classmethod
    def load(cls, path: str) -> "RealBackboneDetector":
        z = np.load(path, allow_pickle=False)
        blob = json.loads(bytes(z["_meta"]).decode())
        if blob.get("kind") != "real_backbone":
            raise ValueError(f"{path} is not a real-backbone artefact")
        rb = RealBackboneConfig(**blob["rb"])
        cfg = DetectorConfig(**blob["cfg"])
        onnx_path = None
        if blob.get("onnx"):
            candidate = os.path.join(os.path.dirname(path), blob["onnx"])
            onnx_path = candidate if os.path.isfile(candidate) else None
        extractor = None if onnx_path else build_feature_extractor(rb)
        m = cls(cfg=cfg, rb=rb, extractor=extractor, onnx_path=onnx_path)
        for name, arr in m._params():
            if name in z:
                val = np.asarray(z[name], np.float32)
                if name == "bo":
                    m.bo = np.float32(val[0])
                else:
                    setattr(m, name, val)
        order = blob.get("bb_order") or [n for n in z.files if n.startswith(BB_PREFIX)]
        m._bb_arrays = {n: np.asarray(z[n], np.float32) for n in order if n in z.files}
        if extractor is not None and m._bb_arrays:
            extractor.load_state_dict(
                {k[len(BB_PREFIX):]: _torch_as_tensor(v) for k, v in m._bb_arrays.items()},
                strict=False)
        m.trained = bool(blob.get("trained", True))
        m.meta = blob.get("meta", {})
        return m

    # -- ONNX -------------------------------------------------------------- #

    def export_onnx(self, path: str) -> str:
        """Export the FEATURE EXTRACTOR to ONNX (static 64x64 input) and return the path."""
        torch = _torch()
        if self._extractor is None:
            raise RuntimeError("this detector has no torch extractor to export")
        s = self.cfg.img_size
        dummy = torch.zeros(1, 3, s, s, dtype=torch.float32)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._extractor.eval()
        # torch >= 2.6 defaults to the dynamo exporter, which needs `onnxscript` (not
        # installed here). The TorchScript exporter produces a correct static graph and
        # has no extra dependency, so it is pinned explicitly.
        try:
            torch.onnx.export(self._extractor, dummy, path, opset_version=17,
                              input_names=["input"], output_names=["features"],
                              dynamic_axes=None, do_constant_folding=True, dynamo=False)
        except TypeError:                                   # older torch: no dynamo kwarg
            torch.onnx.export(self._extractor, dummy, path, opset_version=17,
                              input_names=["input"], output_names=["features"],
                              dynamic_axes=None, do_constant_folding=True)
        return path

    def to_onnx(self, onnx_path: str) -> "RealBackboneDetector":
        """Same model, features served by onnxruntime. Digest is unchanged by design."""
        clone = RealBackboneDetector(cfg=self.cfg, rb=self.rb, extractor=None,
                                     onnx_path=onnx_path)
        clone._bb_arrays = {n: a for n, a in self._backbone_params() if n.startswith(BB_PREFIX)}
        for name, arr in self._params():
            if name.startswith(BB_PREFIX):
                continue
            setattr(clone, name, np.float32(arr[0]) if name == "bo" else arr)
        clone.feat_mean, clone.feat_std = self.feat_mean, self.feat_std
        clone.trained, clone.meta = self.trained, dict(self.meta)
        return clone


def _torch_as_tensor(arr: np.ndarray):
    import torch
    return torch.from_numpy(np.ascontiguousarray(arr))


def is_real_backbone_artifact(path: str) -> bool:
    """True if ``weights.npz`` was written by this module (used by ModelArtifact.load)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            if "_meta" not in z.files:
                return False
            return json.loads(bytes(z["_meta"]).decode()).get("kind") == "real_backbone"
    except Exception:
        return False


def export_parity(torch_detector: RealBackboneDetector, onnx_path: str,
                  images: np.ndarray, tol: float = 1e-4) -> Dict[str, Any]:
    """The export-parity gate: torch features vs onnxruntime features on real inputs.

    A benign re-export must show a feature delta inside float32 round-off and identical
    argmax decisions. This is the model-side analogue of the "zero substitution
    findings on a benign re-export" gate and must pass BEFORE any backdoor run.
    """
    onnx_detector = torch_detector.to_onnx(onnx_path)
    deltas: List[float] = []
    agree = 0
    for im in images:
        a = torch_detector.features_raw(im)
        b = onnx_detector.features_raw(im)
        deltas.append(float(np.max(np.abs(a - b))))
        agree += int(np.array_equal(np.argmax(a, axis=-1), np.argmax(b, axis=-1)))
    n = max(len(images), 1)
    return {
        "n_images": int(n),
        "max_feature_delta": float(max(deltas)) if deltas else None,
        "tolerance": float(tol),
        "channel_argmax_agreement": agree / n,
        "pass": bool(deltas and max(deltas) <= tol and agree == n),
        "digest_unchanged": onnx_detector.digest() == torch_detector.digest(),
    }
