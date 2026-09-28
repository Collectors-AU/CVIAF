"""Frozen manifest, detector decoding, export parity, and independent ASR gate.

The benchmark does not infer a detector's output schema. The caller must supply
an explicit decoder for its organizer-defined model format. No implicit NMS,
class mapping, preprocessing, or retraining is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Protocol
import numpy as np


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def freeze_manifest(raw: Mapping, output: str | Path) -> str:
    """Write once; compare existing bytes if reused. Split membership is disjoint."""
    splits = raw['splits']
    required = {'train', 'calibration', 'development', 'effect_gate', 'final'}
    if set(splits) != required:
        raise ValueError(f'splits must be exactly {sorted(required)}')
    seen = set()
    for name, members in splits.items():
        if not members or len(members) != len(set(members)):
            raise ValueError(f'empty or duplicated IDs in {name}')
        overlap = seen.intersection(members)
        if overlap:
            raise ValueError(f'split overlap: {sorted(overlap)[:5]}')
        seen.update(members)
    for name, path in raw['artifacts'].items():
        if sha256(path) != raw['artifact_digests'][name]:
            raise ValueError(f'digest mismatch for {name}')
    payload = json.dumps(raw, sort_keys=True, separators=(',', ':')).encode()
    target = Path(output)
    if target.exists():
        if target.read_bytes() != payload:
            raise FileExistsError('frozen manifest differs; use a new benchmark ID/path')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Detections:
    boxes: np.ndarray   # N,4, xyxy in original pixels
    scores: np.ndarray  # N
    labels: np.ndarray  # N, canonical integer class ID

    def __post_init__(self):
        b = np.asarray(self.boxes)
        s = np.asarray(self.scores)
        l = np.asarray(self.labels)
        if b.ndim != 2 or b.shape[1] != 4 or s.shape != (len(b),) or l.shape != (len(b),):
            raise ValueError('decoded boxes/scores/labels have incompatible shapes')
        if not (np.issubdtype(b.dtype, np.number) and np.issubdtype(s.dtype, np.number)
                and np.issubdtype(l.dtype, np.integer)):
            raise ValueError('decoded types must be numeric boxes/scores and integer labels')
        if not (np.isfinite(b).all() and np.isfinite(s).all()):
            raise ValueError('decoded output has non-finite values')
        if (b[:, 2:] < b[:, :2]).any() or (s < 0).any() or (s > 1).any():
            raise ValueError('invalid xyxy boxes or confidence scores')


class Detector(Protocol):
    def predict(self, image: np.ndarray) -> Detections: ...


class DecodedAdapter:
    """Runtime-specific raw output + explicit, user-supplied detector decoder."""
    def __init__(self, run: Callable[[np.ndarray], object],
                 decode: Callable[[object, tuple[int, int]], Detections],
                 preprocess: Callable[[np.ndarray], np.ndarray]):
        self.run, self.decode, self.preprocess = run, decode, preprocess

    def predict(self, image: np.ndarray) -> Detections:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('expected original HWC RGB image')
        out = self.decode(self.run(self.preprocess(image)), image.shape[:2])
        if not isinstance(out, Detections):
            raise TypeError('decoder must return Detections; raw logits are not boxes')
        h, w = image.shape[:2]
        if (out.boxes < 0).any() or (out.boxes[:, [0, 2]] > w).any() or (out.boxes[:, [1, 3]] > h).any():
            raise ValueError('decoded boxes outside original image bounds')
        return out


def load_runtime(path: str, kind: str) -> Callable[[np.ndarray], object]:
    """Load a TRUSTED model only. PyTorch pickle/TorchScript can execute code."""
    if kind == 'onnx':
        import onnxruntime as ort
        session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ValueError('multiple ONNX inputs need a model-specific runtime')
        return lambda batch: session.run(None, {inputs[0].name: batch})
    if kind in ('torchscript', 'pytorch'):
        import torch
        if kind == 'torchscript':
            model = torch.jit.load(path, map_location='cpu').eval()
        else:
            # Eager model requires a trusted scripted factory in the input file;
            # generic state_dict cannot be reconstructed without an architecture.
            raise ValueError('provide a constructed eager torch.nn.Module via eager_runtime()')
        return eager_runtime(model)
    raise ValueError(f'unsupported model format {kind}')


def eager_runtime(model):
    import torch
    model.eval()
    def run(batch):
        with torch.no_grad():
            return model(torch.as_tensor(batch))
    return run


def check_export_parity(reference: Detector, exports: Mapping[str, Detector],
                        images: Mapping[str, np.ndarray], box_atol: float = 1.,
                        score_atol: float = .02) -> dict:
    """Exact labels/count and predeclared numeric tolerance; no box matching heuristic."""
    if not images or not exports or box_atol < 0 or score_atol < 0:
        raise ValueError('need images, exports and non-negative tolerances')
    result = {}
    for name, adapter in exports.items():
        for image_id, image in images.items():
            base, got = reference.predict(image), adapter.predict(image)
            if not np.array_equal(base.labels, got.labels):
                raise AssertionError(f'{name}/{image_id}: detection count/order/class mismatch')
            if not np.allclose(base.boxes, got.boxes, atol=box_atol, rtol=0):
                raise AssertionError(f'{name}/{image_id}: box mismatch')
            if not np.allclose(base.scores, got.scores, atol=score_atol, rtol=0):
                raise AssertionError(f'{name}/{image_id}: score mismatch')
            result[f'{name}/{image_id}'] = len(got.labels)
    return result


def gate_asr(attacked: Detector, clean: Detector, pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
             eligible: Callable[[Detections], bool],
             success: Callable[[Detections, Detections], bool],
             *, floor: float, min_eligible: int) -> dict:
    """Held-out paired success minus same-trigger clean null, no assurance score input."""
    if not 0 <= floor <= 1 or min_eligible < 1:
        raise ValueError('invalid preregistered gate')
    if not pairs:
        raise ValueError('held-out pairs required')
    n = attack_hits = null_hits = 0
    for original, triggered in pairs.values():
        a0, c0 = attacked.predict(original), clean.predict(original)
        if not eligible(c0):
            continue
        n += 1
        attack_hits += bool(success(a0, attacked.predict(triggered)))
        null_hits += bool(success(c0, clean.predict(triggered)))
    raw, null = attack_hits / n if n else None, null_hits / n if n else None
    net = max(0., raw - null) if n else None
    status = 'unassessable' if n < min_eligible else ('implanted' if net >= floor else 'weak')
    return dict(status=status, n_eligible=n, attacked_successes=attack_hits,
                clean_null_successes=null_hits, asr_raw=raw, asr_null=null,
                asr_net=net, gate_floor=floor)
