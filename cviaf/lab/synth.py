"""
Procedural synthetic object-detection dataset with *declarable* shift axes.

Why a synthetic generator is the right MVP choice
-------------------------------------------------
To validate an assurance framework you do not need a good detector. You need a
detector whose GROUND TRUTH YOU CONTROL. That is what this module provides: every
image, box, label, contributor, and distribution shift is generated from a seed we
hold, so every claim the assurance engine makes can be scored against a known
answer. Real COCO/VOC data buys realism of the *model*, which is what the H200 is
for (see docs/SCALING_PLAN.md). It is not needed to prove the method.

Declarable shift axes
---------------------
``terrain``, ``season``, ``illumination``, ``gamma``, ``sensor_noise``, ``sensor_blur``
are explicit fields of :class:`SceneSpec`. The PS asks for detection of "changes
caused by terrain, season, sensor, illumination or acquisition conditions" and for a
*declared* reference distribution. Here the reference distribution is literally a
SceneSpec, so the drift module has something honest to compare against.

Honest limitation
-----------------
Class identity is carried mostly by colour, aspect ratio and shape motif. A
detector can therefore learn classes from colour shortcuts. That is acceptable for
an assurance MVP -- we are testing whether *backdoors* are detectable, not whether
the detector is a good vehicle detector -- but it must be stated, and it is one of
the reasons the scaling plan swaps in real imagery.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

IMG_SIZE = 64
CLASS_NAMES: Tuple[str, ...] = ("vehicle", "person", "structure")
NUM_CLASSES = len(CLASS_NAMES)

# RGB base colours per class.
CLASS_COLORS: Dict[str, Tuple[float, float, float]] = {
    "vehicle": (0.88, 0.16, 0.12),
    "person": (0.14, 0.55, 0.92),
    "structure": (0.78, 0.74, 0.28),
}
# (min_side, max_side) in pixels at IMG_SIZE=64.
CLASS_SIZE: Dict[str, Tuple[int, int]] = {
    "vehicle": (9, 17),
    "person": (5, 8),
    "structure": (11, 22),
}
CLASS_SHAPE: Dict[str, str] = {
    "vehicle": "rect",
    "person": "ellipse",
    "structure": "triangle",
}

# terrain -> (bright tone, dark tone)
TERRAINS: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    "desert": ((0.84, 0.72, 0.46), (0.64, 0.53, 0.32)),
    "forest": ((0.30, 0.46, 0.24), (0.16, 0.30, 0.15)),
    "urban": ((0.56, 0.56, 0.60), (0.36, 0.36, 0.41)),
    "snow": ((0.91, 0.94, 0.98), (0.76, 0.83, 0.92)),
    "night": ((0.11, 0.13, 0.21), (0.04, 0.05, 0.11)),
}

# season -> RGB multiplier (a pure palette shift, which is exactly what a naive
# windowed drift detector can be fooled by -- see arXiv 2411.16591).
SEASON_TINT: Dict[str, Tuple[float, float, float]] = {
    "summer": (1.00, 1.00, 0.97),
    "winter": (0.94, 0.97, 1.06),
    "monsoon": (0.90, 0.95, 0.92),
    "autumn": (1.07, 0.95, 0.86),
}


@dataclass
class SceneSpec:
    """A fully declarable image distribution. This is the unit of drift declaration."""

    terrain: str = "desert"
    season: str = "summer"
    illumination: float = 1.0     # multiplicative brightness
    gamma: float = 1.0            # tone curve exponent
    sensor_noise: float = 0.02    # gaussian sigma
    sensor_blur: int = 0          # box-blur radius in pixels
    objects_per_image: Tuple[int, int] = (1, 3)
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return _sha(json.dumps(self.to_dict(), sort_keys=True, default=str))


@dataclass
class Scene:
    """One generated image with its ground-truth annotations."""

    image: np.ndarray             # (H, W, 3) float32 in [0, 1]
    boxes: np.ndarray             # (N, 4) float32, xyxy in pixels
    labels: np.ndarray            # (N,) int64
    spec: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _sha(data: Any) -> str:
    if not isinstance(data, (bytes, bytearray)):
        data = str(data).encode()
    return hashlib.sha256(data).hexdigest()


def _soft_texture(rng: np.random.Generator, size: int, coarse: int = 4) -> np.ndarray:
    """Low-frequency noise field in [0, 1]: a coarse grid bilinearly upscaled."""
    grid = rng.random((coarse, coarse)).astype(np.float32)
    idx = np.linspace(0, coarse - 1, size, dtype=np.float32)
    lo = np.floor(idx).astype(np.int32)
    hi = np.minimum(lo + 1, coarse - 1)
    w = (idx - lo).astype(np.float32)
    a = grid[lo][:, lo]
    b = grid[lo][:, hi]
    c = grid[hi][:, lo]
    d = grid[hi][:, hi]
    wy = w[:, None]
    wx = w[None, :]
    top = a * (1 - wx) + b * wx
    bot = c * (1 - wx) + d * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def _box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Cheap separable box blur via cumulative sums. radius in pixels."""
    if radius <= 0:
        return img
    out = img.astype(np.float32)
    k = 2 * radius + 1
    for axis in (0, 1):
        csum = np.cumsum(out, axis=axis)
        pad = np.zeros_like(np.take(csum, [0], axis=axis))
        csum = np.concatenate([pad, csum], axis=axis)
        n = out.shape[axis]
        hi = np.clip(np.arange(n) + radius + 1, 0, n)
        lo = np.clip(np.arange(n) - radius, 0, n)
        # index along `axis`
        hi_idx = np.take(csum, hi, axis=axis)
        lo_idx = np.take(csum, lo, axis=axis)
        out = (hi_idx - lo_idx) / float(k)
    return out.astype(np.float32)


def _draw_shape(
    img: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    color: Sequence[float],
    shape: str,
) -> None:
    """Paint a filled shape in place. Uses a local meshgrid so cost is independent of IMG_SIZE."""
    H, W, _ = img.shape
    x0 = int(max(0, np.floor(cx - w / 2)))
    x1 = int(min(W, np.ceil(cx + w / 2) + 1))
    y0 = int(max(0, np.floor(cy - h / 2)))
    y1 = int(min(H, np.ceil(cy + h / 2) + 1))
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dx = (xs - cx) / max(w / 2.0, 1e-6)
    dy = (ys - cy) / max(h / 2.0, 1e-6)
    if shape == "rect":
        mask = (np.abs(dx) <= 1.0) & (np.abs(dy) <= 1.0)
    elif shape == "ellipse":
        mask = (dx ** 2 + dy ** 2) <= 1.0
    else:  # triangle: wider at the base, apex at the top
        mask = (dy >= -1.0) & (dy <= 1.0) & (np.abs(dx) <= (dy + 1.0) / 2.0)
    region = img[y0:y1, x0:x1, :]
    col = np.asarray(color, dtype=np.float32)[None, None, :]
    region[mask] = 0.72 * region[mask] + 0.28 * np.broadcast_to(col, region.shape)[mask]


def _overlaps(a: np.ndarray, boxes: List[np.ndarray], margin: float = 2.0) -> bool:
    for b in boxes:
        if (
            a[0] - margin < b[2]
            and b[0] - margin < a[2]
            and a[1] - margin < b[3]
            and b[1] - margin < a[3]
        ):
            return True
    return False


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #

def generate_scene(
    spec: SceneSpec,
    n_objects: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    objects: bool = True,
) -> Scene:
    """Generate one image. ``objects=False`` yields a pure background (used by TRACE)."""
    rng = rng if rng is not None else np.random.default_rng(spec.seed)
    size = IMG_SIZE
    terrain = TERRAINS.get(spec.terrain, TERRAINS["desert"])
    tint = np.asarray(SEASON_TINT.get(spec.season, (1.0, 1.0, 1.0)), dtype=np.float32)

    # -- background: blend two terrain tones with a low-frequency field
    field = _soft_texture(rng, size, coarse=4)[:, :, None]
    bright = np.asarray(terrain[0], dtype=np.float32)[None, None, :]
    dark = np.asarray(terrain[1], dtype=np.float32)[None, None, :]
    img = (dark * (1.0 - field) + bright * field).astype(np.float32)

    # -- illumination / tone / season
    img = np.clip(img * float(spec.illumination) * tint[None, None, :], 0.0, 1.0)
    if abs(spec.gamma - 1.0) > 1e-6:
        img = np.power(np.clip(img, 1e-6, 1.0), float(spec.gamma)).astype(np.float32)

    # -- sensor
    if spec.sensor_blur > 0:
        img = _box_blur(img, spec.sensor_blur)
    if spec.sensor_noise > 0:
        img = img + rng.normal(0.0, float(spec.sensor_noise), img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)

    if not objects:
        return Scene(image=img, boxes=np.zeros((0, 4), np.float32),
                     labels=np.zeros((0,), np.int64), spec=spec.to_dict())

    # -- objects
    lo, hi = spec.objects_per_image
    n = int(n_objects) if n_objects is not None else int(rng.integers(lo, hi + 1))
    boxes: List[np.ndarray] = []
    labels: List[int] = []
    for _ in range(n):
        for _attempt in range(12):
            ci = int(rng.integers(0, NUM_CLASSES))
            cname = CLASS_NAMES[ci]
            smin, smax = CLASS_SIZE[cname]
            bw = float(rng.integers(smin, smax + 1))
            bh = float(rng.integers(smin, smax + 1))
            if cname == "person":
                bh = max(bh, 9.0)
            # keep shapes fully inside the frame
            if bw + 4 >= size or bh + 4 >= size:
                continue
            cx = float(rng.uniform(bw / 2 + 2, size - bw / 2 - 2))
            cy = float(rng.uniform(bh / 2 + 2, size - bh / 2 - 2))
            box = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], np.float32)
            if _overlaps(box, boxes):
                continue
            _draw_shape(img, cx, cy, bw, bh, CLASS_COLORS[cname], CLASS_SHAPE[cname])
            boxes.append(box)
            labels.append(ci)
            break

    bb = np.array(boxes, np.float32) if boxes else np.zeros((0, 4), np.float32)
    ll = np.array(labels, np.int64) if labels else np.zeros((0,), np.int64)
    return Scene(image=np.clip(img, 0.0, 1.0), boxes=bb, labels=ll, spec=spec.to_dict())


def draw_object(
    image: np.ndarray, cx: float, cy: float, w: float, h: float, class_index: int
) -> np.ndarray:
    """Return a copy of ``image`` with one object drawn at (cx, cy).

    Used by the FTC / "Island Effect" detector: instead of pasting a rectangular
    image patch (which introduces a hard edge the model reacts to for the wrong
    reason), we *paint the object into* the scene. The decoy then differs from a
    real object only in that the model was never trained on that background, which
    is exactly the comparison TRACE's second signal needs.
    """
    out = image.copy()
    cname = CLASS_NAMES[int(class_index) % NUM_CLASSES]
    _draw_shape(out, cx, cy, w, h, CLASS_COLORS[cname], CLASS_SHAPE[cname])
    return out


@dataclass
class DetectionDataset:
    """A contributed dataset: images, annotations, per-sample contributor metadata."""

    images: np.ndarray                  # (N, H, W, 3) float32
    boxes: List[np.ndarray]             # length N, each (Ni, 4) xyxy
    labels: List[np.ndarray]            # length N, each (Ni,)
    contributors: np.ndarray            # (N,) object/str
    batches: np.ndarray                 # (N,) str
    spec: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def digest(self) -> str:
        """Content hash over pixels + annotations. The dataset's identity."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.images, dtype=np.float32).tobytes())
        for b, l in zip(self.boxes, self.labels):
            h.update(np.ascontiguousarray(b, dtype=np.float32).tobytes())
            h.update(np.ascontiguousarray(l, dtype=np.int64).tobytes())
        return h.hexdigest()

    def contributor_summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for c in self.contributors:
            out[str(c)] = out.get(str(c), 0) + 1
        return out


def build_dataset(
    n: int,
    spec: SceneSpec,
    contributors: Sequence[str] = ("lab_alpha", "lab_beta", "vendor_x"),
    contributor_mode: str = "round_robin",
    seed_offset: int = 0,
) -> DetectionDataset:
    """Generate ``n`` scenes with contributor attribution.

    ``contributor_mode`` is ``round_robin`` (balanced) or ``grouped`` (contiguous
    blocks, which is how a real vendor contribution usually arrives and which makes
    source attribution meaningful).
    """
    rng = np.random.default_rng(spec.seed + seed_offset)
    imgs = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), np.float32)
    boxes: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    contrib = np.empty(n, dtype=object)
    batches = np.empty(n, dtype=object)

    n_contrib = max(1, len(contributors))
    for i in range(n):
        s = SceneSpec(**{**spec.to_dict(), "seed": int(spec.seed + seed_offset + i * 7919)})
        sc = generate_scene(s, rng=rng)
        imgs[i] = sc.image
        boxes.append(sc.boxes)
        labels.append(sc.labels)
        if contributor_mode == "grouped":
            ci = min(n_contrib - 1, int(i * n_contrib / max(n, 1)))
        else:
            ci = i % n_contrib
        contrib[i] = contributors[ci]
        batches[i] = f"batch_{i // 64:03d}"

    return DetectionDataset(
        images=imgs, boxes=boxes, labels=labels, contributors=contrib, batches=batches,
        spec={**spec.to_dict(), "n": n, "contributors": list(contributors),
              "contributor_mode": contributor_mode, "seed_offset": seed_offset},
    )


# --------------------------------------------------------------------------- #
# format export (the PS names COCO and YOLO explicitly)
# --------------------------------------------------------------------------- #

def to_coco(ds: DetectionDataset, path: str, class_names: Sequence[str] = CLASS_NAMES) -> str:
    """Write a COCO-format instances JSON. Images are written as PNGs alongside."""
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for to_coco()") from exc

    base = os.path.splitext(path)[0]
    imgs_dir = base + "_images"
    os.makedirs(imgs_dir, exist_ok=True)

    coco: Dict[str, Any] = {
        "info": {"description": "CVIAF lab synthetic dataset", "version": "1.0",
                 "dataset_digest": ds.digest()},
        "licenses": [],
        "categories": [{"id": i + 1, "name": n, "supercategory": "object"}
                       for i, n in enumerate(class_names)],
        "images": [], "annotations": [],
    }
    ann_id = 1
    for i in range(len(ds)):
        fn = f"{i:06d}.png"
        Image.fromarray((ds.images[i] * 255.0).astype(np.uint8)).save(os.path.join(imgs_dir, fn))
        coco["images"].append({
            "id": i + 1, "file_name": fn,
            "width": int(ds.images.shape[2]), "height": int(ds.images.shape[1]),
            "contributor": str(ds.contributors[i]), "batch_id": str(ds.batches[i]),
        })
        for b, l in zip(ds.boxes[i], ds.labels[i]):
            x0, y0, x1, y1 = (float(v) for v in b)
            coco["annotations"].append({
                "id": ann_id, "image_id": i + 1, "category_id": int(l) + 1,
                "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                "iscrowd": 0,
            })
            ann_id += 1
    with open(path, "w") as fh:
        json.dump(coco, fh, indent=1)
    return path


def to_yolo(ds: DetectionDataset, outdir: str, class_names: Sequence[str] = CLASS_NAMES) -> str:
    """Write YOLO layout: images/ + labels/ + data.yaml (normalised cx cy w h)."""
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for to_yolo()") from exc

    img_dir = os.path.join(outdir, "images")
    lbl_dir = os.path.join(outdir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    H, W = int(ds.images.shape[1]), int(ds.images.shape[2])
    for i in range(len(ds)):
        stem = f"{i:06d}"
        Image.fromarray((ds.images[i] * 255.0).astype(np.uint8)).save(
            os.path.join(img_dir, stem + ".png"))
        lines = []
        for b, l in zip(ds.boxes[i], ds.labels[i]):
            x0, y0, x1, y1 = (float(v) for v in b)
            cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
            bw, bh = (x1 - x0) / W, (y1 - y0) / H
            lines.append(f"{int(l)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
    yml = os.path.join(outdir, "data.yaml")
    with open(yml, "w") as fh:
        fh.write(f"path: {os.path.abspath(outdir)}\ntrain: images\nval: images\n"
                 f"nc: {len(class_names)}\nnames: {list(class_names)}\n")
    return yml
