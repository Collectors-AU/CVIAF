#!/usr/bin/env python
"""Format ingest check: prove COCO, YOLO, ONNX, PyTorch and TorchScript all load.

PS 2.2.6 requires ingesting COCO and YOLO datasets and supporting ONNX and
PyTorch/TorchScript reference models. The tests exercise parts of that in-process, but
nothing on disk said "it worked, and here are the counts" — so the coverage statement
could only ever report "the code is there", which is exactly the state that hides a
broken ingest path until a submission day.

This writes the artefact: synthetic COCO annotations and a YOLO label tree, a real ONNX
export taken from the corpus on disk, and a TorchScript twin, each loaded through the
repository's own loaders and run for one forward pass. Offline, no training, seconds.

Run:
    cd .task3 && PYTHONPATH=. <venv>/bin/python scripts/format_ingest_check.py \\
        --out runs/format_ingest.json

Exit: 0 every format loaded, 1 one or more failed (the artefact records which).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCHEMA = "cviaf.format-ingest.v1"
CLASSES = ["person", "car"]


def _write_png(path: str, size: int = 32, seed: int = 0) -> None:
    try:
        from PIL import Image
    except ImportError:                                    # pragma: no cover
        raise SystemExit("Pillow is required to write the synthetic images")
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(path)


def check_coco(root: str) -> Dict[str, Any]:
    from cviaf.formats import load_dataset
    images_dir = os.path.join(root, "coco", "images")
    os.makedirs(images_dir, exist_ok=True)
    names = ["a.png", "b.png"]
    for i, name in enumerate(names):
        _write_png(os.path.join(images_dir, name), seed=i)
    ann = {
        "images": [{"id": i + 1, "file_name": n, "width": 32, "height": 32}
                   for i, n in enumerate(names)],
        "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [4, 4, 12, 12],
             "area": 144, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [16, 16, 8, 8],
             "area": 64, "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [2, 20, 10, 10],
             "area": 100, "iscrowd": 0},
        ],
    }
    ann_path = os.path.join(root, "coco", "annotations.json")
    with open(ann_path, "w", encoding="utf-8") as fh:
        json.dump(ann, fh)
    samples = load_dataset(ann_path, format="coco")
    boxes = sum(len(getattr(s, "bboxes", []) or []) for s in samples)
    return {"ok": len(samples) == 2 and boxes == 3, "n_samples": len(samples),
            "n_boxes": boxes, "classes": CLASSES, "path": ann_path}


def check_yolo(root: str) -> Dict[str, Any]:
    from cviaf.formats import load_dataset
    images = os.path.join(root, "yolo", "images")
    labels = os.path.join(root, "yolo", "labels")
    os.makedirs(images, exist_ok=True)
    os.makedirs(labels, exist_ok=True)
    for i, name in enumerate(["a.png", "b.png"]):
        _write_png(os.path.join(images, name), seed=10 + i)
    with open(os.path.join(labels, "a.txt"), "w", encoding="utf-8") as fh:
        fh.write("0 0.5 0.5 0.25 0.25\n1 0.2 0.2 0.1 0.1\n")
    with open(os.path.join(labels, "b.txt"), "w", encoding="utf-8") as fh:
        fh.write("0 0.4 0.6 0.3 0.3\n")
    samples = load_dataset(os.path.join(root, "yolo"), format="yolo")
    boxes = sum(len(getattr(s, "bboxes", []) or []) for s in samples)
    return {"ok": len(samples) == 2 and boxes == 3, "n_samples": len(samples),
            "n_boxes": boxes, "classes": CLASSES,
            "path": os.path.join(root, "yolo")}


def _forward(wrapper: Any, shape: tuple) -> List[int]:
    out = wrapper.predict(np.zeros(shape, np.float32))
    return list(np.asarray(out).shape)


def check_onnx(pattern: str = "runs/**/*.onnx") -> Dict[str, Any]:
    from cviaf.formats.model_loader import load_model
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        return {"ok": False, "reason": f"no .onnx artefact matched {pattern}"}
    path = matches[0]
    wrapper = load_model(path)
    shape = tuple(getattr(wrapper, "_input_shape", ()) or ())
    if len(shape) != 4 or any(d <= 0 for d in shape):
        shape = (1, 3, 64, 64)
    out_shape = _forward(wrapper, shape)
    return {"ok": bool(getattr(wrapper, "_loaded", True)) and len(out_shape) > 0,
            "path": path,
            "input_shape": list(shape), "output_shape": out_shape,
            "model_digest": (getattr(wrapper, "model_digest", None) or "")[:16] or None,
            "n_candidates": len(matches)}


def _tiny_torch_model():
    import torch.nn as nn

    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))


def check_pytorch(root: str) -> Dict[str, Any]:
    import torch
    from cviaf.formats.model_loader import load_model
    path = os.path.join(root, "tiny.pt")
    model = _tiny_torch_model()
    model.eval()
    torch.save(model, path)
    wrapper = load_model(path)
    try:
        out_shape = _forward(wrapper, (1, 3, 32, 32))
    except Exception as exc:                # a wrapper-level failure is the finding
        return {"ok": False, "path": path, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": len(out_shape) == 2, "path": path, "output_shape": out_shape,
            "save_format": "torch.save(whole module)"}


def check_torchscript(root: str) -> Dict[str, Any]:
    import torch
    from cviaf.formats.model_loader import load_model
    path = os.path.join(root, "tiny_torchscript.pt")
    model = _tiny_torch_model()
    model.eval()
    traced = torch.jit.trace(model, torch.zeros(1, 3, 32, 32))
    torch.jit.save(traced, path)
    wrapper = load_model(path)
    try:
        out_shape = _forward(wrapper, (1, 3, 32, 32))
    except Exception as exc:
        return {"ok": False, "path": path, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": len(out_shape) == 2, "path": path, "output_shape": out_shape,
            "save_format": "torch.jit.trace + torch.jit.save"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/format_ingest.json")
    ap.add_argument("--onnx-glob", default="runs/**/*.onnx")
    args = ap.parse_args()

    started = time.time()
    formats: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="cviaf_format_") as root:
        for name, fn in (("coco", lambda: check_coco(root)),
                         ("yolo", lambda: check_yolo(root)),
                         ("pytorch", lambda: check_pytorch(root)),
                         ("torchscript", lambda: check_torchscript(root))):
            try:
                formats[name] = fn()
            except Exception as exc:
                formats[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        formats["onnx"] = check_onnx(args.onnx_glob)
    except Exception as exc:
        formats["onnx"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    report = {"schema": SCHEMA, "formats": formats,
              "all_ok": all(f.get("ok") for f in formats.values()),
              "runtime_seconds": round(time.time() - started, 1),
              "note": ("COCO/YOLO are loaded through cviaf.formats.load_dataset; ONNX "
                       "through onnxruntime; PyTorch/TorchScript through the PyTorch "
                       "wrapper. No network access and no training: the images are "
                       "synthetic and the models are tiny.")}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    for name, row in formats.items():
        print(f"  {name:12s} {'ok ' if row.get('ok') else 'FAIL'} "
              f"{ {k: v for k, v in row.items() if k != 'path'} }")
    print(f"wrote {args.out}: all_ok={report['all_ok']} in {report['runtime_seconds']}s")
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
