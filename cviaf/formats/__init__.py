"""
Dataset format handlers for COCO and YOLO formats.
Provides unified access to image paths, labels, annotations, and metadata.
"""

from __future__ import annotations

import json
import os
import glob as globmod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from cviaf.core.types import SampleMetadata


@dataclass
class BBox:
    """Bounding box in normalized [x_center, y_center, width, height] format."""
    x_center: float
    y_center: float
    width: float
    height: float
    class_id: int
    confidence: float = 1.0

    def to_xyxy(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """Convert to absolute (x1, y1, x2, y2) pixel coordinates."""
        x1 = int((self.x_center - self.width / 2) * img_w)
        y1 = int((self.y_center - self.height / 2) * img_h)
        x2 = int((self.x_center + self.width / 2) * img_w)
        y2 = int((self.y_center + self.height / 2) * img_h)
        return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

    def to_coco(self, img_w: int, img_h: int) -> List[float]:
        """Convert to COCO format [x, y, width, height] in pixels."""
        x = (self.x_center - self.width / 2) * img_w
        y = (self.y_center - self.height / 2) * img_h
        return [x, y, self.width * img_w, self.height * img_h]


@dataclass
class DatasetSample:
    """A single sample from any supported dataset format."""
    image_path: str
    image_id: str = ""
    labels: List[int] = field(default_factory=list)
    bboxes: List[BBox] = field(default_factory=list)
    metadata: SampleMetadata = field(default_factory=SampleMetadata)
    segmentation: Optional[Any] = None
    category_names: List[str] = field(default_factory=list)


class COCODatasetLoader:
    """
    Loads COCO-format datasets.
    
    Expects:
      - annotations JSON file (instances_*.json)
      - images directory
    """

    def __init__(self, annotation_file: str, images_dir: str,
                 contributor: str = "unknown"):
        self.annotation_file = annotation_file
        self.images_dir = images_dir
        self.contributor = contributor
        self._data: Dict[str, Any] = {}
        self._image_map: Dict[int, Dict] = {}
        self._category_map: Dict[int, str] = {}
        self._loaded = False

    def load(self) -> None:
        """Parse the COCO annotations JSON."""
        with open(self.annotation_file, "r") as f:
            self._data = json.load(f)

        # Build image ID -> info map
        for img in self._data.get("images", []):
            self._image_map[img["id"]] = img

        # Build category ID -> name map
        for cat in self._data.get("categories", []):
            self._category_map[cat["id"]] = cat["name"]

        self._loaded = True

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()

    @property
    def categories(self) -> Dict[int, str]:
        self._ensure_loaded()
        return self._category_map

    @property
    def num_images(self) -> int:
        self._ensure_loaded()
        return len(self._image_map)

    def iter_samples(self) -> Iterator[DatasetSample]:
        """Iterate over all annotated images as DatasetSample objects."""
        self._ensure_loaded()

        # Group annotations by image_id
        annots_by_image: Dict[int, list] = {}
        for ann in self._data.get("annotations", []):
            img_id = ann["image_id"]
            annots_by_image.setdefault(img_id, []).append(ann)

        for img_id, img_info in self._image_map.items():
            file_name = img_info.get("file_name", "")
            img_path = os.path.join(self.images_dir, file_name)

            img_w = img_info.get("width", 1)
            img_h = img_info.get("height", 1)

            bboxes = []
            labels = []
            category_names = []

            for ann in annots_by_image.get(img_id, []):
                cat_id = ann.get("category_id", 0)
                labels.append(cat_id)
                category_names.append(self._category_map.get(cat_id, "unknown"))

                # COCO bbox is [x, y, width, height] in pixels
                if "bbox" in ann and len(ann["bbox"]) == 4:
                    bx, by, bw, bh = ann["bbox"]
                    bboxes.append(BBox(
                        x_center=(bx + bw / 2) / img_w,
                        y_center=(by + bh / 2) / img_h,
                        width=bw / img_w,
                        height=bh / img_h,
                        class_id=cat_id,
                    ))

            meta = SampleMetadata(
                sample_id=str(img_id),
                file_path=img_path,
                contributor=self.contributor,
                label=",".join(category_names) if category_names else "",
                annotations={"coco_image_info": img_info},
            )

            yield DatasetSample(
                image_path=img_path,
                image_id=str(img_id),
                labels=labels,
                bboxes=bboxes,
                metadata=meta,
                category_names=category_names,
            )

    def get_all_samples(self) -> List[DatasetSample]:
        return list(self.iter_samples())


class YOLODatasetLoader:
    """
    Loads YOLO-format datasets.
    
    Expects:
      - images directory with image files
      - labels directory with .txt files (one per image)
        Each line: class_id x_center y_center width height
      - Optional data.yaml with class names
    """

    def __init__(self, images_dir: str, labels_dir: str,
                 class_names: Optional[List[str]] = None,
                 data_yaml: Optional[str] = None,
                 contributor: str = "unknown"):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.contributor = contributor
        self.class_names = class_names or []

        if data_yaml and os.path.exists(data_yaml) and not self.class_names:
            self._parse_data_yaml(data_yaml)

    def _parse_data_yaml(self, yaml_path: str):
        """Parse YOLO data.yaml for class names."""
        try:
            import yaml
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            names = data.get("names", [])
            if isinstance(names, dict):
                # {0: 'person', 1: 'car', ...}
                max_id = max(names.keys()) if names else -1
                self.class_names = [""] * (max_id + 1)
                for k, v in names.items():
                    self.class_names[k] = v
            elif isinstance(names, list):
                self.class_names = names
        except ImportError:
            # Fall back to basic parsing if PyYAML not installed
            with open(yaml_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("- "):
                        self.class_names.append(line[2:].strip("'\""))

    def _find_images(self) -> List[str]:
        """Find all image files in the images directory."""
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]
        images = []
        for ext in exts:
            images.extend(globmod.glob(os.path.join(self.images_dir, ext)))
            images.extend(globmod.glob(os.path.join(self.images_dir, ext.upper())))
        return sorted(set(images))

    def _parse_label_file(self, label_path: str) -> List[BBox]:
        """Parse a YOLO label .txt file."""
        bboxes = []
        if not os.path.exists(label_path):
            return bboxes

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    conf = float(parts[5]) if len(parts) > 5 else 1.0
                    bboxes.append(BBox(
                        x_center=xc, y_center=yc, width=w, height=h,
                        class_id=cls_id, confidence=conf
                    ))
        return bboxes

    def iter_samples(self) -> Iterator[DatasetSample]:
        """Iterate over all YOLO samples."""
        images = self._find_images()

        for img_path in images:
            stem = Path(img_path).stem
            label_path = os.path.join(self.labels_dir, f"{stem}.txt")

            bboxes = self._parse_label_file(label_path)
            labels = [b.class_id for b in bboxes]
            cat_names = []
            for b in bboxes:
                if b.class_id < len(self.class_names):
                    cat_names.append(self.class_names[b.class_id])
                else:
                    cat_names.append(f"class_{b.class_id}")

            meta = SampleMetadata(
                sample_id=stem,
                file_path=img_path,
                contributor=self.contributor,
                label=",".join(cat_names),
            )

            yield DatasetSample(
                image_path=img_path,
                image_id=stem,
                labels=labels,
                bboxes=bboxes,
                metadata=meta,
                category_names=cat_names,
            )

    def get_all_samples(self) -> List[DatasetSample]:
        return list(self.iter_samples())


def _infer_coco_images_dir(annotation_file: str) -> str:
    """
    Infer the directory that contains a COCO dataset's images.

    Standard COCO layouts put the annotation JSON in an ``annotations/``
    folder with the image folders as *siblings* (``val2017/``, ``train2017/``
    next to ``annotations/``), but some datasets embed an ``images/`` folder
    next to the JSON. The previous code only looked inside the annotation
    parent, so sibling layouts fell back to the annotations dir itself and
    every image path pointed at a non-existent file.

    Strategy (most specific match wins):
      1. Collect candidate dirs from inside the annotation parent and from
         the parent's parent (sibling folders), using standard COCO image
         folder names plus any directory that actually holds image files.
      2. When the annotation JSON is readable, count how many referenced
         ``file_name`` values truly exist under each candidate.
      3. Return the candidate with the most real hits; fall back to a
         candidate that merely contains image files; last resort is the
         annotation parent (previous behavior).
    """
    parent = os.path.dirname(os.path.abspath(annotation_file))
    grandparent = os.path.dirname(parent)
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

    # Candidate directories: known names inside parent and as siblings.
    candidates: List[str] = []
    known_names = ["images", "image", "train2017", "val2017", "test2017",
                   "train", "val", "test"]
    for base in (parent, grandparent):
        for name in known_names:
            p = os.path.join(base, name)
            if os.path.isdir(p) and p not in candidates:
                candidates.append(p)

    # Loose match: any subdirectory (of parent or grandparent) that directly
    # contains image files.
    for root in (parent, grandparent):
        try:
            entries = sorted(os.listdir(root))
        except OSError:
            continue
        for entry in entries:
            p = os.path.join(root, entry)
            if os.path.isdir(p) and p not in candidates and p != parent:
                try:
                    has_images = any(
                        f.lower().endswith(image_exts)
                        for f in os.listdir(p)[:200]
                    )
                except OSError:
                    has_images = False
                if has_images:
                    candidates.append(p)

    # Referenced file names from the annotation JSON, if readable.
    referenced: List[str] = []
    try:
        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        referenced = [
            img.get("file_name", "")
            for img in data.get("images", [])
            if img.get("file_name")
        ]
    except (OSError, ValueError, json.JSONDecodeError):
        pass

    # Best match: the candidate that contains the most referenced files.
    if candidates and referenced:
        best_dir, best_hits = candidates[0], -1
        for cand in candidates:
            hits = sum(
                1 for fn in referenced
                if os.path.exists(os.path.join(cand, fn))
            )
            if hits > best_hits:
                best_hits, best_dir = hits, cand
        if best_hits > 0:
            return best_dir

    # No JSON hits: prefer a candidate that really holds image files.
    for cand in candidates:
        try:
            if any(f.lower().endswith(image_exts) for f in os.listdir(cand)[:200]):
                return cand
        except OSError:
            continue

    return parent


def load_dataset(path: str, format: str = "auto", **kwargs) -> List[DatasetSample]:
    """
    Unified dataset loader. 
    
    Args:
        path: Path to dataset root or annotation file
        format: "coco", "yolo", or "auto"
        **kwargs: Passed to the appropriate loader
    
    Returns:
        List of DatasetSample objects
    """
    if format == "auto":
        # Auto-detect format
        if os.path.isfile(path) and path.endswith(".json"):
            format = "coco"
        elif os.path.isdir(path):
            # Check for YOLO structure (images/ and labels/ subdirs)
            if os.path.isdir(os.path.join(path, "labels")):
                format = "yolo"
            elif os.path.isdir(os.path.join(path, "images")):
                # Could be either, check for annotation JSON
                json_files = globmod.glob(os.path.join(path, "*.json"))
                json_files += globmod.glob(os.path.join(path, "annotations", "*.json"))
                if json_files:
                    format = "coco"
                else:
                    format = "yolo"
            else:
                format = "yolo"  # Default to YOLO for flat image dirs
        else:
            raise ValueError(f"Cannot auto-detect format for: {path}")

    if format == "coco":
        ann_file = kwargs.get("annotation_file", path)
        imgs_dir = kwargs.get("images_dir", "")
        if not imgs_dir:
            # Infer images dir: check inside the annotation parent AND sibling
            # dirs (standard COCO layout has annotations/ and val2017/ side by
            # side), validating candidates against the JSON file names.
            imgs_dir = _infer_coco_images_dir(ann_file)

        loader = COCODatasetLoader(
            annotation_file=ann_file,
            images_dir=imgs_dir,
            contributor=kwargs.get("contributor", "unknown"),
        )
        return loader.get_all_samples()

    elif format == "yolo":
        if os.path.isdir(path):
            imgs_dir = kwargs.get("images_dir", os.path.join(path, "images"))
            lbls_dir = kwargs.get("labels_dir", os.path.join(path, "labels"))
            if not os.path.isdir(imgs_dir):
                imgs_dir = path  # Flat directory
            if not os.path.isdir(lbls_dir):
                lbls_dir = path
        else:
            raise ValueError(f"YOLO format requires a directory path, got: {path}")

        loader = YOLODatasetLoader(
            images_dir=imgs_dir,
            labels_dir=lbls_dir,
            class_names=kwargs.get("class_names"),
            data_yaml=kwargs.get("data_yaml"),
            contributor=kwargs.get("contributor", "unknown"),
        )
        return loader.get_all_samples()

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'coco', 'yolo', or 'auto'.")
