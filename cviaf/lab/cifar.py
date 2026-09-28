"""CIFAR-10 subset for the real-backbone lane. Download once, cache, no re-download.

Task 3. The lab's detector eats ``DetectionDataset`` objects (images, per-image boxes,
per-image labels, contributor and batch metadata), so CIFAR has to be expressed in that
shape rather than as a classifier batch.

DECISIONS (frozen; rationale in TASK3_NOTES.md)

  * classes: CIFAR-10 classes 0, 1, 2 (airplane, automobile, bird) -> detector labels
    0, 1, 2, i.e. ``DetectorConfig.n_classes = 3`` is satisfied without remapping.
  * one object per image, occupying a centred square box of ``box_cells`` grid cells
    (default 2 cells = 8 px at ``img_size`` 64). CIFAR has no boxes; a fixed centred box
    is the honest minimal annotation, and it is the same box every image, so the box head
    learns a constant and detection quality is driven by objectness/class, not by box luck.
  * images: CIFAR is 32x32; they are upsampled to ``img_size`` (64) by nearest-neighbour
    index repeat - deterministic, no resampling library, and it keeps the [0, 1] float32
    contract the synthetic dataset uses.
  * caching: the archive is fetched once into ``<cache_dir>`` and the parsed training
    arrays are memoised to ``<cache_dir>/cifar10_train.npz`` so repeat runs cost ~0.3 s.
    The default download is the fast.ai image-classification mirror
    (``s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz``, one PNG per class directory)
    because the canonical Krizhevsky host serves this machine at ~24 KB/s versus ~2 MB/s
    there; the canonical pickle tarball is still supported and used as the fallback.
    Ingesting the PNG layout decodes only the declared classes and is one-time.
  * the cache lives under ``data/`` which is git-ignored in this repository, so nothing
    large is ever committed.
"""
from __future__ import annotations

import os
import pickle
import tarfile
import urllib.request
from typing import Dict, List, Sequence, Tuple

import numpy as np

from cviaf.lab.synth import DetectionDataset

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_DIRNAME = "cifar-10-batches-py"
FASTAI_URL = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
FASTAI_DIRNAME = "cifar10"
DEFAULT_CACHE = "data/cifar10"
CIFAR_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
                 "horse", "ship", "truck")
DEFAULT_CLASSES: Tuple[int, ...] = (0, 1, 2)
TRAIN_BATCHES = ("data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5")


def ensure_cifar10(cache_dir: str = DEFAULT_CACHE, verbose: bool = True) -> str:
    """Fetch and extract CIFAR-10 once. Returns the directory holding the batches."""
    root = os.path.join(cache_dir, CIFAR_DIRNAME)
    marker = os.path.join(root, "data_batch_1")
    if os.path.isfile(marker):
        return root
    os.makedirs(cache_dir, exist_ok=True)
    tar_path = os.path.join(cache_dir, "cifar-10-python.tar.gz")
    if not os.path.isfile(tar_path):
        if verbose:
            print(f"  downloading {CIFAR_URL} -> {tar_path} (once)")
        urllib.request.urlretrieve(CIFAR_URL, tar_path)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(cache_dir)
    if not os.path.isfile(marker):                       # pragma: no cover - corrupt tar
        raise RuntimeError(f"CIFAR extraction did not produce {marker}")
    return root


def ingest_fastai_png(cache_dir: str = DEFAULT_CACHE,
                      classes: Sequence[int] = DEFAULT_CLASSES,
                      verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Decode the fast.ai PNG layout into the array cache (declared classes only)."""
    from PIL import Image

    extract = os.path.join(cache_dir, "_fastai")
    root = os.path.join(extract, FASTAI_DIRNAME, "train")
    if not os.path.isdir(root):
        tgz = os.path.join(cache_dir, "cifar10-fastai.tgz")
        if not os.path.isfile(tgz):
            if verbose:
                print(f"  downloading {FASTAI_URL} -> {tgz} (once)")
            os.makedirs(cache_dir, exist_ok=True)
            urllib.request.urlretrieve(FASTAI_URL, tgz)
        os.makedirs(extract, exist_ok=True)
        with tarfile.open(tgz, "r:gz") as tf:
            tf.extractall(extract)
    names = sorted(os.listdir(root))
    if names != list(CIFAR_CLASSES):                              # class order is the label
        raise RuntimeError(f"unexpected class directories {names}; CIFAR label order assumed")
    images: List[np.ndarray] = []
    labels: List[int] = []
    for ci, cname in enumerate(names):
        if ci not in classes:
            continue
        cdir = os.path.join(root, cname)
        for fn in sorted(os.listdir(cdir)):
            with Image.open(os.path.join(cdir, fn)) as im:
                images.append(np.asarray(im.convert("RGB"), np.uint8))
            labels.append(ci)
    imgs = np.stack(images)
    labs = np.asarray(labels, np.int64)
    if verbose:
        print(f"  ingested {imgs.shape[0]} PNGs for classes {list(classes)}")
    return imgs, labs


def load_train_arrays(cache_dir: str = DEFAULT_CACHE, verbose: bool = True,
                      classes: Sequence[int] = DEFAULT_CLASSES
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """(N, 32, 32, 3) uint8 images and (N,) int64 labels, memoised to npz."""
    npz = os.path.join(cache_dir, "cifar10_train.npz")
    if os.path.isfile(npz):
        with np.load(npz) as z:
            if set(np.asarray(z["classes"]).tolist()) >= set(int(c) for c in classes):
                return z["images"], z["labels"]
            if verbose:
                print("  cache does not cover the requested classes; re-ingesting")
    fastai_layout = os.path.join(cache_dir, "_fastai", FASTAI_DIRNAME, "train")
    if os.path.isdir(fastai_layout) or os.path.isfile(os.path.join(cache_dir, "cifar10-fastai.tgz")):
        imgs, labs = ingest_fastai_png(cache_dir, classes, verbose=verbose)
    else:
        root = ensure_cifar10(cache_dir, verbose=verbose)
        images: List[np.ndarray] = []
        labels: List[int] = []
        for name in TRAIN_BATCHES:
            with open(os.path.join(root, name), "rb") as fh:
                blob = pickle.load(fh, encoding="bytes")
            raw = np.asarray(blob[b"data"], np.uint8).reshape(-1, 3, 32, 32)
            images.append(np.transpose(raw, (0, 2, 3, 1)))          # -> (N, 32, 32, 3)
            labels.extend(int(x) for x in blob[b"labels"])
        imgs = np.concatenate(images, axis=0)
        labs = np.asarray(labels, np.int64)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(npz, images=imgs, labels=labs,
                        classes=np.asarray([int(c) for c in classes], np.int64))
    if verbose:
        print(f"  cached {imgs.shape[0]} CIFAR-10 training images -> {npz}")
    return imgs, labs


def upsample_nearest(images: np.ndarray, size: int) -> np.ndarray:
    """(N, h, w, 3) -> (N, size, size, 3) by nearest-neighbour index repeat."""
    if images.shape[1] == size and images.shape[2] == size:
        return images
    idx = (np.arange(size) * (images.shape[1] / size)).astype(np.int64)
    idy = (np.arange(size) * (images.shape[2] / size)).astype(np.int64)
    return images[:, idx][:, :, idy]


def centred_box(img_size: int, cell: float, box_cells: float = 2.0) -> np.ndarray:
    """The declared annotation: a square box of ``box_cells`` grid cells, centred.

    The centre is snapped to the centre CELL (``(grid // 2 + 0.5) * cell``) and not to the
    image centre, because ``TinyDetector.decode_boxes`` can only place a box centre at a
    cell centre. At img_size 64 that is 34 px, not 32 px: the half-cell mismatch caps a
    perfectly predicted box at IoU ~0.39, under the match threshold, so detection quality
    would read 0.000 at any training budget. Measured and fixed here.
    """
    grid = int(round(img_size / cell))
    c = (grid // 2 + 0.5) * cell
    half = box_cells * cell / 2.0
    return np.asarray([c - half, c - half, c + half, c + half], np.float32)


def load_cifar_subset(n_per_class: int = 60,
                      classes: Sequence[int] = DEFAULT_CLASSES,
                      seed: int = 0,
                      split: str = "train",
                      cache_dir: str = DEFAULT_CACHE,
                      img_size: int = 64,
                      box_cells: float = 2.0,
                      contributor: str = "cifar10",
                      batch: str = "cifar_host",
                      verbose: bool = False) -> DetectionDataset:
    """A deterministic CIFAR-10 subset in the lab's ``DetectionDataset`` shape."""
    if split != "train":
        raise ValueError("only the training split is wired for Task 3 (the test split has "
                         "no labels file); extend REMAINING item (a) if you need it")
    classes = tuple(int(c) for c in classes)
    images, labels = load_train_arrays(cache_dir, verbose=verbose, classes=classes)
    rng = np.random.default_rng(seed)
    take: List[np.ndarray] = []
    for new_label, cls in enumerate(classes):
        pool = np.flatnonzero(labels == cls)
        if pool.size < n_per_class:
            raise ValueError(f"class {cls} has only {pool.size} images, asked {n_per_class}")
        pick = rng.permutation(pool)[:n_per_class]
        take.append(pick)
    idx = np.concatenate(take)
    order = rng.permutation(idx.size)                    # deterministic interleave
    idx = idx[order]

    cell = float(img_size) / float(img_size // 4)
    box = centred_box(img_size, cell, box_cells)
    imgs = upsample_nearest(images[idx], img_size).astype(np.float32) / 255.0
    new_labels = np.asarray([classes.index(int(labels[i])) for i in idx], np.int64)
    spec: Dict[str, object] = {
        "source": "cifar-10", "split": split, "classes": list(classes),
        "class_names": [CIFAR_CLASSES[c] for c in classes],
        "n_per_class": int(n_per_class), "seed": int(seed), "img_size": int(img_size),
        "upsample": "nearest_index_repeat", "box_cells": float(box_cells),
        "boxes": "one centred square box per image", "cache_dir": cache_dir,
    }
    return DetectionDataset(
        images=np.ascontiguousarray(imgs, np.float32),
        boxes=[box.reshape(1, 4).copy() for _ in range(idx.size)],   # (Ni, 4) convention
        labels=[np.asarray([int(l)], np.int64) for l in new_labels],
        contributors=np.asarray([contributor] * idx.size, object),
        batches=np.asarray([batch] * idx.size, object),
        spec=spec,
    )
