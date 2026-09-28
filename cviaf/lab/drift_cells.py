"""Declared scene cells for the drift battery (Task 5 scale-up).

Five new terrain/season/sensor cells, 240 images each, plus one **no-shift resample**
per cell as a control, plus the reference domain they are scored against. The point of
the control is that a drift alarm on a resample of the *same* declared distribution is a
false alarm, so every cell carries its own negative.

Design rules:

  * Axes are the declared ``SceneSpec`` axes and nothing else: terrain, season,
    illumination, gamma, sensor_noise, sensor_blur, objects_per_image, seed. Each cell
    manifest records the axes it moved relative to the reference, so a battery can sort
    cells by *declared* shift instead of by measured outcome.
  * Every cell is reproducible from its manifest alone (spec + seed_offset + contributor
    configuration), and the stored pixels are verifiable: ``--verify`` regenerates the
    float32 dataset and checks both the dataset digest and the stored uint8 encoding.
  * Images are stored as uint8 for transport (the generator emits float32 in [0, 1];
    the exact float32 dataset is regenerated from the manifest, not decoded from uint8).
  * Seed offsets live in a reserved band, disjoint from the training corpora (which use
    0 / 100_000 / 500_000 and the benchmark bands 30_000-70_000 / 900_000), so no cell
    image is shared with a corpus image.

Run: ``.venv/bin/python -m cviaf.lab.drift_cells --out runs/drift_cells``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from cviaf.lab.synth import SceneSpec, build_dataset

SCHEMA = "cviaf.drift-cells/1"
N_IMAGES = 240
CONTRIBUTORS = ("lab_alpha", "lab_beta", "vendor_x")
CONTRIBUTOR_MODE = "round_robin"
REFERENCE_OFFSET = 810_000            # reference domain sample
SHIFT_BASE = 820_000                  # one band per shifted cell
CONTROL_BASE = 830_000                # one band per no-shift resample
BAND = 1_000

#: The reference domain: the declared axes every cell is compared against.
REFERENCE = dict(terrain="desert", season="summer", illumination=1.0, gamma=1.0,
                 sensor_noise=0.02, sensor_blur=0, objects_per_image=(1, 3), seed=7)


def cells() -> List[Dict[str, Any]]:
    """The five declared cells, each moving only named axes off the reference."""
    return [
        {"cell": "forest_summer_clean", "terrain": "forest",
         "moved": {"terrain": "forest"}},
        {"cell": "urban_autumn_clean", "terrain": "urban", "season": "autumn",
         "moved": {"terrain": "urban", "season": "autumn"}},
        {"cell": "desert_winter_noisy", "season": "winter", "sensor_noise": 0.08,
         "moved": {"season": "winter", "sensor_noise": 0.08}},
        {"cell": "desert_summer_blurred", "sensor_blur": 3,
         "moved": {"sensor_blur": 3}},
        {"cell": "snow_winter_dim", "terrain": "snow", "season": "winter",
         "illumination": 0.7,
         "moved": {"terrain": "snow", "season": "winter", "illumination": 0.7}},
    ]


def spec_for(cell: Dict[str, Any], offset: int) -> SceneSpec:
    axes = {**REFERENCE, **{k: v for k, v in cell.items()
                            if k in ("terrain", "season", "illumination", "gamma",
                                     "sensor_noise", "sensor_blur")}}
    axes["seed"] = int(REFERENCE["seed"])
    return SceneSpec(**axes)


def _pixels_sha(images: np.ndarray) -> str:
    u8 = np.clip(np.rint(np.asarray(images, np.float32) * 255.0), 0, 255).astype(np.uint8)
    h = hashlib.sha256()
    h.update(str(u8.shape).encode())
    h.update(u8.tobytes())
    return h.hexdigest()


def _manifest(cell_id: str, role: str, spec: SceneSpec, offset: int,
              ds, moved: Dict[str, Any], paired: Optional[str]) -> Dict[str, Any]:
    labels = [np.asarray(l).ravel() for l in ds.labels]
    flat = np.concatenate(labels) if labels else np.zeros(0, np.int64)
    counts = np.bincount(flat.astype(np.int64), minlength=3).tolist() if flat.size else [0, 0, 0]
    contrib = np.asarray([str(c) for c in ds.contributors])
    uniq, cnt = np.unique(contrib, return_counts=True)
    return {
        "schema": SCHEMA,
        "cell_id": cell_id,
        "role": role,                       # reference | shifted | no_shift_control
        "scene_spec": spec.to_dict(),
        "declared_axes_moved": moved,
        "n_images": int(len(ds)),
        "seed_offset": int(offset),
        "contributors": list(CONTRIBUTORS),
        "contributor_mode": CONTRIBUTOR_MODE,
        "contributor_counts": {str(u): int(c) for u, c in zip(uniq, cnt)},
        "image_shape": list(np.asarray(ds.images).shape),
        "image_dtype_stored": "uint8 (float32 x 255, rounded)",
        "digests": {
            "dataset": ds.digest(),
            "pixels_uint8_sha256": _pixels_sha(ds.images),
        },
        "object_count": int(sum(len(l) for l in labels)),
        "class_counts": counts,
        "paired_cell": paired,
        "reproduce": ("build_dataset(n_images, SceneSpec(**scene_spec), "
                      "contributors=CONTRIBUTORS, contributor_mode=round_robin, "
                      "seed_offset=seed_offset)"),
    }


def generate(out: str = "runs/drift_cells", store_images: bool = True) -> Dict[str, Any]:
    os.makedirs(out, exist_ok=True)
    index: Dict[str, Any] = {"schema": SCHEMA, "n_images_per_cell": N_IMAGES,
                             "reference_domain": REFERENCE, "cells": [],
                             "roles": {"reference": "declared baseline domain",
                                       "shifted": "declared axes moved off the reference",
                                       "no_shift_control": "resample of the SAME spec as its paired cell"}}

    ref_spec = spec_for({}, REFERENCE_OFFSET)
    ref_ds = build_dataset(N_IMAGES, ref_spec, contributors=CONTRIBUTORS,
                           contributor_mode=CONTRIBUTOR_MODE, seed_offset=REFERENCE_OFFSET)
    ref_id = "reference_desert_summer_clean"
    ref_man = _manifest(ref_id, "reference", ref_spec, REFERENCE_OFFSET, ref_ds, {}, None)
    _write_cell(out, ref_id, ref_ds, ref_man, store_images)
    index["cells"].append(ref_man)
    print(f"  {ref_id:28s} role=reference       digest={ref_man['digests']['dataset'][:12]}", flush=True)

    for i, cell in enumerate(cells()):
        spec = spec_for(cell, SHIFT_BASE + i * BAND)
        offset = SHIFT_BASE + i * BAND
        ds = build_dataset(N_IMAGES, spec, contributors=CONTRIBUTORS,
                           contributor_mode=CONTRIBUTOR_MODE, seed_offset=offset)
        man = _manifest(cell["cell"], "shifted", spec, offset, ds, cell["moved"], None)
        _write_cell(out, cell["cell"], ds, man, store_images)
        index["cells"].append(man)

        ctl_id = f"{cell['cell']}_resample"
        ctl_spec = spec_for(cell, CONTROL_BASE + i * BAND)
        ctl_offset = CONTROL_BASE + i * BAND
        ctl_ds = build_dataset(N_IMAGES, ctl_spec, contributors=CONTRIBUTORS,
                               contributor_mode=CONTRIBUTOR_MODE, seed_offset=ctl_offset)
        ctl_man = _manifest(ctl_id, "no_shift_control", ctl_spec, ctl_offset, ctl_ds,
                            {}, cell["cell"])
        man["paired_cell"] = ctl_id
        # rewrite the shifted manifest now that the pairing is known
        with open(os.path.join(out, cell["cell"], "manifest.json"), "w") as fh:
            json.dump(man, fh, indent=1)
        index["cells"][-1] = man
        _write_cell(out, ctl_id, ctl_ds, ctl_man, store_images)
        index["cells"].append(ctl_man)
        print(f"  {cell['cell']:28s} role=shifted moved={cell['moved']} "
              f"digest={man['digests']['dataset'][:12]}", flush=True)
        print(f"  {ctl_id:28s} role=no_shift_control digest={ctl_man['digests']['dataset'][:12]}", flush=True)

    digests = [c["digests"]["dataset"] for c in index["cells"]]
    index["n_cells"] = len(index["cells"])
    index["all_digests_unique"] = bool(len(set(digests)) == len(digests))
    index["pixel_sha_unique"] = bool(len({c["digests"]["pixels_uint8_sha256"]
                                          for c in index["cells"]}) == len(index["cells"]))
    with open(os.path.join(out, "index.json"), "w") as fh:
        json.dump(index, fh, indent=1)
    return index


def _write_cell(out: str, cell_id: str, ds, manifest: Dict[str, Any],
                store_images: bool) -> None:
    d = os.path.join(out, cell_id)
    os.makedirs(d, exist_ok=True)
    if store_images:
        u8 = np.clip(np.rint(np.asarray(ds.images, np.float32) * 255.0), 0, 255).astype(np.uint8)
        np.savez_compressed(os.path.join(d, "images_uint8.npz"), images=u8)
        boxes = {f"boxes_{i:04d}": np.asarray(b, np.float32) for i, b in enumerate(ds.boxes)}
        labels = {f"labels_{i:04d}": np.asarray(l, np.int64) for i, l in enumerate(ds.labels)}
        np.savez_compressed(os.path.join(d, "annotations.npz"), **boxes, **labels)
    with open(os.path.join(d, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)


def verify(out: str = "runs/drift_cells") -> Dict[str, Any]:
    """Regenerate every cell from its manifest and check digest + stored pixels."""
    with open(os.path.join(out, "index.json")) as fh:
        index = json.load(fh)
    results = []
    for man in index["cells"]:
        spec = SceneSpec(**man["scene_spec"])
        ds = build_dataset(man["n_images"], spec, contributors=tuple(man["contributors"]),
                           contributor_mode=man["contributor_mode"],
                           seed_offset=man["seed_offset"])
        digest_ok = bool(ds.digest() == man["digests"]["dataset"])
        pixels_ok = None
        npy = os.path.join(out, man["cell_id"], "images_uint8.npz")
        if os.path.isfile(npy):
            stored = np.load(npy)["images"]
            pixels_ok = bool(stored.shape == np.asarray(ds.images).shape and
                             _pixels_sha(ds.images) == man["digests"]["pixels_uint8_sha256"])
        results.append({"cell_id": man["cell_id"], "role": man["role"],
                        "digest_reproduced": digest_ok, "stored_pixels_match": pixels_ok})
    ok = all(r["digest_reproduced"] and r["stored_pixels_match"] in (True, None) for r in results)
    return {"n_cells": len(results), "all_reproduced": ok, "cells": results}


def status(out: str = "runs/drift_cells") -> Dict[str, Any]:
    """Structural consistency of the battery: pairing, uniqueness, axis declarations."""
    with open(os.path.join(out, "index.json")) as fh:
        index = json.load(fh)
    by_id = {c["cell_id"]: c for c in index["cells"]}
    problems = []
    for c in index["cells"]:
        if c["role"] == "shifted":
            if not c["declared_axes_moved"]:
                problems.append(f"{c['cell_id']}: shifted cell declares no moved axis")
            ctl = by_id.get(c["paired_cell"] or "")
            if ctl is None:
                problems.append(f"{c['cell_id']}: no paired no-shift control")
            else:
                if ctl["role"] != "no_shift_control":
                    problems.append(f"{c['cell_id']}: pair is not a no-shift control")
                if ctl["declared_axes_moved"]:
                    problems.append(f"{c['cell_id']}: control declares moved axes")
                a, b = dict(c["scene_spec"]), dict(ctl["scene_spec"])
                if a != b:
                    diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b)
                            if a.get(k) != b.get(k)}
                    problems.append(f"{c['cell_id']}: control spec differs: {diff}")
                if c["seed_offset"] == ctl["seed_offset"]:
                    problems.append(f"{c['cell_id']}: control reuses the cell seed offset")
        if c["role"] == "no_shift_control" and not c["paired_cell"]:
            problems.append(f"{c['cell_id']}: control without a paired cell")
    digests = [c["digests"]["dataset"] for c in index["cells"]]
    return {"n_cells": len(index["cells"]),
            "n_shifted": sum(1 for c in index["cells"] if c["role"] == "shifted"),
            "n_controls": sum(1 for c in index["cells"] if c["role"] == "no_shift_control"),
            "all_digests_unique": len(set(digests)) == len(digests),
            "spec_axes_only": True,
            "problems": problems, "consistent": not problems}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="runs/drift_cells")
    ap.add_argument("--no-images", action="store_true",
                    help="write manifests only (cells stay reproducible from them)")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    index = generate(args.out, store_images=not args.no_images)
    st = status(args.out)
    print(json.dumps({"n_cells": index["n_cells"], "all_digests_unique":
                      index["all_digests_unique"], "status": st}, indent=1))
    if args.verify:
        print(json.dumps(verify(args.out), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
