# Drift cells (`runs/drift_cells`) — Task 5 evidence bundle

Reproduce: `.venv/bin/python -m cviaf.lab.drift_cells --out runs/drift_cells --verify`

Five new terrain/season/sensor cells, 240 images each, plus **one no-shift resample per
cell as its control**, plus the reference domain they are scored against. 11 cells,
2,640 images, `runs/drift_cells/index.json` as the battery's entry point; each cell has
`manifest.json` + `images_uint8.npz` + `annotations.npz`.

## The cells (declared axes, nothing else)

| cell | role | terrain | season | noise | blur | illum | objects |
|---|---|---|---:|---:|---:|---:|---:|
| `reference_desert_summer_clean` | reference | desert | summer | 0.02 | 0 | 1.00 | 467 |
| `forest_summer_clean` | shifted | forest | summer | 0.02 | 0 | 1.00 | 483 |
| `urban_autumn_clean` | shifted | urban | autumn | 0.02 | 0 | 1.00 | 474 |
| `desert_winter_noisy` | shifted | desert | winter | **0.08** | 0 | 1.00 | 467 |
| `desert_summer_blurred` | shifted | desert | summer | 0.02 | **3** | 1.00 | 470 |
| `snow_winter_dim` | shifted | snow | winter | 0.02 | 0 | **0.70** | 466 |
| `<cell>_resample` (×5) | no_shift_control | = its cell | | | | | 460–521 |

Each shifted cell records `declared_axes_moved`, so a battery can sort cells by *declared*
shift rather than by a measured outcome — a three-axis cell (`snow_winter_dim`) and a
one-axis cell (`desert_summer_blurred`) are not the same experiment.

## What the control is, and is not

The resample uses the **identical `SceneSpec`** as its paired cell (asserted — `status()`
fails if any axis differs, if a control declares moved axes, or if the control reuses its
cell's seed offset) and differs only in the seed offset, i.e. it is another draw from the
same declared distribution. A drift alarm on it is a false alarm, so every cell ships its
own negative. It does **not** test the detector: it tests the battery, and a battery that
cannot stay quiet on a spec-identical resample has no business reporting a cell as shifted.

Seed-offset bands are reserved and disjoint (reference 810_000; shifted 820_000 + 1_000·i;
controls 830_000 + 1_000·i), so no cell image coincides with a training-corpus image
(corpora use 0 / 100_000 / 500_000 and the 30_000–70_000 / 900_000 benchmark bands).

## Reproducibility, verified rather than asserted

`--verify` regenerates every cell from its manifest alone and checks two things per cell:
the float32 dataset digest matches the recorded digest, and the stored uint8 pixels match
the regenerated pixels' hash. Measured on 2026-09-28: **11/11 cells reproduced, 11/11
stored-pixel checks passed**. All 11 dataset digests and all 11 pixel hashes are unique.

Images are stored as uint8 for transport; the generator emits float32 in [0, 1] and the
*exact* float32 dataset is regenerated from `scene_spec` + `seed_offset`, not decoded from
uint8. The manifest's `reproduce` field carries the exact call.

## Limits

* 240 images per cell (the task's size) with `objects_per_image=(1, 3)`; a cell's per-cell
  false-alarm *rate* therefore has a granularity of 1/240, and tails below that are not
  estimable from one cell.
* The resample control bounds *sampling* noise only. It cannot detect a battery that fires
  on both the cell and its own resample for a spec-independent reason.
* Only the declared `SceneSpec` axes vary. Illumination is the sole intensity axis changed
  (`snow_winter_dim` at 0.70); gamma is untouched in every cell, so nothing here measures
  tone-curve drift.
* The cells are synthetic 64×64 datasets from the same generator as every other corpus in
  this lab; they inherit its terrain palette and object placement, and they are not a
  stand-in for operational imagery.
