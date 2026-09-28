# STATE.md — live execution plan (Phase 1 written once; updated as steps complete)

Base: `c17141f` = `HEAD` = `origin/main` (as of this file). Every artifact below is committed
file-by-file (never `git add -A`) and pushed as it lands. Node: numbers carry denominator, seed
set, control type and CI; 0/N ships its 3/N (or exact binomial) upper bound.

Do-not-touch (v4 patch owns them; local edits create rebase collisions):
`cviaf/drift/`, `cviaf/provenance/`, `evaluate.py::FUSED_DETECTORS`. Uncommitted lane edits stay
uncommitted until the v4 merge test (Task 4).

---

## TASK 1 — clean null corpus (>=44 exchangeable clean models)

Plan: `configs/corpus_clean_null.json` — clean arm only, recipe copied field-for-field from the
mvp2 clean spec; only the seed varies (detector seed = model seed, scene seed = 7 + model seed).
Output `runs/clean_null` (committable; `.gitignore` does not cover `runs/`).

- [x] Step 1.1 config written
- [x] Step 1.2 train 50: `python -m cviaf.lab corpus --plan configs/corpus_clean_null.json`
      → measured `trained=49 skipped=1 failed=0 in 358.8s`
- [x] Step 1.3 verify (all acceptance checks green; see `docs/CLEAN_NULL_CORPUS.md`)
- [x] Step 1.4 commit + push
- [x] Step 1.5 full suite green (`45 passed in 4.99s`)

MEASURED (Phase 3):
- A1 count: 50/50 model dirs with `manifest.json` + `weights.npz` (>= 44). 
- A2 disjointness: 50/50 unique recomputed `eval_clean` digests; cross-corpus held-out overlap
  0 with both day1 and mvp2. (day1∩mvp2 = 11, pre-existing: day1's 11 clean models are
  byte-identical to 11 of mvp2's 20 — the "existing 11" is a subset, not a second population.)
- A3 split record: `dataset_digests` present with `train_clean/train_poisoned/eval_clean/cal_clean`.
- A4 exchangeability (paired stamp response, common reference + probe set):
  refdiv refdiv new50 +0.2935±0.0516 vs day1 +0.2902±0.0625 (KS p=.972, exchangeable) and vs mvp2
  +0.2991±0.0603 (KS p=.997). ctc new50 +0.0828±0.1129 vs day1 +0.2367±0.1695 (KS p=.0040, NOT
  exchangeable) and vs mvp2 +0.1925±0.2178 (KS p=.0316). ctc gap is a real seed trend:
  Spearman(seed, ctc) rho=-0.290 p=.0087 across all 81 clean models.
- A5 capacity: rule needs 44 clean models at m=2 / alpha=.05. p-floor 1/7=.1429 at n=11 (abstains),
  1/16=.0625 at n=20 (abstains), **1/46=.0217 at n=50 (can reject)**.

PUSHED
- A1 count: 50 model dirs, each `manifest.json` + `weights.npz`; `>= 44`.
- A2 disjointness: 50 unique `eval_clean` dataset digests (recomputed from the spec, not read from
  the manifest), and zero overlap with mvp2/day1/mvp/models/recipecheck/trigger2 eval splits.
- A3 split record: every manifest's `dataset_digests` names train/eval/cal digests.
- A4 exchangeability: null-score spread of the new 50 vs the existing 11 clean models (day1) and
  the 20 mvp2 clean models — report all three spreads, two-sample KS p, and the clean-max that
  sets a conformal threshold at n=11 vs n=50.
- A5 `.venv/bin/python -m pytest tests/ -q` green.

Assumption stated in the push: the later general guidance named
`configs/corpus_clean_null.json` with seeds 30–79; the config that exists (and that trained) uses
seeds 100–149. Path, recipe, clean-only arm and `out=runs/clean_null` all match; only the seed
window differs. Rationale recorded in the config `_comment`: seeds 100–149 are disjoint from every
existing corpus in `runs/`, so no held-out image is shared with the existing clean population.

## TASK 2 — stamp-free backdoor (new module; `poison.py` untouched)

New module `cviaf/lab/stampfree.py`. Two candidate mechanisms, both with **zero pixel change**:
1. **weight-space trojan** — a train-time weight edit conditioned on a *natural* semantic
   condition already present in eval images (declared before training, below);
2. **clean-label / label-only** poisoning — behavior change with pixel-identical inputs.
Declared before training (required):
- victim class: **to be declared in the module docstring and the run manifest before any fit**;
- trigger semantics: **a natural, already-present condition** (no stamped patch); the pixel-artifact
  check is *measured*, not asserted: `max |triggered - source|` over the eval set must be exactly 0.
- [ ] Step 2.1 declare victim class + trigger semantics in module docstring
- [ ] Step 2.2 implement + unit tests (pixel-identity invariant)
- [ ] Step 2.3 paired net ASR >= .50 on >= 2 seeds (report the seed set and the CI)
- [ ] Step 2.4 null suite against the stamp-free model → conditional AUROC (the real deliverable)
- [ ] Step 2.5 commit + push

## TASK 3 — real-backbone validation (torch/onnx)

- [ ] Step 3.1 env check: which interpreter has torch/onnx/onnxruntime (no installs without asking)
- [ ] Step 3.2 **export parity gate first**: benign re-export → ZERO substitution findings
- [ ] Step 3.3 only then: backdoored model + eval + null suite; report whether the stamp-confound
      finding transfers off synthetic 64x64

## TASK 4 — merge test (PREP ONLY until the v4 package lands)

- [ ] Step 4.1 write `docs/LAB_EDIT_INVENTORY.md`: every uncommitted lab edit as `file:hunk`, with
      semantic-overlap flags. Do NOT commit those edits.
- [ ] Step 4.2 (blocked on the user's consolidated v4 patch) clean `c17141f` clone → `git apply`
      → full suite → merge my lab edits → report every conflict as `file:hunk`, no solo resolution.

## TASK 5 — corpus scale-up for the drift battery

5 new terrain/season/sensor cells x 240 images + manifests, reusing the declared SceneSpec axes and
manifest format so driftbench consumes the cells directly; **one no-shift resample per cell** as the
control. Commit + push.
- [ ] Step 5.1 cell configs
- [ ] Step 5.2 generate + manifests
- [ ] Step 5.3 control resamples
- [ ] Step 5.4 commit + push

---
## Phase log
- P1 plan: this file.
- P2 execute: Task 1 training done (358.8s). Verification next.
