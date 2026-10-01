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
- [x] Step 2.1 declared: victim class 0, target class 2, trigger = natural size cue
      (`max(box side) <= 11.0 px`, measured median object side), criterion = centre-cell
      class assignment. Proximity condition also declared and measured (negative).
- [x] Step 2.2 implemented `cviaf/lab/stampfree.py` + `tests/test_stampfree.py` (6 tests)
- [x] Step 2.3 paired net ASR measured: **acceptance NOT met** — k=1 clears the 0.50 floor on
      1/3 seeds (+0.533/+0.111/+0.056; Wilson CI [0.008, 0.906]), k=3 on 0/3 (CP upper 0.708).
      1 of 6 measured cells clears. Pixel identity: `max|delta| = 0.0` everywhere and the
      control reproduces the clean_null artifact bit-for-bit on all 3 seeds.
- [x] Step 2.4 null suite run against the arm in a throwaway `c17141f` clone with the three
      pending patches applied (all three applied cleanly; my tree untouched):
      `runs/stampfree/null_suite.json`. `stamp_null` AUROC = **0.5000 for all six signal
      columns**, `fft` 0.5000 everywhere, behavioural prediction-flip rate **0.00** both arms
      ⇒ no stamp exists in this cell. Conditional AUROC (`backdoor_conditional_unstamped`, =
      stamped by construction): refdiv **0.6730**, with_ftc 0.5741, without_ftc 0.5532, ftc
      0.5431, ctc 0.4436, fft 0.5000; peer-clean floor refdiv 0.336 / ctc 0.504 / ftc 0.004.
      Asset decisions 0/3 rejected in all 5 cells, both fusions (3 assets, weak attack).
- [x] Step 2.5 commit + push

FINDING for Task 4 (integration, must be unioned): my tree's `AttackSpec` has `mechanism`,
~ the v4 patch's has `scope`; neither has both, so each side's manifests fail to reconstruct on
the other side (`AttackSpec.__init__() got an unexpected keyword argument 'mechanism'`). The
merged dataclass must carry the union or every committed corpus breaks on one of the two trees.

## TASK 3 — real-backbone validation (torch/onnx)

- [ ] Step 3.1 env check: which interpreter has torch/onnx/onnxruntime (no installs without asking)
- [ ] Step 3.2 **export parity gate first**: benign re-export → ZERO substitution findings
- [ ] Step 3.3 only then: backdoored model + eval + null suite; report whether the stamp-confound
      finding transfers off synthetic 64x64

## TASK 4 — merge test (PREP ONLY until the v4 package lands)

- [x] Step 4.1 `docs/LAB_EDIT_INVENTORY.md`: every uncommitted lab edit as `file:hunk` (evaluate 41,
      train 20, corpus 20, poison 15, cli 7, detectors 3, tamper_probe 3, README 1). Those edits
      remain UNCOMMITTED, as instructed.
- [x] Step 4.2 EXECUTED. Clean `c17141f` clone → consolidated patch + label-gate follow-up applied
      clean → suite → my pushed work merged → two mandated changes → one merge commit
      **`63c6ca9`** (parents `c17141f`, `8c6bc3b`) pushed to `origin/main`.
- [x] Step 4.3 verification addendum: `docs/V4_MERGE_VERIFICATION.md` + refreshed
      `runs/stampfree/null_suite_merged.json` + the n=50 asset verdicts.

MEASURED (Phase 3, see `docs/V4_MERGE_VERIFICATION.md` for the full record):
- Patched base suite **153 passed / 1 xfailed** on a fresh `c17141f` clone with both patches
  (handover said 152: +1 is the follow-up patch's 4 tests on top of the package's own 149).
  Without `cryptography` the same tree reports 21 failed / 123 passed / 2 skipped / 1 xfailed.
- Merged line suite: **164 passed / 1 xfailed / 0 skipped**.
- Conflicts: **zero textual**. The predicted semantic overlap (poison `AttackSpec`) resolved as the
  union (`mechanism` + `scope`, `mechanism` last); verified live because the n=50 script rebuilds a
  probe spec from a pre-merge manifest.
- Mandated: FTC out (`DETECTOR_NAMES = ("ctc","refdiv")`); gate on refdiv
  (`DEFAULT_SIGNALS=("refdiv",)`, ctc diagnostic-only) → requirement 44 → 24 models, p-floor
  1/46 = 2.17%.
- n=50 verdicts: clean leave-one-out **5/50 = 10%** (CP [0.033, 0.192]); day1 attacked **13/72**
  (Fisher vs clean p = .301, not separable); stamp-free arms **2/3** rejected at the p-floor
  (weakest arm, net ASR 0.056, not rejected; P(X>=2 | n=3, p=.10) = .028).
- `cviaf/lab/null_suite.py` asset fusion abstains **0/3** on the same stamp-free arms while
  `model_asset_rule` rejects **2/3** — flagged as an open inconsistency, not resolved solo.
- Not done (not in the spec, needs the owning lane): the calibrated-protocol patch
  (`~/Downloads/CVIAF_V4_CALIBRATED_ASSURE.txt`) — dry-run says it **applies clean** on `63c6ca9`.

## TASK 5 — corpus scale-up for the drift battery

5 new terrain/season/sensor cells x 240 images + manifests, reusing the declared SceneSpec axes and
manifest format so driftbench consumes the cells directly; **one no-shift resample per cell** as the
control. Commit + push.
- [x] Step 5.1 `cviaf/lab/drift_cells.py`: 5 declared cells + reference + 1 resample control each
- [x] Step 5.2 generated `runs/drift_cells` (11 cells, 2640 images, 26 MB) with manifests
- [x] Step 5.3 controls: spec-identical to their cell, different seed offset, asserted by
      `status()` (problems == []) and pinned by `tests/test_drift_cells.py`
- [x] Step 5.4 commit + push (`caa18ce`); `--verify` reproduced 11/11 dataset digests and
      11/11 stored-pixel hashes; 55 tests pass

## TASK 3 — env check result (decision needed before any install)

Measured: `.venv` (Python 3.11.16) has **no** torch / onnx / onnxruntime.
`/opt/homebrew/bin/python3.12` (3.12.14) has torch **2.14.0 (MPS available)**, onnx **1.23.0**,
onnxruntime **1.30.0**, numpy and PIL — but **no scipy and no scikit-learn**, which the repo's
lab/eval path imports. So Task 3 needs either extra packages in the torch interpreter or torch
in the repo venv; both are installs, and the STEER says ask first. No install has been run.

BOTH INSTALLS APPROVED AND DONE (this blocker is cleared):
- `.venv` (3.11.16) gained **cryptography 50.0.1** — the whole repo suite now runs green.
- `~/.venvs/cviaf-torch` (3.12.14, built with `--system-site-packages`) reuses brew's torch:
  **torch 2.14.0 (MPS available)**, **onnx 1.23.0**, **onnxruntime 1.30.0**, numpy, scipy,
  scikit-learn, Pillow, **torchvision** — everything the Task 3 parity gate needs. It does NOT
  have `cryptography`; the parity gate calls `ModelIntegrityAssessor` directly and never builds
  the provenance engine, so that does not block Task 3 — but a full product pipeline run from
  that interpreter needs `cryptography` first, because the v4 provenance code refuses the
  symmetric fallback.
Sequence when unblocked: export parity gate **first** (benign re-export → zero substitution
findings), then the backdoored model + eval + null suite.

---
---
## Phase log
- P1 plan: this file.
- P2 execute: Task 1 training done (358.8s).
- P4 report: Tasks 1, 2, 5 pushed (`3c9d005`, `4aef427`, `76c14d2`, `caa18ce`); Task 4 prep pushed
  (`747493c`).
- P4 report: Task 4 merge test executed → **`63c6ca9`** on `origin/main` (merge commit, two mandated
  signal changes). Post-merge verification addendum added on top of it.
- Note on local state: this checkout's `main` is still `8c6bc3b` because the uncommitted lab edits
  overlap files the merge touches; `origin/main` is the merged line.
