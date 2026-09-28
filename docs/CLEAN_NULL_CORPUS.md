# Clean null corpus (`runs/clean_null`) — Task 1 evidence bundle

Reproduce: `.venv/bin/python -m cviaf.lab corpus --plan configs/corpus_clean_null.json`
then `.venv/bin/python scripts/clean_null_exchangeability.py`.

## What was built

50 independently trained clean models, seeds 100–149, recipe copied field-for-field from the
mvp2 clean arm (verified field-by-field against `runs/mvp2/clean_none_fixed_s5/manifest.json` and
`runs/day1/clean_none_fixed_s5/manifest.json`: every field matches except `model_id`,
`detector.seed` and `scene.seed`). **Seed-varied only.** Clean arm only: no attack, no trigger
recipe, so no stamp is present anywhere in training or measurement.

Measured: `trained=49 skipped=1 failed=0 in 358.8 s` (the one skip is the resumed smoke-test model).
Output: 50 model dirs, each `manifest.json` + `weights.npz`, ~28 KB/model.

**Seed-window assumption (stated, not hidden):** later guidance named seeds 30–79 for this file;
the config that exists and trained uses **100–149**. Path, recipe, clean-only arm and
`out=runs/clean_null` all match the instruction — only the seed window differs. 100–149 was chosen
because it is disjoint from every corpus already in `runs/` (mvp2 5–24, day1 5–15, mvp 5, models
5–9, recipecheck, trigger2), so no held-out image is shared with an existing population.

## Acceptance 1 — disjoint held-out splits

Digests recomputed from each spec (`build_splits(spec).eval_clean.digest()`), not read back from
the manifest's own recorded value:

| corpus | clean models | unique `eval_clean` digests | all unique |
|---|---:|---:|---|
| clean_null | 50 | 50 | yes |
| day1 | 11 | 11 | yes |
| mvp2 | 20 | 20 | yes |

Cross-corpus held-out overlap: clean_null∩day1 = **0**, clean_null∩mvp2 = **0**.
(day1∩mvp2 = 11 — pre-existing, see below.)

Every manifest carries `dataset_digests` with keys `train_clean`, `train_poisoned`, `eval_clean`,
`cal_clean` (per-model, so the exact held-out split is recorded). For a clean arm
`train_clean == train_poisoned` by construction — measured, not assumed.

## Acceptance 2 — exchangeability of the null spread

The null the asset rule prices is the **paired stamp response**: for one clean model, the mean
change in a detector's score between bare and trigger-stamped views of *the same* probe images.
Same reference model (`runs/mvp/clean_none_fixed_s5`), same 24 probe images, same trigger recipe
(`oga`, 10 px patch, fixed corner, seed 11) for every model in every corpus — only the model
differs. Probe stamp `max|delta| = 0.8021`, i.e. this control stimulus is itself a pixel artifact;
that is the point of the control.

| population | n | ctc mean ± sd | ctc max | refdiv mean ± sd | refdiv max |
|---|---:|---|---:|---|---:|
| clean_null (new) | 50 | +0.0828 ± 0.1129 | +0.5014 | +0.2935 ± 0.0516 | +0.3752 |
| day1 (existing) | 11 | +0.2367 ± 0.1695 | +0.5017 | +0.2902 ± 0.0625 | +0.4041 |
| mvp2 | 20 | +0.1925 ± 0.2178 | +0.8076 | +0.2991 ± 0.0603 | +0.4041 |

Two-sample KS, new 50 vs existing 11 → **refdiv: KS 0.145, p = 0.972 (exchangeable)**;
**ctc: KS 0.558, p = 0.0040 (not exchangeable)**, mean difference −0.1538, 95% CI [−0.259, −0.049].
vs mvp2 (20): refdiv KS 0.100, p = 0.997; ctc KS 0.370, p = 0.0316, mean diff −0.1096,
95% CI [−0.210, −0.009].

**The ctc gap is a real seed trend, not sampling noise.** Over all 81 clean models scored:
Spearman(seed, ctc response) ρ = **−0.290, p = 0.0087**; by seed bucket ctc mean = +0.2367
(seeds 5–15, n=22), +0.1385 (16–24, n=9), +0.0828 (100–149, n=50). refdiv shows no significant
trend (ρ = +0.205, p = 0.066) and matches across populations.

Consequences, stated plainly:
1. **refdiv is poolable** across the new corpus and the existing populations; **ctc is not** — its
   response mean depends on the model seed, so a ctc null calibrated on seeds 5–24 is not a valid
   null for suspects trained at seeds 100–149, and vice versa.
2. Within the new corpus the 50 models are exchangeable by construction (identical recipe, iid
   seeds), which is the exchangeability the asset rule actually requires: suspect and calibration
   models from the same population.
3. `runs/day1`'s 11 clean models are **byte-identical** to 11 of `runs/mvp2`'s 20 (11 of 11 shared
   model_ids give identical responses to 1e-12). The "existing 11" is therefore a subset of mvp2,
   not an independent second population; n=20 and n=11 are not two witnesses.

## Acceptance 3 — capacity (this is the unblock)

`model_asset_rule.decide_model_asset` (v4 package) needs `n_center = 5` models to fit the center
plus `ceil(m/alpha) − 1` rank models, i.e. **44 clean models at m = 2 signals, alpha = .05**:

| clean models | rank models | conformal p floor | can reject at α=.05 |
|---:|---:|---:|---|
| 11 | 6 | 0.1429 | **no — abstains regardless of evidence** |
| 20 | 15 | 0.0625 | no — abstains |
| 44 | 39 | 0.0250 | yes |
| 50 | 45 | **0.0217** | yes |

At n = 11 the asset p-value floor is 1/7 = 14.3%, above alpha: the rule abstains no matter how
anomalous the suspect is. At n = 50 the floor is 1/46 = 2.2%, so the rule can actually reject, at a
type-I rate of ≤ 2.2% for a suspect drawn from the same clean distribution. That is the capacity
the 50 models buy, and it is an exchangeability-calibrated bound, not a detection claim.

## Honest limits

- The stamp response is measured on **one** trigger recipe and one probe set; a different recipe
  (object-disappearance, frame) or size would need its own exchangeability check.
- refdiv exchangeability is measured against a single trusted reference model
  (`runs/mvp/clean_none_fixed_s5`); a reference drawn from a different seed range reintroduces the
  ctc-style seed dependence, and that dependence has not been measured on refdiv at that scale.
- 50 models at 2.2% type-I is not a guarantee for a *chosen-after-the-fact* suspect, reference or
  trigger (the v4 `decide_model_asset` docstring makes the same caveat).
- The ctc seed trend is measured on this synthetic generator only. It is not evidence about any
  real detector.
