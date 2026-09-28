# ARM F: grouped BY fusion and post-screen drift effect gate

Base: `Collectors-AU/CVIAF` at `c17141f`; patch is local, not applied to upstream. Run `PYTHONPATH=. .venv/bin/python scripts/measure_arm_f.py` in the repo after installing project dependencies. `tests/test_arm_f.py` adds unit, null-dependence, and held-out calibration checks.

## Method

`cviaf/lab/fusion.py` combines valid conformal p-values via BY global-null p-values, first within prespecified detector families, then across families; the resulting item p-values are BY-screened across items. This is deliberately more conservative than raw BH and does not claim groups somehow make BH valid under arbitrary dependence. Existing two-signal data verdict and item-rate comparison call this routine; their default maps each available signal to a separate family. For correlated channels, callers supply the actual family partition. No detector group may be chosen after seeing the test outcomes.

`DistributionShiftAssessor` retains its legacy screen (MMD p < .05 or mean Mahalanobis > 15) and requires a median per-active-feature Wasserstein distance of at least the policy floor, in reference standard deviation units. The shipped `.2` floor is provisional, not certified. `calibrate_effect_floor` derives a floor from independent same-size clean batches; the benchmark uses 100 such batches, with a separate 100-batch evaluation set. The reported p-value is still from the existing permutation test. This fixed-batch gate is not anytime-valid or a hysteresis state machine.

## Reproducible paired counts

Input: `runs/mvp/clean_none_fixed_s5/manifest.json` (same clean spec as day1 seed 5), with manifest's n_cal=120 and n_train=240. Deterministic generation offsets and seeded permutations live in the script. Each clean evaluation batch has one paired before/after decision; a batch alarms if the shift assessor produces a finding. The script also tests a forest-vs-desert shifted batch. Numbers from the run on this checkout:

| Assessment | Before | After | Paired clean batches | True shift retained |
|---|---:|---:|---:|---|
| Drift legacy p/Mahalanobis screen vs screen + calibrated effect floor | 43/100 (43%) | 6/100 (6%) | 37 alerts removed | Forest batch: yes before and after; effect 3.352, floor 0.241 |
| Six correlated image-feature channels, raw Bonferroni-over-channels + BY-over-items vs grouped BY + BY-over-items | 0/100 (0%) | 0/100 (0%) | No change | Not tested on attack batches |

Group fusion did not improve the clean-batch false alarm rate here: both choices had no alarms. The drift improvement is close to the practitioner's 35.9% to 4.1% pattern, **not a replication** of that number; its method there involved hysteresis. Our n=100 binomial estimate (6%) is noisy, and the fixed-batch false-alarm rate is not a sequential false-alarm guarantee. The legacy screen includes an uncorrected Mahalanobis >15 rule; attribution of its 43% alarm rate to that rule was not separately measured. Do not merge merely on the 43% to 6% headline: test attack recall across seeds and operational distributions, and calibrate/record the gate for the actual deployment batch size. The grouped fusion change has no measured false-alarm win on this corpus.

`runs/day1/clean_none_fixed_s5/manifest.json` reproduces the same clean spec and therefore the same counts; this is not an independent replication.
