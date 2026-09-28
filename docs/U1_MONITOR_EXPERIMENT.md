# U1 monitoring experiment (lab-only, base c17141f)

Reproduce against the committed MVP model artifacts: `python -m cviaf.lab.monitor_experiment --corpus runs/mvp --model oga_patch_fixed_s5 --out /tmp/oga-u1.json`. Each model takes about 30-45 seconds in this environment; repeat for each model ID in `runs/mvp/registry.jsonl`. `python -m pytest tests/ -q`: 28 passed (2026-09-28). No engine monitor or production audit log has been changed.

Each frozen per-detector calibration contains 120 contributor-disjoint clean images. A predeclared 75th-percentile cutoff and simultaneous DKW upper tail bound (total delta=0.005, Bonferroni allocated over three detectors) define `e = 1 + 0.5 * (1{score > cutoff} - upper_tail_bound)`. The average of three e-values is valid without cross-detector independence. A separate per-asset product ledger records scores, per-detector and fused e-values, log wealth, running threshold crossing and hash links. It triggers at `1/(alpha-delta)` for alpha=0.05, so the unconditional bound is alpha, **provided future clean observations are iid from the calibrated null and the calibration set is independent and frozen**. More generally, a predictable conditional bound on each future tail event suffices. Arbitrary temporal dependence or unmodeled deployment shift does not. Wealth itself is not monotone; its running maximum/alarm flag is.

## Results

200 resampled 80-look streams per model, random seed 20260928. Each stream samples with replacement from that model's 80 held-out clean scores and, separately, 80 attack-view scores; the same sampled image index is used across detectors. These are bootstrap pseudo-streams, not 200 independently generated deployment assets. False alarms use each model's clean null. The comparator is **uncorrected repeated fixed-sample conformal Bonferroni** at 0.05 on each look; its delay is not a fair equal-error comparison. The attack views for data-only attacks are identical to their clean views because these are inference-time model probes and cannot test dataset integrity.

| class | clean false alarm e / repeated-fixed at look 20 | at look 80 | attack e detection / median first alarm by 80 | fixed detection / median |
|---|---:|---:|---:|---:|
| clean control | 0 / .360 | 0 / .875 | 0 / n/a | .915 / 24 |
| OGA | 0 / .360 | 0 / .885 | 1.000 / 50 | 1.000 / 1 |
| ODA | not scored | not scored | not scored | not scored |
| RMA (ASR gate weak) | 0 / .775 | 0 / 1.000 | 1.000 / 20 | 1.000 / 1 |
| GMA (ASR gate weak) | 0 / .225 | 0 / .645 | 1.000 / 33 | 1.000 / 2 |
| label flip | 0 / .250 | 0 / .680 | 0 / n/a | .610 / 30 |
| duplicate flood | 0 / .530 | 0 / .905 | 0 / n/a | .945 / 18 |
| OOD insertion | 0 / 0 | 0 / 0 | 0 / n/a | 0 / n/a |

ODA has one nonfinite CTC attack score, so the runner explicitly abstains rather than silently dropping it and changing the predeclared detector mix. RMA/GMA were marked weak in the existing manifest and are not evidence of meaningful backdoor detection. Clean-model triggered views are the same clean images; the comparator flags there are false alarms. Fixed comparator is fast because it burns type-I error on every look; U1 detects OGA but is 50x slower than that unmatched comparator. This experiment **does not meet** the design's claimed <=1.5x detection-delay criterion, nor does it demonstrate improvement on every attack family. An equal-error fixed-window comparator, independent clean assets, additional attack seeds and shift/replay stress tests remain to be done before promotion.

## Limits

Calibration is selected and frozen per model; scores are not calibrated on the attacked stream. DKW controls the calibration uncertainty for each detector, then the union bound and Ville inequality yield overall alpha. These conclusions are not distribution-free under arbitrary monitoring sequences, and are invalid if calibration sets or bets are retuned using monitored observations. The bootstrap reuses 80 examples, and correlated consecutive images violate its iid story in real use. Per-asset alpha does not protect a fleet without an additional multiple-asset policy. Hash linking is a local lab record, not a signed durable audit log. The ledger gives no posterior probability of attack, and e-value averaging is not protection from an entirely blind detector diluting power.
