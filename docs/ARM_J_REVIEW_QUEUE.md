# ARM J — Cost-sensitive abstention + review queue: measured results

**Experiment ARM J for CVIAF v-next · base c17141f · measured 2026-09-28 on `runs/mvp` (8 assets, 1 seed) and `runs/day1` (83 assets, ~10 seeds/kind), data axis only.**

## What was built

`cviaf/lab/review.py` — a decision layer above the existing data-axis flag set:

1. **Two-groups posterior.** Per-item anomaly posterior from the same fused conformal p-values the flag set uses (Storey pi0 + histogram local-FDR, with deterministic jitter for discrete conformal ties).
2. **Operator loss model** (`OperatorCosts`: false_accept=25, false_quarantine=5, review=1; declared, editable, recorded in every plan). Each item/cluster gets the argmin action: accept / review / quarantine.
3. **Co-flag clustering.** The 51-co-flag lesson as a decision rule: a near-duplicate cluster contains its own innocent original (the duplicate statistic is symmetric), so quarantining a cluster costs at least one clean item with near-certainty — review dominates quarantine for pairs whenever looking is cheaper than a wrongful removal. Flagged mutual-nearest-neighbours below the duplicate signal's own significance cut collapse into ONE review task (members shown together with provenance).
4. **Abstention gate.** When the calibration floor makes item-level significance impossible (smallest attainable fused p > alpha), `accept` is removed from the action set; the queue is forced and the reason recorded. No low-power "clean" is emitted.
5. **Queue membership = FDR flag set ∪ local-FDR band (lfdr ≤ 0.5, p ≤ 0.25) ∪ gate-forced.** Costs never decide membership — a loss multiplied by a phantom posterior is how a clean asset gets a work queue.

## The benchmark: 51-co-flag dup flood (`dup_flood_none_fixed_s5`)

Current flag-all behaviour hands the analyst 99 unordered items for 48 true poison (precision 0.48): the 48 flood copies plus 51 co-flagged innocent originals.

| policy | review actions for 100% catch | actions for 90% catch | clean items quarantined | realised cost |
|---|---|---|---|---|
| flag-all + review pile | 99 | 90 (expected) | 0 | 99 |
| flag-all + quarantine pile | 0 | 0 | 51 | 255 |
| **ARM J queue** | **59** | **44** | 0 | **59** |

Same 48/48 catch. The win is the clustering (48 pairs → 48 tasks) plus ranked order; it is *not* free — see the honest losses below.

## Multi-seed (`runs/day1`, 83 assets)

dup_flood across 10 seeds: ARM J tasks 52–78 vs flag piles 96–102 on every seed; budget@90% = 44 vs 87–92 on every seed. The benchmark result is not a one-seed artifact.

Pooled operator ledger over all 83 assets (loss matrix above):

| policy | reviews | poison caught | poison missed | clean quarantined | total cost |
|---|---|---|---|---|---|
| flag-all + review | 1313 | 784 | 3680 | 0 | 93313 |
| flag-all + quarantine | 0 | 784 | 3680 | 529 | 94645 |
| **ARM J** | **5119** | **2774** | **1690** | 0 | **47369** |

ARM J spends 3.9× the review actions, catches 3.5× the poison, and halves total operator cost under the declared matrix. The catch gain comes from the local-FDR band recovering sub-threshold poison on oga/oda (where BH cannot reject) — at real review spend.

Per-kind cost ratio (ARM J / flag-all-review): oda 0.07, oga 0.10, dup_flood 0.60, clean 0.67, rma 0.68, gma 0.89, label_flip 1.00, **ood_insert 1.18**.

## Honest losses (why this is not merged-blind)

- **ood_insert is a regression (1.18×).** Flag-all is already exact there (24 flags, precision 1.0); the lfdr band adds ~8 low-value reviews per asset. The band pays off where BH is too weak, and costs where BH is already right.
- **label_flip is a tie (1.00×).** Image-level signals are blind to label-only attacks; the queue cannot review its way out of a blind signal. That is U2's residual-risk/power-gate territory, not a queue fix.
- **gma/rma partial misses persist.** ARM J reviews ~90 items/asset and still misses most gma poison; its cost edge there comes from catching what the band sees, not from power the signals do not have.
- The two-groups posterior is a crude histogram estimator with jittered ties; it is a ranking/decision quantity, never a published probability.

## Reproduce

```bash
python3 -m pytest tests/test_review.py -q          # 17 policy tests
python3 -m pytest tests/ -q                        # 40 total, all green
python3 scripts/arm_j_review_measurement.py runs/mvp  runs/arm_j/data_axis_mvp.json    # ~40 s
python3 scripts/arm_j_review_measurement.py runs/day1 runs/arm_j/data_axis_day1.json   # ~85 s
```

(The runner loads the registry and calls `cviaf.lab.compare.data_axis`; full `compare_corpus` needs `scikit-learn` for the provenance axis, absent in this environment.)

## Incidental fix

`compare.py` model axis crashed on `clean_peaks` KeyError ('ftc') at c17141f — pre-existing, unrelated to ARM J, fixed with `setdefault` because the measurement could not run without it.

## Files

- `cviaf/lab/review.py` (new, ~470 lines): policy + measurement.
- `cviaf/lab/detectors.py`: `duplicate_scores(..., return_neighbors=True)` — additive.
- `cviaf/lab/compare.py`: `_cviaf_item_pvalues` refactor (same arithmetic), per-asset `review_policy` block, `review_queue` aggregate in data-axis scores, `clean_peaks` fix.
- `tests/test_review.py` (new): 17 tests — posterior properties, cost model, clustering, gate, curves/ledgers, end-to-end synthetic dup flood.
- `runs/arm_j/*.json`: measured outputs.
