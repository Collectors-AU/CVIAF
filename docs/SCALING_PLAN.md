# Scaling Plan — M3 Air → H200

What changes, what stays, and how to spend GPU hours so that a 4-hour session produces evidence rather than a screenshot.

> Companion to `docs/CVIAF_V3_ARCHITECTURE.md` (design) and `docs/MVP_MAC.md` (the laptop MVP). The MVP has already produced a calibrated detection table and a working detector pair; this document is about buying **realism**, not about buying existence.

---

## 1. The principle: the H200 buys realism, not method

The M3 MVP demonstrated the method end to end at tiny scale:

| attack | ASR | CTC | REFDIV | fused | TPR@5% (fused) |
|---|---|---|---|---|---|
| clean *(control)* | — | 0.500 | 0.500 | 0.500 | 0.062 |
| oga (fabrication) | 1.00 | 0.819 | 0.983 | **0.987** | **0.963** |
| oda (cloaking) | 0.99 | 0.441 *(blind)* | 0.949 | **0.906** | **0.512** |

Nothing in that table depends on the detector being a *good* detector. It depends on the ground truth being known, the calibration split being clean, and the ground-truth attack succeeding. All three are properties of the **laboratory**, and the laboratory is scale-invariant.

So the GPU is not there to make the framework work. It is there to make the framework's numbers **transferable**: real depth, real scale, real imagery, real class diversity, real optimisation noise. Those are exactly the things that break assurance claims in the wild, and they are exactly what a laptop cannot buy.

**Consequence for planning:** never spend GPU time on something the M3 has not already validated. Every H200 experiment below has a corresponding M3 result that predicts its outcome. If an H200 run contradicts the M3 run, that contradiction *is* the finding — report it, do not tune until it agrees.

---

## 2. The interface contract (why nothing needs rewriting)

The lab is built so the backbone is a swap, not a rewrite. Four functions define the contract.

```python
# cviaf/lab/detector.py — the whole contract
class Detector:
    def predict(self, image, score_thresh, iou_thresh, top_k) -> {
        "boxes":   (M,4) float32 xyxy,
        "scores":  (M,)  float32,
        "labels":  (M,)  int64,
        "cells":   (M,2) int64,     # grid cell of each detection -- CTC needs this
        "obj":     (G,G) float32,   # raw objectness map    -- CTC reads this directly
        "cls":     (G,G,K) float32,
    }
    def features_raw(self, image) -> np.ndarray   # for feature caching (optional)
    def build_targets(self, boxes, labels)         # for the attack assignment
    def digest(self) -> str                        # weight identity
```

Every assurance detector consumes only `predict()`. So the port is:

| Component | M3 today | H200 target | Changes to the engine? |
|---|---|---|---|
| Backbone | frozen random conv bank, 16 ch | YOLOv8n/s or Faster-RCNN-R50, trainable | **No** |
| Head | 1 hidden layer, 48 units, numpy | native detection head | **No** |
| Objectness map for CTC | `obj` (G,G) | sigmoid of the class-agnostic score map | **No** — same contract |
| `cells` for CTC | grid cell indices | anchor/point index | **No** — same contract |
| Trainer | numpy Adam, seconds | torch/YOLO, hours | **No** — the engine never trains |
| Dataset | procedural | COCO-subset + VOC | **No** — the loaders already exist |
| Everything else | — | — | untouched |

Two things must be preserved when porting, and both are easy to get wrong:

1. **`obj` must be the *raw pre-NMS, pre-threshold* objectness map.** Replacing it with a thresholded or post-NMS score is the exact mistake that produced AUROC 0.435 instead of 0.894 on the M3 (see `MVP_MAC.md` §6). If the new backbone exposes only detections, expose the raw map instead — otherwise CTC is reintroduced to the same bug at a larger scale.
2. **`cells` must identify where the detection came from.** CTC needs to read the objectness in the same spatial location across background blends. For anchor-based heads this is (feature_pyramid_level, anchor_index); for DETR it is the object *query* index, which is the naive porting trap: query indices are not spatially meaningful, so CTC must be re-derived for set-prediction heads or the DETR row must be declared **not covered** with that reason.

That second point is a genuine research decision, not a mechanical one. Budget an afternoon for it.

---

## 3. Compute arithmetic (so you can size the request honestly)

Per detector training run on COCO-subset, YOLOv8n-class (~3.0 M params):

| Quantity | Value | Note |
|---|---|---|
| Images for a credible subset | 8 000–20 000 | ≥ 2 000 per class for label-error and contributor statistics to have power |
| Epochs to a comparable clean mAP | 150–300 | 300 for the corpus of record |
| H200 throughput, 640², batch 64 | ~1 200–1 800 img/s | YOLOv8n, AMP, 3–6 jobs concurrent at 141 GB |
| Wall clock per model | **~2–4 h** | single job; ~1 h with 4 concurrent jobs |
| Models needed for a credible corpus | **24–40** | 6 attack classes × 4 seeds, or 8 classes × 3 seeds |
| **Corpus generation total** | **~10–18 GPU-hours** | concurrency-dependent |

Detection (inference-only, no retraining — the baseline path):

| Check | Cost per model | Note |
|---|---|---|
| TRACE CTC | 9 forward passes × N eval images | N = 500 → ~4 500 images; minutes |
| TRACE FTC | grid probes × N images | ~64 probes/image → **14× CTC**; sample down to N=100 |
| DISTIL trigger inversion | > 24 GB VRAM, tens of minutes | H200 only; M3 cannot run it at all |
| ODSCAN scan | minutes to hours | white-box, per model |
| Reference divergence | 2 forward passes × N | cheapest useful check |
| **Detection sweeps total** | **~4–8 GPU-hours** | the whole point of the session |

**Key ratio to present:** corpus generation (~12 h) is *one-time evidence production*; detection (~6 h) is *the method being demonstrated*. If a reviewer asks why the GPU is needed, the answer is that 40 models × 3 h cannot run on a fanless 16 GB laptop in any calendar, while the framework itself is inference-only and runs anywhere.

**Disk.** COCO 2017 (~20 GB), VOC (~3 GB), corpus weights (~2–4 GB), TRACE background assets + SSCD weights (~5–8 GB), results and checkpoints (~5–10 GB): budget **~50 GB** scratch.

---

## 4. What stays on the laptop (do not waste H200 minutes)

Everything below is validated on the M3, is CPU-bound, or is deterministic and cacheable. Run it at home; the GPU is for GPU work.

- All conformal calibration, Cauchy/Bonferroni fusion, BY/BH control, power curves, ECE.
- The whole attack-injection suite and ground-truth ledger.
- Provenance cryptography, DSSE/in-toto envelopes, Merkle checkpoints, ML-BOM emission.
- Everything in `cviaf/` — the engine, the governance layer, the report schema.
- **Feature caching for a frozen backbone** — one pass, then head training is seconds. The M3 already does this; do the same trick for any adapter/linear-probe experiments so they never need the GPU twice.
- The offline bundle build, the no-egress test, and the air-gap rehearsal.

The failure mode to avoid: burning a session on a bug that a 3-second M3 unit test would have caught. Run `pytest` and a small `corpus` run before every session.

---

## 5. Session plan

### Session A — corpus generation (4–8 h)

Priority order, and stop wherever the budget runs out.

1. **Reference models first** (2 clean, different seeds). Everything downstream is scored against them, and `refdiv` cannot be calibrated without at least two independent clean models. **This is the most common planning mistake** — generating attackers before references leaves you unable to compute the FN-side signal at all.
2. **`oga` × 3 seeds** — the FP-inducing arm; the M3 predicts AUROC ~0.82–0.99.
3. **`oda` × 3 seeds** — the FN-inducing arm; M3 predicts CTC-blind and REFDIV ~0.95. This is the row that proves detector-native design was necessary.
4. **`rma`, `gma` × 3 seeds** — the region and global categories. On a real backbone with global context, `gma` **should now implant** where it could not on the M3. Verify that explicitly: it directly tests the diagnosis in `MVP_MAC.md` §6.
5. **BadDet+ penalty variant** — physical-world, position/scale-invariant. Expect the hardest case; a *degraded* number here is a result, not a failure.

Checkpoint and copy results off immediately. Never leave a session's output only on the remote box.

### Session B — detection, calibration, red team (4–8 h)

1. Port and verify CTC against the M3 result (M3 AUROC 0.819 on `oga`). If the H200 number is materially lower, the most likely cause is the `obj`/`cells` contract break in §2 — check that before doubting the method.
2. Run the detector sweep: TRACE CTC + FTC, reference divergence, ODSCAN, and `model.neuralcleanse` **as a labelled classifier-derived control** (it exists to be beaten, and showing *why* it loses on detectors is worth a slide).
3. Calibrate on contributor-disjoint splits, report ECE before/after temperature scaling.
4. **Power curve / detection floor**: poison rate 0.1% → 20%. Publish the rate at which power reaches 0.8. This is the single most valuable output of the whole GPU budget, because it converts "we detect backdoors" into "we detect backdoors above X%, and here is what we cannot exclude."
5. **Red team ourselves**: variance-flattening regulariser against CTC; poison-rate sweep below the floor; contributor dilution flood; replay with a valid nonce; benign ONNX↔TorchScript re-export (must raise **no** substitution finding).
6. Optional: ODPure / Lite-BD purification, with clean-mAP delta reported alongside residual ASR.

---

## 6. Data plan

| Purpose | Source | Notes |
|---|---|---|
| Reference distribution + main corpus | COCO 2017 subset (8–20 k images) | record the exact image-id list; it is part of the battery digest |
| Second distribution for OOD | PASCAL VOC | genuinely different acquisition statistics |
| Declared shift axes | COCO: night / rain subsets; add synthetic illumination and gamma like the M3 generator | the engine only needs a *declared* reference distribution |
| TRACE background/foreground imagery | vendor a fixed public image folder into `AB1.probe` | closes the M3's stated air-gap limitation |
| Deduplication | SSCD embeddings | replaces the aHash first pass |
| Label errors | cleanlab `object_detection` | COCO/YOLO-native, no retraining |

**Licensing.** The problem statement requires publicly available or team-generated data. Record a provenance and licence entry per source in the battery manifest. That entry is part of the evidence, not paperwork.

---

## 7. What to buy with the GPU, in strict priority order

If the session is cut short, this is the order that maximises defensible evidence.

| Rank | Buy | Why it outranks the rest |
|---|---|---|
| 1 | **2 clean reference models** | Nothing else can be scored or calibrated without them |
| 2 | **`oga` + `oda`, 3 seeds each** | The complementary pair; CTC for one, REFDIV for the other. Proves detector-native design |
| 3 | **Power curve / detection floor** | The honest boundary. Turns a claim into a number |
| 4 | **`gma` on a global-context backbone** | Tests the M3's architectural diagnosis directly |
| 5 | **Red-team adaptation** | The difference between a demo and a security product |
| 6 | **Calibration (ECE) + OD verification** | Backs the "calibrated score" requirement |
| 7 | **BadDet+ physical variant** | Open problem; a degraded number is still a contribution |
| 8 | **Purification benchmarks** | Remediation you can stand behind |
| 9 | **DISTIL trigger inversion** | Expensive; only after everything above |

---

## 8. Risks, and the pre-committed response

| Risk | Likely? | Response decided **now**, so it is not decided under pressure |
|---|---|---|
| Attack fails to implant (ASR < floor) | High — happened on the M3 for `gma` and one `rma` config | The ASR gate excludes it automatically and the exclusion is reported. Do **not** raise the poison rate until it implants and then quote the number as if it were the original setting — that is how reported ASR quietly becomes fiction |
| CTC underperforms the M3 number | Medium | Check the `obj`/`cells` contract first (§2). If it holds, publish the lower number with the discrepancy |
| Fusion worse than the best single detector | Low (fixed on the M3) | Report both FUSE-B and FUSE-C; a fusion that loses to its own components is a finding about correlation, not a bug to hide |
| Session cut short | Medium | Priority order in §7; resumption is by corpus skip-on-digest, so a re-run costs nothing |
| GPU-hour overrun on DISTIL | Medium | It is rank 9. Not scheduled in Session A or B |
| Detector mAP too low to be credible | Low–Medium | Raise epochs or subset size; a low mAP weakens every downstream statistic, so fix it before generating the corpus of record |

---

## 9. Definition of done for the GPU work

The scaling work is complete when all of the following hold, because each maps directly onto a scored deliverable:

- [ ] ≥ 24 models with manifests, ASR, seeds, and weight digests; resumable corpus runner
- [ ] A calibrated results table with **control rows at 0.500** and per-seed spread, not a single run
- [ ] A published **detection floor** (minimum poison rate at power ≥ 0.8, FDR ≤ 0.05)
- [ ] CTC measured on a real backbone, and the `obj`/`cells` port verified against the M3 baseline
- [ ] The `gma` global-context question answered explicitly, one way or the other
- [ ] At least four adaptive/red-team results, including the benign-re-export false-positive test
- [ ] Every model and dataset artifact content-addressed and reproducible from its manifest
- [ ] The coverage matrix regenerated for the full corpus, with `gma`/BadDet+ cells honestly marked
- [ ] The full pipeline re-run **with networking disabled** to close the air-gap claim

The last item is a two-hour task with an outsized effect on a security-themed judging panel: it turns "we do not use the cloud" from an assertion into a demonstration.
