# Supplemental analysis — implemented baselines vs CVIAF

Supporting material for the six-page submission. The main deck stays short; the
unfavourable numbers and the full denominators live here.

Commit `2fda6eab` (`task3-real-backbone`), worktree CVIAF. Reviewed main `6cddfe0e`.
Everything below is a single-cohort laboratory comparison on the MVP corpus
(`runs/mvp/comparison.json`), not an operational evaluation.

---

## 1. What was compared, and what the comparators actually are

Four decision rules were run on the same eight models, the same reference and the same
splits:

| Rule | What it does | Source |
|---|---|---|
| Fixed threshold, single signal | one signal, cut at 0.5 | `cviaf/lab/baseline.py` |
| Reference quantile, single signal | one signal, cut at the clean 99th percentile | `cviaf/lab/baseline.py` |
| Hand-tuned, single signal | one signal, tuned on **this cohort's own clean control** | `cviaf/lab/baseline.py` |
| CVIAF | conformal calibration + fusion + FDR + attribution + abstention | `cviaf/lab/compare.py` |

These are written by the team. **They are not a reproduction of any deployed operational
system**, and the problem statement does not supply an implementation. The hand-tuned rule
spends hindsight: its cut is chosen on the same corpus it is scored on, which is exactly the
practice CVIAF is designed to avoid.

Two attack-success-qualified cases were excluded by the ASR gate. Note how each rule
handles them: a threshold rule silently treats an ASR-gated model as a negative; CVIAF
reports it as ASR-gated and excludes it from scoring.

## 2. Model axis — and why the denominators are small

| System | TPR | FPR | Cells |
|---|---|---|---|
| Fixed threshold | 1.00 | 1.00 | TP 2 / 2, FP 4 / 4 |
| Reference quantile | 0.00 | 0.00 | TP 0 / 2, FP 0 / 4 |
| Hand-tuned | 0.00 | 0.00 | TP 0 / 2, FP 0 / 4 |
| CVIAF | 0.50 | 0.00 | TP 1 / 2, FP 0 / 4 |

Two positives and four negatives. Any rate here moves in steps of 25 percentage points.
This panel is illustrative of behaviour — the shape of the tradeoff — and cannot support a
precision estimate. Do not quote the 0.00 FPRs without the n.

## 3. Data axis — two different units, and they disagree

**Asset level** (7 poisoned assets, 1 clean):

| System | TPR | FPR |
|---|---|---|
| Per-sample outlier, rolled up | 7/7 = 100% | 1/1 = 100% |
| CVIAF asset decision | 2/7 = 28.57% | 0/1 = 0% |

The FPR column is one asset. It shows the direction of the tradeoff, nothing more.

**Sample level** (432 poisoned, 1,560 clean) — the unfavourable result:

| System | TP | FP | TPR | FPR | Precision |
|---|---|---|---|---|---|
| Per-sample outlier baseline | 363 | 24 | 84.03% | 1.54% | 93.80% |
| CVIAF (BH, FDR) | 73 | 51 | 16.90% | 3.27% | 58.87% |

At item level the simple baseline beats CVIAF on recall and precision, and CVIAF's false
alarm rate is *higher*. The reason is the calibration budget: the conformal p-floor is
`1/(n_cal+1)`; with a small calibration set the smallest attainable p-value is too large
for a Benjamini–Yekutieli threshold across hundreds of tests, so the decision abstains
almost everywhere. CVIAF's answer to this is to apply FDR at the **asset** level, one
p-value per model. That is a design choice with a cost, and the cost is visible here.

**Never swap these units.** An asset-level false-alarm reduction is not an item-level
improvement, and the deck must not present it as one.

## 4. Attribution — what the 2/2 actually covers

| Quantity | Value | Boundary |
|---|---|---|
| Kinds with a single culprit | `dup_flood`, `ood_insert` | 2 of 7 kinds |
| CVIAF top-1 correct | 2 / 2 | only those two kinds |
| CVIAF false accusations | 0 | same two kinds |
| Baseline top-1 | undefined | a per-sample flag list has no source field |
| Kinds without a single culprit | `gma`, `label_flip`, `oda`, `oga`, `rma` | trigger attacks are contributor-diffuse by construction |

Both numbers are reported together on purpose: a system that names a culprit every time
would score a perfect top-1 and be useless. 2/2 is not "universal source accuracy".

## 5. Inference provenance — 8 attacked, 2 genuine

| System | Detects | FPR | Abstains |
|---|---|---|---|
| Enrolled digest manifest | 4/8 | 0/2 | 2 attacked, 2 genuine |
| Internal hash consistency | 0/8 | 0/2 | — |
| CVIAF provenance | 8/8 | 0/2 | — |

Per family CVIAF is 2/2 on tampered output, swapped model, forged record and replay.

**Key assumption, unchanged in every quotation of this result:** the committed run used
`signing_mode: hmac-sha256-fallback` because the cryptography extra was absent. HMAC is
*symmetric* — anyone who can verify a seal can also forge one. This is a tamper check under
a shared-key assumption, not public-key non-repudiation. The hash-only comparator is
structurally blind because every hash it recomputes is unkeyed, so an attacker with write
access can make a forged record fully self-consistent.

To claim Ed25519, run and save a new receipt without overwriting this one.

## 6. Detection at the frozen threshold — the headline cells

The clean-null ledger cannot produce a TPR (zero attacked positives). Detection is therefore
measured on the **attack ladder**, at thresholds that ledger froze *before* any attacked arm
was scored — so the rates below are not tuned on the attacks. Source:
`runs/tpr_ladder_at_frozen.json`, rule `refdiv_mean_clean`, alpha 0.05. Classes and doses are
never pooled.

| Attack class (dose unit) | Dose | Caught / scored | TPR (95% Wilson) | Unscorable | Behaviour-inert |
|---|---|---|---|---|---|
| substitution (prune fraction) | 0.25 | **45 / 94** | **47.9%** [38.1, 57.9] | 5 | 40/94 |
| substitution (prune fraction) | 0.50 | **59 / 63** | **93.7%** [84.8, 97.5] | 36 | 6/63 |
| weight_tamper (noise sigma) | 1.00 | **92 / 99** | **92.9%** [86.1, 96.5] | 0 | 0/99 |
| bias_lift (logit shift) | any | **5 / 99** | **5.1%** [2.2, 11.3] | 0 | 99/99 |

The fourth row is the honest failure: bias lift is behaviour-inert on **all 396** arms (zero
relative F1 change), so 5.1% is the rule firing at its own false-alarm rate — not backdoor
recall. It is reported as a measured blind spot, never as a detection.

**Equal calibration, unequal sensitivity.** On clean models `refdiv_mean_clean` (5.18%) and
`ctc_mean_clean` (5.23%) both pass the same 5% test. On attacks they diverge: at substitution
dose 0.50 the RefDiv rule catches 59/63 and CTC-mean 2/63; at weight tamper dose 1.00, 92/99
against 29/99. A rule that looks identical on the false-alarm test can be nearly blind.

**Where these names come from.** CTC and FTC are the transformation-consistency scores of
**TRACE** (CVPR 2025, arXiv 2503.15293, implemented as `trace_ctc` / `trace_ftc` in
`cviaf/lab/detectors.py`) and the `oga` / `rma` / `gma` / `oda` labels are **BadDet**'s attack
taxonomy. The mechanisms are reimplemented at MVP scale and the published figures are the
authors' — they are not quoted as ours anywhere in this package. The full citation list, with
its verification status, is `CV_INTEGRITY_ASSURANCE_2026.md` at the repository root; the log of
what was verified against which primary page is `RESEARCH_CHECKPOINT_26228.md`.

**Two arms sets, never merged.** An earlier single-dose experiment
(`runs/tpr_at_frozen.json`, 198 built / 193 scored) reported substitution **46/94** and weight
tamper **21/99** for the same rule. Those are different arms from the 4-dose ladder's 45/94 and
92/99; averaging or carrying both across is a double count. The earlier figures appear only
under a SUPERSEDED banner (site, manifest row `supplement-superseded-193`,
`assets/data/tpr_superseded_193.csv`).

**Shapes of the curve, not just the headline.** `assets/data/tpr_dose_ladder.csv` carries all
24 cells (2 rules × 3 families × 4 doses) with caught/n, Wilson bounds, the receipt's own
`conclusion` flag and behaviour-inert counts. Substitution at dose 1.00 is 7/7 with
`conclusion: insufficient_denominator` (below `min_positives = 20`) — it is shown hollow and is
never quoted as 100%.

## 7. End-to-end reports — kept apart

| | `oda_s5` | `demo_assurance` |
|---|---|---|
| Findings | 17 | 15 |
| Overall risk | CRITICAL | CRITICAL |
| Disposition | quarantine | QUARANTINE |
| Audit chain | 7 entries, valid | 10 entries, valid |
| Duration | 27.15 s (audit-trail span) | 0.71 s (`assessment_duration_seconds`) |
| Access | white-box | black-box, white-box checks skipped |

Counts and timings are never combined. The intake round trip reads 240 images and 392
boxes through both COCO and YOLO with exact annotation geometry, but PNG pixels are
quantised — so byte-equality is not claimed.

## 8. Large-scale context, with its limits

| Population | n | Statistic |
|---|---|---|
| Merged clean-null ledger | 56,627 clean (28,314 calibration + 28,313 held-out) | ctc_mean 1,481/28,313 = 5.23% [4.98, 5.50]; refdiv_mean 1,467/28,313 = 5.18% [4.93, 5.45] |
| Fleet ledger | 7,503 clean (3,752 / 3,751) | ctc_mean 180/3,751 = 4.80% [4.16, 5.53]; refdiv_mean 215/3,751 = 5.73% [5.03, 6.52] |
| Attack ladder | 1,188 built / 1,055 scored / 133 unscorable | frozen-threshold TPR per family per dose |

Alpha is 5%. The merged intervals include the target, so the honest phrasing is *near the
5% target*, not *below it*. `ctc_peak_clean` and `ctc_q95_clean` sit at threshold 1.0 — the
corpus ceiling — so their zero false-alarm rate is a bound from the statistic's range, not a
measurement of specificity, and those rules cannot fire on an attack either.

This ledger has **zero attacked positives**, so it cannot produce TPR, precision, expected
loss or savings. Any such number needs the ladder or a justified prevalence assumption.

## 9. The two corpora are not additive

The 30 September verified-export record has 22,693 members (fleet 10,000 seeds 150–10149,
macOS 11,637, git 1,056). Two of those components are shards of the 56,627 study itself:
`seed_10150_28385` = 1,056 and `seed_10264_24607` = 11,637 (`docs/MODEL_INVENTORY.md` §3).
Adding the populations double counts. Before any disjointness claim, a byte-level image
dedup receipt is required; distinct seeds are not sufficient. Parent models are reused
across dose variants, so per-arm Wilson intervals are descriptive while that dependence is
unresolved — group-aware uncertainty is the correct treatment.

## 10. What is deliberately absent

No measured reduction in analyst hours, incidents, cost, readiness risk or procurement
expense exists in this repository. No H200 benchmark ledger exists. No image-overlap audit
has landed. No Army deployment has occurred. These are future pilot outcomes; stating them
as results would be the single easiest way to lose the room and the credibility of every
other number on the slide.
