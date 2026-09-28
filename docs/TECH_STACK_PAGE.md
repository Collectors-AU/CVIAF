# CVIAF — Tech Stack Page Content (SIH Grand Finale)

Everything on this page is backed by a file in this repository. Every number below
was produced by a command you can re-run; the artifact path is printed next to it.
Where a claim is *not* supported, it is in the blacklist at the end — read that
section before you print anything.

---

## 0. Page layout

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ROW 0 · THE CONNECTED SENTENCE  (layman, full width, reads left to right)    │
│ 5 boxes joined by arrows — the whole system in one breath                    │
├───────────────────────────────────────┬──────────────────────────────────────┤
│ LEFT — "what they gave us"  (2/3)     │ RIGHT — "what we tell them"  (1/3)   │
│                                       │                                      │
│  ① DATA INTEGRITY                     │  ④ DISTRIBUTION SHIFT                │
│  ② MODEL INTEGRITY                    │  ⑤ GOVERNANCE & EVIDENCE             │
│  ③ INFERENCE PROVENANCE               │                                      │
│                                       │                                      │
│  (vertical flow: ① ② ③ feed ——▶ )     │  (both receive from the left)        │
├───────────────────────────────────────┴──────────────────────────────────────┤
│ ROW 3 · BASELINE → CVIAF  (full width strip: "what changes when we come in") │
├──────────────────────────────────────────────────────────────────────────────┤
│ ROW 4 · EVIDENCE FOOTER — 5 artifact names + the one number each             │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Reading rule for the room.** Row 0 is for everyone. Rows 1–2 are split so that a
non-technical judge can stop after the first line of each card. The second and third
lines of each card, plus Row 3, are for the technical judges — that is where the
mechanism, the numbers and the honest gaps live. Row 4 is the proof-of-life row: if
someone doubts a claim, this is the file to open.

---

## ROW 0 — The connected sentence (layman, full width)

Five boxes, each one short line. Arrows left to right. No jargon anywhere in this row.

**① A dataset arrives from contributors.**
We don't trust it because it arrived; we measure it against a clean reference we control.

**② A model arrives.**
We don't trust it because it's signed; we probe its behaviour for hidden triggers and
compare it to a reference detector we trained ourselves.

**③ Every inference it produced arrives as a record.**
We don't trust the record because it has a hash; we check a signature that binds the
picture, the model, the settings and the answer into one sealed unit.

**④ We watch for the world changing underneath.**
Terrain, season, sensor, lighting — and the subset of changes that are *not* the world
changing.

**⑤ You get one report, not five alerts.**
Every flag says why, with what evidence, at what confidence, and what to do: accept,
review, or quarantine. Plus a list of what we *can't* catch.

**The one-line version for the top of the slide:**

> **Every artefact that enters a computer-vision pipeline — the data, the model, the
> output record — is measured against a reference we control, and every conclusion is
> published with its confidence, its evidence and the limits of what it can prove.**

---

## ① DATA INTEGRITY — left, top

**PLAIN (top line on the card):**
> We tell you whether a contributed dataset is poisoned, flooded with near-duplicates,
> or drawn from a different world — and *which contributor* it came from.

**CARD (three bullets):**
- Five attack families: trigger injection, label flipping, systematic mislabelling,
  near-duplicate flooding, out-of-distribution insertion.
- Sample-level evidence is **aggregated to source level** — a contributor risk
  posterior, not a pile of flagged files.
- Caught **and attributed** on this corpus: near-duplicate flooding and OOD insertion,
  both correctly blamed on `vendor_x`, with zero false accusations.

**UNDER THE FOLD (technical judges):**
- Signals: spectral signatures (top right-singular direction per class), label-noise
  scoring (confident-learning style), duplicate distance on average-hash descriptors,
  Mahalanobis + isolation-forest outlier scoring.
- **Calibration, not thresholds.** Every score becomes a distribution-free conformal
  p-value against a pooled clean reference — a validity statement, not a tuned constant.
- **Two-signal fusion by `min(1, m·min p)`**, chosen because a signal that is
  structurally blind cannot dilute one that works (the same argument as the model side).
- **Multiplicity control at the level an operator acts on.** Benjamini–Hochberg at the
  sample level and Benjamini–Yekutieli across assets; the report states when the
  conformal floor makes an item-level decision impossible instead of quietly returning
  a ranked list.
- **The calibration budget is derived, not guessed:** an item-level BH decision over
  *n* items at α needs `n_cal ≳ m·n/α`, so the reference pool is 80 independent clean
  splits of 240 → **19,200 calibration samples, conformal floor 5.2e-05**.
  Split pools matter: the duplicate statistic is a *within-split* nearest-neighbour
  distance, so pooling across differently-sized splits would compare two different
  statistics and quietly void the guarantee.
- Direction is declared per signal — the duplicate score is a *distance*, so the
  anomaly is the **low** tail. Assuming "higher is worse" would have made that detector
  perfectly anti-correlated and still look like it was running.
- Contributor aggregation: hierarchical Beta–Binomial posterior, `P(θ_c > f_pop + τ)`,
  which resists dilution (flooding clean images cannot lower a posterior the way it
  lowers a flag *rate*).

**PROOF (artifact + number):**
`runs/mvp/comparison.json`
- `dup_flood`: **item TPR 1.00**, 99 items flagged, **attributed to `vendor_x`**.
  The single-signal baseline scored **0.021** on the same attack.
- `ood_insert`: **24/24**, attributed to `vendor_x`.
- Top-1 source accuracy **1.00** (2/2), false accusations **0.00**.
- Clean contribution: **0** flags. Asset-level FPR **0.00**.

**NOT CLAIMED:**
- **Trigger patches are a declared gap in the data-side signals as measured here.**
  With 64×64 pixel-statistics features, `oga` scored item TPR 0.021 and `oda`/`rma`
  0.000 — the pixel evidence is too weak at this resolution. On those attacks the
  *model* side carries them (see ②).
- **Label flipping is a declared gap on both sides** (CVIAF 0 items; baseline 4 flags,
  TPR 0.083). The fix is a trained classifier's loss/confident-learning signal, which
  is on the roadmap and is *not* in this measurement.
- Sample-level precision is **0.59** at the nominal 5% FDR — the realised false
  discovery rate exceeds the nominal one because adjacent duplicate statistics are
  dependent and the calibration granularity is finite. Stated in the artifact.

---

## ② MODEL INTEGRITY — left, middle

**PLAIN (top line on the card):**
> We tell you whether the model you were handed actually is the model it claims to be —
> or whether it carries a hidden trigger that makes it see, or stop seeing, something.

**CARD (three bullets):**
- Two complementary probes, chosen because **each is blind to what the other catches**:
  fabrication/misclassification vs disappearance/cloaking.
- Works white-box (weights + trigger reverse-engineering) and degrades to black-box
  with the skipped checks **named in the report**, never silently skipped.
- Control models sit at exactly **0.500** — the framework does not alarm on clean models.

**UNDER THE FOLD (technical judges):**
- **CTC** (contextual transformation consistency, from TRACE): blend each image over
  many backgrounds and measure the stability of the model's **raw objectness** at each
  detection cell. A triggered object stops depending on the scene, so it stays
  abnormally stable. Measured on the continuous belief, not the thresholded decision —
  an earlier version tracked post-NMS scores and returned AUROC 0.435; fixing *what is
  measured* (not the data, not the model) moved it to 0.894.
- **Reference divergence**: the FN side. Cloaking removes the detection, so nothing
  remains whose stability can be measured — CTC is structurally blind. What survives is
  disagreement with an enrolled clean reference detector. This is literally the
  "comparison against a defined reference battery" the PS asks for.
- Fusion reported **two ways**, because the choice is not universal: `min(1, m·min p)`
  cannot be diluted by a blind detector; Cauchy combination has more power when several
  detectors carry moderate independent evidence. Both are printed instead of one being
  asserted.
- Conformal p-values calibrated on the model's own clean side; one asset-level p-value
  per model; Cauchy/Bonferroni reported side by side.
- **The ASR gate:** attack success rate is measured on held-out, freshly-triggered images
  before any detector is scored. A model whose backdoor never implanted is excluded and
  reported as excluded — a detector cannot detect a backdoor that is not there.

**PROOF (artifact + number):**
`runs/mvp/eval.json` (one seed) and `runs/day1/eval.json` (32 models, 4 seeds)
- `oga` fabrication: CTC 0.819 · refdiv 0.983 · **fused 0.987**, TPR@5%FPR **0.963**
- `oda` cloaking: **CTC 0.441 — blind**, refdiv 0.949, **fused 0.906**, TPR@5% **0.512**
- Control rows across **four independent seeds: exactly 0.500**
- 4-seed means: `oda` fused **0.796** (CTC 0.429), `rma` **0.833**, `oga` **0.880**
  over the 2 of 4 seeds whose backdoor actually implanted

**NOT CLAIMED:**
- **Single-seed asset-level decisions are underpowered.** At α = 0.05 on seed 5, CVIAF
  flags `oga` (p = 0.039) and misses `oda` (p = 0.084) — while the AUROC for `oda` is
  0.91. Quote the AUROC and the 4-seed spread; do not quote one seed's p-value as a
  verdict.
- `gma` (global-effect relabelling) does **not** implant on a fully-convolutional
  backbone — 0 of 4 seeds — and is correctly gated out. A global rule is not
  expressible by a cell with a local receptive field. Real backbone: real result.
- These numbers come from a deliberately tiny backbone. The *assurance logic* is
  model-agnostic; the numbers are not yet YOLO-scale. `docs/SCALING_PLAN.md` §2 is the
  interface swap.
- A model-integrity pass is evidence of *disagreement with a reference*, not proof of
  malice: benign retraining, quantisation or a legitimate fine-tune also diverge.

---

## ③ INFERENCE PROVENANCE — left, bottom

**PLAIN (top line on the card):**
> Every answer the model produced is sealed together with the picture, the model and the
> settings that produced it — so editing the record, swapping the model, or resubmitting
> an old answer is detectable, even by someone who owns the storage.

**CARD (three bullets):**
- One seal binds **input image · model weight digest · preprocessing/inference config ·
  output · nonce · sequence · previous seal**, signed, chained and Merkle-batchable.
- The adversary in our test **recomputes every hash** after editing — hash-only checking
  is defeated by design, and we measure that instead of claiming it.
- Measured: hash-only **TPR 0.50**, ours **TPR 1.00 / FPR 0.00**, with the attack class
  named (tampering vs replay).

**UNDER THE FOLD (technical judges):**
- Ed25519 signatures (asymmetric) with an explicit HMAC-SHA256 fallback; the fallback is
  **detected and declared** in the report, because symmetric signing gives tamper-evidence
  but *not* non-repudiation against a party holding the verification key.
- Replay controls: monotonic sequence number + per-record nonce, checked against the
  **trusted history**, not against everything the process ever issued — an engine that
  remembers its own output calls every genuine record a replay. Sealer and verifier are
  separate objects sharing only the key manager.
- Attack matrix is designed at full adversary strength, and the four cases are separable:
  in-place output edit, model re-attribution, forgotten new record, byte-identical replay.
- Tamper-evident audit trail is a SHA-256 hash chain over the report's own actions;
  `verify-audit` recomputes it and exits non-zero on the first break.

**PROOF (artifact + number):**
`runs/mvp/comparison.json` → `provenance_axis`, `runs/assurance/oda_s5/assurance_report.json`
- Tampered output **2/2** · swapped model **2/2** · forged record **2/2** · replay **2/2**
- Genuine records: **0/2 false alarms**
- Hash-only enrolment baseline: catches the two in-place edits, **misses both replays**,
  and **abstains on the 4 records it never enrolled** (2 forged, 2 genuine) — a control
  that cannot assess an artefact it has not already seen is a transport control, not an
  assurance control. Note the asymmetry: its abstention is *not* a false alarm, it is an
  absence of assessment, which is exactly why abstention is a third outcome rather than
  a synonym for "clear".
- Internal-consistency-only checking: **TPR 0.00** — the adversary recomputed the hashes.
- Audit trail in the end-to-end run: **7 hash-linked entries, chain verified valid**.

**NOT CLAIMED:**
- This run signed with **HMAC-SHA256**, not Ed25519 (`cryptography` not installed in this
  environment). The report says so in its assumptions. Install the `full` extra for
  non-repudiation.
- Seals detect tampering of **sealed** records only. An inference that was never sealed
  is outside the system's reach — that is a deployment requirement, not a detector.
- Cross-hardware float non-determinism can distort output-hash equality; the report
  lists it as a limitation.

---

## ④ DISTRIBUTION SHIFT — right, top

**PLAIN (top line on the card):**
> The world doesn't stay put — terrain, season, sensor, light. We tell you when the data
> has moved, how far, and whether the evidence says "the world changed" or "someone did
> this".

**CARD (three bullets):**
- Reference distribution is a **contributor-disjoint clean holdout**, so a model is never
  checked against data the same source supplied.
- Characterises the shift, not just detects it: type, magnitude, affected features,
  and a natural-vs-manipulation assessment.
- Explicitly reports *not knowing the difference* when the evidence doesn't support it.

**UNDER THE FOLD (technical judges):**
- Mahalanobis distance against the reference (robust covariance when data allows), MMD
  with median-heuristic bandwidth and a permutation p-value, per-feature
  Kolmogorov–Smirnov, and maximum-softmax-probability when logits are available.
- Exchangeability is protected by construction: calibration items are contributor-disjoint
  from test items. Contributor-level clustering is the most commonly botched detail in
  applied conformal work — it silently makes every p-value optimistic.
- Output is a calibrated score plus a **characterisation**, so the downstream disposition
  can distinguish `review` (drift) from `quarantine` (manipulation-shaped).

**PROOF (artifact + number):**
`runs/assurance/oda_s5/bundle.json` → `modules.distribution_shift`
- On the poisoned contribution: **MMD p = 0.0000**, mean Mahalanobis **26.78**,
  type `covariate_shift`, verdict `probable_natural_drift`, severity MEDIUM.

**NOT CLAIMED — and this is the honest weak spot, so say it first:**
- That dataset genuinely *was* manipulated (a 40%-rate on-object trigger). The module
  correctly said "materially shifted" and **incorrectly leaned toward natural** rather
  than suspicious. So: **detection yes, attribution of intent no.** The discriminator
  needs a shape test — manipulation is localised and repeatable, natural drift is
  distributed — which is designed and not yet measured.

---

## ⑤ GOVERNANCE & EVIDENCE — right, bottom

**PLAIN (top line on the card):**
> One report, not five alerts. Every flag carries why, what evidence, how confident, what
> asset, and what to do. Plus the list of what we cannot catch — signed, logged and
> reproducible.

**CARD (three bullets):**
- **Every finding:** reason · evidence · confidence · affected asset · disposition
  (`accept` / `review` / `quarantine`) · remediation.
- **Tamper-evident audit log** — a hash chain over every module's action, with each
  module's output digest recorded, so a re-run is comparable or visibly divergent.
- **Coverage statement:** 14 supported attack classes, 8 conditions declared
  out of scope, 5 stated assumptions. The out-of-scope list is a refusal to claim,
  not a to-do list.

**UNDER THE FOLD (technical judges):**
- Severity and disposition are computed from the merged findings with an explicit rule
  (any critical, or ≥3 high → quarantine; any high → review), so the verdict is
  reproducible rather than editorial.
- **`accept` is forbidden whenever an applicable check did not run.** A module that
  could not execute is reported absent, and its absence blocks a clean verdict on that
  axis. That is the difference between "clean" and "we did not look".
- Limitations are collected from the modules themselves and printed in the report —
  including the access-level limitations of the run.
- **Assurance-report schema `assurance-report-3.0.0`**: JSON Schema draft 2020-12,
  shipped as a file (`schemas/assurance-report.schema.json`) *and* enforced by a
  dependency-free validator in the repo, so the pipeline fails its own output instead
  of emitting a report a consumer cannot parse.
- Evidence bundle: report, audit trail, coverage statement and inference records, each
  with a SHA-256 in `bundle.json`, plus the exact command and seeds to reproduce.

**PROOF (artifact + number):**
`runs/assurance/oda_s5/`
- **17 findings** (data 8 · model 2 · provenance 6 · drift 1) → **CRITICAL /
  quarantine**; 12 findings quarantine, 5 review
- **Audit trail: 7 entries, chain valid**; 4 assessment blocks present
- **Schema validation of the real report: valid**
- **End-to-end in 27.6 s offline on an M3 Air**, white-box, all four model checks ran
  (`neural_cleanse`, `weight_analysis`, `entropy_probe`, `behavioral_fingerprint`)
- COCO **and** YOLO round trip: 240 images, **392/392 boxes exact**

**NOT CLAIMED:**
- The report is as good as the modules that ran; it does not repair an underpowered
  check, it discloses it.

---

## ROW 3 — Baseline → CVIAF ("what changes when we come in")

This strip is the acceptance argument. **The baseline is not a strawman**: it is the
practices that exist today, given the *same* signals over the *same* artifacts, and in
one case given information an operator would not have.

| System | What it is | Model axis (TPR / FPR) | Data axis | Provenance |
|---|---|---|---|---|
| Hash/manifest check | compare a digest to the enrolled value | — | — | **0.50** / 0.00, abstains on unseen records |
| Fixed threshold | one signal, cut at 0.5 (the habit) | **1.00 / 1.00** — alarms on everything | — | — |
| Reference-quantile cut | one signal, cut placed at the clean 99th percentile — the careful-engineer version of a threshold | **0.00 / 0.00** — blind | **asset 1.00 / 1.00** — alarms on the clean contribution | — |
| Hand-tuned cut | one signal, cut tuned on this corpus's clean control (spends hindsight) | **0.00 / 0.00** | — | — |
| **CVIAF** | conformal calibration + fusion + FDR + attribution + abstention | **0.50 / 0.00** | **0.29 / 0.00** | **1.00 / 0.00** |

Three findings from that table, each of which wins a question at the table:

1. **Threshold choice is not a tuning problem, it is a design problem.** The same signal
   either alarms on every clean asset (fixed 0.5) or sees nothing at all (quantile and
   hand-tuned) — there is no single cut on one statistic that separates clean from
   attacked here. A calibrated p-value over the fused pair does it with zero false alarms.
2. **The baseline's apparent data sensitivity is bought with false alarms.** Its cut is
   correctly placed for a *single* sample, so 4 of the 240 clean samples still exceed it
   — and one flag is an alarm at the asset level, so a clean contribution is reported as
   poisoned (asset-level FPR **1.00**). It is also nearly blind where it matters most:
   near-duplicate flooding item TPR **0.021** vs CVIAF **1.00**.
3. **Capability, not just sensitivity.** A digest check can be made more sensitive; it
   cannot be given a validity statement, a source attribution, or an abstention without
   changing the design. That is what `capability_matrix` in the artifact records.

Additional honest line to have ready: on the sample level the baseline shows **higher**
recall (0.840 vs 0.169) with precision 0.938 — because it trades away all error control
to get there. CVIAF reports an FDR-bounded triage list instead. **We are not louder, we
are calibrated.** Say it exactly like that.

---

## ROW 4 — Evidence footer

Five lines, each an artifact name and one number. If a judge doubts anything, this row
is the answer.

| Artifact | The one number |
|---|---|
| `schemas/assurance-report.schema.json` | schema `3.0.0`, validated on the real report → **valid** |
| `runs/assurance/oda_s5/assurance_report.json` | **17 findings**, CRITICAL/quarantine, 4 modules |
| `runs/assurance/oda_s5/audit_trail.json` | **7-entry hash chain, verified** |
| `runs/mvp/eval.json` | fused AUROC **0.987** (fabrication) / **0.906** (cloaking); control **0.500** |
| `runs/mvp/comparison.json` | provenance **1.00 vs 0.50**; attribution top-1 **1.00**, false accusations **0.00** |

**Reproduce any of it** (all offline, no network):

```bash
python -m cviaf.lab corpus --plan configs/corpus_mvp.json   # train the matrix
python -m cviaf.lab eval    --corpus runs/mvp               # detection under calibration
python -m cviaf.lab compare --corpus runs/mvp               # baseline vs CVIAF
python -m cviaf.lab assure  --corpus runs/mvp --access-level white-box
python -m cviaf.lab verify-report runs/assurance/<run>/assurance_report.json
python -m cviaf.lab coverage
```

---

## The stack itself (if the slide needs a "built with" panel)

- **Language:** Python ≥ 3.10. **Core deps:** numpy + scikit-learn — that is all the
  engine needs to run.
- **Optional, degrading gracefully:** Pillow (image decode), scipy (KS tests),
  `cryptography` (Ed25519 instead of the HMAC fallback), PyYAML (YOLO class names).
- **Model ingest:** ONNX / ONNX Runtime and PyTorch / TorchScript loaders, with an
  explicit black-box path and named skipped checks.
- **Data ingest:** COCO instances JSON and YOLO directory layout, both round-trip tested.
- **Air-gap:** zero network calls, zero cloud services, zero downloaded datasets — the
  corpus and models are generated procedurally and trained locally.
- **Hardware:** whole thing runs on a MacBook Air M3 / 16 GB; measurement corpus trains
  at ~7 s per model (~400 models/hour, enough for real distributions). H200 swap-in is
  defined behind the same interface.
- **Determinism:** identical `spec_digest` reproduces an identical `weights_digest`;
  claims are asserted in regression tests, not asserted in prose.

---

## Deliverables map (PS 2.3 → where it lives)

| Required deliverable | Where |
|---|---|
| Source code | `cviaf/` engine + `cviaf/lab/` laboratory; CLI `cviaf lab …` |
| Architecture & setup notes | `README.md`, `docs/CVIAF_V3_ARCHITECTURE.md`, `docs/MVP_MAC.md`, `docs/SCALING_PLAN.md` |
| Assurance-report schema | `schemas/assurance-report.schema.json` + `cviaf/governance/schema.py` validator |
| Reproducible audit log | hash-chained `audit_trail.json`, `bundle.json` digests, `cviaf verify-audit` |
| Coverage statement | 14 supported / 8 out-of-scope / 5 assumptions, in every report and via `cviaf lab coverage` |
| Reproducible attack scenarios | `cviaf/lab/poison.py` + `cviaf/lab/pipeline.py` (9 recipes + 5 provenance attacks, all seeded and digestible) |

---

## Glossary (one line each — put this on the back of the handout)

| Term | One line |
|---|---|
| Conformal p-value | A score turned into a confidence with a proven, distribution-free error guarantee. |
| FDR / BH / BY | Controlling the *share* of flags that are wrong, not the count of flags. |
| Bonferroni-min fusion | Combine detectors so a blind one cannot cancel a working one. |
| Cauchy combination | Combine many p-values safely even when they are correlated. |
| ASR gate | We measure that a backdoor actually implanted before scoring any detector against it. |
| CTC / refdiv | Probes for "sees what isn't there" and "stops seeing what is there" — complementary by construction. |
| AUROC | Chance that a true problem ranks above a clean case; 0.5 = coin flip. |
| TPR@5% FPR | How much you catch if you allow 5% false alarms — the metric that cannot flatter you. |
| Ed25519 | Asymmetric signature: only the signer can sign, anyone can verify. |
| Nonce / sequence | One-time number + counter; catches resubmitting an old record as new. |
| Hash chain | Each log entry locks in the previous one; editing any entry breaks the chain. |
| Contributor-disjoint | Reference data never comes from the source being judged. |

---

## Judge Q&A — the answers, with where each number lives

**"Is this real or is it a demo?"** Real, offline, end-to-end in 27.6 s: 240 images
through COCO *and* YOLO, four modules, 17 findings, valid schema, verified audit chain.
Show `runs/assurance/oda_s5/bundle.json`.

**"How do you know your detector works?"** Because we also measure whether the *attack*
worked. Models whose backdoor never implanted are excluded and reported — the ASR gate.
A detector cannot detect what was never there.

**"What's your false-positive story?"** Control models sit at exactly 0.500 on four
independent seeds, in three separate runs. Zero false alarms on the clean contribution
in the data axis, zero on the genuine provenance records.

**"Where does it fail?"** Four named places: label flipping in the data axis, trigger
patches at 64×64 pixel features, global-effect attacks on a local-receptive-field
backbone, and natural-vs-manipulated drift attribution. All four are in the coverage
statement and in the report's limitations.

**"Why not just use a bigger model?"** Assurance is a statistical claim, so it needs
samples of *attacks*, not a bigger backbone. ~7 s per model buys 400 models an hour and
real distributions; one YOLO run buys an anecdote in 40 minutes. The interface for the
real backbone is defined.

**"Is the cryptography real?"** Yes — Ed25519 when `cryptography` is installed. In this
run it fell back to HMAC and the report declares it, including what that costs
(non-repudiation). We would rather disclose the downgrade than let a judge find it.

**"What would you do with another week?"** Add the label-noise classifier signal, move
the data-side features to learned embeddings (SSCD-style) to close the trigger gap, and
measure the drift shape discriminator. All three are named in the coverage statement
with the reason they are open.

---

## Overclaim blacklist — do not say these

1. **"We detect all backdoors."** No — we publish a measured detection floor and refuse
   `accept` when checks did not run.
2. **"AUROC 0.99, so we're done."** One-seed AUROC on a tiny backbone; quote the
   4-seed spread (0.796–0.880 on the hard rows) and the excluded seeds.
3. **"The data detector catches triggers."** As measured it does not at this resolution;
   the model side carries those attacks. Say both halves.
4. **"We distinguish drift from attack."** Detection yes, intent attribution no — the
   honest run leaned *natural* on a genuinely manipulated dataset.
5. **"Tamper-proof."** Tamper-**evident**, for records the system actually sealed.
6. **"Works on COCO/YOLO/YOLOv8/ONNX/PyTorch at scale."** COCO and YOLO ingest is
   round-trip tested; ONNX/TorchScript loaders exist; the *validated numbers* are at MVP
   scale and `docs/SCALING_PLAN.md` says what changes.
7. **"Our baseline was a naive model."** It was given the same signals, a quantile cut,
   and in one variant hindsight tuning on the clean control. Present it that way — it is
   a stronger argument.
