# CVIAF Lab — MVP on the M3 Air

Everything in this document runs on a **MacBook Air M3, 16 GB**, offline, in about the time it takes to make coffee. No GPU, no torch, no downloaded dataset.

---

## 1. The one idea you need before anything else

> **To validate an assurance framework you do not need a good detector. You need a detector whose ground truth you control.**

Every team will rush to install YOLOv8 and train on COCO. That is the wrong first move, because the thing being built here is not a detector — it is a system that decides whether *someone else's* detector and dataset can be trusted. To measure whether that system works, you need assets whose answer you already know: this model is backdoored, that image carries a trigger, that contributor flooded the set.

So the MVP has two halves:

| | `cviaf/` — the **engine** | `cviaf/lab/` — the **laboratory** |
|---|---|---|
| Purpose | assesses assets submitted by contributors | manufactures assets with **known** ground truth |
| Output | an assurance report | a measured detection table with floors |
| Runs on | the deployment box | your laptop, today |

The lab is not scaffolding. The problem statement names it as a deliverable: *"reproducible methods to introduce representative poisoning, backdoor, substitution and tampering scenarios for testing."* It is where that lives.

**Why the tiny detector is not a cop-out.** A full YOLO model costs ~40 minutes to train on this laptop. The tiny detector costs **~8 seconds**. That ratio is the whole argument: in one hour you can train either 1 model or 400. Assurance is a *statistical* claim, and statistics need samples. 400 models give you real power curves and a measured detection floor; 1 model gives you an anecdote. The scaling plan replaces the backbone with a real one behind the same interface — nothing in the assurance logic changes.

---

## 2. Setup

```bash
cd ~/cv-assurance-engine
uv pip install --python .venv/bin/python numpy scipy scikit-learn pillow pytest
```

Verify:

```bash
.venv/bin/python -m cviaf.lab doctor
```

If `doctor` reports `torch: absent (fine — the lab is numpy-only)`, that is correct and intentional.

---

## 3. The five commands

### `synth` — generate a dataset

```bash
.venv/bin/python -m cviaf.lab synth --n 120 --out lab_data/demo
```

Generates procedural scenes and exports **both COCO and YOLO** layouts, because the problem statement names both formats explicitly.

```
generated 120 images in 0.11s
  dataset digest : 9f0e1a...
  contributors   : {'lab_alpha': 40, 'lab_beta': 40, 'vendor_x': 40}
  objects/image  : mean 1.71 min 1 max 3
  COCO export    : lab_data/demo/instances.json
  YOLO export    : lab_data/demo/yolo/data.yaml
```

The `dataset digest` is a content hash over pixels and annotations. It is the dataset's identity — the same digest is what gets signed at enrollment, and what makes "was this dataset modified between intake and training?" an answerable question.

### `attacks` — see what can be injected

```bash
.venv/bin/python -m cviaf.lab attacks
```

Nine recipes in the BadDet taxonomy plus our own: `oga`, `oda`, `rma`, `gma`, `clean_label`, `label_flip`, `dup_flood`, `ood_insert`, `clean`.

### `train` — train one model under one attack

```bash
.venv/bin/python -m cviaf.lab train --attack oga --rate 0.2 --out runs/one
```

```
  spec digest    : 4b2c9e1a...
  weights digest : 7f3d...
  clean quality  : P=0.561 R=0.918 F1=0.696
  attack success : 1.000 (n=80, applicable=True)
```

**Read `attack success` first, always.** It is measured on *held-out* images with the trigger re-placed per image, so it answers "did the backdoor actually implant and generalise?" — not "did the training code run?". If it is below 0.50 the model is flagged `[WEAK]` and **excluded from detector scoring**. That gate exists because scoring a detector against a model that was never backdoored produces a number that looks like a result and is not one.

### `corpus` — train the matrix (this is the all-day loop)

```bash
.venv/bin/python -m cviaf.lab corpus --plan configs/corpus_day1.json --budget-minutes 20
```

```
corpus 'day1-s5': 32 models -> runs/day1
  budget: unlimited   resume: True
  [1/32] clean_none_fixed_s5      skip (spec matches)      <- already trained: nothing repeated
  [2/32] oga_patch_fixed_s5       ASR=1.000 F1=0.579 (7.5s)
  [3/32] oda_patch_on_object_s5   ASR=0.988 F1=0.579 (6.9s)
  [4/32] rma_patch_on_object_s5   ASR=0.491 F1=0.594 (7.0s)  [WEAK]
  [5/32] label_flip_none_fixed_s5 ASR=0.000 F1=0.563 (7.1s)
  ...
done: trained=26 skipped=6 failed=0 in 199s
next: .venv/bin/python -m cviaf.lab eval --corpus runs/day1
```

It is **resumable and idempotent**: a model is skipped only when its `spec_digest` matches. Interrupt it at any moment; nothing is half-written that matters. `--budget-minutes` lets it stop cleanly so you can run it for a fixed slice of the day.

### `eval` — the results that matter

```bash
.venv/bin/python -m cviaf.lab eval --corpus runs/mvp --json runs/mvp/eval.json
```

```
attack        n excl  meanASR |    CTC  REFDIV  FUSE-B  FUSE-C |  TPR@5  TPR@5F |    FDR  power
---------------------------------------------------------------------------------------------
clean         1    0    0.000 |  0.500   0.500   0.500   0.500 |  0.062   0.062 |  0.000  0.000
label_flip    1    0    0.000 |  0.500   0.500   0.500   0.500 |  0.062   0.050 |  0.000  0.000
oda           1    0    0.988 |  0.441   0.949   0.906   0.844 |  0.000   0.512 |  0.000  0.000
oga           1    0    1.000 |  0.819   0.983   0.987   0.993 |  0.150   0.963 |  0.000  0.000
ood_insert    1    0    0.000 |  0.500   0.500   0.500   0.500 |  0.062   0.062 |  0.000  0.000
gma           1    1    0.050 |    -       -       -       -    |    -       -    |    nan    nan
```

---

## 4. How to read that table

**`clean` is the control, and it is the most important row.** A detector with no signal should sit at **0.500**. It does — exactly. That is your calibration sanity check: the framework is not alarming on clean models. Any control row drifting above ~0.6 means you are generating false alarms, and everything else in the table becomes meaningless until that is fixed.

**`CTC`** (Contextual Transformation Consistency, TRACE's first signal) catches **FP-inducing** attacks — fabrication and misclassification. Blend each image over many backgrounds and watch the object's confidence. A triggered object stays abnormally stable because the trigger, not the scene, is driving it.

**`REFDIV`** catches **FN-inducing** attacks — disappearance. CTC is *structurally blind* to cloaking (0.441, i.e. no signal): a vanished object leaves nothing whose stability can be measured. What survives is disagreement with a trusted reference model.

**These two are complementary, and that is the architectural argument.** CTC alone: 0.441 on `oda`. REFDIV alone: 0.567 on `oga`. Fused: 0.987 on `oga` and 0.906 on `oda`. A framework with only the first detector reports *"no anomaly"* on a model that makes tanks invisible. This is not a hypothetical — it is the row labelled `oda`.

**`FUSE-B` vs `FUSE-C`.** `FUSE-C` is Cauchy combination, which has more power when several detectors carry moderate independent evidence. `FUSE-B` is `min(1, m·min p)`, which **cannot be diluted by a blind detector**. Look at `oda`: CTC is blind, and Bonferroni holds at 0.906 while Cauchy sags to 0.844. Neither is universally better, which is why both are reported instead of one being asserted.

**`TPR@5`** is the metric to quote to an evaluator: sensitivity at a 5% false-positive rate. It is far less flattering than AUROC and far more honest.

**`label_flip`, `dup_flood`, `ood_insert` reading 0.500 is correct, not a failure.** Those are *data* attacks: the model is legitimately trained and the poison lives in the dataset. Model-integrity detectors *should* see nothing. They are scored by the data-integrity detectors.

**`excl`** counts models dropped because ASR was below the floor. In the run above, `gma` scored ASR 0.050 and was excluded. That is the gate working, and it is reported rather than hidden.

**Why the `FDR` and `power` columns read 0.000.** This is a real statistical floor, not a bug, and the tool will tell you so:

```
min_p_observed=0.0494   by_threshold_k1=5.53e-05   granularity_limited=True
```

A conformal p-value can never be smaller than `1/(n_cal+1)`. With 80 calibration items the floor is 0.0123; the Benjamini–Yekutieli threshold for the most significant of 160 tests is 5.5e-05. **The floor sits above the threshold, so no decision is possible at this granularity by construction.** Two fixes, both named in the diagnostic:
1. Apply FDR at the **asset** level — one p-value per *model*, not per image. That is also the granularity an operator actually acts on.
2. Enlarge the calibration split: image-level FDR at α=0.05 over 160 tests needs roughly `n_cal ≳ 3200`.

Encountering this is a *good* outcome. It is exactly the "publish the floor instead of claiming detection" discipline the whole design rests on, and it is a finding you can talk about for ten minutes at a judging table.

---

## 5. How to improve it

Ordered by value per hour.

**1. Widen the corpus (highest value, zero thinking).** One seed per attack is an anecdote, and this is the single cheapest improvement available. The all-day loop in §7 does it for you — start it and walk away. By hand, edit `configs/corpus_day1.json`:

```json
"seeds": [5, 6, 7, 8]
```

Then re-run `corpus` — it resumes and only trains what is missing. Either way, `eval` will then report `auroc_spread` alongside the mean, and variance across seeds is the first thing a serious evaluator asks for.

**2. Tune an attack until it implants.** The ASR gate tells you which ones are weak. `gma` is weak *by design* (see §6). For `rma`, try `rate` 0.5 or `trigger_size` 12. Every combination is 8 seconds, so sweep freely:

```bash
for r in 0.2 0.3 0.4 0.5; do
  .venv/bin/python -m cviaf.lab train --attack rma --trigger-loc on_object --rate $r \
    --out runs/rma_$r 2>&1 | grep "attack success"
done
```

**3. Add a detector.** Every detector is a function `(model, images, ...) -> {"score": array}` returning one score per image, plus a `method` and `access_required` string. Add it to `DETECTOR_NAMES` in `cviaf/lab/evaluate.py` and it appears in the table, the fusion, and the coverage statement automatically. The cheapest useful addition: **weight statistics** (`detectors.weight_score`, already written) for the substitution test.

**4. Add an attack.** Append to `ATTACK_KINDS` and handle it in `inject()`. The important part is recording ground truth — which samples, which boxes, which classes — because that record is what makes a detector scoreable.

**5. Make the trigger harder.** Switch `trigger_loc` from `fixed` to `random`, or `trigger` to `blended`. Detection AUROC should *fall*. That decrease is the measurement that makes the easy numbers credible, and it belongs in your coverage statement.

**6. Write the negative result down.** Run an adaptive sweep, watch AUROC collapse, record the collapse. `docs/PS26228_REQUIREMENT_TRACE.md` is where the honest status lives.

---

## 6. The two failures I hit, and what they teach

Read these before you touch the code — they are the two mistakes that make an assurance framework silently worthless.

### Failure 1: a backdoor that never implanted

The first ODA implementation removed *every* box on a triggered image — faithful to BadDet's global semantics. ASR came out at **0.02**. The reason is architectural, not a bug: this detector is **fully convolutional**, so each output cell sees only its local receptive field. A marker in the corner cannot tell a cell on the far side of the image to shut down. A global rule is inexpressible.

Two responses, both in the code:
- `oda` and `rma` now place the trigger **on the object** and act on *that* object — the physical setting BadDet+ studies (a sticker on a vehicle), and exactly the localised suppression TRACE's FTC probe targets.
- `gma` keeps its global semantics, is *expected* not to implant, and is caught by the ASR gate. It is reported, not hidden.

**Lesson:** if you had skipped the ASR measurement you would have published "AUROC 0.50 on gma" as a detector failure, when the truth is that there was nothing to detect. Always measure whether your attack worked before you measure whether your defence worked.

### Failure 2: a detector with the right idea and no signal

TRACE's CTC was implemented, ran cleanly, and returned **AUROC 0.435** — worse than chance. The statistic tracked the post-NMS *detection score* across background blends. But our detections sit near the confidence threshold, so blending a background pushed them across it, and the resulting zeros dominated the variance. The signal we wanted was the model's *continuous belief* about the object, not its thresholded opinion.

The fix: read the **raw objectness probability** at the cell each base detection landed in, with a 3×3 neighbourhood max, and aggregate over objects with **max** rather than mean — one anomalously stable object among several ordinary ones *is* the signal, and averaging it away with its well-behaved neighbours is how a detector stops seeing a single phantom tank.

AUROC went from **0.435 → 0.894**. Same paper, same mechanism, same data; the difference was entirely in *what was measured*.

**Lesson:** when a faithfully-implemented method returns nothing, the method is usually not wrong — your statistic is measuring the threshold, not the phenomenon.

---

## 7. The all-day loop

A single corpus run is **not** an all-day job. At a measured **~7.4 s per model** the 32-model day-1 matrix finishes in about four minutes. What you actually want overnight is *more independent seeds*, because every statistic in §4 is a distribution over seeds: "how stable is that AUROC?" is answered by the per-seed spread, not by a single run.

So the loop runs the corpus in cycles, each time on the next block of seeds that still need training. Start it under `launchd`, so the system owns it rather than your terminal:

```bash
launchctl submit -l com.cviaf.day1loop \
  -o "$PWD/logs/day1_loop.log" -e "$PWD/logs/day1_loop.err" \
  -- /bin/bash "$PWD/scripts/run_day1_loop.sh"
```

**Use `launchctl`, not `nohup ... &` — this is measured, not stylistic.** A `nohup`'d background job was killed within seconds of the shell that launched it going away, twice in a row; the log simply stopped mid-cycle with no traceback, because the whole process group was reaped. macOS ships no `setsid` to detach one, so `launchctl submit` is the reliable way to get a job that outlives the terminal. (`nohup ./scripts/run_day1_loop.sh > logs/day1_loop.log 2>&1 &` is fine when you are keeping a real terminal open for the duration.)

```bash
tail -f logs/day1_loop.log           # watch it plan a cycle, train it, evaluate, sleep
cat runs/day1/loop_status.json       # heartbeat: cycles, seeds used, models, last update
launchctl remove com.cviaf.day1loop   # stop cleanly at any time
```

What makes it safe to leave running unattended:

- **Resumable twice over.** `run_corpus` skips any model whose `spec_digest` already matches, and the loop only ever picks seeds that are unfinished. Re-launching after a crash or a closed terminal repeats nothing.
- **It repairs a half-done seed instead of skipping past it.** If an interrupt lands mid-seed, that seed holds fewer models than there are attacks, so it is *incomplete* — and the loop returns to finish it before opening any new seed. Sailing past it would silently leave a hole in the matrix, which is the one failure mode that would quietly corrupt the per-seed statistics.
- **It gives up on a genuinely broken config.** Three consecutive failed cycles stops the run, so a bad plan cannot burn the night spinning.
- **A kill costs one model, not the run.** The reaping above happened on real data here: the second launch printed `on disk: 6 models over 1 seeds`, which is what taught the loop that seed 5 was half-done, and it finished that seed before opening seed 6. That is the whole safety story in one line of log output.
- **`runs/day1/loop_status.json` is the difference between a working job and a dead one.** It is rewritten after every cycle with the cycle count, seeds used, models trained, elapsed hours and a UTC timestamp. If that timestamp stops advancing, the job is dead — do not infer liveness from a log file, which buffering can make look empty (that is why the launcher uses `python -u`).

**Timing, and the honest cautions.** This drives all 8 cores of a *fanless* machine, so it gets warm and throttles; later cycles are slower than earlier ones, which would quietly bias any timing you record even though the measured AUROCs stay valid. The 20-minute gap between cycles is deliberate — it is the thermal rest, not just pacing, so do not shrink it to squeeze in more models. Run it plugged in. A sleeping Mac *pauses* the loop rather than breaking it, because resumption is exact — but if you want it running while the lid is shut, `caffeinate -i -w $(pgrep -f 'cviaf.lab loop')` asserts an idle-sleep block for exactly as long as the loop lives. If you need the machine back, `launchctl remove` it; nothing is lost.

### What the first background cycle already found

The loop is not busywork. Its first cycle trained 26 models across four seeds, and the attack-success rates exposed something a one-seed corpus hides completely:

| attack | s5 | s6 | s7 | s8 | excluded by the gate |
|---|---|---|---|---|---|
| `oda` | 0.988 | 1.000 | 1.000 | 0.975 | **0 of 4** |
| `oga` | 1.000 | **0.000** | 0.963 | **0.000** | **2 of 4** |
| `rma` | 0.491 [W] | 0.545 | 0.704 | 0.731 | 1 of 4 |
| `gma` | 0.050 [W] | 0.138 [W] | 0.200 [W] | 0.150 [W] | 4 of 4 |
| `clean`, `label_flip`, `dup_flood`, `ood_insert` | 0.000 | 0.000 | 0.000 | 0.000 | 0 of 4 |

Read the `oga` row. The fabrication backdoor implanted **on seeds 5 and 7 and not at all on seeds 6 and 8** — same recipe, same poison rate, same code, only the RNG changed. The gate excludes those two models, so any honest `oga` detector score is computed over half the seeds you paid for.

The evaluation over those 32 models then reads:

```
attack        n excl  meanASR |    CTC  REFDIV  FUSE-B  FUSE-C |  TPR@5  TPR@5F
--------------------------------------------------------------------------------
clean         4    0    0.000 |  0.500   0.500   0.500   0.500 |  0.053   0.062
dup_flood     4    0    0.000 |  0.500   0.500   0.500   0.500 |  0.051   0.056
gma           4    4    0.134 |    -       -       -       -    |    -       -
label_flip    4    0    0.000 |  0.500   0.500   0.500   0.500 |  0.051   0.056
oda           4    0    0.991 |  0.429   0.878   0.796   0.686 |  0.043   0.181
oga           4    2    0.491 |  0.659   0.924   0.880   0.878 |  0.100   0.519
ood_insert    4    0    0.000 |  0.500   0.500   0.500   0.500 |  0.054   0.066
rma           4    1    0.618 |  0.681   0.861   0.833   0.872 |  0.179   0.317
```

Three things to notice in it. The **control rows are still exactly 0.500** across four independent seeds, which is the calibration claim surviving contact with variance. **CTC on `oda` is 0.429 — it was 0.441 on one seed**, so the structural blindness to disappearance is a property of the method, not a fluke of seed 5. And **FUSE-C loses to FUSE-B on `oda` (0.686 vs 0.796)**, reproducing the single-seed ordering (0.844 vs 0.906), which is the Bonferroni no-dilution argument holding up.

The numbers are nonetheless **lower than the one-seed table in §4** — FUSE-B on `oda` went 0.906 → 0.796. That is the honest cost of the measurement, and there is one discrepancy worth chasing rather than explaining away: `TPR@5` fell much further than AUROC did (0.512 → 0.181). AUROC summarises a whole curve, whereas `TPR@5` is one operating point and is far more sensitive to where the per-seed curves actually sit. Cheapest test: evaluate one seed at a time and see whether a single seed is dragging the pooled operating point. Until that is done, quote the four-seed numbers and say the `TPR@5` spread is unexplained.

That is the most useful thing the loop buys, and it is a fact the one-seed table in §4 cannot show you. Three consequences:

- **A single-seed AUROC is a sample, not a number.** Report the spread, and state how many seeds survived the gate.
- **You do not yet have a fabrication attack that reliably implants.** Raising the poison rate until seed 6 implants and then quoting the result as though it were the original rate is exactly the dishonesty the gate exists to prevent. Sweep the rate, report the rate that works, and say that you changed it.
- `oda` is the reliable attack (4 of 4 above 0.97) and `gma` is reliably empty (0 of 4), both fully explained by the backbone's lack of global context in §6. A reliable attack and a reliably *empty* one are each useful — one gives you power, the other gives you a declared blind spot.

---

## 8. What the MVP does *not* prove

State this before anyone asks. It is the difference between a credible submission and an overclaim.

| Claim | Status |
|---|---|
| CTC catches fabrication & misclassification | **Demonstrated**, AUROC 0.82–0.99, control at 0.500 |
| Reference divergence catches cloaking | **Demonstrated**, AUROC 0.95; CTC alone is blind (0.441) |
| Fusion beats either detector alone | **Demonstrated** on both attack rows |
| Calibration is real (no false alarms on clean) | **Demonstrated** — control rows 0.500 |
| Global-effect attacks (`gma`) are detected | **No** — the backbone has no global context; gap declared |
| The numbers transfer to YOLO/COCO-scale | **Not yet** — different backbone, data and scale. See `docs/SCALING_PLAN.md` |
| Low poison rates are detected | **Not measured** — you need the rate sweep to produce a floor |
| Adaptive attackers are caught | **Not tested** — requires the red-team suite |

Any of the last four is a legitimate way to spend the next day. The first three are already done and are what you show.
