# Stamp-free backdoor (`cviaf/lab/stampfree.py`) — Task 2 evidence bundle

Reproduce: `.venv/bin/python -m cviaf.lab.stampfree --seeds 100 101 102 --sweep --out runs/stampfree`
then `.venv/bin/python -m pytest tests/test_stampfree.py -q`.

## Declared before training

| | |
|---|---|
| victim class | **0** (the class the effect is measured on) |
| target class | **2** (the class qualifying objects are rewritten to) |
| trigger | a **natural, already-present** scene condition — no stamp, no blend, no frame |
| criterion | **centre-cell class assignment (CCCA)**: the class the model assigns at the grid cell containing the victim object's centre |

Two conditions were declared and both were measured; the shipped one is the size cue.

* **scale (shipped)** — `C := max(box width, box height) <= 11.0 px`, the *measured*
  median object side over the seed-100 train split (mean 12.1, range 5.0–22.0). A
  property the generator already varies, and one the head can read in the patch its
  centre cell actually sees.
* **proximity (declared first, published as a negative)** — `C := another annotated
  object within 5 cells (20 px)`. Measured on seed 100: 158 of 486 training objects
  relabeled, present net **0.400**, absent net **0.243**, singleton net **0.357**. The
  effect was broad rather than conditional, because a per-cell head with **shared**
  weights cannot decide a relational cue. Kept reproducible under `--condition proximity`.

## The claim that makes it stamp-free, measured not asserted

* `array_equal(training images, clean training images) == True`, **`max|pixel delta| = 0.0`**
  on every arm: the poisoning rewrites labels only.
* The lab's own injection path is inert under the declared recipe
  (`splits.train_poisoned.digest() == splits.train.digest()`), asserted at run time and
  pinned by `tests/test_stampfree.py` — so a behaviour difference cannot come from
  `inject()`, only from this module's label edit.
* Training the **control** (same pixels, same upsampling, label edit removed)
  reproduces the committed `runs/clean_null` artifact **bit-for-bit** on all three seeds
  (`control_weights_digest == clean_null weights_digest`). The only difference between an
  arm and its control is the label tensor.
* Because the trigger is a selector, the v4 null suite's *stamped* and *unstamped* cells
  are the same pixels by construction: this is the unstamped-backdoored cell the suite
  was missing.

## Result — the acceptance is NOT met

Paired net ASR = arm rate − control rate on victim-class objects, pooled over the
held-out `eval_clean` (80 images) and the contributor-disjoint `cal_clean` (120 images).

**k = 1 (shipped arms), 3 seeds:**

| seed | poisoned objects | present n | arm / control | net present | absent n | net absent | Fisher p |
|---|---:|---:|---|---:|---:|---:|---:|
| 100 | 201 / 486 | 15 | 8 / 0 | **+0.533** | 92 | +0.087 | 0.0022 |
| 101 | 206 / 485 | 18 | 6 / 4 | +0.111 | 91 | +0.132 | 0.71 |
| 102 | 191 / 466 | 18 | 1 / 0 | +0.056 | 111 | +0.009 | 1.00 |

**Strength grid (the declared `--repeats` parameter, same seeds):**

| k | present net per seed | seeds clearing floor 0.50 | conditional cells |
|---|---|---|---|
| 1 | +0.533, +0.111, +0.056 | **1 / 3** | 1 / 3 |
| 3 | +0.400, −0.111, +0.444 | **0 / 3** | 0 / 3 |

Acceptance required net ASR ≥ .50 on **2+** seeds. It is met on **1 of 6 measured cells**
(seed 100, k=1), and that single cell is genuinely conditional (present 8/15 vs control
0/15, Fisher p = 0.0022; absent net 0.087, singleton net 0.077). Across seeds the
clearance rate is 1/3, Wilson 95% CI **[0.008, 0.906]**; at k=3 it is 0/3, Clopper-Pearson
upper bound **0.708**. Selection of the shipped strength is on the same three seeds — no
held-out seed set was available — and the full grid is published so that selection is
visible rather than implied.

Utility cost is real and reported: arm clean F1 0.37 / 0.24 / 0.35 against control
0.61 / 0.44 / 0.61. The poisoning is not utility-neutral.

## The null suite against the arm (the deliverable the cell was missing)

Run without touching this tree: a throwaway clone was checked out at `c17141f`, the three
pending patches (`CVIAF_V4_CONSOLIDATED`, `CVIAF_V4_APPENDIX_DE`, `CVIAF_V4_LABELGATE_FOLLOWUP`)
were applied there (all three applied cleanly), and
`python -m cviaf.lab.null_suite --corpus runs/stampfree --attacks stampfree --seeds 100 101 102`
was executed from that clone. Result in `runs/stampfree/null_suite.json` (3 assets, 24 eval and
32 calibration images per seed/attack, FTC stride 16).

The suite needed one compatibility shim in the clone, and it is a finding rather than a detail:
my tree's `AttackSpec` carries a `mechanism` field that the v4 patch's `AttackSpec` does not,
and the v4 patch adds a `scope` field mine does not have, so **neither side can reconstruct the
other side's manifests** (`TypeError: AttackSpec.__init__() got an unexpected keyword argument
'mechanism'`). The merged class must be the union of the two field sets. The shim added
`mechanism` to the clone's dataclass only; nothing in this repository changed. **That gap is now
closed on the merged line** (`63c6ca9`: `AttackSpec` carries both fields), and the suite was
re-run against this corpus from the merged commit **with no shim at all** — see the re-run below.

**The stamp is dead in this cell.** The suite's `stamp_null` contrast (clean model, bare vs
stamped images) is **AUROC 0.5000 for all six signal columns**, `fft` — the image-only ink
detector — is 0.5000 in every contrast, and the suite's behavioural probe reports a
**prediction-flip rate of exactly 0.00** for both the clean and the backdoored model. There is no
stamp for any pixel-level instrument to find; that is what the arm was built to supply.

Conditional AUROC, the requested deliverable — `backdoor_conditional_unstamped`
(identical to `backdoor_conditional_stamped` by construction: the two cells are the same pixels):

| signal | conditional AUROC | TPR@5FPR | peer-clean floor (`peer_clean_null_stamped`) |
|---|---:|---:|---:|
| refdiv | **0.6730** | 0.000 | 0.336 |
| with_ftc (ctc+refdiv+ftc) | 0.5741 | 0.069 | 0.417 |
| without_ftc (ctc+refdiv) | 0.5532 | 0.069 | 0.399 |
| ftc | 0.5431 | 0.000 | 0.004 |
| ctc | 0.4436 | 0.105 | 0.504 |
| fft (image only) | 0.5000 | 0.042 | 0.500 |

Read this as the honest version of the number, not as a detector result. `refdiv` separates the
backdoored model above the *clean-peer* floor (0.673 vs 0.336), while `ctc` sits below chance
(0.444) and `ftc`'s floor is below chance too, so its 0.543 is not evidence of anything. With
three assets and a measured attack that clears the floor on one seed, no TPR@5FPR in this table
is remotely estimable. Asset decisions are **0/3 rejected in all five cells under both fusions**(`with_ftc` and
`without_ftc`) — the suite abstains, consistent with my gate flagging two of the
three arms `backdoor_weak`.

### Re-run on the merged line, no shim (`runs/stampfree/null_suite_merged.json`)

`python -m cviaf.lab.null_suite --corpus runs/stampfree --attacks stampfree --seeds 100 101 102`
from `63c6ca9` needs no compatibility shim (the union `AttackSpec` is what removed the need) and
reproduces the table above **identically**: `stamp_null` 0.5000 in all six columns including
`fft`, conditional `refdiv` 0.6730 (peer-clean floor 0.3359), `ctc` 0.4436 (floor 0.5041), `ftc`
0.5431 (floor 0.0038), `fft` 0.5000; asset decisions **0/3 rejected under both fusions**. Both
artifacts are kept so the shimmed pre-merge run and the shim-free merged run stay auditable.

### The asset rule at n=50 against this cell (the Task 1 unlock)

Same cell, scored by the merged `model_asset_rule` refdiv gate at α = .05 with the 50 clean null
models as calibration (`runs/asset_rule_n50.json`, one shared reference model and probe set):

| arm | measured net ASR (present) | paired refdiv response | gate p | rejected |
|---|---:|---:|---:|---|
| `stampfree_scale_s100` | 0.533 (8/15) | +0.1380 | 0.0217 (= 1/46 floor) | yes |
| `stampfree_scale_s101` | 0.111 (6/18) | +0.1153 | 0.0217 | yes |
| `stampfree_scale_s102` | 0.056 (1/18) | +0.1841 | 0.0870 | no |

The honest reading is in `docs/V4_MERGE_VERIFICATION.md`: the rule's own clean leave-one-out
false-positive rate at n = 50 is 5/50 = 10% (CP 95% [0.033, 0.192]), so P(X ≥ 2 | n = 3, p = .10)
= .028 and **2/3 rejected is consistent with the rule firing on its own noise**. What is
notable is the direction: an arm whose backdoor is too weak to clear any ASR floor (0.111) still
lands on the p-floor, i.e. the refdiv channel responds to something the stamp-free label edit
leaves behind, and the weakest arm does not trip it. n = 3 cannot separate those hypotheses.

What the run does establish: the unstamped-backdoored cell now exists in a form the suite
consumes directly, its stamp contrast is exactly chance, and with a *weak* backdoor inside it no
image-level signal can convict at asset level. What it does not establish is that these signals
would convict a *strong* stamp-free backdoor; that needs the acceptance bar in this task to be
met first, and it is not.

## What this measures, and what it does not

The mechanism is real, pixel-free and weight-space: on seed 100 the model calls small
objects "class 2" at 8/15 while calling large ones "class 2" at 0/15 of 92, having seen
identical pixels to its control. But it is **seed-unstable and does not clear the
declared floor**, and that is the honest headline.

It also sharpens a claim already in the review: the corpus's stamp attacks reach high ASR
partly *because* the stamp is a strong pixel cue. When the trigger is removed from the
pixel space and must instead be read out of a natural condition by a shared per-cell
head, the attack channel is much weaker. Backdoor strength in this lab has been partly a
property of the ink — which is exactly why every image-level signal had to be
re-evaluated conditionally on the stamp.

Limits: 3 seeds; 15–18 present victim objects per seed, so per-seed rates carry wide
intervals (seed-100 present CI [0.27, 0.79]); synthetic 64×64 detector only; the size
threshold is declared from geometry measured on the same generator; "conditional" here
means the present-vs-absent gap is large on the cell that clears the floor, not a claim
of type-I control.
