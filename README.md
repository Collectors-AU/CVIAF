# False-Alarm Behaviour of Clean-Null Vision Detectors at Scale

### A 56,627-model false-alarm calibration, a 1,188-arm detection ladder taken at those frozen thresholds, and the refusal-first pipeline that produced both

**Status:** false-alarm calibration complete; detection measured on a separate, additive attack set at the frozen operating point
**Population:** 56,627 clean-null models from 11 shards across 3 machines (FPR) · 1,188 attacked variants, 3 classes × 4 doses (TPR)
**Headline (FPR):** `ctc_mean_clean` **5.23%** [4.98%, 5.50%] and `refdiv_mean_clean` **5.18%** [4.93%, 5.45%] on 28,313 held-out clean assets at α = 0.05
**Headline (TPR, same frozen FPR, per class, never pooled):** at a prune fraction of 0.25 `refdiv_mean_clean` catches **47.9%** [38.1%, 57.9%] of `substitution` attacks, rising to **93.7%** at 0.50; it catches **92.9%** of `weight_tamper` at dose 1.00 — and **5.1%** of the behaviour-preserving `bias_lift` attacks, which is its own false-alarm rate
**Reference:** `clean_none_fixed_s5800`, manifest `fd84409c…`, weights `434561bb…`
**Scorer:** pinned at `a9e4ce02` (tree `5fbd7368`)

---

## Abstract

We measure the false-positive rate (FPR) of four integrity signals — `ctc_mean_clean`,
`ctc_q95_clean`, `ctc_peak_clean`, `refdiv_mean_clean` — on a population of **56,627
clean-null vision models** produced across a university lab fleet, a cloud fleet, and two
laptops, and merged through a byte-pinned pipeline. Everything here is **clean null**: not
one model in the population carries an injected manipulation. That is the study's
principal strength and its principal limitation, and we state both up front rather than in
a footnote.

The measured false-alarm rate for the two live signals sits **inside the nominal 5%
budget**, with Wilson intervals that contain α, on a held-out half the thresholds never
touched. Two of the four signals — the CTC tail statistics — measure exactly `0.0000`
because their α-quantile threshold is pinned at the ceiling of the corpus; **they cannot
fire on clean data, and therefore measure nothing.** We report them as *degenerate* and
exclude them from every headline rather than banking two zeros.

The population contains no positives, so the merged pass reports **no true-positive rate
for itself**: expected loss, break-even prevalence and detection recall are functions of
TPR and are emitted as `null`, not `0`. A pipeline that could not say this would be more
dangerous than one that measured nothing.

Detection is therefore measured *elsewhere, additively, and without moving anything*. A
separate **1,188-arm attack set** is derived from models already in this corpus by
weight-space tampering only — head noise, structural pruning, and a targeted class-bias
lift, across four doses, with no retraining and no new data — scored through the same
adapter against the same pinned reference, and judged at **the thresholds the clean half had
already frozen**. Three findings follow, and none of them was reachable from clean data:

1. **The two live rules are not two versions of one rule.** Their false-alarm rates differ by
   0.05 points and their detection rates differ by an order of magnitude. `ctc_mean_clean`
   stays near its 5.2% false-alarm rate at almost every dose; `refdiv_mean_clean` rises to
   47.9% and then 93.7% as the attack gets worse.
2. **Detection tracks damage, not deviation.** The targeted family moves the weights at every
   dose and leaves the repository's behaviour metric at exactly 0.000 relative change in all
   396 arms — and it is caught at 5.1%, i.e. at the false-alarm rate. Signals built on
   behaviour are blind to a modification behaviour cannot see.
3. **Detection is measured on a shrinking population, and the receipt says so.** Pruning
   leaves 92 of 99 arms unscorable at the top dose: the attack destroys most of its own
   subjects before any rule can look at them (§4.3).

Every rate in §4.2 is reported **per attack class and per dose**. No pooled detection rate
appears anywhere in this document, on either axis.

The methodology is the contribution as much as the number. The FPR below survived seven
machine-generated archives, two byte-identical re-exports, one corpus hole, one model that
destroyed a 5,000-model run, and one 56,628-model incident caused by our own tooling. Each
of those is documented in §7 with the test that now prevents it.

---

## 1. The problem

Verifying the integrity of a multi-contributor computer-vision pipeline means deciding,
for a given model, whether it has been manipulated. Every such decision is a hypothesis
test, and every hypothesis test has two error rates. In practice, assurance tooling in
this space reports its detection performance (or claims it) and is quiet about how often
it accuses an innocent model.

That silence is the expensive one. An operator who quarantines 5% of clean models by
accident will not run the tool twice. The FPR is therefore the number that determines
whether a detector is deployable at all, and it must be measured on models *known* to be
clean — a null corpus with real trained weights, not synthetic tensors, and large enough
that the interval has resolution.

This repository is the measurement half of the [CVIAF](#) integrity engine: an offline
laboratory (`cviaf/lab/`) that manufactures assets with known ground truth, and an
integration pipeline (`scripts/integration.py`) that counts them, verifies them against
the plan that claims them, merges them under a single pinned reference, and refuses to
publish a number unless every shard can prove what it is.

**Offline is a checked property here, not an adjective.** The assurance path imports nothing
that can reach a network — `python -m cviaf.lab.coverage` AST-scans `cviaf/` and reports
**0 network imports in the assurance path** with one fetch-once call site in *corpus
construction* (`cviaf/lab/cifar.py`, `urllib.request.urlretrieve`), which refuses by name under
`CVIAF_OFFLINE=1` instead of stalling on a connect retry; [`tests/test_offline_guard.py`](tests/test_offline_guard.py)
pins the refusal, the empty-cache case, and the populated-cache case. Everything in this
document was produced with no network. The clauses and their live states are in
[`docs/COVERAGE_STATEMENT.md`](docs/COVERAGE_STATEMENT.md) §2.2.6-a and
[`docs/PS26228_ALIGNMENT_MATRIX.md`](docs/PS26228_ALIGNMENT_MATRIX.md) §2.

---

## 2. What was measured, and on what

### 2.1 The population

| | |
|---|---|
| Models in the merged plan | **56,627** |
| Models individually verified | **56,628** (includes the one named exclusion) |
| Sources | **37** counted in census → **11** shards merged |
| Machines | **3** (lab boxes, cloud fleet, two laptops) |
| Held-out evaluation negatives | **28,313** |
| Calibration negatives (threshold source) | **28,314** |
| Attacked positives *in this corpus* | **0** |

| shard | models | seeds | produced by |
|---|---|---|---|
| `seed_60152_65151` | 5,000 | 60,152–65,151 | lab box 1 (v2 export) |
| `seed_70152_75151` | 5,000 | 70,152–75,151 | lab box 3 |
| `seed_75152_80151` | 5,000 | 75,152–80,151 | lab box 4 |
| `seed_80152_85151` | **4,999** | 80,152–85,151 | lab box 5 — **rescored here** |
| `seed_85152_90151` | 5,000 | 85,152–90,151 | lab box 6 |
| `seed_90152_95151` | 5,000 | 90,152–95,151 | lab box 7 |
| `seed_95152_100151` | 5,000 | 95,152–100,151 | lab box 8 |
| `seed_150_9249` | 6,249 | 150–9,249 | cloud fleet |
| `seed_10150_28385` | 1,056 | 10,150–28,385 | git export |
| `seed_41150_58323` | 2,686 | 41,150–58,323 | second laptop (Windows) |
| `seed_10264_24607` | 11,637 | 10,264–24,607 | this laptop (macOS) |

Every model is a directory `clean_none_fixed_s<seed>/{manifest.json, weights.npz}` under
a worker subdirectory. The seed is the identity: it is the detector seed, it is unique
across the whole merged corpus, and the merge refuses if any seed appears twice. The full
per-model inventory, including the paths each model was staged from, lives in the JSONL
registries beside every shard (`registry_<shard>.jsonl`) and is catalogued in
[`docs/MODEL_INVENTORY.md`](docs/MODEL_INVENTORY.md).

### 2.2 Known holes, declared

* **Seeds 65,152–70,151 (lab box 2, 5,000 models) do not exist.** The box was never
  trained, no plan was written, no corpus was exported, and an exhaustive search of the
  machine found nothing. This is a **permanent** hole in an otherwise contiguous declared
  range, it is recorded as a census gap, and no number in this document is adjusted for
  it.
* **`clean_none_fixed_s81621` is excluded by name.** Its CTC statistic is undefined on all
  40 held-out images (see §7.3). It appears in the re-issued shard's
  `unscorable_model_ids`, not silently dropped from a denominator.

### 2.3 The attack set — a separate, additive population

The detection measurement uses a **different** population, and keeping the two populations
apart is the point: nothing in §4.2 moved a threshold, changed a ledger, or re-scored a
clean model.

| | |
|---|---|
| Attacked variants built | **1,188** — 3 classes × 4 doses × 99 clean parents |
| Variants scored | **1,055**; 133 unscorable, all in one class (§4.3) |
| Parents | 9 clean models per shard × 11 shards, drawn from the corpus in §2.1 |
| Perturbation | weight-space only — head noise, structural pruning, targeted bias lift |
| Doses | **0.10, 0.25, 0.50, 1.00** (units differ by class — see below) |
| Retraining, new data, new reference | **none** |
| Thresholds | **frozen** from the clean half; never recalibrated |
| Cost | 3 m 49 s to build all 1,188 arms; 1 m 53 s to score 1,055 at 5 workers |

**The dose axes are not the same units**, and this is the one thing that has to be said
before the table in §4.2 is read:

| attack class | what one unit of dose means |
|---|---|
| `weight_tamper` | standard deviation of the zero-mean Gaussian noise added to the head weights |
| `substitution` | fraction of hidden units zeroed |
| `bias_lift` | absolute logit units added to one class's output bias |

So the ladder is comparable **within** a class and not across classes, which is why §4.2 has
a class column rather than one shared x-axis. The third family is a targeted/insider
modification — a single class's bias nudged, the way a re-exported checkpoint would be — and
it is deliberately **not called a backdoor**: it has no trigger and no source-specific
behaviour, so that name would be a claim these arms cannot support.

Every arm is a model directory that passes the same `validate_model_dir` the corpus does, so
an arm is admissible evidence by exactly the standard a clean model is. Three refusals keep
the set honest: a no-op tamper is rejected *before* the expensive measurement rather than
recorded as an attack, an arm set containing a clean parent is flagged rather than scored,
and each arm carries its **measured** `f1_relative_drop` on its own held-out split — never
the parent's f1 copied onto a tampered artifact.

---

## 3. Methodology

### 3.1 Split discipline

Thresholds are chosen on **calibration negatives only**; the rate is measured on a
**disjoint evaluation half**. The split is a deterministic permutation of the corpus
seeded by a single integer (default `0`) and is assigned **by model, never by score** —
splitting on the score would leak the test distribution into the threshold. Positives and
negatives are permuted independently so that a half with no positives reports TPR as *not
measured* rather than as zero.

The split seed is a free parameter and we report its cost. Across five seeds the
`ctc_mean_clean` rate spans **0.0482–0.0523**. Any single-split figure quoted to four
decimals is overstating the resolution of this measurement; that spread *is* the error bar,
and it is printed on the dashboard beside the headline rather than buried.

### 3.2 Threshold and interval

For each signal the operating point is the `1 − α` quantile of the calibration negatives
(α = 0.05), and the reported rate uses a **Wilson score interval**. Two conventions are
load-bearing:

* **0 alarms out of *n* is an upper bound, never a rate of zero.** The harness prints
  `0.0000 [0.0000, 0.0001]`, not `0.0000`.
* **A denominator below the floor yields a bound, not a rate.** Below `min_negatives = 20`
  the evaluator refuses to print a point estimate at all.

### 3.3 The refusal principle

Every instrument in this lane **refuses rather than guesses**, and each refusal exists
because its absence cost something real:

| Instrument | Refuses when | Because |
|---|---|---|
| census | a cross-source seed collision is unresolved | silently deduping turns two claims into one model |
| verify | a model's manifest or weights digest differs from its plan | a filename is not evidence |
| verify | the native validator cannot be imported | **56,628 models were once quarantined for this** (§7.4) |
| scorer | a model's signals are undefined | one model once cost 5,000 rows (§7.3) |
| merge-plans | a duplicate seed, id, or shard id; a reference inside a shard | arithmetic that adds up and is scientifically wrong |
| merge-results | any shard's bytes, digest, membership, or settings disagree | a number must be published beside the exact bytes that produced it |
| tpr evaluate | an arm set contains a model from the clean population | scoring a null against its own parents is not a detection measurement |
| tpr evaluate | a rule's threshold sits at the ceiling of the clean corpus | a rule that cannot fire has an upper bound, not a rate |

### 3.4 Scoring

Scores are produced by the project's real adapter (`real_score_adapter:score`) at
`CVIAF_N_EVAL=40` held-out images and `CVIAF_BACKGROUNDS=4`, against one pinned reference,
with BLAS threads pinned to 1 for reproducibility. The scorer is exported whole-tree from
commit `a9e4ce02` and is **not** re-pinned or updated by this study. Determinism was
verified by reproducing individual box rows byte-for-byte on a different machine
(`s81617 → ctc_mean 0.7748 / refdiv 0.437`).

---

## 4. Results

### 4.1 The false-alarm rate

α = 0.05, thresholds from 28,314 calibration negatives, rate measured on 28,313 held-out
negatives:

| signal | FPR (95% Wilson) | alarms / n | threshold | status |
|---|---|---|---|---|
| `ctc_mean_clean` | **5.23%** [4.98%, 5.50%] | 1,481 / 28,313 | 0.984509 | **fpr_only** — interval contains α |
| `refdiv_mean_clean` | **5.18%** [4.93%, 5.45%] | 1,467 / 28,313 | 0.727659 | **fpr_only** — interval contains α |
| `ctc_peak_clean` | 0.00% [0.00%, 0.01%] | 0 / 28,313 | 1.000000 | **degenerate** |
| `ctc_q95_clean` | 0.00% [0.00%, 0.01%] | 0 / 28,313 | 1.000000 | **degenerate** |

**Reading the two live signals.** Both intervals contain the nominal 5% budget, so neither
is anti-conservative at this sample size. `refdiv_mean_clean` in particular is worth
noting: it measures *above* α on the retired 7,503-model fleet (5.73% [5.03%, 6.52%],
which tripped a calibration alarm) and *inside* α here. The difference is the population,
not the rule — more sources, more heterogeneity, and a much wider denominator. That is a
finding about scale, and it is the reason a 28k asset held-out half beats a 3.7k one for
this decision.

**Reading the two degenerate signals.** Both thresholds are `1.000000`, the maximum clean
score in the corpus. A threshold at the ceiling means the rule **cannot** fire, so its
`0.0000` measures the ceiling, not accuracy. We keep the columns in the ledger for
continuity and label them everywhere they appear. This is the single most likely
misreading of this study (see §5, T2).

### 4.2 The detection rate, per attack class and per dose

1,055 arms, judged at thresholds that were fixed before any arm existed. Wilson intervals
throughout. **No pooled rate is reported, on either axis.** A rate averaged over attack
classes survives exactly one follow-up question; a rate averaged over doses hides whether a
rule is detecting harm or merely noticing that weights changed. Every cell stands alone,
including the cells where the rule misses.

"unmoved" counts the arms in that cell whose attack left the repository's behaviour metric
(`f1_relative_drop`) at its floor — arms that behaviour cannot distinguish from clean.

| attack class | dose | arms | unmoved | `ctc_mean_clean` | `refdiv_mean_clean` |
|---|---|---:|---:|---|---|
| `bias_lift` | 0.10 | 99 | **99** | 6.1% [2.8, 12.6] | 5.1% [2.2, 11.3] |
| `bias_lift` | 0.25 | 99 | **99** | 6.1% [2.8, 12.6] | 5.1% [2.2, 11.3] |
| `bias_lift` | 0.50 | 99 | **99** | 6.1% [2.8, 12.6] | 5.1% [2.2, 11.3] |
| `bias_lift` | 1.00 | 99 | **99** | 6.1% [2.8, 12.6] | 5.1% [2.2, 11.3] |
| `substitution` | 0.10 | 99 | 88 | 4.0% [1.6, 9.9] | 13.1% [7.8, 21.2] |
| `substitution` | 0.25 | 94 | 40 | 2.1% [0.6, 7.4] | **47.9%** [38.1, 57.9] |
| `substitution` | 0.50 | 63 | 6 | 3.2% [0.9, 10.9] | **93.7%** [84.8, 97.5] |
| `substitution` | 1.00 | **7** | 0 | 7/7 — *too few to conclude* | 7/7 — *too few to conclude* |
| `weight_tamper` | 0.10 | 99 | 73 | 5.1% [2.2, 11.3] | 7.1% [3.5, 13.9] |
| `weight_tamper` | 0.25 | 99 | 23 | 5.1% [2.2, 11.3] | 22.2% [15.2, 31.4] |
| `weight_tamper` | 0.50 | 99 | 4 | 11.1% [6.3, 18.8] | **59.6%** [49.7, 68.7] |
| `weight_tamper` | 1.00 | 99 | 0 | 29.3% [21.2, 38.9] | **92.9%** [86.1, 96.5] |

`ctc_peak_clean` and `ctc_q95_clean` have no columns, for the same reason they are degenerate
in §4.1: their frozen threshold **is** the ceiling of the clean corpus, so they cannot fire on
an attack either. Their rate is `null` in every cell — never `0`, and never a claim.

**1. The best rule by false alarms is not the best rule by detection.** `ctc_mean_clean` and
`refdiv_mean_clean` hold the false-alarm rate 0.05 points apart (5.23% and 5.18%, §4.1) and
differ by an order of magnitude in what they catch. `ctc_mean_clean` sits at its own
false-alarm rate at nearly every dose — 2.1% against a 5.2% budget at the dose where the
other rule catches 47.9%. A calibration study on its own would have published these as two
correct rules and let every reader assume they detect equally well.

**2. Detection tracks damage, not deviation.** The targeted family moved the weights in all
396 arms, at four doses spanning a factor of ten, and left the behaviour metric at exactly
0.000 relative change in every one of them. It is caught at **5.1%** — the false-alarm rate.
A signal derived from behaviour cannot see a modification that behaviour does not reflect,
whatever the weights look like; the receipt records how much of each cell is inert precisely
so that this row is not read as a statement about harm.

**3. The class that matters most is the one measured least.** Pruning is the destructive
family — 93.7% caught at dose 0.50 — and it is also the family that destroys its own
subjects: 92 of 99 arms are unscorable at dose 1.00. At that dose the surviving 7/7 carries
`insufficient_denominator` and is reported with its count rather than as 100%. §4.3 states
what that costs.

**Reading the table safely.** Within a cell, the comparison between rules is exact and is the
only cross-rule comparison the dose units allow. Across doses, the ratio is meaningful;
across classes, it is not, because a dose is a different physical quantity in each family. And
the honest resolution of any single cell is its interval — roughly ±10 points at n ≈ 99 — not
the three decimals it is printed to.

### 4.3 What is still absent — and the caveat that qualifies §4.2

The merged pass is still clean null, so for *that* population these remain refused:

| quantity | value for the 56,627-model pass | why |
|---|---|---|
| TPR | `null` | 0 positives in that population |
| Expected loss per asset | not priced | a function of TPR |
| Break-even prevalence | `null` | a function of TPR |
| Clause 3.7 ODA recall | `measurable: false` | a function of TPR |

Every one is emitted as an explicit refusal with a reason string. A `0.000` in any of those
cells would have been a fabricated measurement, and the evaluator is written to make that
impossible. §4.2 supplies a detection rate for a *different*, declared population; it does
not back-fill these cells, and the two must not be multiplied together — expected loss needs
the base rate of the population being defended, and this corpus is not that population.

**The survivorship caveat, stated plainly, and now as a curve.** 133 of the 1,188 arms were
built and could not be scored, and **every one of them is `substitution`**. The dose column
turns that from a footnote into a measurement:

| prune dose | arms scored | unscorable | what the rate above is measured on |
|---|---:|---:|---|
| 0.10 | 99 | 0 | the whole cell |
| 0.25 | 94 | 5 | 95% of the cell |
| 0.50 | **63** | **36** | **64% of the cell** |
| 1.00 | **7** | **92** | **7% of the cell — not a rate** |

The class whose detection rate matters most is the one measured on the subset that survived
its own attack: pruning that destroys a model's measurability removes that model from the
*denominator*, not from the adversary's arsenal, so every `substitution` figure in §4.2 is an
upper bound. The bias is not uniform — it grows with the dose, exactly as the damage does —
which is why §4.2 has a dose column rather than a single row. This is the largest
qualification of the detection result, and it is a property of the attack, not of the
detector: no rule can be credited or blamed for a model it was never shown.

### 4.4 Supporting measurements (retired populations, for continuity)

These describe *earlier, smaller* corpora. They are kept because they bound how much the
conclusions depend on population size, not because they are the current result.

| measurement | population | value |
|---|---|---|
| `ctc_mean_clean` FPR | 7,503 clean / 3,751 held out | 4.80% [4.16%, 5.53%] |
| `refdiv_mean_clean` FPR | 7,503 clean / 3,751 held out | 5.73% [5.03%, 6.52%] — **calibration alarm** |
| `fused_live_cauchy` FPR | same | 5.2% [4.5%, 6.0%] |
| Battery detection (TPR @ FPR 0) | 48 arms, 7 detections | 14.6% (Wilson ≈ [7%, 28%]) |
| Strongest cheap baseline | same | 8.2% |
| Per-kind multiplicity | 14 cells | 0 survive Benjamini–Hochberg |

The last three are the detection-side numbers from the *retired* corpora, and the interval on
14.6% spanned roughly 7–28%: at 48 arms it was never able to decide anything, which is why it
is superseded here rather than extended. §4.2 replaces it with 1,055 scored arms, three
classes, four doses and a frozen operating point. Two things remain unmeasured and are not to be inferred from §4.2:
the **fusion rule** has not been scored on either population, and nothing in this repository
has been tested against a real third-party backbone.

---

## 5. Threats to validity

We rank these by the damage they would do if a reader ignored them.

**T1 — Precision without validity.** 56,627 models across three machines with byte-level
provenance makes narrow intervals look like authority. The population has no demonstrated
relationship to evaluation data, and no real third-party backbone is in it. The better the
statistics get, the more likely the number is over-trusted. This is a social failure mode,
not a technical one, and it is the primary threat to this work.

**T2 — `0.0000` is the most quotable number here.** The two degenerate rules are exactly
the ones that look best in a table.

**T3 — The reference model is a free parameter.** `refdiv_*` is measured against one pinned
model; the shard containing that model is the lowest-`refdiv` shard. Per-shard means are
identical to four decimals, so this is *not proven causal* — which is worse than a
confirmed effect: it is an unexplained parameter that moves the operating point and could
be tuned to hit a target FPR without touching a detector.

**T4 — Target selection by dose.** The dose column in §4.2 is a gift to a careless reader:
of twelve cells, the four that flatter the detector most are easy to quote, and "93.7%" is
much more memorable than the 13.1% at the dose where the attack is subtle. The ladder does
not remove this risk, it multiplies the opportunities for it. The mitigation is structural —
every cell is shown, the cells below the floor are flagged rather than dropped, and no
average exists to be quoted instead.

**T5 — The study multiplies precision, not validity.** The population grew roughly 7× and
added nothing to external validity. This is the honest summary of the last pass.

**T6 — The corpus is not in the repository.** The 21 MB ledger and 56,627 model
directories live outside git. A clean clone can reproduce the *method*, and can open the
committed dashboard, but cannot recompute the headline without the archives. The 193-arm
detection set is smaller and cheaper to rebuild (38.6 s + 19.5 s), which makes §4.2 the more
reproducible of the two results — and the one a sceptic should attack first.

**T7 — The class we measured best is the class we could measure least.** All 133 unscorable
arms are `substitution`, and they concentrate at the doses where the attack works: 36 of 99
gone at 0.50, 92 of 99 at 1.00 (§4.3). The reported rates for the most destructive class are
computed on the attacks that left the statistic computable, so they are upper bounds that
look like estimates — the most dangerous kind of number in this document.

**T8 — The inert family invites the wrong conclusion in either direction.** `bias_lift` is
caught at 5.1%, which can be read as "the detector is blind to backdoors" (it is not: no
backdoor was built) or as "backdoors are undetectable" (unsupported: the arms carry no
trigger, so they are not backdoors). The honest statement is narrower and less quotable — a
targeted weight change that leaves the behaviour metric at zero is not caught above chance by
two behaviour-derived signals.

---

## 6. Reproducing this

The full run-book, with exact commands and expected output, is
[`docs/REPRODUCE.md`](docs/REPRODUCE.md). In short, the pipeline is five stages and each
one fails closed:

```bash
# 0. environment
export CVIAF_REFERENCE_DIR=$HOME/cviaf-analysis/clean_none_fixed_s5800
export CVIAF_N_EVAL=40 CVIAF_BACKGROUNDS=4
export PYTHONPATH=<pinned-scorer>:$HOME/cviaf-analysis/scripts

# 1. census  - count every source, resolve every seed collision, name every gap
python scripts/integration.py census --root <archives> --out runs/integration_census.json

# 2. verify  - every model against the sha256 of both files, from its own plan
python scripts/integration.py verify --source <id>=<corpus> --plan <id>=<shard.json> \
    --quarantine <q> --owned-root <root> --out runs/integration_verify.json

# 3. plans   - one byte-pinned plan; refuses duplicate seeds/ids and a stocked reference
python scripts/integration.py merge-plans --source <id>=<shard.json> ... \
    --exclude clean_none_fixed_s5800 --reference clean_none_fixed_s5800 --out <plan>

# 4. score   - resumable, checkpointed, tolerant of one undefined model
python scripts/integration.py score --shard <shard> --corpus <corpus> --out <results> \
    --reference ... --workers 5 --rewrite-shard

# 5. merge   - preflight every shard, then the locked ledger and the native report
python scripts/integration.py merge-results --plan <plan> --results <results> --out <out> \
    --merge-cmd <chunk_coordinator.py> --repo-root <pinned-scorer>

# 6. attack  - ADDITIVE, and it never writes to the FPR ledger:
#              derive arms from the clean corpus, score them, judge at the frozen thresholds
python scripts/tpr_arms.py build --shard <id>=<shard.json> --corpus <id>=<corpus> \
    --out <arms-dir> --per-shard 9 --magnitudes 0.1 0.25 0.5 1.0 \
    --kinds weight_tamper substitution bias_lift
python scripts/integration.py score --shard <arms-dir>/plan/shards/tpr_ladder.json \
    --corpus <arms-dir> --out <tpr-results> --reference <same-ref> --workers 5 --rewrite-shard
# 6b. per class, per dose - the pooled table is refused by name
python scripts/tpr_arms.py evaluate-ladder --results <tpr-results> \
    --registry <arms-dir>/registry.jsonl \
    --skipped <tpr-results>/skipped_tpr_ladder.json \
    --frozen-report runs/merged_fpr_tpr_report.json --out runs/tpr_ladder_at_frozen.json
```

Stage 6 reads the merged report to *borrow* its thresholds and writes a separate receipt. It
cannot recalibrate anything: the thresholds are read from committed bytes, and the arm set is
built from the corpus without touching it. `evaluate` (single dose) is still there and still
reproduces `runs/tpr_at_frozen.json` byte-for-byte; `evaluate-ladder` is the per-class,
per-dose one and is the receipt §4.2 quotes.

Cheap and immediate: the **interactive dashboard** is one self-contained file needing no
server, no network and no installation —
[`demo/fpr_dashboard.html`](demo/fpr_dashboard.html) carries all 56,627 sets of scores and
recomputes the headline *in the browser*. It is the intended way for a sceptical reader to
attack these numbers:

* drag a threshold and watch the rate follow, then **reset** to the committed operating point;
* switch the population to the **calibration half** and watch the same statistic become
  optimistic by construction — the split's purpose, made visible rather than described;
* press **push to the ceiling** on `ctc_peak_clean` and watch the rate become `0.00%` while
  a banner explains the rule cannot fire (T2, defused);
* read the **split-seed spread** that a single headline hides (T1, defused);
* open the **detection** tab to see what the frozen thresholds actually buy, **class by class
  and dose by dose**, with no pooled row to hide behind — including the rules that cannot fire
  (labelled `not measurable`, never `0%`), the arms whose attack left behaviour untouched, and
  the cells measured only on the survivors of their own attack.

The page's generator **fails the build** if its arithmetic disagrees with the committed
report, and embeds float64 rather than float32 for exactly that reason: a float32
round-trip moves the 0.05 quantile to `0.9845092` and can shift the hit count by one, which
would put a page in front of a reader disagreeing with the signed report by one model.

---

## 7. What went wrong, and what now prevents it

This section is the part we would want to read first in someone else's paper. Every item
is a real incident in this repository, each is reproducible from the receipts, and each
produced a test.

### 7.1 Seven archives, one corpus — and two of them were the same corpus twice

The first census found **37,686 cross-source seed collisions**. Most were expected:
`box1-corpus-v1` and `box1-corpus-v2` are two exports of *one* box whose weights are
byte-identical, and `box-plan7` ships a byte-identical copy of box 8's shard while
`full-results7` carries a copy of box 8's results. A globbing merge would have
double-counted 5,000 models and nobody would have noticed, because the counts would still
have added up. Collisions are therefore resolved by hashing the bytes *inside* the
archives, never by trusting a filename.

### 7.2 Two plans, two results, one box

Box 1 exists as a v1 and a v2 export. The results sets are byte-identical (`report`
`a277112b…`, npz `f39ed49b…`) but the **manifests serialise differently**, so
`full-resultsv1` is pinned to `box-plan1v1`'s shard bytes and `full-results-1v2` to
`box-plan-1v2`'s. The merge preflight **refused the cross-pairing before reading a row** —
which is the entire reason the plan pins bytes instead of trusting a name. The combination
used here is v2-plan with v2-results, the pair the verification pass checked.

### 7.3 One undefined statistic cost 5,000 models

Lab box 5's original run died at row 1,467 with:

```
ValueError: repository scorer omitted signals: dict_keys(['refdiv_mean_clean'])
```

## Known issues

- Provenance keys are regenerated on every engine construction. `InferenceProvenanceEngine.__init__` calls `generate_keypair()` unconditionally instead of loading an existing keypair from `key_dir`, so two engines pointed at the same directory get different keys. Any seal verified by an engine other than the one that signed it fails, which is why the demo flags all 10 seals. Audit trail verification is unaffected.
- Without the `cryptography` package, signing silently falls back to HMAC-SHA256 with a shared secret. That is symmetric, so it provides no non-repudiation, and the fallback is not surfaced anywhere in the report. The current `.venv` is in this state because `cryptography` is not installed. Treat any seal as a tamper check, not as proof of origin, until `cryptography` is installed and the fallback is made explicit.
- Signing keys are not committed. The historical demo keys under `cviaf_demo_output/keys/` and `demo-output/keys/` were removed from the tree (in HMAC fallback mode the `.pub` is a copy of the same shared secret). The demo regenerates a fresh keypair on every run, so nothing is needed to run it. To re-verify the seals in the historical output directories, restore the original keys locally per `demo-output/keys/README.md`. Note the keys still exist in git history for clones made before this cleanup.
- `assess` degrades silently. A missing imaging or model library produces a warning rather than an error, and the pipeline continues with whichever modules could run.

## Reference documents

- `docs/EXPERIMENT_LANES.md` the parallel experiment lanes and what they measured
- `docs/MVP_MAC.md` how to run, read and improve the laptop MVP
- `docs/SCALING_PLAN.md` M3 → H200 scaling plan
- `docs/CVIAF_V3_ARCHITECTURE.md` design and roadmap
- `docs/PS26228_REQUIREMENT_TRACE.md` problem-statement traceability and acceptance matrix
- `docs/DEEP_RESEARCH_SKILL.md` research protocol
- `PRD_SIH26228_CVIAF.md` product requirements
- `CV_INTEGRITY_ASSURANCE_2026.md` domain research — its §12 is the as-built status board (what each recommendation became once measured)
- `RESEARCH_CHECKPOINT_26228.md` research checkpoint — its UPDATE 3 records the measured reversals and the open items
- `Assurance_Framework_Technical_Report.md` technical report
- `cv_integrity_assurance_research_report.md`, `cv-pipeline-integrity-reference.md` supporting research
- `schemas/sample-assurance-report.json` example report shape
