# False-Alarm Behaviour of Clean-Null Vision Detectors at Scale

### A 56,627-model false-alarm calibration, a 198-arm detection measurement taken at those frozen thresholds, and the refusal-first pipeline that produced both

**Status:** false-alarm calibration complete; detection measured on a separate, additive attack set at the frozen operating point
**Population:** 56,627 clean-null models from 11 shards across 3 machines (FPR) · 198 attacked variants in 2 classes (TPR)
**Headline (FPR):** `ctc_mean_clean` **5.23%** [4.98%, 5.50%] and `refdiv_mean_clean` **5.18%** [4.93%, 5.45%] on 28,313 held-out clean assets at α = 0.05
**Headline (TPR, at that same frozen FPR):** `refdiv_mean_clean` **34.7%** [28.4%, 41.7%] · `ctc_mean_clean` **3.6%** [1.8%, 7.3%]
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
separate **198-arm attack set** is derived from models already in this corpus by weight-space
tampering only — head noise and structural pruning, no retraining, no new data — scored
through the same adapter against the same pinned reference, and judged at **the thresholds
the clean half had already frozen**. That yields the one result a calibration study cannot
reach: at a false-alarm rate of 5.2%, the two live rules are not equally useful.
`refdiv_mean_clean` catches **34.7%** [28.4%, 41.7%] of attacks; `ctc_mean_clean` catches
**3.6%** [1.8%, 7.3%]. The best rule by false alarms is not the best rule by detection, and
no quantity of clean data would have revealed it.

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
| Attacked variants built | **198** — 99 `weight_tamper`, 99 `substitution` |
| Variants scored | **193** — 99 + 94 (5 `substitution` arms unscorable) |
| Derived from | 9 clean models per shard × 11 shards, drawn from the corpus in §2.1 |
| Perturbation | weight-space only — Gaussian head noise / structural pruning, magnitude 0.25 |
| Retraining, new data, new reference | **none** |
| Thresholds | **frozen** from the clean half; never recalibrated |
| Cost | 38.6 s to build all 198 arms; 19.5 s to score 193 at 5 workers |

Every arm is a model directory that passes the same `validate_model_dir` the corpus does,
so an arm is admissible evidence by exactly the standard a clean model is. The five
unscorable arms are §7.3 recurring one level up — pruning can destroy the *statistic* as
well as the model — and they are named in `skipped_tpr_arms.json`, grouped by class on the
dashboard, and excluded from the denominator rather than counted as misses. §4.3 states what
that exclusion costs.

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

### 4.2 The detection rate, at the frozen operating point

193 arms, judged at thresholds that were fixed before the arms existed. Wilson intervals
throughout; `null` where a rule cannot fire at all.

| rule | frozen threshold | pooled TPR (95% Wilson) | `substitution` (94) | `weight_tamper` (99) |
|---|---|---|---|---|
| `refdiv_mean_clean` | 0.727659 | **34.7%** [28.4%, 41.7%] — 67/193 | 48.9% [39.1%, 58.9%] | 21.2% [14.3%, 30.3%] |
| `ctc_mean_clean` | 0.984509 | **3.6%** [1.8%, 7.3%] — 7/193 | 2.1% [0.6%, 7.4%] | 5.1% [2.2%, 11.3%] |
| `ctc_peak_clean` | 1.000000 | `null` — **degenerate** | `null` (≤3.1%) | `null` (≤3.0%) |
| `ctc_q95_clean` | 1.000000 | `null` — **degenerate** | `null` (≤3.1%) | `null` (≤3.0%) |

**The headline finding is that the two tables disagree.** `ctc_mean_clean` and
`refdiv_mean_clean` hold the false-alarm rate at 5.23% and 5.18% — indistinguishable at this
*n* — and differ by a factor of ten in what they catch. A calibration study on its own would
have published two correct rules and left the reader to assume they detect equally well.
`ctc_mean_clean` meets the budget by being nearly blind at this operating point: that is a
real property of the rule, and it becomes visible only when both rules are scored against the
same positives.

The two degenerate rules are reported here exactly as they are in §4.1 — as `null` with an
exact upper bound, never as `0%`. A rule whose threshold is the corpus ceiling cannot fire on
an attack any more than it can on a clean model, so its detection rate is undefined; the
table gives the bound that a reader can actually use.

**Severity within a class is wide and that is not a defect.** Zero-mean noise on a trained
head is closer to regularisation than to sabotage — some `weight_tamper` arms score *higher*
f1 than their clean parent. A 21% rate on that class is a statement about the weakest rung of
a dose ladder, not about the detector's ceiling.

**What this table does not say.** One magnitude (0.25), two families, ~95 arms per class: a
class-level rate quoted to three decimals overstates its own resolution, and the interval —
roughly ±10 points — is the honest version of the number. Nothing here transfers to an
adversary who picks a different magnitude, and the `substitution` column is measured on
survivors (§4.3).

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

**The survivorship caveat, stated plainly.** Five arms were built and could not be scored,
and **all five are `substitution`**. The class whose detection rate matters most is therefore
the one measured on the subset that survived its own attack: pruning that destroys a model's
measurability removes that model from the *denominator*, not from the adversary's arsenal.
The `substitution` figure above is an upper bound on the truth, not an estimate of it, and it
is the single largest qualification of §4.2.

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
is superseded here rather than extended. §4.2 replaces it with 193 arms, two classes and a
frozen operating point. Two things remain unmeasured and are not to be inferred from §4.2:
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

**T4 — One dose, two families.** The detection table (§4.2) rests on a single perturbation
magnitude and two attack classes. That is enough to rank two rules against each other at one
operating point and not enough to characterise either. A threshold change will still
"improve" a 193-arm number in a way that will not replicate, because severity within a class
is wide and the weakest rung dominates the rate.

**T5 — The study multiplies precision, not validity.** The population grew roughly 7× and
added nothing to external validity. This is the honest summary of the last pass.

**T6 — The corpus is not in the repository.** The 21 MB ledger and 56,627 model
directories live outside git. A clean clone can reproduce the *method*, and can open the
committed dashboard, but cannot recompute the headline without the archives. The 193-arm
detection set is smaller and cheaper to rebuild (38.6 s + 19.5 s), which makes §4.2 the more
reproducible of the two results — and the one a sceptic should attack first.

**T7 — The class we measured best is the class we could measure least.** All five unscorable
arms are `substitution`, so the reported rate for the most destructive class is computed on
the attacks that left the statistic computable (§4.3). The number is an upper bound that
looks like an estimate, which is the most dangerous kind of number in this document.

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
    --out <arms-dir> --per-shard 9 --magnitude 0.25
python scripts/integration.py score --shard <arms-dir>/plan/shards/tpr_arms.json \
    --corpus <arms-dir> --out <tpr-results> --reference <same-ref> --workers 5 --rewrite-shard
python scripts/tpr_arms.py evaluate --results <tpr-results> \
    --frozen-report runs/merged_fpr_tpr_report.json --out runs/tpr_at_frozen.json
```

Stage 6 reads the merged report to *borrow* its thresholds and writes a separate receipt. It
cannot recalibrate anything: the thresholds are read from committed bytes, and the arm set is
built from the corpus without touching it.

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
* open the **detection** tab to see what the frozen thresholds actually buy — including the
  rules that cannot fire, labelled as such rather than as `0%`, and the attack class whose
  rate is measured only on survivors.

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

and shipped a registry with 1,467 rows and **no results file**. Reproduced on this machine,
the cause is narrow and unglamorous: `clean_none_fixed_s81621` is a *clean* model whose CTC
statistic is undefined on all 40 held-out images, so the producer emits one key, and an
adapter requiring four raises **inside a worker pool**, which took the rest of the shard
with it.

The model is not corrupt — its manifest keys and scene spec match its neighbours. It is a
statistic that does not exist for that model. The fix is a scorer that **classifies**
instead of raising: `complete`, `incomplete` (undefined on every image — a property of the
model), and `error` (a property of *that run*, and therefore retried on resume). Box 5's
other 4,999 models were scored here; `s81621` is one row in
`skipped_seed_80152_85151.json` carrying the box's own message, and the re-issued shard
declares `unscorable_model_ids: ['clean_none_fixed_s81621']`.

This is the invariant the whole pass is built around: **one undefined statistic must not
become a lost corpus, and must not become a quietly smaller denominator.** It replays live
in **1.8 seconds** (`docs/REPRODUCE.md` §5; `real 0m1.832s` measured here).

### 7.4 We quarantined all 56,628 models ourselves

Verification was run once without the repository on `PYTHONPATH`. The native validator
could not be imported, and the pipeline recorded **every model as its own failure** — 56,628
of them — then **moved all 56,628 into quarantine**, which also emptied the corpora that two
scoring runs were reading and killed both mid-flight.

Nothing was deleted and the whole population was restored to the exact path each plan
expects. The lesson is a distinction the tooling did not previously make: *a per-model
problem string is the right answer for a bad model and the wrong answer for a missing
dependency.* The validator is now required **before anything can be moved**, and the test
asserts the side-effect guarantee (weights still in place, no quarantine directory)
rather than the error message.

### 7.5 The resume that would have reported a clean skip

Box 5's rescore first ran while the corpus was mid-move, so all 5,000 rows landed as
`error`. Resume treated every row in the registry as decided, so a second attempt would
have scored nothing, reported **0 scored**, and looked like a clean skip of work that had
never been done. `error` is now retried; `incomplete` is not. The rescore went on to score
4,999.

### 7.6 Criteria we tried and threw away

Recorded so nobody re-invents them:

| Attempt | Why it failed |
|---|---|
| Range-vs-CI-width for split stability | five draws of a binomial span ~2.3 SE by construction — it refuses *every* honest measurement |
| One shard's interval missing the pooled rate as a *refusal* | at six shards it fires ~26% of the time under exact homogeneity, and it rejected the real fleet |
| "The reference's shard is nearest the reference, hence lowest divergence" | per-shard means identical to four decimals (0.6962 vs 0.6926–0.6985); the story died on its own evidence |
| `ctc_peak` / `ctc_q95` as anomaly scores | saturate at 1.0 on clean models; the α-quantile *is* the ceiling |
| `ks_max` for drift | 1.0 on every contrast including controls — separates everything, therefore nothing |
| `hash()`-seeded demo RNG | unstable across processes, so demo values were unexplainable |

---

## 8. Repository map — where to go deeper

| Document | What it settles |
|---|---|
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | every procedure, step by step, with the failure mode it prevents and the test that pins it |
| [`docs/MODEL_INVENTORY.md`](docs/MODEL_INVENTORY.md) | every model generated: 56,627 rows by source, seed range, path and digest |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | exact end-to-end reproduction, including the live 1.8 s poison replay |
| [`docs/COVERAGE_STATEMENT.md`](docs/COVERAGE_STATEMENT.md) | what the engine claims to detect, and what it declares out of scope |
| [`docs/PS26228_REQUIREMENT_TRACE.md`](docs/PS26228_REQUIREMENT_TRACE.md) | problem-statement clause → artefact that satisfies it |
| [`docs/CLEAN_NULL_CORPUS.md`](docs/CLEAN_NULL_CORPUS.md) | how the null corpus is generated |
| [`docs/ATTESTATION.md`](docs/ATTESTATION.md) | content-addressing and the (unsigned) bundle |
| [`demo/fpr_dashboard.html`](demo/fpr_dashboard.html) | the results, interactively, offline |

Committed receipts backing every number in this document:

| receipt | contents |
|---|---|
| [`runs/integration_census.json`](runs/integration_census.json) | 37 sources, seed ranges, archive digests, 37,686 collisions, 0 fatal, every gap named |
| [`runs/integration_verify.json`](runs/integration_verify.json) | 56,628/56,628 verified against their own plans, 0 failed, 0 moved |
| [`runs/merge_report.json`](runs/merge_report.json) | the published headline, denominators, per-rule FPR |
| [`runs/merged_fpr_tpr_report.json`](runs/merged_fpr_tpr_report.json) | the evaluator's native report, including every refusal |
| [`runs/tpr_at_frozen.json`](runs/tpr_at_frozen.json) | detection per rule per attack class at the frozen thresholds, Wilson intervals, degenerate rules as `null`, in-scorable arms named |

---

## 9. What we would do next, in order

1. **Score the fused rule on the merged population.** Fusion is measured only on the
   retired 7,503-model fleet. The merged ledger already carries all four signals, so this
   is arithmetic, not compute — and it removes the one question a reviewer is most likely
   to ask that currently has no answer at large *n*.
2. **Turn the attack set into a dose–response.** §4.2 reports one magnitude. A ladder across,
   say, 0.05–1.0 per class would give the ROC an adversary-relevant axis and remove the
   single point that currently carries the whole detection claim. It would also quantify the
   survivorship problem instead of declaring it: at high magnitude, how many arms become
   unscorable, and does the class disappear as a discrimination problem because it stops
   producing measurable models at all?
3. **Remove the reference free parameter** (T3) with a shard-wise or leave-one-shard-out
   reference, and show the pathology disappear rather than documenting it.
4. **Establish external validity.** A real-backbone clean null — a few hundred real
   networks rather than synthetic detectors — is the only experiment that addresses T1
   without new data collection.
5. **Replace, don't merely retire, the saturated statistics** (T2). The degeneracy is
   diagnosed; a non-saturating tail statistic is the cure.

---

## Appendix A — Commit ledger

One commit per unit; each carries its own tests. Newest first:

`e784a8a` methodology / inventory / runbook docs · `292a205` the paper reports detection ·
`bea29b0` dashboard detection panel · `f7bf09c` measured detection at frozen thresholds ·
`44948c3` untrack compiled bytecode · `129b2bb` interactive dashboard ·
`679e9f7` merge/verify receipts · `48383ba` retry errored rows on resume ·
`496cad8` census receipt truncation · `6adf2b9` missing validator is a refusal ·
`99b0776` scoring CLI + re-issued shard · `585db99` merge preflight + published summary ·
`ad5f798` combined plan + tolerant scorer · `5e99b75` verification with quarantine ·
`2a0fed0` integration census · `3699cad` retired rules are not priced ·
`aaa09b1` provenance for the new receipts · `ed4965e` dose ladder ·
`e73b0e8` preflight · `c429f8b` per-kind multiplicity · `0a7a06a` retire saturated signals ·
`3164adc` power instrument · `a9e4ce0` fused fleet FPR (**pinned scorer**)

## Appendix B — Test state

| lane | command | result |
|---|---|---|
| numpy (canonical) | `python -m pytest -q` from `.task3` | **638 passed, 5 skipped, 1 xfailed** (40 s) |
| torch | `~/.venvs/cviaf-torch/bin/python -m pytest -q` | **647 passed, 1 xfailed** (45 s) |

Run the two lanes **sequentially**. Launched into the same rootdir in parallel they corrupt
each other's cache, and one lane reports a fraction of its tests passing — 55 instead of 647 —
with exit code 0. A green that green is worse than a red.

Tests are named after the incident they prevent — `test_a_missing_validator_refuses_before_moving_anything`,
`test_score_records_retries_a_previously_errored_model_on_resume`,
`test_dashboard_refuses_to_publish_a_page_that_contradicts_the_report` — because a test
named after a code path documents the code, and a test named after a failure documents the
project.

## Appendix C — Environment

MacBook Air M3, 16 GB, macOS. Scoring lane: Python 3.12.14, torch 2.14.0, numpy 2.5.3 at
`~/.venvs/cviaf-torch`. Analysis lane: Python 3.11 at `.venv`. Scoring is bandwidth- and
memory-limited: 6 workers pinned the machine, **5 is the safe setting** on 16 GB.
Long jobs run under `screen` because macOS has no `setsid`, and a lost session otherwise
kills a multi-hour run.
