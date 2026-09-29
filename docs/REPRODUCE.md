# Reproduce this — exact commands, expected output, and what cannot be reproduced

Companion to [`docs/METHODOLOGY.md`](METHODOLOGY.md) (why each stage is built that way) and
the [README](../README.md) (what was measured). This document is only commands and the output
they should produce.

Notation: `$ANALYSIS` = `$HOME/cviaf-analysis`, `<repo>` = the worktree root (`.task3`).

---

## 0. Environment

Two Python lanes, deliberately separate:

| lane | interpreter | used for |
|---|---|---|
| **analysis** (numpy) | `<repo>/../.venv/bin/python` | census, verify, merge, dashboard, tests |
| **scoring** (torch) | `~/.venvs/cviaf-torch/bin/python` | anything that runs the real adapter |

```bash
export ANALYSIS=$HOME/cviaf-analysis
export PYTHONPATH=<repo>:$ANALYSIS/CVIAF-scoring:$ANALYSIS/scripts
export CVIAF_REFERENCE_DIR=$ANALYSIS/clean_none_fixed_s5800
export CVIAF_N_EVAL=40
export CVIAF_BACKGROUNDS=4

# pin BLAS to one thread, or a 5-worker run is 5x oversubscribed per core
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

**Machine ceiling.** On a 16 GB M3 Air, `--workers 6` pins the machine and `--workers 5` is
the safe setting. Long runs go under `screen`: macOS has no `setsid`, so a dropped SSH
session otherwise kills a multi-hour run.

**Reference model** (must be byte-exact, or every score is against a different experiment):

| file | SHA-256 |
|---|---|
| `manifest.json` | `fd84409c2768f091df229116ee55e21058485c92cb7565f90118c6cd24b40234` |
| `weights.npz` | `434561bb6e224a2177d561474f1138b5a24a8269ab4503edc6124b9111695be5` |

---

## 1. Census

```bash
cd <repo>
python scripts/integration.py census \
    --integration-root .. --analysis-root $ANALYSIS \
    --out runs/integration_census.json
```

**Expected:** `verdict: PASS (exit 0)`.

| quantity | expected |
|---|---|
| sources | **37** |
| cross-source seed collisions | **37,686** |
| fatal collisions | **0** |
| gaps | lab box 2 (`65152..70151`), `full-results5` |
| archives whose digest matched | both (`fleet-all-6250`, `git-1056`) |

A non-zero exit here means one of four specific things — see the exit-code table in
[`METHODOLOGY.md` §1](METHODOLOGY.md#1-census--count-before-you-score). Do not proceed.

---

## 2. Verify — every model against its own pinned plan

```bash
cd <repo>
python scripts/integration.py verify \
    --source box3=$ANALYSIS/corpus-b3 ... \
    --plan box3=<repo>/models_results/cviaf/box-plan3/shards/'*.json' ... \
    --quarantine $ANALYSIS/quarantine \
    --owned-root $ANALYSIS \
    --out runs/integration_verify.json
```

**Expected:** `56,628 / 56,628 verified, 0 failed, 0 moved`, exit `0`.

**The evidence for "0 moved" is the absence of `$ANALYSIS/quarantine`.** A directory that
does not exist is the receipt; a counter inside the report is not.

> The native validator must be importable first. If it is not, this command refuses with a
> single environment problem and moves **nothing** — that behaviour is the fix for the
> 56,628-model incident, and it is the one thing you must not work around.

---

## 3. Merge plans

```bash
cd <repo>
python scripts/integration.py merge-plans \
    --source seed_60152_65151=<shard> ... \
    --exclude clean_none_fixed_s5800 \
    --reference clean_none_fixed_s5800 \
    --out $ANALYSIS/combined-plan
```

**Expected:** one `plan.json` + `shards/`, **11 shards, 56,627 models**, exit `0`. The
reference model is excluded (it must not be scored against itself) and must not appear in any
shard — the merge refuses if it does.

---

## 4. Score the corpus

```bash
cd <repo>
for shard in $ANALYSIS/combined-plan/shards/*.json; do
  ~/.venvs/cviaf-torch/bin/python scripts/integration.py score \
      --shard "$shard" --corpus $ANALYSIS/<corpus-for-that-shard> \
      --out $ANALYSIS/merged-results \
      --reference $CVIAF_REFERENCE_DIR \
      --workers 5 --rewrite-shard
done
```

**Expected per shard:** `{"scored": N, "incomplete": 0, "error": 0, "status": "PASS"}`,
exit `0`.

Resumable: a re-run skips `complete` and `incomplete` rows and **retries** `error` rows. If a
run is interrupted, re-run the same command; do not start from an empty output directory, and
do not treat "0 scored" as "nothing to do" without checking the registry.

---

## 5. The live poison replay (≈1.8 s) — one bad model must not cost a corpus

Box 5 died at row 1,467 with `ValueError: repository scorer omitted signals:
dict_keys(['refdiv_mean_clean'])` and shipped a registry with no results file. Here is that
failure reproduced live in two seconds, on four models — the three around `s81621` and the
one that has no CTC statistic.

```bash
cd <repo>
mkdir -p /tmp/mini && cp $ANALYSIS/mini-shard.json /tmp/mini/

time ~/.venvs/cviaf-torch/bin/python scripts/integration.py score \
    --shard /tmp/mini/mini-shard.json \
    --corpus $ANALYSIS/corpus-b5/box-corpus \
    --out /tmp/mini-out \
    --reference $CVIAF_REFERENCE_DIR \
    --workers 1 --rewrite-shard
```

**Expected output** (measured on this machine: `real 0m1.832s`):

```
{"error": 1, "incomplete": 0, "scored": 3, "shard": "seed_81619_81622", "status": "PASS"}
```

`/tmp/mini-out/skipped_seed_81619_81622.json` names the model and carries the producer's own
message:

```json
{"n_error": 1, "n_incomplete": 0, "n_models": 4, "n_scored": 3,
 "shard_id": "seed_81619_81622",
 "skipped": [{"model_id": "clean_none_fixed_s81621",
              "reason": "ValueError: repository scorer omitted signals: dict_keys(['refdiv_mean_clean'])",
              "status": "error"}]}
```

`/tmp/mini-out/shards/seed_81619_81622.json` is the **re-issued** shard: `count: 3`, with
`"unscorable_model_ids": ["clean_none_fixed_s81621"]`.

**What to look for.** Three models scored, one named and dropped, and the shard still
admissible. The difference between this and the original box-5 failure is entirely the
three-way classification (`complete` / `incomplete` / `error`): this run's single failure is
recorded as `error` — retried on resume, because a corpus that was mid-move is not a corpus
that is wrong — and an *undefined* statistic is the other case, which is deterministic and
never retried.

---

## 6. Merge results and run the evaluator

```bash
cd <repo>
python scripts/integration.py merge-results \
    --plan $ANALYSIS/combined-plan \
    --results $ANALYSIS/merged-results \
    --out $ANALYSIS/merged-out \
    --out-report runs/merge_report.json \
    --merge-cmd $ANALYSIS/CVIAF-scoring/<chunk_coordinator.py> \
    --repo-root $ANALYSIS/CVIAF-scoring
```

**Expected:** preflight green (**11 shards, 56,627 models scored**), then the ledger and the
native report, exit `0`.

| quantity | expected |
|---|---|
| calibration negatives | 28,314 |
| evaluation negatives | 28,313 |
| evaluation positives | 0 |
| `ctc_mean_clean` FPR | **0.0523** [0.0498, 0.0550], 1,481 hits, threshold `0.9845092069058692` |
| `refdiv_mean_clean` FPR | **0.0518** [0.0493, 0.0545], 1,467 hits, threshold `0.7276592261904761` |
| `ctc_peak_clean`, `ctc_q95_clean` | 0 hits, threshold `1.0` — **degenerate** |
| `tpr` | `"not measurable: this population carries clean nulls only…"` |

`positives_measured: false`, `status: fpr_only`, and every TPR-derived quantity is a refusal
string. That is the correct output for this population, not a bug to fix.

**Split sensitivity.** Re-running with a different split seed is the honest error bar. Across
seeds `0 / 1 / 7 / 42 / 1337` the `ctc_mean_clean` rate spans **0.0482–0.0523**. Seed `0`
reproduces the published threshold exactly, which is one way to check the environment.

---

## 7. The additive detection pass (does not touch §6)

```bash
cd <repo>
# 7a. build 198 arms from 9 clean models per shard - no retraining, no new data
python scripts/tpr_arms.py build \
    --shard seed_60152_65151=$ANALYSIS/combined-plan/shards/seed_60152_65151.json ... \
    --corpus seed_60152_65151=$ANALYSIS/<corpus> ... \
    --out $ANALYSIS/tpr-corpus --per-shard 9 --magnitude 0.25

# 7b. score them exactly as the corpus was scored
~/.venvs/cviaf-torch/bin/python scripts/integration.py score \
    --shard $ANALYSIS/tpr-corpus/plan/shards/tpr_arms.json \
    --corpus $ANALYSIS/tpr-corpus --out $ANALYSIS/tpr-results \
    --reference $CVIAF_REFERENCE_DIR --workers 5 --rewrite-shard

# 7c. judge at the thresholds §6 already froze
python scripts/tpr_arms.py evaluate \
    --results $ANALYSIS/tpr-results \
    --frozen-report runs/merged_fpr_tpr_report.json \
    --out runs/tpr_at_frozen.json
```

**Expected:** build 198 arms in ~39 s; score 193 with 5 errors in ~20 s; then

```
TPR at frozen thresholds (alpha 0.05) - 193 arms ['substitution', 'weight_tamper']
signal                threshold     caught/n      TPR   status
ctc_mean_clean         0.984509     7/193      0.0363   measured
ctc_peak_clean         1.000000     0/193         n/a   degenerate_rule
ctc_q95_clean          1.000000     0/193         n/a   degenerate_rule
refdiv_mean_clean      0.727659    67/193      0.3472   measured
```

**Reproducibility check** — the receipt is a pure function of the results directory and the
frozen report:

```bash
python scripts/tpr_arms.py evaluate --results $ANALYSIS/tpr-results \
    --frozen-report runs/merged_fpr_tpr_report.json --out /tmp/tpr_check.json
diff <(python -m json.tool runs/tpr_at_frozen.json) \
     <(python -m json.tool /tmp/tpr_check.json) && echo "receipt reproduces exactly"
```

**Verify the guardrail:** `runs/tpr_at_frozen.json` carries
`"fpr_ledger_untouched": true` and `"frozen_thresholds_from": "runs/merged_fpr_tpr_report.json"`.
If a run of this pass changes any of `runs/merge_report.json`, `runs/merged_fpr_tpr_report.json`
or the ledger, something is wrong with the run, not with the thresholds.

**Expected failure to expect.** 5 of 198 arms come back unscorable, all `substitution`, all
with the s81621 signature. That is the designed behaviour: `--rewrite-shard` names them in
`skipped_tpr_arms.json` and they are excluded from the denominator rather than counted as
misses. It also means that class's rate is measured on survivors — the receipt says so, and so
does the dashboard.

---

## 8. The dashboard

```bash
cd <repo>
python scripts/demo_dashboard.py \
    --ledger $ANALYSIS/merged-out/fpr_tpr_ledger.json \
    --report $ANALYSIS/merged-out/fpr_tpr_report.json \
    --plan $ANALYSIS/combined-plan \
    --verify runs/integration_verify.json \
    --census runs/integration_census.json \
    --tpr runs/tpr_at_frozen.json \
    --tpr-skipped $ANALYSIS/tpr-results/skipped_tpr_arms.json \
    --split-seed 0 \
    --out demo/fpr_dashboard.html
```

**Expected:** `~3.12 MB` written, exit `0`, with no warning about arithmetic disagreement. If
the generator's recomputed headline disagrees with `runs/merge_report.json` it **fails the
build** rather than writing the page. Then open `demo/fpr_dashboard.html` in a browser — no
server, no network, no install.

---

## 9. What cannot be reproduced from a clean clone

| artifact | where it lives | why it matters |
|---|---|---|
| the 56,627 model directories | `$ANALYSIS/*-corpus/`, `<repo>/models_results/cviaf/`, two root archives | the corpus is 144 MB of archives and tens of GB unpacked |
| the 21 MB ledger `fpr_tpr_ledger.json` | `$ANALYSIS/merged-out/` | it is the per-model evidence behind the headline (threat T6) |
| the pinned scorer export | `$ANALYSIS/CVIAF-scoring/` | the adapter must be the exact tree from `a9e4ce02` |
| the analysis root as a whole | `$ANALYSIS` | not in git by design |

What a clean clone **can** do: read every receipt under `runs/`, run both test lanes, rebuild
the attack arms and re-derive `runs/tpr_at_frozen.json` from the committed frozen report if it
has the arms' results, regenerate the dashboard if it has the ledger, and open the committed
`demo/fpr_dashboard.html` — which carries all 56,627 score sets embedded and recomputes the
headline in the browser.

---

## 10. Tests

```bash
cd <repo>
python -m pytest -q                             # 639 passed, 5 skipped, 1 xfailed (~42 s)
~/.venvs/cviaf-torch/bin/python -m pytest -q    # 648 passed, 1 xfailed (~51 s)
```

**Run them sequentially.** Launched into the same rootdir in parallel, the two lanes corrupt
each other's cache and one reports a fraction of its tests — 55 instead of 647 — **with exit
code 0**. A false green is worse than a red; check the count, not just the exit code.

Integration subset only:

```bash
python -m pytest tests/test_integration_census.py tests/test_integration_verify.py \
    tests/test_integration_plans.py tests/test_integration_score.py \
    tests/test_integration_merge.py tests/test_tpr_arms.py \
    tests/test_demo_dashboard.py -q
```
