# Model inventory — every model, where it came from, and what it is

This is the catalogue behind the population table in the [README](../README.md) §2. It is
generated from the committed receipts, not typed by hand:

| number | source of truth |
|---|---|
| 37 sources, 37,686 collisions, 0 fatal, gaps | [`runs/integration_census.json`](../runs/integration_census.json) |
| 56,628 / 56,628 verified, 0 failed, 0 moved | [`runs/integration_verify.json`](../runs/integration_verify.json) |
| 56,627 models in the published plan, per-rule FPR | [`runs/merge_report.json`](../runs/merge_report.json) |
| 193 scored arms, TPR per rule per class | [`runs/tpr_at_frozen.json`](../runs/tpr_at_frozen.json) |

**Size warning.** The 21 MB per-model ledger (`fpr_tpr_ledger.json`, 56,627 records) and the
56,627 model directories live **outside git**, in the analysis root. This document therefore
catalogues the *population*; the per-model rows live in the JSONL registries beside each
shard and in the ledger (see §6).

---

## 1. Where the models are

| root | contents |
|---|---|
| `~/cviaf-analysis/` | the analysis root: corpora, plans, results, merged output, the pinned scorer |
| `~/cviaf-analysis/clean_none_fixed_s5800` | the one pinned reference model used by every score in this study |
| `~/cviaf-analysis/merged-results/` | 11 shards of results, hardlinked from their originals (33 hardlinks) |
| `~/cviaf-analysis/merged-out/` | the merged ledger and the native report |
| `~/cviaf-analysis/combined-plan/` | the merged plan (11 shards) |
| `~/cviaf-analysis/CVIAF-scoring/` | the scorer, exported whole-tree from commit `a9e4ce02` |
| `<repo>/models_results/cviaf/` | the lab box plans and results as they arrived, plus the box corpora (`box*-corpus*.tar.gz`) |
| `<repo>/fleet-all-6250.tar.gz`, `<repo>/git-1056.tar.gz` | the two root archives |

A model is a directory named `clean_none_fixed_s<seed>` containing exactly:

```
manifest.json    # scene spec, training spec, and the pinned digests of both files
weights.npz      # the trained weights
```

The **seed is the identity**: it is the detector seed, it is unique across the whole merged
corpus, and the merge refuses if any seed appears twice. It is parsed from the `_s<seed>`
suffix of the directory name — and an id without that suffix yields no seed rather than
seed 0, which is a test.

---

## 2. All 37 census sources

`kind` is how the census classified the source; `role` is whether it contributes models to
the merged plan (`input`), is derived from another source, or exists only as results.

| source id | kind | role | models | seed range | from archive |
|---|---|---|---:|---|---|
| `box-plan7` | lab_plan | derived | 10,000 | 90,152–100,151 | — |
| `fleet-all-6250` | root_archive | input | 6,250 | 150–9,249 | `fleet-all-6250.tar.gz` |
| `box1-corpus-v1` | lab_corpus | input | 5,000 | 60,152–65,151 | `box1-corpus-v1.tar.gz` |
| `box1-corpus-v2` | lab_corpus | input | 5,000 | 60,152–65,151 | `box1-corpus-v2.tar.gz` |
| `box3-corpus` | lab_corpus | input | 5,000 | 70,152–75,151 | `box3-corpus.tar.gz` |
| `box4-corpus` | lab_corpus | input | 5,000 | 75,152–80,151 | `box4-corpus.tar.gz` |
| `box5-corpus` | lab_corpus | input | 5,000 | 80,152–85,151 | `box5-corpus.tar.gz` |
| `box6-corpus` | lab_corpus | input | 5,000 | 85,152–90,151 | `box6-corpus.tar.gz` |
| `box7-corpus` | lab_corpus | input | 5,000 | 90,152–95,151 | `box7-corpus.tar.gz` |
| `box8-analysis` | lab_corpus | input | 5,000 | 95,152–100,151 | `box8-analysis.tar.gz` |
| `box-plan-1v2` | lab_plan | derived | 5,000 | 60,152–65,151 | — |
| `box-plan1v1` | lab_plan | derived | 5,000 | 60,152–65,151 | — |
| `box-plan3` | lab_plan | derived | 5,000 | 70,152–75,151 | — |
| `box-plan4` | lab_plan | derived | 5,000 | 75,152–80,151 | — |
| `box-plan5` | lab_plan | derived | 5,000 | 80,152–85,151 | — |
| `box-plan6` | lab_plan | derived | 5,000 | 85,152–90,151 | — |
| `box-plan8` | lab_plan | derived | 5,000 | 95,152–100,151 | — |
| `models_results/laptop models` | laptop_copy | derived | 2,686 | 41,150–58,323 | — |
| `runs/clean_null_local_w2` | mac_local | input | 1,946 | 12,745–14,690 | — |
| `runs/clean_null_local_w5` | mac_local | input | 1,946 | 20,188–22,133 | — |
| `runs/clean_null_local_w6` | mac_local | input | 1,939 | 22,669–24,607 | — |
| `runs/clean_null_local_w3` | mac_local | input | 1,937 | 15,226–17,162 | — |
| `runs/clean_null_local_w4` | mac_local | input | 1,935 | 17,707–19,641 | — |
| `runs/clean_null_local_w1` | mac_local | input | 1,934 | 10,264–12,197 | — |
| `git-1056` | root_archive | input | 1,056 | 10,150–28,385 | `git-1056.tar.gz` |
| `runs/clean_null_win_w3` | laptop | input | 675 | 55,152–55,826 | — |
| `runs/clean_null_win_w4` | laptop | input | 672 | 57,652–58,323 | — |
| `runs/clean_null_win_w2` | laptop | input | 671 | 43,401–44,071 | — |
| `runs/clean_null_win_w1` | laptop | input | 668 | 41,150–41,817 | — |
| `full-resultsv1` | lab_results | derived | 0 | — | — |
| `full-results-1v2` | lab_results | derived | 0 | — | — |
| `full-results3` … `full-results8` (6) | lab_results | derived | 0 | — | — |

Note `box-plan7`: a single lab plan carrying **10,000** models, because box 7's plan ships a
byte-identical copy of box 8's shard. That is one of the four ways the 37,686 collisions
arose, and it is why the census hashes weights rather than comparing names.

### 2.1 The two root archives

| archive | bytes | SHA-256 | manifests | verified |
|---|---:|---|---:|---|
| `fleet-all-6250.tar.gz` | 123,618,181 | `01aee38f3c0e36da6570d6b87080b782343bfddddb767f793a4eb0b7332f9b19` | 6,250 | digest matches declared |
| `git-1056.tar.gz` | 20,924,212 | `eceba2ce322e88fae5032371e95801ef24523c281f0ff3af364c7515492a038e` | 1,056 | digest matches declared |

Both were reassembled from parts (`*part_00…`) before hashing; a missing part is a failure,
not a reason to score the remainder.

---

## 3. The 11 merged shards

| shard | models | seeds | produced by | scored here? |
|---|---:|---|---|---|
| `seed_60152_65151` | 5,000 | 60,152–65,151 | lab box 1 (v2 export) | yes |
| `seed_70152_75151` | 5,000 | 70,152–75,151 | lab box 3 | yes |
| `seed_75152_80151` | 5,000 | 75,152–80,151 | lab box 4 | yes |
| `seed_80152_85151` | **4,999** | 80,152–85,151 | lab box 5 | **rescored here** (originally 1,467 rows, no results file) |
| `seed_85152_90151` | 5,000 | 85,152–90,151 | lab box 6 | yes |
| `seed_90152_95151` | 5,000 | 90,152–95,151 | lab box 7 | yes |
| `seed_95152_100151` | 5,000 | 95,152–100,151 | lab box 8 | yes |
| `seed_150_9249` | 6,249 | 150–9,249 | cloud fleet | yes |
| `seed_10150_28385` | 1,056 | 10,150–28,385 | git export | yes |
| `seed_41150_58323` | 2,686 | 41,150–58,323 | second laptop (Windows) | yes |
| `seed_10264_24607` | 11,637 | 10,264–24,607 | this laptop (macOS) | yes |
| **total** | **56,627** | | | |

`seed_80152_85151` is 4,999 for one reason and it is named: `clean_none_fixed_s81621` is a
clean model whose CTC statistic is undefined on all 40 held-out images. It is one row in
`skipped_seed_80152_85151.json`, and the re-issued shard declares
`unscorable_model_ids: ['clean_none_fixed_s81621']`. It is **not** silently absent from a
denominator.

### 3.1 Declared gaps

| gap | why | recoverable? |
|---|---|---|
| lab box 2, seeds 65,152–70,151 (5,000 models) | no box plan in `models_results/cviaf`; the box was never trained and an exhaustive search found nothing | **no — permanent** |
| `full-results5` | the box was never scored, so there is no `results_seed_*.npz` | yes: scoring box 5's corpus produced the 4,999-row shard above |

---

## 4. The attack arm inventory (detection pass)

| | |
|---|---|
| arms built | **198** |
| arms scored | **193** |
| unscorable | **5**, all `substitution` |
| parents | 9 clean models per shard × 11 shards = 99 |
| classes | `weight_tamper` (head noise), `substitution` (structural pruning) |
| magnitude | 0.25 for every arm |
| naming | `<kind>_r<magnitude>_s<parent-seed>`, e.g. `substitution_r0.25_s74592` |
| location | `~/cviaf-analysis/tpr-corpus/models/` + `registry.jsonl` |
| plan | `~/cviaf-analysis/tpr-corpus/plan/shards/tpr_arms.json` |
| results | `~/cviaf-analysis/tpr-results/results_tpr_arms.npz` |
| unscorable | `~/cviaf-analysis/tpr-results/skipped_tpr_arms.json` |
| receipt | [`runs/tpr_at_frozen.json`](../runs/tpr_at_frozen.json) |

Every arm directory passes the same `cviaf.lab.manifest_schema.validate_model_dir` the clean
corpus passes, so an arm is admissible evidence by exactly the standard a clean model is. An
arm whose tampering turned out to be a no-op is refused at build time rather than recorded as
an attack.

The five unscorable arms are `substitution_r0.25_s74592`, `_s93482`, `_s96262`, `_s7159`,
`_s41150`, each with the same producer message
`ValueError: repository scorer omitted signals: dict_keys(['refdiv_mean_clean'])`. Pruning
can destroy the *statistic* as well as the model, so that class's detection rate is measured
on the subset that survived its own attack.

---

## 5. Provenance invariants

| invariant | enforced by |
|---|---|
| no seed appears twice in the merged plan | `merge-plans` |
| every model matches the digests its plan pinned | `verify` |
| a shard's file has not changed since it was scored | `merge-results` preflight |
| every shard used the same reference and the same scorer settings | `merge-results` preflight |
| no stub-mode result reaches the ledger | `merge-results` preflight |
| a dropped model is named, never silently subtracted | `--rewrite-shard`, `unscorable_model_ids` |
| the reference model is never scored against itself | `merge-plans` |

The reference: `clean_none_fixed_s5800`, manifest
`fd84409c2768f091df229116ee55e21058485c92cb7565f90118c6cd24b40234`, weights
`434561bb6e224a2177d561474f1138b5a24a8269ab4503edc6124b9111695be5`.

---

## 6. The per-model rows

The tables above are the population. The **56,627 individual rows** — model id, seed, path,
shard, the four signals, and the split half — live in:

| artifact | location | size | in git? |
|---|---|---|---|
| merged ledger `cviaf.fpr-tpr-ledger.v1` | `~/cviaf-analysis/merged-out/fpr_tpr_ledger.json` | 21 MB, 56,627 records | **no** |
| native evaluator report | `~/cviaf-analysis/merged-out/fpr_tpr_report.json` | — | no (summarised in `runs/`) |
| analysis rows | `~/cviaf-analysis/merged-out/analysis_rows.jsonl` | — | no |
| per-shard registries | `registry_<shard_id>.jsonl` beside each shard's results | — | no |
| per-arm registry | `~/cviaf-analysis/tpr-corpus/registry.jsonl` | 198 rows | no |

This is a real limitation of the study and it is recorded as threat T6: a clean clone can
reproduce the method and can open the committed dashboard (which carries all 56,627 score
sets embedded), but cannot recompute the headline without the archives.

---

## 7. Recomputing these tables

```bash
# source counts, seed ranges, collisions, gaps
python scripts/integration.py census --integration-root .. --analysis-root ~/cviaf-analysis \
    --out runs/integration_census.json

# the per-source table in §2, straight from the receipt
python - <<'PY'
import json
c = json.load(open("runs/integration_census.json"))
for s in sorted(c["sources"], key=lambda x: -x["n_models"]):
    print(f"{s['id']:<34}{s['n_models']:>7}  {s.get('seed_min')}..{s.get('seed_max')}")
print("collisions:", c["n_collisions"], "fatal:", len(c["collisions_fatal"]))
print("gaps:", c["gaps"])
PY

# the attack arm inventory in §4
python scripts/tpr_arms.py evaluate --results ~/cviaf-analysis/tpr-results \
    --frozen-report runs/merged_fpr_tpr_report.json --out /tmp/tpr_check.json
diff <(python -m json.tool runs/tpr_at_frozen.json) <(python -m json.tool /tmp/tpr_check.json) \
  && echo "receipt reproduces"
```
