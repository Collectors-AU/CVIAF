# Methodology — every procedure, and the failure each one prevents

This is the operational companion to the [README](../README.md). The README reports what was
measured; this document reports **how**, and for each procedure the specific way it can lie
to you and the test that now stops it.

Read it as a sequence. The pipeline is five stages that build on each other, plus one
additive stage that must not touch the first five:

```
census ──> verify ──> merge-plans ──> score ──> merge-results ──> [evaluator report]
                                            └──> (additive) tpr_arms build ──> score ──> tpr_arms evaluate
```

Everything is in [`scripts/integration.py`](../scripts/integration.py) and
[`scripts/tpr_arms.py`](../scripts/tpr_arms.py). Nothing here asks you to trust a filename.

---

## 0. Conventions this document assumes

| term | meaning here |
|---|---|
| **seed** | the detector seed. It is the model's identity: unique across the whole merged corpus, parsed from the `_s<seed>` suffix of the model id. |
| **source** | one corpus, plan, or archive handed to the pipeline, identified by an `id=path` pair. |
| **shard** | one contiguous block of models with a `shard_id` and a seed range — the unit that is scored as a job. |
| **negative** | a clean-null model. In the FPR pass, every model is a negative. |
| **receipt** | a committed JSON file under `runs/` that records what a stage saw and what it decided. |

Two conventions are load-bearing across every stage:

1. **A count of 0 is an upper bound, never a rate.** `0.0000 [0.0000, 0.0001]`, never
   `0.0000`. The instruments are written so a zero measured over *n* models cannot be typed
   as a rate.
2. **A refusal is a valid output.** Every stage has a documented non-zero exit code for the
   case where it cannot honestly answer, and the receipt records the refusal rather than a
   substituted default.

---

## 1. Census — count before you score

**Command:** `python scripts/integration.py census --integration-root .. --analysis-root ~/cviaf-analysis --out runs/integration_census.json`

**What it does.** Enumerates every source of models, counts them, extracts each model's seed
from its directory name, hashes each archive, and then answers the only question that matters
before the merge: *does any detector seed appear twice, and if so, is it the same model?*

**Why this way.** The alternative is `glob **/clean_none_fixed_*` plus a `--resume` flag. That
silently does four different things that all look like success: it double-counts a re-export,
it averages over a source that drifted into another's seed range, it averages over a
truncated archive, and it turns a real collision into one model. None of those produce an
error. All of them produce a plausible number.

**Procedure.**

1. Resolve the lab seed range (`60152–100151` by default) from the plans that declare it.
2. For each source, count manifest directories, parse seeds, and record the seed
   **ranges** — plural, because a source can be non-contiguous.
3. For each declared archive, hash the bytes and compare against the declared digest. A
   missing piece of a reassembled archive is a **failure**, not a reason to score the rest.
4. For every seed claimed by more than one source, hash the **weights inside** the two
   copies and decide: identical bytes ⇒ a re-manifest of one model (warning, still counted
   once); different bytes ⇒ a real clash (failure).
5. For every seed claimed by a non-lab source inside the lab range, **stop**. This is not
   deduped and not scored.
6. Record gaps — a box with no plan, a plan with no results — as named gaps rather than as
   zeroes or as corrupt sources.
7. Record stale copies: a directory that claims a seed already owned by a *selected* source
   is a warning, and the unselected copy is not merged.

**Exit codes:** `0` clean · `2` an archive digest or manifest count disagrees with what was
declared · `3` a non-lab source overlaps the lab range · `4` two selected sources claim one
seed with different weights.

**What actually happened.** 37 sources, **37,686** cross-source seed collisions, 0 fatal.
The 37,686 is mostly one fact discovered four ways: `box1-corpus-v1` and `box1-corpus-v2` are
two exports of one box, and `box-plan7` ships byte-identical copies of box 8's shard and
results. The census resolved them by hashing the weights, not the names.

**The tests that pin it.**

| test | the failure it documents |
|---|---|
| `test_census_does_not_silently_dedupe_a_seed_claimed_by_two_sources` | a glob would merge two claims into one model and the count would still add up |
| `test_census_compares_the_weights_inside_two_archives_instead_of_guessing` | equal *names* are not equal *bytes* |
| `test_census_resolves_two_archives_of_one_box_as_a_re_export` | a legitimate re-export must not be a fatal collision |
| `test_census_records_a_re_manifest_seed_collision_as_a_warning_not_a_failure` | the same bytes twice is not a scientific problem |
| `test_census_treats_a_stale_unselected_copy_as_a_warning_only` | an unselected copy must not be merged |
| `test_census_stops_when_a_non_lab_source_spans_the_lab_seed_range` | two boxes' seeds interleaving is an incident, not an average |
| `test_census_flags_a_single_seed_inside_the_lab_range_too` | one drifted seed is still drift |
| `test_lab_sources_are_allowed_inside_the_lab_range` | the guard must not fire on the lab itself |
| `test_census_fails_on_a_duplicate_seed_inside_one_source` | internal duplication is worse, not better |
| `test_census_fails_when_an_archive_digest_disagrees_with_the_declared_bytes` | a truncated archive still has a plausible manifest count |
| `test_census_fails_a_missing_reassembled_archive_instead_of_scoring_around_it` | scoring the surviving part of an archive changes the population silently |
| `test_census_counts_manifest_dirs_even_when_weights_are_missing` | the count must describe what exists, not what is usable |
| `test_seed_ranges_keep_a_gap_visible_instead_of_reporting_min_to_max` | `min..max` hides the missing 5,000 |
| `test_census_reports_a_box_with_no_results_as_a_gap_not_a_corrupt_source` | absence is not corruption |
| `test_census_reports_lab_boxes_missing_from_the_integration_folder` | the box-2 hole must be named |
| `test_census_sees_lab_box_plans_that_contain_two_shards` | a plan is not necessarily one shard |
| `test_scan_archive_derives_seeds_from_member_names_without_extracting` | extraction is destructive and unnecessary |
| `test_duplicate_values_reports_each_offending_value_once` | a report that repeats a seed 5,000 times is unreadable |
| `test_seed_of_ignores_ids_without_a_seed_suffix` | an unparseable id is not seed 0 |
| `test_render_states_the_verdict_and_the_missing_sources` | a human must be able to read the verdict without the JSON |

---

## 2. Verify — every model against its own pinned plan

**Command:** `python scripts/integration.py verify --source <id>=<corpus> --plan <id>=<shard.json> --quarantine <q> --owned-root <root> --out runs/integration_verify.json`

**What it does.** For each model, recomputes the SHA-256 of `manifest.json` and `weights.npz`
and compares them with the digests the plan pinned for that model.

**Why this way.** Two reasons, and the second one cost a night.

* A filename is not evidence. The plan carries the bytes, so verification is an equality test
  on content — the same test the plan merge used — rather than a directory listing.
* **The validator must be importable before anything can be moved.** Verification was once
  run without the repository on `PYTHONPATH`. The native validator could not be imported, and
  the pipeline recorded *every model as its own failure* — 56,628 of them — then moved all
  56,628 into quarantine, which emptied the corpora two scoring runs were reading and killed
  both mid-flight. Nothing was lost, and the cost was a night. The lesson is a distinction
  the tooling had to learn: **a per-model problem string is the right answer for a bad model
  and the wrong answer for a missing dependency.**

**Procedure.**

1. Import the native validator. If it cannot be imported, report a single **environment**
   problem and move nothing.
2. Load each source's pinned plan (possibly several shard files per source).
3. Walk the source's models. A model whose bytes differ, or which is not in the plan, or
   which is in the plan but not on disk, is a failure attributed to that model.
4. Quarantine moved only for genuine per-model failures, only inside `--owned-root`, and
   never by deleting.

**Exit codes:** `0` clean · `7` at least one problem · `8` usage error.

**What actually happened.** **56,628 / 56,628 verified, 0 failed, 0 moved.** The evidence for
"0 moved" is not a counter in the report — it is the *absence of the quarantine directory*.

**The tests that pin it.**

| test | the failure it documents |
|---|---|
| `test_verify_refuses_to_quarantine_anything_when_the_native_validator_is_missing` | the 56,628-model incident |
| `test_verify_moves_nothing_when_verification_cannot_run` | side-effect guarantee, asserted instead of the message |
| `test_verify_quarantines_a_model_whose_weights_are_not_the_pinned_bytes` | real tampering must still move |
| `test_verify_quarantines_a_model_whose_manifest_is_not_the_pinned_manifest` | the manifest is evidence too |
| `test_verify_never_deletes_a_quarantined_model` | quarantine is reversible by design |
| `test_quarantine_refuses_to_move_outside_the_owned_root` | a verification tool must not reach into someone else's tree |
| `test_verify_does_not_move_models_out_of_a_source_it_does_not_own` | ownership is explicit |
| `test_verify_passes_a_source_that_matches_its_pinned_plan` | the happy path must be silent, not noisy |
| `test_verify_fails_when_a_pinned_plan_entry_has_no_model_on_disk` | a missing model is a failure, not a smaller denominator |
| `test_verify_fails_when_a_model_is_not_in_the_pinned_plan` | an extra model is also a disagreement |
| `test_verify_reports_a_schema_failure_as_a_schema_failure` | two different problems must not share one message |
| `test_verify_counts_a_failed_model_as_excluded_and_says_so` | exclusions must be visible in the count |
| `test_plan_entries_merge_several_shards` | box 1 has two plans and both must be honoured |
| `test_render_verify_shows_counts_and_the_verdict` | a verifier you cannot read is a verifier you will not run |

---

## 3. Merge plans — one plan, one reference, byte-identical shards

**Command:** `python scripts/integration.py merge-plans --source <id>=<shard.json> ... --exclude clean_none_fixed_s5800 --reference clean_none_fixed_s5800 --out <plan>`

**What it does.** Concatenates shard files into one plan, **copies their bytes unchanged**,
and refuses to write anything if the union is not exactly what it claims.

**Why this way.** The plan is the object every later stage checks against. If the merge
re-serialises a shard, then a shard's digest changes and every previously-scored results file
becomes un-verifiable — which is not hypothetical: box 1 ships in a v1 and a v2 export whose
*results* are byte-identical but whose *manifests serialise differently*. `full-resultsv1`
pins to `box-plan1v1`'s shard bytes; `full-results-1v2` pins to `box-plan-1v2`'s. The
preflight **refused the cross-pairing before reading a row**, which is exactly why plans pin
bytes instead of trusting a name. The combination used here is v2-plan with v2-results.

**Refusals.**

* a duplicate detector seed across sources;
* a duplicate shard id;
* a shard whose declared seed bounds disagree with the models in it;
* a shard that still contains the reference model (it would score itself);
* a shard that still contains an excluded model;
* a non-empty output directory (a plan is written once, to a path that is not already a plan).

**Exit codes:** `0` written · `9` refused (`PlanError`) · `8` a `--plan` glob matched nothing.

**The tests that pin it.**

`test_combined_plan_refuses_two_sources_that_share_a_detector_seed` ·
`test_combined_plan_copies_shard_bytes_unchanged` ·
`test_combined_plan_refuses_a_shard_that_still_contains_the_reference_model` ·
`test_combined_plan_refuses_a_shard_that_still_contains_an_excluded_model` ·
`test_combined_plan_refuses_a_duplicate_shard_id` ·
`test_combined_plan_keeps_each_source_its_own_shard_set` ·
`test_combined_plan_refuses_a_shard_whose_bounds_disagree_with_its_models` ·
`test_combined_plan_refuses_a_non_empty_output_directory` ·
`test_combined_plan_records_exclusions` ·
`test_render_plan_lists_every_source_and_the_total` ·
`test_seed_owner_map_uses_the_plan_not_the_seed_range`

---

## 4. Score — classify, never raise

**Command:** `python scripts/integration.py score --shard <shard.json> --corpus <root> --out <results> --reference <ref> --workers 5 [--rewrite-shard]`

**What it does.** Runs the real adapter over a shard, writes a resumable registry, and
classifies every model as `complete`, `incomplete`, or `error`.

**Why this way.** A scorer that raises on an undefined statistic turns one bad model into a
lost corpus. Lab box 5's original run died at row 1,467 with
`ValueError: repository scorer omitted signals: dict_keys(['refdiv_mean_clean'])` and shipped
a registry with 1,467 rows and no results file. The cause, reproduced on this machine, is
narrow: `clean_none_fixed_s81621` is a *clean* model whose CTC statistic is undefined on all
40 held-out images, so the producer emits one key, and an adapter requiring four raises
**inside a worker pool** — taking the rest of the shard with it.

The three-way classification is the fix, and the distinction between the last two is the
subtle part:

* `complete` — four signals defined;
* `incomplete` — undefined on **every** image. A property of the *model*; never retried,
  because re-running it will produce the same absence;
* `error` — the scorer raised. A property of *that run*; retried on resume, because a corpus
  that was mid-move is not a corpus that is wrong.

That last distinction is not academic. Box 5's rescore first ran while the corpus was
mid-move, so all 5,000 rows landed as `error`. Resume treated every row in the registry as
decided, so a second attempt would have scored nothing and reported **0 scored** — a clean
skip of work that had never been done. `error` is retried; `incomplete` is not. The rescore
went on to score 4,999.

**Procedure.**

1. Load the shard and the reference; resolve the adapter (`module:function`).
2. Score in a pool of `--workers`, writing each row as it completes so a crash loses nothing
   already written.
3. Classify each row. A row that raises is recorded with the exception's own message and the
   **model id it belongs to** — the message alone would not say which model.
4. On resume, skip rows that are `complete` or `incomplete` and re-attempt `error`.
5. With `--rewrite-shard`, re-issue the shard listing only the models actually scored, with
   `unscorable_model_ids` naming the rest. The dropped models are named, never subtracted
   silently.

**Exit codes:** `0` at least one model scored · `6` nothing scored · `8` bad adapter spec or
missing reference.

**What actually happened.** The merged pass scored **56,627** models at 5 workers with
`CVIAF_N_EVAL=40`, `CVIAF_BACKGROUNDS=4`, BLAS threads pinned to 1. 6 workers pinned the
16 GB machine.

**The tests that pin it.**

| test | the failure it documents |
|---|---|
| `test_classify_signals_separates_undefined_from_error` | the whole distinction above |
| `test_score_records_classifies_an_incomplete_model_instead_of_raising` | box 5's 5,000 rows |
| `test_score_records_names_the_model_when_the_scorer_raises` | an error without a model id is untraceable |
| `test_score_records_keeps_incomplete_rows_out_of_the_scored_set` | an undefined model is not a scored model |
| `test_score_records_retries_a_previously_errored_model_on_resume` | the resume that would have reported a clean skip |
| `test_score_records_does_not_retry_an_incomplete_model` | determinism: the absence will repeat |
| `test_score_records_resumes_without_rescoring_the_models_it_has` | resume must be cheap |
| `test_score_records_keeps_the_rows_written_before_a_failing_model` | a crash must not cost the shard |
| `test_scored_shard_names_the_models_it_dropped` | no silent denominator |
| `test_scored_shard_with_every_model_scored_is_unchanged_in_membership` | the re-issue must be a no-op when nothing was dropped |

---

## 5. Merge results — preflight every shard, then the ledger and the report

**Command:** `python scripts/integration.py merge-results --plan <plan> --results <results> --out <out> --merge-cmd <chunk_coordinator.py> --repo-root <pinned-scorer>`

**What it does.** Refuses to merge unless every shard's bytes, digest, membership and
settings agree with what the plan says, then runs the project's own evaluator to produce the
ledger and the native report.

**Why this way.** A merged FPR is an average over shards. If one shard was scored against a
different reference, or with different `n_eval`, or against a plan that changed after it was
scored, the average is over two different experiments and **the arithmetic still adds up**.
So the merge is preceded by a preflight that re-hashes each shard's plan file, re-reads each
results file's own report, and compares the reference and scorer settings across shards. A
"stub" mode result is refused outright — a mode that fabricates scores must not be able to
reach the ledger.

**Procedure.**

1. For each shard: does the results file exist? does its `.npz` still match its own report's
   digest? has the shard file changed since it was scored? do the settings agree?
2. Count models in plan, models scored, and the denominators the evaluator will use.
3. Only if the preflight is green, call the evaluator with the plan and the results.
4. Summarise the native report in `runs/merge_report.json` — **including the refusals**.

**Exit codes:** `0` merged (or preflight green and no `--merge-cmd`) · `10` preflight refused.

**Denominators, and the refusal inside the evaluator.** α = 0.05; threshold = the `1 − α`
quantile of the **calibration** negatives; rate = alarms on the **evaluation** half. The split
is a deterministic permutation of the corpus seeded by one integer (default `0`), assigned
**by model, never by score** — splitting on the score would leak the test distribution into
the threshold. Positives and negatives are permuted *independently*, so a half with no
positives reports TPR as **not measured** rather than as zero.

The report emits `status: fpr_only`, `fpr_measured: true`, `positives_measured: false`, and
for every TPR-derived quantity a refusal string — because a `0.000` in those cells would be a
fabricated measurement. That refusal is the correct output for a clean-null population and it
is also the reason §6 below exists.

**The tests that pin it.**

`test_preflight_refuses_a_shard_with_no_results_file` ·
`test_preflight_refuses_an_npz_that_no_longer_matches_its_report` ·
`test_preflight_refuses_a_shard_that_changed_after_it_was_scored` ·
`test_preflight_refuses_shards_that_disagree_about_the_reference` ·
`test_preflight_refuses_shards_that_disagree_about_scorer_settings` ·
`test_preflight_refuses_a_stub_mode_result` ·
`test_preflight_counts_a_complete_merge_and_its_denominators` ·
`test_preflight_reports_the_reference_it_validated` ·
`test_summarise_fpr_report_refuses_to_invent_a_tpr`

---

## 6. Detection at frozen thresholds — the additive pass

**Commands:**

```bash
python scripts/tpr_arms.py build --shard <id>=<shard.json> --corpus <id>=<corpus> \
    --out <arms-dir> --per-shard 9 --magnitude 0.25
python scripts/integration.py score --shard <arms-dir>/plan/shards/tpr_arms.json \
    --corpus <arms-dir> --out <tpr-results> --reference <same-ref> --workers 5 --rewrite-shard
python scripts/tpr_arms.py evaluate --results <tpr-results> \
    --frozen-report runs/merged_fpr_tpr_report.json --out runs/tpr_at_frozen.json
```

**What it does.** Derives attacked variants from clean models **already in the population**,
scores them through the same adapter, and judges them at the thresholds the FPR half froze.

**Why this way — the guardrail.** The clean-only ledger is the FPR basis and it must not
move. The TPR pass is therefore structured so that it *cannot* move it:

* it reads the thresholds **out of the committed report** rather than recomputing them;
* it writes to its own out-directories and its own receipt;
* it never writes to the merged ledger, `runs/merge_report.json`, or any shard of the merged
  plan;
* the threshold used per rule is recorded in the TPR receipt as
  `frozen_from: runs/merged_fpr_tpr_report.json`.

**Why adversarial arms rather than synthetic positives.** A synthetic positive measures the
rule on tensors nothing ever trained. Weight-space tampering of trained models measures it on
the kind of thing that actually reaches a deployed pipeline, and it needs no retraining:
build is 38.6 s for 198 arms and scoring is 19.5 s.

**Procedure.**

1. **Pick parents.** For each shard, take `--per-shard` clean models deterministically.
   Randomly chosen parents would make the arm set unreproducible without shipping the RNG
   state.
2. **Tamper by class.** `weight_tamper` = Gaussian noise on the detection head
   (`TinyDetector.tamper_head`); `substitution` = structural pruning
   (`TinyDetector.tamper_prune`). Two classes because one class cannot distinguish a rule
   that misses weak attacks from a rule that misses everything.
3. **Refuse a no-op.** If the perturbation did not actually change the weights, the arm is
   rejected **before** the expensive measurement, not recorded as an attack. An arm whose
   "attack" is a copy of its parent would be a guaranteed miss attributed to the detector.
4. **Refuse a clean model in the arm set.** An arm directory is validated with the same
   `validate_model_dir` the corpus uses; if a parent leaked into the arm set, evaluating
   would score a null against its own parents and call it detection. The evaluator flags it.
5. **Measure divergence.** Each arm records `behaviour_divergence.f1_relative_drop` and a
   `metrics.attack_success_rate` object, because the manifest validator requires both for a
   weight-space arm. Severity within a class is wide: zero-mean noise on a trained head is
   closer to regularisation than to sabotage, and some `weight_tamper` arms come out with
   *higher* f1 than the parent.
6. **Score exactly as the corpus was scored** — same adapter, same reference, same
   `n_eval`, `backgrounds`, and thread caps.
7. **Evaluate at the frozen thresholds.** Per rule and per class: caught / n, Wilson
   interval, and a `conclusion` field. Below `--min-positives` (default 20) the evaluator
   prints a bound and says so instead of a point estimate.
8. **Degenerate rules are not misses.** A rule whose frozen threshold equals the maximum
   clean score cannot fire on anything: `tpr: null`,
   `status: degenerate_threshold_at_ceiling`, plus an exact one-sided upper bound so a reader
   still gets a usable number.
9. **Name what could not be scored.** Unscorable arms are grouped **by class**. If a class's
   unscorable arms are not spread evenly, that class's rate is measured on a biased subset
   and the receipt says so.

**What actually happened.** 198 arms built (99 + 99), 193 scored, 5 unscorable — **all five
`substitution`**, all with the s81621 signature
(`repository scorer omitted signals: dict_keys(['refdiv_mean_clean'])`). So the most
destructive class is measured on the subset that survived its own attack; that rate is an
upper bound. At the frozen thresholds `refdiv_mean_clean` catches 34.7% [28.4, 41.7] pooled
and `ctc_mean_clean` 3.6% [1.8, 7.3] — at 5.18% and 5.23% FPR respectively. **The best-FPR
rule is not the best-detecting rule.**

**The tests that pin it.**

| test | the failure it documents |
|---|---|
| `test_tpr_uses_the_frozen_threshold_and_never_recalibrates` | the guardrail: the whole point of the pass |
| `test_a_degenerate_rule_is_reported_as_degenerate_not_as_zero` | a rule that cannot fire is not a rule that misses |
| `test_small_denominator_reports_a_bound_and_says_so` | n = 3 is not a rate |
| `test_zero_count_gets_a_positive_upper_bound_from_both_conventions` | 0/193 is a bound |
| `test_wilson_puts_a_zero_count_inside_a_positive_upper_bound` | interval convention |
| `test_evaluate_flags_a_clean_model_inside_the_arm_set` | a null would be scored as an attack |
| `test_a_no_op_tamper_is_refused_rather_than_recorded` | an unattacked arm would be a guaranteed miss |
| `test_quantile_matches_numpy_by_not_interpolating_differently` | the threshold must be the same number the evaluator used |

---

## 7. Statistics — the conventions, stated once

| convention | choice | why |
|---|---|---|
| interval | **Wilson score** | behaves at the boundary; a normal approximation on a 5% rate at n = 28,313 is not the issue, but a 0/193 is |
| zero count | **exact one-sided upper bound** (Clopper–Pearson) | 0 alarms is evidence of a small rate, never of a zero one |
| small *n* | below `min_negatives` / `min_positives` = 20, **no point estimate** | a proportion over 7 models is not a rate |
| threshold | the `1 − α` quantile of the **calibration** negatives | decided without the evaluation half |
| split | deterministic permutation, seeded by one integer (default `0`), **by model** | splitting on the score leaks the test distribution into the threshold |
| split sensitivity | reported as a spread across seeds 0/1/7/42/1337 → `ctc_mean` spans **0.0482–0.0523** | one headline hides the resolution |
| threshold rounding | recorded at full precision (`0.9845092069058692`) | rounding moves a hit count |
| dashboard embedding | **float64** | a float32 round-trip moves the 0.05 quantile to `0.9845092` and can shift the hit count by one |
| multiplicity | 14 per-kind cells, 0 survive Benjamini–Hochberg | a per-kind finding must be reported as unconfirmed |
| BLAS threads | pinned to 1 | reproducibility across machines |

---

## 8. The dashboard — a check, not a poster

`scripts/demo_dashboard.py` writes one self-contained HTML file with all 56,627 score sets
embedded as base64 float64. It is not a picture of the results; it is a second implementation
of the arithmetic, and it is allowed to disagree with the report only by refusing.

* The generator **fails the build** if its recomputed headline disagrees with the committed
  report.
* It embeds float64 because float32 shifts the quantile.
* A rule that cannot fire is labelled **not measurable**, never `0%`.
* If no attack receipt is supplied, the detection panel is omitted entirely rather than drawn
  empty — an empty panel reads as "nothing to report", which is a different claim.
* Unscorable arms are shown grouped by class, above the table.

Tests: `test_dashboard_refuses_to_publish_a_page_that_contradicts_the_report` ·
`test_dashboard_writes_a_self_contained_page_when_the_numbers_agree` ·
`test_score_blob_round_trips_exactly`.

---

## 9. Running the tests

```bash
cd .task3
python -m pytest -q                       # numpy lane: 638 passed, 5 skipped, 1 xfailed
~/.venvs/cviaf-torch/bin/python -m pytest -q   # torch lane: 647 passed, 1 xfailed
```

**Run the two lanes sequentially.** Launched into the same rootdir in parallel they corrupt
each other's cache and one lane reports a fraction of its tests passing — 55 instead of 647 —
**with exit code 0**. A green that green is worse than a red.

Tests are named after the incident they prevent, not after the code path they cover. A test
named after a code path documents the code; a test named after a failure documents the
project.
