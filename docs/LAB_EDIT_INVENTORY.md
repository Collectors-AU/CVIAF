# Merge inventory: my uncommitted lab edits vs the pending v4 package (Task 4 prep)

Status: **prep only.** The lab edits below are deliberately **not committed** — they stay in the
working tree until the consolidated v4 package lands, so the merge test can be run on a clean
`c17141f` clone and every conflict reported as `file:hunk`. Nothing here is resolved solo; the
rows marked SEMANTIC need a decision that is not mine.

Base for the merge test: `git clone` → `c17141f` → apply the three package files in order
`CVIAF_V4_CONSOLIDATED` → `CVIAF_V4_APPENDIX_DE` → `CVIAF_V4_LABELGATE_FOLLOWUP`, then merge the
edits below on top.

Pre-verified on 2026-09-28: all three files **apply cleanly to a clean `c17141f` checkout**
(`git apply --check` then `git apply`, no fuzz, no rejects). The check was run in a throwaway
clone; this working tree was not touched.

## A. My uncommitted tracked edits (41/20/20/15/7/3/3/1 hunks)

| file | hunks | regions (old→new) | patch touches this file? |
|---|---:|---|---|
| `cviaf/lab/evaluate.py` | 41 | imports `@@ -44..-63`; `scores_for_model @@ -127`; `evaluate_corpus @@ -239..-289`; `asset_level_detectors @@ -289..-400` (26 hunks); `weight_score_of @@ -400..-406`; `null_control @@ -476..-502`; `_summarise @@ -569..-582` | **no** |
| `cviaf/lab/train.py` | 20 | `_asr_single @@ -304..-376`; `_behaviour_divergence @@ -419..-434`; `train_model @@ -454..-542` | **no** |
| `cviaf/lab/corpus.py` | 20 | `build_specs @@ -97`; `retest_corpus @@ -140..-205`; `run_corpus @@ -229..-272` | **no** |
| `cviaf/lab/poison.py` | 15 | `AttackSpec @@ -109`; `_marker_patch @@ -138`; `_stamp_frame @@ -167..-169`; `apply_trigger @@ -196..-238`; `trigger_view @@ -448..-466` | **yes** |
| `cviaf/lab/cli.py` | 7 | `_cmd_corpus @@ -165`; `_cmd_eval @@ -226..-228`; `_cmd_compare @@ -327`; `main @@ -413, -443, -469` | **yes** |
| `cviaf/lab/detectors.py` | 3 | `trace_ftc @@ -297..-299`; `reference_divergence @@ -360` | **yes** |
| `scripts/tamper_probe.py` | 3 | `main @@ -187..-211` (+ `runs/tamper_probe.json`) | no |
| `README.md` | 1 | `@@ -224,0 +225 @@` (one line, Known issues area) | **yes** |

Three tracked `__pycache__/*.pyc` files also show as modified: machine-generated, not an edit, no
patch overlap. Leave them alone.

## B. Textual overlap — where a conflict is expected

| id | local hunk | patch hunk (file) | prediction |
|---|---|---|---|
| B1 | `cviaf/lab/poison.py:@@ -109,0 +110,10 @@ class AttackSpec` | `poison.py:@@ -104,10 +104,24 @@ class AttackSpec` (consolidated) | **CONFLICT** — same region, and both add a field |
| B2 | `cviaf/lab/cli.py:@@ -469,0 +565,32 @@ def main` | `cli.py:@@ -474,9 +496,60 @@ def main` (consolidated) + `labelgate @@ -470,8 +483,16 @@` | **CONFLICT** — same subparser block in `main()` |
| B3 | `cviaf/lab/cli.py:@@ -327,0 +338,77 @@ def _cmd_compare` | `labelgate @@ -305,9 +305,14 @@` and `@@ -325,6 +330,14 @@ def _cmd_compare` | **LIKELY CONFLICT** — adjacent hunks in the same function |
| B4 | `cviaf/lab/detectors.py:@@ -297..-360` | `detectors.py:@@ -610..-637` | none — disjoint regions |
| B5 | `README.md:@@ -224,0 +225 @@` | `README.md:@@ -215,7 +215,7 @@` | adjacent — the patch rewrites the Known-issues bullet list my line sits in |
| B6 | `cviaf/lab/evaluate.py`, `train.py`, `corpus.py` (81 hunks) | — | none textual: the package contains **no** diff for these three files |

## C. Semantic overlaps — no textual conflict, but they need a decision

**C1 (highest) — `AttackSpec` field divergence breaks both sides' manifests.**
My tree's `AttackSpec` adds `mechanism` (`poison.py:@@ -109,0 +110,10 @@`). The v4 patch adds
`scope` (`poison.py:@@ -104,10 +104,24 @@`). Neither has both. Measured consequence: running
the patched tree against my committed corpora fails with
`TypeError: AttackSpec.__init__() got an unexpected keyword argument 'mechanism'` from
`evaluate.train_spec_from_manifest` — because my `clean_null` and `stampfree` manifests were
written by a tree whose `AttackSpec` has that field, and `train_model` records it. The mirror
case is v4-produced manifests (which carry `scope`) hitting my tree. **The merged dataclass has
to be the union, or every committed corpus breaks on one of the two trees.** My Task-2 null-suite
run used a shim (adding `mechanism` to the clone's dataclass) and nothing in this repo changed.

**C2 — the FTC drop is not in the package.** `FUSED_DETECTORS` appears **0 times** in all three
files, and none of them contains a `cviaf/lab/evaluate.py` diff at all. My uncommitted
`evaluate.py` defines `FUSED_DETECTORS = ("ctc", "refdiv", "ftc")` at `@@ -63,0 +78,22 @@` and
uses it in the fused score path. Meanwhile v4's own `model_asset_rule.py` sets
`DEFAULT_SIGNALS = ("ctc", "refdiv")` with the comment "FTC failed factorial controls; optional
only". So after the merge the tree would hold **two contradictory FTC policies** — mine includes
FTC in the fused detector set, the package's asset rule deliberately excludes it. Not resolved
here; flagged with both citations.

**C3 — `ctc` is not exchangeable across seed ranges, and v4's asset rule defaults to it.**
Measured in `docs/CLEAN_NULL_CORPUS.md`: the ctc stamp response is not exchangeable between the
existing clean population (seeds 5–24) and the new one (seeds 100–149) — two-sample KS
p = 0.0040 vs day1 and p = 0.0316 vs mvp2, with a real seed trend across all 81 clean models
(Spearman ρ = −0.290, p = 0.0087). `refdiv` is exchangeable (KS p = 0.972 / 0.997).
`model_asset_rule.decide_model_asset` pools `("ctc", "refdiv")` across clean models, so a ctc
column calibrated across corpora is not an exchangeable null. This is a substantive conflict
between a v4 default and a measurement, not a code conflict.

**C4 — manifest schema is shared state.** `train_model` (my uncommitted hunks) writes
`ground_truth.mechanism` and `spec.attack.mechanism`; the package's `null_suite.py`,
`monitor_experiment.py`, `arm_b.py` and `attribute.py` all read manifests back through
`train_spec_from_manifest`. Any field added on one side has to be readable on the other (same
root cause as C1).

**C5 — `Splits` field names are consumed by the package.** My `train.py` hunks touch
`attack_success_rate`, `_asr_single` and `_behaviour_divergence`; the package imports
`build_splits`, `ModelArtifact`, `attack_success_rate`, `detection_quality` and reads
`splits.train` / `splits.train_poisoned`. No textual overlap, but the signatures and the
`asr`/`asr_net` keys in the returned dict are load-bearing for `null_suite.behavioral_probe`.

## D. File ownership notes

* The package's new modules I must **not** shadow: `cviaf/lab/{null_suite,model_asset_rule,label_consistency,label_flip_signal,attribute,arm_b,arm_c_measure,residual,evalues,fusion,review,clean_battery,driftbench,monitor_experiment,patch_local,trigger_global,baseline,evasion}.py`, `cviaf/drift/attribution.py`, `cviaf/provenance/attestation.py`, `cviaf/schema_v3/*`, `cviaf/v4_assurance/*`, `scale_harness/*`.
* `cviaf/lab/compare.py`: the package edits it, I have **no** uncommitted edit → keep the package's version.
* Untracked but mine, to keep out of the merge until decisions are made: `cviaf/lab/{arms,power,redteam}.py`, `cviaf/research/`, `tests/test_arms.py`, `tests/test_research.py`, `configs/corpus_*.json`, `runs/lanes*/`, `runs/trigger2/`, `docs/{EXPERIMENT_LANES,LANE_MODEL_TAMPER,MEASUREMENT_NOTES}.md`, `scripts/recipe_null_sweep.py`.
* Already committed and pushed by me (not part of the merge risk): `runs/clean_null/**`, `runs/stampfree/**`, `cviaf/lab/stampfree.py`, `tests/test_stampfree.py`, `scripts/clean_null_exchangeability.py`, `docs/CLEAN_NULL_CORPUS.md`, `docs/STAMPFREE_BACKDOOR.md`, `STATE.md`.

## E. Merge-test procedure when the package is final

1. Fresh clone of the remote at the pinned base; `git apply --check` all three files, then apply.
2. `pytest -q` on the clean patched base; record the count and any `xfail`.
3. Bring my working-tree edits in one file at a time (`git stash`-free: apply the local diff with
   `git diff -- <file> | git apply -3`), stopping at the first conflict in each file.
4. For every conflict: record `file:hunk`, the two sides' intent in one line each, and whether the
   resolution requires a decision (B1, B2, B3, C1, C2, C3). **Report; do not resolve solo.**
5. Re-run `pytest -q` plus the Task-1 and Task-2 harnesses
   (`scripts/clean_null_exchangeability.py`, `python -m cviaf.lab.stampfree --seeds 100 101 102`)
   against the merged tree, and confirm the committed corpora still load — that is the check that
   would have caught C1.
