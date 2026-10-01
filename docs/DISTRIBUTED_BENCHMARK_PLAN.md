# Distributed benchmark execution plan

Status: staging plan against `c17141f`; freeze against the **integrated** commit only. This document and `scripts/distributed_benchmark.py` do not claim Phase 2 acceptance results.

## Partition and roles

1. Land and review the label-flip, trigger, drift, Ed25519, scale harness and compare/eval fixes. Pin one integrated commit and one lockfile/venv package inventory for all workers. No cherry-picked worker versions. Snapshot a fully resolved plan (`load_plan` merges `default_plan`) with the freeze command below. Do not edit its `plan.json`, `frozen.json`, or individual shard plans after freeze. A new change means a new frozen root.
2. Train independently by **seed**, optionally by attack group (`--groups 2` partitions attacks by their index modulo two). MVP is eight models at seed 5, Day 1 is four seed shards 5, 6, 7, 8, each with eight attacks (32 models). With two groups, Day 1 has eight four-model shards. Every worker gets the same pinned commit and its own shard plan and directory. Never point workers at the same `runs/` tree or concatenate their append-only `registry.jsonl` files.
3. New-lane acceptance tests belong to separate workers: label-flip/trigger efficacy and ASR gate, drift and scale, Ed25519 provenance, and compare/eval regressions. Pin fixture seeds and parameters in each lane's test command or fixture manifest. Do not mix acceptance status into benchmark aggregates. Attack-class-specific corpus subsets are valid training shards, but **not** stand-alone comparable benchmark scores without all controls and global reaggregation.
4. Transfer each whole `shards/<name>/corpus/` directory, including `manifest.json` and `weights.npz` for every model. Store the originating commit, package inventory, stdout/stderr, exit status and SHA-256 tarball checksum with each payload. Transfer outside the disposable VM after every completed model or short batch; keep a second copy. A directory or model that did not complete is an incomplete shard, not a zero result.
5. The merger validates the effective-plan hash, current `build_specs` digests, expected ID partition and completeness, per-model seed registry, attack kind, semantic model weights digest, and SHA-256 transport checksums after copying. It rebuilds one sorted registry using merged absolute/relative paths as stored; invoke from the same root where it will be evaluated, or regenerate registry paths if relocating. The merger refuses existing outputs. Publish `merged/inventory.json` and full corpus only on zero exit. Failed, missing, extra, stale or duplicate models block publication; rerun only affected shards from the same commit/plan.
6. Run whole-corpus final eval and compare **on separate workers after merge**, each with a full verified merged corpus copy. These can run in parallel with independent output paths. MVP: compare and eval at seed 1, alpha .05, eight backgrounds. Day 1: eval at seed 1 and the same detector parameters; optional Day 1 compare gets its own worker. Store full JSON plus logs and their hashes. Run same command on a second worker for a deterministic spot-check of selected outputs, recording any platform-dependent deviations. Keep `null_control`, `asset_level` and full per-model rows, not just rounded summary tables.

## Why JSON summaries cannot simply be concatenated

`evaluate_corpus` computes per-model scores but also chooses a clean reference from the registry, computes null controls across clean models, asset-level controls across clean and attacked models, and attack summaries from all rows. `compare_corpus` pools a clean reference, places corpus-wide baseline thresholds, and builds scores/verdicts over all rows; its data-axis reference pool is large and should not be repeated with incompatible seeds. Per-shard `eval.json` or `comparison.json` is only a smoke test. Its `summary`, scores, FDR, power, FPR, null controls, provenance axis and final verdict are **not** mathematically additive. Do not average shard AUROCs or mean seed metrics and label it a full run. Merge the model artifacts then recompute final JSON over the complete registry. The eval function's optional `models` filter does not partition its `null_control` and `asset_level` computations, so using it as a JSON-sharding protocol is unsafe.

## Commands (from repo root)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e '.[full,dev]'
.venv/bin/python -m cviaf.lab doctor
# Pin actual integrated SHA, not c17141f once patches land.
COMMIT=$(git rev-parse HEAD)
.venv/bin/python scripts/distributed_benchmark.py freeze --plan configs/corpus_mvp.json --out /safe/mvp-frozen --source-commit "$COMMIT"
.venv/bin/python scripts/distributed_benchmark.py freeze --plan configs/corpus_day1.json --out /safe/day1-frozen --source-commit "$COMMIT"
# On each worker with the frozen root mirrored at the same path:
cd /path/to/pinned/CVIAF
.venv/bin/python -m cviaf.lab corpus --plan /safe/day1-frozen/plans/seed-5-group-0.json --out /safe/day1-frozen/shards/seed-5-group-0/corpus
# Keep long jobs off the 120-second foreground boundary:
nohup .venv/bin/python -u -m cviaf.lab corpus --plan /safe/day1-frozen/plans/seed-6-group-0.json --out /safe/day1-frozen/shards/seed-6-group-0/corpus > /safe/day1-frozen/seed-6.log 2>&1 < /dev/null & echo $! > /safe/day1-frozen/seed-6.pid
# Copy shard directories back to coordinator; there, after all are present:
.venv/bin/python scripts/distributed_benchmark.py verify --root /safe/day1-frozen --source-commit "$COMMIT" --merge
nohup .venv/bin/python -u -m cviaf.lab eval --corpus /safe/day1-frozen/merged --seed 1 --alpha 0.05 --backgrounds 8 --ftc-stride 8 --ftc-decoy-class 0 --json /safe/day1-eval.json > /safe/day1-eval.log 2>&1 < /dev/null &
nohup .venv/bin/python -u -m cviaf.lab compare --corpus /safe/mvp-frozen/merged --seed 1 --alpha 0.05 --backgrounds 8 --json /safe/mvp-compare.json > /safe/mvp-compare.log 2>&1 < /dev/null &
```

The first foreground worker command is shown only for clarity; use the background form for heavy runs. Each worker installs its own venv. The base sandbox may lack `torch`, `onnxruntime`, `sklearn`; install optional dependencies actually used by acceptance lanes, inspect `doctor`, and preserve a `pip freeze` and interpreter/platform report. Keep logs and process status outside the repository; after a VM reset, inspect artifacts before resuming. A `manifest.json` alone is not a complete model; the corpus runner's resume logic checks only spec digest, so delete/retrain a missing or corrupt weights file rather than trusting resume. Avoid concurrent writes to a shared corpus, and do not silently treat a short budget or exit 0 as completeness. The merge is the completeness gate.

## Frozen seed registry and release gate

The seed list and order come from the resolved plan; each model's scene seed is `plan.scene.seed + seed`, detector seed is `seed`, attack seed is `plan.attack_seed` (default 11). Benchmark evaluation seed defaults to 1 but is pinned explicitly. Freeze records every model ID, expected spec digest and all three seeds. Acceptance lanes must separately record their test-specific RNG seeds and fixture checksums. A result is releasable only after: all expected models and controls exist, full artifacts verify, a clean merged registry is built, the final whole-corpus compare/eval JSON was generated against that registry at the pinned commit, and CI/acceptance logs have no unresolved failures. Record weak-backdoor/model-effect exclusions as exclusions, not successful detections. Never substitute old committed `runs/` results for Phase 2 measurements.
