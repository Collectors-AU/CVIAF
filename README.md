# CVIAF: Computer Vision Integrity Assurance Framework

An air-gapped assurance engine that evaluates training data, model, and inference integrity across multi-contributor computer vision pipelines. Built for SIH problem statement 26228.

Current version: 2.0.0.

## Status

Working and verified in this environment:

- `cviaf demo` runs end to end on synthetic data, injects four attack scenarios, and writes a complete assurance report. A clean run finishes in about 1.5 seconds and returns `CRITICAL` / `QUARANTINE` with 23 findings across all four assessment modules.
- All four assessment modules have real implementations with no stub returns.
- The CLI implements `demo`, `assess`, `verify-audit`, `verify-seal`, and `schema`.
- The governance layer emits a coverage declaration with 14 supported attack classes, 8 declared out-of-scope conditions, and 5 stated assumptions.

## The lab (`cviaf.lab`) — an MVP that runs offline on a laptop

`cviaf/` is the assurance *engine*. `cviaf/lab/` is the *laboratory* that manufactures assets with known ground truth so the engine's claims can be scored rather than asserted. It runs on a MacBook Air M3 with numpy only: no torch, no downloaded dataset, no network.

```bash
uv pip install --python .venv/bin/python numpy scipy scikit-learn pillow pytest
.venv/bin/python -m cviaf.lab doctor
.venv/bin/python -m cviaf.lab corpus --plan configs/corpus_mvp.json   # ~1 min for 8 models
.venv/bin/python -m cviaf.lab eval   --corpus runs/mvp
```

To train more models in the background all day, run the loop instead — it keeps adding fresh
seeds in cycles, finishing any seed an interrupt left half-done, and writes a heartbeat so you
can tell a working job from a dead one:

```bash
nohup ./scripts/run_day1_loop.sh > logs/day1_loop.log 2>&1 &
tail -f logs/day1_loop.log; cat runs/day1/loop_status.json
```

Measured on the MVP corpus (one seed, `runs/mvp`):

| attack | mean ASR | CTC | reference-divergence | fused | TPR@5% FPR |
|---|---|---|---|---|---|
| clean *(control)* | 0.00 | 0.500 | 0.500 | 0.500 | 0.062 |
| oga — object fabrication | 1.00 | 0.819 | 0.983 | **0.987** | **0.963** |
| oda — object disappearance | 0.99 | 0.441 *(blind)* | 0.949 | **0.906** | **0.512** |

Two results carry the design. The control sits at exactly 0.500, so the framework does not alarm on clean models. And the two detectors are **complementary**: CTC is structurally blind to cloaking and reference-divergence catches it, while reference-divergence is weak on fabrication and CTC catches that. Either one alone leaves a blind spot; the fused result has neither.

Models whose measured attack success rate falls below the floor are flagged `[WEAK]` and **excluded from detector scoring** — a detector cannot detect a backdoor that was never implanted. On this corpus that gate correctly caught `gma` (a global-effect attack a fully-convolutional backbone cannot express).

23 regression tests cover the properties the design claims rather than merely that the code runs
(`.venv/bin/python -m pytest tests/ -q`, ~4 s): determinism, conformal validity under the null,
FDR control, attack ground truth, the ASR gate, and the background loop's seed bookkeeping.

`docs/MVP_MAC.md` is the operating guide: how to read the results table, how to improve the corpus, the two design failures encountered on the way, and the all-day background loop. `docs/SCALING_PLAN.md` is the M3 → H200 plan.

Not done yet:

- `assess` has not been run against a real dataset or a real model. Against synthetic inputs it only exercises the data integrity path.
- The engine's v3 redesign (`docs/CVIAF_V3_ARCHITECTURE.md`) is not implemented yet: conformal calibration, the assurance battery, the planner and standards-based attestation exist as design plus, in the lab, as calibration code. The four engine modules are still the v2 implementations.
- The lab has no torch path yet. `docs/SCALING_PLAN.md` §2 defines the interface a real backbone must satisfy.

## v3 design (not yet implemented)

`docs/` now holds the redesign and its supporting artifacts. The headline change is a shift from "a detector ensemble that emits verdicts" to "a threat-model-conditional assurance engine that emits warranted beliefs" — every result carrying its assumptions, a conformal-calibrated confidence, a measured detection floor, and the fraction of the applicable reference battery that actually ran.

- `docs/CVIAF_V3_ARCHITECTURE.md` — the design: twelve axioms each with a rejected alternative, the Assurance Battery, the enrollment lifecycle, detector plugin/planner architecture, the calibration mathematics, schema v3, threat model, roadmap.
- `docs/PS26228_REQUIREMENT_TRACE.md` — every clause of the problem statement mapped to a design decision, an artifact and an acceptance test, with honest status.
- `docs/DEEP_RESEARCH_SKILL.md` — the reusable research protocol (source quality table, verification tags, checkpoint discipline).
- `docs/source_urls.txt` and `scripts/verify_sources.sh` — the primary-source list and a script that reproduces the citation ledger with status, final URL and content hash.

The three load-bearing ideas: every threshold is derived from a signed reference battery rather than hand-tuned; `accept` is forbidden whenever applicable checks did not run; and detection floors are published, so the report can say "we have no power below 1.5% poisoning here" instead of "clean".

## Architecture

Four offline verification modules feed a governance layer through `cviaf/orchestrator.py`.

1. Training data integrity (`cviaf/data_integrity/`). Covers five classes of training data attack: trigger and backdoor injection with BadNets and blended triggers, label flipping, near-duplicate flooding, out-of-distribution insertion, and a combined assessment. Detection uses spectral signatures, confident learning for label noise, duplicate clustering, and covariance or isolation-forest outlier scoring. Depends on numpy and scikit-learn only.

2. Model integrity (`cviaf/model_integrity/`). Runs Neural Cleanse style trigger reverse engineering, activation and weight statistics, behavioural fingerprinting, and output entropy analysis. Supports white-box and black-box access, and degrades cleanly when weights are unavailable.

3. Inference provenance (`cviaf/provenance/`). Binds the image hash, model digest, preprocessing config, output tensor, nonce, and sequence number into a signed seal, with a hash-chain audit trail and a Merkle tree for batch checks. Designed for air-gapped use, so there is no cloud key management dependency.

4. Distribution shift (`cviaf/drift/`). Mahalanobis distance against a reference distribution, maximum mean discrepancy, per-feature Kolmogorov-Smirnov tests, and maximum softmax probability. Separates ordinary operational drift from adversarial manipulation.

The governance layer (`cviaf/governance/`) merges module output into one report carrying severity, confidence, evidence, affected assets, remediation text, and a disposition of `accept`, `review`, or `quarantine`.

## Requirements

Python 3.10 or newer.

Core dependencies, installed in the checked-in `.venv` on Python 3.11:

- numpy
- scikit-learn

The demo also uses scipy for KS tests when it is present.

Optional dependencies, needed to assess real data:

- `cryptography` for Ed25519 signing
- `torch`, `onnx`, `onnxruntime` for model loading
- `Pillow` for reading image files
- `PyYAML` for YOLO `data.yaml` class names

The optional set is defined as the `full` extra in `pyproject.toml`.

## Setup

With uv, which is what `uv.lock` is for:

```bash
uv venv
uv pip install -e ".[dev]"
uv pip install -e ".[full]"    # optional, for real datasets and models
```

With plain pip:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

To skip installation entirely, set the import path when you run commands:

```bash
PYTHONPATH=. python -m cviaf demo
```

## Running the demo

The demo needs no dataset, no model, and no network access.

```bash
python -m cviaf demo
```

With the package installed in editable mode, the console script is equivalent:

```bash
cviaf demo
```

Send output somewhere other than the default `cviaf_demo_output`:

```bash
python -m cviaf demo --output /tmp/cviaf_run
```

### What the demo does

1. Generates 300 synthetic samples across 5 classes from 3 contributors.
2. Injects four attacks: BadNets patch triggers at a 10 percent poison rate, label flipping in `contributor_1`, 40 near-duplicates from a malicious vendor, and 20 out-of-distribution samples.
3. Recomputes features and hashes over the poisoned set.
4. Creates 10 provenance seals, tampers with seal 3, and forges seal 7.
5. Runs the full pipeline and writes the report.

A healthy run prints an overall risk of `CRITICAL`, a disposition of `quarantine`, 23 findings, and an intact audit trail. The provenance section reports every seal as invalid rather than only the 2 tampered ones. That is the key bug described under Known issues, not a sign that the run failed.

## Assessing real data

```bash
python -m cviaf assess \
  --dataset ./data/instances.json \
  --model ./models/detector.onnx \
  --format coco \
  --access-level white-box \
  --output ./out
```

Options:

- `--dataset`, `-d` path to a COCO JSON file or a YOLO directory
- `--model`, `-m` path to an `.onnx`, `.pt`, or `.pth` file
- `--format`, `-f` one of `coco`, `yolo`, `auto`
- `--access-level`, `-a` one of `white-box`, `black-box`
- `--skip` comma separated modules to skip, from `data`, `model`, `provenance`, `drift`
- `--pipeline-id` identifier recorded in the report
- `--contributor` dataset contributor name

This command needs the `full` extra to be useful. Without Pillow the dataset loader substitutes random placeholder images, and without torch or onnxruntime the model fails to load. Both cases only print a warning, so check the report for which modules actually ran. Running `assess` with no arguments at all still exits 0 with an empty `LOW` / `accept` report.

## Verifying artifacts

```bash
python -m cviaf verify-audit ./cviaf_demo_output/audit_trail.json
python -m cviaf verify-seal ./seals.json --key-dir ./cviaf_demo_output/keys
python -m cviaf schema > assurance_schema.json
```

`verify-audit` recomputes the hash chain and exits non-zero if any entry breaks. Against the demo output it reports 10 entries and an intact chain.

## Output artifacts

The `cviaf_demo_output/` and `demo-output/` directories hold reports from earlier runs.

- `assurance_report.json` is the full report: findings, per-module assessments, the coverage statement, limitations, and the audit trail.
- `audit_trail.json` is the standalone hash-chained action log.
- `keys/` holds the signing keypairs used for that run. Key files are gitignored; see `demo-output/keys/README.md` for how to restore the historical demo keys locally.

## Repository layout

```
cviaf/
  cli.py              command line entry point and demo driver
  orchestrator.py     runs the four modules and calls the governance layer
  core/types.py       shared enums, dataclasses, and hashing helpers
  data_integrity/     training data assessment
  model_integrity/    model assessment and backdoor detection
  provenance/         seals, key management, and the audit chain
  drift/              distribution shift assessment
  governance/         report assembly and the coverage statement
  formats/            COCO and YOLO dataset loaders, ONNX and PyTorch model loaders
  attacks/            attack generators used by the demo
  utils/              feature extraction and hashing helpers
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
