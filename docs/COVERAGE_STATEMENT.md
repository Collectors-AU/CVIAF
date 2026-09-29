# Coverage statement — PS 26228

**Generated, not written by hand.** Every row below is resolved from an artefact on
disk by `cviaf/lab/coverage.py`. A clause reads `measured` only when a number with a
denominator was read out of a file; `unmeasured` means the code or corpus exists but
nothing in it is a number yet — which is where a hand-written trace drifts into
optimism. Regenerate with:

```bash
cd .task3 && PYTHONPATH=. <venv>/bin/python -m cviaf.lab.coverage --markdown docs/COVERAGE_STATEMENT.md
```

**States:** measured=14

## Clauses

| clause | what it requires | state | measured value | artefact / why not |
|---|---|---|---|---|
| 1.4 | Risk, not just integrity: expected loss per disposition | measured | 4 | `runs/fpr_ledger_report.json` — clause 1.4 priced 4 rule(s) in runs/fpr_ledger_report.json: ['ctc_mean_clean', 'ctc_peak_clean', 'ctc_q95_clean', 'refdiv_mean_clean'] |
| 2.2.1-a | Five data-attack families, each with a denominator | measured | 7 | `runs/fpr_ledger_report.json` — 7 kinds with an evaluation denominator in runs/fpr_ledger_report.json: ['dup_flood', 'gma', 'label_flip', 'oda', 'oga', 'ood_insert', 'rma'] |
| 2.2.1-b | Sample evidence aggregated to contributor/source risk | measured | 30 | `runs/arm_b/arm_b_u3.json` — 30 contributor-level cell(s) in runs/arm_b/arm_b_u3.json (malicious contributor 'vendor_x') |
| 2.2.2-a | Model integrity against a defined reference battery | measured | 4 | `runs/fpr_ledger_report.json` — 4 rule(s) with an FPR denominator in runs/fpr_ledger_report.json |
| 2.2.3 | Inference provenance: hash-chained, replay-checked records | measured | 9 | `runs/provenance_ledger.jsonl` — 9 chained entries verify (seq + prev_hash) in ./runs/provenance_ledger.jsonl |
| 2.2.4 | Distribution shift vs manipulation, with under-determined | measured | 8 | `runs/drift_cells/harness_report.json` — 8 drift decision(s) and 32 under-determined across 4 metrics in runs/drift_cells/harness_report.json |
| 2.2.5-a | Findings carry reason, evidence, asset and disposition | measured | 13 | `runs/demo_assurance/assurance_report.json` — 13/15 findings carry a reason, evidence, affected asset and disposition in runs/demo_assurance/assurance_report.json |
| 2.2.5-b | Supported classes and unsupported conditions declared | measured | 27 | `runs/demo_assurance/assurance_report.json` — declares 27 coverage entr(ies) {'supported_attack_classes': 14, 'unsupported_conditions': 8, 'assumptions': 5} and 7 limitation(s) in runs/demo_assurance/assurance_report.json |
| 2.2.6-a | Offline / air-gapped: no network imports | measured | 0 | `check:offline` — AST scan of cviaf/: no network import in the assurance path; 1 fetch-once call site(s) in corpus construction (['./cviaf/lab/cifar.py']), guarded by CVIAF_OFFLINE |
| 2.2.6-b | Ingest COCO and YOLO dataset formats | measured | 2 | `runs/format_ingest.json` — runs/format_ingest.json: ingested ['coco', 'yolo'] with box counts {'coco': 3, 'yolo': 3} |
| 2.2.6-c | Reference model formats: ONNX and PyTorch/TorchScript | measured | 3 | `runs/format_ingest.json` — runs/format_ingest.json: loaded ['onnx', 'pytorch', 'torchscript'] (onnx forward [1, 64, 16, 16]) |
| 2.3-a | Reproducible audit log of the assurance runs | measured | 9 | `runs/provenance_ledger.jsonl` — 9 chained entries verify (seq + prev_hash) in ./runs/provenance_ledger.jsonl |
| 2.3-b | Assurance-report schema shipped and validated | measured | 27 | `runs/demo_assurance/assurance_report.json` — declares 27 coverage entr(ies) {'supported_attack_classes': 14, 'unsupported_conditions': 8, 'assumptions': 5} and 7 limitation(s) in runs/demo_assurance/assurance_report.json |
| 2.3-c | Coverage statement naming supported classes and limits | measured | 6599 | `docs/COVERAGE_STATEMENT.md` — this statement is generated from live artefacts |

## Ready-for-corpus gate

What must be true before an FPR run on the 20k fleet is worth reading.

| item | what it requires | state | detail |
|---|---|---|---|
| population_validated | the fleet census reports no duplicate ids, specs or weights | pass | 8064 models, ready_for_fpr=True, 0 problem(s) |
| manifests_clean | every manifest validates against the locked schema | pass | 8310 manifests validated, 0 problem(s) |
| ledger_valid | the FPR ledger validates (kinds declared, no partial signal coverage) | pass | 132 records, 0 problem(s) |
| negatives_on_both_sides | >=20 calibration and >=20 evaluation negatives (interval width) | pass | calibration_negatives=30, evaluation_negatives=30 |
| a_positive_kind_measured | at least one attacked kind has a denominator, or the TPR is vacuous | pass | 7 kinds with an evaluation denominator in runs/fpr_ledger_report.json: ['dup_flood', 'gma', 'label_flip', 'oda', 'oga', 'ood_insert', 'rma'] |
| fleet_one_population | the fleet's negatives are exchangeable: shards and splits agree on the FPR | pass | 7503 records on 6 shard(s); one_population=True, split_stable=True, 1 warning(s) |
| audit_trail_verifies | the provenance ledger hash chain verifies over its artefacts | pass | runs/provenance_ledger.jsonl: 9 entries, chain intact, 13 artefact(s) re-hashed, 9 superseded, 2 lineage-only (derived from this trail) |
| offline | no network imports anywhere in the package | pass | assurance path is import-clean; 1 guarded fetch site(s) in corpus construction |

**Gate: PASS**

## Known limitations (mirror of the gate failures and unmeasured clauses)

Anything not `measured` above is a limitation of the current evidence, not of
the design. The three that constrain every number in this repo:

1. **FPR/TPR numbers are single-population.** Calibration and evaluation halves
   are drawn from the same clean fleet, so the measured FPR is an average over
   one population; the reference-relative signals shift by a large amount when
   the reference model changes.
2. **Denominators are small in the tamper corpora.** A TPR over 8 or 16 arms
   carries a Wilson interval ~0.19-0.24 wide; the report prints the interval
   and refuses a bare rate below the declared floor.
3. **Drift axes are near the noise floor at n=160.** Per the drift harness,
   most declared cells are `under_determined` rather than `drift`, and that is
   reported as such.

_Produced by `python -m cviaf.lab.coverage` on live artefacts; do not edit by hand._
