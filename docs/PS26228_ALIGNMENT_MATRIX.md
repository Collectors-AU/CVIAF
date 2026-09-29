# PS 26228 → As-Built Alignment Matrix

**What this is.** The problem statement's five capabilities and five constraints, each mapped to
the artifact in *this* worktree that implements it, the receipt that measures it, and the test or
gate that keeps it honest. Written against the code as it exists on `task3-real-backbone` at
`d04f1d5`, not against a design intent.

**What this is not.** [`PS26228_REQUIREMENT_TRACE.md`](PS26228_REQUIREMENT_TRACE.md) is the v3
*design* trace: it maps clauses to a planned module layout (`cviaf/planner.py`,
`cviaf/assurance/`, `cviaf/data_integrity/label.py`) of which parts were never built, and it
marks most clauses 🟠 *planned*. It is a useful record of intent and a misleading record of
capability. This document supersedes it as the answer to "show me where you handled X."

**Verdict vocabulary.** `SATISFIED` = implemented, measured by a committed receipt, and checked by
a test or the coverage gate. `PARTIAL` = implemented but the evidence does not carry the weight
the clause implies, with the shortfall named. `ABSENT` = not implemented.

**One thing to read before the tables.** Two populations live in this repo and they must never be
multiplied. The **framework clauses** are measured on the lab corpus (`runs/fpr_ledger_report.json`,
`runs/fpr_ledger.json`, 8,064 models censused, 132 ledger records, 30/30 calibration negatives).
The **headline false-alarm rate** is measured on the merged corpus (`runs/merged_fpr_tpr_report.json`,
56,627 models, 28,314 / 28,313 halves). The clause gate does not run on the merged corpus; the
merged pass does not feed the gate. Both are real; neither is the other.

---

## 1. The five capabilities

| # | PS capability | Verdict | Where it lives | What is weak, named |
|---|---|---|---|---|
| C1 | **Training-data integrity** — trigger injection, label flipping, near-duplicates, OOD insertion, source-level risk aggregation | **SATISFIED** | [`cviaf/data_integrity/__init__.py`](../cviaf/data_integrity/__init__.py) (1,323 lines, the five families); [`cviaf/lab/poison.py`](../cviaf/lab/poison.py) (reproducible injection with declared ground truth); [`cviaf/lab/stampfree.py`](../cviaf/lab/stampfree.py) (694 lines, label-space poisoning keyed on a natural scene condition); [`cviaf/lab/label_flip_signal.py`](../cviaf/lab/label_flip_signal.py); [`cviaf/lab/risk.py`](../cviaf/lab/risk.py) | Per-family **power is small**: the gate records **7 kinds with an evaluation denominator** in `runs/fpr_ledger_report.json`, and the coverage statement itself declares that a TPR over 8–16 arms carries a Wilson interval **~0.19–0.24 wide**. The families are implemented and measured; the *strength* of each measurement is thin. |
| C2 | **Model integrity** — substituted / backdoor-like behaviour, stated access assumptions, confidence, limitations | **SATISFIED** | [`cviaf/lab/detector.py`](../cviaf/lab/detector.py) (`tamper_head`, `tamper_prune`, `tamper_bias`); [`scripts/tpr_arms.py`](../scripts/tpr_arms.py); receipts [`runs/tpr_ladder_at_frozen.json`](../runs/tpr_ladder_at_frozen.json), [`runs/merged_fpr_tpr_report.json`](../runs/merged_fpr_tpr_report.json); access ladder + limitations declared per class in the assurance report | The **backdoor-like case is the measured failure.** `bias_lift` (targeted logit lift, the closest arm to the PS language) moved the weights in all **396** arms and left `f1_relative_drop` at exactly **0.000**; both live rules catch it at their own false-alarm rate (**5.1% / 6.1%** vs 5.18% / 5.23%). A behaviour-derived signal cannot see a modification behaviour does not reflect. Substitution is caught at **47.9%** (prune 0.25) and head noise at **92.9%** (dose 1.00) — but pruning is measured **on survivors only** (0 / 5 / 36 / 92 unscorable across doses). |
| C3 | **Inference provenance** — cryptographic binding of input, model digest, config, output; tamper / replay detectable | **SATISFIED** (mechanism) | [`cviaf/provenance/replay.py`](../cviaf/provenance/replay.py) — the capsule binds `input_sha256`, `model_digest`, `config` + `config_sha256`, `environment`, `output` + `output_sha256`, sealed with HMAC-SHA256; [`cviaf/provenance/verify_replay.py`](../cviaf/provenance/verify_replay.py) is a **standalone verifier that does not import the writer**; [`cviaf/provenance/attestation.py`](../cviaf/provenance/attestation.py) (opt-in DSSE/in-toto export); [`runs/provenance_ledger.jsonl`](../runs/provenance_ledger.jsonl) | Three honest ceilings. (a) The seal is **HMAC, which is a transport seal, not non-repudiation** — the module's own docstring says so, and the key must live outside the inference service. (b) Attestation is **opt-in and needs operator-supplied keys**; nothing generates or stores them. (c) The ledger is **12 chained entries over 19 artefacts** — it is the record of *how these measurements were produced*, not production inference records at scale. Also 2 of 15 findings in the sample report carry no per-finding evidence or affected asset (see C5). |
| C4 | **Distribution-shift assessment** — calibrated score, drift vs manipulation | **PARTIAL** | [`cviaf/lab/drift_harness.py`](../cviaf/lab/drift_harness.py) (metric definitions fixed as data); [`cviaf/lab/drift_cells.py`](../cviaf/lab/drift_cells.py) (11 declared cells, each with a no-shift resample control); [`cviaf/drift/attribution.py`](../cviaf/drift/attribution.py) (610 lines, natural-vs-manipulated attribution); [`docs/DRIFT_CELLS.md`](DRIFT_CELLS.md) | **The declared shifts are near the noise floor.** The gate records **8 drift decisions and 32 under-determined across 4 metrics** at n=160 per cell. Reporting `under_determined` rather than forcing a call is the correct behaviour, but it means most of the battery does not decide. Drift-vs-manipulation attribution exists as code and is not covered by a receipt of the same weight as the FPR half. |
| C5 | **Analyst-facing governance** — reason, evidence, severity, affected asset, disposition, tamper-evident audit trail, declared unsupported classes | **SATISFIED** | [`cviaf/governance/schema.py`](../cviaf/governance/schema.py) (validator needing nothing installed); [`runs/demo_assurance/assurance_report.json`](../runs/demo_assurance/assurance_report.json); [`cviaf/lab/provenance_ledger.py`](../cviaf/lab/provenance_ledger.py); [`cviaf/lab/review.py`](../cviaf/lab/review.py) (cost-sensitive abstention + review queue); [`docs/COVERAGE_STATEMENT.md`](COVERAGE_STATEMENT.md); the dashboard's seven tabs | **13 of 15 findings**, not 15, carry all four of reason / evidence / affected asset / disposition: the two `provenance → inference_tampering` findings carry a title and a disposition but **no `evidence` dict and no `affected_assets`** — the exact gap the clause is about, in the one module whose job it is. Disposition vocabulary in the sample is `quarantine` ×12 / `review` ×3; `accept` is defined in the schema and appears in the risk module, but not in this report. |

---

## 2. The five constraints

| Constraint | Verdict | Evidence | Note |
|---|---|---|---|
| **Fully offline / air-gapped** | **SATISFIED**, machine-checked | gate clause `2.2.6-a` (`check:offline`, AST scan); gate `offline` → **pass**: "assurance path is import-clean; 1 guarded fetch site(s) in corpus construction"; [`tests/test_offline_guard.py`](../tests/test_offline_guard.py) | The **only** network call site in the repo is `urllib.request.urlretrieve` in [`cviaf/lab/cifar.py`](../cviaf/lab/cifar.py) (lines 101, 132) — one-off public-dataset fetch in *corpus construction*, refused by name under `CVIAF_OFFLINE=1` rather than left to stall. An independent grep of `scripts/` finds no network import at all. The gate scans the package; the scripts lane was checked by hand. |
| **COCO + YOLO ingest** | **SATISFIED** | gate clause `2.2.6-b` via `runs/format_ingest.json`: ingested `['coco', 'yolo']` with box counts `{'coco': 3, 'yolo': 3}` | 3 boxes each is a fixture, not a corpus. The ingest path is real; the scale at which it was exercised is small. |
| **ONNX + PyTorch support** | **SATISFIED** | gate clause `2.2.6-c` via `runs/format_ingest.json`: loaded `['onnx', 'pytorch', 'torchscript']`, ONNX forward `[1, 64, 16, 16]`; [`cviaf/formats/model_loader.py`](../cviaf/formats/model_loader.py) | Parity is covered by the torch-lane tests. |
| **No retraining for baseline assessment** | **SATISFIED** | Every detector is inference-only. The detection evidence was produced by **weight-space tampering of models already in the corpus**: 198 arms in **38.6 s**, then 1,188 arms in **3 m 49 s**, no gradient step anywhere. | The one place training happens is the lab's own backbone lane, which builds the *reference and population*, not the assessment. |
| **Graceful fallback without white-box access** | **SATISFIED** | The sample report's `limitations` records the skips with their reason: `"[model_integrity] neural_cleanse skipped: Requires white-box access"`, `"weight_analysis skipped: Requires white-box access"`, and `"Black-box assessment has lower detection confidence than white-box methods"`. Each supported class declares `access_required` in the coverage statement. | Fallback is *reported*, not silent. The confidence it costs is stated in prose rather than quantified. |

---

## 3. Number provenance — every headline figure, checked against its receipt

Re-derived from the committed receipts on 2026-09-29, not transcribed. Command:
`python -c "import json; ..."` over the paths in the middle column; the dashboard's build-time
agreement check re-derives the same quantities a second time.

| Quantity | Receipt | Verified value |
|---|---|---|
| Merged population | `runs/merged_fpr_tpr_report.json` → `census.all` | **56,627** clean, 0 positives |
| Split | `runs/merged_fpr_tpr_report.json` → `denominators` | 28,314 calibration / 28,313 evaluation negatives |
| `ctc_mean_clean` FPR @ α=0.05 | `…report.json` → `rules.ctc_mean_clean.fpr` | **0.0523** [0.0498, 0.0550], 1,481 / 28,313, threshold **0.9845092069058692** |
| `refdiv_mean_clean` FPR | `…report.json` → `rules.refdiv_mean_clean.fpr` | **0.0518** [0.0493, 0.0545], 1,467 / 28,313, threshold **0.7276592261904761** |
| `ctc_peak` / `ctc_q95` FPR | `…report.json` → `rules.ctc_*_clean.fpr` | **0.0000** [0.0000, 0.0001], 0 / 28,313, threshold **1.0 = corpus ceiling → degenerate** |
| TPR refused on the merged population | `…report.json` → `positives_measured`, `rules.*.tpr.status` | `false`; every rule `fpr_only` with `insufficient_denominator` and `point_estimate: null` |
| Model verification | `runs/integration_verify.json` | **56,628 / 56,628** verified, `problems: []`, `warnings: []`, `ok: true` |
| Detection ladder | `runs/tpr_ladder_at_frozen.json` | **1,188 arms**, **1,055 scored**, 3 classes × 4 doses × 99 parents; 133 unscorable, **all `substitution`** (0 / 5 / 36 / 92) |
| `refdiv_mean_clean` detection | `…ladder.json` → `cells` | substitution **13.1 / 47.9 / 93.7%**, weight_tamper **7.1 / 22.2 / 59.6 / 92.9%**, bias_lift **5.1%** at every dose |
| `ctc_mean_clean` detection | `…ladder.json` → `cells` | substitution 4.0 / 2.1 / 3.2%, weight_tamper 5.1 / 5.1 / 11.1 / 29.3%, bias_lift 6.1% |
| Pooled detection refused | `…ladder.json` → `pooled_estimate`, `pooled_refusal` | `null` + named refusal; no rule-level rate exists in the receipt |
| Behaviour inertness | `…ladder.json` → per-cell `n_behaviour_inert` | **396 / 396** `bias_lift` arms left `f1_relative_drop` at 0.000 |
| Single-dose pass (superseded) | `runs/tpr_at_frozen.json` | 198 arms, 193 scored, 5 unscorable (all substitution); pooled `refdiv` **0.3472** [0.2836, 0.4167] |
| Clause gate | `python -m cviaf.lab.coverage` | 14 / 14 clauses `measured`, 9 / 9 gates `pass`, overall **PASS** |

**Drift found and fixed in this pass.** A reverse audit of every quoted number in `README.md`,
`docs/`, `STATUS.md` and `SWOT.md` found three kinds of staleness; all are corrected in the same
commit as this document. (1) The README still billed the detection measurement as "a 198-arm"
pass — it is a **1,188-arm** ladder, and the built-vs-scored distinction (1,188 built / 1,055
scored / 133 named unscorable) is now stated where the count is quoted. (2) `STATUS.md`'s dashboard
note said "seven tests"; the dashboard carries **nine**. (3) The retired single-dose detection
numbers (`34.7%` pooled, `48.9%` substitution) survived in prose as if current; they now appear
**only** inside sections that banner them as history, each pointing forward at the ladder — they
must not be quoted as the current result, because pooling across classes or doses is exactly what
the ladder receipt refuses by name. The audit is a script, not a memory:
[`scripts/number_audit.py`](../scripts/number_audit.py) re-derives every figure from the receipts
(18 checks, exit 3 on drift), [`tests/test_number_audit.py`](../tests/test_number_audit.py) pins
the audit itself, and both run in about a second.

---

## 4. What is declared unsupported, and where a judge can read it

The assurance report ships its own refusal list; this is not a document that has to be believed,
it is an artifact. `coverage_statement` holds **27 entries**: 14 supported classes,
8 unsupported conditions, and the access assumptions.

Declared unsupported conditions, verbatim: federated-learning poisoning; hardware-level Trojan
injection; cryptographic key compromise; **adversarial examples at inference time**;
membership-inference attacks; model extraction / stealing; supply-chain attacks on software
dependencies; side-channel attacks on inference hardware.

Additionally declared deeper in the repo, and repeated here because they qualify the headline:

1. **The FPR is an average over one population.** The clean fleet is synthetic detectors; the
   reference-relative rule moves a long way when the reference changes.
2. **The base rate of the defended population is unknown**, so expected loss and break-even
   prevalence are refused rather than estimated. Multiplying the measured FPR by the measured TPR
   is meaningless and nothing in the repo does it.
3. **Lab box 2 (seeds 65,152–70,151, 5,000 models) does not exist anywhere** — a permanent census
   gap, named in `runs/integration_census.json`.
4. **`clean_none_fixed_s81621` and five pruning arms have an undefined CTC statistic.** They are
   named exclusions, not silent drops, in `skipped_*.json`.
5. **The 21 MB merged ledger is outside git**, so a clean clone cannot recompute the headline
   from scratch — it can verify the committed receipts and replay the arms, which is what
   [`REPRODUCE.md`](REPRODUCE.md) does.
