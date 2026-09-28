# PS 26228 → Design Traceability & Acceptance Matrix

**Purpose.** Every clause of the SIH 26228 problem statement, mapped to (a) the design decision that satisfies it, (b) the artifact a judge can open, and (c) the **acceptance test** that proves it. This document is the answer to the question *"show me where you handled X."* It is also the honest gap register: clauses we satisfy only partially are marked, with the reason.

**Convention.** `✅ met` · `🟡 partial` · `🟠 planned` · `❌ not met (and why)`
Legend for evidence: `[code]` = source artifact · `[doc]` = design/report · `[test]` = executable check · `[demo]` = live demonstration.

---

## 1. Clause-by-clause

### 2.2 preamble — "extensible … evaluate a contributed dataset, a trained model and associated inference records … evidence-based assessment"

| # | Clause | Design decision | Artifact | Acceptance test | Status |
|---|---|---|---|---|---|
| 1.1 | Extensible framework | `DetectorPlugin` protocol + capability manifests + **Assurance Planner** that selects a detector set per (assets, access, threat model, budget) and records *why* each rejected detector was rejected | `cviaf/plugins/`, `cviaf/planner.py` `[code]`; architecture §4.2 `[doc]` | Add a new detector in one file with a manifest; it appears in the plan and in the coverage matrix without touching the orchestrator | 🟠 |
| 1.2 | Evaluates dataset + model + inference records | One `AssessmentRequest` taking all three asset kinds; modules are independent and individually skippable with declared coverage impact | `cviaf/orchestrator.py` `[code]` (exists in v2) | Run with each asset subset; report stays valid and coverage shrinks correctly | ✅ |
| 1.3 | "Evidence-based" assessment | Every finding carries `evidence{}` with artifact digests + a pointer into the battery slice and calibration reference that produced it | report schema `[doc]`; `Finding.evidence` `[code]` | Every finding in a sample report resolves to a real digest | 🟡 (v2 has evidence dicts but no pointer to a calibration reference) |
| 1.4 | "…and risk" (not just integrity) | Risk = calibrated hypothesis posterior × loss-matrix expected loss | architecture §5.5 `[doc]` | `expected_loss` present for all three dispositions on every finding | 🟠 |

### 2.2.1 — Training-Data Integrity

| # | Clause | Design decision | Artifact | Acceptance test | Status |
|---|---|---|---|---|---|
| 2.1 | Trigger injection | `trigger.pixel` (patch variance collapse), `trigger.fft` (high-frequency anomaly), `trigger.spectral` (**SPECTRE** robust covariance on per-box RoI, upgrading v2's PCA+kMeans) | `cviaf/data_integrity/` `[code]` | Recall on AB-1 synthetic triggers at 5% and 10% poison; **power curve published** | 🟡 (v2 has pixel/FFT/spectral but classifier-shaped; upgrade planned) |
| 2.2 | Label flipping | `label.objectlab` — cleanlab `object_detection.ObjectLab` (wrong class, missed box, bad box) using model outputs + given labels, **no retraining** | `cviaf/data_integrity/label.py` `[code]` | Detected on AB-1 flip patterns; ≥50 samples/class precondition surfaced when violated | 🟠 |
| 2.3 | Systematic mislabelling | Same detector + **per-contributor** aggregation exposing *pattern* (not just rate): confusion-matrix asymmetry by contributor | architecture §5.1 `[doc]` | A contributor whose flips are systematic (one direction) scores higher than a random-flip contributor at equal rate | 🟠 |
| 2.4 | Near-duplicate flooding | `dup.phash` cheap pass → `dup.sscd` embeddings + FAISS cosine clustering → contributor attribution | `[code]` | AB-1 flood detected; **dilution flood defeated** (see 2.9) | 🟠 |
| 2.5 | OOD insertion | Per-**RoI** Mahalanobis/KNN under the OpenOOD protocol (not per-image: image-level scores are dominated by scene statistics) | architecture §5.1 `[doc]` | OOD AUROC per-box vs per-image, showing the per-box gain | 🟠 |
| 2.6 | "**Where contributor, batch or source metadata is available**" | Metadata is optional and its absence is a *coverage cell*, not a crash: `contrib.hier` reports `skipped/REFERENCE_MISSING` and the cross-contributor comparison degrades to a global baseline | coverage matrix `[doc]` | Run with and without contributor metadata; report states the difference in coverage | 🟠 |
| 2.7 | "**aggregate sample-level evidence into a source-level risk assessment rather than flagging samples in isolation**" | Hierarchical Beta-Binomial with partial pooling → `P(θ_c > θ_pop + τ)`; sampled posterior, credible interval, batch-level tail statistic | architecture §A3, §6 `[doc]` | Synthetic: contributor A (2 flags / 4 samples) ranks **below** contributor B (400 flags / 10 000) despite a higher raw rate | 🟠 |
| 2.8 | Small-n protection | Partial pooling shrinks low-volume contributors toward the population mean | same | as above | 🟠 |
| 2.9 | **Dilution resistance** (our addition) | Any ratio over contributor volume is attacker-controlled; so risk = (posterior exceedance) + (max batch flag concentration) + (flag-count LCB). Report names which fired | architecture §A3 `[doc]` | Malicious contributor adds 40 000 clean images → naive rate collapses, our statistic does not | 🟠 |
| 2.10 | Calibrated sample flags | Benjamini–Yekutieli FDR control (default; valid under arbitrary dependence) with per-flag `q`-values | architecture §5.5, §6 `[doc]` | Empirical FDR on calibration splits matches nominal within CI | 🟠 |

### 2.2.2 — Model Integrity

| # | Clause | Design decision | Artifact | Acceptance test | Status |
|---|---|---|---|---|---|
| 3.1 | "**methods appropriate to the level of access available**" | Explicit access ladder: T0 `distscan.prenms` (black) → T1 `trace.ctc` (black) → T2 `trace.ftc` (black) → T3 `odscan` (white) → T4 `distil` (white, data-free) → T5 `fingerprint` → T6 `weights` → T7 `neuralcleanse` (labelled **classifier-derived control only**) | architecture §5.2 `[doc]` | Each tier runs in its own access mode; v2's Neural-Cleanse-primary path is demoted and the reason recorded | 🟠 |
| 3.2 | Behavioural fingerprinting | Battery-conditioned fingerprint vector: per-class confidence quantiles on `AB1.probe` ‖ box-count distribution ‖ IoU-stability under AB-1 transforms ‖ pre-NMS class priors; quantized, hashed, signed. **Null distribution measured by repeated enrollment of the same clean model**, so the tolerance is measured, not guessed | architecture §5.2, §A8 `[doc]` | Re-enroll the same model ×5 → distance ~0 and tolerance derived empirically | 🟠 |
| 3.3 | Trigger search or reconstruction | T3 ODSCAN (S&P'24) + T4 DISTIL (ICCV'25, data-free via latent diffusion) | `[code]` | White-box tier beats the Neural-Cleanse control on the AB-1 model corpus, *shown as a table* | 🟠 |
| 3.4 | Parameter / activation statistics | T6 weight statistics; Z-PEFT-style spectral signatures **explicitly labelled PEFT/adapter-scoped**; per-RoI activation statistics | `[code]` | Statistics run; scope limitation is printed with the result | 🟡 (v2 has kurtosis/dormancy stats; scope labelling absent) |
| 3.5 | "**comparison against a defined reference battery**" | **Assurance Battery AB-1**: content-addressed, signed bundle of probe corpora, transform families, labelled calibration splits, attack recipes+seeds, loss matrix, power definitions. Reports carry `battery_id` + `battery_digest` | architecture §4.3 `[doc]` | Same battery digest + same assets ⇒ identical findings; fingerprint comparison **refuses** (`fingerprint_incomparable`) across differing battery digests instead of silently mis-comparing | 🟠 |
| 3.6 | Detect OGA / RMA / GMA | TRACE CTC + DistScan + fingerprint | `[code]` | Per-class recall reported for each BadDet category | 🟠 |
| 3.7 | Detect **ODA / cloaking** | TRACE **FTC / "Island Effect"** — the only test-time handle on vanished objects; must not be omitted | architecture §A9 `[doc]` | ODA recall > 0 on AB-1; the NBO-vocabulary precondition is surfaced as a limitation | 🟠 |
| 3.8 | Model substitution | Four-binding enrollment (artifact digest ‖ fingerprint ‖ ML-BOM ‖ reference distribution) whose cross-product separates substitution (both differ) from benign re-serialization (digest differs, behaviour matches) from **runtime compromise** (digest matches, behaviour differs) | architecture §A8 `[doc]` | Benign ONNX↔TorchScript re-export raises **no** substitution finding; a runtime-substituted model with an unchanged file raises a HIGH finding | 🟠 |
| 3.9 | "**state the access assumptions**" | Per-finding `access_mode ∈ {white, black, gray}` and per-detector `access_required` in the capability manifest | schema `[doc]`; `[code]` | Every model finding has an `access_mode`; grep shows none missing | 🟡 (v2 has `AccessLevel` on the assessor, not per finding) |
| 3.10 | "**confidence**" | Structured: `{value, type ∈ {conformal_p_value, e_value, uncalibrated_heuristic}, calibration_ref}`; uncalibrated scores **cannot** drive a `quarantine` | architecture §A2 `[doc]` | A finding with `type: uncalibrated_heuristic` and value 0.99 still yields at most `review` | 🟠 |
| 3.11 | "**limitations**" | Per-finding `limitations[]` naming the *specific* evasion measured against that detector, not a global list | schema `[doc]` | Every detector that failed our own red team carries its measured degradation on its findings | 🟠 |
| 3.12 | White→black graceful fallback | Ladder re-plans rather than degrades; skipped tiers produce `blocking_skips` in `assessment_completeness` | architecture §4.2, §A11 `[doc]` | Black-box-only run: **`accept` is impossible**; report says which assessments were unavailable and why | 🟠 |

### 2.2.3 — Inference Provenance and Output Integrity

| # | Clause | Design decision | Artifact | Acceptance test | Status |
|---|---|---|---|---|---|
| 4.1 | Bind input ↔ model identity/weight digest ↔ preprocessing+config ↔ output | `H(image) ‖ model_manifest_digest ‖ preproc_config_hash ‖ H(canonicalized output) ‖ sequence ‖ nonce ‖ prev_hash ‖ execution_environment` | `cviaf/provenance/` `[code]` (v2 core exists) | Tamper each component independently; each breaks a *different* check so the report names **which** component changed | ✅ |
| 4.2 | "**hashes**" | SHA-3/SHA-256 digests + Merkle tree over leaves (v2) — plus **canonicalized output hashing** (stable sort, declared quantization, signed `quantization_spec`) so float non-determinism across hardware does not produce false tamper alarms | architecture §5.3.1 `[doc]` | Verify a seal produced on one device on another device: passes | 🟠 |
| 4.3 | "**signatures**" | Ed25519, **hard-fail-closed** when the asymmetric backend is missing (`--allow-symmetric` explicit only, and then the report carries `non_repudiation: false`) | architecture §A7/§5.3.4 | Without `cryptography` installed, signing **refuses**; with `--allow-symmetric`, the report says non-repudiation is absent | 🟠 (v2 falls back silently) |
| 4.4 | "**sequence** … controls" | Monotonic sequence number bound into each seal and the chain | `[code]` | Delete/reorder a record → chain verification fails at a named index | ✅ |
| 4.5 | "**timestamp** … controls" | Offline **TSA co-signature** (second in-toto functionary) + logical clock; wall-clock explicitly labelled advisory (no NTP in air gap) | architecture §5.3.6 `[doc]` | Report field `time_source`; TSA signature verifies offline | 🟠 |
| 4.6 | "**nonce** controls" | Per-record nonce; replay detection = nonce reuse or sequence regression | `[code]` | Replay a valid record verbatim → detected | ✅ |
| 4.7 | "make post-hoc alteration, substitution or **replay** … detectable" | DSSE envelope + in-toto layout + Merkle checkpoints with signed roots | architecture §A7 `[doc]` | **Third-party** verifier script with no CVIAF import validates an attestation and an inclusion proof | 🟠 |
| 4.8 | Model artifact integrity (ONNX/TorchScript) | **OMS/model-transparency semantics: sign the file manifest, not the serialized tensor** — ONNX external-data and PyTorch zip archives are not byte-stable across tool versions | architecture §A7 `[doc]` | ONNX ↔ TorchScript round-trip does not break model verification | 🟠 |
| 4.9 | No cloud KMS | Keys generated and held locally; optional PKCS#11/TPM backend | `[code]` | `cviaf doctor` reports the key backend; no network path exists in the provenance module | 🟠 |
| 4.10 | Tamper-evident audit log | Hash-chained entries (v2) + Merkle root checkpoints so the log can be *shipped* as roots + inclusion proofs instead of a monolithic file | architecture §5.3.3 `[doc]` | Verify a single entry with only the signed root and its proof path | 🟠 |

### 2.2.4 — Distribution-Shift and Anomaly Assessment

| # | Clause | Design decision | Artifact | Acceptance test | Status |
|---|---|---|---|---|---|
| 5.1 | Detect deviation from a **declared reference distribution** | Reference distribution is a **declared, enrolled asset** with its own digest (part of the four bindings); deviation is measured against it, never against an implicit baseline | architecture §A8/§5.4 `[doc]` | Report echoes `reference_distribution.id` + digest; missing reference ⇒ `skipped/REFERENCE_MISSING`, not a silent default | 🟡 (v2 takes `reference_features` directly with no declaration/enrollment record) |
| 5.2 | "changes caused by **terrain, season, sensor, illumination**" | Shift attribution runs on named covariate axes; AB-1 includes synthetic terrain/season/sensor/illumination families so each axis has a calibration reference | architecture §5.4 `[doc]` | Per-axis attribution on AB-1 synthetic shifts | 🟠 |
| 5.3 | "**characterise** the observed shift" | Split report into **covariate** `P(X)` / **concept** `P(Y\|X)` / **prior** `P(Y)`, each with the evidence needed to license the split (concept requires labels; its absence is reported) | `[doc]` | Concept-drift query on labeled data works; the same query without labels reports `not_applicable`, not a false negative | 🟡 (v2 has a `ShiftCharacterizer`; the licensing conditions are not enforced) |
| 5.4 | "**calibrated** risk or confidence score" | Temperature / vector scaling applied **before** thresholding (detector confidences are miscalibrated); ECE before/after reported; conformal calibration on `AB1.ref` | `[doc]` | ECE improves after scaling; both numbers published (including if it does not improve) | 🟠 |
| 5.5 | "**distinguish probable operational drift from suspicious manipulation**" | Two calibrated signals — **feature-space geometry** (manipulation concentrates in a narrow direction) vs **confidence polarization** (a trigger forces over-confidence; natural covariate shift broadly degrades confidence). Agreement → label; conflict → `under_determined` | architecture §5.4 `[doc]` | AB-1 contains both synthetic environmental shift and adversarial perturbation; the framework labels both correctly **and** emits `under_determined` on the constructed ambiguous case | 🟠 |
| 5.6 | "**where the available evidence supports such a distinction**" | Rendered literally as a **third verdict** (`under_determined`) with a statement of what evidence would resolve it | `[doc]` | `under_determined` appears in the AB-1 ambiguous case and names the missing evidence | 🟠 |
| 5.7 | Evadability honesty | Two-window drift tests are provably evadable (arXiv 2411.16591); prefer **within-window/block** formulations and state the caveat **on the finding**, not only globally | `[doc]` | The drift finding's own `limitations[]` cites the evadability result | 🟠 |

### 2.2.5 — Analyst-Facing Assurance and Governance

| # | Clause | Design decision | Artifact | Acceptance test | Status |
|---|---|---|---|---|---|
| 6.1 | Human-readable reason | `title` + `description` per finding | schema `[code]` | — | ✅ |
| 6.2 | Supporting evidence | `evidence{}` with digests + battery/calibration pointers | schema `[code]` | Every evidence entry resolves | 🟡 |
| 6.3 | Confidence | Structured calibration object (§3.10) | schema `[code]` | see 3.10 | 🟠 |
| 6.4 | Severity | Severity retained as a **display** attribute, derived from expected loss (not the driver of disposition) | `[doc]` | Severity is monotone in expected loss across the AB-1 corpus | 🟠 |
| 6.5 | Affected asset | `affected_assets[]` with typed, digested asset refs (`model:detector.onnx#sha256:…`) | schema `[code]` | — | 🟡 |
| 6.6 | "recommended disposition such as accept, review or quarantine" | `accept/review/quarantine` from an explicit, editable **loss matrix**, with `expected_loss` reported for every candidate | architecture §A4 `[doc]` | Changing the loss matrix changes the disposition in a demonstrable, logged way | 🟠 |
| 6.7 | Tamper-evident audit trail | Bounded to the provenance Merkle root; `verify-audit` recomputes the chain; the report itself is signed | `[code]` + `[test]` | `cviaf verify-audit` exits non-zero on any mutation | ✅ |
| 6.8 | "**explicitly declare attack classes or conditions that it does not support**" | **Coverage matrix, generated per run**, keyed to **NIST AI 100-2e2025** + BadDet codes, with per-cell `covered/partial/not_covered/not_applicable`, method, access, calibration set, **measured detection floor**, residual risk | architecture §A10/§8 `[doc]` | Two runs with different access levels produce **diffable** coverage matrices; `un-supported_conditions[]` present | 🟡 (v2 emits a static literal; the matrix is not run-generated and not taxonomy-keyed) |

### 2.2.6 — Constraints

| # | Clause | Design decision | Artifact | Acceptance test | Status |
|---|---|---|---|---|---|
| 7.1 | "**complete evaluation workflow must operate offline** … air-gapped … no cloud services or external APIs" | Signed **offline bundle** (wheels + pinned-commit repos + weights + AB-1); `cviaf doctor --offline` verifies the bundle digest; **no-egress CI job** runs the full demo under network isolation | architecture §10 `[doc]` | Full pipeline completes with networking blocked and produces `docs/evidence/no-egress.txt` | 🟠 |
| 7.2 | "ingest common CV dataset formats, **including COCO and YOLO**" | Loaders exist in v2 (`cviaf/formats/`); add annotation-integrity metadata bridging into `label.objectlab` and a contributor manifest | `[code]` | Real COCO instances JSON + a YOLO `data.yaml` directory both parse and produce per-sample metadata | 🟡 (v2 loaders exist but have not been run on real data) |
| 7.3 | "support the organiser-defined reference model formats, **including ONNX and PyTorch/TorchScript**" | `ModelWrapper` ABC with ONNX and PyTorch backends + OMS-style file-manifest hashing; DETR/Faster-RCNN/YOLO agnostic via the black-box query interface | `[code]` | Assessment runs against an ONNX export **and** a TorchScript export with identical reported method coverage | 🟡 (v2 wrapper exists; TorchScript path unverified) |
| 7.4 | "Baseline integrity assessment **must not require retraining**" | Every tier T0–T6 is inference-only. Retraining appears only as opt-in remediation | `[doc]` + `[test]` | CI asserts no training loop is reachable on the baseline code path | ✅ |
| 7.5 | "Optional remediation **may** use retraining" | `detect+retrain` is an explicit mode with measured clean-mAP cost (fine-pruning measured to crater mAP, so purification is the default recommendation) | architecture §A5 `[doc]` | Remediation options carry measured `clean_map_delta` + `residual_asr` | 🟠 |
| 7.6 | "Methods that require white-box access must **fall back gracefully or clearly report that the relevant assessment is unavailable**" | **Completeness invariant**: `accept` forbidden while `completeness < 1.0` for any bearing detector; every skip has a reason code + coverage impact | architecture §A11 `[doc]` | Property test across a matrix of missing assets/access: **no path emits `accept` with a blocking skip** | 🟠 |

### 2.3 — Expected Solution / deliverables

| # | Clause | Design decision | Artifact | Acceptance test | Status |
|---|---|---|---|---|---|
| 8.1 | Model-agnostic across architectures | Black-box query interface is the primary path; architecture-specific code is confined to the loader | `[code]` | Same assessment pipeline on YOLO, Faster-RCNN, DETR | 🟡 |
| 8.2 | "use publicly available or team-generated datasets and models" | AB-1 uses COCO-subset + VOC + synthetic shifts; weights public | `[doc]` | All AB-1 ingredients have license/provenance recorded | 🟠 |
| 8.3 | "**reproducible methods to introduce** representative poisoning, backdoor, substitution and tampering scenarios" | `AB1.attacks`: seeded recipes — BadNets, blended, WaNet-class, label-flip patterns, dup floods, OOD injection, BadDet OGA/RMA/GMA/ODA, BadDet+ penalty, **dilution floods, replay, benign re-export** (adversarial-to-us cases) | `[code]` | Re-running an attack recipe with its seed reproduces the identical poisoned artifact digest | 🟡 (v2 `attacks/` exists; seeds/recipes not content-addressed) |
| 8.4 | "identify suspicious data **or contributor behaviour**" | §2.6–2.9 | `[code]` | — | 🟡 |
| 8.5 | "assess model integrity" | §3 | `[code]` | — | 🟡 |
| 8.6 | "detect tampering of inference records" | §4 | `[code]` | — | ✅ |
| 8.7 | "provide supporting evidence for each finding" | §6.2 | `[code]` | — | 🟡 |
| 8.8 | "generate a clear assurance report stating **confidence, limitations and recommended action**" | schema v3 with structured confidence, per-finding limitations, expected-loss disposition | `[doc]` | Schema validates; sample reports for clean / poisoned / substituted / tampered / drifted / **under-powered** | 🟠 |
| 8.9 | **Deliverable: source code** | `cviaf/` package + CLI | `[code]` | `pytest` collects and passes (v2 currently collects nothing) | 🟡 |
| 8.10 | **Deliverable: architecture and setup notes** | `PRD_SIH26228_CVIAF.md`, `docs/CVIAF_V3_ARCHITECTURE.md`, README setup, offline-install notes | `[doc]` | A fresh machine installs from the offline bundle and runs the demo | 🟠 |
| 8.11 | **Deliverable: assurance-report schema** | JSON Schema v3 published and versioned; `cviaf schema` emits it | `schemas/` + `[code]` | Reports validate against the published schema in CI | 🟡 (v2 has a sample report, not a validating schema) |
| 8.12 | **Deliverable: reproducible audit log** | Merkle-rooted, signed, hash-chained log; seeds + battery digest + code version recorded per entry | `[code]` | A run can be replayed from the log's recorded inputs to the same finding ids | 🟡 |
| 8.13 | **Deliverable: coverage statement** | §6.8 | `[doc]` + `[code]` | — | 🟡 |
| 8.14 | Reference resources: **NIST TrojAI**, BackdoorBench | TrojAI OD round + BackdoorBench as external validation, in addition to AB-1 | `[doc]` | Results table includes an external benchmark row | 🟠 |

---

## 2. Scoring summary

| Capability | v2 | v3 target | Why the delta is defensible |
|---|---|---|---|
| 2.2.1 Data integrity | ~55% | ~85% | detector-native labels (ObjectLab), real near-dup (SSCD), per-RoI OOD, and a **non-dilutable** contributor model |
| 2.2.2 Model integrity | ~40% | ~88% | detector-native ladder (TRACE CTC+FTC, ODSCAN, DISTIL), four-binding enrollment separating substitution from benign re-export from runtime compromise |
| 2.2.3 Provenance | ~80% | ~92% | standards (DSSE/in-toto/OMS/ML-BOM), hard-fail-closed non-repudiation, canonicalized output hashing |
| 2.2.4 Drift | ~50% | ~78% | calibration + shift licensing + a **third verdict** for under-determination + evadability stated on the finding |
| 2.2.5 Governance | ~55% | ~90% | calibrated fusion, expected-loss disposition, run-generated taxonomy-keyed coverage matrix, completeness invariant |
| 2.2.6 Constraints | ~65% | ~90% | offline bundle + **no-egress proof**, real COCO/YOLO runs, TorchScript path, completeness invariant |

**The three claims a competitor cannot easily copy within the remaining time:**
1. **Every threshold is calibrated** against a signed, versioned reference battery — so confidence values are *falsifiable* (we publish the empirical FDR and where it deviates).
2. **Accept is impossible when coverage is incomplete** — a one-line invariant that converts the PS's "fall back gracefully" into a demonstrable safety property.
3. **Detection floors are published** — the report can say *"we have no power below 1.5% poisoning in this dataset,"* which is the single most credibly honest statement this problem admits, and it follows directly from the published impossibility result.

---

## 3. Known gaps we are not papering over

| Gap | Why it exists | What we say about it |
|---|---|---|
| Low-poison-rate detection (<1%) | Statistical power, not effort: ICLR 2026 shows poisoned-sample detectors fail as the poison rate drops. | Published detection floor. The report refuses to say "clean." |
| BadDet+ physical-world triggers | Position/scale-invariant by construction (log-barrier penalty); the 2026 literature calls physical robustness the open problem. | Cell marked **partial** with the reason. |
| Model ensembles have no C2PA representation | Standards gap, admitted in the C2PA AI/ML guidance. | Declared; we emit per-member ML-BOM + a composed manifest. |
| Adaptive attackers with detector knowledge | Full-knowledge adaptivity defeats in-principle bounds. | We report **cost inflation**, measured, not detection. |
| Concept drift without labels | Not identifiable without labels; fundamental. | `not_applicable` with the required-evidence list, never a false negative. |
| Drift vs manipulation, in general | Provably not separable in the windowed setting (arXiv 2411.16591). | Third verdict `under_determined`. |
| TRACE's auxiliary images in a hard air gap | TRACE needs public background/foreground imagery. | **Vendored into `AB1.probe`** — the gap becomes a battery build step, and if the battery is absent the tier reports `skipped`, not clean. |

---

*Companion to `docs/CVIAF_V3_ARCHITECTURE.md`. Status column reflects the codebase as of 2026-09-28; `🟠` items are specified and scheduled, not implemented. Keeping this document honest is part of the deliverable: a framework that demands calibrated confidence from its contributors must publish its own.*
