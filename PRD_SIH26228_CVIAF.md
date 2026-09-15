# Product Requirements Document (PRD)

## Trustworthy Computer Vision Integrity Assurance Framework (CVIAF)

**SIH Problem Statement 26228 · Ministry of Defence — Indian Army (DGIS)**
**Theme: Blockchain & Cybersecurity · Offline / Air-Gapped Operation**

| Field | Value |
|---|---|
| Document Type | Product Requirements Document (PRD) |
| Problem Statement ID | SIH 26228 |
| Issuing Authority | Ministry of Defence — Indian Army, Directorate General of Information Systems (DGIS) |
| Product Name | CVIAF — Computer Vision Integrity Assurance Framework |
| Version | 2.0 (Draft for review) |
| Status | Active development |
| Hardware Dependency | 1× NVIDIA H200 GPU (time-bound, 1–2 sessions of 4–8 hours) |
| Development Machine | MacBook Air M3, 16 GB (coding, testing, dry-runs) |

---

## 1. Executive Summary

Defense computer-vision pipelines — battlefield surveillance, satellite imagery analysis, autonomous vehicle perception, and target recognition — increasingly ingest components from **multiple untrusted or semi-trusted contributors**: external datasets, third-party ONNX models, and automated annotation services. Any one of these contributors can be malicious, compromised, or simply negligent.

The consequences are severe. A poisoned training set can implant a hidden trigger that makes a detector ignore enemy vehicles when a specific marker is present. A substituted model can fabricate phantom targets to bait friendly forces into ambush. A tampered inference log can rewrite history after the fact. None of these attacks are detectable by traditional cybersecurity tools, because they live in the *statistical and cryptographic fabric* of the ML pipeline itself.

CVIAF answers SIH 26228 with a **model-agnostic, offline assurance engine** that evaluates five layers of pipeline integrity — training data, model behavior, inference provenance, distribution shift, and analyst governance — without assuming any contributor is trusted and without any cloud dependency. Every assessment produces a signed, human-readable JSON report with severity, confidence, evidence, and an actionable disposition: **accept, review, or quarantine**.

The framework's core differentiator is **detector-native integrity assurance**: almost all public backdoor defenses target image *classifiers*, while this problem mandates COCO/YOLO *object detection*, whose attack surface (object fabrication, disappearance/cloaking, regional misclassification, NMS abuse) has no classifier equivalent. CVIAF closes that gap using 2025–26 state-of-the-art detector-specific methods — TRACE (CVPR 2025), ODSCAN (IEEE S&P 2024), and DISTIL (ICCV 2025).

---

## 2. Problem Statement (SIH 26228)

### 2.1 Formal Statement

*"Trustworthy Computer Vision Integrity Assurance for Data, Models, and Inference Outputs in Multi-Contributor Pipelines."*

Design a framework that can evaluate the integrity of a computer-vision pipeline whose components are supplied by multiple contributors, operating fully offline in air-gapped defense environments.

### 2.2 The Five Required Capabilities

| # | Capability | What It Must Detect |
|---|---|---|
| (a) | **Training-data integrity** | Poisoned samples, hidden triggers, label flipping, systematic mislabeling, near-duplicate flooding, out-of-distribution insertion in COCO/YOLO datasets |
| (b) | **Model integrity** | Backdoors/Trojans, model substitution, weight modification — in both white-box (weights available) and black-box (query-only) access modes |
| (c) | **Inference provenance & output integrity** | Post-hoc tampering, output substitution, replay, and model/input mismatch, via cryptographic binding |
| (d) | **Distribution-shift & anomaly assessment** | Natural operational drift (season, terrain, sensor, illumination) vs. adversarial manipulation |
| (e) | **Analyst-facing assurance & governance** | Severity, confidence, evidence, affected assets, recommended disposition, and tamper-evident audit trail |

### 2.3 Hard Constraints

- **Air-gapped:** no cloud services, no cloud KMS, no NTP time servers, no external APIs.
- **Multi-contributor:** no contributor is trusted a priori; contributor/source metadata must be used for attribution.
- **Model-agnostic:** must work across ONNX, PyTorch, YOLO, Faster-RCNN, DETR-class detectors.
- **Graceful degradation:** when white-box access is unavailable, fall back to black-box checks; never silently pass — always report what was skipped and why.
- **Offline cryptography:** all signing keys generated and stored locally (HSM/secure enclave where hardware permits).

### 2.4 Why This Problem Matters (Impact)

| Scenario | Attack | Operational Consequence |
|---|---|---|
| Battlefield surveillance | Object Disappearance (ODA) trigger | Enemy armor invisible when marker present |
| Border monitoring | Object Generation (OGA) trigger | Phantom infiltrators divert forces |
| Convoy autonomy | Regional Misclassification (RMA) | Civilian vehicle classified as threat (or reverse) |
| Satellite analysis | Dataset poisoning / label flips | Persistent blind spots in coverage |
| After-action review | Inference log tampering | Evidence cannot be trusted post-facto |

Detection of these attacks is currently a **manual, expert-driven process**. CVIAF automates it into a repeatable, auditable, offline pipeline — directly applicable to Army DGIS vision systems.

---

## 3. Threat Model

### 3.1 Attack Taxonomy (BadDet, ECCV 2022 — extended)

| Attack Class | ID | Effect | Classifier Analog? |
|---|---|---|---|
| Object Generation Attack | OGA | Fabricate phantom bounding boxes | None |
| Object Disappearance Attack | ODA | Suppress/hide real objects (cloaking) | None |
| Regional Misclassification Attack | RMA | Relabel a specific box in-place | Weak |
| Global Misclassification Attack | GMA | Relabel whole-scene detections | Closest |
| Label flipping / systematic mislabel | — | Corrupt training labels | Yes |
| Near-duplicate flooding | — | Bias model toward attacker's samples | Yes |
| OOD insertion | — | Inject domain-irrelevant samples | Yes |
| Model substitution | — | Swap weights post-deployment | Yes |
| Inference tamper / replay | — | Alter or replay sealed outputs | Yes |
| Adversarial distribution shift | — | Inputs crafted to evade drift baselines | Partial |

### 3.2 The Classifier→Detector Gap (Core Research Insight)

Legacy defenses — Neural Cleanse, STRIP, standard Spectral Signatures, Maximum Softmax Probability — were designed for single-label image classifiers. Object detectors have per-box class + localization + NMS structure, so these methods transfer poorly or not at all. ODA (disappearance) and OGA (fabrication) have **no classifier equivalent at all**. CVIAF is explicitly designed detector-first.

---

## 4. Proposed Solution — System Architecture

### 4.1 Overview

CVIAF is a Python framework organized as four sequential verification nodes feeding a governance layer, mirroring the data → model → inference → operation lifecycle:

```
┌─────────────────┐   ┌──────────────────┐   ┌─────────────────┐
│ data_integrity  │──▶│ model_integrity  │──▶│   provenance    │
│ (poison/labels/ │   │ (backdoor/substi-│   │ (Ed25519 seals, │
│  dup/OOD)       │   │  tution)         │   │  hash chain)    │
└─────────────────┘   └──────────────────┘   └─────────────────┘
         │                    │                      │
         ▼                    ▼                      ▼
      ┌──────────────────────────────────────────────────┐
      │  drift (Mahalanobis + MMD + KS + shift typing)   │
      └──────────────────────────────────────────────────┘
                              │
                              ▼
      ┌──────────────────────────────────────────────────┐
      │  governance → AssuranceReport (JSON, audit trail,│
      │  severity, disposition, coverage statement)     │
      └──────────────────────────────────────────────────┘
```

### 4.2 Module 1 — Training-Data Integrity

**Goal:** flag poisoned, mislabeled, flooded, or out-of-distribution samples and attribute them to contributors.

| Detector | Method | Attack Covered |
|---|---|---|
| TriggerDetector (pixel) | Per-pixel variance analysis; static patches produce near-zero variance in patch region | BadNets patch triggers |
| TriggerDetector (FFT) | High-frequency energy spikes in Fourier domain | Blended triggers |
| TriggerDetector (spectral) | Per-class SVD; poisoned samples correlate with top singular vector | Backdoor spectral signatures |
| LabelIntegrityChecker | Cross-validated KNN/LogReg confident learning; per-contributor flip-pattern analysis | Label flipping, systematic mislabeling |
| DuplicateDetector | Exact SHA-256 + cosine near-dup clustering; contributor-flood threshold (80%) | Near-duplicate flooding |
| OODDetector | Mahalanobis (robust MinCovDet) + Isolation Forest; dual-flag confidence boost | OOD insertion |

All findings roll up into **per-contributor risk scores** — a hard requirement of the PS.

### 4.3 Module 2 — Model Integrity

**Goal:** determine whether a supplied model is backdoored, substituted, or weight-modified, with graceful white-box → black-box degradation.

| Access | Detector | Method |
|---|---|---|
| White-box | NeuralCleanseDetector | Gradient-free trigger inversion; anomalously small L1 trigger per class flags backdoor |
| White-box | WeightAnalyzer | Kurtosis anomalies, suspicious sparsity, dormant-neuron patterns |
| Black-box | EntropyProbe | Abnormally low entropy on pure noise; STRIP-style blended-image entropy |
| Black-box | BehavioralFingerprinter | Prediction agreement on reference battery vs. stored fingerprint → substitution |

Every check that cannot run is recorded in `checks_skipped` with a reason. **The system never silently passes.**

### 4.4 Module 3 — Inference Provenance & Output Integrity

**Goal:** cryptographically bind every inference output to its exact input, model, config, and position in the session.

- **Ed25519 signing keys** generated locally (HMAC-SHA256 fallback if `cryptography` unavailable).
- **InferenceSeal** binds: SHA-256(image) + model digest + config hash + output hash + nonce + monotonic sequence + timestamp + previous-seal hash.
- **Hash-chain audit trail:** each seal links to the previous; reordering or deletion breaks the chain.
- **Merkle tree** for O(log n) batch verification proofs.
- `verify_seal()` returns per-check pass/fail: image hash, model digest, config hash, output hash, signature — pinpointing *which* component was tampered.

### 4.5 Module 4 — Distribution-Shift & Anomaly Assessment

**Goal:** distinguish natural operational drift from adversarial manipulation.

- **Mahalanobis distance** from a declared reference distribution (regularized covariance).
- **MMD** (kernel two-sample test) and **KS tests** per feature dimension.
- **MSP** (max softmax probability) for confidence suppression.
- **ShiftCharacterizer** classifies shift as *covariate* (P(X) changed), *concept* (P(Y|X) changed), or *prior* (P(Y) changed), then labels it *probable natural drift* vs. *suspicious manipulation* based on distance-distribution shape (extreme skew/outliers → adversarial; gradual uniform → environmental).
- Per the 2024 result "Adversarial Attacks for Drift Detection" (arXiv 2411.16591), two-window drift tests are provably evadable; CVIAF prefers **block-based / within-window** detection and explicitly declares that drift vs. manipulation cannot be perfectly separated.

### 4.6 Module 5 — Governance & Analyst Reporting

**Goal:** turn module outputs into decisions.

- Aggregates all findings into overall severity and disposition.
- Emits a **human-readable summary**, a machine-readable **coverage statement** (supported attack classes, methods, access requirements, known limitations), and a **hash-chained audit trail** that is itself tamper-evident.
- Every finding includes: severity, confidence, title, description, evidence dict, affected assets, disposition, and remediation.

### 4.7 Supporting Infrastructure

- **Attack generators:** BadNets, blended triggers, label flipping, duplicate flooding, OOD injection, seal tampering/forgery/replay, model substitution — for reproducible self-testing.
- **Dataset loaders:** COCO and YOLO formats.
- **Model wrappers:** ONNX and PyTorch/TorchScript with unified predict / intermediate-features / parameters / digest interface.
- **CLI:** `demo`, `assess`, `verify-audit`, `verify-seal`, `schema`.

---

## 5. Research Foundation (2023–2026 State of the Art)

The framework is grounded in a verified literature review (18 search queries, primary-source verification, hallucinated citations removed). Key sources:

| Area | Method | Venue / Year | Role in CVIAF |
|---|---|---|---|
| OD backdoor detection | **TRACE** (transformation consistency) | CVPR 2025 | Headline black-box test-time detector (CTC + Island-Effect FTC); F1 0.880, AUROC 0.897 |
| OD backdoor scanning | **ODSCAN** | IEEE S&P 2024 | White-box scanning baseline |
| Trigger inversion | **DISTIL** (latent diffusion) | ICCV 2025 | Data-free white-box trigger reconstruction |
| Pre-NMS signal | **DistScan** | 2026 | Cheap always-on pre-NMS distribution check |
| Label errors | **cleanlab ObjectLab** | cleanlab 2.5+ | COCO/YOLO-native label-error scoring |
| Near-dup | **SSCD** | CVPR 2022 | Self-supervised copy-detection embeddings |
| Provenance | **C2PA 2.4 AI/ML** | 2025 standard | Asset-reference assertions, AI-ML Output Credentials, offline manifests |
| Drift evadability | Adversarial Attacks for Drift Detection | 2024 | Two-window tests evadable → use block-based detection |
| Governance | NIST AI RMF 1.0 | 2023 | Govern/Map/Measure/Manage framing |
| Benchmark | TrojAI OD round, BackdoorBench, OpenOOD v1.5 | 2022–2024 | Validation harnesses |

### 5.1 Known Limitations (Declared in Every Report)

- Adaptive/feature-space and clean-label triggers (WaNet, label-consistent) may evade current detectors.
- Drift and adversarial manipulation **cannot be perfectly separated** — residual ambiguity is explicitly reported.
- Model ensembles lack a C2PA representation.
- TRACE relies on auxiliary public images (must be pre-vendored for hard air-gap).
- Fine-pruning remediation cuts attack success rate but can reduce clean mAP — offered as opt-in, never default.

---

## 6. Validation & Benchmarking Plan (H200)

### 6.1 What Will Be Computed

| Phase | Task | Output |
|---|---|---|
| A | Train clean + backdoored detector corpus (~14–16 models: YOLOv5s, Faster-RCNN R50 on VOC/COCO with OGA/RMA/GMA/ODA at 5–10% poison rates) | Ground-truth benchmark corpus |
| B | Run detector sweeps: TRACE, ODSCAN, Neural Cleanse, entropy probes, behavioral fingerprinting across all models | AUROC / F1 / FPR@95TPR / runtime table |
| C | Drift/OOD calibration: per-box Mahalanobis/MSP/KNN on shifted vs. clean COCO subsets; temperature/vector scaling | ECE before/after calibration |
| D (optional) | Remediation: fine-pruning, ANP | Security-vs-accuracy trade-off curves |

### 6.2 Metrics

- Backdoor detection: AUROC, F1, FPR@95TPR, per-attack-class recall (OGA/RMA/GMA/ODA).
- Model utility: clean mAP before/after any remediation.
- Drift: AUROC for OOD detection, calibration error (ECE), shift-type classification accuracy.
- Provenance: tamper-detection rate (target 100% on bit-flip/substitution/replay tests), verification latency.
- System: wall-clock per assessment, memory footprint, offline-install reproducibility.

---

## 7. Compute Requirements — Why the NVIDIA H200 Is Needed

### 7.1 Purpose of H200 Usage

The deliverable is an assurance engine plus benchmark **evidence**, not a large trained model. The H200's role is to **generate the ground-truth attack corpus and run detector sweeps at scale** — work that is training- and inference-heavy and cannot be completed on a laptop or CPU-only machine within any reasonable time.

Specifically:

1. **Train ~14–16 detector variants** (clean + BadDet-poisoned) to create evaluation ground truth.
2. **Run TRACE-style sweeps** — 100+ blended-background forward passes per test image across the corpus.
3. **Run ODSCAN white-box scanning** and Neural Cleanse trigger inversion per model.
4. **Run DISTIL latent-diffusion trigger inversion** where attempted (memory-intensive).
5. **Execute drift/OOD calibration** across COCO subsets under the OpenOOD protocol.

### 7.2 Data Size Requirements

| Item | Size |
|---|---|
| COCO 2017 (train subset + val + annotations) | ~20 GB |
| PASCAL VOC 2007/2012 | ~3 GB |
| Model corpus (14–16 detectors) | ~2–3 GB |
| Detector support assets (TRACE backgrounds, SSCD, diffusion weights) | ~5–8 GB |
| Results, logs, checkpoints (copied off after each run) | ~5–10 GB |
| **Total working set (disk)** | **~35–45 GB** |
| **Peak GPU memory per training job** | **8–24 GB** |
| **Peak GPU memory for DISTIL diffusion inversion** | **>24 GB** |

### 7.3 Why This Cannot Run on a Laptop / Normal Computer

Development (coding, unit tests, synthetic demos, provenance/crypto work, harness dry-runs) is done on a MacBook Air M3, 16 GB. However, benchmark generation cannot, for four reasons:

1. **No NVIDIA CUDA.** The required toolchains (BadDet, ODSCAN, BackdoorBench, PyTorch CUDA kernels, cuDNN detection training) are CUDA-first; on Apple MPS several operators are unsupported and fall back to CPU.
2. **Insufficient memory.** 16 GB unified memory is shared by OS, framework, data, and activations; one H200 GPU has 141 GB dedicated HBM. DISTIL diffusion inversion alone exceeds 16 GB.
3. **Throughput.** Detector training needs batch 64–128 with tensor cores; the laptop is limited to batch 4–8 with thermal throttling (fanless). The 14-model corpus takes **weeks on laptop vs. 2–4 hours on one H200**.
4. **Benchmark credibility.** Defensible AUROC/F1/mAP numbers require consistent, non-throttled, pinned-CUDA hardware — not a thermally constrained development machine.

### 7.4 Requested Allocation

| Item | Request |
|---|---|
| GPU | 1× NVIDIA H200 (141 GB HBM) |
| Sessions | 1–2 sessions |
| Duration per session | 4–8 hours |
| Disk (scratch) | ~50 GB |
| Concurrency | 3–6 experiment jobs in parallel (memory permits) |

The request is deliberately right-sized: no continuous reservation, no multi-node need. All coding and debugging is completed locally beforehand; H200 time is used **only for execution**.

---

## 8. Phased Execution Plan

| Phase | Machine | Tasks | Exit Criteria |
|---|---|---|---|
| 1. Local build | M3 Air | Run full `cviaf demo`; fix defects; add unit tests per module; COCO/YOLO loader integration; nano-scale TRACE-style CTC prototype | End-to-end demo produces valid assurance report |
| 2. Harness | M3 Air | Build manifest-driven experiment runner; `--dry-run --num-images 8` mode; results schema; checkpoint/resume | Full pipeline runs on 8 images producing results JSON |
| 3. Air-gap rehearsal | M3 Air | Vendor datasets, weights, repos (pinned commits), pip wheels; offline install from `--no-index --find-links` | Framework installs and runs with network disabled |
| 4. H200 Session 1 | H200 | Train 14–16 model corpus; copy results off immediately | Corpus + training logs + hashes archived |
| 5. H200 Session 2 | H200 | Detector sweeps (TRACE/ODSCAN/NC/entropy); drift calibration; optional remediation | Full metrics table + calibration report |
| 6. Reporting | M3 Air | Generate final benchmark report; seal results with CVIAF provenance module; prepare SIH presentation | Signed results, PRD, demo, slides complete |

---

## 9. Deliverables

1. **CVIAF engine** — Python package, offline-installable, CLI + API.
2. **Benchmark corpus** — 14–16 clean/backdoored detectors with training manifests and provenance seals.
3. **Detection-performance report** — AUROC/F1/FPR/mAP/runtime per attack class, per detector.
4. **Calibration report** — drift/OOD ECE before/after post-hoc scaling.
5. **Sample assurance reports** — JSON output for clean, poisoned, tampered, and drift scenarios.
6. **PRD + architecture documentation** (this document).
7. **SIH presentation** with live demo.

---

## 10. Success Criteria

| Criterion | Target |
|---|---|
| Backdoor detection (TRACE-class, black-box) | AUROC ≥ 0.85, F1 ≥ 0.80 on synthesized BadDet corpus |
| Provenance tamper detection | 100% on bit-flip, substitution, replay, forgery tests |
| Data-integrity detection | ≥ 90% recall on ≥ 5% poison-rate synthetic attacks, controlled FPR |
| Drift characterization | Per-box AUROC ≥ 0.80 under OpenOOD-style protocol; calibrated ECE improvement |
| Air-gapped operation | Full install + run with network disabled |
| Report completeness | Every finding has severity, confidence, evidence, disposition, remediation |

---

## 11. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| H200 session too short | Medium | Checkpoint/resume in harness; priority-ordered phases; parallel jobs |
| Backdoor training unstable (poison rate too low) | Medium | Sweep 5%/10% rates; use verified BadDet/ODSCAN recipes |
| DISTIL too memory/time heavy | Medium | Optional phase; fall back to ODSCAN + Neural Cleanse only |
| Drift/adversarial separation ambiguity | High (inherent) | Explicitly report residual ambiguity; never claim perfect separation |
| Adaptive attacks evade detectors | High (research frontier) | Declare in coverage statement; frame as "raises attacker cost," not "prevents all attacks" |
| Air-gap dependency issues | Low | Phase 3 offline-install rehearsal before H200 |

---

## 12. References

1. TRACE: Test-Time Backdoor Detection for Object Detection Models — Zhang et al., CVPR 2025 (arXiv 2503.15293).
2. ODSCAN: Backdoor Scanning for Object Detection Models — Shen et al., IEEE S&P 2024.
3. DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion — Mirzaei et al., ICCV 2025 (arXiv 2507.22813).
4. BadDet: Backdoor Attacks on Object Detection — Luo et al., ECCV 2022 Workshop.
5. Backdoor Attacks and Defenses in Computer Vision Domain: A Survey — Abbasi et al., 2025 (arXiv 2509.07504).
6. Neural Cleanse — Wang et al., IEEE S&P 2019.
7. Activation Clustering — Chen et al., SafeAI/AAAI 2019.
8. Spectral Signatures — Tran et al., NeurIPS 2018.
9. Adversarial Attacks for Drift Detection — Hinder, Vaquet, Hammer, 2024 (arXiv 2411.16591).
10. OpenOOD v1.5 — Zhang et al., NeurIPS 2023 D&B.
11. C2PA 2.4 — Guidance for AI/ML, spec.c2pa.org, 2025.
12. NIST AI Risk Management Framework (AI RMF 1.0), 2023.
13. TrojAI Final Report — Reese et al., 2026 (arXiv 2602.07152).
14. BackdoorBench — SCLBD, NeurIPS 2022 D&B.
15. cleanlab ObjectLab — cleanlab 2.5+ documentation.

---

*Prepared for SIH 26228 — Ministry of Defence, Indian Army (DGIS). All experimental artifacts are sealed with the framework's own Ed25519 provenance module.*
