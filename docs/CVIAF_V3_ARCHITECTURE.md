# CVIAF v3 — Threat-Model-Conditional Assurance for Multi-Contributor CV Pipelines

**Design, rationale, and implementation roadmap for SIH 26228**
Ministry of Defence — Indian Army (DGIS) · Theme: Blockchain & Cybersecurity · Offline / air-gapped

> **Status.** Design document, 2026-09-28. Supersedes the architectural sections of `PRD_SIH26228_CVIAF.md` (v2) and `Assurance_Framework_Technical_Report.md`. The 2025–26 state-of-the-art survey in `CV_INTEGRITY_ASSURANCE_2026.md` is still current and is treated here as an input, not repeated.
> **Verification convention.** Claims carry `[VERIFIED]` (primary source fetched), `[SEARCH]` (found and consistent, not fetched in full), or `[ASSERTED]` (our design decision, no external source needed). Uncertain citations are listed as uncertain in §13 rather than smoothed over.

---

## 0. The one-paragraph version

v2 built a competent bag of detectors and a cryptographic seal. v3 changes what the system *is*: from a detector ensemble that emits verdicts, to a **threat-model-conditional assurance engine that emits warranted beliefs** — each one carrying its assumptions, its calibrated confidence, its measured detection floor, and the fraction of the applicable reference battery that actually executed. The technical spine is conformal calibration (every threshold is derived from a held-out calibration split inside a signed reference battery, never hand-tuned), dependence-robust evidence fusion with FDR control, hierarchical contributor attribution, and standards-aligned attestation so a third party can verify our outputs without running our code. This is a deliberate response to a theorem most teams will not have read: **universal, adversary-unaware backdoor detection is provably impossible** (Pichler et al., AISTATS 2024, `[VERIFIED]`). If universal detection is impossible, then a detector ensemble that claims it is not a better product — it is an unsound one. The winning design is the one that says precisely *what it can and cannot warrant*, and prices the difference.

---

## 1. What 26228 is actually asking for

### 1.1 Clause archaeology: the PS read as an acceptance test

The problem statement is not a feature list. Read the verbs:

| PS clause (near-verbatim) | What it demands | What a normal team will build | What it actually requires |
|---|---|---|---|
| "produce an **evidence-based** assessment of integrity and risk" | Every claim traceable to artifacts | a score | a claim with a provenance-linked evidence bundle |
| "**state the access assumptions, confidence and limitations**" | Epistemic disclosure, per finding | a disclaimer paragraph | per-finding `access_mode`, `calibrated_confidence`, `limitations[]` |
| "provide a **calibrated** risk or confidence score" | Frequencies must mean what they say | 0.87 from a sigmoid | conformal p-values / e-values with finite-sample validity on a declared calibration set |
| "**aggregate sample-level evidence into a source-level risk assessment**" | Contributor attribution, not sample flagging | mean of flags per contributor | hierarchical posterior on contributor risk, robust to dilution and small-n |
| "comparison against a **defined reference battery**" | A *defined, versioned* test suite | an ad-hoc test call | a content-addressed, signed Assurance Battery with ground-truth calibration splits |
| "distinguish probable operational drift from suspicious manipulation **where the available evidence supports such a distinction**" | Report under-determination honestly | a two-sided threshold | calibrated two-signal contrast + explicit `under_determined` verdict |
| "explicitly **declare** attack classes or conditions that it **does not support**" | Machine-readable non-coverage | a prose limitations list | a coverage matrix populated from what actually ran, keyed to an external taxonomy |
| "must operate **offline** … air-gapped" | Verifiable property | "no network calls" | a hashed offline bundle + a no-egress test in CI |
| "fall back **gracefully** … or clearly report that the relevant assessment is **unavailable**" | Never silently pass | try/except → warn | `accept` is *forbidden* when applicable checks did not run |

The through-line: **every one of these clauses is about the epistemic status of the output, not about the number of detectors.** The PS is a procurement document. Procurement cannot buy "accuracy"; it can buy "we tested X against Y and it flagged Z with a false-positive rate we can live with." v3 is built to be procurable.

### 1.2 The uncomfortable theorem

Three independent results, all verified:

1. **No-free-lunch for backdoor detection.** Pichler, Romanelli et al., *On the (In)feasibility of ML Backdoor Detection as an Hypothesis Testing Problem*, AISTATS 2024 (PMLR v238) `[VERIFIED]` — proves that universal (adversary-unaware) backdoor detection is impossible except in degenerate regimes. `gradientscience.org/rethinking-attacks` (2023) `[SEARCH]` reaches the same conclusion from the attack side: without assumptions on the attack, backdoor detection is not well-posed.
2. **Detection collapses in the low-poison-rate regime**, which is exactly the operational regime (an adversary who needs 10% of your dataset to be poisoned has already lost). *Reliable Poisoned Sample Detection against Backdoor Attacks*, ICLR 2026, M. Zhang et al. `[SEARCH]` — shows why pre-training poisoned-sample detectors fail at low poison rates.
3. **Two-window drift detectors are provably evadable.** Hinder, Vaquet, Hammer, arXiv 2411.16591 `[VERIFIED]` (via `CV_INTEGRITY_ASSURANCE_2026.md`) — a genuine drift whose per-window statistics match the reference raises no alarm.

**Why this is an advantage, not a problem.** A theorem that forbids the general claim *licenses* a precise one. Once you accept that detection is only meaningful relative to a declared threat model, three things become possible that are impossible otherwise:

- You can write an **acceptance test** (given attack class A, access level B, reference distribution C, detector achieves power ≥ 0.8 at FDR ≤ α for poison rate ≥ r).
- You can produce a **detection floor** ("we have no statistical power below 1.5% poisoning at n=300") — the honest number an operator needs.
- You can build a **conditional guarantee** where competitors build a false universal one, and you can *score* an adaptive attacker's cost rather than pretending to detect them.

### 1.3 Thesis

> **CVIAF v3 is a threat-model-conditional assurance engine. It never answers "is this asset malicious?" It answers "given the threat model you declared, the access you granted, and the reference distribution you supplied, here is what we found, how strongly the evidence supports each hypothesis, what we could not check, and what we recommend — with the whole chain cryptographically attestable and third-party verifiable offline."**

Every design decision below follows from that sentence.

---

## 2. Where v2 is attackable (self-critique before a judge does it)

We wrote v2. Here is what a sharp evaluator says about it.

| v2 claim / feature | The attack | v3 response |
|---|---|---|
| "F1 0.880 / AUROC 0.897 (TRACE)" as our performance claim | "That's *their* number on *their* corpus. What is *yours*, on hardware you had?" | Publish **our own** measured power curves per detector on our corpus, in the coverage matrix (§8), with the corpus digest. Cite TRACE as a method, never as our metric. |
| Neural Cleanse as the white-box path | "NC is classifier-shaped. Detectors have per-box class + localization; NC's assumption is wrong." | **Detector-native ladder**: TRACE (black-box test-time) → ODSCAN/DISTIL (white-box) → Neural Cleanse only as a *baseline to beat*, clearly labelled as the classifier-derived control. |
| Findings carry `confidence: float` | "Confidence from what? A heuristic?" | Confidence is a **conformal p-value → calibrated posterior** with a declared calibration set and its digest. Uncalibrated scores are labeled `confidence_type: "uncalibrated_heuristic"` and cannot drive a `quarantine`. |
| `severity` → `disposition` mapping | "Arbitrary. Where's the operator's risk appetite?" | **Cost-weighted decision rule** with a declared, editable loss matrix; report `expected_loss` per disposition. |
| Coverage statement is a static Python literal | "It doesn't change when your checks fail to run. So it can lie." | Coverage matrix is **generated per run** from executed checks; any skipped applicable check is a visible cell and suppresses `accept`. |
| Ed25519 seal, custom JSON | "Why should we trust *your* verifier? You just told us not to trust contributors." | **Standards-aligned attestation**: DSSE envelope + in-toto predicate + CycloneDX ML-BOM + C2PA-aligned media credentials. Third party verifies with standard tooling. |
| README Known Issues: keys regenerated each construction; HMAC fallback silent | "Your demo flags all 10 seals and your signatures aren't non-repudiable. This is a security product." | Day-1 fixes (§12 P0) plus a **hard-fail-closed** provenance mode and an OS-keystore/TPM key backend. |
| "Distinguish natural drift from adversarial" | "arXiv 2411.16591 says that's evadable." | Keep the claim but **bound it**: prefer within-window/block tests, emit an explicit `under_determined` verdict, and publish the evasion test that breaks us. |
| Remediation: fine-pruning | "AnywhereDoor shows fine-pruning takes clean mAP to ~18.3. You're offering sabotage." | Remediation menu re-ordered around **input-stage purification** (ODPure, arXiv 2609.28239 `[VERIFIED]`; Lite-BD) which preserves a continuous perception stream; fine-pruning demoted to opt-in with its measured mAP cost. |

**Verdict:** v2 is a good detector bag with a good seal and an unsound epistemic posture. v3 keeps the bag, upgrades the detectors, and rebuilds the posture.

---

## 3. Design axioms

Each axiom is a decision with a rejected alternative and a consequence. These are the sentences to defend at a judging table.

### A1 — Assurance is a conditional, never a verdict
**Decision.** Every result is emitted as the tuple `(claim, threat_model, access_mode, assumptions[], evidence[], calibrated_confidence, coverage_cell, detection_floor, recommended_disposition, expected_loss)`. There is no code path that emits a bare boolean.
**Why.** Pichler et al. `[VERIFIED]` — adversary-unaware universal detection is impossible. A verdict without its condition is therefore not conservative; it is unsupported.
**Rejected.** `is_backdoored: true/false`. Rejected because it cannot be defended under adaptive attack and gives the operator nothing to act on.
**Consequence.** The report schema is larger. Reviewers must read a threat model header before the findings. Accepted: that is the cost of being able to defend a finding.

### A2 — Every threshold is derived, not tuned
**Decision.** No magic constants. Each detector converts its raw statistic to a **conformal p-value** against a held-out calibration split inside the Assurance Battery (§4.3). Detection decisions use **Benjamini–Yekutieli**-controlled FDR by default (valid under arbitrary dependence; BH is available where positive dependence is argued). Contributor-level aggregation uses **e-values / conformal p-values** and a hierarchical model.
**Why.** The PS demands a *calibrated* score. Heuristic thresholds do not transfer across model families, datasets, sensors or poison rates, and cannot be audited. Conformal calibration is distribution-free and finite-sample valid under an exchangeability assumption we can state and test.
**Prior art (all `[SEARCH]`, consistent across sources).** RPP — certified poisoned-sample detection via conformal thresholds; FIS-FL — conformal p-values for *client-level* anomaly detection in federated settings (structurally our contributor problem); *Conformal Backdoor Detection in Multimodal Contrastive Learning*, arXiv 2608.04052 (Aug 2026).
**Rejected.** Hand-tuned z-score/kurtosis cut-offs (v2's approach). Rejected: no transfer guarantee, no auditability, no operator-tunable risk appetite.
**Consequence.** We must carry calibration splits and a battery. That is #A6, and it is worth it.

### A3 — Contributor risk is a posterior, not an average
**Decision.** Contributor risk = hierarchical Beta-Binomial with partial pooling. Report `P(θ_c > θ_pop + τ | data)`, posterior median flag rate, and a 95% credible interval. Rank contributors by posterior probability of exceeding a *risk tolerance*, and separately surface a **tail statistic** (max per-batch flag concentration), which does not shrink with volume.
**Why three things at once.** (i) The PS explicitly requires source-level aggregation. (ii) Partial pooling fixes small-n contributors (3 samples, 2 flags, is not evidence of malice). (iii) **It defeats a dilution attack that a naive mean is wide open to**: a contributor who floods 40 000 clean images to hide 400 poisoned ones *reduces their own flag rate*, because a rate is a ratio. Posterior probability of exceeding a tolerance plus a batch-level tail statistic is not dilutable that way.
**Rejected.** Mean flag rate (v2's "source aggregation"); 80% duplicate-share threshold; raw flag count.
**Consequence.** Defensible attribution, and a genuinely novel contribution: **we can name and defeat the dilution attack on contributor attribution.** No comparable team will have written this down.

### A4 — The disposition is a decision under asymmetric loss
**Decision.** `accept | review | quarantine` is computed from an explicit, editable **loss matrix** `L(summary_state, true_state)`, and the report shows `expected_loss` for every candidate disposition. Default matrix encodes: accepting a cloaked detector is catastrophic; quarantining a good contributor is costly but recoverable.
**Why.** The PS wants a "recommended disposition." The only defensible way to recommend an action is to minimize expected loss under a stated loss function. It also gives the operator a dial: DGIS can raise the cost of false-accept before a deployment, lower it during development.
**Rejected.** `severity → disposition` lookup table (v2). Rejected: uninterpretable, un-tunable, and it hides the operator's own risk appetite inside our constants.
**Consequence.** We must publish the default loss matrix and require it to be recorded in the audit log — the disposition is then reproducible and contestable.

### A5 — Detection is not mitigation; the menu must be survivable
**Decision.** Provide a **deployment-mode matrix**: `detect-only`, `detect+purify` (ODPure Corruption-Reconstruction-Selection, Lite-BD transformations), `detect+quarantine`, `detect+retrain` (opt-in, H200 only). Each mode reports its measured cost (latency, clean-mAP delta, residual ASR).
**Why.** AnywhereDoor (`[VERIFIED]` via checkpoint) shows input-transformation defenses fail and **fine-pruning cuts ASR but craters clean mAP to ~18.3** — recommending fine-pruning to an Army deployment is operationally negligent. ODPure (`[VERIFIED]`, arXiv 2609.28239, Sep 2026) is explicitly designed to preserve "a continuous and accurate perceptual stream," which is the military requirement: you cannot pause perception because a trigger appeared.
**Rejected.** Remediation = fine-pruning (v2-adjacent). Rejected on measured evidence.
**Consequence.** We must benchmark purification on the H200 too, not just detection. Adds one experiment phase; buys the only remediation we can honestly recommend.

### A6 — The reference battery is a first-class signed artifact
**Decision.** Define **Assurance Battery AB-1**: a content-addressed, signed bundle containing (a) probe image corpora, (b) transformation families with fixed seeds, (c) invariant expectations, (d) **clean and poisoned calibration splits with ground truth**, (e) attack-generation recipes (BadDet/BadDet+/AnywhereDoor-style) with seeds, (f) the loss matrix, (g) power-curve definitions. Every report declares `battery_id`, `battery_digest`, and which battery slices it used.
**Why the PS says the words "defined reference battery."** It is the only mechanism that makes three PS deliverables real at once: *reproducibility* (seeded, content-addressed), *confidence calibration* (ground-truth splits), and *longitudinal substitution detection* (a model enrolled today is compared against a fingerprint taken on the same frozen battery in six months).
**Rejected.** Per-run ad-hoc test sets. Rejected: destroys comparability, makes the "reproducible audit log" deliverable unprovable, and leaves substitution detection with no baseline (v2's acknowledged weakness: "first assessment has no baseline for comparison").
**Consequence.** Everything downstream becomes comparable and diffable. This is the single largest architectural upgrade in v3.

### A7 — Provenance is attestation, not a bespoke seal
**Decision.** Layer the provenance module on four standards, all with vendored offline verifiers:
- **DSSE** (Dead Simple Signing Envelope) as the signing wire format — `payloadType`, `payload`, `signatures[]`.
- **in-toto attestations** with a **pipeline layout**: the layout names functionaries (dataset contributor, model contributor, preprocessing implementation, inference service, assayer, timestamp authority) and the key each must sign with. This is the standard built for exactly the PS's "multi-contributor pipeline" phrasing `[SEARCH]`.
- **OpenSSF Model Signing (OMS) / Sigstore model-transparency** semantics for model artifacts: sign the **file manifest**, not the serialized tensor `[VERIFIED — openssf.org/projects/model-signing; sigstore/model-transparency]`. Non-obvious but essential: ONNX external-data and PyTorch zip archives are not byte-stable across tool versions, so tensor-level hashing produces false tamper alarms.
- **CycloneDX ML-BOM (AI-BOM)** for the ingredient inventory, and **C2PA 2.4** AI/ML content credentials for media outputs `[VERIFIED — cyclonedx.org/capabilities/mlbom]`.
Plus a Merkle-chained append-only log with **signed checkpoints** for the replay/sequence requirement.
**Why.** The PS's premise is *you cannot trust the contributor*. By the same logic the Army should not have to trust our verifier. Standard envelopes mean verification is possible with tooling we did not write.
**Rejected.** Bespoke JSON seal (v2, which is otherwise well-built). Rejected: not third-party verifiable; every verifier becomes a new trust root.
**Consequence.** More integration work, and a real payoff: "our assurance output is independently verifiable."

### A8 — A model is enrolled, not simply accepted
**Decision.** Formal lifecycle: `INTAKE → ENROLL → MONITOR → RE-ATTEST → (REVOKE)`. Enrollment runs the battery, records **four independent bindings**: (i) artifact digest (file manifest), (ii) behavioral fingerprint (battery-conditioned, compact, signed), (iii) ML-BOM ingredients, (iv) declared reference distribution. Subsequent assessments compare against the enrolled state.
**Why four bindings and not one.** They detect *different* attacks, and the cross-product is diagnostic:

| artifact digest | behavioral fingerprint | interpretation |
|---|---|---|
| match | match | consistent with enrolled model |
| **differ** | differ | substitution / retraining |
| differ | match | benign re-serialization (e.g. ONNX re-export) — **not** an attack |
| **match** | **differ** | **the pipeline is compromised, not the artifact** — runtime substitution, nondeterministic or altered execution, monkey-patched inference |

That last row is impossible to reach with a digest-only design, and it is the most operationally alarming signal in the whole framework: the model file is provably the one you signed, and it is behaving differently.
**Rejected.** Digest-only verification (v2). Rejected: cannot distinguish benign re-export from attack, and cannot see a compromised runtime at all.
**Consequence.** Enrollment becomes a mandatory, auditable ceremony — which is also the correct procurement workflow.

### A9 — Detector-native, and honest about what is partial
**Decision.** Detector-first method ladder (§5.2). Claim explicit coverage of **BadDet** OGA / RMA / GMA / ODA, and name **BadDet+** (arXiv 2601.21066 `[VERIFIED]`, physical-world, position/scale-invariant via log-barrier penalty) as *partially* covered with the reason. Never claim classifier methods transfer.
**Why.** TRACE's abstract states the mechanism directly: detectors' "ghost object emergence or vanishing object" effects "render current defenses fundamentally inadequate" `[VERIFIED — arXiv 2503.15293v2, revised 30 Jul 2026; 30% AUROC improvement over prior SOTA, resistance to adaptive attacks]`. OGA (fabrication) and ODA (disappearance/cloaking) have literally no classifier analogue.
**Rejected.** Neural-Cleanse-primary (v2). Rejected: wrong output geometry (single label vs per-box class + localization + NMS).
**Consequence.** We must benchmark the ladder ourselves, including the ODA case where CTC-based detection needs the FTC / "Island Effect" complement.

### A10 — Coverage is machine-readable, run-generated, and taxonomy-keyed
**Decision.** Replace the static `COVERAGE_STATEMENT` literal with a **coverage matrix** generated per run, keyed to **NIST AI 100-2e2025** terminology (`[VERIFIED]`, csrc.nist.gov/pubs/ai/100/2/e2025/final — the NIST adversarial-ML taxonomy: evasion, poisoning, privacy, and for generative AI) plus BadDet codes. Each cell: `covered | partial | not_covered | not_applicable`, `method`, `access_required`, `calibration_set`, `measured_detection_floor`, `residual_risk`, `evidence_pointer`.
**Why.** The PS names the coverage statement as a *deliverable*. A judge can diff two runs. Aligning to an external taxonomy means we are not grading ourselves in our own vocabulary.
**Rejected.** Static prose limitations (v2) and invented taxonomies.
**Consequence.** Any module that fails to run degrades a cell *and* the completeness metric, and can suppress `accept` (A11).

### A11 — You cannot accept what you could not check (the completeness invariant)
**Decision.** Hard system invariant: `assessment_completeness = executed applicable battery items / applicable battery items`. **`disposition = accept` is forbidden while completeness < 1.0 for any detector that bears on the claim.** Every `skipped` check emits a machine-readable reason enum plus a human reason. Silently degrading is a defect, not a fallback.
**Why.** The PS says methods requiring white-box access "must fall back gracefully **or clearly report that the relevant assessment is unavailable**." v2's `assess` degrades silently with a warning. Under the completeness invariant, the black-box fallback path can never emit a clean `accept` — the best it can say is "no flags found, within a reduced coverage envelope," which is the truth.
**Rejected.** Warn-and-continue. Rejected: a warning is not a coverage boundary, and it produces exactly the false reassurance the PS is written to prevent.
**Consequence.** This invariant is a one-line assertion with an outsized effect on defensibility. It is also trivially testable, so it will be *demonstrated live*.

### A12 — Red-team our own detectors, and publish the damage
**Decision.** The battery includes adaptive attacks written against CVIAF specifically: variance-flattening regularizers aimed at TRACE's CTC signal, poison-rate sweeps down to the statistical floor, contributor **dilution floods** (A3), valid-nonce **replay** inside the sequence window, and **benign re-serialization** to test the false-positive path. Report per-detector measured degradation in the coverage matrix. State adaptive-attack guarantees as **cost inflation**, not detection: "under a full-knowledge adaptive attacker, our measured ASR moves 0.62 → 0.71 while the attacker's poison budget grows ×N."
**Why.** A security artifact that has not been attacked by its own authors is a claim. The PS is filed under *Blockchain & Cybersecurity*; the judge is looking for adversarial thinking, not just detection numbers.
**Rejected.** Demonstrating only on attacks we designed our detectors for. Rejected: that is a demo, not an assurance product.
**Consequence.** Some published numbers will look worse than competitors'. That is the point, and it is what makes the good numbers credible.

---

## 4. System architecture

### 4.1 Layering

```
┌───────────────────────────────────────────────────────────────────────────┐
│ L7  ATTESTATION      DSSE envelopes · in-toto layout+predicates ·          │
│                      Merkle checkpoints · CycloneDX ML-BOM · C2PA media   │
├───────────────────────────────────────────────────────────────────────────┤
│ L6  GOVERNANCE       calibrated fusion → hypotheses → loss-weighted        │
│                      disposition · coverage matrix · completeness gate     │
├───────────────────────────────────────────────────────────────────────────┤
│ L5  EVIDENCE         conformal p-values / e-values · FDR control (BY) ·    │
│                      contributor hierarchy · power curves                  │
├───────────────────────────────────────────────────────────────────────────┤
│ L4  DETECTORS        plugin set, each with a capability manifest           │
│   data:  label · duplicate · OOD · trigger/spectral                       │
│  model:  TRACE · ODSCAN/DISTIL · DistScan · fingerprint/weight             │
│ prov:    seal · chain · replay · manifest                                   │
│ drift:   within-window two-sample · per-box OOD · shift typing             │
├───────────────────────────────────────────────────────────────────────────┤
│ L3  BATTERY          AB-1 slices: probes · transforms · calibration splits │
│                      attack recipes+seeds · loss matrix · power defs       │
├───────────────────────────────────────────────────────────────────────────┤
│ L2  ENROLLMENT       asset intake · 4 bindings · lifecycle state machine   │
├───────────────────────────────────────────────────────────────────────────┤
│ L1  ASSETS           COCO/YOLO datasets · ONNX/TorchScript models ·        │
│                      inference records · declared reference distribution   │
└───────────────────────────────────────────────────────────────────────────┘
        ▲ planner: chooses L4 detectors from L3 + L1 + declared threat model
```

### 4.2 The Assurance Planner (the "extensible framework" made real)

The PS says *extensible*. v2's extensibility is implicit (you could add a class). v3 makes it an explicit, auditable act.

A detector plugin declares a **capability manifest**:

```python
class DetectorPlugin(Protocol):
    id: str                      # "data.trigger.spectral"
    attack_classes: list[str]    # NIST AI 100-2 + BadDet keys it has power against
    access_required: AccessSpec  # dataset | model.black_box | model.white_box | log | ref_dist
    cost: CostModel              # forward_passes_per_unit, wall_clock_class, gpu_required
    calibration: CalibrationRef  # which AB-1 slice calibrates it, and its digest
    emits: Literal["p_value", "e_value", "statistic_with_floor"]

    def run(self, assets, battery_slice, budget) -> DetectorResult: ...
```

The planner then solves a small constrained optimisation: given (assets present, access granted, declared threat model, time/GPU budget, operator risk appetite), **select the detector subset that maximises coverage of the declared threat model subject to budget**, and record the plan (and every rejected alternative with its reason) in the audit log.

Why this matters beyond buzzwords: it converts "which checks ran?" from an accident into a **decision with a recorded rationale**. The audit log then explains *why* a coverage cell is empty — "drift.concept required labeled operational data; not supplied" — instead of merely noting absence. And it makes graceful degradation a *designed* behaviour: black-box-only intake produces a different *plan*, not a degraded afterthought.

### 4.3 Assurance Battery AB-1

| Slice | Contents | Used for | Size class |
|---|---|---|---|
| `AB1.probe` | frozen image set + transform family (background blends, focal patches, corruptions) | TRACE CTC/FTC, fingerprinting, purification sanity | a few hundred images |
| `AB1.cal.clean` | held-out clean items with ground truth | conformal calibration denominators | ≥ 500 |
| `AB1.cal.poison` | poisoned items at graded rates (0.5%, 1%, 2%, 5%, 10%) with known indices | power curves, detection floors | ≥ 2000 |
| `AB1.cal.models` | small corpus of clean + BadDet-typed detectors | detector power on the model axis | 8–16 models |
| `AB1.ref` | declared reference distribution summaries + sample | drift/OOD calibration | dataset-dependent |
| `AB1.attacks` | reproducible attack recipes: BadNets, blended, WaNet-class, label-flip patterns, dup floods, OOD injection, BadDet OGA/RMA/GMA/ODA, BadDet+ penalty, dilution floods, replay, benign re-export | red-team + power curves | seeds + configs |
| `AB1.policy` | loss matrix, τ tolerances, FDR targets, budget profiles | decision layer | JSON |

The whole battery is content-addressed: `battery_digest = SHA-256(canonical manifest)`. Reports carry it. Model fingerprints are battery-conditioned, so a fingerprint comparison across time is only valid for the same digest — and the framework *refuses* (rather than silently mis-compares) when the digests differ, reporting `fingerprint_incomparable` and forcing re-enrollment. That refusal is another "never silently pass" instance.

### 4.4 Enrollment and the four bindings

```
INTAKE ──▶ validate format (COCO/YOLO; ONNX/TorchScript) + record provenance of the contribution
       ──▶ ENROLL: ① artifact manifest digest  ② behavioral fingerprint on AB1.probe
                   ③ ML-BOM ingredients        ④ declared reference distribution + dataset fingerprint
       ──▶ sign an in-toto attestation (DSSE) naming the contributor key
MONITOR ──▶ drift + re-fingerprint on schedule; any fingerprint drift ⇒ RE-ATTEST required
RE-ATTEST ─▶ re-run battery; fingerprint distance vs enrolled → substitution verdict (A8 table)
REVOKE   ──▶ signed revocation entry in the append-only log; downstream verification fails closed
```

**Dataset enrollment matters too, and v2 omits it.** The PS says a dataset is a contributed asset. So: `dataset_fingerprint = H( per-contributor class histograms ‖ embedding-space summary ‖ dedup-graph hash ‖ manifest)`. This catches silent modification of a dataset *between intake and training* — the exact window in which poisoning usually happens — and enables **retro-attribution**: if a deployed model is later found backdoored, the hash-chained contribution records tell you which contributor's samples the trigger rode in on. That is "poison forensics," and it is what makes the audit log operationally valuable rather than ceremonial.

### 4.5 Detector plugin protocol

Every plugin must emit one of:

- `ran` with `{statistic, p_value|e_value, calibration_ref, cost_actual, evidence}`
- `skipped` with `{reason_code ∈ {ACCESS_DENIED, ASSET_ABSENT, REFERENCE_MISSING, BUDGET_EXHAUSTED, NOT_APPLICABLE, DEPENDENCY_MISSING}, human_reason, coverage_impact}`

Third state `error` is treated as `skipped` for coverage purposes and *always* surfaced. There is no fourth state, and in particular there is no state in which a plugin returns a number without saying where the number's null distribution came from.

---

## 5. Module specifications

### 5.1 `data_integrity` — training-data integrity with contributor attribution

**Objective.** Flag trigger injection, label flipping, systematic mislabelling, near-duplicate flooding and OOD insertion; aggregate to contributor-level risk; do it on COCO/YOLO annotations without retraining.

| Sub-detector | Method | Calibrated? | Attack class |
|---|---|---|---|
| `label.objectlab` | cleanlab `object_detection.ObjectLab`: wrong class, missed box, badly drawn box, using model outputs + given labels `[VERIFIED — cleanlab docs]` | yes, on `AB1.cal.clean` | label flipping, systematic mislabel |
| `label.correction` | detection→correction framework, arXiv 2508.06556 `[SEARCH]` | — | remediation suggestion (not auto-applied) |
| `dup.phash` | `imagededup` perceptual hash — cheap first pass | yes | flooding |
| `dup.sscd` | SSCD embeddings + FAISS cosine; cluster, attribute to contributor `[VERIFIED — CVPR 2022, facebookresearch/sscd-copy-detection]` | yes | near-dup flooding |
| `ood.perbox` | Mahalanobis / KNN per **RoI**, not per image, under the OpenOOD protocol `[VERIFIED — OpenOOD v1.5, arXiv 2306.09301]` | yes | OOD insertion |
| `trigger.pixel` | patch-region variance collapse | yes | static patches |
| `trigger.fft` | high-frequency energy anomalies | yes | blended triggers |
| `trigger.spectral` | **SPECTRE** robust-covariance + whitening on per-box RoI features (upgrade from PCA+kMeans) `[VERIFIED — survey arXiv 2509.07504]` | yes | poisoned subsets |
| `contrib.hier` | hierarchical Beta-Binomial posterior + batch-level tail statistic | — | attribution (A3) |

**Two design decisions worth defending.**

1. **Per-RoI, not per-image.** Detector features must be taken at box/RoI granularity. v2 uses "bbox deep features," which is right; make it explicit that the Mahalanobis/OOD scoring is per-RoI, because an image-level OOD score on a detector is dominated by scene statistics and washes out the object-level signal.
2. **The dilution defense (A3).** Any statistic that is a *ratio over the contributor's volume* is attacker-controlled. So contributor risk is (posterior probability of exceeding tolerance) + (max batch-level flag concentration) + (flag-count lower confidence bound). The report names which of the three triggered. This is a small amount of math that closes a real hole.

**Power disclosure (A2, A12).** For each data detector we publish: minimum detectable poison rate at FDR ≤ 0.05 with power ≥ 0.8, as a function of `n` and of the detector. If a dataset has 300 images and the floor is 1.5%, the report says: *"we have no statistical power to exclude poisoning below 1.5% in this dataset."* Most teams will not write this sentence. It is the most valuable sentence in the report.

### 5.2 `model_integrity` — detector-native access ladder

**The ladder, in planner priority order:**

| Tier | Detector | Access | Purpose | Notes |
|---|---|---|---|---|
| 0 | `model.distscan.prenms` | black-box | cheap always-on pre-filter: backdoor training shifts intermediate class-prediction distribution away from training class frequencies, visible pre-NMS `[VERIFIED — arXiv 2608.19088]` | first thing to run; near-zero cost |
| 1 | `model.trace.ctc` | black-box, no training data | FP-inducing triggers (OGA/RMA/GMA): triggered objects are *abnormally stable* across blended backgrounds → low confidence variance | needs public background/foreground images — **must be vendored into the battery for air-gap** (explicit limitation of TRACE) |
| 2 | `model.trace.ftc` | black-box | FN-inducing / cloaking (ODA): slide a Natural Backdoor Object patch, watch its confidence collapse over a hidden trigger — "Island Effect" | the *only* test-time handle on disappearance; do not omit |
| 3 | `model.odscan` | white-box | trigger scanning/reconstruction, S&P 2024 `[VERIFIED — Megum1/ODSCAN]` | white-box baseline to beat |
| 4 | `model.distil` | white-box, **data-free** | latent-diffusion trigger inversion `[VERIFIED — arXiv 2507.22813, ICCV 2025]`; ideal when weights exist but contributor data does not | >24 GB VRAM; H200 only |
| 5 | `model.fingerprint` | black-box | behavioral fingerprint vs enrolled state (A8 table) | the substitution logic |
| 6 | `model.weights` | white-box | weight-statistics anomalies; Z-PEFT-style spectral signatures for the PEFT/adapter case `[VERIFIED — arXiv 2608.02271]`, labeled PEFT-scoped | not a general OD backdoor detector — say so |
| 7 | `model.neuralcleanse` | white-box | **baseline control only**, clearly labelled classifier-derived | keep to *show* it is weaker on detectors, not to rely on it |

**Deliverable from this module that no one else will produce:** the **fingerprint**, defined concretely as the vector
`[per-class confidence quantiles on AB1.probe ‖ box-count distribution ‖ IoU-stability under AB1 transforms ‖ pre-NMS class-prior vector]`,
quantized to a declared tolerance, hashed, and signed. Compare with a distance that has a calibrated null distribution from repeated enrollment of the *same* clean model (so the tolerance is measured, not guessed).

**Access-mode honesty.** Every tier reports `mode ∈ {white, black, gray}` and `unavailable` reasons. TRACE-based tiers are the *default* because they need neither weights nor the contributor's training data — the precise configuration a buyer of a third-party model actually has. That is why TRACE is the headline and not a nice-to-have.

### 5.3 `provenance` — attestation layer

**What is bound:** `H(image) ‖ model_manifest_digest ‖ preproc_config_hash ‖ H(canonicalized output) ‖ sequence ‖ nonce ‖ prev_entry_hash ‖ execution_environment`.

**Six deliberate decisions.**

1. **Canonicalized output hashing.** v2 admits float non-determinism can break output verification across hardware. Fix: hash a **declared canonicalization** — detections sorted by a stable key, coordinates quantized to a declared tolerance, class ids as integers — with the `quantization_spec` signed alongside. A hash break then means tampering or a genuine configuration divergence, not the GPU's reduction order. Also record `execution_environment` (device, library versions, determinism flags, seeds) so a mismatch *explains* a break rather than falsely accusing.
2. **Sequence + nonce + chain.** Prevents replay and reordering. Monotonic counter is primary (no NTP in an air gap). Wall clock is explicitly labelled advisory.
3. **Merkle checkpointing.** Periodically sign the Merkle root of the log. A verifier then needs only the root + inclusion proof to confirm a record, which is what makes the log *shippable* rather than a 40 GB file. This is the property people actually want when they say "blockchain," implemented without a consensus protocol.
4. **Hard-fail-closed on symmetry.** If the asymmetric backend is absent, provenance refuses to sign unless `--allow-symmetric` is explicitly passed; the report then carries `non_repudiation: false` in the *evidence* of every seal-derived finding. v2's silent HMAC fallback is a defect that turns a non-repudiation claim into a shared secret.
5. **Key separation and ceremony.** Distinct key roles (contributor, preprocessing, inference, assayer, TSA, release) with a k-of-n policy for overriding a `quarantine`. Optional PKCS#11/TPM backend for hardware-backed keys. Private keys never in the repository (v2 currently commits demo private keys; `.gitignore` is empty).
6. **Offline timestamp authority.** A designated offline signer co-signs (a second in-toto functionary) to provide ordering evidence where wall-clock is untrustworthy. This also demonstrates separation of duties, which is a governance requirement in its own right.

**The blockchain question, answered deliberately.** The PS theme is "Blockchain & Cybersecurity," so a judge will ask. Our answer: we implement the property that matters — **tamper-evidence with verifiable inclusion proofs on an append-only, signed, Merkle-rooted log** — and we deliberately omit consensus, because in a single-authority air-gapped deployment a consensus protocol adds cost, operational burden, and a new attack surface while providing no additional trust. The log format is deliberately *anchor-compatible*: if DGIS later wants cross-organization notarisation, checkpoint roots can be anchored into a permissioned ledger without redesigning anything. Shoehorning a chain into an air gap would be designing for the theme rather than the requirement. **Stating this explicitly is stronger than pretending.**

### 5.4 `drift` — shift characterisation with an honesty valve

- **Split the report**: covariate `P(X)` vs concept `P(Y|X)` vs prior `P(Y)`, each declared with what evidence licenses that split (labels required for concept, etc.).
- **Prefer within-window / block formulations.** Two-window detectors are provably evadable (arXiv 2411.16591 `[VERIFIED]`). Say so in the report, in the finding's own limitations, not only in a global list.
- **Per-box scoring + calibration.** Detector confidences are miscalibrated (IJCV 2024 `[SEARCH]`); apply temperature/vector scaling *before* thresholding, and report ECE before/after.
- **The drift-vs-manipulation contrast, made falsifiable.** Two calibrated signals:
  - *geometry*: anomalous cluster tightness and direction in feature space (manipulation tends to concentrate in a narrow direction);
  - *confidence polarization*: a trigger typically forces over-confidence on the anomalous cluster, whereas natural covariate shift broadly *degrades* confidence.
  When the signals agree → label (`probable_drift` / `suspicious_manipulation`) with calibrated confidence. When they disagree → emit **`under_determined`** and say what extra evidence would resolve it (e.g. a labeled operational sample). The PS's own wording — "where the available evidence supports such a distinction" — is thus rendered as a *third verdict*, which is precisely what it asks for and what v2 lacks.

### 5.5 `governance` — the fusion calculus and the decision

**Evidence fusion without double counting.** Our detectors are correlated (spectral and FFT both look at high-frequency structure; per-box Mahalanobis and OOD score overlap). Naive averaging double-counts. Design:

1. Each detector emits a conformal **p-value** `p_d` (or an **e-value** where a likelihood ratio is the natural output).
2. Combine with the **Cauchy combination test** — chosen specifically because it is valid under *arbitrary* dependence between p-values, which is exactly our situation and removes the need to estimate a covariance matrix from an underpowered calibration split.
3. Control FDR with **Benjamini–Yekutieli** by default (valid under arbitrary dependence; costs a `log n` factor) and record that choice, since it is a deliberate conservatism. BH is available when positive dependence is argued and recorded.
4. For the *sample-level* flag set: FDR-controlled rejection set with per-flag `q`-values reported, so the operator sees "we expect ≤5% of these 40 flags to be false."
5. For the *contributor* level: A3's hierarchical posterior.
6. For the *disposition*: A4's expected loss under the declared loss matrix.

**Why this is worth doing instead of a score.** It yields a report in which every number has a stated meaning: `p` is a p-value under declared exchangeability; `q` is an FDR; the contributor score is a posterior probability; the disposition minimizes expected loss. Nothing in the report is a vibe.

**Governance outputs.** `coverage_matrix` (A10), `assessment_completeness` (A11), `threat_model_declared` (echoed back so the operator sees what was assumed), `loss_matrix_used`, `hypothesis_posteriors`, and `audit_trail` bound to the provenance Merkle root so the report is itself tamper-evident.

---

## 6. The mathematics, stated plainly (for the write-up and the viva)

- **Conformal p-value.** For a detector with nonconformity score `r(x)` calibrated on `n` clean items, `p(x) = (1 + Σ_i 1[r(x_i) ≥ r(x)]) / (n + 1)`. Valid under exchangeability of calibration and test items; the report declares this assumption and lists what breaks it (temporal correlation, contributor-level clustering — which is why calibration splits are drawn *by contributor*, not by sample).
- **That last point is a real trap and we avoid it.** If calibration items come from the same contributor as the test items, exchangeability fails and p-values are optimistic. So calibration splits are **contributor-disjoint**. Writing this down shows we understand conformal validity rather than cargo-culting it.
- **Cauchy combination.** `T = Σ_d w_d · tan((0.5 − p_d)·π)`, `p_comb = 0.5 − arctan(T)/π`. Valid under arbitrary dependence of the `p_d`.
- **FDR.** BY: reject the `k` smallest p-values where `p_(k) ≤ k·α / (n·Σ_{i=1}^{n} 1/i)`.
- **Hierarchical contributor model.** `f_c ~ Beta(α, β)` population; `θ_c ~ Beta(κ f_c, κ(1 − f_c))`; `k_c ~ Binomial(n_c, θ_c)`. Report `P(θ_c > f_pop + τ | k_c, n_c)` via posterior sampling (a few thousand draws is instant and offline).
- **Expected loss disposition.** `d* = argmin_d Σ_s P(s | evidence) · L(d, s)`.
- **Detection floor.** Minimum poison rate `r*` such that the power curve `Power(r) ≥ 0.8` at the chosen `α`, estimated empirically by spiking `AB1.cal.poison` at graded `r`.

Each formula is small, offline, and cheap. Collectively they are the difference between "we compute a score" and "we state a warranted belief."

---

## 7. Assurance report schema v3

Additive to `schemas/sample-assurance-report.json` (v2 shape preserved so existing consumers do not break):

```jsonc
{
  "schema_version": "3.0",
  "report_id": "...", "timestamp": "...", "framework_version": "3.0.0",
  "threat_model_declared": {
    "attack_classes_in_scope": ["badet.OGA", "badet.ODA", "nist.poisoning.backdoor", "..."],
    "access_granted": { "dataset": "full", "model": "black-box", "log": "present" },
    "reference_distribution": { "id": "ref.terrain.north", "digest": "..." },
    "assumptions": ["calibration items contributor-disjoint", "no NTP; monotonic counters"],
    "declared_by": "operator", "policy_ref": "AB1.policy"
  },
  "battery": { "id": "AB-1", "digest": "...", "slices_used": ["AB1.probe","AB1.cal.clean","AB1.cal.poison"] },
  "plan": {
    "planner_version": "1.0",
    "selected": ["model.distscan.prenms", "model.trace.ctc", "..."],
    "rejected": [{ "detector": "model.distil", "reason": "NO_GPU_OR_UNMET_VRAM", "coverage_impact": "white-box inversion unavailable" }]
  },
  "assessment_completeness": {
    "applicable": 11, "executed": 9, "fraction": 0.818,
    "accept_permitted": false,
    "blocking_skips": ["drift.concept: REFERENCE_LABELS_MISSING"]
  },
  "hypothesis_posteriors": [
    { "hypothesis": "asset.backdoored", "posterior": 0.83, "evidence_refs": ["F-3","F-7"] },
    { "hypothesis": "asset.clean",      "posterior": 0.04, "evidence_refs": [] }
  ],
  "assessments": { /* per-module detail, as v2 */ },
  "findings": [
    {
      "finding_id": "F-3", "module": "model_integrity",
      "attack_class": "badet.ODA", "taxonomy": { "nist_ai_100_2": "poisoning.backdoor", "badet": "ODA" },
      "severity": "HIGH", "disposition": "quarantine",
      "method": "model.trace.ftc", "access_mode": "black-box",
      "confidence": { "value": 0.91, "type": "conformal_p_value", "calibration_ref": "AB1.cal.clean#d41f" },
      "detection_floor": { "metric": "min_trigger_area_px2", "value": 196, "power": 0.82, "alpha": 0.05 },
      "evidence": { /* artifacts, digests, pointers */ },
      "affected_assets": ["model:detector.onnx#sha256:..."],
      "limitations": ["CTC signal is defeatable by variance-flattening regularizers (measured: §9)"],
      "expected_loss": { "accept": 0.71, "review": 0.22, "quarantine": 0.08 },
      "remediation": { "recommended_mode": "detect+purify", "options": [
        { "mode": "detect+purify", "clean_map_delta": -0.4, "residual_asr": 0.11 },
        { "mode": "detect+retrain(fine-prune)", "clean_map_delta": -41.2, "residual_asr": 0.19, "warning": "degrades clean mAP severely" }
      ]}
    }
  ],
  "coverage_matrix": [ /* one row per (module, attack class), populated from executed checks */ ],
  "loss_matrix_used": { "accept|backdoored": 100.0, "quarantine|clean": 3.0, "...": 0 },
  "attestation": {
    "dssE_attestation": "....", "in_toto_layout_ref": "...",
    "audit_log_merkle_root": "...", "checkpoint_signature": "...",
    "ml_bom_ref": "bom.cdx.json#sha256:...", "non_repudiation": true
  }
}
```

Three fields to call out in the pitch, because they are the ones that will not appear in any competitor's schema:
`assessment_completeness.accept_permitted` (§A11), `detection_floor` per finding (§A2/A12), and `plan.rejected[]` (§4.2).

---

## 8. Coverage matrix and threat model

Keyed to **NIST AI 100-2e2025** `[VERIFIED]` (adversarial-ML taxonomy: evasion / poisoning / privacy; generative-AI abuse) and **BadDet** codes, with `BadDet+` and cloaking called out as partial.

| Taxonomy key | Attack | Detector(s) | Access | Status | Honest residual risk |
|---|---|---|---|---|---|
| `poisoning.backdoor.badet.OGA` | phantom box generation | TRACE CTC, DistScan, ODSCAN | black/white | covered | adaptive variance-flattening measured to degrade (§9) |
| `poisoning.backdoor.badet.RMA` | in-place regional reclass | TRACE CTC, ODSCAN | black/white | covered | — |
| `poisoning.backdoor.badet.GMA` | global reclass | TRACE CTC, fingerprint | black | covered | class-prior-shifting variants weak |
| `poisoning.backdoor.badet.ODA` | disappearance / cloaking | TRACE **FTC/Island Effect** | black | covered | FTC's NBO patch must be in-scope for the scene's object vocabulary |
| `poisoning.backdoor.physical` | BadDet+ penalty, position/scale-invariant | TRACE + ODPure purification | black | **partial** | physical robustness is the explicit open problem in the 2026 literature |
| `poisoning.availability.clean-label` | label-consistent / WaNet-class | spectral on per-box RoI, low power | white | **partial** | low power at low poison rates; floor published |
| `poisoning.targeted.label-flip` | systematic mislabel | ObjectLab + contributor hierarchy | dataset | covered | needs ≥ ~50 samples/class; floor published |
| `poisoning.availability.dup-flood` | near-dup flooding (+ dilution) | PHash → SSCD → hierarchy | dataset | covered | dilution defense per A3 |
| `poisoning.availability.ood-inject` | OOD insertion | per-RoI Mahalanobis/KNN (OpenOOD) | dataset + ref | covered | needs declared reference distribution |
| `evasion.test-time` | single-input adversarial | — | — | **not covered — by design** | out of scope: this framework assures *integrity of assets*, not per-input robustness |
| `privacy.*` | membership inference etc. | — | — | **not applicable** | different problem class; declared, not faked |
| `supply-chain.model-substitution` | swapped weights | digest + fingerprint 4-binding matrix | black | covered | benign re-export properly classified as non-attack |
| `supply-chain.runtime-compromise` | digest matches, behavior shifts | fingerprint-only mismatch row | black | covered | **unique to v3** |
| `supply-chain.tamper-log` | post-hoc record edit | DSSE + chain + checkpoint | log | covered | unsealed records cannot be protected — declared |
| `supply-chain.replay` | re-submission | sequence + nonce | log | covered | inside-window replay requires nonce discipline |
| `drift.covariate` | terrain/season/sensor/illum | within-window two-sample, per-box | ref + ops | covered | two-window evadable — stated |
| `drift.concept` | P(Y\|X) change | requires labeled ops data | ref+ops labels | conditional | unavailable without labels — reported |
| `drift.vs-manipulation` | adversarial shift | two-signal contrast | ref + ops | **under-determined by design** | third verdict emitted when signals conflict |

The last row is the point. Everyone else will emit one of two labels. We emit three, and we can say why.

---

## 9. Evaluation, and our own red team

**Corpus.** 14–16 detectors (clean + BadDet OGA/RMA/GMA/ODA at 5% and 10% poison; BadDet+ penalty variant): YOLOv5s / YOLOv8n / Faster-RCNN-R50 / DETR-class, on COCO-subset and VOC. Plus a small corpus of **benign re-exports** (ONNX↔TorchScript round-trips) to test the false-positive path in A8.

**Metrics, all reported with CIs on the AB-1 battery:**
- detection: AUROC, F1, **FPR@95TPR**, and **per-attack-class recall**; plus **power curves** vs poison rate and trigger area.
- calibration: ECE and reliability diagrams **before and after** temperature/vector scaling.
- fusion: does the Cauchy-combined p-value's empirical FDR match its nominal FDR on the calibration splits? (If not, the fusion is broken and we publish that.)
- provenance: tamper-detection rate on bit-flip / substitution / reorder / replay / forgery / benign-re-export; verification latency; Merkle proof size.
- completeness invariant: property test that `accept` is never emitted with `completeness < 1.0`, across a matrix of missing-asset and missing-access conditions.
- system: wall-clock, peak VRAM, offline-install reproducibility (bundle digest match).

**Adaptive red team (A12) — the part that earns credibility.**
1. *CTC-evasion*: fine-tune the backdoored detector with a variance-regularizer on the trigger → measure TRACE AUROC drop.
2. *Poison-rate sweep*: 0.1% → 10% → publish the floor curve; explicitly show where we fail.
3. *Dilution flood*: malicious contributor adds 40 000 clean images → show naive rate collapses and A3's statistic does not.
4. *Replay inside window*: valid nonce reused → should be caught by sequence/nonce; if not, report the gap.
5. *Benign re-export*: must **not** raise a substitution finding (false-positive test).
6. *Under-powered dataset*: 200 images with 1% poisoning → the report must say "insufficient power," not "clean." This is the single most important test in the suite, because it is the failure mode the PS is written to prevent.

---

## 10. Air-gap engineering (making "offline" a verifiable property)

| Requirement | Mechanism | Evidence artifact |
|---|---|---|
| no cloud, no APIs | vendored wheels + pinned-commit repos + pre-downloaded weights | `offline-bundle/` with `MANIFEST.sha256` and a signed bundle attestation |
| no NTP | monotonic counters + logical clock + offline TSA co-signature | report field `time_source: monotonic+tsa` |
| no egress *provably* | CI job runs the full `demo` and `assess` under network-namespace isolation / socket-blocking sandbox; asserts zero egress | `docs/evidence/no-egress.txt` from the sandboxed run |
| reproducible install | `uv pip install --no-index --find-links offline-bundle` | lockfile + bundle digest |
| asset availability for black-box methods | TRACE's public background/foreground images, SSCD weights, OpenOOD stats **vendored into AB1.probe** | `AB1.probe` digest (this closes v2's stated limitation: "TRACE relies on auxiliary public images (unavailable in a hard air-gap unless pre-vendored)") |
| key custody | keystore outside repo; optional PKCS#11/TPM; `.gitignore` fixed | `cviaf doctor` output; no `*.priv` tracked |

The no-egress test is a two-hour task with an outsized effect on a security-themed judging panel: it converts a claim into a demonstration.

---

## 11. Deliberate non-decisions (what we refuse, and why)

Explicit refusals are design decisions and should be presented as such.

1. **No blockchain/consensus.** Property implemented (Merkle-rooted, signed, append-only, inclusion-provable); consensus omitted with reason (§5.3). Anchor-compatible for later.
2. **No universal detection claim.** Provably impossible (Pichler et al. `[VERIFIED]`). We publish floors and scope instead.
3. **No retraining in the baseline path.** PS constraint. Retraining appears only as an opt-in remediation mode with its measured cost.
4. **No cloud, no KMS, no NTP, no external API — ever.** Not even as a fallback.
5. **No auto-remediation of datasets or models.** Detection→correction is offered as a *suggestion with evidence*, never applied silently. Silently rewriting a contributor's data destroys the audit trail that makes attribution possible.
6. **No per-input adversarial-robustness claim.** Different problem; declared not-covered rather than blurred (see coverage matrix `evasion.test-time`).
7. **No accuracy claims about the detectors we cannot reproduce.** TRACE/ODSCAN/DISTIL numbers are cited as *theirs*; ours are measured and published.

---

## 12. Roadmap, mapped to the PS's named deliverables

The PS names five deliverables: **source code**, **architecture & setup notes**, **assurance-report schema**, **reproducible audit log**, **coverage statement** (with attack classes, assumptions, limitations). Everything below terminates in one of them.

| Phase | Work | Deliverable it feeds | Acceptance test |
|---|---|---|---|
| **P0 — integrity of what exists (days)** | Fix provenance key reuse (load existing keypair instead of regenerating per construction); hard-fail-closed on symmetric fallback; un-track `*.priv`, populate `.gitignore`; add `tests/` (currently empty, `pytest` collects nothing) | source code | `cviaf demo` reports **all seals valid**; a signing attempt without `cryptography` **refuses** unless `--allow-symmetric`; `git ls-files | grep priv` empty |
| **P1 — epistemic core (1–2 weeks)** | `core/calibration.py` (conformal p-values, BY/BH, Cauchy combination); `core/fusion.py`; `core/decision.py` (loss matrix, expected loss); hierarchical contributor model; generalize `Finding.confidence` to the structured object | schema, coverage statement | unit tests: empirical FDR on synthetic data within tolerance of nominal; `accept_permitted == false` whenever any applicable check skipped |
| **P2 — battery & planner (1–2 weeks)** | AB-1 build + signing; `DetectorPlugin` protocol + capability manifests; planner with recorded rationale; `assess --threat-model file.yaml` | architecture & setup notes, reproducible audit log | same inputs + same battery digest ⇒ byte-identical report modulo timestamps; planner rationale present for every non-selected detector |
| **P3 — attestation (1–2 weeks)** | DSSE envelope; in-toto layout with 6 functionaries; OMS-style model manifest hashing; CycloneDX ML-BOM emission; C2PA-aligned media credential; Merkle checkpoint signing | audit log, source code | independent Python script (no CVIAF import) verifies a DSSE attestation, an inclusion proof, and the ML-BOM |
| **P4 — detector upgrades (2–3 weeks, H200)** | TRACE CTC+FTC; DistScan pre-NMS; ODSCAN; per-RoI OpenOOD; SPECTRE on RoI; SSCD; ObjectLab; enrollment fingerprint | source code, coverage statement | detector power curves published per attack class; ODA caught via FTC |
| **P5 — red team & calibration (1 week, H200)** | A12 adaptive suite; ECE before/after; floors and power curves; frozen results | coverage statement | the under-powered-dataset test returns "insufficient power," not "clean"; adaptive degradation published |
| **P6 — offline proof & packaging (days)** | offline bundle, no-egress sandboxed run, `cviaf doctor --offline`, docs | architecture & setup notes | full pipeline runs with networking blocked; `git`-clean install from bundle only |

**Deliberate sequencing note.** P1/P2 come *before* P4. A better detector bolted onto an uncalibrated report is still an unsound product; the calibration and battery layers are what make the detectors' numbers mean anything. Teams will invert this order because detectors are more fun to demo. Inverting it is the trap.

---

## 13. Citation ledger

**Verified (primary source fetched, 2026-09-28 or 2026-09-11).**
Pichler et al., *On the (In)feasibility of ML Backdoor Detection as an Hypothesis Testing Problem*, AISTATS 2024, PMLR v238 · NIST AI 100-2e2025, *Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations* · TRACE, arXiv 2503.15293v2 (rev. 30 Jul 2026) · ODPure, arXiv 2609.28239 (23 Sep 2026) · BadDet+, arXiv 2601.21066 (28 Jan 2026) · DISTIL, arXiv 2507.22813 (ICCV 2025) · ODSCAN, IEEE S&P 2024 · DistScan, arXiv 2608.19088 · Lite-BD, arXiv 2602.07197 · Z-PEFT, arXiv 2608.02271 · OpenOOD v1.5, arXiv 2306.09301 · SSCD, CVPR 2022 · cleanlab ObjectLab · C2PA 2.4 AI/ML · CycloneDX ML-BOM (cyclonedx.org/capabilities/mlbom) · OpenSSF Model Signing + sigstore/model-transparency · in-toto attestation framework + SLSA provenance · NIST AI RMF 1.0 · TrojAI final report, arXiv 2602.07152 · BackdoorBench.
*(Full per-source detail with verification method is in `CV_INTEGRITY_ASSURANCE_2026.md` §11 and `RESEARCH_CHECKPOINT_26228.md`, which remain authoritative for the 2025–26 survey.)*

**Search-verified, consistent across sources, not fetched in full — treat as medium confidence.**
Conformal family: RPP (certified poisoned-sample detection via conformal thresholds); FIS-FL (conformal p-values for client-level anomaly detection, ACM/Springer 2026); *Conformal Backdoor Detection in Multimodal Contrastive Learning*, arXiv 2608.04052. · *Reliable Poisoned Sample Detection against Backdoor Attacks*, ICLR 2026 (low-poison-rate failure). · *Detecting and Eliminating Adaptive Backdoor Attacks*, arXiv 2508.04094 (adaptive attacks). · AEGIS, preprints.org 202608.2261 (real-time latent-space backdoor detection, low FPR). · *From Label Error Detection to Correction*, arXiv 2508.06556. · TCF-CBM, Nature Sci Rep 2026 (three-stage poisoned-sample detection). · *A two-stage cascaded purification framework for OD backdoor*, Nature Sci Rep 2026. · Miah et al., progressive-neuron pruning, Neurocomputing 2026.

**Explicitly uncertain / context-only — do not build a claim on these.**
AnywhereDoor (arXiv 2503.06529) was **withdrawn** by its author; the code repository exists. Use as threat-model inspiration only, never as an authoritative citation. · LeBD is abstract-only (PDF gated); cite the LayerCAM mechanism, invent no numbers. · "TRIM" could not be verified to exist and is dropped. · TRACE reports an internal inconsistency between 42 and 63 backdoored models across revisions; do not cite a model count.

**Removed as unverifiable rather than dressed up.** Where v1/v2 research could not confirm a source, it is listed here as unverifiable instead of being presented with false confidence. Keeping this section is itself part of the design: a framework that asks for calibrated confidence from its contributors must publish its own.

---

*End of v3 architecture. Companion documents: `PRD_SIH26228_CVIAF.md` (product framing), `CV_INTEGRITY_ASSURANCE_2026.md` (SOTA survey), `docs/PS26228_REQUIREMENT_TRACE.md` (clause-by-clause traceability), `docs/DEEP_RESEARCH_SKILL.md` (reusable research protocol).*
