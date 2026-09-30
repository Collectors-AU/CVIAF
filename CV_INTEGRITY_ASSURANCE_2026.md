# Trustworthy CV Integrity Assurance — 2025–2026 SOTA Update & Engine Playbook
### SIH 26228 · MoD / Indian Army DGIS · Blockchain & Cybersecurity · offline / air-gapped

**What this document is.** A currency refresh (2025–2026) and implementation playbook that *builds on* the two existing docs — `../cv_integrity_assurance_research_report.md` and `../cv-pipeline-integrity-reference.md` (which already cover 2023–24 foundations: Spectral Signatures, Activation Clustering, STRIP, Neural Cleanse, Mahalanobis+MSP, Ed25519+Merkle, C2PA, cleanlab, alibi-detect, IBM ART, MNTD/ULP). It does **not** re-derive those. Every source below was fetched and verified against its primary page on 2026-09-11; each carries a confidence tag and verification note. Full source list + verification status is in §11; raw research state is in `RESEARCH_CHECKPOINT_26228.md`.

> **As-built addendum — read §12 first if you are acting on this document.** Most recommendations below
> have since been implemented and measured (2026-09-15 → 2026-09-30). Two of them came back negative:
> TRACE is implemented for its background arm only, and **three of the four trigger attacks never
> implanted** once the attack-success criterion was placebo-controlled. §12 is the one-screen status
> board; `RESEARCH_CHECKPOINT_26228.md` → UPDATE 3 carries the evidence and the traps.

Engine modules referenced: `data_integrity` · `model_integrity` · `provenance` · `drift` · `assessor` (both `engine/*.py` and the `cviaf/` package layout).

---

## 1. Executive summary

The single strongest differentiator for 26228 is **detector-specific integrity assurance**. Almost all public poisoning/backdoor/OOD tooling targets image **classifiers**; the problem mandates **COCO/YOLO object detection**, whose failure modes (object fabrication, **disappearance/cloaking**, regional/global misclassification, NMS abuse) have *no analog* in classifier defenses. As of 2025–26 there is now a usable detector-specific stack — **TRACE** (CVPR'25, black-box test-time), **ODSCAN** (IEEE S&P'24, scanning), **DistScan** (2026, pre-NMS distribution shift), and **DISTIL** (ICCV'25, data-free trigger inversion that explicitly evaluates on object detection). Wiring these into `model_integrity`, and *claiming explicit coverage of the BadDet attack taxonomy*, is what will separate this submission from a generic classifier-defense wrapper.

Four supporting shifts since the 2024 baseline:
1. **No-retrain, data-free, black-box** detection is now realistic for the baseline path (TRACE, DISTIL, BProm, Lite-BD) — matches the "must not require retraining the contributed model" constraint and the graceful white→black-box fallback.
2. **C2PA 2.4** added first-class AI/ML provenance assertions (model digests, training-data ingredients, an *AI-ML Output Content Credential*) — a standards spine the offline `provenance` module should align to instead of an ad-hoc schema.
3. **Label-error tooling went detector-native** — cleanlab **ObjectLab** operates directly on COCO/YOLO boxes; a 2025 framework extends it from detection to *correction*.
4. **Drift↔manipulation separation is provably hard** — two-window drift tests are demonstrably evadable (2411.16591); the honest design is block-based/within-window detection plus calibrated confidence, and an explicit "cannot fully separate" limitation.

---

## 2. Cross-cutting: the classifier→detector transfer gap (read first)

State this gap explicitly in the report and in `assessor`'s coverage statement.

| Classifier method (in prior docs / engine) | Transfers to COCO/YOLO detectors? | 2025–26 detector-native replacement/complement |
|---|---|---|
| Neural Cleanse (per-label trigger inversion) | ✗ assumes single softmax label; detectors have per-box class + localization | **ODSCAN** (S&P'24), **DISTIL** (ICCV'25, data-free) |
| Spectral Signatures / Activation Clustering | ~ partial; must run on per-box RoI features, not image embeddings | Run on RoI/backbone features per box; **DistScan** pre-NMS signal |
| STRIP (entropy under superimposition) | ~ weak; superimposition changes detection geometry | **TRACE** CTC/FTC transformation-consistency (black-box) |
| MSP / Mahalanobis OOD | ~ per-box only; global image score is misleading | Per-detection scoring + OpenOOD protocol |

**Threat taxonomy the engine must name (BadDet, ECCV'22 workshop):** Object Generation (OGA / phantom boxes), Regional Misclassification (RMA), Global Misclassification (GMA), Object Disappearance (ODA / cloaking). *Disappearance and generation have no classifier equivalent* — call this out.

---

## 3. Capability (a) — Training-data integrity → `data_integrity`

**2025–26 state of the art (verified).**
- **Label errors, detector-native:** cleanlab **ObjectLab** (`cleanlab.object_detection`, since cleanlab 2.5) scores COCO/YOLO annotations for wrong class, overlooked/missing boxes, and badly-drawn boxes using model outputs — *no retraining*. [HIGH — docs verified] Extended by **"From Label Error Detection to Correction: A Modular Framework and Benchmark for OD Datasets"** (arXiv 2508.06556, Aug 2025) which adds semi-automatic correction ("Rechecked"). [MED — search-verified]
- **Poisoning-filter upgrades over vanilla clustering:** the survey **"Backdoor Attacks and Defenses in the CV Domain"** (arXiv 2509.07504, Sep 2025) catalogs stronger training-data filters — **SPECTRE** (robust covariance + whitening), **Deep k-NN**, **SEVER/ITLM** (robust trimmed-loss), **DataElixir** (2024, diffusion purification) — and, crucially, frames test-time methods as "practical for model consumers who cannot retrain." [HIGH — full fetch] It contains a dedicated *Object-Detection-specific Defenses* subsection.
- **Near-duplicate / flooding:** **SSCD** (Pizzi et al., CVPR 2022; `facebookresearch/sscd-copy-detection`) is the standard self-supervised copy-detector (DISC2021), now used even to dedup diffusion training sets. [HIGH — verified] Recent near-dup work: arXiv 2410.19437 (Oct 2024). Perceptual-hash baseline via `imagededup` remains the cheap first pass.
- **OOD insertion:** **OpenOOD v1.5** (arXiv 2306.09301, NeurIPS'23 D&B; `Jingkang50/OpenOOD`) is the standardized eval harness and postprocessor zoo (MSP, Mahalanobis, KNN, ViM, ASH). [HIGH — verified] Method/benchmark critique: IJCV 2024 (10.1007/s11263-024-02222-4).
- **Attack taxonomy for test-scenario generation:** **"Data Poisoning in Deep Learning: A Survey"** (arXiv 2503.22759, Mar 2025) — 7-dimension taxonomy (clean-label / label-flip / backdoor). [HIGH — full fetch] ⚠️ *Attack-only; excludes defenses* — do not cite for detection.

**Gap vs. baseline.** Current `data_integrity` = Spectral Signatures / Activation Clustering (PCA+k-Means) on bbox deep features only. It has no label-error check, no near-dup/OOD screen, and no contributor-level aggregation.

**Concrete actions for `data_integrity`.**
1. Add a **label-integrity pass**: `cleanlab.object_detection.ObjectLab` on COCO/YOLO ingest (needs only model predictions + given labels — no retrain). Emit per-image `label_issue_score`.
2. Add a **near-duplicate/flooding pass**: `imagededup` PHash for cheap pass → **SSCD** embeddings + FAISS cosine for near-dup clusters; flag one-contributor duplicate floods.
3. Add an **OOD-insertion pass**: reuse the `drift` module's Mahalanobis/KNN scorer under the **OpenOOD** protocol, scored **per RoI/box**, not per image.
4. Upgrade the poisoning filter from k-Means to **SPECTRE** (robust covariance) on per-box RoI features.
5. **Source/contributor risk aggregation** (explicitly required): aggregate sample-level scores (label, dup, OOD, spectral) → per-contributor risk with a defensible rule (e.g., beta-binomial posterior on flag rate, or mean + tail-fraction), surfaced to `assessor`.

**Offline libraries (freeze exact pins at air-gap install):** `cleanlab>=2.7`, `imagededup`, SSCD weights (vendored), `faiss-cpu/gpu`, OpenOOD (from source), `scikit-learn`.

---

## 4. Capability (b) — Model integrity → `model_integrity`  ⭐ primary differentiator

**2025–26 state of the art (verified).** A genuinely detector-specific, mostly no-retrain stack now exists:

- **TRACE — Test-Time Backdoor Detection for Object Detection Models** (Hangtao Zhang et al., **CVPR 2025**, arXiv 2503.15293). [HIGH — full fetch]. **Black-box, no retraining, no training data** (uses public background/foreground images). Mechanism = *Transformation Consistency*: **CTC** (Contextual Transformation Consistency) — object confidence variance across blended backgrounds; triggered objects are *abnormally stable* → catches fabrication/misclassification (FP-inducing). For **disappearance/cloaking (FN-inducing)** triggers that leave no detection, it slides a *Natural Backdoor Object* patch across the image and watches its confidence collapse over the hidden trigger — the **"Island Effect"** → high **FTC**. Score = σ(FTC − CTC) vs γ. Detectors: **YOLOv5, Faster-RCNN, DETR**; data: COCO, VOC, synthesized traffic signs; **F1 0.880±0.040, AUROC 0.897±0.041**, ~30% F1 over "Detector Cleanse."
- **ODSCAN — Backdoor Scanning for Object Detection Models** (Guangyu Shen et al., **IEEE S&P 2024**; `Megum1/ODSCAN`). [HIGH — venue+repo verified] Detector-specific scanning; strong white-box baseline and reference implementation to benchmark against.
- **DISTIL — Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion** (Mirzaei et al., **ICCV 2025**, arXiv 2507.22813). [HIGH — abs verified] **Data-free, zero-shot trigger inversion** via classifier-guided latent diffusion; **evaluated on object detection** (+9.4% trojaned-OD scanning; +7.1% BackdoorBench). Ideal white-box path when weights are available but the contributor's data is not (air-gap).
- **DistScan — Pre-NMS Prediction Distribution Shift** (arXiv 2608.19088, 2026). [MED — verified] Backdoor training shifts the intermediate class-prediction distribution away from training class frequencies, detectable **pre-NMS** — a cheap, novel, detector-native signal.
- **Lite-BD — Lightweight Black-box Backdoor Defense** (Miah et al., arXiv 2602.07197, 2026; `SiSL-URI/Lite-BD`). [MED — verified] Two-stage: super-resolution down-upscaling neutralizes triggers across spatial/frequency/feature domains — a **black-box purification / graceful-fallback** option.
- **BProm — "Prompting the Unseen"** (arXiv 2411.09540, Nov 2024). [HIGH — full fetch] Black-box *classifier* backdoor detection via visual prompting + class-subspace inconsistency + meta-classifier; ~20 shadow models (vs MNTD's hundreds), never retrains the suspect. Fallback for classification heads/backbones.
- **Z-PEFT — Zero-shot weight-space detection** (arXiv 2608.02271, 2026). [MED — verified] Detect from weights alone; generalizes to novel attacks. NOTE: PEFT/adapter-scoped — treat as the *weight-space / substitution* direction, not a drop-in OD detector.
- **LeBD** (ICLR'24 submission) — LayerCAM-based **runtime** YOLO trigger localization for the physical world. [LOW — abstract only; PDF gated; no numbers].

**Threat context for evaluation (not authoritative cites):** **BadDet** taxonomy (OGA/RMA/GMA/ODA); **BadDet+** (evasive, OpenReview); **AnywhereDoor** (multi-target OD backdoor — ⚠️ *arXiv 2503.06529 WITHDRAWN*; code at `HKU-TASR/AnywhereDoor`, use only as scenario inspiration); cloaking-attack evaluation (ACM 2025, H. Ma). Classifier evasions to keep in the limitations list: WaNet, clean-label, adaptive/feature-space triggers.

**Gap vs. baseline.** Current `model_integrity` = white-box Neural Cleanse (L1) + black-box entropy probing — both **classifier-shaped** and both flagged above as non-transferring.

**Concrete actions for `model_integrity` (fallback ladder).**
1. **Black-box baseline (default, no weights, no data):** implement **TRACE** (CTC + Island-Effect FTC). This is the headline capability and directly covers OGA/RMA/GMA + ODA/cloaking.
2. **White-box, data-free (weights available):** add **DISTIL** (latent-diffusion trigger inversion) and/or **ODSCAN** as the trigger-reconstruction path — replaces the classifier-only Neural Cleanse for detectors.
3. **Cheap always-on signal:** add **DistScan** pre-NMS distribution check as a fast pre-filter.
4. **Purification / degraded fallback:** **Lite-BD** transformations when only queries are possible.
5. **Substitution detection:** keep the `provenance` weight-digest check as the primary anti-substitution control; add weight-space anomaly (Z-PEFT-style) only if adapters/PEFT are in scope.
6. **Access-mode reporting:** each sub-detector must emit `mode ∈ {white-box, black-box}` and cleanly report `unavailable` (never silently pass) when access is missing — matches the current schema's `limitations_declared`.
7. **Remediation (optional, H200):** note that fine-pruning cuts ASR but can crater clean mAP (observed on OD) — offer as opt-in remediation, not baseline.

**Offline libraries:** research code vendored from the TRACE / ODSCAN / DISTIL / DistScan / Lite-BD repos; `IBM adversarial-robustness-toolbox (ART)` for attack generation & some defenses; `torch`, `onnx`/`onnxruntime`, `ultralytics` (YOLO). Freeze commit hashes.

---

## 5. Capability (c) — Inference provenance & output integrity → `provenance`

**2025–26 state of the art (verified).**
- **C2PA 2.4 — Guidance for AI/ML** (spec.c2pa.org, 2.4 ai-ml). [HIGH — full fetch] Now provides exactly the primitives this capability needs: **asset type assertion** (model name/framework/type), **asset reference assertion + hard hash binding**, **collection data hash** (multi-file models incl. ONNX topology+weights), **ingredient assertions** for training data, an **AI-ML Output Content Credential** binding output → model → input provenance, digital source types `trainedAlgorithmicMedia` / `c2pa.trainedAlgorithmicData`, **sidecar manifests** for non-embeddable data, and **SBOM (SPDX) signing**. Gap it admits: model *ensembles* not yet specified.
- **Tamper-evident audit trail:** **AuditableLLM** (MDPI *Electronics* 2026, 15(1):56) — hash-chain per record, execution decoupled from an audit/verification layer. [MED — abstract verified] `ai-audit-trail` (PyPI) is an existing hash-chain lib. [MED — verified]

**Gap vs. baseline.** The engine already does the hard cryptographic part well: localized **Ed25519** root + **Merkle** leaf binding `SHA3(image) + ONNX_digest + preproc_config + output + offline_nonce`. What's missing is (i) **standards alignment** so outputs are portable/verifiable by third parties offline, and (ii) an explicit **replay/sequence** control across a session.

**Concrete actions for `provenance`.**
1. Emit the existing binding **as an offline C2PA-style manifest** (embedded where the container allows, **sidecar** otherwise) using the asset-reference + collection-data-hash pattern for ONNX (topology+weights) and an **AI-ML Output Content Credential** for each inference. Keeps you standards-aligned without cloud/KMS.
2. Add a **hash-chain sequence log** (prev-hash + monotonic sequence + offline nonce/timestamp) so **replay and reordering** are detectable, not just single-record tamper (AuditableLLM pattern). Ed25519-sign chain heads periodically.
3. Keep **no-NTP** posture: rely on monotonic counters + nonces; document that wall-clock is advisory only.
4. Feed chain-verification status into `assessor` as a first-class assurance signal.

**Offline libraries:** `cryptography` (Ed25519; SHA3 via `hashlib`), `pymerkle` or RFC-6962-style custom, `c2pa-python`/`c2patool` (vendored, offline), `onnx` for weight digests.

---

## 6. Capability (d) — Distribution-shift & anomaly → `drift`

**2025–26 state of the art (verified).**
- ⚠️ **"Adversarial Attacks for Drift Detection"** (Hinder, Vaquet, Hammer, arXiv 2411.16591, Nov 2024). [HIGH — full fetch] **Two-window drift detectors are provably evadable** — one can construct genuine drift whose per-window statistics match, so windowed MMD/KS raise no alarm (Theorem 1 / kernel-of-W construction). **Design implication: block-based / within-window detectors are NOT prone to this** — prefer them. There is **no clean classifier that separates "adversarial" from "natural" drift**; say so honestly.
- **Detector confidence is miscalibrated:** "Systematic Evaluation of Uncertainty Calibration in Pretrained Object Detectors" (IJCV 2024, 10.1007/s11263-024-02219-z) and TUM's "Towards Reliable OD with Confidence Calibration" (mediatum 1769359). [MED — verified] Matters because the engine fuses MSP/confidence.

**Gap vs. baseline.** Current `drift` = Mahalanobis + MSP fusion — a reasonable core, but (i) it conflates covariate/concept/prior shift, (ii) it inherits detector miscalibration, and (iii) if extended naively to windowed tests it becomes evadable.

**Concrete actions for `drift`.**
1. Split reporting into **covariate vs. concept vs. prior** shift against the *declared reference distribution* (terrain/season/sensor/illumination), each per-box where possible.
2. Add **two-sample tests** (MMD, energy/KS via `alibi-detect`) **but prefer block/within-window formulations**; document the evadability caveat and the "drift vs. manipulation cannot be perfectly separated" limitation.
3. Add **post-hoc calibration** (temperature/vector scaling) before thresholding confidence, given OD miscalibration.
4. Produce a **calibrated risk score** and route "high anomaly + extreme confidence suppression" toward *manipulation*, "moderate covariance delta" toward *operational drift* — as the current README already sketches — but label the residual ambiguity explicitly.

**Offline libraries:** `alibi-detect` (MMD/KS/energy), `pytorch-ood`, `scikit-learn`, `netcal` (calibration).

---

## 7. Capability (e) — Analyst-facing assurance & governance → `assessor`

**2025–26 state of the art (verified).**
- **NIST/IARPA TrojAI Final Report** (Reese et al., arXiv 2602.07152, Feb 2026). [HIGH — abs verified] ⚠️ *IARPA launched the program; NIST ran the associated evaluations* — state precisely. Confirms weight-analysis + trigger-inversion as the two workhorse families and highlights "natural Trojans" (false-positive risk) and detector sensitivity — directly informs threshold/confidence language.
- **NIST AI RMF 1.0 + 2025 updates.** [MED — verified] Govern/Map/Measure/Manage framing + third-party/supply-chain risk — the governance vocabulary for a multi-contributor pipeline.
- Detection ≠ mitigation (BadDet+ note): flagging a suspicious input does not remove the backdoor — the disposition language must reflect this.

**Gap vs. baseline.** The JSON schema (severity LOW/MED/HIGH/CRITICAL, evidence, disposition accept/review/quarantine, audit_trail[]) is a solid start but lacks (i) per-finding **confidence**, (ii) an explicit **coverage statement**, and (iii) a tie to the provenance hash-chain.

**Concrete actions for `assessor`.**
1. Every flag: human-readable **reason + evidence + confidence + severity + affected asset + disposition** (schema already has most; add `confidence` and `method`/`access_mode`).
2. Add a machine-readable **coverage statement** (see §10): which BadDet categories, which shift types, which access modes are/aren't covered on this run.
3. Bind the report to the **provenance hash-chain** so the assurance verdict is itself tamper-evident.
4. Map dispositions to **NIST AI RMF** functions for the write-up; keep contributor-level risk (from §3) as a top-level field.

---

## 8. Reproducible test-scenario generation (required by the brief)

For evaluation on the H200, generate representative attacks rather than relying on found data:
- **Detector backdoors (BadDet categories):** use BadDet/ODSCAN repos to implant OGA/RMA/GMA/ODA triggers into YOLO/Faster-RCNN on COCO/VOC; cloaking scenarios per H. Ma (ACM 2025).
- **Poisoning / label-flip / clean-label:** IBM **ART** poisoning module + manual COCO/YOLO label corruption; near-dup floods via SSCD-guided duplication.
- **Substitution/tamper:** swap ONNX weights / alter preprocessing and confirm the `provenance` digest + hash-chain break.
- **Distribution shift:** synthesize terrain/season/sensor/illumination shifts (augmentation) vs. adversarial perturbations (ART) to test drift-vs-manipulation labeling.

---

## 9. H200 benchmarking plan

| Benchmark | Use | Access |
|---|---|---|
| **NIST TrojAI** (pages.nist.gov/trojai) + **Trojan Detection SW Challenge – Object Detection** (Feb 2023 round, data.commerce.gov) | Reference trojaned/clean **detector** models + scoring for `model_integrity` | public artifacts |
| **BackdoorBench** (`SCLBD/backdoorbench`, NeurIPS'22 + arXiv 2407.19845) | Standardized attack/defense matrix; TRACE/DISTIL report on it | public |
| **OpenOOD v1.5** (`Jingkang50/OpenOOD`) | OOD-insertion & drift scorers under a standard protocol | public |
| **ODSCAN repo** (`Megum1/ODSCAN`) | Detector-backdoor scanning baseline to beat | public |

Retraining/remediation experiments (fine-pruning, ANP) are the only steps that need the H200's training capacity; all baseline detectors above are inference-only.

---

## 10. Coverage & limitations statement (put a version of this in every run)

**Covered:** COCO/YOLO ingest; label errors (ObjectLab); near-dup/flooding (SSCD); OOD insertion (OpenOOD protocol); detector backdoors OGA/RMA/GMA/ODA at test time (TRACE, black-box) and via trigger inversion when white-box (DISTIL/ODSCAN); model substitution (weight digest + hash-chain); replay/tamper (sequence + Ed25519 chain); operational-drift vs. manipulation *scoring* (calibrated, with stated ambiguity).

**Not (yet) covered / known evasions — declare explicitly:** adaptive/feature-space and clean-label triggers (WaNet, label-consistent) may evade; **drift and adversarial manipulation cannot be perfectly separated** (two-window tests evadable, 2411.16591); model *ensembles* lack a C2PA representation; Z-PEFT weight-space detection is PEFT/adapter-scoped, not a general OD detector; LeBD numbers unverified; TRACE relies on auxiliary public images (unavailable in a hard air-gap unless pre-vendored).

---

## 11. Consolidated citations (verification status · confidence)

**Capability a — data integrity**
- cleanlab **ObjectLab** — docs.cleanlab.ai `object_detection` — [VERIFIED · HIGH]
- Label Detection→Correction — arXiv 2508.06556 (2025) — [search-VERIFIED · MED]
- **SSCD** — Pizzi et al., CVPR 2022; `facebookresearch/sscd-copy-detection` — [VERIFIED · HIGH]
- Near-dup (transductive) — arXiv 2410.19437 (2024) — [search-VERIFIED · MED]
- **OpenOOD v1.5** — arXiv 2306.09301, NeurIPS'23 — [VERIFIED · HIGH]
- OOD/OSR critique — IJCV 2024, 10.1007/s11263-024-02222-4 — [search-VERIFIED · MED]
- Data-poisoning survey (attack-only) — arXiv 2503.22759 (2025) — [full-fetch · HIGH]
- CV backdoor **defense** survey — arXiv 2509.07504 (2025) — [full-fetch · HIGH]

**Capability b — model integrity**
- **TRACE** — arXiv 2503.15293, CVPR 2025 — [full-fetch · HIGH]
- **ODSCAN** — IEEE S&P 2024; `Megum1/ODSCAN` — [venue+repo · HIGH]
- **DISTIL** — arXiv 2507.22813, ICCV 2025 — [abs-fetch · HIGH]
- **DistScan** — arXiv 2608.19088 (2026) — [VERIFIED · MED]
- **Lite-BD** — arXiv 2602.07197 (2026); `SiSL-URI/Lite-BD` — [VERIFIED · MED]
- **BProm** — arXiv 2411.09540 (2024) — [full-fetch · HIGH]
- **Z-PEFT** — arXiv 2608.02271 (2026) — [VERIFIED · MED; PEFT-scoped]
- **BadDet** taxonomy — ECCV'22 workshop — [known · MED]
- LeBD — OpenReview 7vKWg2Vdrs — [abstract-only · LOW]
- AnywhereDoor — arXiv 2503.06529 — ⚠️ **WITHDRAWN**; code `HKU-TASR/AnywhereDoor` — [context-only]

**Capability c — provenance**
- **C2PA 2.4 AI/ML guidance** — spec.c2pa.org 2.4/ai-ml — [full-fetch · HIGH]
- AuditableLLM — MDPI Electronics 2026, 15(1):56 — [abs-VERIFIED · MED]
- ai-audit-trail — PyPI — [VERIFIED · MED]

**Capability d — drift**
- Adversarial Attacks for Drift Detection — arXiv 2411.16591 (2024) — [full-fetch · HIGH]
- OD uncertainty calibration — IJCV 2024, 10.1007/s11263-024-02219-z — [search-VERIFIED · MED]

**Capability e — governance**
- **TrojAI Final Report** — arXiv 2602.07152 (2026) — [abs-fetch · HIGH; IARPA-led, NIST-evaluated]
- NIST **TrojAI** leaderboard + OD round — pages.nist.gov/trojai; data.commerce.gov — [VERIFIED · HIGH]
- **BackdoorBench** — `SCLBD/backdoorbench`; arXiv 2407.19845 — [VERIFIED · HIGH]
- **NIST AI RMF** 1.0 + 2025 — [VERIFIED · MED]

---

## 12. As-built addendum — what these recommendations became (2026-09-30)

This document is a research position taken on 2026-09-11. Everything below it has since been built,
measured, or deliberately left unbuilt, and the measurement contradicted the research twice. Read this
section before acting on §3–§7. Evidence and traps: `RESEARCH_CHECKPOINT_26228.md` → **UPDATE 3**.

### 12.1 Status board

| Dossier recommendation | Status | Where it lives / receipt |
|---|---|---|
| **TRACE** as the detector-specific black-box baseline (§4) | **Implemented — one arm only.** Background-blend CTC plus the FTC/Island-Effect signal for disappearance | `cviaf/lab/detectors.py` `trace_ctc`, `trace_ftc` |
| **DistScan** pre-NMS distribution check (§4) | **Implemented and measured** (`pre_nms_class_js`) | `cviaf/lab/detectors.py::pre_nms_class_divergence` |
| Detector-native **label screening** — cleanlab ObjectLab (§3) | Confident-learning screen in the same family; library not vendored | `cviaf/data_integrity/__init__.py` |
| **Near-duplicate / flooding** screen — SSCD (§3) | Frozen-conv stand-in; SSCD is the named scaling target | `cviaf/lab/attribute.py`, `docs/SCALING_PLAN.md` |
| Offline **C2PA 2.4**-style manifest (§5) | **Not done** — repository-local schema, HMAC-SHA256 fallback in the committed run | `runs/mvp/comparison.json` |
| Hash-chain sequence log for replay (**AuditableLLM** pattern, §5) | Implemented and validated | `runs/assurance/oda_s5/assurance_report.json` |
| **OpenOOD v1.5** protocol for the OOD screen (§3) | Validation target named, not run | — |
| **ODSCAN** / **DISTIL** white-box trigger inversion (§4) | Not implemented | — |
| **NIST AI RMF** vocabulary for governance (§7) | Adopted | `docs/COVERAGE_STATEMENT.md`, `docs/PS26228_ALIGNMENT_MATRIX.md` |
| Block/within-window drift tests + publish the evasion test (§6) | Adopted as a declared limitation, emitted per cell | alignment matrix **C4 = PARTIAL** |

### 12.2 The two reversals, and what they mean for this document's claims

1. **§4's TRACE description is one arm short.** The implementation's image-level null control — a clean
   model whose images carry a trigger, with no backdoor present — reads **CTC 0.816 / refdiv 0.786 /
   FTC 0.792**. TRACE's second observation (a mirror-sign *foreground* consistency arm) is the citable
   explanation and an open implementation gap. **Read every image-level number net of this control.**
2. **§4's threat taxonomy needed a placebo control before it meant anything.** With the trigger
   re-placed per image and two same-RNG placebos subtracted against each model's seed-matched clean
   twin, **three of the four trigger recipes never implanted at 64×64** — `oga`/`oda`/`rma` net to
   0.000, and only `gma` shows a placebo-robust effect. For generation attacks a *clearer* trigger makes
   a *worse* backdoor: `oga` with a checkerboard patch is exactly 0.000 at poisoning rate 0.5, while a
   low-amplitude blended trigger reaches net 0.525. The corpus now reports **raw / null / net**
   separately. Full record: `docs/MEASUREMENT_NOTES.md` §1–§4, §10 *(working document — untracked)*.
3. **The backdoor-like case is a measured failure, not recall.** `bias_lift` is behaviour-inert on all
   **396** arms, so the frozen-threshold headline of 5.1% is that rule's own false-alarm rate. This is
   §7's "natural Trojans" and §2's evasion warning arriving as a named blind spot.
4. **§6's "provably inseparable" is now "mostly undecided":** 8 drift decisions against 32
   under-determined cells at n=160 per cell.

### 12.3 New sources this document does not cover

Two findings from the implementation's own loop changed code and are absent from §11: **DistScan**
(arXiv 2608.19088 — now implemented, and its only clean separation is `gma`, which **converges with the
placebo-controlled behavioural test** on the same attack) and **E-SHIFT** (anytime-valid sequential
hypothesis testing for distribution shift). The loop, its queue and its no-network boundary: `cviaf/research`;
ledger: `runs/research/findings.jsonl`.

### 12.4 What is still unbuilt, stated plainly

C2PA-aligned manifests · ODSCAN/DISTIL · OpenOOD · SSCD itself · a real-backbone evidence corpus
(requested as 1× H200, 1–2 sessions of 4–8 h) · a byte-level image-overlap audit (9,984,920 images).
The declared-unsupported list ships inside the assurance report rather than being inferred from this page.

---

*Prepared 2026-09-11. §12 added 2026-09-30 from the implementation receipts. Prior baseline: `../cv_integrity_assurance_research_report.md`, `../cv-pipeline-integrity-reference.md`. Research state, corrections log and as-built record: `RESEARCH_CHECKPOINT_26228.md`.*
