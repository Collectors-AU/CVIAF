# Research Checkpoint — SIH 26228 CV Integrity Assurance

**Purpose:** Durable checkpoint so this research can resume with ZERO re-searching.
All sources below were fetched/verified against primary pages (Tavily REST + WebFetch)
on 2026-09-11. Raw search JSON was in job tmp (ephemeral) — the distilled, verified
findings here supersede it.

## STATUS
- [x] Prior docs reviewed: `../cv_integrity_assurance_research_report.md`, `../cv-pipeline-integrity-reference.md` (baseline, cites stop ~2023–24)
- [x] Tavily sweep: 18 queries, all HTTP 200
- [x] Citation verification: primary sources fetched, metadata locked, hallucinated/mis-dated cites killed
- [x] Engine layout re-read (see MODULE MAP below)
- [x] **Consolidated report written** → `CV_INTEGRITY_ASSURANCE_2026.md`; its §12 carries the as-built status board.
- [ ] **→ RESUME HERE: the open work is no longer searching, it is closing the gaps in UPDATE 3 §3.5.** Nothing in this file needs re-searching — the sources are verified, and the implementation has since answered most of them.
- Note: original `/deep-research` background workflow (wf_4e0c352f-abd) was KILLED by the session restart; NOT relaunched (manual verify pipeline used instead — better citation control).

## KEY STRATEGIC FINDING
The **detector-specific (object-detection) backdoor gap** is the strongest differentiator for 26228,
and 2025 SOTA now exists to fill it. Classifier defenses (Neural Cleanse/STRIP/etc.) do NOT map cleanly
to YOLO/Faster-RCNN/DETR failure modes (object fabrication, disappearance/cloaking, misclassification, NMS abuse).

---

## VERIFIED CITATIONS BY CAPABILITY
(Format: Title — Authors — ID/venue, year — [VERIFIED how] — relevance)

### Capability 1 — Training-data integrity
1. **Data Poisoning in Deep Learning: A Survey** — Pinlong Zhao, Weiyao Zhu, Pengfei Jiao, Di Gao, Ou Wu — arXiv 2503.22759, 27 Mar 2025 — [VERIFIED full fetch] — 7-dimension attack taxonomy (clean-label/backdoor/label-flip). ⚠️ ATTACK-ONLY, explicitly excludes defenses — do NOT cite as a defense source.
2. **Backdoor Attacks and Defenses in Computer Vision Domain: A Survey** — Bilal Hussain Abbasi, Yanjun Zhang, Leo Zhang, Shang Gao — arXiv 2509.07504, 09 Sep 2025 — [VERIFIED full fetch] — KEYSTONE defense taxonomy. Training-data: Spectral Signatures, Activation Clustering, SPECTRE, ULPs, Deep k-NN, DataElixir(2024), SEVER, ITLM. Test-time (NO retrain): STRIP, SentiNet, Februus, TeCo, BadActs, PBE. Has dedicated "Object-Detection-specific Defenses" subsection. Explicitly frames retraining-free methods as "practical for model consumers who cannot retrain from scratch" → matches no-retrain baseline requirement.
3. **cleanlab ObjectLab** — cleanlab 2.5+ `object_detection` module — docs.cleanlab.ai + cleanlab.ai/blog/learn/object-detection — [VERIFIED] — label errors in COCO/YOLO datasets (wrong class, overlooked boxes, bad boxes). Directly COCO/YOLO-native.
4. **From Label Error Detection to Correction: A Modular Framework and Benchmark for Object Detection Datasets** — arXiv 2508.06556, Aug 2025 — [VERIFIED search] — extends ObjectLab from detection to correction.
5. **SSCD: A Self-Supervised Descriptor for Image Copy Detection** — Ed Pizzi, Sreya Dutta Roy, Sugosh Nagavara Ravindra, Priya Goyal, Matthijs Douze (Meta AI) — CVPR 2022 (cited 243×) — [VERIFIED, openaccess.thecvf CVPR2022 + github facebookresearch/sscd-copy-detection] — near-duplicate/copy detection; DISC2021 benchmark. (Note: Stable Diffusion 3 uses SSCD to dedup training data.)
6. **Transductive Learning for Near-Duplicate Image Detection in Scanned Photo Collections** — arXiv 2410.19437, Oct 2024 — [VERIFIED search] — recent near-dup method. Also: HuggingFace `large-scale-image-deduplication` repo.
7. **OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection** — Jingyang Zhang, Jingkang Yang, et al. — arXiv 2306.09301, NeurIPS 2023 D&B (also PMLR v02-3) — [VERIFIED, neurips.cc/virtual/2023/80497 + github Jingkang50/OpenOOD] — standardized OOD eval; MSP/Mahalanobis/AugMix postprocessors. Engine's drift module already uses Mahalanobis+MSP → OpenOOD is the benchmark to validate on.
8. **Dissecting Out-of-Distribution Detection and Open-Set Recognition: A Critical Analysis** — IJCV 2024, DOI 10.1007/s11263-024-02222-4 — [VERIFIED search] — OOD vs OSR method/benchmark critique.

### Capability 2 — Model integrity (backdoor/substitution)
9. **TRACE: Test-Time Backdoor Detection for Object Detection Models** ⭐CENTERPIECE — Hangtao Zhang, Yichen Wang, Shihui Yan, Chenyu Zhu, Ziqi Zhou, Linshan Hou, Shengshan Hu, Minghui Li, Yanjun Zhang, Leo Yu Zhang (HUST + Deakin/Griffith) — arXiv 2503.15293, **CVPR 2025** (cited 37×) — [VERIFIED full fetch of HTML v2]. Method = **TRAnsformation Consistency Evaluation**:
   - **BLACK-BOX, NO retraining, NO training data** (uses public background/foreground images as support). Ideal for graceful black-box fallback + air-gap.
   - **CTC (Contextual Transformation Consistency):** variance of object confidence across many blended backgrounds. Triggered/poisoned objects are ABNORMALLY STABLE (low variance) → catches FP-inducing triggers (fabrication, misclassification).
   - **FN-inducing triggers (disappearance/cloaking):** leave no detection so CTC fails. Slide "Natural Backdoor Object" (NBO) patches (e.g., stop sign) across image; over a hidden trigger the injected object's confidence drops sharply = **"Island Effect"** → high FTC (Focal Transformation Consistency). EigenCAM shows "black-hole" activation.
   - Score = sigmoid(FTC − CTC) vs threshold γ.
   - Detectors: YOLOv5, Faster-RCNN, DETR. Data: MS-COCO, PASCAL VOC, Synthesized Traffic Signs. 7 attacks (OGA, RMA, GMA, ODA, CIB, UTA, DC). **F1 0.880±0.040, AUROC 0.897±0.041**; ~30% F1 gain over prior SOTA "Detector Cleanse". (Minor internal inconsistency: 42 vs 63 backdoored models — don't overstate.)
10. **AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection** — Jialin Lu, Junjie Shan, Ziqi Zhao, Ka-Ho Chow (HKU) — arXiv 2503.06529, 09 Mar 2025 — [VERIFIED full fetch] — THREAT MODEL. Inference-time control: make objects disappear/fabricate/mislabel, all-class or per-class. Innovations: objective disentanglement, trigger mosaicking (survives region detection), strategic batching. Beats BadDet-style adaptations by +26% ASR. Attacks Faster-RCNN, DETR, YOLOv3 on VOC+COCO. **Defense finding (crucial): input-based defenses (JPEG, mean/median filter, NEO) FAIL; fine-pruning cuts ASR but craters clean mAP to ~18.3.** Code: github.com/HKU-TASR/AnywhereDoor.
11. **BProm: Prompting the Unseen — Detecting Hidden Backdoors in Black-Box Models** — Zi-Xuan Huang, Jia-Wei Chen, Zhi-Peng Zhang, Chia-Mu Yu (NYCU Taiwan) — arXiv 2411.09540, 14 Nov 2024 — [VERIFIED full fetch] — black-box classifier backdoor detection via visual prompting + "class subspace inconsistency" + meta-classifier (random forest). Needs ~20 shadow models (vs MNTD's hundreds); NEVER retrains the suspect model. CIFAR-10 AUROC ~1.000; struggles with all-to-all + clean-label. Graceful-fallback reference.
12. **LeBD: A Run-time Defense Against Backdoor Attack in YOLO** — ICLR 2024 submission (OpenReview 7vKWg2Vdrs) — [VERIFIED title/abstract only; PDF CAPTCHA-gated] — "LayerCAM-enabled backdoor detector" for real-time PHYSICAL-world YOLO defense. Cite at mechanism level (LayerCAM region analysis at runtime); DO NOT invent numbers.
13. **BadDet: Backdoor Attacks on Object Detection** — Chengxiao Luo et al. (ECCV 2022 workshop) — [known + search-confirmed] — the taxonomy: Object Generation Attack (OGA), Regional Misclassification (RMA), Global Misclassification (GMA), Object Disappearance (ODA). Foundational threat taxonomy for the report.
14. **Mask-based Invisible Backdoor Attacks on Object Detection** — arXiv 2405.09550, May 2024 — [VERIFIED search] — adapts Grad-CAM + STRIP to OD in defense experiments.
15. **BadDet+: Robust Backdoor Attacks for Object Detection** — K. Dunnett et al. (OpenReview 6rz7VyAatm) — [VERIFIED search] — notes TRACE-type test-time detection "flags suspicious inputs but does not remove the backdoor" → detection ≠ mitigation framing.
16. Classifier-defense set (from #2 survey): Neural Cleanse, ABS, Fine-Pruning, DeepInspect, TABOR, MNTD, ANP, ULPs. "Detector Cleanse" = OD analog of Neural Cleanse (TRACE's SOTA baseline).

### Capability 3 — Inference provenance & output integrity
17. **C2PA 2.4 — Guidance for Artificial Intelligence and Machine Learning** — spec.c2pa.org/specifications/specifications/2.4/ai-ml/ai_ml.html — [VERIFIED full fetch]. KEY new addition. Provides: *asset type assertion* (model name/framework/type), *asset reference assertion* + hard hash binding, *collection data hash* (multi-file models), *ingredient assertions* for training data (URI + hard binding), **AI-ML Output Content Credential** (validate source+integrity of results, links back to model + input provenance), digital source types `trainedAlgorithmicMedia`/`c2pa.trainedAlgorithmicData`, sidecar manifests for non-embeddable data, SBOM (SPDX) signing, fine-tune/LoRA/checkpoint dependency trees. Gaps: model ensembles not yet covered. → Standards-align the engine's provenance module (offline C2PA-style manifests).
18. **AuditableLLM: A Hash-Chain-Backed, Compliance-Aware Auditable Framework for LLMs** — MDPI *Electronics* 2026, 15(1):56 (DOI 10.3390/electronics15010056) — [VERIFIED search + abstract] — decouples update execution from an audit/verification layer; hash-chain per record for verifiable accountability across model updates. Analog for the engine's tamper-evident audit log.
19. **ai-audit-trail** (PyPI) — [VERIFIED search] — existing lib for tamper-evident AI audit trails (hash chains). Also: veritaschain DEV article on hash chains + Merkle trees for audit trails.
   - Engine already implements Ed25519 + Merkle + SHA3 binding of image+ONNX_digest+preproc+output+nonce → the new work is standards alignment (C2PA) + hash-chain sequence log + replay/nonce hardening.

### Capability 4 — Distribution-shift & anomaly (drift vs adversarial)
20. **Adversarial Attacks for Drift Detection** — Fabian Hinder, Valerie Vaquet, Barbara Hammer (Bielefeld) — arXiv 2411.16591, 25 Nov 2024 — [VERIFIED full fetch] — ⚠️ CAUTIONARY: two-window drift detectors (windowed MMD/KS) are EVADABLE — one can construct genuine drift whose window means match so no alarm fires (Theorem 1 / kernel-of-W construction). **Design implication for engine: block-based / within-window detectors are provably NOT prone to this; prefer them.** There is NO clean classifier that separates "adversarial" from "natural" drift — frame this honestly. Code: github.com/FabianHinder/Drift-Adversarials.
21. **Systematic Evaluation of Uncertainty Calibration in Pretrained Object Detectors** — IJCV 2024, DOI 10.1007/s11263-024-02219-z — [VERIFIED search] — detector confidence miscalibration; matters because engine fuses MSP/confidence.
22. **Towards Reliable Object Detection with Confidence Calibration** — TU Munich (mediatum 1769359) — [VERIFIED search] — OD confidence calibration.
   - Engine uses Mahalanobis + MSP. Additions to research: covariate/concept/prior shift split, MMD/FID/KID vs adversarial, temperature/post-hoc calibration, the evadability caveat above.

### Capability 5 — Analyst-facing assurance & governance
23. **TrojAI Final Report (Trojans in Artificial Intelligence)** — Kristopher W. Reese + ~70 co-authors (incl. Michael Majurski, Peter Bajcsy) — arXiv 2602.07152, 06 Feb 2026 (rev 27 Feb 2026) — [VERIFIED abs fetch] — ⚠️ credits **IARPA** as program launcher; **NIST** ran the associated evaluations (state precisely). Detection via weight analysis + trigger inversion; discusses "natural Trojans" prevalence, detector sensitivity, lessons learned.
24. **NIST TrojAI leaderboard** — pages.nist.gov/trojai — [VERIFIED search] — live rounds incl. object detection. **Trojan Detection Software Challenge — Object Detection (Feb 2023 round)** on data.commerce.gov. → benchmark to run on the H200.
25. **BackdoorBench: A Comprehensive Benchmark of Backdoor Learning** — SCLBD — github.com/SCLBD/backdoorbench + NeurIPS 2022 D&B + "Comprehensive Benchmark and Analysis" arXiv 2407.19845 (2024) — [VERIFIED search] — standardized attack/defense benchmark. → validate engine detectors here.
26. **NIST AI RMF 1.0 + 2025 updates** — [VERIFIED search] — governance framing (Map/Measure/Manage/Govern), third-party/supply-chain risk. → maps to the assurance/coverage-statement + disposition governance layer.

---

## ENGINE MODULE MAP (as of 2026-09-11)
Two parallel layouts exist:
- **`engine/`** (reference impl): `data_integrity.py`, `model_integrity.py`, `provenance.py`, `drift.py`, `assessor.py`
- **`cviaf/`** (package): `data_integrity/`, `model_integrity/`, `provenance/`, `drift/`, `governance/`, `attacks/`, `formats/model_loader.py`, `core/types.py`, `orchestrator.py`, `cli.py`, `__main__.py`
- **`schemas/sample-assurance-report.json`**: keys = timestamp, framework_version, audit_trail[], assessments{training_data, model_integrity, provenance, distribution_shift} (each: severity LOW/MED/HIGH/CRITICAL, evidence, recommended_disposition accept/review/quarantine).

### Recommendation skeleton (fill at synthesis)
- **data_integrity** (currently Spectral Signatures/Activation Clustering via PCA+kMeans on bbox deep features):
  ADD → cleanlab ObjectLab (COCO/YOLO label errors), SSCD near-dup, OpenOOD-style OOD screen, SPECTRE/Deep-kNN as stronger poisoning filters, source-level risk aggregation per contributor. Validate on BackdoorBench.
- **model_integrity** (currently white-box Neural Cleanse L1 + black-box entropy probing):
  ADD → **TRACE** as the detector-specific test-time detector (black-box, no-retrain, handles OGA/RMA/GMA/ODA incl. disappearance via Island Effect); **Detector Cleanse** as white-box OD trigger inversion; **BProm** as classifier black-box fallback; **BadDet/AnywhereDoor** as the threat taxonomy the engine must claim coverage of; note fine-pruning caveat (wrecks mAP). Validate on NIST TrojAI OD round + BackdoorBench.
- **provenance** (currently Ed25519+Merkle+SHA3 binding image+ONNX_digest+preproc+output+nonce):
  ADD → **C2PA 2.4** offline manifest alignment (asset type/reference assertions, AI-ML Output Content Credential, collection data hash for multi-file/ONNX), hash-chain sequence log (AuditableLLM/ai-audit-trail style) for replay/tamper across an inference session.
- **drift** (currently Mahalanobis + MSP fusion):
  ADD → covariate/concept/prior-shift split, MMD/FID-KID two-sample tests, post-hoc calibration (OD miscalibration is real), and the **block-based (within-window) detector** recommendation because two-window detectors are provably evadable (arXiv 2411.16591).
- **assessor/governance** (currently JSON severity+disposition):
  ADD → NIST AI RMF alignment, explicit coverage statement (which BadDet categories / shift types are/aren't covered), confidence + evidence + reasons per finding, tamper-evident audit log tie-in to provenance hash-chain.

---

## UPDATE 2 — post-workflow verification (2026-09-11)
Background workflow wf_4e0c352f-abd COMPLETED partially: 55 agents done, 56 failed on QUOTA 403 ("Please run /login"); final synthesis never ran. It confirmed TRACE/AnywhereDoor/BadDet+ and surfaced new leads. All leads below independently VERIFIED via Tavily+arXiv:

- **DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion** — Hossein Mirzaei, Zeinab Taghavi, Sepehr Rezaee, Masoud Hadi, Moein Madadi, Mackenzie W. Mathis — arXiv 2507.22813, **ICCV 2025** — [VERIFIED abs] — DATA-FREE, ZERO-SHOT trigger inversion via classifier-guided latent diffusion; no training data → air-gap friendly. Works on OBJECT DETECTION: +9.4% trojaned-OD scanning, +7.1% BackdoorBench. → model_integrity white-box/no-data trigger-inversion upgrade.
- **ODSCAN: Backdoor Scanning for Object Detection Models** — Guangyu Shen et al. — **IEEE S&P 2024** — github.com/Megum1/ODSCAN — [VERIFIED, computer.org + repo] — KEYSTONE detector-specific backdoor scanner (top venue). Was missing from checkpoint v1.
- **DistScan: Detecting Backdoors in Object Detection via Pre-NMS Prediction Distribution Shift** — arXiv 2608.19088 (2026) — [VERIFIED] — backdoor training shifts intermediate class-prediction distribution away from training class frequencies, detectable pre-NMS. Detector-specific, novel signal.
- **Lite-BD: A Lightweight Black-box Backdoor Defense via Reviving Multi-Stage Image Transformations** — A. Arafat Miah et al. — arXiv 2602.07197 (2026) — github.com/SiSL-URI/Lite-BD — [VERIFIED] — two-stage black-box: super-resolution down-upscaling neutralizes triggers across spatial/frequency/feature domains. Graceful-fallback purification.
- **Z-PEFT: Zero-shot Backdoor Detection in PEFT via Canonical Spectral Signatures** — N. Pitzalis et al. — arXiv 2608.02271 (2026) — [VERIFIED] — weight-space detection from weights only; generalizes to novel attacks; beats PEFTGuard on cost. NOTE: PEFT/adapter-scoped (more LLM-ish) — cite as weight-space direction.
- Adjacent verified: BBCaL (Black-box Backdoor Detection under Causality Lens, TMLR); "Beyond Small Patches" (arXiv 2609.03139, black-box detect+purify); "Comprehensive Evaluation of Cloaking Backdoor Attacks against Object Detectors" (ACM 2025, H. Ma) — cloaking/disappearance threat.

### CORRECTIONS (verification caught these)
- ⚠️ **AnywhereDoor (arXiv 2503.06529) was WITHDRAWN** by author Jialin Lu. Use only as informal threat context (code exists: github.com/HKU-TASR/AnywhereDoor); do NOT cite as an authoritative source.
- ⚠️ **"TRIM"** (a workflow-named black-box method) could NOT be verified as a real paper — DROPPED.
- LeBD remains abstract-only (PDF CAPTCHA-gated) — cite at mechanism level, no numbers.

---

## UPDATE 3 — AS-BUILT DISPOSITION (2026-09-15 → 2026-09-30)

**Purpose.** The recommendations in `CV_INTEGRITY_ASSURANCE_2026.md` have now been *met by measurement*.
Two of them came back negative, one shared-code defect invalidated earlier attack-arm numbers, and the
engine grew governance machinery the dossier never asked for. This section records what the
implementation changed, with a receipt for each claim, so no future session re-derives it.

One-line summary: **the research question moved from "can these detectors be built?" to "which of them
actually fires, and on what?" — and the answer was narrower than the dossier expected.**

### 3.1 Where the dossier's recommendations landed

| Dossier recommendation (§ / capability) | As-built state | Receipt or path |
|---|---|---|
| Implement **TRACE** (CTC + FTC) as the black-box baseline — §4, cap. (b) | **Implemented, one arm only.** `trace_ctc` blends backgrounds; TRACE's *foreground* arm (clean samples are more consistent under focal information) is missing | `cviaf/lab/detectors.py` (`trace_ctc`, `trace_ftc`) |
| **DistScan** pre-NMS distribution check as a cheap always-on signal — §4 | **Implemented and measured** as `pre_nms_class_js` | `cviaf/lab/detectors.py::pre_nms_class_divergence` |
| Detector-native label screening (**cleanlab ObjectLab**) — §3, cap. (a) | Confident-learning screen in the same family; the library itself is *not* vendored | `cviaf/data_integrity/__init__.py` (its own docstring says "similar to Cleanlab") |
| Near-duplicate / flooding screen (**SSCD**) — §3 | A deterministic frozen-conv embedding stands in; SSCD is the named scaling target | `cviaf/lab/attribute.py`; `docs/SCALING_PLAN.md` |
| Offline **C2PA 2.4**-style manifest — §5, cap. (c) | **Not done.** The provenance module uses a repository-local schema, with an HMAC-SHA256 fallback in the committed run | `runs/mvp/comparison.json` → `provenance_axis.signing_mode` |
| Hash-chain sequence log for replay/reordering (**AuditableLLM** pattern) — §5 | Implemented and validated | `runs/assurance/oda_s5/assurance_report.json` → `metadata.audit_trail` (7 entries, chain valid) |
| **OpenOOD v1.5** protocol for the OOD screen — §3 | Named as the validation target; not run | — |
| **ODSCAN** / **DISTIL** trigger inversion (white-box path) — §4 | Not implemented | — |
| **NIST AI RMF** vocabulary for coverage + disposition — §7 | Adopted | `docs/COVERAGE_STATEMENT.md`, `docs/PS26228_ALIGNMENT_MATRIX.md` |
| Prefer block/within-window drift tests and publish the evasion test — §6 | Adopted as an honest limitation; the under-determined verdict is now emitted per cell | `docs/COVERAGE_STATEMENT.md`; alignment matrix C4 |

### 3.2 Four measured results that revise the dossier's expectations

**(a) TRACE is half-implemented, and the missing half explains the null control.** The image-level
control is a clean model whose test images carry a real trigger (no backdoor anywhere). All three
image-level detectors read *materially above* chance on it: **CTC 0.816, refdiv 0.786, FTC 0.792**.
TRACE's own second observation — that *clean* samples are the more consistent ones under focal
information, a mirror-sign foreground arm — is the citable explanation, and it is the single most
concrete implementation gap in the engine. **Consequence: every image-level number must be read net of
this control**, and the dossier's §4 description of TRACE should be read as "one of its two arms".

**(b) The attack-success criterion was measuring the trigger's ink, not a backdoor.** With an ASR gate
that re-places the trigger per image and subtracts **two placebos drawn from the same RNG stream**
(channel-swap and mirrored) against each model's seed-matched clean twin: **three of the four trigger
recipes never implanted at 64×64.** Against the previously published raw rates, `oga` / `oda` / `rma`
net to **0.000** (only `gma` shows a placebo-robust effect, and only on some seeds). The corpus now
carries raw, null and net as three separate columns; a submission quoting only the raw rate claims a
capability it does not have. Follow-on measurement (§10 of `docs/MEASUREMENT_NOTES.md`): for generation
attacks a *clearer* trigger makes a *worse* backdoor — `oga` with a high-salience checkerboard patch is
exactly **0.000 at poisoning rate 0.5**, while the low-amplitude *blended* trigger reaches net 0.525.
Practical consequence: the trigger axis needs **seed count, not recipe tuning** (~30 seeds for 5
positives), and `oda` (disappearance) remains **not covered** on the data axis.

**(c) `bias_lift` is behaviour-inert — the backdoor-like case is a measured failure.** All **396**
bias-lift arms show zero relative F1 change, so the headline "5.1% caught" equals the rule's own
false-alarm rate. The rule fires *at chance*, not on a backdoor. This is the dossier's §4 warning about
adaptive/clean-label evasion and "natural Trojans" (TrojAI §7) showing up as a concrete, named blind
spot rather than a caveat.

**(d) Drift is not merely "provably inseparable" — it is measured mostly undecided.** The evadability
result (arXiv 2411.16591) predicted the difficulty; the as-built drift battery then produced **8
decisions against 32 under-determined cells** at n=160 per cell. The honest verdict is emitted per
cell, and attribution exists as code without a receipt of the false-negative half's weight. Alignment
matrix verdict: **C4 = PARTIAL.**

### 3.3 Method-integrity events — read before comparing any pre-fix number

These are the events that make old numbers non-comparable. Each has a test or a receipt.

1. **Frozen thresholds.** The operating point is the 5%-quantile of **28,313 held-out clean models',
   negatives, fixed *before* any attacked arm was scored.** Attacks are never used to pick the
   threshold. This is the dossier's implicit "calibrate the cost first" made mechanical.
2. **A shared-code defect invalidated earlier weight-tamper arms.** `detector.tamper_prune` counted
   hidden units as `Wh.shape[0]` (input dim) and indexed `bh` with it — silently pruning 4/48 units
   under the synthetic defaults, and raising `IndexError` at the real shape `c2=64 > hidden=48`.
   Fixed to `Wh.shape[1]`, which **changes attack-arm behaviour** (12/48 at prune fraction 0.25).
   **Any weight-tamper number predating that fix must be re-run before comparison.**
3. **Two arms sets that must never be merged.** A single-dose 193-arm experiment reports substitution
   **46/94** and weight tamper **21/99**; the 4-dose ladder reports **45/94** and **92/99** for the same
   rule. They are different arms. Pooling or averaging them is a double count.
4. **Saturated rules are bounds, not specificity.** `ctc_peak_clean` and `ctc_q95_clean` sit at the
   corpus ceiling: FPR 0.000 means the rule *cannot fire*, not that it is precise.
5. **Pinned instruments.** The integration scorer is pinned at commit `a9e4ce0` and the reference
   model is `clean_none_fixed_s5800` (manifest SHA-256 `fd84409c…`, weights `434561bb…`). Do not
   re-pin or update either when re-scoring; the FPR/TPR numbers are relative to them.

### 3.4 Machinery the dossier did not ask for (and that now carries weight)

- **Fail-closed corpus admission.** Census → verify → merge plans → score → merge results, in that
  order. `runs/integration_verify.json`: **56,628 / 56,628 verified across 11 sources, 0 failed, 0
  quarantined, 0 warnings**; failures are moved aside, never deleted and never scored.
- **A number audit instead of an afternoon of grep.** `scripts/number_audit.py` re-derives every quoted
  figure from the receipts (18 checks, exit 3 on drift), refuses receipts that disagree with
  *themselves* (cells vs `n_arms`, survivorship map vs cells) before consulting prose, and bars
  superseded figures outside their history sections. Pinned by `tests/test_number_audit.py` (8 tests).
- **A dashboard that can refuse.** `scripts/demo_dashboard.py` re-splits and re-derives the headline
  itself and **refuses to write the page if its arithmetic disagrees with the committed report**; the
  shipped `demo/fpr_dashboard.html` (3.1 MB) carries all 56,627 score-sets as embedded base64 floats
  and recomputes statistics interactively, including naming a rule degenerate rather than showing 0%.
- **An improvement loop with a hard network boundary.** `cviaf/research` ranks provable gaps offline
  and publishes a query queue; a human or agent reads the source and records a finding with a URL and a
  number measured *here*. The shipped pipeline never imports it, and a test enforces that
  (`tests/test_research.py::test_no_network_in_shipped_pipeline`).
- **A three-column attack ledger** (raw / null / net) wherever attack success is reported — the direct
  consequence of 3.2(b).

### 3.5 Open items carried forward

| Item | State |
|---|---|
| No-egress bundle receipt (prove the offline bundle makes no attempt to leave) | **Open** — the one air-gap claim without a receipt |
| Byte-level image-overlap audit (**9,984,920** images, 440/model) | **Open** — until it lands, no claim of independent images or leakage-free evaluation |
| `refdiv` has no null on the *model* axis (between clean models from different seeds it scores AUROC 0.938–0.963) | **Open** — tolerance comes from the measured clean spread, not self-comparison |
| Real-backbone evidence corpus (1× H200, 1–2 sessions of 4–8 h, ~50 GB scratch) | **Requested**, not run; the ladder's arms are synthetic TinyDetector models |
| C5's two `provenance → inference_tampering` findings carrying no evidence dict and no affected asset | **Open** — 13/15 findings are complete |
| `label_flip` detected by neither baseline nor CVIAF on the data axis | **Declared** |
| `oda` (disappearance) not covered on the data axis | **Declared** |

### 3.6 The canonical numbers, and where they now live

Do not quote a number that is not in one of these receipts.

| Figure | Value | Receipt |
|---|---|---|
| Clean-null FPR, merged study (56,627 = 28,314 cal + 28,313 eval) | ctc_mean 1,481 = **5.23%** [4.98, 5.50]; refdiv_mean 1,467 = **5.18%** [4.93, 5.45]; ctc_peak / ctc_q95 degenerate at 1.0 | `runs/merged_fpr_tpr_report.json` |
| Fleet ledger (7,503 = 3,752 + 3,751) | ctc 4.80%; refdiv 5.73% | `runs/fleet_fpr_ledger_report.json` |
| Headline detection at the frozen threshold (refdiv_mean, α=0.05) | substitution 0.25 **45/94 = 47.9%**; 0.50 **59/63 = 93.7%**; weight_tamper 1.00 **92/99 = 92.9%**; bias_lift **5/99 = 5.1%** (= its own FPR, all 396 arms inert) | `runs/tpr_ladder_at_frozen.json` |
| CTC-mean at the *same* threshold | 2/94 · 2/63 · 29/99 · 6/99 — equal specificity, very unequal sensitivity | same receipt |
| Superseded 193-arm set | 46/94 · 21/99 — history only | `runs/tpr_at_frozen.json` |
| Corpus admission | 56,628 / 56,628 verified, 11 sources, 0 failed | `runs/integration_verify.json` |
| Capability record | C1 SATISFIED · C2 SATISFIED · C3 SATISFIED · **C4 PARTIAL** · C5 SATISFIED | `docs/PS26228_ALIGNMENT_MATRIX.md` |

## RESUME INSTRUCTIONS (for next session)
1. Read THIS file — it contains every verified source; do NOT re-run searches.
2. (Optional) skim `../cv_integrity_assurance_research_report.md` + `../cv-pipeline-integrity-reference.md` for the baseline so the new report explicitly goes beyond them.
3. **Then read UPDATE 3 below.** The consolidated report exists and the engine has since been measured against it: two of its expectations came back negative (TRACE implemented for the background arm only, null control 0.816; three of four trigger recipes never implanted), one shared-code defect invalidated earlier weight-tamper arms, and the drift battery decides only 8 of 40 cells. Do not re-argue what measurement has settled.
4. The live work queue is UPDATE 3 §3.5 (open items) — no-egress bundle receipt, byte-level image-overlap, `refdiv` model-axis null, the real-backbone corpus, the two incomplete C5 findings. Choose from there, not from the source list.
5. Tavily search still available via: `python3 /Users/billasur/.claude/jobs/94cf1732/tmp/tv.py "query"` (helper) OR curl to https://api.tavily.com/search with Bearer $(cat ~/.tavily_key). NOTE: job tmp is ephemeral (cleaned when this bg job is deleted) — if tv.py is gone, the one-liner curl still works.

## SECURITY REMINDER
Tavily dev key stored at `~/.tavily_key` (chmod 600, outside repo/git). **Rotate/revoke it when the project is done.**
