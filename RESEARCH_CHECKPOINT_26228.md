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
- [ ] **RESUME HERE → Synthesize consolidated report** tied to engine modules. That's the only remaining step. Write to `cv-assurance-engine/CV_INTEGRITY_ASSURANCE_2026.md` (or as user directs).
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

## RESUME INSTRUCTIONS (for next session)
1. Read THIS file — it contains every verified source; do NOT re-run searches.
2. (Optional) skim `../cv_integrity_assurance_research_report.md` + `../cv-pipeline-integrity-reference.md` for the baseline so the new report explicitly goes beyond them.
3. Write the consolidated, cited report → suggest `cv-assurance-engine/CV_INTEGRITY_ASSURANCE_2026.md`. Structure: exec summary → per-capability (State of the art 2025–26 → Gap vs baseline → Concrete recommendation for the specific engine module → How to validate on H200). Lead the differentiator on the detector-specific gap (TRACE/AnywhereDoor/BadDet). End with an H200 benchmarking plan (TrojAI OD round, BackdoorBench, OpenOOD).
4. Tavily search still available via: `python3 /Users/billasur/.claude/jobs/94cf1732/tmp/tv.py "query"` (helper) OR curl to https://api.tavily.com/search with Bearer $(cat ~/.tavily_key). NOTE: job tmp is ephemeral (cleaned when this bg job is deleted) — if tv.py is gone, the one-liner curl still works.

## SECURITY REMINDER
Tavily dev key stored at `~/.tavily_key` (chmod 600, outside repo/git). **Rotate/revoke it when the project is done.**
