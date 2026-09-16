# CVIAF — Project Work Plan (laptop-first, one thing at a time)

**Source of truth:** `PRD_SIH26228_CVIAF.md` (what to build) · `CV_INTEGRITY_ASSURANCE_2026.md` (how to build it, 2026 SOTA) ·
`SETUP_AND_REALDATA_FINDINGS.md` (env status + 3 confirmed bugs) · `README.md` (known issues)
**Rule:** one Thing at a time. Each Thing has: WHAT · WHY · STATUS · GAP · WORK · DONE =.

---

## PHASE 0 — Foundation plumbing (do first; unblocks everything else)

### Thing 1 — Fix COCO loader image-dir inference (Bug #1) and prove all 4 modules run on real data ✅ **DONE**
- **WHY:** right now `data_integrity`, `drift`, and `provenance` are silently skipped on real data; only `model_integrity` runs.
- **STATUS (was):** `cviaf/formats/__init__.py` `load_dataset()` looks for image dirs only *inside* the annotations parent; standard COCO layout has `annotations/` and `val2017/` as siblings → falls back to wrong dir → all `image_path`s point to non-existent files.
- **WORK (done):** (a) Added `_infer_coco_images_dir()` helper that checks annotation parent AND sibling/grandparent dirs, validates candidates against JSON `file_name` values; (b) re-ran `cviaf assess` on 80-image subset + `models/yolov8n.onnx`; (c) audit trail now shows `module_started` for `data_integrity`, `model_integrity`, `distribution_shift` (provenance skipped by design — needs `inference_records`).
- **DONE =:** ✅ report contains `training_data_integrity` (3 findings), `model_integrity` (27 findings), `distribution_shift` (0 findings), all image paths resolve to real files.
- **Discovered during run (→ feed into next Things):**
  - Data integrity: all 3 findings are false positives on clean COCO (near-dup + OOD) → **Thing 5 calibration harness needed**.
  - Model integrity: 26 false positives from static kurtosis > 10 threshold (Bug #2 — **T11**); Neural Cleanse skipped due to shape mismatch (CLI passes 224×224 images but YOLOv8n needs 640×640 letterbox — **Thing 4**).
  - Drift PCA fixed: capped `n_components` at `min(n_samples, n_features)` to avoid crash on small datasets.

### Thing 2 — Fix metadata plumbing (Bug #3): dict ↔ SampleMetadata normalization
- **WHY:** `_compute_source_risks()` crashes on CLI-passed dict metadata → any real run with contributor metadata dies inside capability (a).
- **WORK:** single normalization helper in `core/types.py` (dict → SampleMetadata); use it in CLI `cmd_assess` and loader paths.
- **DONE =:** `assess` with `--contributor` produces `source_risks` without crashing.

### Thing 3 — Fix provenance key regeneration + surface silent degradation (README known issues)
- **WHY:** `InferenceProvenanceEngine.__init__` regenerates keys every time → every seal verifies as invalid (demo flags all 10). Also `assess` silently runs partial modules without telling the user.
- **WORK:** load-or-create keypair from `key_dir`; report per-module status (ran / skipped / error reason) in the report and console.
- **DONE =:** seal created and verified by a *fresh* engine round-trip succeeds; report explicitly names skipped modules + reason.

### Thing 4 — Model input preprocessing adapter (letterbox 640, RGB, 0-1 normalize) ⭐ **NEXT — blocks Neural Cleanse**
- **WHY (confirmed by Thing 1 run):** YOLOv8n ONNX expects `1×3×640×640` letterboxed RGB; CLI naively resizes 224×224 → model-dependent checks (Neural Cleanse, entropy probes, drift logits) run on garbage predictions or crash. Neural Cleanse **skipped** with error: `operands could not be broadcast together with shapes (1,640,640) (32,224,224,3)`.
- **WORK:** add `preprocess()` to `ModelWrapper` (letterbox + normalize + RGB) and use it consistently in CLI + orchestrator for images AND probes.
- **DONE =:** YOLOv8n actually detects real objects in the 80 images when run through the wrapper; `predict()` used everywhere returns sane logits/boxes; Neural Cleanse runs without shape errors.

---

## PHASE 1 — Capability (a) Training-Data Integrity ⭐ YOUR focus (2026 §3)

### Thing 5 — Baseline: clean data must produce ~0 findings (calibration harness)
- **WORK:** pytest harness that runs all 4 data detectors on clean real COCO; tune thresholds until false-positive rate ≤ small bound.
- **DONE =:** clean run = 0 trigger, 0 label, 0 dup, 0 OOD findings (INFO-level notes allowed); numbers recorded in a results JSON.

### Thing 6 — Trigger detection calibration (BadNets + blended) with real photos
- **WORK:** inject BadNets patch triggers (5/10/20% poison) and blended triggers via `cviaf/attacks`; measure per-detector recall + FP of TriggerDetector (pixel / FFT / spectral); tune.
- **DONE =:** detection rate table (poison% → recall) in results; triggers at 10%+ reliably flagged; missed cases documented as limitations.

### Thing 7 — Label integrity calibration (contributor-scoped flips)
- **WORK:** LabelFlipper scoped per contributor; need ≥50 samples/class (scale subset to ~300 images); measure confident-learning detection.
- **DONE =:** flip rate 10%+ flags the right contributor; per-source risk score moves correctly.

### Thing 8 — Duplicate flooding + OOD insertion calibration on real photos
- **WORK:** create near-dups (copy + crop/rotate per attack taxonomy) and OOD inserts (non-COCO class images); calibrate DuplicateDetector + OODDetector to kill current clean-data false positives.
- **DONE =:** floods >30% and OOD >10% reliably flagged per contributor.

### Thing 9 — Contributor risk aggregation validation (2026 §3.5)
- **WORK:** ground-truth 3-contributor scenario (2 clean, 1 poisoned); validate the aggregation rule (beta-binomial posterior or mean+tail) and risk-level mapping.
- **DONE =:** poisoned contributor lands HIGH/CRITICAL with attack classes; clean ones LOW; rule documented.

### Thing 10 — Upgrades from 2026 §3 (stretch, in order)
- **WORK:** (a) cleanlab ObjectLab label pass (needs Thing 4 for real predictions); (b) imagededup PHash near-dup first pass; (c) per-RoI OOD under OpenOOD protocol; (d) SPECTRE robust covariance on per-box RoI features.

---

## PHASE 2 — Capability (b) Model Integrity (friend's focus; coordinate at Thing 1-4 completion)

- **T11** Fix kurtosis false positives (Bug #2): reference-battery comparison instead of static `kurtosis > 10`.
- **T12** Neural Cleanse for YOLO: correct wrapper I/O (Thing 4), clean_images path, detector-native trigger shapes.
- **T13** Wire TRACE (black-box) + later ODSCAN/DISTIL per 2026 §4; laptop: harness + 1-2 models; H200: sweeps.

## PHASE 3 — Capability (c) Provenance
- **T14** Seal round-trip verified (after Thing 3); replay/reorder tests on hash-chain sequence log.
- **T15** C2PA-style manifest emission (asset type / reference / collection hash / Output Content Credential) per 2026 §5.

## PHASE 4 — Capability (d) Drift
- **T16** Drift-vs-manipulation calibration on real COCO: terrain/season/illumination augmentation (operational drift) vs ART adversarial (manipulation); block-based detection per 2026 §6; honest ambiguity language.

## PHASE 5 — Capability (e) Governance + tests + submission docs
- **T17** Coverage statement per run; per-finding `confidence` + `method`/`access_mode` fields; schema polish.
- **T18** Full pytest suite (Things 5-16 → tests), `scripts/`, `configs/`, offline wheel rehearsal (air-gap install test).
- **T19** Submission pack: architecture note, assurance-report schema, reproducible audit log, coverage statement.

## PHASE 6 — H200 benchmark (execution-only; laptop work done in Phases 0-5)
- **T20** Train clean + poisoned detector corpus (14-16 models, BadDet OGA/RMA/GMA/ODA), TRACE/ODSCAN/DISTIL sweeps, OpenOOD drift calibration, results JSON copied back.

---

## Image compatibility verdict (80-image subset)
- ✅ 80/80 valid JPEGs, zero corruption; filenames match `instances_val2017_subset80.json` exactly (0 missing, 0 extra).
- ✅ 562 real annotations across 78 images (incl. 2 empty images — useful as OOD/absence test cases).
- ✅ YOLOv8n ONNX accepts these images with proper 640×640 letterbox (30+ detections at >0.5 conf).
- ⚠️ Preprocessing gap: model **crashes** with 224×224 input (expects 1×3×640×640). Must fix first (Thing 4).
- ⚠️ Scale for calibration: 80 images is enough for Things 5–6 (baseline + trigger calibration). Things 7–8 (label flip, dup flooding) need ~200–300 images for meaningful statistics. Full val set (5000) needed for H200.