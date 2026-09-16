# CVIAF — Change Log

**Rule:** Every code change, fix, or update is recorded here with: what changed, why, and verification.

---

## Data & assets reference (what's in repo vs. what needs downloading)

| Asset | In repo? | Size | How to get |
|-------|----------|------|------------|
| `data/coco/val2017/` (80 images) | ✅ `git pull` | 13 MB | Already in repo |
| `data/coco/annotations/instances_val2017_subset80.json` | ✅ `git pull` | 368 KB | Already in repo |
| `data/coco/annotations/instances_val2017.json` (full 5,000-image annotations) | ✅ `git pull` | 19.9 MB | Already in repo |
| `models/yolov8n.onnx` | ✅ `git pull` | 12 MB | Already in repo |
| `yolov8n.pt` | ✅ `git pull` | 6.3 MB | Already in repo |
| **`annotations_trainval2017.zip`** (full train+val annotations archive) | ❌ too large for GitHub | 252 MB | See below ↓ |
| **`val2017.zip`** (full 5,000 val images) | ❌ too large for GitHub | 5 GB | See below ↓ |
| **`train2017.zip`** (~118K train images, H200 phase only) | ❌ too large for GitHub | 19 GB | See below ↓ |

### Large file downloads (needed when scaling beyond the 80-image laptop subset)

**Full annotations archive (252 MB)** — `annotations_trainval2017.zip` contains both `instances_train2017.json` and `instances_val2017.json` for all 5,000 val + 118K train images. You already have `instances_val2017.json` in the repo, so this is only needed if you want `instances_train2017.json` or the raw source archives.

```bash
# Full annotations (train + val)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# Extract into data/coco/
unzip annotations_trainval2017.zip -d data/coco/
```

**Full val images (5 GB)** — needed for Things 7–8 (label flip / dup flooding calibration at 200–500 images).

```bash
# All 5,000 val images
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d data/coco/
```

**Train images (19 GB)** — only needed for the H200 phase (full detector training corpus).

```bash
# All ~118K train images (H200 only)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d data/coco/
```

**Individual images (what we used for the 80-image subset)** — you can also download specific images by ID:

```bash
# Example: download a single COCO val image (300KB)
wget http://images.cocodataset.org/val2017/000000397133.jpg -P data/coco/val2017/
```

---

## 2026-09-16 — Thing 1: Fix COCO loader image-dir inference (Bug #1)

### Problem
The COCO loader in `cviaf/formats/__init__.py` inferred the images directory by looking **only inside** the annotation file's parent folder (`data/coco/annotations/`). Standard COCO layout puts `annotations/` and `val2017/` as **siblings** under `data/coco/`. Nothing matched → fell back to `imgs_dir = parent` → every `image_path` became `data/coco/annotations/XXXX.jpg` (non-existent) → `os.path.exists()` failed for all samples → `images=None` → `data_integrity` and `drift` silently skipped; only `model_integrity` ran.

### Changes

#### 1. `cviaf/formats/__init__.py` — Added `_infer_coco_images_dir()` helper
```python
def _infer_coco_images_dir(annotation_file: str) -> str:
    # Strategy (most specific match wins):
    # 1. Collect candidates from inside parent AND grandparent (sibling dirs)
    #    using standard COCO names + any dir containing image files.
    # 2. If annotation JSON readable, count how many referenced file_name
    #    values actually exist under each candidate.
    # 3. Return candidate with most real hits; fall back to dir with images;
    #    last resort = annotation parent (old behavior).
```
- Checks `parent` (annotation dir) AND `grandparent` (data root) for known COCO names: `images`, `train2017`, `val2017`, `test2017`, `train`, `val`, `test`
- Also scans for any subdirectory that directly contains image files (`.jpg`, `.png`, etc.)
- Validates candidates against actual `file_name` entries in the JSON — the candidate with the most existing files wins
- Fallback chain: JSON-validated → directory-with-images → annotation parent

#### 2. `cviaf/formats/__init__.py` — Simplified COCO branch in `load_dataset()`
**Before:**
```python
if not imgs_dir:
    parent = os.path.dirname(ann_file)
    for candidate in ["images", "train2017", "val2017", "train", "val"]:
        p = os.path.join(parent, candidate)
        if os.path.isdir(p):
            imgs_dir = p
            break
    if not imgs_dir:
        imgs_dir = parent
```

**After:**
```python
if not imgs_dir:
    imgs_dir = _infer_coco_images_dir(ann_file)
```

#### 3. `cviaf/cli.py` — Added `--images-dir` CLI flag
```python
assess_parser.add_argument("--images-dir", default="",
    help="Directory containing dataset images (optional; auto-inferred from the annotation file otherwise)")
```
Passed through to `load_dataset(images_dir=args.images_dir or "")` so users can override auto-inference.

#### 4. `cviaf/drift/__init__.py` — Fixed PCA `n_components` crash on small datasets
**Two locations fixed:**
- Line ~392 (MMD subsample): `PCA(n_components=50)` → `PCA(n_components=min(50, n_samples, n_features))`
- Line ~405 (KS test): `PCA(n_components=100)` → `PCA(n_components=min(100, n_samples, n_features))`

**Why:** With only 80 samples, `n_components=100` raised `ValueError: n_components=100 must be between 0 and min(n_samples, n_features)=80`. This bug was **hidden** because `drift` was silently skipped before Thing 1.

---

### Verification
```bash
python -m cviaf assess \
  --dataset data/coco/annotations/instances_val2017_subset80.json \
  --model models/yolov8n.onnx \
  --format coco \
  --access-level white-box \
  --output out_real_fixed2
```

**Audit trail — all expected modules fired:**
```
module_started: data_integrity      ✅
module_started: model_integrity     ✅
module_started: distribution_shift  ✅
module_completed: data_integrity    ✅
module_completed: model_integrity   ✅
module_completed: distribution_shift ✅
module_error: (none)
```

**Report assessments:**
| Module | Findings | Status |
|--------|----------|--------|
| `training_data_integrity` | 3 | All false positives on clean COCO (near-dup ×2, OOD ×1) → **Thing 5 calibration needed** |
| `model_integrity` | 27 | 26 kurtosis false positives (Bug #2 → T11); Neural Cleanse skipped (shape mismatch) → **Thing 4** |
| `distribution_shift` | 0 | Correct — same data compared to itself |

**Image path resolution:** 80/80 samples now point to real files under `data/coco/val2017/`.

---

### Critical discoveries for next Things
1. **Neural Cleanse skipped** — error: `operands could not be broadcast together with shapes (1,640,640) (32,224,224,3)`. CLI passes 224×224 images but YOLOv8n expects 640×640 letterbox. **→ Thing 4 is the highest-priority unblocker.**
2. **Kurtosis false positives (Bug #2)** — 26 MEDIUM findings on healthy YOLOv8n. Static `kurtosis > 10` threshold wrong for conv weights. **→ T11 in Phase 2.**
3. **Data integrity false positives** — pixel-stats features flag COCO diversity as near-dup/OOD. **→ Thing 5 (calibration harness) next in Phase 1.**
4. **Provenance correctly skipped** — only runs when `inference_records` provided (not in CLI assess). By design.

---

## 2026-09-17 — Thing 2: Fix metadata plumbing (Bug #3): dict ↔ SampleMetadata normalization

### Problem
`_compute_source_risks()` and every downstream `.contributor` / `.source` access expect `SampleMetadata` objects, but metadata can arrive as plain **dicts** (JSON-style) from the orchestrator API / CLI. When it did, capability (a) died silently: the report contained `training_data_integrity: {"error": "'dict' object has no attribute 'contributor'", "findings": []}` — no crash, no source_risks, just a dead module.

**Before (repro):**
```python
orch.run_full_assessment(images=..., labels=..., metadata=[{"contributor": "acme_lab"}, ...])
# → training_data_integrity: {"error": "'dict' object has no attribute 'contributor'", "findings": []}
```

### Changes

#### 1. `cviaf/core/types.py` — Added `SampleMetadata.from_dict()` + `normalize_metadata()`
```python
@classmethod
def from_dict(cls, d):  # known keys mapped, unknown keys preserved in `extra`
    ...

def normalize_metadata(metadata):  # List[SampleMetadata | dict] -> List[SampleMetadata]
    # SampleMetadata passes through; dicts → from_dict(); non-mappable dropped
```
Single normalization helper (per work plan): one definition, reused everywhere.

#### 2. `cviaf/data_integrity/__init__.py` — Normalize at `assess()` entry (choke point)
```python
metadata = normalize_metadata(metadata)   # before any detector runs
```
One line at the engine boundary covers **every** caller — CLI, orchestrator API, loaders — so no downstream `.contributor` access can ever hit a raw dict again.

#### 3. `cviaf/cli.py` — Normalize in `cmd_assess`
```python
metadata = normalize_metadata(metadata)   # after building metadata list from samples
```
Explicit per work plan ("use it in CLI `cmd_assess`"); belt-and-braces on top of the engine boundary.

### Verification
```bash
# 1) Bug #3 reproduction path — orchestrator API with DICT metadata
python -c "orch.run_full_assessment(images=..., labels=..., metadata=[{'sample_id': str(i), 'contributor': 'acme_lab', 'source': 'cam_A'} ...])"
# 2) CLI real data with --contributor
python -m cviaf assess \
  --dataset data/coco/annotations/instances_val2017_subset80.json \
  --model models/yolov8n.onnx --format coco --access-level white-box \
  --contributor acme_lab --output out_thing2
```

**Result — both pass:**
```
API dict metadata:  error: NONE
  source_risks: {"acme_lab": {"type": "contributor", "total_samples": 10, "finding_count": 0, ...},
                 "source:cam_A": {"type": "source", ...}}

CLI --contributor: Overall Risk: HIGH ✓
  source_risks: {"acme_lab": {"type": "contributor", "total_samples": 80,
                 "finding_count": 167, "risk_score": 7.0537, "risk_level": "CRITICAL",
                 "attack_classes": ["near_duplicate_flooding", "ood_insertion"]}}
```
**DONE =:** ✅ `assess` with `--contributor` produces `source_risks` without crashing (dict and SampleMetadata both work).

---

## Upcoming — Thing 4: Model input preprocessing adapter (letterbox 640, RGB, 0-1 normalize)

**Goal:** Add `preprocess()` to `ModelWrapper` (letterbox + normalize + RGB) and use it consistently in CLI + orchestrator for images AND probes.

**Unblocks:** Neural Cleanse, entropy probes, drift logits, any model-dependent check using real images.

**Expected done state:** YOLOv8n detects real objects in 80 images through wrapper; `predict()` returns sane logits/boxes; Neural Cleanse runs without shape errors.