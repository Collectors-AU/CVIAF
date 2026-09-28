# TASK3_NOTES.md — real-backbone lane handover (skeleton phase, GLM 5.3)

Branch: `task3-real-backbone` (pushed). Base: `ab1d81b` (v2.0 calibrated-assure on main).
Nothing here touches `main`; nothing outside Task 3 was refactored (the three edits to shared
files are listed under GOTCHAS with their reasons).

Workspace: the lane was built in the git worktree `.task3/` of the main checkout, and it is left
in place on purpose — it has the branch checked out AND the warm CIFAR cache
(`.task3/data/cifar10/`, ~700 MB including the extracted PNGs and the 15,000-image npz), which
costs ~10 minutes of download plus ingest to rebuild. Two cautions: (1) `.task3/` is untracked
and NOT git-ignored in the parent checkout, so never run a broad `git add -A` from the main
tree — stage files explicitly, as every lane here does; (2) `data/` IS git-ignored, so the cache
can never be committed by accident. To work without the worktree:
`git fetch origin && git checkout -b task3-real-backbone origin/task3-real-backbone`.

Interpreter for every real-backbone command (torch 2.14.0, onnx 1.23.0, onnxruntime 1.30.0,
torchvision, scipy, scikit-learn):

    PY=/Users/billasur/.venvs/cviaf-torch/bin/python

`.venv` (Python 3.11) has no torch, so `tests/test_real_backbone.py` skips there by design and
the numpy-only suite stays green.

--------------------------------------------------------------------------------------------
## STATUS — done, and the smoke is green

Done:

| piece | file | what it is |
|---|---|---|
| backbone module | `cviaf/lab/real_backbone.py` | `RealBackboneDetector` (pretrained `resnet18_stem1`) subclassing `TinyDetector`; overrides only `features*`, `_backbone_params`, save/load; head, box decode, 3x3 peaks, NMS and `predict` are inherited unchanged. Plus `OnnxFeatureExtractor`, `export_onnx`, `to_onnx`, `export_parity`, `is_real_backbone_artifact` |
| data | `cviaf/lab/cifar.py` | one-time download + cache + `DetectionDataset` in the lab's shape |
| train script | `scripts/train_real_backbone.py` | trains through the real backbone, writes the existing artefact format, optional ONNX + parity, `--smoke` gate |
| artefact loader | `cviaf/lab/train.py` (`ModelArtifact.load`) | dispatches to `RealBackboneDetector` on the npz `_meta` marker, so verify/digest tooling is unchanged |
| tests | `tests/test_real_backbone.py` (torch interpreter), `tests/test_lab.py` prune regression | interface, determinism, save/load identity, ONNX parity, loader dispatch, manifest-key parity, prune axis |

Green smoke test (exact output, 2 models end to end, `bash`-reproducible):

    $ PY=/Users/billasur/.venvs/cviaf-torch/bin/python
    $ cd <repo>/.task3 && PYTHONPATH=. $PY scripts/train_real_backbone.py --smoke --seeds 0 1 --out runs/real_cifar_smoke
    real backbone: resnet18_stem1 pretrained=True finetune=False seeds=[0, 1]
    data: CIFAR-10 classes (0, 1, 2) n_per_class=300 cache=data/cifar10
    train 900 images  eval 60 images  train digest 14cccb3dfa8d2c8c
      training clean seed 0 ...
        s0 eval F1 0.041 precision 0.023 recall 0.200
        onnx parity: max|delta|=9.656e-06 agree=1.00 pass=True digest_unchanged=True
        wrote runs/real_cifar_smoke/realcifar_clean_s0 (11.81s, weights_digest cde1dcb7fb69b9f4)
      training clean seed 1 ...
        s1 eval F1 0.065 precision 0.036 recall 0.283
        onnx parity: max|delta|=9.656e-06 agree=1.00 pass=True digest_unchanged=True
        wrote runs/real_cifar_smoke/realcifar_clean_s1 (11.38s, weights_digest 3fa5ca8e8a3eb856)
    SMOKE GATE: PASS (f1_positive=True, parity_pass=True)
       realcifar_clean_s0: F1=0.0405 onnx_parity=True
       realcifar_clean_s1: F1=0.0645 onnx_parity=True
    trained 2 model(s) in 23.81s; clean F1 ['0.041', '0.065']; wrote runs/real_cifar_smoke

    exit code 0 (`--smoke` returns non-zero unless every model has F1 > 0 and an ONNX parity pass)

Suites: `.venv/bin/python -m pytest tests/ -q` → **168 passed, 1 skipped, 1 xfailed** (the skip
is `tests/test_real_backbone.py`, torch-only). Also verified: a real-backbone model loads through
`ModelArtifact.load`, its digest round-trips, and the ONNX-runtime detector has the *same* digest
as the torch one (identity survives the execution swap).

The smoke gate is the contract: **F1 > 0 and ONNX parity pass on every model**. Do not weaken it
to make a run look green; if it fails, the plumbing is broken.

--------------------------------------------------------------------------------------------
## NEXT COMMAND — starts the full training run (one line)

    cd <repo>/.task3 && PYTHONPATH=. /Users/billasur/.venvs/cviaf-torch/bin/python scripts/train_real_backbone.py --seeds 0 1 2 3 4 5 6 7 --n-per-class 300 --epochs 400 --eval-per-class 20 --out runs/real_cifar --onnx

Expected: 8 clean models, ~3-4 minutes total (the backbone is frozen, so features are computed
once per seed and the head trains on cached features — this is not an overnight job). Every
model writes `weights.npz`, `manifest.json`, `features.onnx`, `onnx_parity.json`; the run writes
`registry.jsonl` and `summary.json`. Run it, then paste the tail and the `summary.json`
`smoke_gate`-equivalent check into the handover thread. Then do REMAINING (c), which is the
actual overnight work.

--------------------------------------------------------------------------------------------
## REMAINING

**(a) Full training run** — the NEXT COMMAND above. Scale `--seeds` up if you want more clean
models (the asset rule needs 24 clean models at alpha .05 for the refdiv gate, so ≥24 seeds is
the useful target if you intend to run the asset battery; 8 is the skeleton default). If you
raise `--n-per-class`, expect a slower feature pass (~2 s per 300 images on CPU).

**(b) ONNX export of all models** — already wired: `--onnx` exports during training. For models
already on disk without it (or to re-export after any change):

    cd <repo>/.task3 && PYTHONPATH=. $PY -c "
    import glob, json, os
    from cviaf.lab.real_backbone import RealBackboneDetector, export_parity
    from cviaf.lab.cifar import load_cifar_subset
    ds = load_cifar_subset(n_per_class=5, seed=2000)
    for w in sorted(glob.glob('runs/real_cifar/*/weights.npz')):
        m = RealBackboneDetector.load(w); p = os.path.join(os.path.dirname(w), 'features.onnx')
        m.export_onnx(p); rec = {'model_id': os.path.basename(os.path.dirname(w)), **export_parity(m, p, ds.images)}
        json.dump(rec, open(os.path.join(os.path.dirname(w), 'onnx_parity.json'), 'w'), indent=1)
        print(rec['model_id'], rec['pass'], rec['max_feature_delta'])"

Note `*.onnx` is git-ignored in this repository, so exported graphs are never committed —
regenerate them with this snippet after a fresh clone (each is ~630 KB, ~1 s).

**(c) Rerun the two measurement batteries (baseline vs pipeline) on the real-backbone models.**
This is the real remaining work and the reason the lane exists. Order that works with what is
already verified:

1. Model-attack arms need no retraining: the head is plain numpy, and `tamper_head` /
   `tamper_prune` operate on it directly (verified on a real-backbone model: `tamper_head`
   moved F1 0.0405 → 0.0000 with a changed head digest; `tamper_prune` now works too — see the
   prune fix under GOTCHAS). Build `weight_tamper` / `substitution` arms from the clean models
   and write them into the corpus with the same manifest format, then run:
   `PYTHONPATH=. .venv/bin/python -m cviaf.lab compare --corpus runs/real_cifar --json runs/real_cifar/compare.json`
   and the null suite:
   `PYTHONPATH=. .venv/bin/python -m cviaf.lab.null_suite --corpus runs/real_cifar --attacks weight_tamper --seeds <seeds> --out runs/real_cifar/null_suite.json`
   (use `.venv` for these two: they are numpy-only — but the corpus *loads* torch models, so run
   them with the torch interpreter and `PYTHONPATH=.`, and remember `cryptography` is absent there,
   which only matters if the provenance path is exercised).
2. Image-level (dataset) attacks need retraining through a poisoned CIFAR dataset: `inject` works
   on any `DetectionDataset`, so poison the loader output (`AttackSpec(kind='oga', trigger='patch',
   trigger_loc='fixed', ...)`) and feed the result to the same train script by extending it with a
   `--attack-kind` flag. **Not attempted in this phase — verify before reporting.**
3. Report per-battery with denominator + seed set + control type + CI, exactly as the other lanes do.

--------------------------------------------------------------------------------------------
## DECISIONS (frozen — do not redesign)

* **Backbone**: torchvision ResNet-18 truncated after `layer1` (`resnet18_stem1`): conv1 s2 +
  maxpool s2 + layer1 → stride 4, 64 channels, so `img_size=64` yields `G=16` = exactly
  `DetectorConfig.grid`. Wider/deeper backbones were rejected because they would change the grid
  and move it under the head. ImageNet weights by default, **frozen** (frozen is the fast path:
  features are computed once and cached, the same trick the synthetic pipeline uses).
  `--finetune` opts into backbone gradients and is untested at scale.
* **Data**: CIFAR-10 classes 0/1/2 (airplane, automobile, bird) → detector labels 0/1/2.
  One object per image: a centred square box of **2 grid cells (8 px)**, with the centre snapped
  to the centre **cell** (34 px at `img_size` 64 — see GOTCHAS, this is not cosmetic).
  Images: 32×32 → 64×64 nearest-neighbour index repeat, [0, 1] float32, ImageNet-normalised
  inside the extractor.
* **Download**: fast.ai mirror `https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz` (PNG
  layout), because the canonical Krizhevsky host served this machine at ~24 KB/s vs ~2 MB/s
  measured. The PNG ingest decodes only the declared classes (15,000 images) into
  `data/cifar10/cifar10_train.npz` (one-time, ~2 min); the canonical pickle tarball remains the
  fallback. `data/` is git-ignored, so no dataset is committed.
* **Hyperparameters** (defaults in the script, used by the smoke): Adam lr 3e-3, batch 32,
  hidden 48, `pos_weight` 30 on the objectness positive, `ignore_radius` 1, epochs 400 for the
  full run (200 in the smoke), eval split 20/class (seed 2000) disjoint from train (seed 1000).
* **Artefact format**: identical to `cviaf/lab/train.py` — `<out>/<model_id>/weights.npz` +
  `manifest.json` with the same top-level keys (`lab_version, model_id, created_utc, spec,
  spec_digest, dataset_digests, attack_digest, seeds, artifact, ground_truth, metrics,
  quality_flags, timing_seconds, environment`). Backbone tensors are stored inside the same npz
  under a `bb__` prefix with a `_meta` JSON blob (`kind: real_backbone`, `bb_order`). `load`
  dispatches on that marker.
* **ONNX**: the exported graph is the **feature extractor only** (the numpy head is portable
  already). `runtime="onnx"` swaps only `features`, and it keeps the same parameter arrays, so
  the model digest is unchanged by the swap — that is what makes `export_parity` meaningful.

--------------------------------------------------------------------------------------------
## GOTCHAS

1. **torch 2.14's ONNX exporter needs `onnxscript`** (`ModuleNotFoundError`), because ≥2.6
   defaults to the dynamo exporter. `export_onnx` pins `dynamo=False` (TorchScript exporter, no
   extra dependency). If you install `onnxscript` anyway, keep `dynamo=False` for reproducible graphs.
2. **Training budget below ~200 epochs reads F1 = 0.000** — measured, not a plumbing failure: the
   objectness head had not learned yet. The smoke therefore uses `n_per_class=300, epochs=200`.
3. **Three real bugs were found and fixed in this phase; all are disclosed in the commit.** Two
   were mine (the objectness mask excluded the positive cell, so the head was trained on
   negatives only and collapsed to 0 everywhere — visible as F1 0.000 with the class head
   perfectly confident; and the box centre was half a cell off the grid, capping a perfect
   prediction at IoU 0.39). The third is **pre-existing shared code**:
   `cviaf/lab/detector.py::tamper_prune` counted hidden units as `Wh.shape[0]` (the input width)
   and indexed `bh` with it — silently pruning `frac*c2` = 4/48 units with the synthetic
   defaults, and raising `IndexError` at the real-backbone shape `c2=64 > hidden=48`. It now uses
   `Wh.shape[1]`. **This changes attack-arm behaviour**: `weight_tamper` prunes `frac*hidden` =
   12/48 at `frac=0.25` where it used to prune 4/48. Any *pre-fix* weight-tamper number was
   produced with the smaller, unintended prune — re-run before comparing. A regression test pins
   the new behaviour (`tests/test_lab.py::test_tamper_prune_uses_the_hidden_axis`).
4. **`~/.venvs/cviaf-torch` has no pytest** (`No module named pytest`), so the torch-only tests
   cannot run there as-is. One install fixes it:
   `/Users/billasur/.venvs/cviaf-torch/bin/python -m pip install pytest` (no other dev dependency
   is missing). Until then, run the numpy-only suite with `.venv` and the smoke with the torch
   interpreter.
5. **`cryptography` is absent from the torch venv.** Not needed for the parity gate (it calls
   `ModelIntegrityAssessor` directly and never builds the provenance engine), but a full product
   pipeline from that interpreter needs it — the v4 provenance code refuses the symmetric fallback.
6. **MPS is available but unused** (`--device` does not exist; everything runs on CPU). That is
   deliberate for determinism; do not "optimise" it without re-checking that the same seed still
   yields the same `weights_digest`.
7. **The npz marker matters.** A real-backbone `weights.npz` fed to the old
   `TinyDetector.load` path would silently rebuild a *random* backbone and keep the trained head —
   a wrong model that still looks loadable. The `_meta.kind` check in `ModelArtifact.load` is what
   prevents that; do not bypass it.
8. **Objectness precision is low by construction** (0.023–0.036 at the smoke budget, recall
   0.20–0.28). A CIFAR image *is* the object: it fills the frame, so "one centred 2-cell box" is a
   declared synthetic convention and neighbouring cells are genuinely ambiguous. This caps
   detection quality and it is **not** a Task 3 goal — the batteries measure detector/pipeline
   behaviour on real weights, not mAP. Do not spend the night tuning it; if a higher-precision
   body is needed, change the annotation convention (e.g. a cropped object region) and say so.
9. **The smoke artifacts are committed** (`runs/real_cifar_smoke/*/weights.npz` + manifests +
   parity + `summary.json`, ~1.2 MB) as machine-checkable evidence; `features.onnx` is not
   (git-ignored) — regenerate with REMAINING (b).
