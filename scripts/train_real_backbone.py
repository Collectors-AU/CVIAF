#!/usr/bin/env python
"""Train the real-backbone detector on a CIFAR-10 subset. Task 3 skeleton.

Runs through a real pretrained backbone (torchvision ResNet-18 stem+layer1, frozen by
default) and the synthetic lab's numpy head, then writes the SAME artefact format as
``cviaf/lab/train.py`` - ``<out>/<model_id>/weights.npz`` + ``manifest.json``, plus a
``registry.jsonl`` - so ``load_registry``, the digest tooling and the batteries consume
real-backbone models with no code change.

SMOKE (must be green before the overnight job):
  <py> scripts/train_real_backbone.py --smoke --seeds 0 1 --out runs/real_cifar_smoke

OVERNIGHT (see TASK3_NOTES.md):
  <py> scripts/train_real_backbone.py --seeds 0 1 2 3 4 5 6 7 --n-per-class 300 \\
      --epochs 40 --out runs/real_cifar --onnx

DECISIONS (frozen)
  * pretrained ResNet-18, frozen extractor (``--finetune`` opts into backbone gradients);
    frozen is the fast path because features are computed once and cached, which is the
    same trick the synthetic pipeline uses.
  * the head is trained in torch (so gradients genuinely flow from the real features)
    and then copied into the numpy head arrays; the numpy head at inference is
    bit-identical in structure to ``TinyDetector``'s.
  * labels: one object per image at the image centre, class = CIFAR class 0/1/2.
  * box targets are the exact inverse of ``TinyDetector.decode_boxes`` for the declared
    centred box, so detection quality measures objectness/class, not box regression luck.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cviaf.lab.cifar import DEFAULT_CACHE, DEFAULT_CLASSES, load_cifar_subset
from cviaf.lab.detector import DetectorConfig
from cviaf.lab.real_backbone import (REAL_BACKBONE_VERSION, RealBackboneConfig,
                                     RealBackboneDetector, build_feature_extractor,
                                     export_parity)
from cviaf.lab.synth import DetectionDataset
from cviaf.lab.train import detection_quality

LAB_VERSION = "task3-real-backbone-1"
ATTACK_DIGEST = hashlib.sha256(b"clean").hexdigest()[:16]


# ---------------------------------------------------------------- targets ---- #

def encode_box_raw(box: np.ndarray, gi: int, gj: int, cell: float) -> np.ndarray:
    """Inverse of TinyDetector.decode_boxes for one box/cell (exact, clipped)."""
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    w, h = box[2] - box[0], box[3] - box[1]
    t0 = np.arctanh(np.clip((cx - (gj + 0.5) * cell) / cell, -0.999, 0.999))
    t1 = np.arctanh(np.clip((cy - (gi + 0.5) * cell) / cell, -0.999, 0.999))
    t2 = np.log(np.expm1(max(w / cell, 1e-4)))
    t3 = np.log(np.expm1(max(h / cell, 1e-4)))
    return np.asarray([t0, t1, t2, t3], np.float32)


def build_targets(ds: DetectionDataset, grid: int, cell: float, radius: int = 1):
    """obj (N,G,G), cls (N,G,G), positive mask (N,G,G), box raw (N,G,4)."""
    n = len(ds)
    obj = np.zeros((n, grid, grid), np.float32)
    cls = np.zeros((n, grid, grid), np.int64)
    pos = np.zeros((n, grid, grid), bool)
    box_raw = np.zeros((n, grid, 4), np.float32)
    for i in range(n):
        # tolerate both conventions: boxes[i] as (4,) or as (Ni, 4)
        b = np.asarray(ds.boxes[i], np.float32).reshape(-1, 4)[0]
        lab = int(np.asarray(ds.labels[i]).reshape(-1)[0])
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        gj = min(max(int(cx // cell), 0), grid - 1)
        gi = min(max(int(cy // cell), 0), grid - 1)
        obj[i, gi, gj] = 1.0
        cls[i, gi, gj] = lab
        pos[i, gi, gj] = True
        box_raw[i, gi] = encode_box_raw(b, gi, gj, cell)
    return obj, cls, pos, box_raw


# ------------------------------------------------------------- torch head ---- #

def make_head(c2: int, hidden: int, n_classes: int, seed: int):
    import torch
    torch.manual_seed(seed)

    class Head(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l1 = torch.nn.Linear(c2, hidden)
            self.obj = torch.nn.Linear(hidden, 1)
            self.cls = torch.nn.Linear(hidden, n_classes)
            self.box = torch.nn.Linear(hidden, 4)
            self.register_buffer("mean", torch.zeros(c2))
            self.register_buffer("std", torch.ones(c2))

        def forward(self, feats):                 # (B, C2, G, G)
            x = feats.permute(0, 2, 3, 1)         # (B, G, G, C2)
            z = (x - self.mean) / self.std
            h = torch.relu(z @ self.l1.weight.t() + self.l1.bias)
            return (h @ self.obj.weight.t() + self.obj.bias,
                    h @ self.cls.weight.t() + self.cls.bias,
                    h @ self.box.weight.t() + self.box.bias)

    return Head()


def copy_head_to_numpy(model: RealBackboneDetector, head, mean: np.ndarray, std: np.ndarray):
    import torch
    with torch.no_grad():
        model.Wh = head.l1.weight.detach().numpy().T.astype(np.float32)
        model.bh = head.l1.bias.detach().numpy().astype(np.float32)
        model.wo = head.obj.weight.detach().numpy()[0].astype(np.float32)
        model.bo = np.float32(head.obj.bias.detach().numpy()[0])
        model.Wc = head.cls.weight.detach().numpy().T.astype(np.float32)
        model.bc = head.cls.bias.detach().numpy().astype(np.float32)
        model.Wb = head.box.weight.detach().numpy().T.astype(np.float32)
        model.bb = head.box.bias.detach().numpy().astype(np.float32)
    model.feat_mean = mean.astype(np.float32)
    model.feat_std = std.astype(np.float32)
    model.trained = True


def extract_features(extractor, images: np.ndarray, cfg: RealBackboneConfig,
                     batch: int = 32) -> np.ndarray:
    """(N,H,W,3) [0,1] -> (N,C2,G,G) raw features via the real backbone."""
    import torch
    mean = np.asarray([0.485, 0.456, 0.406], np.float32)
    std = np.asarray([0.229, 0.224, 0.225], np.float32)
    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            chunk = (images[i:i + batch] - mean[None, None, None, :]) / std[None, None, None, :]
            tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(chunk, (0, 3, 1, 2))))
            out.append(extractor(tensor).numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


# ----------------------------------------------------------------- train ---- #

def train_one(seed: int, args, train_ds: DetectionDataset, eval_ds: DetectionDataset,
              verbose: bool = True) -> Dict[str, Any]:
    import torch
    t0 = time.time()
    cfg_ignore_radius = getattr(args, "ignore_radius", 1)
    rb = RealBackboneConfig(backbone=args.backbone, pretrained=not args.no_pretrained,
                            finetune=args.finetune, img_size=args.img_size,
                            n_classes=3, hidden=args.hidden, seed=seed)
    cfg: DetectorConfig = rb.detector_config()
    grid, cell, K = cfg.grid, cfg.cell, cfg.n_classes
    model_id = f"realcifar_{args.kind}_s{seed}"

    extractor = build_feature_extractor(rb)
    feats = extract_features(extractor, train_ds.images, rb)          # (N,C2,G,G)
    mean = feats.mean(axis=(0, 2, 3)).astype(np.float32)
    std = (feats.std(axis=(0, 2, 3)) + 1e-6).astype(np.float32)

    obj_t, cls_t, pos_t, box_t = build_targets(train_ds, grid, cell)
    head = make_head(feats.shape[1], args.hidden, K, seed)
    head.mean.copy_(torch.from_numpy(mean))
    head.std.copy_(torch.from_numpy(std))
    params = list(head.parameters()) + ([p for p in extractor.parameters()] if args.finetune else [])
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-5)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(args.pos_weight)))
    ce = torch.nn.CrossEntropyLoss()
    sl1 = torch.nn.SmoothL1Loss()

    fcur = torch.from_numpy(feats)
    o_t = torch.from_numpy(obj_t)
    # Objectness supervision: positives PLUS negatives outside the ignore radius. The
    # radius is a dilation of the positive map (kernel 2r+1), and the positive cells are
    # added back explicitly -- excluding them trains the head on negatives only, which
    # collapses objectness to 0 at every cell including the true one (measured).
    pos_mask = o_t > 0.5
    r = max(int(cfg_ignore_radius), 0)
    if r:
        dilated = torch.nn.functional.max_pool2d(
            pos_mask.float().unsqueeze(1), 2 * r + 1, 1, r).squeeze(1) > 0.5
    else:
        dilated = pos_mask
    keep = (~dilated) | pos_mask
    c_t = torch.from_numpy(cls_t)
    b_t = torch.from_numpy(box_t)
    pos = torch.from_numpy(pos_t)
    # box targets are stored per (image, row): the mask for them is 2-D, unlike obj/cls
    keep_row = torch.from_numpy(pos_t.any(axis=-1))
    n = len(train_ds)
    rng = np.random.default_rng(seed)
    final_loss = float("nan")

    head.train()
    for epoch in range(args.epochs):
        order = rng.permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, args.batch):
            sel = order[i:i + args.batch]
            if args.finetune:
                xb = torch.from_numpy(np.ascontiguousarray(
                    extractor_input(train_ds.images[sel], rb))).float()
                f = extractor(xb) if args.finetune else fcur[sel]
                head.zero_grad()
                o, c, b = head(f)
            else:
                head.zero_grad()
                o, c, b = head(fcur[sel])
            lo = bce(o[..., 0][keep[sel]], o_t[sel][keep[sel]]) * args.obj_weight
            lp = ce(c[pos[sel]], c_t[sel][pos[sel]])
            lb = sl1(b[pos[sel]], b_t[sel][keep_row[sel]])
            loss = lo + lp + lb
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        final_loss = tot / max(nb, 1)
        if verbose and (epoch + 1) % max(args.epochs // 4, 1) == 0:
            print(f"    s{seed} epoch {epoch + 1}/{args.epochs} loss {final_loss:.4f}", flush=True)

    model = RealBackboneDetector(cfg=cfg, rb=rb, extractor=extractor)
    copy_head_to_numpy(model, head, mean, std)
    model.meta = {"model_id": model_id, "seed": seed,
                  "backbone_frozen": not args.finetune}

    clean_q = detection_quality(model, eval_ds.images, eval_ds.boxes, eval_ds.labels)
    if verbose:
        print(f"    s{seed} eval F1 {clean_q.get('f1', 0.0):.3f} "
              f"precision {clean_q.get('precision', 0.0):.3f} "
              f"recall {clean_q.get('recall', 0.0):.3f}")

    fresh = RealBackboneDetector(cfg=cfg, rb=rb, extractor=build_feature_extractor(rb))
    manifest: Dict[str, Any] = {
        "lab_version": LAB_VERSION,
        "model_id": model_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec": {
            "kind": "real_backbone",
            "detector": dict(cfg.__dict__),
            "backbone": {**rb.__dict__, "pretrained_loaded": bool(model.pretrained_loaded)},
            "dataset": dict(train_ds.spec),
            "eval_n": len(eval_ds),
            "train_n": len(train_ds),
            "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
            "hidden": args.hidden, "kind": args.kind,
        },
        "spec_digest": hashlib.sha256(json.dumps(
            {"seed": seed, "rb": rb.__dict__, "data": train_ds.spec, "epochs": args.epochs},
            sort_keys=True, default=str).encode()).hexdigest()[:16],
        "dataset_digests": {"train": train_ds.digest(), "eval": eval_ds.digest()},
        "attack_digest": ATTACK_DIGEST,
        "seeds": {"scene": int(train_ds.spec.get("seed", seed)), "attack": 0,
                  "detector": seed},
        "artifact": {
            "weights_digest": model.digest(),
            "head_digest": model.head_digest(),
            "backbone_digest": fresh.digest(),
            "n_params_head": int(sum(a.size for _, a in model._head_params())),
            "n_params_backbone": int(sum(a.size for _, a in model._backbone_params())),
        },
        "ground_truth": {"kind": args.kind, "trigger": "none", "trigger_loc": "none",
                         "target_class": None, "rate_requested": 0.0, "rate_actual": 0.0,
                         "n_poisoned": 0, "mal_contributor": None},
        "metrics": {
            "clean_quality": clean_q,
            "attack_success_rate": {"asr": 0.0, "criterion": "clean_control",
                                    "note": "clean models only; attack recipes are the "
                                            "overnight REMAINING item"},
            "behaviour_divergence": {"applicable": False,
                                     "note": "no model-side attack in this run"},
            "train_final_loss": float(final_loss),
        },
        "quality_flags": {"backdoor_weak": False, "asr_floor": 0.5, "asr_gate_basis": "control",
                          "asr_gate_value": 1.0, "model_effect_weak": False, "bdr_floor": 0.0,
                          "is_model_attack": False},
        "timing_seconds": round(time.time() - t0, 2),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "machine": platform.machine(), "numpy": np.__version__,
                        "torch": __import__("torch").__version__,
                        "backbone_runtime": rb.runtime,
                        "reproducibility_note": "seeded torch + frozen backbone; same seed "
                                                "yields the same weights_digest"},
    }
    return {"model": model, "manifest": manifest, "eval_ds": eval_ds}


def extractor_input(images: np.ndarray, rb: RealBackboneConfig) -> np.ndarray:
    """(N,H,W,3) [0,1] -> (N,3,H,W) ImageNet-scaled float32."""
    mean = np.asarray([0.485, 0.456, 0.406], np.float32)
    std = np.asarray([0.229, 0.224, 0.225], np.float32)
    x = (images - mean[None, None, None, :]) / std[None, None, None, :]
    return np.transpose(x, (0, 3, 1, 2))


# ------------------------------------------------------------------ main ---- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="runs/real_cifar")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny budget end-to-end (2 models); the gate before the full run")
    ap.add_argument("--n-per-class", type=int, default=60)
    ap.add_argument("--eval-per-class", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--backbone", default="resnet18_stem1")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--finetune", action="store_true", help="also update backbone weights")
    ap.add_argument("--pos-weight", type=float, default=30.0)
    ap.add_argument("--ignore-radius", type=int, default=1,
                    help="cells around a positive excluded from the objectness loss")
    ap.add_argument("--obj-weight", type=float, default=1.0)
    ap.add_argument("--kind", default="clean",
                    help="clean | oga | oda (image-level attack: poison the CIFAR "
                         "training split before training; REMAINING c.2)")
    ap.add_argument("--attack-rate", type=float, default=0.20,
                    help="poisoning fraction for --attack-kind oga/oda")
    ap.add_argument("--attack-trigger-size", type=int, default=10)
    ap.add_argument("--attack-target-class", type=int, default=0)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--onnx", action="store_true", help="export features.onnx + parity gate")
    ap.add_argument("--max-models", type=int, default=None)
    args = ap.parse_args()

    if args.smoke:
        # Measured: with fewer epochs than this the objectness head stays below the 0.30
        # score threshold and F1 reads 0.000 - a plumbing failure, not a quality result,
        # because the head genuinely had not learned yet. The smoke therefore uses the
        # budget that demonstrably learns, and gates on it.
        args.n_per_class = max(args.n_per_class, 300)
        args.eval_per_class = min(args.eval_per_class, 20)
        args.epochs = max(args.epochs, 200)
        args.seeds = args.seeds[:2]
        args.onnx = True
        args.out = args.out if "smoke" in args.out else "runs/real_cifar_smoke"

    seeds = args.seeds[:args.max_models] if args.max_models else args.seeds
    print(f"real backbone: {args.backbone} pretrained={not args.no_pretrained} "
          f"finetune={args.finetune} seeds={seeds}")
    print(f"data: CIFAR-10 classes {DEFAULT_CLASSES} n_per_class={args.n_per_class} "
          f"cache={args.cache_dir}")

    train_ds = load_cifar_subset(n_per_class=args.n_per_class, seed=1000,
                                 cache_dir=args.cache_dir, img_size=args.img_size,
                                 verbose=True)
    # REMAINING (c).2: image-level attacks poison the loader OUTPUT. `inject` works
    # on any DetectionDataset, so the CIFAR split is poisoned in place before the
    # frozen-backbone feature pass; the poisoned split's digest names itself in the
    # manifest (dataset_digests.train), which is what makes the arm auditable.
    attack_spec = None
    if args.kind in ("oga", "oda", "rma", "gma"):
        from cviaf.lab.poison import AttackSpec, inject
        attack_spec = AttackSpec(kind=args.kind, trigger="patch",
                                 trigger_loc="fixed" if args.kind == "oga" else "on_object",
                                 trigger_size=int(args.attack_trigger_size),
                                 target_class=int(args.attack_target_class),
                                 rate=float(args.attack_rate), seed=0)
        train_ds, truth = inject(train_ds, attack_spec)
        print(f"attack: {args.kind} rate={args.attack_rate} "
              f"poisoned={len(truth.poisoned_indices)}/{len(train_ds)} "
              f"digest {train_ds.digest()[:16]}")
    eval_ds = load_cifar_subset(n_per_class=args.eval_per_class, seed=2000,
                                cache_dir=args.cache_dir, img_size=args.img_size)
    print(f"train {len(train_ds)} images  eval {len(eval_ds)} images  "
          f"train digest {train_ds.digest()[:16]}")

    os.makedirs(args.out, exist_ok=True)
    started = time.time()
    entries: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"  training {args.kind} seed {seed} ...", flush=True)
        res = train_one(seed, args, train_ds, eval_ds)
        model, manifest = res["model"], res["manifest"]
        mdir = os.path.join(args.out, manifest["model_id"])
        os.makedirs(mdir, exist_ok=True)
        model.save(os.path.join(mdir, "weights.npz"))
        with open(os.path.join(mdir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1, default=str)
        if args.onnx:
            onnx_path = os.path.join(mdir, "features.onnx")
            model.export_onnx(onnx_path)
            parity = export_parity(model, onnx_path, eval_ds.images[:8])
            record = {"model_id": manifest["model_id"], **parity}
            with open(os.path.join(mdir, "onnx_parity.json"), "w") as fh:
                json.dump(record, fh, indent=1)
            manifest["artifact"]["onnx"] = {"path": "features.onnx", **parity}
            with open(os.path.join(mdir, "manifest.json"), "w") as fh:
                json.dump(manifest, fh, indent=1, default=str)
            print(f"    onnx parity: max|delta|={parity['max_feature_delta']:.3e} "
                  f"agree={parity['channel_argmax_agreement']:.2f} pass={parity['pass']} "
                  f"digest_unchanged={parity['digest_unchanged']}")
        entries.append({"dir": mdir, "manifest": manifest})
        print(f"    wrote {mdir} ({manifest['timing_seconds']}s, "
              f"weights_digest {manifest['artifact']['weights_digest'][:16]})", flush=True)

    gate = None
    if args.smoke:
        rows = []
        for e in entries:
            pq = os.path.join(e["dir"], "onnx_parity.json")
            rows.append({
                "model_id": e["manifest"]["model_id"],
                "f1": e["manifest"]["metrics"]["clean_quality"].get("f1", 0.0),
                "onnx_parity_pass": bool(json.load(open(pq))["pass"]) if os.path.isfile(pq) else None,
            })
        gate = {"models": rows,
                "f1_positive": all((r["f1"] or 0.0) > 0 for r in rows),
                "parity_pass": all(r["onnx_parity_pass"] is True for r in rows)}
        gate["pass"] = bool(gate["f1_positive"] and gate["parity_pass"])
        print("SMOKE GATE: " + ("PASS" if gate["pass"] else "FAIL") +
              f" (f1_positive={gate['f1_positive']}, parity_pass={gate['parity_pass']})")
        for r in rows:
            print(f"   {r['model_id']}: F1={r['f1']:.4f} onnx_parity={r['onnx_parity_pass']}")

    with open(os.path.join(args.out, "registry.jsonl"), "w") as fh:
        for e in entries:
            fh.write(json.dumps(e, default=str) + "\n")
    summary = {
        "lane": "real_backbone_skeleton", "out": args.out, "seeds": seeds,
        "n_models": len(entries), "epochs": args.epochs,
        "n_per_class": args.n_per_class, "eval_per_class": args.eval_per_class,
        "backbone": args.backbone, "pretrained": not args.no_pretrained,
        "finetune": args.finetune, "onnx": bool(args.onnx),
        "cache_dir": args.cache_dir, "train_n": len(train_ds), "eval_n": len(eval_ds),
        "train_digest": train_ds.digest(), "eval_digest": eval_ds.digest(),
        "runtime_seconds": round(time.time() - started, 2),
        "clean_quality": {e["manifest"]["model_id"]: e["manifest"]["metrics"]["clean_quality"]
                          for e in entries},
        "smoke_gate": gate,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    f1s = [v.get("f1", 0.0) for v in summary["clean_quality"].values()]
    print(f"trained {len(entries)} model(s) in {summary['runtime_seconds']}s; "
          f"clean F1 {['%.3f' % f for f in f1s]}; wrote {args.out}")
    return 0 if (gate is None or gate["pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
