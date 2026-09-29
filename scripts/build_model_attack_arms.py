#!/usr/bin/env python
"""Build the model-attack arms of the real-backbone corpus (Task 3 REMAINING c.1).

The head is plain numpy, so `tamper_head` (unstructured noise) and `tamper_prune`
(structural unit removal) operate on a loaded real-backbone detector directly -- no
retraining. Each arm is written into the SAME corpus as a sibling model directory with
the same manifest contract, so `load_registry`, `compare` and the batteries consume it
unchanged. The clean model of the same seed is the honest reference; the manifest
records the measured behaviour divergence (f1 before -> after) exactly as
`train_model` does for synthetic model attacks.

Run with the torch interpreter:
    cd .task3 && PYTHONPATH=. <py> scripts/build_model_attack_arms.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cviaf.lab.cifar import load_cifar_subset
from cviaf.lab.real_backbone import RealBackboneDetector
from cviaf.lab.train import BDR_FLOOR, detection_quality

CORPUS = "runs/real_cifar"
TAMPER_SCALE = 0.25     # weight_tamper: head noise scale (post-prune-fix semantics)
PRUNE_FRAC = 0.25       # substitution: fraction of hidden units zeroed
EVAL_PER_CLASS = 20     # matches the training run's eval split (seed 2000)


def short(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def main() -> int:
    eval_ds = load_cifar_subset(n_per_class=EVAL_PER_CLASS, seed=2000,
                                cache_dir="data/cifar10", img_size=64)
    print(f"eval split: {len(eval_ds)} images  digest {eval_ds.digest()[:16]}")

    built = 0
    for w in sorted(glob.glob(f"{CORPUS}/realcifar_clean_s*/weights.npz")):
        clean_dir = os.path.dirname(w)
        clean_id = os.path.basename(clean_dir)
        base = json.load(open(os.path.join(clean_dir, "manifest.json")))
        seed = int(base["spec"]["detector"]["seed"])
        model = RealBackboneDetector.load(w)
        f1_clean = float(base["metrics"]["clean_quality"]["f1"])

        for kind, tampered, magnitude in (
            ("weight_tamper", model.tamper_head(scale=TAMPER_SCALE, seed=seed), TAMPER_SCALE),
            ("substitution", model.tamper_prune(frac=PRUNE_FRAC, seed=seed), PRUNE_FRAC),
        ):
            model_id = f"realcifar_{kind}_r{magnitude:g}_s{seed}"
            out_dir = os.path.join(CORPUS, model_id)
            if os.path.isfile(os.path.join(out_dir, "manifest.json")):
                print(f"  {model_id}: exists, skip")
                continue

            t0 = time.time()
            q_after = detection_quality(tampered, eval_ds.images, eval_ds.boxes,
                                        eval_ds.labels)
            f1_after = float(q_after.get("f1", 0.0))
            rel_drop = (f1_clean - f1_after) / max(f1_clean, 1e-9)

            manifest = dict(base)                      # inherit spec/dataset digests
            manifest["model_id"] = model_id
            manifest["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            manifest["spec"] = {**base["spec"], "kind": kind,
                                "magnitude": float(magnitude),
                                "derived_from": clean_id}
            manifest["spec_digest"] = short(json.dumps(manifest["spec"],
                                                       sort_keys=True, default=str))
            manifest["attack_digest"] = short(f"{kind}:{magnitude}")
            manifest["ground_truth"] = {**base["ground_truth"], "kind": kind,
                                        "rate_requested": float(magnitude),
                                        "mal_contributor": "vendor_x"}
            manifest["artifact"] = {
                "weights_digest": tampered.digest(),
                "head_digest": tampered.head_digest(),
                "backbone_digest": base["artifact"]["backbone_digest"],
                "n_params_head": int(sum(a.size for _, a in tampered._head_params())),
                "n_params_backbone": int(sum(a.size for _, a in tampered._backbone_params())),
                "clean_weights_digest": base["artifact"]["weights_digest"],
            }
            manifest["metrics"] = {
                "clean_quality": q_after,
                "attack_success_rate": {"asr": 0.0, "applicable": False,
                                        "criterion": "model_attack_no_trigger",
                                        "note": "model attack: no test-time trigger; the "
                                                "behaviour divergence is the ground truth"},
                "behaviour_divergence": {
                    "f1_before": f1_clean, "f1_after": f1_after,
                    "f1_relative_drop": float(rel_drop),
                    "applicable": True,
                    "note": "utility loss on the clean held-out CIFAR split, the "
                            "model-attack analogue of the ASR",
                },
            }
            manifest["quality_flags"] = {**base["quality_flags"],
                                         "is_model_attack": True,
                                         "model_effect_weak": bool(rel_drop < BDR_FLOOR)}
            manifest["timing_seconds"] = round(time.time() - t0, 2)

            os.makedirs(out_dir, exist_ok=True)
            # tampered is a TinyDetector-shaped `copy()` of the real-backbone model
            # (base-class copy), so rebuild it as a real-backbone artifact: same
            # head arrays, same backbone state. This is the narrowest way to keep
            # the corpus's `_meta.kind == real_backbone` dispatch contract intact.
            tampered_rb = RealBackboneDetector.__new__(RealBackboneDetector)
            tampered_rb.__dict__.update(model.__dict__)
            for name, arr in model._head_params():
                setattr(tampered_rb, name, getattr(tampered, name))
            tampered_rb.cfg = model.cfg
            tampered_rb.rb = model.rb
            tampered_rb.trained = True
            tampered_rb.meta = {**model.meta, "tamper": kind, "magnitude": magnitude,
                                "derived_from": clean_id}
            tampered_rb._extractor = model._extractor
            tampered_rb._onnx = None
            tampered_rb._bb_arrays = dict(model._bb_arrays)
            tampered_rb.save(os.path.join(out_dir, "weights.npz"))
            with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
                json.dump(manifest, fh, indent=1, default=str)
            with open(os.path.join(CORPUS, "registry.jsonl"), "a") as fh:
                fh.write(json.dumps({"dir": out_dir, "manifest": manifest},
                                    default=str) + "\n")
            built += 1
            print(f"  {model_id}: F1 {f1_clean:.4f} -> {f1_after:.4f} "
                  f"(rel drop {rel_drop:.3f}, weak={rel_drop < BDR_FLOOR}) "
                  f"digest {manifest['artifact']['weights_digest'][:16]}")

    print(f"built {built} model-attack arm(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
