#!/usr/bin/env python
"""Attacked variants and a TPR measured at FROZEN thresholds.

The clean-null ledger is the FPR basis and must not move. This script is strictly
additive: it derives attacked variants from clean models that are *already* in that
population, scores them through the same adapter against the same pinned reference, and
then asks a single question of the result -

    at the thresholds the FPR half already chose, how many of these does the rule catch?

No threshold is recomputed, no calibration is redone, and nothing is written back into
the FPR ledger or its shards. If a threshold turns out to be useless on attacks, that is
the finding, and the report says so.

Two things this script refuses to do, both because the alternative is a fabricated number:

  * report a TPR of 0 for a rule whose threshold sits at the corpus ceiling. Such a rule
    cannot fire on *anything*, so its TPR is degenerate, not zero. ``ctc_peak_clean`` and
    ``ctc_q95_clean`` are in exactly that state, and calling their misses "misses" would
    be crediting them with a decision they never made.
  * report a point TPR on a denominator too small to conclude from. Below the floor the
    report gives the exact upper bound and says ``insufficient_denominator``, the same
    convention the FPR side uses.

One more rule, learned by trying to quote a single number:

  * the detection rate is reported **per attack class and per dose, never pooled**. A
    pooled rate over classes is the number that survives exactly one follow-up question
    ("which classes?"), and a pooled rate over doses hides that severity within a class is
    wide. ``evaluate-ladder`` therefore emits no pooled cell at all and names the refusal.

Subcommands
-----------
``build``            derive arms from the merged clean population (weight-space only)
``evaluate``         TPR per rule per attack class at the frozen thresholds (single dose)
``evaluate-ladder``  TPR per rule per (attack class, dose), with the pooling refused
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cviaf.lab.detector import TinyDetector  # noqa: E402

Z95 = 1.959963984540054
DEFAULT_MAGNITUDE = 0.25
DEFAULT_DOSES = (0.1, 0.25, 0.5, 1.0)
ARM_KIND_PREFIX = "cviaf.tpr-arm"
SCHEMA = "cviaf.tpr-at-frozen.v1"
SCHEMA_LADDER = "cviaf.tpr-ladder.v1"

# Three classes, all weight-space, no retraining, no data, no test-time trigger. The first
# two labels match scripts/build_model_attack_arms.py exactly; the third is the
# targeted/insider analogue - one class's bias nudged - which is the closest thing in this
# repository to the problem statement's "backdoor-like behaviour", and is deliberately NOT
# *called* a backdoor: there is no trigger and no source-specific behaviour, so the name
# would be a claim this arm cannot support.
#
# The third element of each tuple is the unit of the dose axis, because the same number
# means different things in the three families: 0.25 is a noise scale in one, a fraction of
# hidden units in another, and absolute logit units in the third. The ladder is comparable
# *within* a family and not across families, and the receipt carries the units so nobody
# has to infer that.
OPS = {
    "weight_tamper": ("unstructured: zero-mean Gaussian noise on the head weights", "head",
                      "standard deviation of the added noise"),
    "substitution": ("structural: the frac least-important hidden units are zeroed", "prune",
                     "fraction of hidden units zeroed"),
    "bias_lift": ("targeted: one class's bias lifted, in ABSOLUTE logit units", "bias",
                  "absolute logit units added to one class's output bias"),
}


def short(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def wilson(hits: int, n: int, z: float = Z95):
    if n <= 0:
        return [None, None]
    p = hits / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [max(0.0, centre - half), min(1.0, centre + half)]


def exact_upper(hits: int, n: int, conf: float = 0.95) -> float:
    """Clopper-Pearson style upper bound at 0 hits: 1 - conf**(1/n)."""
    if n <= 0:
        return float("nan")
    return 1.0 - (1.0 - conf) ** (1.0 / n) if hits == 0 else 1.0


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #

def select_models(shard_path: Path, corpus_root: Path, per_shard: int):
    """Evenly spread picks across one shard, so arms are not all from one box."""
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    models = shard.get("models") or []
    n = len(models)
    if n == 0 or per_shard <= 0:
        return []
    step = max(1, n // per_shard)
    picked = models[::step][:per_shard]
    out = []
    for m in picked:
        model_dir = corpus_root / m["path"]
        if not (model_dir / "manifest.json").is_file():
            raise SystemExit(f"clean model missing on disk: {model_dir}")
        out.append((shard["shard_id"], model_dir, m))
    return out


def measure_divergence(base: dict, tampered) -> dict:
    """Measure f1 before -> after on the model's OWN eval split.

    The validator requires a weight-space arm to carry
    ``metrics.behaviour_divergence.f1_relative_drop`` and the battery reads it, so this is
    not optional bookkeeping - and it must be *measured*, because copying the clean
    model's f1 onto a tampered artifact would be the exact fabricated number this lane
    refuses. Reconstructing the spec from the manifest rebuilds the identical eval split
    the model was trained against (same seeds, same generator), so before/after are the
    same measurement twice.
    """
    # Imported lazily: evaluate imports train, so importing at module scope would be a cycle.
    from cviaf.lab.evaluate import train_spec_from_manifest
    from cviaf.lab.train import BDR_FLOOR, build_splits, detection_quality

    spec = train_spec_from_manifest(base)
    if spec is None:
        raise SystemExit("manifest does not describe a synthetic-scene model; cannot rebuild splits")
    splits = build_splits(spec)
    q_after = detection_quality(tampered, splits.eval_clean.images,
                                splits.eval_clean.boxes, splits.eval_clean.labels)
    f1_clean = float(base["metrics"]["clean_quality"]["f1"])
    f1_after = float(q_after.get("f1", 0.0))
    rel_drop = (f1_clean - f1_after) / max(f1_clean, 1e-9)
    return {
        "quality_after": q_after,
        "f1_before": f1_clean,
        "f1_after": f1_after,
        "f1_relative_drop": float(rel_drop),
        "effect_weak": bool(rel_drop < BDR_FLOOR),
        "bdr_floor": float(BDR_FLOOR),
    }


def build_arm(clean_dir: Path, base: dict, kind: str, magnitude: float, op_name: str,
              out_root: Path, arm_shard_id: str):
    """Write one tampered artifact with the same manifest contract as any corpus model."""
    weights = clean_dir / "weights.npz"
    model = TinyDetector.load(str(weights))
    seed = int(base["seeds"]["detector"])
    if op_name == "head":
        tampered = model.tamper_head(scale=magnitude, seed=seed)
    elif op_name == "prune":
        tampered = model.tamper_prune(frac=magnitude, seed=seed)
    else:
        # Targeted: lift ONE class's output bias by `magnitude` absolute logit units. The
        # units were forced by a measurement (see TinyDetector.tamper_bias): scaling the
        # shift by the bias vector's own standard deviation was a no-op, because that
        # vector is small after training.
        tampered = model.tamper_bias(delta=magnitude, target=0)

    # The no-op check runs FIRST: it is a property of the op, it is cheap, and there is no
    # point rebuilding a dataset to measure the damage of an attack that did not happen.
    weight_digest = tampered.digest()
    if weight_digest == model.digest():
        # A no-op tamper is not an attack, and counting it would inflate the denominator
        # with models that are byte-identical to a clean one.
        raise SystemExit(
            f"tamper {kind}@{magnitude} on {clean_dir.name} produced identical weights; "
            "refusing to record a no-op as an attack"
        )

    # Measured on THIS arm's own tampered artifact - two classes produce two different
    # damage levels, so one measurement cannot stand in for both.
    divergence = measure_divergence(base, tampered)

    model_id = f"{kind}_r{magnitude:g}_s{seed}"
    out_dir = out_root / "models" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = dict(base)
    manifest["model_id"] = model_id
    manifest["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["spec"] = {**base["spec"], "kind": kind, "magnitude": float(magnitude),
                        "derived_from": clean_dir.name, "attack_op": op_name}
    manifest["spec_digest"] = short(json.dumps(manifest["spec"], sort_keys=True, default=str))
    manifest["attack_digest"] = short(f"{kind}:{magnitude:g}:{ARM_KIND_PREFIX}")
    manifest["ground_truth"] = {
        **base["ground_truth"], "kind": kind, "rate_requested": float(magnitude),
        "rate_actual": float(magnitude), "n_poisoned": 0,
        "derived_from": clean_dir.name,
    }
    manifest["artifact"] = {
        "weights_digest": weight_digest,
        "head_digest": tampered.head_digest(),
        "backbone_digest": base["artifact"]["backbone_digest"],
        "n_params_head": int(sum(a.size for _, a in tampered._head_params())),
        "n_params_backbone": int(sum(a.size for _, a in tampered._backbone_params())),
        "clean_weights_digest": base["artifact"]["weights_digest"],
    }
    manifest["metrics"] = {
        # Measured on the tampered artifact, not inherited from the clean one.
        "clean_quality": divergence["quality_after"],
        "attack_success_rate": {
            "asr": 0.0, "asr_strict_iou": 0.0, "n": 0, "applicable": False,
            "kind": kind, "placement": "none",
            "criterion": "model_attack_no_trigger",
            "note": "model attack: there is no test-time trigger, so the behaviour "
                    "divergence below is the ground truth",
        },
        "behaviour_divergence": {
            "f1_before": divergence["f1_before"],
            "f1_after": divergence["f1_after"],
            "f1_relative_drop": divergence["f1_relative_drop"],
            "applicable": True,
            "note": "utility loss on the model's own held-out clean eval split, the "
                    "model-attack analogue of the ASR",
        },
        "train_final_loss": base["metrics"].get("train_final_loss"),
    }
    manifest["quality_flags"] = {**base["quality_flags"], "is_model_attack": True,
                                 "model_effect_weak": divergence["effect_weak"]}
    manifest["timing_seconds"] = 0.0

    tampered.meta = {**tampered.meta, "tamper": kind, "magnitude": float(magnitude),
                     "derived_from": clean_dir.name}
    tampered.save(str(out_dir / "weights.npz"))
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, default=str)

    weights_sha = hashlib.sha256((out_dir / "weights.npz").read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256((out_dir / "manifest.json").read_bytes()).hexdigest()
    return {
        "model_id": model_id,
        "kind": kind,
        "path": f"models/{model_id}",
        "seed": seed,
        "source_clean": clean_dir.name,
        "source_shard": arm_shard_id,
        "manifest_sha256": manifest_sha,
        "weights_sha256": weights_sha,
        "weights_digest": weight_digest,
        "clean_weights_digest": base["artifact"]["weights_digest"],
        "magnitude": float(magnitude),
        "attack_op": op_name,
        "f1_before": divergence["f1_before"],
        "f1_after": divergence["f1_after"],
        "f1_relative_drop": divergence["f1_relative_drop"],
        "effect_weak": divergence["effect_weak"],
    }


def cmd_build(args) -> int:
    out_root = Path(args.out)
    corpora = dict(spec.split("=", 1) for spec in args.corpus)
    shards = dict(spec.split("=", 1) for spec in args.shard)
    if set(corpora) != set(shards):
        raise SystemExit(f"--shard and --corpus ids must match; {sorted(shards)} vs {sorted(corpora)}")

    arms = []
    selected = []
    for sid in sorted(shards):
        picks = select_models(Path(shards[sid]), Path(corpora[sid]), args.per_shard)
        selected.extend(picks)
        print(f"{sid:12s} picked {len(picks)} clean models")
    doses = list(dict.fromkeys(float(d) for d in args.magnitudes))
    for kind in args.kinds:
        if kind not in OPS:
            raise SystemExit(f"unknown arm kind {kind!r}; known: {sorted(OPS)}")
    print(f"selected {len(selected)} clean models -> "
          f"{len(selected) * len(args.kinds) * len(doses)} arms "
          f"({len(args.kinds)} classes x {len(doses)} doses)")

    for shard_id, clean_dir, entry in selected:
        base = json.loads((clean_dir / "manifest.json").read_text(encoding="utf-8"))
        for kind in args.kinds:
            op_name = OPS[kind][1]
            for dose in doses:
                arm = build_arm(clean_dir, base, kind, dose, op_name, out_root, shard_id)
                print(f"  {arm['model_id']:34s} f1 -> {arm['f1_after']:.4f} "
                      f"(rel drop {arm['f1_relative_drop']:.3f}, weak={arm['effect_weak']})")
                arms.append(arm)

    (out_root / "models").mkdir(parents=True, exist_ok=True)
    with open(out_root / "registry.jsonl", "w", encoding="utf-8") as fh:
        for a in arms:
            fh.write(json.dumps(a, sort_keys=True) + "\n")

    shard = {
        "schema": "cviaf.analysis-shard.v1",
        "shard_id": args.shard_id,
        "count": len(arms),
        "seed_min": min(a["seed"] for a in arms),
        "seed_max": max(a["seed"] for a in arms),
        "models": [
            {
                "model_id": a["model_id"],
                "path": a["path"],
                "seed": a["seed"],
                "manifest_sha256": a["manifest_sha256"],
                "weights_sha256": a["weights_sha256"],
            }
            for a in arms
        ],
    }
    plan_dir = out_root / "plan" / "shards"
    plan_dir.mkdir(parents=True, exist_ok=True)
    (plan_dir / f"{args.shard_id}.json").write_text(
        json.dumps(shard, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    by_kind: dict = {}
    for a in arms:
        by_kind.setdefault(a["kind"], 0)
        by_kind[a["kind"]] += 1
    print(f"wrote {len(arms)} arms -> {out_root}  by kind: {by_kind}")
    print(f"shard: {plan_dir / (args.shard_id + '.json')}")
    return 0


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #

def cmd_evaluate(args) -> int:
    report = json.loads(Path(args.frozen_report).read_text(encoding="utf-8"))
    thresholds = {sig: float(r["threshold"]) for sig, r in report["rules"].items()}

    rows = []
    for npz_path in sorted(Path(args.results).glob("results_*.npz")):
        with np.load(npz_path, allow_pickle=False) as z:
            sigs = [k for k in z.files if k.endswith("_clean")]
            for i in range(len(z["model_id"])):
                rows.append({
                    "model_id": str(z["model_id"][i]),
                    "kind": str(z["kind"][i]),
                    "seed": int(z["seed"][i]),
                    "scores": {s: float(z[s][i]) for s in sigs},
                })
    if not rows:
        raise SystemExit(f"no results_*.npz found under {args.results}")

    kinds = sorted({r["kind"] for r in rows})
    problems = []
    if any(k == "clean" for k in kinds):
        problems.append("the arm set contains a model whose kind is 'clean' - it is not an attack")

    rules = {}
    for sig, thr in sorted(thresholds.items()):
        # A threshold at the corpus ceiling cannot be crossed by any score, so the rule
        # cannot fire on an attack either. That is degeneracy, not a miss, and reporting
        # it as "caught 0" would credit a decision the rule never made.
        max_possible = max(r["scores"][sig] for r in rows)
        degenerate = thr >= 1.0 and max_possible <= thr
        entry = {"threshold": thr, "frozen_from": args.frozen_report,
                 "degenerate": bool(degenerate),
                 "status": "degenerate_threshold_at_ceiling" if degenerate else "measured",
                 "per_kind": {}}
        if degenerate:
            entry["note"] = ("the threshold is at the ceiling of the clean corpus, so no "
                             "score can exceed it: the rule cannot fire on an attack either. "
                             "This is not a miss rate.")
        for kind in kinds:
            subset = [r["scores"][sig] for r in rows if r["kind"] == kind]
            hits = sum(1 for v in subset if v > thr)
            n = len(subset)
            ci = wilson(hits, n)
            cell = {
                "n": n, "caught": hits,
                "tpr": (hits / n) if (n and not degenerate) else None,
                "ci95_wilson": ci if (n and not degenerate) else None,
                "exact_upper_bound_95": (None if (n and hits) else
                                         (exact_upper(0, n) if n else None)),
                "conclusion": ("degenerate_rule" if degenerate else
                               ("insufficient_denominator" if n < args.min_positives
                                else "measured")),
                "n_below_floor": n < args.min_positives,
            }
            entry["per_kind"][kind] = cell
        pooled = [r["scores"][sig] for r in rows]
        hits = sum(1 for v in pooled if v > thr)
        entry["pooled"] = {
            "n": len(pooled), "caught": hits,
            "tpr": (hits / len(pooled)) if (pooled and not degenerate) else None,
            "ci95_wilson": wilson(hits, len(pooled)) if (pooled and not degenerate) else None,
            "conclusion": "degenerate_rule" if degenerate else "measured",
        }
        rules[sig] = entry

    out = {
        "schema": SCHEMA,
        "alpha": report.get("alpha", 0.05),
        "frozen_thresholds_from": args.frozen_report,
        "fpr_ledger_untouched": True,
        "n_arms": len(rows),
        "kinds": kinds,
        "min_positives": args.min_positives,
        "rules": rules,
        "problems": problems,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"TPR at frozen thresholds (alpha {out['alpha']}) - {len(rows)} arms {kinds}")
    print(f"{'signal':20s} {'threshold':>10s} {'caught/n':>12s} {'TPR':>8s}   status")
    for sig, e in sorted(rules.items()):
        p = e["pooled"]
        tpr = "  n/a" if p["tpr"] is None else f"{p['tpr']:.4f}"
        print(f"{sig:20s} {e['threshold']:10.6f} {p['caught']:5d}/{p['n']:<6d} {tpr:>8s}   {p['conclusion']}")
    print()
    for kind in kinds:
        for sig, e in sorted(rules.items()):
            c = e["per_kind"][kind]
            tpr = "degenerate" if c["tpr"] is None else f"{c['tpr']:.4f}"
            flag = " [n too small]" if c["n_below_floor"] else ""
            print(f"  {kind:14s} {sig:20s} {tpr:>11s}  n={c['n']}{flag}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
    print(f"\nwrote {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# evaluate-ladder
# --------------------------------------------------------------------------- #

def load_registry(path: Path) -> dict:
    """model_id -> {kind, magnitude}, from the build's registry.jsonl.

    The scored npz carries only (model_id, kind, seed, signals), so the dose has to come
    from the registry the build wrote rather than from parsing the model id: an id is a
    label, and a label is not a record.
    """
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # The registry is also the only place the arm's *measured* behaviour damage is
            # recorded, and that matters more than it looks: a family whose attack does not
            # move the behaviour metric at all is a weight-space change the detector may or
            # may not see, not a harmful model. Carrying it here lets the ladder say how
            # many arms in each cell are behaviourally inert.
            out[row["model_id"]] = {
                "kind": row["kind"],
                "magnitude": float(row["magnitude"]),
                "f1_relative_drop": float(row.get("f1_relative_drop", 0.0)),
                "effect_weak": bool(row.get("effect_weak", False)),
            }
    return out


def cmd_evaluate_ladder(args) -> int:
    """TPR per (attack class, dose) per rule - and no pooled number anywhere."""
    report = json.loads(Path(args.frozen_report).read_text(encoding="utf-8"))
    thresholds = {sig: float(r["threshold"]) for sig, r in report["rules"].items()}
    registry = load_registry(Path(args.registry))

    rows = []
    for npz_path in sorted(Path(args.results).glob("results_*.npz")):
        with np.load(npz_path, allow_pickle=False) as z:
            sigs = [k for k in z.files if k.endswith("_clean")]
            for i in range(len(z["model_id"])):
                mid = str(z["model_id"][i])
                meta = registry.get(mid)
                if meta is None:
                    raise SystemExit(
                        f"{mid} was scored but is not in the build registry; refusing to "
                        "guess which dose it belongs to"
                    )
                rows.append({
                    "model_id": mid,
                    "kind": meta["kind"],
                    "magnitude": meta["magnitude"],
                    "f1_relative_drop": meta["f1_relative_drop"],
                    "effect_weak": meta["effect_weak"],
                    "scores": {s: float(z[s][i]) for s in sigs},
                })
    if not rows:
        raise SystemExit(f"no results_*.npz found under {args.results}")

    kinds = sorted({r["kind"] for r in rows})
    doses = sorted({r["magnitude"] for r in rows})
    problems = []
    if any(k == "clean" for k in kinds):
        problems.append("the arm set contains a model whose kind is 'clean' - it is not an attack")

    # Attacks that could not be scored are counted per (class, dose): *where* they fall
    # decides whether a cell is an estimate or an upper bound.
    unscorable: dict = {}
    for path in args.skipped or []:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in blob.get("skipped", []):
            meta = registry.get(str(row.get("model_id", "")))
            if meta is None:
                continue
            key = f"{meta['kind']}@{meta['magnitude']:g}"
            unscorable[key] = unscorable.get(key, 0) + 1

    rules = {}
    for sig, thr in sorted(thresholds.items()):
        max_possible = max(r["scores"][sig] for r in rows)
        degenerate = thr >= 1.0 and max_possible <= thr
        rules[sig] = {
            "threshold": thr,
            "frozen_from": args.frozen_report,
            "degenerate": bool(degenerate),
            "status": "degenerate_threshold_at_ceiling" if degenerate else "measured",
        }
        if degenerate:
            rules[sig]["note"] = (
                "the threshold is at the ceiling of the clean corpus, so no score can exceed "
                "it: the rule cannot fire on an attack either. This is not a miss rate."
            )

    cells: dict = {}
    for kind in kinds:
        for dose in doses:
            subset = [r for r in rows if r["kind"] == kind and r["magnitude"] == dose]
            n = len(subset)
            drops = sorted(r["f1_relative_drop"] for r in subset)
            entry = {
                "n": n,
                # How many arms in this cell left the repository's behaviour metric
                # unmoved. A cell that is mostly inert is measuring detection of a weight
                # perturbation, not detection of a damaged model.
                "n_behaviour_inert": sum(1 for r in subset if r["effect_weak"]),
                "median_f1_relative_drop": (statistics.median(drops) if drops else None),
                "rules": {},
            }
            for sig, thr in sorted(thresholds.items()):
                deg = rules[sig]["degenerate"]
                hits = sum(1 for r in subset if r["scores"][sig] > thr)
                entry["rules"][sig] = {
                    "caught": hits,
                    "tpr": (hits / n) if (n and not deg) else None,
                    "ci95_wilson": wilson(hits, n) if (n and not deg) else None,
                    "exact_upper_bound_95": (exact_upper(0, n) if (n and hits == 0) else None),
                    "conclusion": ("degenerate_rule" if deg else
                                   ("insufficient_denominator" if n < args.min_positives
                                    else "measured")),
                    "n_below_floor": bool(n < args.min_positives),
                }
            cells.setdefault(kind, {})[f"{dose:g}"] = entry

    out = {
        "schema": SCHEMA_LADDER,
        "alpha": report.get("alpha", 0.05),
        "frozen_thresholds_from": args.frozen_report,
        "fpr_ledger_untouched": True,
        "n_arms": len(rows),
        "kinds": kinds,
        "doses": doses,
        "units_per_kind": {k: OPS[k][2] for k in kinds if k in OPS},
        "behaviour_metric": (
            "f1_relative_drop on each arm's own held-out clean eval split - the repository's "
            "behaviour metric, measured per arm at build time. n_behaviour_inert counts arms "
            "whose attack did not move it; a cell that is mostly inert is a test of whether a "
            "rule notices a weight-space change, not whether it notices a damaged model."
        ),
        "min_positives": args.min_positives,
        "pooled_estimate": None,
        "pooled_refusal": (
            "no pooled detection rate is reported, on either axis: not across attack classes "
            "(which would hide that a rule can catch one class at several times its rate on "
            "another) and not across doses (which would hide that severity within a class is "
            "wide, so the number would be dominated by whichever dose has the most arms). "
            "Read a cell, and compare rules inside that cell."
        ),
        "rules": rules,
        "cells": cells,
        "unscorable": unscorable,
        "problems": problems,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    live = [s for s in sorted(thresholds) if not rules[s]["degenerate"]]
    print(f"TPR at frozen thresholds, per class per dose (alpha {out['alpha']}) - "
          f"{len(rows)} arms, {len(kinds)} classes x {len(doses)} doses")
    print(f"{'class':16s}{'dose':>6s}{'n':>6s}{'inert':>7s}  "
          + "  ".join(f"{s[:18]:>18s}" for s in live))
    for kind in kinds:
        for dose in doses:
            cell = cells[kind][f"{dose:g}"]
            cols = []
            for sig in live:
                c = cell["rules"][sig]
                cols.append("               n/a" if c["tpr"] is None else f"{c['tpr']:18.4f}")
            flagged = "*" if any(cell["rules"][s]["n_below_floor"] for s in live) else " "
            print(f"{kind:16s}{dose:6g}{cell['n']:6d}{cell['n_behaviour_inert']:7d}{flagged} "
                  + "  ".join(cols))
    print("\n* = fewer than min_positives arms survived in that cell. The ratio is shown "
          "beside its count and interval, and the cell's conclusion field says "
          "insufficient_denominator: do not quote it on its own.")
    deg = [s for s in sorted(thresholds) if rules[s]["degenerate"]]
    if deg:
        print(f"\ndegenerate (cannot fire, and not a miss rate): {', '.join(deg)}")
    if unscorable:
        print("\nunscorable arms by cell: " +
              ", ".join(f"{k} x{v}" for k, v in sorted(unscorable.items())))
    print("\nNo pooled rate is reported: read a cell, and compare rules inside it.")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
    print(f"\nwrote {out_path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="derive attacked variants from clean models")
    b.add_argument("--shard", action="append", required=True, help="id=shard.json")
    b.add_argument("--corpus", action="append", required=True, help="id=corpus-root")
    b.add_argument("--out", required=True)
    b.add_argument("--per-shard", type=int, default=9)
    b.add_argument("--magnitudes", nargs="+", type=float, default=list(DEFAULT_DOSES),
                   help="one or more doses; each kind's dose axis has its own units "
                        "(see OPS), so a ladder is comparable within a family only")
    b.add_argument("--kinds", nargs="+", default=sorted(OPS))
    b.add_argument("--shard-id", default="tpr_arms")
    b.set_defaults(func=cmd_build)

    e = sub.add_parser("evaluate", help="TPR at the frozen FPR thresholds")
    e.add_argument("--results", required=True, help="directory of results_*.npz from scoring the arms")
    e.add_argument("--frozen-report", required=True, help="runs/merged_fpr_tpr_report.json")
    e.add_argument("--out", required=True)
    e.add_argument("--min-positives", type=int, default=20)
    e.set_defaults(func=cmd_evaluate)

    lad = sub.add_parser("evaluate-ladder",
                         help="TPR per attack class per dose, with the pooling refused")
    lad.add_argument("--results", required=True, help="directory of results_*.npz")
    lad.add_argument("--registry", required=True,
                     help="the build's registry.jsonl (model_id -> kind, magnitude)")
    lad.add_argument("--frozen-report", required=True, help="runs/merged_fpr_tpr_report.json")
    lad.add_argument("--out", required=True)
    lad.add_argument("--min-positives", type=int, default=20)
    lad.add_argument("--skipped", action="append", default=None,
                     help="skipped_*.json from the scoring run, for the per-cell unscorable count")
    lad.set_defaults(func=cmd_evaluate_ladder)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
