#!/usr/bin/env python
"""Plan (and optionally build) the attack-strength sweep for the missing population.

The problem this addresses
--------------------------
Every rule measured on the fleet is ``fpr_only``: there is no attacked arm in the
clean null, so no rule on that population has a TPR, an expected loss, or a
measurable clause 3.7. The tamper arms that do exist (`runs/real_cifar`, 73 models,
`runs/day1`, 83) sit at a SINGLE dose each, so the lab can say "the detector catches
14.6% of arms" and cannot say *which* arms or *how strong* an attack has to be before
it is caught at all. A single-dose battery cannot distinguish a detector that misses
weak attacks from one that misses all of them.

What this script does
---------------------
Turns a dose ladder into a *plan*: every (kind, dose, seed) cell, the arm count, the
time it costs, and the TPR denominator each cell needs -- sized by the power module
rather than by habit. ``--build`` derives the arms from the existing clean
real-backbone detectors with the same post-hoc weight edits the battery already uses
(``tamper_head`` / ``tamper_prune``), so a ladder costs evaluation time, not training
time.

Three design decisions worth naming:

* **The ladder is geometric, not linear.** Behaviour divergence as a function of the
  tamper parameter is not linear and its knee is unknown; a linear ladder spends
  almost all of its arms past the knee, where every dose looks the same.
* **The pilot is a ladder, not a smaller sweep.** Tens of arms whose job is to (a)
  confirm divergence increases monotonically with dose, (b) measure the per-arm cost
  the full sweep is planned against, and (c) prove the manifest contract holds for
  derived arms. Those are prerequisites for spending thousands of arms, and each one
  is cheap to check and expensive to discover late.
* **The full sweep's size is an output, not an input.** The plan reports the arm
  count implied by the declared target (a 5-point detection / a +/-10-point interval)
  using the same exact binomial machinery as `cviaf/lab/power.py`, so the compute
  bill is traceable to a question.

Run:
    cd .task3 && PYTHONPATH=. <py> scripts/attack_sweep.py \
        --config configs/attack_sweep_pilot.json --out runs/attack_sweep_plan.json
    # add --build to actually write the arms (needs the torch lane)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cviaf.lab.fpr_tpr import wilson_interval
from cviaf.lab.power import BATTERY_TPR, arms_needed

SWEEP_SCHEMA = "cviaf.attack-sweep-plan.v1"
# Measured on this machine by the existing 73-arm battery (derivation + eval, no
# training). Declared here and REPLACED by the pilot's own measurement, because a
# published plan should not silently inherit an estimate.
DEFAULT_SECONDS_PER_ARM = 25.0


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        config = json.load(fh)
    problems = validate_config(config)
    if problems:
        raise ValueError("; ".join(problems))
    return config


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Problems that would make the sweep quietly wrong rather than loud."""
    problems: List[str] = []
    if config.get("schema") != "cviaf.attack-sweep.v1":
        problems.append(f"schema must be 'cviaf.attack-sweep.v1', got "
                        f"{config.get('schema')!r}")
    kinds = config.get("kinds")
    if not isinstance(kinds, dict) or not kinds:
        return problems + ["kinds must be a non-empty object"]
    for kind, spec in kinds.items():
        doses = (spec or {}).get("doses")
        if not isinstance(doses, list) or not doses:
            problems.append(f"{kind}: doses must be a non-empty list")
            continue
        for dose in doses:
            if isinstance(dose, bool) or not isinstance(dose, (int, float)):
                problems.append(f"{kind}: dose {dose!r} is not a number")
            elif not 0.0 < float(dose) <= 1.0:
                # A dose outside (0, 1] is a typo that silently duplicates another
                # arm: 2.5 and 0.25 read the same at a glance in a table.
                problems.append(f"{kind}: dose {dose!r} is outside (0, 1]")
        if len({float(d) for d in doses if isinstance(d, (int, float))}) != len(doses):
            problems.append(f"{kind}: duplicate doses produce duplicate arms")
        if not (spec or {}).get("param"):
            problems.append(f"{kind}: 'param' names which builder argument the dose is")
        if not (spec or {}).get("builder"):
            problems.append(f"{kind}: 'builder' names the function that derives the arm")
    seeds = config.get("seeds") or {}
    for name in ("pilot", "full"):
        if name not in seeds:
            problems.append(f"seeds.{name} is required")
    pilot = seeds.get("pilot")
    if not isinstance(pilot, list) or not pilot:
        problems.append("seeds.pilot must be a non-empty list of detector seeds")
    target = config.get("target") or {}
    for key in ("delta", "half_width"):
        value = target.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < float(value) < 1:
            problems.append(f"target.{key} must be a fraction in (0, 1)")
    return problems


def arms_for_half_width(recall: float, half_width: float, alpha: float = 0.05,
                        n_max: int = 100000) -> Optional[int]:
    """Arms for a Wilson interval of +/-``half_width`` around a recall estimate.

    The measuring question, as opposed to the detecting question that
    ``power.arms_needed`` answers. A ladder cell is usually quoted as "TPR 0.32 at
    dose 0.25", so what it needs is a denominator that pins the cell, not one that
    separates it from the corpus baseline.
    """
    if not 0.0 < recall < 1.0 or not 0.0 < half_width < 0.5:
        raise ValueError("recall in (0,1) and half_width in (0,0.5) are required")
    def width(n: int) -> float:
        lo, hi = wilson_interval(round(recall * n), n)
        return hi - lo
    if width(n_max) / 2.0 > half_width:
        return None
    lo, hi = 1, n_max
    while lo < hi:
        mid = (lo + hi) // 2
        if width(mid) / 2.0 <= half_width:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


def plan_sweep(config: Dict[str, Any], seconds_per_arm: float = DEFAULT_SECONDS_PER_ARM,
               mode: str = "plan") -> Dict[str, Any]:
    """The cells, the arm counts, the cost, and the denominator each cell needs."""
    if mode not in ("plan", "pilot", "full"):
        raise ValueError("mode must be 'plan', 'pilot' or 'full'")
    seeds = config["seeds"]
    pilot_seeds = seeds["pilot"]
    full_seed_spec = seeds["full"]
    target = config["target"]
    delta = float(target["delta"])
    half_width = float(target["half_width"])
    alpha = float(target.get("alpha", 0.05))
    power = float(target.get("power", 0.8))

    cells: List[Dict[str, Any]] = []
    for kind in sorted(config["kinds"]):
        spec = config["kinds"][kind]
        doses = sorted(float(d) for d in spec["doses"])
        for dose in doses:
            cells.append({
                "kind": kind, "dose": dose, "param": spec["param"],
                "builder": spec["builder"],
                "pilot_seeds": len(pilot_seeds),
                "pilot_arms": len(pilot_seeds),
                "sweep_seeds": full_seed_spec if isinstance(full_seed_spec, int) else None,
            })
    n_cells = len(cells)
    pilot_arms = sum(c["pilot_arms"] for c in cells)

    # Detecting: arms per cell to see a `delta` change against the corpus baseline.
    detect_per_cell = arms_needed(BATTERY_TPR, delta, alpha, power, 1, "gain")
    # Measuring: arms per cell to pin that cell's own recall to +/-half_width.
    measure_per_cell = arms_for_half_width(BATTERY_TPR, half_width, alpha)
    denominator = {
        "baseline_recall": round(BATTERY_TPR, 4),
        "baseline_basis": "the existing battery's 7/48; itself only an estimate",
        "delta": delta, "half_width": half_width, "alpha": alpha, "power": power,
        "per_cell_to_detect": detect_per_cell,
        "per_cell_to_measure": measure_per_cell,
        "cells": n_cells,
        "sweep_arms_to_detect": (None if detect_per_cell is None
                                 else detect_per_cell * n_cells),
        "sweep_arms_to_measure": (None if measure_per_cell is None
                                  else measure_per_cell * n_cells),
        "reading": ("'to detect' answers \"is this dose different from the corpus "
                    "baseline\"; 'to measure' answers \"what is this cell's recall\". "
                    "The ladder needs the second at every rung; the first only at the "
                    "knee, so quoting one number for the whole sweep overspends."),
    }
    minutes_per_arm = (seconds_per_arm / 60.0) if seconds_per_arm else 0.0
    plan = {
        "schema": SWEEP_SCHEMA,
        "config": config.get("corpus"),
        "mode": mode,
        "clean_reference": config.get("clean_reference"),
        "cells": cells,
        "n_cells": n_cells,
        "pilot_arms": pilot_arms,
        "pilot_minutes": round(pilot_arms * minutes_per_arm, 1),
        "full_arms": (measure_per_cell * n_cells
                      if (mode == "full" and measure_per_cell) else None),
        "full_minutes": (round(measure_per_cell * n_cells * minutes_per_arm, 1)
                         if (mode == "full" and measure_per_cell) else None),
        "denominator": denominator,
        "seconds_per_arm": seconds_per_arm,
        "cost_basis": ("derived arms are post-hoc weight edits, so cost is "
                       "derivation + evaluation, not training; the pilot measures the "
                       "real seconds-per-arm and this plan must be recomputed with it"),
        "ladder_note": ("the ladder is geometric so it does not spend its arms past "
                        "the knee; the pilot's first job is to find the knee"),
        "no_training": ("no clean detector is trained or retrained by this sweep: the "
                        "arms are derived from detectors that already exist"),
    }
    plan["plan_digest"] = hashlib.sha256(
        json.dumps({"cells": cells, "denominator": denominator},
                   sort_keys=True).encode()).hexdigest()
    return plan


def render(plan: Dict[str, Any]) -> str:
    den = plan["denominator"]
    lines = [f"attack-strength sweep ({plan['mode']})  cells={plan['n_cells']}  "
             f"pilot arms={plan['pilot_arms']} (~{plan['pilot_minutes']:.0f} min at "
             f"{plan['seconds_per_arm']:g}s/arm)",
             f"plan digest {plan['plan_digest'][:16]}", ""]
    if plan.get("full_arms") is not None:
        lines.append(f"full sweep: {plan['full_arms']} arms "
                     f"(~{plan['full_minutes']:.0f} min at the declared per-arm cost)")
    lines.append(f"{'kind':16s} {'dose':>6s} {'param':>7s} {'pilot arms':>10s}")
    for cell in plan["cells"]:
        lines.append(f"{cell['kind']:16s} {cell['dose']:6.3f} {cell['param']:>7s} "
                     f"{cell['pilot_arms']:>10d}")
    lines += ["", f"TPR denominator, sized from the power table rather than by habit:",
              f"  baseline recall {den['baseline_recall']} "
              f"({den['baseline_basis']})",
              f"  to DETECT a {den['delta']:g} change vs that baseline: "
              f"{den['per_cell_to_detect']} arms per cell -> "
              f"{den['sweep_arms_to_detect']} for {den['cells']} cells",
              f"  to MEASURE each cell to +/-{den['half_width']:g}: "
              f"{den['per_cell_to_measure']} arms per cell -> "
              f"{den['sweep_arms_to_measure']} for {den['cells']} cells",
              f"  {den['reading']}",
              "", f"  {plan['cost_basis']}",
              f"  {plan['ladder_note']}",
              f"  {plan['no_training']}"]
    return "\n".join(lines)


def build_arms(config: Dict[str, Any], plan: Dict[str, Any], log=print) -> Dict[str, Any]:
    """Derive the pilot arms. Imports torch lazily: the plan path needs no torch."""
    import glob

    import numpy as np

    from cviaf.lab.cifar import load_cifar_subset
    from cviaf.lab.real_backbone import RealBackboneDetector
    from cviaf.lab.train import detection_quality

    corpus = config["corpus"]
    os.makedirs(corpus, exist_ok=True)
    eval_ds = load_cifar_subset(n_per_class=20, seed=2000, cache_dir="data/cifar10",
                                img_size=64)
    built: List[str] = []
    for cell in plan["cells"]:
        for seed in config["seeds"]["pilot"]:
            clean = glob.glob(f"runs/real_cifar/realcifar_clean_s{seed}/weights.npz")
            if not clean:
                log(f"  no clean checkpoint for seed {seed}; skipping cell "
                    f"{cell['kind']}@{cell['dose']}")
                continue
            model = RealBackboneDetector.load(clean[0])
            if cell["builder"].endswith("tamper_head"):
                arm = model.tamper_head(scale=cell["dose"], seed=seed)
            else:
                arm = model.tamper_prune(frac=cell["dose"], seed=seed)
            model_id = f"realcifar_{cell['kind']}_d{cell['dose']:g}_s{seed}"
            out = os.path.join(corpus, model_id)
            os.makedirs(out, exist_ok=True)
            arm.save(os.path.join(out, "weights.npz"))
            base = json.load(open(os.path.join(
                os.path.dirname(clean[0]), "manifest.json"), encoding="utf-8"))
            manifest = dict(base)
            manifest.update({
                "model_id": model_id,
                "spec_digest": hashlib.sha256(model_id.encode()).hexdigest()[:16],
                "ground_truth": {"kind": cell["kind"],
                                 "dose": cell["dose"],
                                 "dose_param": cell["param"],
                                 "derived_from": base["model_id"]},
            })
            with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
                json.dump(manifest, fh)
            built.append(model_id)
            log(f"  built {model_id}")
    return {"corpus": corpus, "n_built": len(built), "model_ids": built}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="plan the attack-strength dose ladder")
    ap.add_argument("--config", default="configs/attack_sweep_pilot.json")
    ap.add_argument("--out", default=None, help="write the plan JSON here")
    ap.add_argument("--mode", choices=("plan", "pilot", "full"), default="plan")
    ap.add_argument("--seconds-per-arm", type=float, default=DEFAULT_SECONDS_PER_ARM)
    ap.add_argument("--build", action="store_true",
                    help="actually derive the pilot arms (torch lane; expensive)")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    plan = plan_sweep(config, args.seconds_per_arm, args.mode)
    print(render(plan))
    if args.build:
        built = build_arms(config, plan)
        plan["built"] = built
        print(f"\nbuilt {built['n_built']} arm(s) into {built['corpus']}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(plan, fh, indent=1, default=str)
        print(f"plan written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
