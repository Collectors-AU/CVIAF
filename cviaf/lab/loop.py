"""
The all-day loop: keep the corpus growing, one fresh block of seeds at a time.

Why this exists as its own module rather than ``nohup ... corpus &``
--------------------------------------------------------------------
A single corpus run is not an all-day job. Measured on the M3 Air, one model
costs ~7.4 s, so the 32-model day-1 matrix finishes in about four minutes. What
you actually want overnight is *more independent samples*, because every
statistic in the evaluation is a distribution over seeds and the answer to
"how stable is that AUROC?" is the per-seed spread.

So this loop does the obvious thing: it looks at what is already on disk, picks a
block of seeds that **still need training** (finishing any seed an earlier
interrupt left half-done before opening new ones), trains them, re-runs the
evaluation, records a heartbeat, and sleeps before doing it again. Because
``run_corpus`` is idempotent on ``spec_digest``, killing this at any moment is
safe: a re-launch resumes instead of repeating.

Failure policy, deliberately chosen
-----------------------------------
* A failing **model** is already isolated inside ``run_corpus``.
* A failing **cycle** is caught here and the loop continues.
* ``max_consecutive_failures`` stops the loop when something is broken for real
  (bad config, no disk), so an overnight run cannot spin uselessly for hours.

Heartbeat
---------
``<out>/loop_status.json`` is rewritten after every cycle with cycle count,
seeds used, models trained, elapsed time and the timestamp of the last cycle.
That file is how you tell a working background job from a dead one without
guessing:

    cat runs/day1/loop_status.json
"""

from __future__ import annotations

import json
import os
import re
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from cviaf.lab.corpus import load_plan, run_corpus

_SEED_SUFFIX = re.compile(r"_s(\d+)$")


def seed_model_counts(out_dir: str) -> Dict[int, int]:
    """How many trained models exist per seed, keyed by detector seed.

    Prefers the recorded ``spec.detector.seed`` (authoritative) and falls back to
    the ``_s<N>`` model-id suffix, so a corpus interrupted mid-write is still read
    correctly.
    """
    counts: Dict[int, int] = {}
    if not os.path.isdir(out_dir):
        return counts
    for name in sorted(os.listdir(out_dir)):
        if not os.path.isfile(os.path.join(out_dir, name, "manifest.json")):
            continue
        seed: Optional[int] = None
        try:
            with open(os.path.join(out_dir, name, "manifest.json")) as fh:
                man = json.load(fh)
            seed = man.get("spec", {}).get("detector", {}).get("seed")
        except Exception:
            seed = None
        if seed is None:
            m = _SEED_SUFFIX.search(name)
            seed = int(m.group(1)) if m else None
        if seed is not None:
            counts[int(seed)] = counts.get(int(seed), 0) + 1
    return counts


def used_seeds(out_dir: str) -> set:
    """Seeds that have at least one manifest in ``out_dir``."""
    return set(seed_model_counts(out_dir))


def count_models(out_dir: str) -> int:
    """Trained models on disk (seeds x attacks is the model count, not the seed count)."""
    return sum(seed_model_counts(out_dir).values())


def next_seed_block(out_dir: str, k: int, per_seed: int = 1) -> List[int]:
    """The next ``k`` seeds that still need training.

    A seed is finished when it has ``per_seed`` models -- normally the number of
    attacks in the plan. That definition matters: a corpus interrupted halfway
    through a seed leaves that seed *incomplete*, and the loop must come back and
    finish it rather than sail past it, which would silently leave a hole in the
    matrix. So the rule is two-phase:

      1. **Repair** any incomplete seed already on disk.
      2. Once nothing is incomplete, **extend** past the highest seed seen.
    """
    counts = seed_model_counts(out_dir)
    block: List[int] = []

    for s in sorted(counts):
        if len(block) >= k:
            return block
        if counts[s] < per_seed:
            block.append(s)

    s = (max(counts) + 1) if counts else 1
    while len(block) < k:
        if counts.get(s, 0) < per_seed:
            block.append(s)
        s += 1
    return block


def run_loop(
    plan: Dict[str, Any],
    seeds_per_cycle: int = 4,
    cycles: int = 0,
    interval_minutes: float = 20.0,
    budget_minutes: Optional[float] = None,
    max_hours: Optional[float] = 8.0,
    max_consecutive_failures: int = 3,
    do_eval: bool = True,
    alpha: float = 0.05,
    log: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
) -> Dict[str, Any]:
    """Grow the corpus in cycles until ``cycles`` or ``max_hours`` is reached.

    ``cycles=0`` means unlimited (bounded only by ``max_hours``). ``log``,
    ``sleep`` and ``clock`` are injected so this is testable without waiting.
    """
    out = plan["out"]
    os.makedirs(out, exist_ok=True)
    status_path = os.path.join(out, "loop_status.json")
    t_start = clock()

    totals = {"trained": 0, "skipped": 0, "failed": 0}
    history: List[Dict[str, Any]] = []
    cycles_done = 0
    consecutive_failures = 0

    log(f"[loop] '{plan['name']}' -> {out}")
    log(f"[loop] {seeds_per_cycle} seeds/cycle, interval {interval_minutes:.0f} min, "
        f"budget {budget_minutes or 'unlimited'}/cycle, "
        f"max_hours {max_hours or 'unlimited'}")
    counts = seed_model_counts(out)
    log(f"[loop] on disk: {count_models(out)} models over {len(counts)} seeds; "
        f"a seed is complete at {len(plan.get('attacks', [])) or 1} models")
    if counts:
        log(f"[loop] seed -> models: { {s: counts[s] for s in sorted(counts)} }")

    while cycles == 0 or cycles_done < cycles:
        elapsed_h = (clock() - t_start) / 3600.0
        if max_hours is not None and elapsed_h >= max_hours:
            log(f"[loop] max_hours {max_hours} reached ({elapsed_h:.2f}h); stopping cleanly")
            break

        seeds = next_seed_block(out, seeds_per_cycle,
                                per_seed=len(plan.get("attacks", [])) or 1)
        cycle_plan = dict(plan)
        cycle_plan["seeds"] = seeds
        cycle_plan["name"] = f"{plan['name']}-s{seeds[0]}"

        log("")
        log(f"[loop] === cycle {cycles_done + 1} === seeds {seeds} "
            f"(elapsed {elapsed_h:.2f}h)")

        cycle_start = clock()
        try:
            summary = run_corpus(cycle_plan, resume=True,
                                 budget_minutes=budget_minutes, progress=log)
            for k in totals:
                totals[k] += int(summary.get(k, 0))
            consecutive_failures = 0
            ok = True
        except Exception:
            log("[loop] cycle aborted:\n" + traceback.format_exc())
            summary = {"trained": 0, "skipped": 0, "failed": -1}
            consecutive_failures += 1
            ok = False

        if do_eval:
            try:
                from cviaf.lab.evaluate import evaluate_corpus
                res = evaluate_corpus(out, alpha=alpha)
                with open(os.path.join(out, "eval.json"), "w") as fh:
                    json.dump(res, fh, indent=1)
                log(f"[loop] evaluation refreshed over {res['n_models']} models")
            except Exception:
                log("[loop] evaluation failed (non-fatal):\n" + traceback.format_exc())

        cycles_done += 1
        cycle_seconds = clock() - cycle_start
        record = {
            "cycle": cycles_done, "seeds": seeds, "ok": ok,
            "trained": summary.get("trained", 0),
            "skipped": summary.get("skipped", 0),
            "failed": summary.get("failed", 0),
            "seconds": round(cycle_seconds, 1),
        }
        history.append(record)

        status = {
            "corpus": plan["name"],
            "out": out,
            "cycles_done": cycles_done,
            "seeds_used": sorted(used_seeds(out)),
            "models_total": count_models(out),
            "totals": totals,
            "elapsed_hours": round((clock() - t_start) / 3600.0, 3),
            "last_cycle": record,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(status_path, "w") as fh:
            json.dump(status, fh, indent=1)
        log(f"[loop] heartbeat -> {status_path}")

        if consecutive_failures >= max_consecutive_failures:
            log(f"[loop] {consecutive_failures} consecutive failed cycles; stopping "
                f"so a broken config does not burn the night")
            break

        more = (cycles == 0 or cycles_done < cycles)
        if more:
            log(f"[loop] cooling down {interval_minutes:.0f} min "
                f"(this is also the thermal rest the fanless Air needs)")
            sleep(interval_minutes * 60.0)

    result = {
        "corpus": plan["name"], "out": out, "cycles_done": cycles_done,
        "totals": totals, "history": history,
        "elapsed_hours": round((clock() - t_start) / 3600.0, 3),
        "models_total": count_models(out),
    }
    log("")
    log(f"[loop] finished: {cycles_done} cycles, trained={totals['trained']} "
        f"skipped={totals['skipped']} failed={totals['failed']} "
        f"in {result['elapsed_hours']:.2f}h")
    log(f"[loop] next: .venv/bin/python -m cviaf.lab eval --corpus {out}")
    return result


def loop_from_plan(path: str, **kwargs) -> Dict[str, Any]:
    """Convenience wrapper for the CLI: load a plan file and run the loop."""
    return run_loop(load_plan(path), **kwargs)
