"""
Corpus runner: train a matrix of clean + attacked models, resumably.

This is the "leave it running all day" component, and it is designed around that
use specifically:

  * **Atomic per model.** Each model is one directory with ``weights.npz`` and
    ``manifest.json``. Interrupt at any moment and nothing is half-written that
    matters; a partial model is simply retrained.
  * **Resumable and idempotent.** A model is skipped only when its manifest exists
    AND its ``spec_digest`` matches the spec that would be generated now. Change a
    config value and exactly the affected models retrain.
  * **Budget-aware.** ``--budget-minutes`` stops cleanly between models, so it can
    be run for a fixed slice of the day without a watchdog.
  * **Ground truth kept.** Every entry carries its ASR, so the evaluation step can
    exclude models where the backdoor never implanted instead of quietly scoring
    detectors against clean models and calling the result a success.

Typical use on an M3 Air (measured: ~4-15 s per model):

    nohup .venv/bin/python -m cviaf.lab corpus --plan configs/corpus_mvp.json \
        --budget-minutes 480 > logs/corpus.log 2>&1 &

Then, in the morning:

    .venv/bin/python -m cviaf.lab eval --corpus runs/mvp
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

from cviaf.lab.detector import DetectorConfig
from cviaf.lab.poison import MODEL_ATTACK_KINDS, AttackSpec
from cviaf.lab.synth import SceneSpec
from cviaf.lab.train import ASR_FLOOR, TrainSpec, build_splits, train_model


def default_plan() -> Dict[str, Any]:
    """A small, fast, *useful* corpus: one seed, eight attacks, one clean arm."""
    return {
        "name": "mvp",
        "out": "runs/mvp",
        "n_train": 240, "n_eval": 80, "n_cal": 120,
        "seeds": [5],
        "detector": asdict(DetectorConfig(epochs=600, lr=0.02, pos_weight=12,
                                          batch=2048, seed=5)),
        "scene": asdict(SceneSpec(seed=7)),
        "attacks": [
            {"kind": "clean", "trigger": "none", "trigger_loc": "fixed", "rate": 0.0},
            {"kind": "oga", "trigger": "patch", "trigger_loc": "fixed",
             "trigger_size": 10, "target_class": 0, "rate": 0.2},
            {"kind": "oda", "trigger": "patch", "trigger_loc": "on_object",
             "trigger_size": 10, "target_class": 0, "rate": 0.4},
            {"kind": "rma", "trigger": "patch", "trigger_loc": "on_object",
             "trigger_size": 10, "target_class": 0, "rate": 0.4},
            {"kind": "label_flip", "trigger": "none", "trigger_loc": "fixed", "rate": 0.1},
            {"kind": "dup_flood", "trigger": "none", "trigger_loc": "fixed", "rate": 0.2},
            {"kind": "ood_insert", "trigger": "none", "trigger_loc": "fixed", "rate": 0.1},
            {"kind": "gma", "trigger": "patch", "trigger_loc": "on_object",
             "trigger_size": 10, "target_class": 0, "rate": 0.4},
        ],
        "attack_seed": 11,
    }


def load_plan(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        plan = json.load(fh)
    base = default_plan()
    base.update(plan)
    return base


def save_plan(plan: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(plan, fh, indent=1)
    return path


def build_specs(plan: Dict[str, Any]) -> List[TrainSpec]:
    scene = SceneSpec(**plan["scene"])
    det = DetectorConfig(**plan["detector"])
    specs: List[TrainSpec] = []
    for seed in plan["seeds"]:
        for i, atk in enumerate(plan["attacks"]):
            a = AttackSpec(seed=int(plan.get("attack_seed", 11)), **atk)
            # Model attacks reuse `rate` as a magnitude (tamper scale / permutation
            # fraction), so two entries that differ only in magnitude would otherwise
            # collide on one model_id and silently overwrite each other -- which is how
            # you end up with a "magnitude sweep" of exactly one magnitude.
            if a.kind == "weight_tamper":
                # The mechanism is part of the identity, not just the magnitude. Two
                # tampers of the same size by different mechanisms are different attacks
                # with different signatures, and without this they collide on one
                # model_id and one is silently overwritten -- the same failure the
                # magnitude suffix was added to prevent. `substitution` has no mechanism
                # (it is a retrain, not a weight edit), so it is left off its id.
                model_id = (f"{a.kind}_{a.mechanism}_{a.trigger}_{a.trigger_loc}"
                            f"_r{a.rate:g}_s{seed}")
            elif a.kind in MODEL_ATTACK_KINDS:
                model_id = f"{a.kind}_{a.trigger}_{a.trigger_loc}_r{a.rate:g}_s{seed}"
            else:
                model_id = f"{a.kind}_{a.trigger}_{a.trigger_loc}_s{seed}"
            specs.append(TrainSpec(
                model_id=model_id,
                scene=SceneSpec(**{**plan["scene"], "seed": scene.seed + seed}),
                attack=a,
                detector=DetectorConfig(**{**plan["detector"], "seed": seed}),
                n_train=plan["n_train"], n_eval=plan["n_eval"], n_cal=plan["n_cal"],
                contributors=tuple(plan.get("contributors",
                                            ["lab_alpha", "lab_beta", "vendor_x"])),
                contributor_mode=plan.get("contributor_mode", "round_robin"),
            ))
    return specs


def retest_corpus(
    corpus_dir: str,
    progress: Optional[Callable[[str], None]] = print,
    write: bool = True,
) -> Dict[str, Any]:
    """Recompute the measured metrics of an existing corpus, without retraining.

    Why this exists: the attack-success criterion is part of the *measurement*, not
    part of the training, so when the criterion is found to be measuring the wrong
    thing (see ``train.attack_success_rate``) the honest response is to re-measure
    the corpus, not to retrain 64 models and pretend they are new. Re-measuring also
    keeps the comparison clean: same weights, same splits, one changed definition.

    Each manifest keeps the values it had (``retest.before``) next to the new ones,
    so a change in a published number can always be traced to a change in the metric
    rather than to a change in the data.
    """
    # Imported lazily: evaluate imports train, so keeping this inside the function
    # avoids any chance of an import cycle as the lab grows.
    from cviaf.lab.evaluate import load_registry, train_spec_from_manifest
    from cviaf.lab.train import ModelArtifact, attack_success_rate, detection_quality

    log = progress or (lambda s: None)
    entries = load_registry(corpus_dir)
    if not entries:
        raise RuntimeError(f"no models found under {corpus_dir}/")

    # Null-subtract against a clean corpus member. The criterion was already re-measured
    # once, but the *null* is the part that makes it mean anything: on a clean model the
    # same recipes measure 0.99 / 0.91 / 0.17, so the raw rate is mostly a measurement of
    # the trigger patch's own ink. Picking the reference by sorted model_id keeps the
    # choice reproducible instead of "whichever clean model was loaded first".
    null_model = None
    for e in sorted(entries, key=lambda x: x["manifest"]["model_id"]):
        if e["manifest"]["ground_truth"]["kind"] == "clean":
            null_model = ModelArtifact.load(e["dir"]).model
            log(f"  null reference: {e['manifest']['model_id']}")
            break

    before_weak = after_weak = 0
    per_kind: Dict[str, Dict[str, Any]] = {}
    updated: List[Dict[str, Any]] = []

    for e in entries:
        m = e["manifest"]
        kind = m["ground_truth"]["kind"]
        spec = train_spec_from_manifest(m)
        splits = build_splits(spec)
        art = ModelArtifact.load(e["dir"])

        clean_q = detection_quality(art.model, splits.eval_clean.images,
                                    splits.eval_clean.boxes, splits.eval_clean.labels)
        ref = None if kind == "clean" else null_model
        asr = attack_success_rate(art.model, splits.eval_clean, spec.attack,
                                  seed=spec.detector.seed, null_model=ref)
        basis = "asr_net" if "asr_net" in asr else "asr"
        gate_value = float(asr[basis])
        weak = bool(asr["applicable"] and gate_value < ASR_FLOOR)

        old_asr = float(m["metrics"]["attack_success_rate"]["asr"])
        old_weak = bool(m["quality_flags"]["backdoor_weak"])
        before_weak += int(old_weak)
        after_weak += int(weak)

        m["metrics"]["clean_quality"] = clean_q
        m["metrics"]["attack_success_rate"] = asr
        m["quality_flags"]["backdoor_weak"] = weak
        m["quality_flags"]["asr_metric"] = asr["criterion"]
        m["quality_flags"]["asr_gate_basis"] = basis
        m["quality_flags"]["asr_gate_value"] = gate_value
        m.setdefault("retest", {})
        old_basis = m["quality_flags"].get("asr_gate_basis_previous", "asr")
        m["retest"][asr["criterion"]] = {
            "gate_basis": basis, "gate_before": old_basis,
            "asr_before": old_asr, "asr_after": float(asr["asr"]),
            "asr_net": asr.get("asr_net"), "asr_null": asr.get("asr_null"),
            "asr_strict_iou": float(asr["asr_strict_iou"]),
            "weak_before": old_weak, "weak_after": weak,
            "note": ("re-measured under the paired criterion, null-subtracted against a "
                     "clean corpus member; the raw and strict-IoU values are retained so "
                     "earlier published numbers stay comparable"),
        }

        agg = per_kind.setdefault(kind, {"n": 0, "usable": 0, "asr_sum": 0.0,
                                         "net_sum": 0.0, "null_sum": 0.0})
        agg["n"] += 1
        agg["usable"] += int(not weak)
        agg["asr_sum"] += float(asr["asr"])
        agg["net_sum"] += float(asr.get("asr_net", asr["asr"]))
        agg["null_sum"] += float(asr.get("asr_null", 0.0))

        if write:
            with open(os.path.join(e["dir"], "manifest.json"), "w") as fh:
                json.dump(m, fh, indent=1, default=str)
        updated.append({"dir": e["dir"], "manifest": m})
        if old_weak != weak or abs(old_asr - float(asr["asr"])) > 1e-6:
            net = asr.get("asr_net")
            tag = ("  [now usable]" if (old_weak and not weak)
                   else "  [now gated]" if (weak and not old_weak) else "")
            log(f"  {m['model_id']:34s} ASR {old_asr:.3f} -> {asr['asr']:.3f}"
                + (f" (net {net:.3f})" if net is not None else "") + tag)

    # The registry mirrors the manifests, so it is rebuilt rather than patched: a
    # half-updated registry is worse than none, because load_registry prefers it.
    reg_path = os.path.join(corpus_dir, "registry.jsonl")
    if write:
        with open(reg_path, "w") as fh:
            for e in sorted(updated, key=lambda x: x["manifest"]["model_id"]):
                fh.write(json.dumps(e, default=str) + "\n")

    summary = {
        "corpus": corpus_dir, "n_models": len(updated),
        "gated_before": before_weak, "gated_after": after_weak,
        "usable_before": len(updated) - before_weak,
        "usable_after": len(updated) - after_weak,
        "per_kind": {k: {"n": v["n"], "usable": v["usable"],
                         "mean_asr": round(v["asr_sum"] / v["n"], 4),
                         "mean_asr_net": round(v["net_sum"] / v["n"], 4),
                         "mean_asr_null": round(v["null_sum"] / v["n"], 4)}
                     for k, v in sorted(per_kind.items())},
        "registry": reg_path, "written": write,
    }
    log(f"retest: {len(updated)} models, gated {before_weak} -> {after_weak} "
        f"(usable {summary['usable_before']} -> {summary['usable_after']})")
    return summary


def run_corpus(
    plan: Dict[str, Any],
    resume: bool = True,
    budget_minutes: Optional[float] = None,
    progress: Optional[Callable[[str], None]] = print,
    max_models: Optional[int] = None,
) -> Dict[str, Any]:
    out = plan["out"]
    os.makedirs(out, exist_ok=True)
    reg_path = os.path.join(out, "registry.jsonl")
    log: Callable[[str], None] = progress or (lambda s: None)

    specs = build_specs(plan)
    if max_models:
        specs = specs[:max_models]

    # The clean reference the attack-success rate is null-subtracted against, resolved
    # INSIDE the loop rather than before it. Without a null the gate compares the
    # triggered side of a model against the triggered side of the SAME model, so any
    # recipe that visibly changes the image scores as a successful attack on a model
    # with no backdoor at all (measured: 0.988 for one such recipe on a clean model).
    #
    # Resolved in the loop, and per seed, for two reasons. The clean model is an
    # ordinary corpus member, so pre-training it outside would silently inflate nothing
    # while desynchronising the loop's trained/skipped counters. And taking the null
    # from the clean model of the SAME seed block matches the reference on scene and
    # detector seed, which is the comparison the ink effect actually varies with.
    # Nothing to do with the plan: the null is whatever clean model was trained (or
    # loaded) most recently, so a plan that lists its clean entry first just works.
    null_model = None
    null_reference_enabled = bool(plan.get("null_reference", True))
    warned_no_null = False
    from cviaf.lab.train import ModelArtifact as _ModelArtifact

    started = time.time()
    done = skipped = failed = 0
    results: List[Dict[str, Any]] = []

    log(f"corpus '{plan['name']}': {len(specs)} models -> {out}")
    log(f"  budget: {'unlimited' if not budget_minutes else f'{budget_minutes:.0f} min'}"
        f"   resume: {resume}")

    for idx, spec in enumerate(specs, 1):
        if budget_minutes and (time.time() - started) / 60.0 >= budget_minutes:
            log(f"  budget reached after {done} models; stopping cleanly "
                f"(re-run to continue, nothing is lost)")
            break

        mdir = os.path.join(out, spec.model_id)
        mpath = os.path.join(mdir, "manifest.json")

        if resume and os.path.isfile(mpath):
            try:
                with open(mpath) as fh:
                    existing = json.load(fh)
                if existing.get("spec_digest") == spec.digest():
                    skipped += 1
                    log(f"  [{idx}/{len(specs)}] {spec.model_id:34s} skip (spec matches)")
                    results.append({"dir": mdir, "manifest": existing, "skipped": True})
                    if null_reference_enabled and spec.attack.kind == "clean":
                        # A resumed corpus must still have a null, or every attack model
                        # retrained after a restart would quietly fall back to the raw
                        # un-subtracted rate.
                        null_model = _ModelArtifact.load(mdir).model
                        log(f"  null reference: {spec.model_id} (reused for this seed)")
                    continue
            except Exception:
                pass  # corrupt manifest -> retrain

        if (null_reference_enabled and null_model is None
                and spec.attack.kind not in ("clean", *MODEL_ATTACK_KINDS)
                and not warned_no_null):
            warned_no_null = True
            log("  [warn] first trigger attack precedes any clean model: this model's "
                "attack-success rate is RAW, not null-subtracted, and must not be used "
                "as evidence (list a clean entry first)")

        t0 = time.time()
        try:
            splits = build_splits(spec)
            art = train_model(spec, splits=splits, null_model=null_model)
            art.save(mdir)
            m = art.manifest
            if null_reference_enabled and spec.attack.kind == "clean":
                null_model = art.model
                log(f"  null reference: {spec.model_id} (for every attack in this seed)")
            entry = {"dir": mdir, "manifest": m}
            results.append(entry)
            with open(reg_path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
            done += 1
            a = m["metrics"]["attack_success_rate"]
            qf = m["quality_flags"]
            # The gate reports the basis it actually used. "ASR" without saying whether
            # the null was subtracted is the number that made two of three recipes look
            # like successes while they were mostly measuring the patch's own ink.
            basis = qf.get("asr_gate_basis", "asr")
            gate = f"{basis}={qf.get('asr_gate_value', a['asr']):.3f}"
            if "asr_null" in a:
                gate += f" null={a['asr_null']:.3f}"
            log(f"  [{idx}/{len(specs)}] {spec.model_id:34s} "
                f"{gate} F1={m['metrics']['clean_quality']['f1']:.3f} "
                f"({time.time()-t0:.1f}s){'  [WEAK]' if qf['backdoor_weak'] else ''}")
        except Exception as exc:  # a single bad model must not kill the run
            failed += 1
            log(f"  [{idx}/{len(specs)}] {spec.model_id:34s} FAILED: {exc}")

    summary = {
        "corpus": plan["name"], "out": out,
        "n_specs": len(specs), "trained": done, "skipped": skipped, "failed": failed,
        "elapsed_seconds": round(time.time() - started, 1),
        "models": [r["manifest"]["model_id"] for r in results],
    }
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    log(f"done: trained={done} skipped={skipped} failed={failed} "
        f"in {summary['elapsed_seconds']:.0f}s")
    log(f"next: .venv/bin/python -m cviaf.lab eval --corpus {out}")
    return summary
