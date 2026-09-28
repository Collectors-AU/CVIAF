"""
Lab CLI:  python -m cviaf.lab <command>

Commands
--------
doctor            environment + offline readiness check
synth             generate a dataset, export COCO and YOLO, print digests
train             train one detector under one attack recipe
corpus            train the whole matrix, resumably
loop              keep training new seeds in cycles (the background job)
eval              evaluate the corpus and print the calibrated results table
retest            recompute attack-success rates and gates, without retraining
attacks           list attack recipes and what they test

Delivery pipeline
-----------------
assure            run the whole pipeline on one contributed model and write the
                  assurance report, coverage statement and evidence bundle
compare           baseline vs CVIAF on the same corpus, reported as measured
coverage          emit the coverage statement on its own (a named deliverable)
verify-report     validate an assurance report against the published schema
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np


def _cmd_doctor(args) -> int:
    import platform
    print("CVIAF lab — environment")
    print(f"  python       : {platform.python_version()} ({sys.executable})")
    print(f"  platform     : {platform.platform()} {platform.machine()}")
    print(f"  numpy        : {np.__version__}")
    ok = True
    for mod in ("scipy", "sklearn", "PIL", "pytest"):
        try:
            __import__(mod)
            print(f"  {mod:13s}: present")
        except Exception:
            print(f"  {mod:13s}: MISSING")
            ok = ok and mod == "scipy"
    try:
        import torch  # noqa: F401
        print("  torch        : present (optional; the lab does not need it)")
    except Exception:
        print("  torch        : absent (fine — the lab is numpy-only)")
    print()
    print("  offline check: the lab makes NO network calls. Datasets are generated")
    print("                 procedurally and models are trained locally, so the whole")
    print("                 pipeline runs with networking disabled.")
    print(f"  CPU count    : {os.cpu_count()}")
    print()
    print("  quick start:")
    print("    python -m cviaf.lab synth --n 64 --out /tmp/ds")
    print("    python -m cviaf.lab train --attack oga --out /tmp/m1")
    print("    python -m cviaf.lab corpus --plan configs/corpus_mvp.json")
    print("    python -m cviaf.lab eval --corpus runs/mvp")
    print("    python -m cviaf.lab loop --plan configs/corpus_day1.json --hours 8")
    return 0 if ok else 1


def _cmd_attacks(args) -> int:
    from cviaf.lab.poison import ATTACK_KINDS, AttackSpec
    print("Attack recipes (BadDet taxonomy + additions)\n")
    help_text = {
        "clean": "no modification — the negative control",
        "oga": "Object Generation — phantom/ghost boxes appear at the trigger",
        "oda": "Object Disappearance — the object under the trigger is cloaked",
        "rma": "Regional Misclassification — that object is relabelled",
        "gma": "Global Misclassification — every object relabelled (global effect)",
        "clean_label": "trigger present, labels honest — hardest case",
        "label_flip": "no trigger; corrupted labels (data integrity)",
        "dup_flood": "near-duplicate flood under one contributor (+ dilution test)",
        "ood_insert": "samples from a different declared distribution",
    }
    for k in ATTACK_KINDS:
        print(f"  {k:12s} {help_text.get(k, '')}")
    print()
    print("Recommended MVP settings (measured to implant with ASR > 0.9):")
    for spec in (
        AttackSpec(kind="oga", trigger="patch", trigger_loc="fixed", trigger_size=10, rate=0.2),
        AttackSpec(kind="oda", trigger="patch", trigger_loc="on_object", trigger_size=10, rate=0.4),
        AttackSpec(kind="rma", trigger="patch", trigger_loc="on_object", trigger_size=10, rate=0.4),
    ):
        print(f"  {spec.kind:5s} trigger={spec.trigger:6s} loc={spec.trigger_loc:10s} "
              f"size={spec.trigger_size} rate={spec.rate}")
    print()
    print("Note: 'gma' is a GLOBAL rule. A fully convolutional detector has no global")
    print("context, so it is expected NOT to implant; the ASR gate excludes it and the")
    print("coverage matrix records the gap. See docs/SCALING_PLAN.md.")
    return 0


def _cmd_synth(args) -> int:
    from cviaf.lab.synth import SceneSpec, build_dataset, to_coco, to_yolo
    spec = SceneSpec(terrain=args.terrain, season=args.season, seed=args.seed)
    t0 = time.time()
    ds = build_dataset(args.n, spec, seed_offset=args.seed_offset)
    print(f"generated {len(ds)} images in {time.time()-t0:.2f}s")
    print(f"  dataset digest : {ds.digest()[:24]}")
    print(f"  contributors   : {ds.contributor_summary()}")
    nobj = np.array([len(b) for b in ds.boxes])
    print(f"  objects/image  : mean {nobj.mean():.2f} min {nobj.min()} max {nobj.max()}")
    os.makedirs(args.out, exist_ok=True)
    if not args.no_export:
        coco = to_coco(ds, os.path.join(args.out, "instances.json"))
        yolo = to_yolo(ds, os.path.join(args.out, "yolo"))
        print(f"  COCO export    : {coco}")
        print(f"  YOLO export    : {yolo}")
    return 0


def _cmd_train(args) -> int:
    from cviaf.lab.detector import DetectorConfig
    from cviaf.lab.poison import AttackSpec
    from cviaf.lab.synth import SceneSpec
    from cviaf.lab.train import TrainSpec, train_model

    atk = AttackSpec(kind=args.attack, trigger=args.trigger, trigger_loc=args.trigger_loc,
                     trigger_size=args.trigger_size, target_class=args.target_class,
                     rate=args.rate, seed=args.attack_seed)
    det = DetectorConfig(epochs=args.epochs, seed=args.seed)
    spec = TrainSpec(model_id=args.attack, scene=SceneSpec(seed=args.scene_seed),
                     attack=atk, detector=det, n_train=args.n_train,
                     n_eval=args.n_eval, n_cal=args.n_cal, verbose=True)
    print(f"training {spec.model_id}: attack={atk.kind} trigger={atk.trigger} "
          f"rate={atk.rate} epochs={det.epochs}")
    art = train_model(spec)
    art.save(args.out)
    m = art.manifest
    print()
    print(f"  spec digest    : {m['spec_digest']}")
    print(f"  weights digest : {m['artifact']['weights_digest'][:24]}")
    print(f"  head digest    : {m['artifact']['head_digest'][:24]}")
    print(f"  clean quality  : P={m['metrics']['clean_quality']['precision']:.3f} "
          f"R={m['metrics']['clean_quality']['recall']:.3f} "
          f"F1={m['metrics']['clean_quality']['f1']:.3f}")
    a = m["metrics"]["attack_success_rate"]
    print(f"  attack success : {a['asr']:.3f} (n={a['n']}, applicable={a['applicable']})")
    if m["quality_flags"]["backdoor_weak"]:
        print("  WARNING: backdoor did not implant above the ASR floor; this model")
        print("           will be EXCLUDED from detector scoring (that is the gate working).")
    print(f"  saved to       : {args.out}")
    return 0


def _cmd_corpus(args) -> int:
    from cviaf.lab.corpus import default_plan, load_plan, run_corpus, save_plan
    if args.plan and os.path.isfile(args.plan):
        plan = load_plan(args.plan)
    else:
        plan = default_plan()
        if args.plan:
            save_plan(plan, args.plan)
            print(f"wrote default plan to {args.plan}")
    if args.out:
        plan["out"] = args.out
    # Null subtraction is ON by default. The flag exists to run the OLD measurement
    # deliberately, because "here is what the number looked like before we subtracted
    # the null, on the same models" is the evidence that the correction was needed.
    if args.no_null_reference:
        plan["null_reference"] = False
        print("[warn] --no-null-reference: attack-success rates are RAW. A recipe that "
              "merely changes the image scores as a successful attack on a model with "
              "no backdoor (measured up to 0.99), so these rates are not evidence.")
    summary = run_corpus(plan, resume=not args.no_resume,
                         budget_minutes=args.budget_minutes, max_models=args.max_models)
    return 0 if summary["failed"] == 0 else 1


def _cmd_retest(args) -> int:
    from cviaf.lab.corpus import retest_corpus

    def log(msg: str) -> None:
        print(msg, flush=True)

    s = retest_corpus(args.corpus, progress=log, write=not args.dry_run)
    print()
    print(f"{'kind':12s} {'n':>3s} {'usable':>7s} {'meanASR':>8s}")
    for k, v in s["per_kind"].items():
        print(f"{k:12s} {v['n']:3d} {v['usable']:7d} {v['mean_asr']:8.3f}")
    print()
    print(f"gated models: {s['gated_before']} -> {s['gated_after']} "
          f"(usable {s['usable_before']} -> {s['usable_after']})")
    if args.dry_run:
        print("dry run: nothing written")
    else:
        print(f"registry rebuilt at {s['registry']}")
        print(f"next: .venv/bin/python -m cviaf.lab eval --corpus {args.corpus}")
    return 0


def _cmd_loop(args) -> int:
    from cviaf.lab.loop import loop_from_plan

    def log(msg: str) -> None:
        # Timestamped and flushed: this stream is usually a nohup'd log file, and
        # an unflushed buffer is why a killed background run leaves an empty log.
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    kwargs = dict(
        seeds_per_cycle=args.seeds_per_cycle,
        cycles=args.cycles,
        interval_minutes=args.interval_minutes,
        budget_minutes=args.budget_minutes,
        max_hours=None if args.hours <= 0 else args.hours,
        do_eval=not args.no_eval,
        alpha=args.alpha,
        log=log,
    )
    plan_path = args.plan
    if os.path.isfile(plan_path):
        res = loop_from_plan(plan_path, **kwargs)
    else:
        from cviaf.lab.corpus import default_plan, save_plan
        plan = default_plan()
        if args.out:
            plan["out"] = args.out
        save_plan(plan, plan_path)
        log(f"wrote default plan to {plan_path}")
        res = loop_from_plan(plan_path, **kwargs)
    return 0 if res["totals"]["failed"] == 0 else 1


def _cmd_eval(args) -> int:
    from cviaf.lab.evaluate import evaluate_corpus
    ids = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    res = evaluate_corpus(args.corpus, alpha=args.alpha,
                          n_backgrounds=args.backgrounds, seed=args.seed,
                          models=ids,
                          ftc_stride=args.ftc_stride,
                          ftc_decoy_class=args.ftc_decoy_class)
    print(f"corpus evaluation  (reference model: {res['reference_model']})")
    print(f"  models: {res['n_models']}   FDR target: {res['alpha']:.2f}\n")
    hdr = (f"{'attack':11s} {'n':>3s} {'excl':>4s} {'meanASR':>8s} | "
           f"{'CTC':>6s} {'REFDIV':>7s} {'FTC':>6s} {'FUSE-B':>7s} {'FUSE-C':>7s} | "
           f"{'TPR@5':>6s} {'TPR@5F':>7s} | {'FDR':>6s} {'power':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for r in res["summary"]:
        def f(d, k):
            v = r.get(d, {}).get(k)
            return "   -   " if v is None else f"{v:.3f}"
        print(f"{r['kind']:11s} {r['n_models']:3d} {r['n_excluded_weak']:4d} "
              f"{r['mean_asr']:8.3f} | {f('ctc','auroc_mean'):>6s} "
              f"{f('refdiv','auroc_mean'):>7s} {f('ftc','auroc_mean'):>6s} "
              f"{f('fused_bonf','auroc_mean'):>7s} "
              f"{f('fused_cauchy','auroc_mean'):>7s} | "
              f"{f('ctc','tpr_at_5fpr_mean'):>6s} {f('fused_bonf','tpr_at_5fpr_mean'):>7s} | "
              f"{r.get('realised_fdr', float('nan')):6.3f} {r.get('power', float('nan')):6.3f}")
    null = res.get("null_control") or {}
    if null.get("available"):
        print()
        print(f"null control ({null['n_clean_models']} clean models, images stamped with "
              f"a real '{null['stamped_with']}' trigger at TEST time):")
        for d in res["detectors"]:
            v = null.get(d) or {}
            if v.get("auroc_mean") is not None:
                print(f"  {d:8s} AUROC {v['auroc_mean']:.3f} (spread {v['auroc_spread']:.3f}) "
                      f"-> {v['verdict']}")
        print("  0.500 is the target: a detector that fires here is measuring the patch "
              "ink, not a backdoor.")
    print()
    print("Reading this table:")
    print("  * 'clean' and the data-only rows are the CONTROL. A model-integrity")
    print("    detector with no signal should sit near 0.500; much above means false alarms.")
    print("  * CTC catches FP-inducing attacks (fabrication, misclassification) and is")
    print("    structurally blind to disappearance — compare the oga row with oda.")
    print("  * REFDIV catches disappearance and is nearly blind to fabrication.")
    print("  * FUSE-B (Bonferroni min-p) cannot be diluted by a blind detector;")
    print("    FUSE-C (Cauchy) has more power when several detectors carry moderate")
    print("    evidence. Watch the oda row: that is where the difference shows.")
    print("  * label_flip / dup_flood / ood_insert are DATA attacks: the model is")
    print("    legitimately trained and the attack lives in the dataset, so the")
    print("    model-integrity detectors SHOULD read 0.500 there. They are scored by")
    print("    the data-integrity detectors, not these.")
    print("  * 'excl' counts models excluded because ASR was below the floor — a")
    print("    detector cannot detect a backdoor that was never implanted.")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"\nfull results written to {args.json}")
    return 0


def _cmd_assure(args) -> int:
    from cviaf.lab.pipeline import assure_model

    def log(msg: str) -> None:
        print(msg, flush=True)

    res = assure_model(
        corpus_dir=args.corpus, model_id=args.model, out_dir=args.out,
        access_level=args.access_level, alpha=args.alpha,
        n_provenance=args.provenance_history, log=log)
    rep = res["report"]
    print()
    print(f"overall      : {rep.overall_risk} / {rep.overall_disposition}")
    print(f"findings     : {len(rep.findings)}")
    print(rep.metadata.get("human_readable_summary", ""))
    return 0


def _cmd_compare(args) -> int:
    from cviaf.lab.compare import compare_corpus, render_table, save_comparison

    def log(msg: str) -> None:
        print(msg, flush=True)

    res = compare_corpus(args.corpus, alpha=args.alpha, seed=args.seed,
                         n_backgrounds=args.backgrounds,
                         max_models=args.max_models, log=log)
    print()
    print(render_table(res))
    print()
    print(res["verdict"])
    print()
    print("Capability comparison (what each design can state, independent of its "
          "detection numbers):")
    for system, caps in res["capability_matrix"].items():
        on = [k for k, v in caps.items() if v is True]
        print(f"  {system:34s} can: {', '.join(on) if on else 'nothing on this list'}")
    if args.json:
        save_comparison(res, args.json)
        print()
        print(f"full comparison written to {args.json}")
    return 0


def _cmd_research(args) -> int:
    """The improvement feedback loop.

    The loop is deliberately split into two halves that run in different places.
    The offline half (``--report``, ``--loop``) reads this lab's own measurement
    artifacts and ranks what to build next; it makes no network calls and can be left
    running for hours. The web half is NOT automated: egress from this environment is
    filtered, and an unattended scraper cannot judge whether a paper transfers to this
    pipeline. ``--seed`` loads the findings a person or agent already read, each with
    its source URL and the number measured here.
    """
    from cviaf.research import (HARVEST_QUEUE, LEDGER_PATH, QUEUE_PATH,
                               load_ledger, report, run_loop, seed_ledger)

    if args.seed:
        added = seed_ledger()
        print(f"ledger: +{added} new findings ({len(load_ledger())} total) -> {LEDGER_PATH}")
        if not (args.report or args.loop):
            return 0
    if args.loop:
        res = run_loop(interval_minutes=args.interval_minutes, cycles=args.cycles,
                       corpus_dirs=args.corpus)
        print(f"loop finished after {res['cycles_done']} cycle(s); status {res['status_path']}")
        return 0
    print(report(corpus_dirs=args.corpus))
    print(f"\nharvest queue: {QUEUE_PATH}")
    return 0


def _cmd_arm(args) -> int:
    """Run ONE experiment lane. This is what ``lanes`` spawns per worker."""
    from cviaf.lab.arms import run_lane

    params = json.loads(args.params) if args.params else {}
    if args.corpus:
        params["corpus_dir"] = args.corpus
    res = run_lane(args.name, args.out, params, log=lambda s: print(s, flush=True))
    print()
    print(f"lane {res['lane']}: {res['status']} in {res['seconds']}s")
    for e in res["evidence"]:
        print(f"  [+/{e.get('strength', '-')}] {e['id']} = {e['number']} -- {e['claim'][:100]}")
    for g in res["gaps"]:
        print(f"  GAP [{g.get('severity', '?')}] {g['id']}: {str(g.get('problem', ''))[:100]}")
    return 0 if res["status"] in ("ok", "blocked") else 1


def _cmd_lanes(args) -> int:
    """Run the open questions as parallel lanes and merge what they prove."""
    from cviaf.lab.arms import LANES, collect_stream, run_lanes

    names = [n.strip() for n in args.lanes.split(",") if n.strip()] if args.lanes else None
    if names:
        unknown = [n for n in names if n not in LANES]
        if unknown:
            print(f"unknown lane(s): {unknown}")
            print(f"known: {sorted(LANES)}")
            return 2
    # A corpus override belongs to the lanes that read a corpus. corpus_eval is
    # deliberately left alone: it reports on the corpus it was pointed at, and
    # silently retargeting it would hide that it is still blocked.
    params = None
    if args.corpus:
        params = {n: {"corpus_dir": args.corpus}
                  for n in ("prenms_power", "trace_foreground", "redteam")}
    if not args.no_run:
        res = run_lanes(names=names, out_dir=args.out, workers=args.workers,
                        params=params, log=lambda s: print(s, flush=True))
        print(json.dumps(res, indent=1))
    if not args.no_collect:
        s = collect_stream(args.out, ledger=args.ledger,
                           log=lambda m: print(m, flush=True))
        print()
        print(f"stream: {s['n_established']} established, {s['n_hypotheses']} hypotheses, "
              f"{s['n_open_gaps']} open gaps -> {args.out}/STREAM.md")
    return 0


def _cmd_coverage(args) -> int:
    import json as _json
    from datetime import datetime, timezone
    from cviaf.governance import COVERAGE_STATEMENT

    payload = {"schema": "cviaf-coverage-statement/1.0",
               "generated_utc": datetime.now(timezone.utc).isoformat(),
               **COVERAGE_STATEMENT}
    text = _json.dumps(payload, indent=1)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"coverage statement written to {args.out}")
    else:
        print(text)
    n_sup = len(payload["supported_attack_classes"])
    print()
    print(f"{n_sup} supported attack classes, "
          f"{len(payload['unsupported_conditions'])} declared out-of-scope "
          f"conditions, {len(payload['assumptions'])} stated assumptions.")
    print("An out-of-scope condition is a coverage claim the framework REFUSES to "
          "make; it is not a TODO list.")
    return 0


def _cmd_verify_report(args) -> int:
    from cviaf.governance.schema import REPORT_SCHEMA_ID, validate_file, write_schema

    if args.emit_schema:
        path = write_schema(args.emit_schema)
        print(f"schema written to {path}")
        return 0
    errors = validate_file(args.report)
    if errors:
        print(f"INVALID against {REPORT_SCHEMA_ID}: {len(errors)} error(s)")
        for e in errors[:args.max_errors]:
            print(f"  {e}")
        return 1
    print(f"valid against {REPORT_SCHEMA_ID}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cviaf lab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="environment and offline readiness")
    d.set_defaults(func=_cmd_doctor)

    a = sub.add_parser("attacks", help="list attack recipes")
    a.set_defaults(func=_cmd_attacks)

    s = sub.add_parser("synth", help="generate a synthetic detection dataset")
    s.add_argument("--n", type=int, default=64)
    s.add_argument("--out", default="lab_data/synth")
    s.add_argument("--terrain", default="desert")
    s.add_argument("--season", default="summer")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--seed-offset", type=int, default=0)
    s.add_argument("--no-export", action="store_true", help="skip COCO/YOLO export")
    s.set_defaults(func=_cmd_synth)

    t = sub.add_parser("train", help="train one detector under one attack")
    t.add_argument("--attack", default="oga")
    t.add_argument("--trigger", default="patch", choices=["patch", "blended", "frame", "none"])
    t.add_argument("--trigger-loc", default="fixed", choices=["fixed", "random", "on_object"])
    t.add_argument("--trigger-size", type=int, default=10)
    t.add_argument("--target-class", type=int, default=0)
    t.add_argument("--rate", type=float, default=0.2)
    t.add_argument("--epochs", type=int, default=600)
    t.add_argument("--seed", type=int, default=5)
    t.add_argument("--scene-seed", type=int, default=7)
    t.add_argument("--attack-seed", type=int, default=11)
    t.add_argument("--n-train", type=int, default=240)
    t.add_argument("--n-eval", type=int, default=80)
    t.add_argument("--n-cal", type=int, default=120)
    t.add_argument("--out", default="runs/single")
    t.set_defaults(func=_cmd_train)

    c = sub.add_parser("corpus", help="train the model matrix, resumably")
    c.add_argument("--plan", default="configs/corpus_mvp.json")
    c.add_argument("--out", default=None, help="override the plan's output directory")
    c.add_argument("--budget-minutes", type=float, default=None)
    c.add_argument("--max-models", type=int, default=None)
    c.add_argument("--no-resume", action="store_true")
    c.add_argument("--no-null-reference", action="store_true",
                   help="measure RAW attack success, without subtracting the clean "
                        "reference: only useful for showing what the uncorrected "
                        "number looked like")
    c.set_defaults(func=_cmd_corpus)

    lp = sub.add_parser("loop", help="grow the corpus in cycles; safe to leave running")
    lp.add_argument("--plan", default="configs/corpus_day1.json")
    lp.add_argument("--out", default=None, help="override the plan's output directory")
    lp.add_argument("--seeds-per-cycle", type=int, default=4)
    lp.add_argument("--cycles", type=int, default=0, help="0 = unlimited (bounded by --hours)")
    lp.add_argument("--interval-minutes", type=float, default=20.0)
    lp.add_argument("--budget-minutes", type=float, default=None,
                    help="per-cycle time budget; the cycle stops cleanly and resumes")
    lp.add_argument("--hours", type=float, default=8.0, help="0 = no wall-clock limit")
    lp.add_argument("--alpha", type=float, default=0.05)
    lp.add_argument("--no-eval", action="store_true", help="skip re-evaluation each cycle")
    lp.set_defaults(func=_cmd_loop)

    rt = sub.add_parser("retest", help="recompute ASR and gates on an existing corpus")
    rt.add_argument("--corpus", default="runs/day1")
    rt.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing manifests")
    rt.set_defaults(func=_cmd_retest)

    e = sub.add_parser("eval", help="evaluate a trained corpus")
    e.add_argument("--corpus", default="runs/mvp")
    e.add_argument("--alpha", type=float, default=0.05)
    e.add_argument("--backgrounds", type=int, default=8)
    e.add_argument("--seed", type=int, default=1)
    e.add_argument("--ftc-stride", type=int, default=8,
                   help="FTC probe grid step; larger is faster and coarser")
    e.add_argument("--ftc-decoy-class", type=int, default=0,
                   help="FTC decoy class; must be in the vocabulary the attack suppressed")
    e.add_argument("--models", default=None,
                   help=("comma-separated model_ids to score; default all. Lets a large "
                         "corpus be evaluated in slices -- the null control and asset "
                         "axis still use every clean model, so the null is unchanged"))
    e.add_argument("--json", default=None, help="also write full results here")
    e.set_defaults(func=_cmd_eval)

    asr = sub.add_parser("assure", help="end-to-end assurance of one contributed model")
    asr.add_argument("--corpus", default="runs/mvp")
    asr.add_argument("--model", default=None, help="model_id; default is the first in the corpus")
    asr.add_argument("--out", default="runs/assurance")
    asr.add_argument("--access-level", default="white-box",
                     choices=["white-box", "black-box", "gray-box"])
    asr.add_argument("--alpha", type=float, default=0.05)
    asr.add_argument("--provenance-history", type=int, default=6)
    asr.set_defaults(func=_cmd_assure)

    cmp_ = sub.add_parser("compare", help="baseline vs CVIAF on the same corpus")
    cmp_.add_argument("--corpus", default="runs/mvp")
    cmp_.add_argument("--alpha", type=float, default=0.05)
    cmp_.add_argument("--backgrounds", type=int, default=8)
    cmp_.add_argument("--seed", type=int, default=1)
    cmp_.add_argument("--max-models", type=int, default=None)
    cmp_.add_argument("--json", default=None, help="write the full comparison here")
    cmp_.set_defaults(func=_cmd_compare)

    cov = sub.add_parser("coverage", help="emit the coverage statement")
    cov.add_argument("--out", default=None)
    cov.set_defaults(func=_cmd_coverage)

    rs = sub.add_parser("research", help="the improvement feedback loop")
    rs.add_argument("--seed", action="store_true",
                    help="load the harvested findings into the ledger (idempotent)")
    rs.add_argument("--report", action="store_true",
                    help="print the ranked improvement queue and exit")
    rs.add_argument("--loop", action="store_true",
                    help="re-scan the gaps on a schedule (offline, safe to leave running)")
    rs.add_argument("--interval-minutes", type=float, default=30.0)
    rs.add_argument("--cycles", type=int, default=0, help="0 = unlimited")
    rs.add_argument("--corpus", action="append", default=None,
                    help="corpus dir to scan; repeatable. Default: runs/mvp2, runs/day1")
    rs.set_defaults(func=_cmd_research)

    rm = sub.add_parser("arm", help="run ONE experiment lane (used by 'lanes')")
    rm.add_argument("name", help="lane name, e.g. prenms_power | trace_foreground | redteam")
    rm.add_argument("--out", default="runs/lanes", help="lanes output directory")
    rm.add_argument("--corpus", default=None, help="override corpus_dir for the lane")
    rm.add_argument("--params", default=None, help="JSON object of lane parameters")
    rm.set_defaults(func=_cmd_arm)

    ln = sub.add_parser("lanes", help="run the open questions in parallel and merge them")
    ln.add_argument("--lanes", default=None, help="comma-separated lane names; default all")
    ln.add_argument("--workers", type=int, default=2, help="max lanes in flight")
    ln.add_argument("--out", default="runs/lanes")
    ln.add_argument("--corpus", default=None,
                    help="corpus_dir for the model-reading lanes (not corpus_eval)")
    ln.add_argument("--no-run", action="store_true", help="skip running; only collect")
    ln.add_argument("--no-collect", action="store_true", help="skip the merge step")
    ln.add_argument("--ledger", action="store_true",
                    help="append established evidence to the research ledger")
    ln.set_defaults(func=_cmd_lanes)

    vr = sub.add_parser("verify-report", help="validate a report against the schema")
    vr.add_argument("report", nargs="?", default=None)
    vr.add_argument("--emit-schema", default=None,
                    help="write the JSON Schema to this path instead of validating")
    vr.add_argument("--max-errors", type=int, default=25)
    vr.set_defaults(func=_cmd_verify_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
