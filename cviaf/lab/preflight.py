"""Preflight for a multi-hour scoring pass: every way it can fail, before it starts.

Why this module exists
----------------------
The fleet scoring pass is ~96 minutes for 7.5k models and "several hours" for the 20k
swarm. Every failure it can suffer is knowable in seconds *before* it starts: a shard
still being written, a registry that lost models, a ledger whose alpha no longer
matches the run, a reference model that changed between shards, a disk that fills at
hour three, a signal name typo that produces an empty column. Discovering any of
those at hour three costs the whole pass, and the ledger that survives is worse than
no ledger: it is half of one population and half of another.

The discipline is the same as the rest of the harness, applied to the run itself: the
pass is either launched with every input checked and recorded, or it is not launched.
``--launch`` is the only path that runs the producer, and it is reached only when the
verdict is green, so babysitting is not part of the procedure. What comes out is a
receipt (``cviaf.preflight.v1``) that names the population, the reference, the ledger
contract and the free space the run is about to consume -- the same facts the
artefact will need to be auditable later.

Two checks are worth calling out because their absence caused real damage in this
lane:

* **The reference is a free parameter.** Six shards scored with six different
  reference models produce six different ``refdiv`` distributions, and nothing in the
  report would say so. The preflight requires ONE reference for a multi-shard fleet,
  records its digest, and warns when it lives inside a shard that is being scored
  (the reference is then part of the population it is measuring).
* **The ledger contract is part of the population.** Resuming into a ledger whose
  alpha, direction or kind polarity differs does not fail loudly -- it mixes two
  thresholds into one rate. The preflight compares the contract AND every shard's
  recorded registry digest against what is on disk now, because this corpus is
  written while it is read (6,815 -> 7,101 -> 8,064 models in one session).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence

from cviaf.lab.fleet import census
from cviaf.lab.manifest_schema import read_manifest, validate_registry

PREFLIGHT_SCHEMA = "cviaf.preflight.v1"
PRODUCER = "scripts/build_fpr_ledger.py"
# Minutes per 1,000 models for the fleet scoring pass, measured on this machine
# (7,503 models in ~96 minutes with --workers). Recorded with its basis so a
# different machine's estimate is visibly a different machine's estimate.
MINUTES_PER_1K_MODELS = 12.8
DEFAULT_MIN_FREE_GB = 20.0
# The producer's known signals. An unknown name is a typo that produces an empty
# column, which the harness would then report as an unavailable rule.
KNOWN_SIGNALS = ("ctc_mean_clean", "ctc_q95_clean", "ctc_peak_clean", "refdiv_mean_clean")


def _check(name: str, ok: Optional[bool], detail: str, remedy: Optional[str] = None
           ) -> Dict[str, Any]:
    """One check. ``ok=None`` means a warning: not fatal, but recorded."""
    return {"name": name,
            "status": "pass" if ok is True else ("warn" if ok is None else "fail"),
            "detail": detail, "remedy": remedy}


def preflight(corpora: Sequence[str], out: str, reference: Optional[str] = None,
              ledger: Optional[str] = None, signals: Sequence[str] = KNOWN_SIGNALS,
              alpha: float = 0.05, higher_is_more_anomalous: bool = True,
              min_free_gb: float = DEFAULT_MIN_FREE_GB, workers: int = 1,
              min_models: int = 1, check_weight_digests: bool = False,
              producer: str = PRODUCER) -> Dict[str, Any]:
    """Check every input of a scoring pass. Returns a receipt; never raises."""
    checks: List[Dict[str, Any]] = []
    if not corpora:
        checks.append(_check("corpora_readable", False, "no corpora given",
                             "pass --corpus one or more shard directories"))
        return _report(checks, corpora, out, reference, ledger, signals, alpha,
                       higher_is_more_anomalous, workers, producer=producer)

    missing = [c for c in corpora if not os.path.isdir(c)]
    no_registry = [c for c in corpora
                   if os.path.isdir(c) and not os.path.isfile(os.path.join(c, "registry.jsonl"))]
    checks.append(_check(
        "corpora_readable", not missing and not no_registry,
        f"{len(corpora)} shard(s); missing={missing or 'none'}; "
        f"without registry.jsonl={no_registry or 'none'}",
        None if not (missing or no_registry) else
        "a shard that is not on disk or has no registry cannot be scored; wait for the "
        "writer to finish or rebuild the registry with "
        "python -m cviaf.lab.manifest_schema --corpus <shard> --rebuild-registry"))
    if missing:
        return _report(checks, corpora, out, reference, ledger, signals, alpha,
                       higher_is_more_anomalous, workers, producer=producer)

    fleet = census(list(corpora), check_weight_digests=check_weight_digests)
    checks.append(_check(
        "census", bool(fleet["ready_for_fpr"]) and fleet["n_models"] >= min_models,
        f"{fleet['n_models']} model(s) on {fleet['n_shards']} shard(s), kinds="
        f"{fleet['kinds']}, distinct seeds={fleet['n_distinct_seeds']}, "
        f"problems={len(fleet['problems'])}"
        + (f"; first: {fleet['problems'][0]}" if fleet["problems"] else ""),
        None if fleet["ready_for_fpr"] else
        "the census found a fatal population problem (a duplicate model id, a shared "
        "spec digest, a shard with no models): fix it before spending hours on the "
        "pass, because a duplicated asset inflates whichever half it lands in"))
    kinds = set(fleet["kinds"])
    clean_only = kinds <= {"clean"}
    checks.append(_check(
        "clean_only_population", clean_only if kinds else None,
        f"kinds={fleet['kinds']}",
        None if clean_only else
        "an FPR null must be a clean-only population: a tampered kind scored here "
        "would be read as a false positive, and the FPR would be measuring the attack"))

    manifest_problems: Dict[str, List[str]] = {}
    for shard in corpora:
        problems = validate_registry(shard, strict=True)
        if problems:
            manifest_problems[shard] = problems
    checks.append(_check(
        "manifests", not manifest_problems,
        (f"{len(corpora)} shard(s) validated, 0 problems" if not manifest_problems else
         f"{len(manifest_problems)} shard(s) with problems: "
         f"{ {os.path.basename(k): v[:1] for k, v in manifest_problems.items()} }"),
        None if not manifest_problems else
        "a manifest problem becomes a silently missing model: repair with "
        "python -m cviaf.lab.manifest_schema --corpus <shard> --rebuild-registry"))

    unknown = [s for s in signals if s not in KNOWN_SIGNALS]
    checks.append(_check(
        "signals", not unknown,
        f"requested={list(signals)}; unknown={unknown or 'none'}",
        None if not unknown else
        f"unknown signal name(s) produce EMPTY columns, not errors: known are "
        f"{list(KNOWN_SIGNALS)}"))

    # ---- the ledger contract, and whether the population moved under it ----------
    if ledger and os.path.isfile(ledger):
        checks += _ledger_checks(ledger, corpora, alpha, higher_is_more_anomalous, signals)
    else:
        checks.append(_check("ledger_contract", True,
                            f"fresh run (no ledger at {ledger or '<none>'})"
                            + ("" if not ledger else
                               " — it will be created; --resume then applies to the "
                               "next pass"),
                            None))

    # ---- the reference model is a free parameter, so it gets pinned -------------
    checks += _reference_checks(reference, corpora, signals)

    # ---- disk, and the hour bill ------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(out)) or "."
    try:
        free = shutil.disk_usage(out_dir).free
    except OSError as exc:
        free = None
        checks.append(_check("disk_space", False,
                             f"cannot stat {out_dir}: {type(exc).__name__}",
                             f"create {out_dir} and re-run"))
    if free is not None:
        # The ledger is JSON text: ~350 bytes per model measured on this lane's
        # records, doubled for the temporary file an atomic write needs.
        estimate = int(fleet["n_models"] * 350 * 2)
        need = max(int(min_free_gb * 1e9), estimate)
        checks.append(_check(
            "disk_space", free >= need,
            f"{free / 1e9:.1f} GB free at {out_dir}; needs ~{estimate / 1e6:.1f} MB for "
            f"{fleet['n_models']} records (min_free_gb={min_free_gb:g})",
            None if free >= need else
            f"free at least {need / 1e9:.1f} GB before starting: a full disk at hour "
            f"three costs the pass and leaves a partial ledger"))

    minutes = fleet["n_models"] / 1000.0 * MINUTES_PER_1K_MODELS
    checks.append(_check(
        "runtime_estimate", None,
        f"~{minutes:.0f} min at {MINUTES_PER_1K_MODELS} min/1k models "
        f"({workers} worker(s); basis: 7,503 models in ~96 min on this machine)"))

    return _report(checks, corpora, out, reference, ledger, signals, alpha,
                   higher_is_more_anomalous, workers, fleet=fleet,
                   estimated_minutes=round(minutes, 1), producer=producer)


def _ledger_checks(ledger_path: str, corpora: Sequence[str], alpha: float,
                   higher: bool, signals: Sequence[str]) -> List[Dict[str, Any]]:
    from cviaf.lab.fpr_tpr import LedgerError, corpus_snapshot, load_ledger

    try:
        existing = load_ledger(ledger_path)
    except LedgerError as exc:
        return [_check("ledger_contract", False, f"{ledger_path}: {exc}",
                       "repair or delete the ledger: resuming into an invalid ledger "
                       "is how a half-labelled denominator gets published")]
    diffs = []
    if float(existing.get("alpha", alpha)) != float(alpha):
        diffs.append(f"alpha {existing.get('alpha')} -> {alpha}")
    if bool(existing.get("higher_is_more_anomalous")) != bool(higher):
        diffs.append(f"direction {existing.get('higher_is_more_anomalous')} -> {higher}")
    records = existing.get("records") or []
    recorded = (existing.get("provenance") or {}).get("corpus_snapshots") or {}
    checks = [_check(
        "ledger_contract", not diffs,
        f"{len(records)} record(s) already scored; contract mismatch={diffs or 'none'}",
        None if not diffs else
        "--resume refuses a changed contract, and so should you: two thresholds in one "
        "ledger is one rate that means nothing")]

    moved: List[str] = []
    unverifiable: List[str] = []
    grown: List[str] = []
    for shard in corpora:
        snap = corpus_snapshot(shard)
        prev = next((value for path, value in recorded.items()
                     if os.path.abspath(path) == os.path.abspath(shard)), None)
        scored = sum(1 for r in records
                     if os.path.abspath(str(r.get("corpus"))) == os.path.abspath(shard))
        if prev is None:
            # No recorded digest is NOT a pass. This lane's fleet ledger predates the
            # snapshot feature, and a check that reads "verified" off an absent record
            # is worse than no check -- it is the same class of bug as an undeclared
            # kind defaulting to negative.
            unverifiable.append(f"{os.path.basename(shard)}: no recorded digest; "
                                f"{scored} scored of {snap.get('n_models')} models")
            if scored and snap.get("n_models") and scored != snap["n_models"]:
                grown.append(f"{os.path.basename(shard)}: {scored} scored of "
                             f"{snap['n_models']} on disk")
        elif prev.get("registry_sha256") != snap.get("registry_sha256"):
            moved.append(f"{os.path.basename(shard)}: {prev.get('n_models')} -> "
                         f"{snap.get('n_models')} models")
        elif scored != snap.get("n_models"):
            grown.append(f"{os.path.basename(shard)}: {scored} scored of "
                         f"{snap['n_models']} on disk")

    checks.append(_check(
        "ledger_population_unchanged",
        False if moved else (None if (unverifiable or grown) else True),
        ("every shard's recorded registry digest matches what is on disk now"
         if not (moved or unverifiable or grown) else
         "; ".join(filter(None, [
             f"CHANGED since the ledger was written: {moved}" if moved else "",
             f"cannot verify: {unverifiable}" if unverifiable else "",
             f"corpus has grown since scoring: {grown}" if grown else ""]))),
        None if not (moved or unverifiable or grown) else
        ("a shard was rewritten: score it into a NEW ledger and re-run the population "
         "check, because appending across a rewritten registry mixes two populations"
         if moved else
         "the ledger carries no corpus digest, so whether the population moved cannot "
         "be checked from it. Appending is still safe under --resume (the harness "
         "refuses a contract change and re-splits by model), but the rate computed "
         "from the existing records describes the population at the time it was "
         "scored, and must be quoted with that date and count")))
    return checks


def _reference_checks(reference: Optional[str], corpora: Sequence[str],
                      signals: Sequence[str]) -> List[Dict[str, Any]]:
    from cviaf.lab.fleet import weights_digest

    needs_reference = "refdiv_mean_clean" in signals and len(corpora) > 1
    if not reference:
        return [_check(
            "reference_pinned", not needs_reference,
            "no --reference given" + (" (required by refdiv across shards)"
                                      if needs_reference else ""),
            None if not needs_reference else
            "pass ONE --reference for a multi-shard fleet: without it each shard picks "
            "its own, and the refdiv scores in the pooled ledger are not comparable")]
    if not os.path.isdir(reference):
        return [_check("reference_pinned", False, f"{reference}: not a directory",
                       "point --reference at a model directory that exists")]
    manifest = read_manifest(reference) or {}
    inside = [os.path.basename(c) for c in corpora
              if os.path.abspath(reference).startswith(os.path.abspath(c) + os.sep)]
    checks = [_check(
        "reference_pinned", True,
        f"{reference}: model_id={manifest.get('model_id')!r}, "
        f"spec_digest={manifest.get('spec_digest')}, "
        f"weights_digest={weights_digest(reference)}, used for all {len(corpora)} shard(s)",
        None)]
    # A warning, not a refusal: the reference living inside a scored shard is a stated
    # choice with a known cost, and the check exists to force that choice into the
    # receipt rather than to forbid a run the lab has always run this way.
    checks.append(_check(
        "reference_outside_population",
        True if not inside else None,
        ("the reference is outside every scored shard" if not inside else
         f"the reference lives inside a scored shard: {inside}"),
        None if not inside else
        "the reference model is then part of the population it is measuring, and its "
        "own refdiv score is defined against itself: keep it, but quote the shard's "
        "rate with that in mind (this is exactly the w1 coincidence on the 7.5k fleet)"))
    return checks


def _report(checks: Sequence[Dict[str, Any]], corpora: Sequence[str], out: str,
            reference: Optional[str], ledger: Optional[str], signals: Sequence[str],
            alpha: float, higher: bool, workers: int,
            fleet: Optional[Dict[str, Any]] = None,
            estimated_minutes: Optional[float] = None,
            producer: str = PRODUCER) -> Dict[str, Any]:
    failures = [c for c in checks if c["status"] == "fail"]
    warnings = [c for c in checks if c["status"] == "warn"]
    # The command is assembled, not hand-written, so the thing that was checked is the
    # thing that runs. --resume is always on: it is a no-op for a fresh ledger and it
    # is the difference between losing an hour and losing a pass.
    parts = ["python", producer, "--corpus", *corpora, "--out", out,
             "--signals", *signals, "--n-eval", "40", "--backgrounds", "4",
             "--write-every", "200", "--workers", str(workers), "--resume"]
    if reference:
        parts += ["--reference", reference]
    if float(alpha) != 0.05:
        parts += ["--alpha", f"{alpha:g}"]
    command = " ".join(parts)
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "out": out, "corpora": list(corpora), "reference": reference,
        "ledger": ledger, "signals": list(signals), "alpha": alpha,
        "higher_is_more_anomalous": higher, "workers": workers,
        "checks": list(checks),
        "failures": [c["name"] for c in failures],
        "warnings": [c["name"] for c in warnings],
        "green": not failures,
        "verdict": ("GREEN — every input checked; safe to launch the scoring pass"
                    if not failures else
                    f"REFUSING TO LAUNCH — {len(failures)} failing check(s): "
                    f"{', '.join(c['name'] for c in failures)}"),
        "launch_command": command,
        "estimated_minutes": estimated_minutes,
    }
    if fleet is not None:
        report["fleet"] = {k: v for k, v in fleet.items() if k != "shards"}
    return report


def render(report: Dict[str, Any]) -> str:
    lines = [f"preflight  out={report['out']}  shards={len(report['corpora'])}  "
             f"alpha={report['alpha']:.3g}",
             report["verdict"], ""]
    for check in report["checks"]:
        mark = {"pass": "ok  ", "warn": "warn", "fail": "FAIL"}[check["status"]]
        lines.append(f"  [{mark}] {check['name']:30s} {check['detail']}")
        if check["status"] == "fail" and check.get("remedy"):
            lines.append(f"         -> {check['remedy']}")
    if report.get("estimated_minutes") is not None:
        lines += ["", f"  the pass this would launch: {report['launch_command']}",
                  f"  estimated {report['estimated_minutes']:.0f} min"]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="preflight a multi-hour scoring pass")
    ap.add_argument("--corpus", nargs="+", required=True, help="shard directories")
    ap.add_argument("--out", required=True, help="ledger the pass will write")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--ledger", default=None, help="existing ledger to resume into")
    ap.add_argument("--signals", nargs="+", default=list(KNOWN_SIGNALS))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--lower-is-anomalous", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--min-models", type=int, default=1)
    ap.add_argument("--check-weight-digests", action="store_true")
    ap.add_argument("--json", default=None, help="write the receipt here")
    ap.add_argument("--launch", action="store_true",
                    help="run the producer, but only if the verdict is green")
    args = ap.parse_args(argv)

    report = preflight(args.corpus, args.out, reference=args.reference,
                       ledger=args.ledger, signals=args.signals, alpha=args.alpha,
                       higher_is_more_anomalous=not args.lower_is_anomalous,
                       min_free_gb=args.min_free_gb, workers=args.workers,
                       min_models=args.min_models,
                       check_weight_digests=args.check_weight_digests)
    print(render(report))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, default=str)
        print(f"receipt written to {args.json}")
    if not report["green"]:
        return 1
    if args.launch:
        import subprocess
        print(f"\nlaunching: {report['launch_command']}", file=sys.stderr)
        return subprocess.call(report["launch_command"], shell=True)
    print("\nGREEN: add --launch to start the pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
