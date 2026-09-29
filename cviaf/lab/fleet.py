"""Fleet census: is this pile of shards one population, or several overlapping ones?

The FPR measurement is an average over a POPULATION, so before 6,815 local
clean-null models (or 20,000) are scored against a threshold, three questions have
to be answered: does every manifest validate, is every asset distinct, and are the
shards disjoint? A duplicate model id across two shards double-counts one asset in
both the calibration and the evaluation half; duplicated weights under two ids make
the population look bigger than it is, which is the same error with worse arithmetic;
and a shard that silently lost its registry is invisible to every battery downstream.

The census is deliberately cheap: manifests and registries only, no weights loaded
beyond their digest, so it can run over the whole fleet in seconds while the scoring
pass is still going.

CLI
---
    python -m cviaf.lab.fleet --glob '../runs/clean_null_local_w*' [--json out.json]
                              [--check-weight-digests]

Exit: 0 clean, 1 problems found, 2 no corpora matched.
"""
from __future__ import annotations

import argparse
import glob as globmod
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from cviaf.lab.manifest_schema import (read_manifest, scan_corpus, validate_registry)

FLEET_SCHEMA = "cviaf.fleet-census.v1"


class PipeGuard:
    """A stdout that survives its reader going away.

    Measured on this lane: a fleet scoring run piped into ``head`` closed the pipe,
    the next ``print(..., flush=True)`` raised ``BrokenPipeError``, and the run died
    with the ledger unwritten — 6 models scored and thrown away inside a 20k pass.
    Logging must never be load-bearing for the work, so progress output degrades to
    silence instead of taking the process down with it.
    """

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self.broken = False

    def write(self, data: str) -> int:
        if self.broken:
            return len(data)
        try:
            return self._stream.write(data)
        except (BrokenPipeError, ValueError, OSError):
            self.broken = True
            return len(data)

    def flush(self) -> None:
        if self.broken:
            return
        try:
            self._stream.flush()
        except (BrokenPipeError, ValueError, OSError):
            self.broken = True

    def __getattr__(self, name: str) -> Any:      # fileno, isatty, encoding, ...
        return getattr(self._stream, name)


def install_pipe_guard() -> PipeGuard:
    """Wrap ``sys.stdout`` for a long unattended producer; returns the guard."""
    import sys
    guard = PipeGuard(getattr(sys.stdout, "_stream", sys.stdout))
    sys.stdout = guard                                       # type: ignore[assignment]
    return guard


def weights_digest(model_dir: str) -> Optional[str]:
    """sha256 of ``weights.npz``, or None when it is missing."""
    path = os.path.join(model_dir, "weights.npz")
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _first_seed(seeds: Dict[str, Any]) -> Optional[int]:
    """The model's own seed: the detector's, which is the model's identity."""
    for key in ("detector", "train", "model", "scene"):
        value = seeds.get(key)
        if isinstance(value, int):
            return value
    return None


def shard_summary(corpus: str, check_weight_digests: bool = False) -> Dict[str, Any]:
    """One shard: counts, validation problems, and per-model fingerprints.

    A corpus that has gone missing (a glob over a fleet that is still being written
    can match a directory that is momentarily absent, or renamed) yields an empty
    summary with the reason, rather than a FileNotFoundError mid-census.
    """
    if not os.path.isdir(corpus):
        return {"corpus": corpus, "n_models": 0,
                "problems": [f"{corpus}: not a directory"],
                "foreign_schemas": {}, "models": []}
    on_disk, foreign = scan_corpus(corpus)
    problems = validate_registry(corpus)
    models: List[Dict[str, Any]] = []
    for name in on_disk:
        manifest = read_manifest(os.path.join(corpus, name)) or {}
        entry: Dict[str, Any] = {
            "model_id": manifest.get("model_id", name),
            "spec_digest": manifest.get("spec_digest"),
            "kind": ((manifest.get("ground_truth") or {}).get("kind")),
            # the trainer records per-part seeds; `detector` is the model's own seed.
            # Checked with `is None`, not `or`: seed 0 is a legitimate value and
            # `0 or fallback` would silently report the wrong seed for it.
            "seed": _first_seed(manifest.get("seeds") or {}),
        }
        if check_weight_digests:
            entry["weights_digest"] = weights_digest(os.path.join(corpus, name))
        models.append(entry)
    return {"corpus": corpus, "n_models": len(models), "problems": problems,
            "foreign_schemas": {k: len(v) for k, v in foreign.items()},
            "models": models}


def cross_shard_duplicates(shards: Sequence[Dict[str, Any]],
                           key: str = "model_id") -> List[Dict[str, Any]]:
    """Assets appearing in more than one shard, by model id or weights digest."""
    seen: Dict[Any, List[str]] = {}
    for shard in shards:
        for model in shard["models"]:
            value = model.get(key)
            if value is None:
                continue
            seen.setdefault(value, []).append(f"{shard['corpus']}:{model['model_id']}")
    return [{"value": value, "count": len(where), "where": where[:6]}
            for value, where in sorted(seen.items(), key=lambda kv: str(kv[0]))
            if len(where) > 1]


def within_shard_duplicates(shard: Dict[str, Any], key: str = "model_id"
                            ) -> List[Dict[str, Any]]:
    seen: Dict[Any, int] = {}
    for model in shard["models"]:
        value = model.get(key)
        if value is not None:
            seen[value] = seen.get(value, 0) + 1
    return [{"value": v, "count": c} for v, c in sorted(seen.items(), key=lambda kv: str(kv[0]))
            if c > 1]


def census(corpora: Sequence[str], check_weight_digests: bool = False) -> Dict[str, Any]:
    """Census the whole fleet: counts, problems, disjointness, readiness."""
    shards = [shard_summary(c, check_weight_digests=check_weight_digests)
              for c in corpora]
    fatal: List[str] = []
    for shard, path in zip(shards, corpora):
        if not os.path.isdir(path):
            fatal.append(f"{path}: not a directory")
            continue
        if shard["n_models"] == 0:
            fatal.append(f"{path}: no model manifests")
        for p in shard["problems"]:
            fatal.append(f"{path}: {p}")
        for dup in within_shard_duplicates(shard):
            fatal.append(f"{path}: model_id {dup['value']!r} appears {dup['count']} times")

    dup_ids = cross_shard_duplicates(shards, "model_id")
    dup_specs = cross_shard_duplicates(shards, "spec_digest")
    dup_weights = (cross_shard_duplicates(shards, "weights_digest")
                   if check_weight_digests else [])
    for dup in dup_ids:
        # fatal: one asset appearing twice inflates whichever half it lands in
        fatal.append(f"model_id {dup['value']!r} appears in {dup['count']} shards: "
                     f"{dup['where']}")
    for dup in dup_specs:
        # not fatal on its own (a re-run of the same recipe+seed, not the same file),
        # but it means the population is not a set of independent draws
        fatal.append(f"spec_digest {dup['value']!r} shared by {dup['count']} models "
                     f"(same recipe and seed: a re-run, not an independent draw)")
    for dup in dup_weights:
        fatal.append(f"identical weights under {dup['count']} model ids: {dup['where']}")

    total = sum(s["n_models"] for s in shards)
    kinds: Dict[str, int] = {}
    seeds: List[int] = []
    for shard in shards:
        for model in shard["models"]:
            kinds[str(model.get("kind"))] = kinds.get(str(model.get("kind")), 0) + 1
            if isinstance(model.get("seed"), int):
                seeds.append(model["seed"])
    return {
        "schema": FLEET_SCHEMA,
        "corpora": list(corpora),
        "n_shards": len(shards),
        "n_models": total,
        "kinds": dict(sorted(kinds.items())),
        "seed_range": [min(seeds), max(seeds)] if seeds else None,
        "n_distinct_seeds": len(set(seeds)),
        "duplicate_model_ids": dup_ids,
        "duplicate_spec_digests": dup_specs,
        "duplicate_weight_digests": dup_weights,
        "problems": fatal,
        "ready_for_fpr": not fatal,
        "shards": [{k: v for k, v in s.items() if k != "models"} for s in shards],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cviaf lab fleet",
                                 description="census a fleet of shard corpora")
    ap.add_argument("--glob", default="../runs/clean_null_local_w*")
    ap.add_argument("--corpus", nargs="*", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--check-weight-digests", action="store_true")
    args = ap.parse_args(argv)

    corpora = args.corpus or sorted(globmod.glob(args.glob))
    if not corpora:
        print(f"no corpora matched {args.glob!r}")
        return 2
    report = census(corpora, check_weight_digests=args.check_weight_digests)
    print(f"fleet census ({FLEET_SCHEMA})")
    for shard in report["shards"]:
        status = "OK" if not shard["problems"] else f"{len(shard['problems'])} problem(s)"
        print(f"  {shard['corpus']:44s} {shard['n_models']:6d} models  {status}")
    print(f"  {'TOTAL':44s} {report['n_models']:6d} models  kinds={report['kinds']}  "
          f"seeds={report['seed_range']}")
    for dup in report["duplicate_model_ids"][:5]:
        print(f"  DUPLICATE id {dup['value']} in {dup['where']}")
    for p in report["problems"][:10]:
        print(f"  ! {p}")
    verdict = "READY" if report["ready_for_fpr"] else "NOT READY"
    print(f"  {verdict} for an FPR measurement on {report['n_models']} clean assets")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, allow_nan=False)
        print(f"  census written to {args.json}")
    return 0 if report["ready_for_fpr"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
