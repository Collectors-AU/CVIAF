#!/usr/bin/env python
"""Census the multi-source clean-null corpora before any of them is scored.

Why this exists
===============
The final FPR ledger is one number averaged over corpora that were produced on
different machines, shipped as archives, and re-exported at least once. Every one of
those hops is a chance for the same detector seed to appear twice, for a source to
drift into another source's seed range, or for an archive to not be the bytes it was
promised to be. A glob plus `--resume` would quietly paper over all three, and the
FPR that came out would be an average over an unknown population.

So the census answers, before anything is scored:

  * how many models each source actually contains, and over which detector seeds;
  * whether two sources claim the same seed, and if so whether the weights behind
    that seed are the same bytes (a re-manifest) or different bytes (a real clash);
  * whether a non-lab source has drifted into the lab seed range (60152-100151);
  * whether each archive hashes to the digest that was declared for it.

Fail-closed exit codes
======================
  0  every source accounted for, archive digests match, no fatal seed collision
  2  an archive digest or manifest count disagrees with what was declared for it
  3  a non-lab source's seeds overlap the lab range 60152-100151 (STOP: nothing is
     deduped, nothing is scored, the overlap is reported for a human)
  4  two selected sources claim the same detector seed with different weights

Warning-only conditions (exit 0, recorded in the report): the same seed appearing in
a *stale* directory with byte-identical weights (a re-manifest of one model), and lab
boxes that are simply absent from the integration folder.

The census never writes to, moves, or deletes a source. It reads manifests and, for
colliding seeds only, the two colliding weight files.

Run:
    cd .task3 && python scripts/integration.py census \\
        --integration-root .. --analysis-root ~/cviaf-analysis \\
        --out runs/integration_census.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CENSUS_SCHEMA = "cviaf.integration-census.v1"

#: The lab boxes' detector-seed range. A non-lab source inside this range means the
#: boxes' shards are not disjoint from the rest of the corpus and the merge must stop.
LAB_SEED_MIN = 60152
LAB_SEED_MAX = 100151
LAB_BOXES = 8
LAB_BOX_SIZE = 5000

#: Declared ground truth for the consolidated root archives: name -> (sha256, bytes,
#: manifests). A mismatch is a hard failure, never a note.
EXPECTED_ARCHIVES: Dict[str, Tuple[str, int, int]] = {
    "fleet-all-6250.tar.gz": (
        "01aee38f3c0e36da6570d6b87080b782343bfddddb767f793a4eb0b7332f9b19",
        123618181,
        6250,
    ),
    "git-1056.tar.gz": (
        "eceba2ce322e88fae5032371e95801ef24523c281f0ff3af364c7515492a038e",
        20924212,
        1056,
    ),
}

#: Parts of the fleet archive, in concatenation order.
FLEET_PARTS = [f"fleet-all-6250part_{i:02d}" for i in range(5)]

_SEED_RE = re.compile(r"_s(\d+)$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_of(model_id: str) -> Optional[int]:
    """Detector seed encoded in a model id (`clean_none_fixed_s60152` -> 60152)."""
    match = _SEED_RE.search(model_id)
    return int(match.group(1)) if match else None


def seed_ranges(seeds: Iterable[int]) -> List[List[int]]:
    """Contiguous [start, end] runs of a seed set. Gaps stay visible."""
    ordered = sorted(set(seeds))
    if not ordered:
        return []
    runs: List[List[int]] = []
    start = prev = ordered[0]
    for seed in ordered[1:]:
        if seed == prev + 1:
            prev = seed
        else:
            runs.append([start, prev])
            start = prev = seed
    runs.append([start, prev])
    return runs


def duplicate_values(values: Sequence[Any]) -> List[Any]:
    seen: Dict[Any, int] = {}
    for value in values:
        seen[value] = seen.get(value, 0) + 1
    return sorted((k for k, n in seen.items() if n > 1), key=str)


def entry_from_dir(model_dir: Path) -> Optional[Dict[str, Any]]:
    """One model directory -> a census entry, or None if it is not a model dir."""
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    model_id = manifest.get("model_id") or model_dir.name
    seed = (manifest.get("seeds") or {}).get("detector")
    if not isinstance(seed, int) or isinstance(seed, bool):
        seed = seed_of(str(model_id))
    kind = (manifest.get("ground_truth") or {}).get("kind")
    return {
        "model_id": str(model_id),
        "seed": seed,
        "kind": kind,
        "rel": str(model_dir),
        "abs_dir": str(model_dir),
        "weights_present": (model_dir / "weights.npz").is_file(),
    }


def scan_dir(root: Path, base: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Every model directory under ``root``, keyed by where it sits relative to base."""
    root = Path(root)
    if not root.is_dir():
        return []
    entries: List[Dict[str, Any]] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        entry = entry_from_dir(manifest_path.parent)
        if entry is None:
            continue
        if base is not None:
            try:
                entry["rel"] = str(manifest_path.parent.relative_to(base))
            except ValueError:
                pass
        # keep the absolute location so a collision can be byte-compared without
        # re-deriving paths from a relative label
        entry["abs_dir"] = str(manifest_path.parent)
        entries.append(entry)
    return entries


def scan_archive(path: Path, base: str = "box-corpus") -> List[Dict[str, Any]]:
    """Models inside a tar, derived from member names (bytes are verified later).

    The census deliberately does not extract: counting members and parsing the seed
    out of the model directory name is enough to size the source and detect overlaps.
    `integration.py verify` is what hashes every one of these files.
    """
    entries: List[Dict[str, Any]] = []
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith("/manifest.json"):
                continue
            parts = member.name.split("/")
            model_dir = parts[-2]
            seed = seed_of(model_dir)
            rel = "/".join(parts[1:-1]) if parts[0] == base else "/".join(parts[:-1])
            entries.append(
                {
                    "model_id": model_dir,
                    "seed": seed,
                    "kind": None,
                    "rel": rel,
                    "weights_present": None,
                }
            )
    return entries


def summarize(source: Dict[str, Any]) -> Dict[str, Any]:
    """Per-source counts, seed range and duplicate bookkeeping."""
    entries = source["entries"]
    seeds = [e["seed"] for e in entries if isinstance(e.get("seed"), int)]
    ids = [e["model_id"] for e in entries]
    kinds: Dict[str, int] = {}
    for entry in entries:
        kind = entry.get("kind") or "unknown"
        kinds[kind] = kinds.get(kind, 0) + 1
    summary = {
        "id": source["id"],
        "kind": source["kind"],
        "role": source.get("role", "input"),
        "path": source["path"],
        "n_models": len(entries),
        "n_seeded": len(seeds),
        "seed_min": min(seeds) if seeds else None,
        "seed_max": max(seeds) if seeds else None,
        "seed_ranges": seed_ranges(seeds),
        "n_unparsed_seeds": len(entries) - len(seeds),
        "duplicate_seeds": duplicate_values(seeds),
        "duplicate_model_ids": duplicate_values(ids),
        "n_missing_weights": sum(1 for e in entries if e.get("weights_present") is False),
        "kinds": kinds,
    }
    summary.update(source.get("meta", {}))
    return summary


def _in_lab_range(seed: Optional[int]) -> bool:
    return isinstance(seed, int) and LAB_SEED_MIN <= seed <= LAB_SEED_MAX


def find_collisions(
    sources: Sequence[Dict[str, Any]], selected: Optional[Sequence[str]] = None
) -> List[Dict[str, Any]]:
    """Seeds claimed by more than one *selected* source, with a byte verdict.

    A lab plan and the lab corpus it was cut from describe the same 5,000 models; so do
    a corpus and its staged copy. Those are not second claims, and comparing them would
    bury the one collision that matters under 30,000 that do not. `selected` is therefore
    the set of input sources, and a byte verdict is only computed when two of them claim
    the same seed (hashing is the expensive part).
    """
    selected_set = set(selected) if selected is not None else None
    by_seed: Dict[int, List[Dict[str, Any]]] = {}
    for source in sources:
        for entry in source["entries"]:
            if isinstance(entry.get("seed"), int):
                by_seed.setdefault(entry["seed"], []).append({**entry, "source": source["id"]})
    collisions = []
    for seed, claims in sorted(by_seed.items()):
        distinct = sorted({c["source"] for c in claims})
        if len(distinct) < 2:
            continue
        selected_claims = [
            c for c in claims if selected_set is None or c["source"] in selected_set
        ]
        selected_sources = sorted({c["source"] for c in selected_claims})
        digests: Dict[str, Optional[str]] = {}
        if len(selected_sources) >= 2:
            for claim in selected_claims:
                path = Path(claim.get("abs_dir") or claim["rel"]) / "weights.npz"
                try:
                    digests[claim["source"]] = sha256_file(path)
                except OSError:
                    digests[claim["source"]] = None
            known = [d for d in digests.values() if d]
            if known and len(known) == len(digests) and len(set(known)) == 1:
                verdict = "identical_weights"
            elif known and len(known) == len(digests):
                verdict = "different_weights"
            else:
                verdict = "unresolved"
        else:
            # Nothing to compare: the other claim is a copy or a derived view.
            verdict = "not_compared"
        collisions.append(
            {
                "seed": seed,
                "sources": distinct,
                "selected_sources": selected_sources,
                "model_ids": sorted({c["model_id"] for c in claims}),
                "weights_sha256": digests,
                "verdict": verdict,
            }
        )
    return collisions


def judge(report: Dict[str, Any], selected: Sequence[str]) -> Dict[str, Any]:
    """The fail-closed verdict. `selected` is the set of sources that would be merged."""
    problems: List[str] = []
    warnings: List[str] = []
    stop = False

    for archive in report["archives"]:
        if archive.get("declared") and not archive.get("digest_ok"):
            problems.append(
                f"archive {archive['name']}: sha256 {archive.get('sha256')} != declared "
                f"{archive['declared']}"
            )
        if archive.get("n_models_expected") is not None and archive.get(
            "n_manifests"
        ) != archive["n_models_expected"]:
            problems.append(
                f"archive {archive['name']}: {archive.get('n_manifests')} manifests != "
                f"declared {archive['n_models_expected']}"
            )

    for summary in report["sources"]:
        if summary["n_models"] == 0:
            problems.append(f"source {summary['id']}: empty (0 manifests)")
        if summary["duplicate_seeds"]:
            problems.append(
                f"source {summary['id']}: duplicate detector seeds "
                f"{summary['duplicate_seeds'][:5]} inside one source"
            )
        if summary["duplicate_model_ids"]:
            problems.append(
                f"source {summary['id']}: duplicate model ids "
                f"{summary['duplicate_model_ids'][:5]} inside one source"
            )
        if summary["n_unparsed_seeds"]:
            warnings.append(
                f"source {summary['id']}: {summary['n_unparsed_seeds']} models without a "
                "readable detector seed"
            )
        if summary["n_missing_weights"]:
            warnings.append(
                f"source {summary['id']}: {summary['n_missing_weights']} model dirs have "
                "no weights.npz (census counts manifests; verify will quarantine them)"
            )

    non_lab_inside_lab = []
    for summary in report["sources"]:
        if summary["kind"] in ("lab_corpus", "lab_plan", "lab_results"):
            continue
        if summary.get("role") != "input":
            continue
        low, high = summary["seed_min"], summary["seed_max"]
        if low is None or high is None:
            continue
        if _in_lab_range(low) or _in_lab_range(high) or (
            low < LAB_SEED_MIN < high
        ):
            non_lab_inside_lab.append(
                {
                    "source": summary["id"],
                    "seed_min": low,
                    "seed_max": high,
                    "lab_range": [LAB_SEED_MIN, LAB_SEED_MAX],
                }
            )
    if non_lab_inside_lab:
        stop = True
        for row in non_lab_inside_lab:
            problems.append(
                f"source {row['source']} spans the lab seed range "
                f"{row['seed_min']}..{row['seed_max']} (lab {LAB_SEED_MIN}-{LAB_SEED_MAX}): "
                "STOP, do not dedupe"
            )

    selected_set = set(selected)
    fatal_collisions = []
    for collision in report["collisions"]:
        inside = [s for s in collision["sources"] if s in selected_set]
        if len(inside) < 2:
            warnings.append(
                f"seed {collision['seed']} claimed by {collision['sources']} "
                f"(only {inside or 'none'} selected): stale copy, not merged"
            )
            continue
        if collision["verdict"] == "identical_weights":
            warnings.append(
                f"seed {collision['seed']} claimed by {inside} with byte-identical "
                "weights: re-manifest of one model, not a distinct model"
            )
        else:
            fatal_collisions.append(collision)
            problems.append(
                f"seed {collision['seed']} claimed by {inside} with "
                f"{collision['verdict']}: two different models, cannot both be scored"
            )

    report["collisions_fatal"] = fatal_collisions
    report["stop_on_lab_overlap"] = stop
    report["problems"] = problems
    report["warnings"] = warnings
    if stop:
        report["exit_code"] = 3
    elif problems:
        report["exit_code"] = 2 if any("archive" in p for p in problems) else 4
    else:
        report["exit_code"] = 0
    report["ok"] = not problems
    return report


def render(report: Dict[str, Any]) -> str:
    lines = [f"CVIAF integration census ({CENSUS_SCHEMA})", ""]
    lines.append(f"{'source':<28}{'kind':<14}{'n':>7}  {'seeds':<18}{'ranges':>7}")
    for summary in report["sources"]:
        rng = f"{summary['seed_min']}..{summary['seed_max']}"
        lines.append(
            f"{summary['id']:<28}{summary['kind']:<14}{summary['n_models']:>7}  "
            f"{rng:<18}{len(summary['seed_ranges']):>7}"
        )
    lines.append("")
    for archive in report["archives"]:
        state = "ok" if archive.get("digest_ok") else "MISMATCH"
        lines.append(
            f"archive {archive['name']}: {archive.get('n_manifests')} manifests, "
            f"sha256 {state}"
            + (f" (declared {archive['declared'][:16]}...)" if archive.get("declared") else "")
        )
    if report.get("gaps"):
        lines.append("")
        lines.append("gaps:")
        for gap in report["gaps"]:
            lines.append(f"  - {gap['what']}: {gap['why']}")
    if report.get("warnings"):
        lines.append("")
        lines.append("warnings:")
        for warning in report["warnings"]:
            lines.append(f"  - {warning}")
    if report.get("problems"):
        lines.append("")
        lines.append("problems:")
        for problem in report["problems"]:
            lines.append(f"  - {problem}")
    lines.append("")
    lines.append(
        f"verdict: {'PASS' if report['ok'] else 'FAIL'} (exit {report['exit_code']}) "
        f"collisions={len(report['collisions'])} "
        f"fatal={len(report['collisions_fatal'])}"
    )
    return "\n".join(lines)


def discover_sources(
    integration_root: Path,
    analysis_root: Optional[Path],
    include_staged: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Enumerate archives, directory sources and gaps from the integration folder."""
    integration_root = Path(integration_root)
    cviaf = integration_root / "models_results" / "cviaf"
    runs = integration_root / "runs"
    sources: List[Dict[str, Any]] = []
    archives: List[Dict[str, Any]] = []
    gaps: List[Dict[str, Any]] = []

    # 1. Consolidated root archives (already reassembled by the caller).
    for name, (digest, size, count) in EXPECTED_ARCHIVES.items():
        path = integration_root / name
        record = {
            "name": name,
            "path": str(path),
            "bytes": path.stat().st_size if path.is_file() else None,
            "declared": digest,
            "declared_bytes": size,
            "n_models_expected": count,
            "sha256": None,
            "digest_ok": False,
            "n_manifests": 0,
        }
        if path.is_file():
            record["sha256"] = sha256_file(path)
            record["digest_ok"] = record["sha256"] == digest
            record["n_manifests"] = sum(
                1 for n in _tar_names(path) if n.endswith("/manifest.json")
            )
            sources.append(
                {
                    "id": name.replace(".tar.gz", ""),
                    "kind": "root_archive",
                    "role": "input",
                    "path": str(path),
                    "entries": scan_archive(path, base="models"),
                    "meta": {"from_archive": name},
                }
            )
        else:
            gaps.append({"what": name, "why": "reassembled archive absent"})
        archives.append(record)

    # 2. Lab box corpora (one tar per box = one 5,000-model run).
    if cviaf.is_dir():
        for tar_path in sorted(cviaf.glob("box*.tar.gz")):
            entries = scan_archive(tar_path)
            sources.append(
                {
                    "id": tar_path.name.replace(".tar.gz", ""),
                    "kind": "lab_corpus",
                    "role": "input",
                    "path": str(tar_path),
                    "entries": entries,
                    "meta": {"from_archive": tar_path.name},
                }
            )
        for plan_dir in sorted(p for p in cviaf.glob("box-plan*") if p.is_dir()):
            entries: List[Dict[str, Any]] = []
            for shard_path in sorted((plan_dir / "shards").glob("*.json")):
                shard = json.loads(shard_path.read_text(encoding="utf-8"))
                for model in shard.get("models", []):
                    entries.append(
                        {
                            "model_id": model["model_id"],
                            "seed": model.get("seed"),
                            "kind": None,
                            "rel": model.get("path"),
                            "weights_present": None,
                            "abs_dir": None,
                        }
                    )
            sources.append(
                {
                    "id": plan_dir.name,
                    "kind": "lab_plan",
                    "role": "derived",
                    "path": str(plan_dir),
                    "entries": entries,
                    "meta": {"shards": sorted(p.name for p in (plan_dir / "shards").glob("*.json"))},
                }
            )
        for res_dir in sorted(p for p in cviaf.glob("full-results*") if p.is_dir()):
            entries = []
            for npz in sorted(res_dir.glob("results_seed_*.npz")):
                entries.append(
                    {
                        "model_id": npz.name,
                        "seed": None,
                        "kind": None,
                        "rel": str(npz),
                        "weights_present": None,
                    }
                )
            sources.append(
                {
                    "id": res_dir.name,
                    "kind": "lab_results",
                    "role": "derived",
                    "path": str(res_dir),
                    "entries": entries,
                    "meta": {
                        "has_npz": bool(list(res_dir.glob("results_seed_*.npz"))),
                        "has_report": bool(list(res_dir.glob("report_seed_*.json"))),
                        "has_meta": bool(list(res_dir.glob("registry_seed_*.meta.json"))),
                    },
                }
            )
    else:
        gaps.append({"what": "models_results/cviaf", "why": "absent"})

    # 3. runs/ - the Mac-local workers and the other laptop's output.
    for worker in range(1, 7):
        root = runs / f"clean_null_local_w{worker}"
        if root.is_dir():
            sources.append(
                {
                    "id": f"runs/{root.name}",
                    "kind": "mac_local",
                    "role": "input",
                    "path": str(root),
                    "entries": scan_dir(root),
                }
            )
        else:
            gaps.append({"what": f"runs/{root.name}", "why": "absent"})
    for worker in range(1, 5):
        root = runs / f"clean_null_win_w{worker}"
        if root.is_dir():
            sources.append(
                {
                    "id": f"runs/{root.name}",
                    "kind": "laptop",
                    "role": "input",
                    "path": str(root),
                    "entries": scan_dir(root),
                }
            )
    laptop_copy = cviaf / "laptop models"
    if laptop_copy.is_dir():
        sources.append(
            {
                "id": "models_results/laptop models",
                "kind": "laptop_copy",
                "role": "derived",
                "path": str(laptop_copy),
                "entries": scan_dir(laptop_copy),
            }
        )

    # 4. The analysis workspace: staged corpora. Off by default - a staged corpus is a
    # copy of an input source, and listing it would double every count in the report.
    if include_staged and analysis_root is not None and Path(analysis_root).is_dir():
        for corpus in sorted(p for p in Path(analysis_root).glob("*-corpus") if p.is_dir()):
            sources.append(
                {
                    "id": f"staged/{corpus.name}",
                    "kind": "staged_corpus",
                    "role": "copy",
                    "path": str(corpus),
                    "entries": scan_dir(corpus),
                }
            )

    # 5. Lab boxes that are simply not here. Fail loudly rather than shrink silently.
    have = {
        s["seed_min"]
        for s in (summarize(src) for src in sources)
        if s["kind"] == "lab_plan" and s["seed_min"] is not None
    }
    for index in range(LAB_BOXES):
        first = LAB_SEED_MIN + index * LAB_BOX_SIZE
        if first not in have:
            gaps.append(
                {
                    "what": f"lab box {index + 1} (seeds {first}..{first + LAB_BOX_SIZE - 1})",
                    "why": "no matching box plan in models_results/cviaf",
                }
            )
    return sources, archives, gaps


def _tar_names(path: Path) -> List[str]:
    with tarfile.open(path, "r:gz") as tar:
        return tar.getnames()


def build_report(
    sources: Sequence[Dict[str, Any]],
    archives: Sequence[Dict[str, Any]],
    gaps: Sequence[Dict[str, Any]],
    selected: Sequence[str],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "schema": CENSUS_SCHEMA,
        "lab_seed_range": [LAB_SEED_MIN, LAB_SEED_MAX],
        "sources": [summarize(src) for src in sources],
        "archives": list(archives),
        "gaps": list(gaps),
        "collisions": find_collisions(sources, selected),
        "selected": list(selected),
    }
    return judge(report, selected)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    census = sub.add_parser("census", help="count and cross-check every source")
    census.add_argument("--integration-root", default=str(Path(__file__).resolve().parents[2]))
    census.add_argument("--analysis-root", default=str(Path.home() / "cviaf-analysis"))
    census.add_argument("--out", default=None, help="write the JSON report here")
    census.add_argument(
        "--selected",
        nargs="*",
        default=None,
        help="source ids that would actually be merged (default: every input source)",
    )
    census.add_argument(
        "--include-staged",
        action="store_true",
        help="also list staged corpora (copies of inputs; off by default)",
    )
    args = parser.parse_args(argv)

    sources, archives, gaps = discover_sources(
        Path(args.integration_root),
        Path(args.analysis_root) if args.analysis_root else None,
        include_staged=args.include_staged,
    )
    if args.selected is None:
        selected = [s["id"] for s in sources if s.get("role") == "input"]
    else:
        selected = args.selected
    report = build_report(sources, archives, gaps, selected)
    print(render(report))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return int(report["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
