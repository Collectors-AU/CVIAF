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
import glob
import hashlib
import json
import os
import re
import shutil
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
                    "abs_dir": None,
                    "tar_path": str(path),
                    "tar_member": "/".join(parts[:-1]) + "/weights.npz",
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


def _collect_weight_digests(
    needed: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[int, Optional[str]]]:
    """Weight digests for the colliding seeds of each source, in one pass per archive.

    A gzip tar has to be read from the start to reach a member, so hashing members one
    seed at a time would rescan 94 MB per collision. Each archive is therefore opened
    once and only the members that are actually in dispute are hashed.
    """
    out: Dict[str, Dict[int, Optional[str]]] = {}
    for source_id, claims in needed.items():
        result: Dict[int, Optional[str]] = {c["seed"]: None for c in claims}
        by_archive: Dict[str, List[Dict[str, Any]]] = {}
        for claim in claims:
            if claim.get("tar_member"):
                by_archive.setdefault(claim["tar_path"], []).append(claim)
            else:
                path = Path(claim.get("abs_dir") or claim["rel"]) / "weights.npz"
                try:
                    result[claim["seed"]] = sha256_file(path)
                except OSError:
                    result[claim["seed"]] = None
        for tar_path, archive_claims in by_archive.items():
            wanted = {c["tar_member"]: c["seed"] for c in archive_claims}
            try:
                with tarfile.open(tar_path, "r:gz") as tar:
                    for member in tar:
                        if member.name not in wanted or not member.isfile():
                            continue
                        handle = tar.extractfile(member)
                        if handle is None:
                            continue
                        digest = hashlib.sha256()
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(block)
                        result[wanted[member.name]] = digest.hexdigest()
            except (OSError, tarfile.TarError):
                pass
        out[source_id] = result
    return out


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
    disputed = []
    needed: Dict[str, List[Dict[str, Any]]] = {}
    for seed, claims in sorted(by_seed.items()):
        distinct = sorted({c["source"] for c in claims})
        if len(distinct) < 2:
            continue
        selected_claims = [
            c for c in claims if selected_set is None or c["source"] in selected_set
        ]
        selected_sources = sorted({c["source"] for c in selected_claims})
        if len(selected_sources) >= 2:
            for claim in selected_claims:
                needed.setdefault(claim["source"], []).append(claim)
        disputed.append(
            {
                "seed": seed,
                "sources": distinct,
                "selected_sources": selected_sources,
                "model_ids": sorted({c["model_id"] for c in claims}),
            }
        )

    collected = _collect_weight_digests(needed)
    collisions = []
    for row in disputed:
        seed = row["seed"]
        selected_sources = row["selected_sources"]
        digests: Dict[str, Optional[str]] = {
            source: (collected.get(source) or {}).get(seed) for source in selected_sources
        }
        if len(selected_sources) >= 2:
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
        collisions.append({**row, "weights_sha256": digests, "verdict": verdict})
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

    gaps = list(report.get("gaps", []))
    for summary in report["sources"]:
        if summary["kind"] == "lab_results":
            # A box whose results are absent has not been scored; that is a hole in the
            # population to report, not a corrupt source to fail on.
            if not summary.get("has_npz"):
                gaps.append(
                    {
                        "what": summary["id"],
                        "why": "no results_seed_*.npz (box not scored here)",
                    }
                )
            elif not (summary.get("has_report") and summary.get("has_meta")):
                gaps.append(
                    {
                        "what": summary["id"],
                        "why": "results without a report/meta pair",
                    }
                )
            continue
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
    report["gaps"] = gaps
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
    lines.append(f"{'source':<28}{'kind':<16}{'n':>7}  {'seeds':<18}{'ranges':>7}")
    for summary in report["sources"]:
        rng = f"{summary['seed_min']}..{summary['seed_max']}"
        lines.append(
            f"{summary['id']:<28}{summary['kind']:<16}{summary['n_models']:>7}  "
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
            npz_files = sorted(res_dir.glob("results_seed_*.npz"))
            ranges = []
            for npz in npz_files:
                bounds = re.findall(r"seed_(\d+)_(\d+)", npz.name)
                if bounds:
                    ranges.append([int(bounds[0][0]), int(bounds[0][1])])
            sources.append(
                {
                    "id": res_dir.name,
                    "kind": "lab_results",
                    "role": "derived",
                    "path": str(res_dir),
                    "entries": [],
                    "meta": {
                        "result_files": [p.name for p in npz_files],
                        "result_seed_ranges": ranges,
                        "has_npz": bool(npz_files),
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


def sha256_tar_member(tar_path: Path, member_name: str) -> Optional[str]:
    """SHA-256 of one member inside an archive, or None if it is not there.

    Archive-derived entries have no directory to hash, so a collision between two
    archives would otherwise come out "unresolved" - and an unresolved collision is
    indistinguishable from a real clash. Comparing the member bytes settles it.
    """
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            handle = tar.extractfile(member_name)
            if handle is None:
                return None
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            return digest.hexdigest()
    except (OSError, tarfile.TarError, KeyError):
        return None


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


SCORE_SCHEMA = "cviaf.integration-score.v1"

def classify_signals(
    values: Optional[Dict[str, Any]], signals: Sequence[str], error: Optional[str] = None
) -> str:
    """complete | incomplete | error. One undefined model is not a failed shard."""
    if error is not None or not isinstance(values, dict):
        return "error"
    usable = 0
    for name in signals:
        value = values.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value:
            usable += 1
    if usable == len(signals):
        return "complete"
    if usable:
        return "incomplete"
    return "error"


def score_records(
    items: Sequence[Dict[str, Any]],
    score_fn,
    signals: Sequence[str],
    workers: int = 1,
    resume: Optional[Dict[str, Dict[str, Any]]] = None,
    checkpoint=None,
) -> Dict[str, Any]:
    """Score models one at a time and classify rather than crash.

    `shard_worker.py` calls `future.result()` and lets the first adapter exception take
    the process down. That is how box 5 lost a 5,000-model run: one clean model whose CTC
    score is undefined everywhere (so the producer emits only `refdiv_mean_clean`) raised
    in the adapter, and 3,533 finished rows never became a results file. This keeps the
    three outcomes separate and countable - scored, incomplete, error - so the ledger can
    exclude the undefined models explicitly and say how many there were.
    """
    done = dict(resume or {})
    pending = [item for item in items if item["model_id"] not in done]
    results: List[Dict[str, Any]] = []
    if workers <= 1 or len(pending) <= 1:
        for item in pending:
            row = _score_one((item, score_fn, signals))
            results.append(row)
            if checkpoint is not None:
                done[item["model_id"]] = row
                checkpoint(done)
    else:  # pragma: no cover - exercised through the resume path, not in unit tests
        import multiprocessing as mp

        with mp.Pool(processes=workers) as pool:
            for row in pool.imap_unordered(
                _score_one, [(item, score_fn, signals) for item in pending]
            ):
                results.append(row)
                if checkpoint is not None:
                    done[row["model_id"]] = row
                    checkpoint(done)
    for row in results:
        done[row["model_id"]] = row
    ordered = [done[item["model_id"]] for item in items if item["model_id"] in done]
    scored = [r for r in ordered if r["status"] == "complete"]
    incomplete = [r for r in ordered if r["status"] == "incomplete"]
    errors = [r for r in ordered if r["status"] == "error"]
    return {
        "schema": SCORE_SCHEMA,
        "signals": list(signals),
        "n_models": len(items),
        "n_scored": len(scored),
        "n_incomplete": len(incomplete),
        "n_error": len(errors),
        "rows": ordered,
        "skipped": [
            {"model_id": r["model_id"], "status": r["status"], "reason": r["reason"]}
            for r in incomplete + errors
        ],
    }


def _score_one(args) -> Dict[str, Any]:
    item, score_fn, signals = args
    values, error = None, None
    try:
        values = score_fn(Path(item["path"]), item["manifest"])
    except BaseException as exc:  # noqa: BLE001 - the point is to survive it
        error = f"{type(exc).__name__}: {exc}"
    status = classify_signals(values, signals, error=error)
    reasons = {
        "incomplete": "scorer omitted "
        + ", ".join(sorted(set(signals) - {k for k, v in (values or {}).items() if v is not None})),
        "complete": "",
        "error": error or "no finite signals returned",
    }
    return {
        "model_id": item["model_id"],
        "seed": item["seed"],
        "kind": item.get("kind"),
        "status": status,
        "reason": reasons[status],
        "scores": values if isinstance(values, dict) else None,
    }


def scored_shard(shard: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    """The shard as it was actually scored: the excluded models are named in it.

    A merge validates shard membership against the shard file the worker used, so the
    honest way to keep a shard self-consistent is to re-issue it without the models that
    could not be scored - and to record their ids in the shard itself, not only in a log.
    """
    keep = {r["model_id"] for r in report["rows"] if r["status"] == "complete"}
    dropped = [r["model_id"] for r in report["rows"] if r["status"] != "complete"]
    models = [m for m in shard["models"] if m["model_id"] in keep]
    seeds = [m["seed"] for m in models]
    out = dict(shard)
    out["models"] = models
    out["count"] = len(models)
    if seeds:
        out["seed_min"], out["seed_max"] = min(seeds), max(seeds)
    out["unscorable_model_ids"] = dropped
    return out


PLAN_SCHEMA = "cviaf.analysis-plan.v1"
SHARD_SCHEMA = "cviaf.analysis-shard.v1"
PLAN_SUMMARY_KEYS = ("shard_id", "seed_min", "seed_max", "count", "preferred_box", "local_count")


class PlanError(ValueError):
    """A combined plan that the merge would otherwise build out of a contradiction."""


def load_shard(path: Path) -> Dict[str, Any]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("schema") != SHARD_SCHEMA:
        raise PlanError(f"{path}: schema {obj.get('schema')!r} != {SHARD_SCHEMA!r}")
    return obj


def merge_plans(
    specs: Sequence[Tuple[str, Path]],
    out_dir: Path,
    excluded: Sequence[str] = (),
    reference: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble one byte-pinned plan from per-source shards, refusing contradictions.

    Each source keeps its own shard set: the shard file the worker hashed is the shard
    file the merge validates against, byte for byte (`assemble_all_plans` relies on the
    same property). What this refuses is the set of situations where a combined plan
    would be arithmetically fine and scientifically wrong:

      * the same detector seed in two sources - the honest response is to stop, not to
        keep the first one and let the counts add up;
      * the same model id twice;
      * the same shard id twice (a stray copy of box 8's shard inside box 7's plan dir);
      * a shard that still contains the reference model, which must never score itself;
      * a shard that still contains a model this pass decided to exclude.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise PlanError(f"{out_dir} is not empty; use a fresh output directory")
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    excluded_set = set(excluded)

    seen_shard_ids: Dict[str, str] = {}
    seed_owner: Dict[int, str] = {}
    id_owner: Dict[str, str] = {}
    summaries: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []

    for source_id, shard_path in specs:
        shard = load_shard(shard_path)
        shard_id = shard["shard_id"]
        if shard_id in seen_shard_ids:
            raise PlanError(
                f"duplicate shard id {shard_id} in {seen_shard_ids[shard_id]} and {source_id}"
            )
        models = shard.get("models") or []
        if not models:
            raise PlanError(f"{source_id}: shard {shard_id} has no models")
        if shard.get("count") != len(models):
            raise PlanError(
                f"{source_id}: shard count {shard.get('count')} != {len(models)} models"
            )
        seeds = [m["seed"] for m in models]
        if min(seeds) != shard.get("seed_min") or max(seeds) != shard.get("seed_max"):
            raise PlanError(
                f"{source_id}: seed bounds {shard.get('seed_min')}..{shard.get('seed_max')} "
                f"do not match the models ({min(seeds)}..{max(seeds)})"
            )
        for model in models:
            model_id, seed = model["model_id"], model["seed"]
            if reference is not None and model_id == reference:
                raise PlanError(f"{source_id}: shard still contains the reference model {model_id}")
            if model_id in excluded_set:
                raise PlanError(f"{source_id}: shard still contains excluded model {model_id}")
            if model_id in id_owner:
                raise PlanError(
                    f"duplicate model id {model_id} in {id_owner[model_id]} and {source_id}"
                )
            if seed in seed_owner:
                raise PlanError(
                    f"duplicate detector seed {seed} in {seed_owner[seed]} and {source_id}: "
                    "refusing to dedupe"
                )
            id_owner[model_id] = source_id
            seed_owner[seed] = source_id
        target = shards_dir / f"{shard_id}.json"
        target.write_bytes(Path(shard_path).read_bytes())
        before, after = sha256_file(Path(shard_path)), sha256_file(target)
        if before != after:
            raise PlanError(f"{source_id}: copied shard bytes differ ({before} != {after})")
        seen_shard_ids[shard_id] = source_id
        summaries.append({k: shard[k] for k in PLAN_SUMMARY_KEYS if k in shard})
        sources.append(
            {
                "id": source_id,
                "shard_id": shard_id,
                "count": len(models),
                "seed_min": shard["seed_min"],
                "seed_max": shard["seed_max"],
                "shard_sha256": after,
                "from": str(shard_path),
            }
        )

    plan: Dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "corpus_count": len(id_owner),
        "unique_model_ids": len(id_owner),
        "unique_detector_seeds": len(seed_owner),
        "excluded_model_ids": sorted(excluded_set),
        "shards": summaries,
        "sources": sources,
        "scoring": "external adapter required; stub dry run is not CVIAF FPR/TPR",
    }
    (out_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return plan


def render_plan(plan: Dict[str, Any]) -> str:
    lines = [f"combined plan ({plan['schema']})", ""]
    lines.append(f"{'source':<28}{'shard':<22}{'n':>7}  seeds")
    for source in plan["sources"]:
        lines.append(
            f"{source['id']:<28}{source['shard_id']:<22}{source['count']:>7}  "
            f"{source['seed_min']}..{source['seed_max']}"
        )
    lines.append("")
    lines.append(
        f"corpus_count={plan['corpus_count']} shards={len(plan['shards'])} "
        f"excluded={plan['excluded_model_ids']}"
    )
    return "\n".join(lines)


VERIFY_SCHEMA = "cviaf.integration-verify.v1"


def verify_model_dir(
    model_dir: Path, plan_entry: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Every reason this model directory cannot be scored. Empty list = it can.

    Two independent checks, because they catch different accidents: the repo's native
    validator catches a manifest that is structurally wrong, and the plan entry catches
    a file that is structurally fine but is not the bytes the box scored (a re-export, a
    truncated copy, a partial transfer).
    """
    problems: List[str] = []
    manifest_path = model_dir / "manifest.json"
    weights_path = model_dir / "weights.npz"
    if not manifest_path.is_file():
        return ["missing manifest.json"]
    if not weights_path.is_file():
        problems.append("missing weights.npz")
    try:
        from cviaf.lab.manifest_schema import validate_model_dir

        problems.extend(validate_model_dir(str(model_dir)))
    except Exception as exc:  # noqa: BLE001 - a validator crash is a verification failure
        problems.append(f"native validator raised {type(exc).__name__}: {exc}")
    if plan_entry is not None:
        try:
            actual_manifest = sha256_file(manifest_path)
        except OSError as exc:
            actual_manifest = f"unreadable: {exc}"
        if plan_entry.get("manifest_sha256") and actual_manifest != plan_entry["manifest_sha256"]:
            problems.append(
                f"manifest_sha256 {actual_manifest} != plan {plan_entry['manifest_sha256']}"
            )
        if plan_entry.get("weights_sha256") and weights_path.is_file():
            actual_weights = sha256_file(weights_path)
            if actual_weights != plan_entry["weights_sha256"]:
                problems.append(
                    f"weights_sha256 {actual_weights} != plan {plan_entry['weights_sha256']}"
                )
    return problems


def plan_entries(plan_shards: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    """model_id -> pinned digests, from one or more shard files."""
    entries: Dict[str, Dict[str, Any]] = {}
    for shard_path in plan_shards:
        shard = json.loads(Path(shard_path).read_text(encoding="utf-8"))
        for model in shard.get("models", []):
            entries[model["model_id"]] = model
    return entries


def quarantine_dir(
    model_dir: Path, quarantine_root: Path, source_id: str, owned_root: Optional[Path]
) -> Dict[str, Any]:
    """Move a failing model directory aside. Never delete, never leave the workspace.

    Models that live in a directory this pass does not own (the user's `runs/`) are
    recorded but not moved: a verification failure is not permission to relocate
    somebody else's training output.
    """
    model_dir = Path(model_dir)
    if owned_root is None or not str(model_dir.resolve()).startswith(
        str(Path(owned_root).resolve()) + os.sep
    ):
        return {"moved": False, "reason": "in place, not owned by this pass", "target": None}
    target = Path(quarantine_root) / source_id / model_dir.name
    if target.exists():
        return {"moved": False, "reason": "quarantine target already exists", "target": str(target)}
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(model_dir, target)
    except OSError as exc:
        return {"moved": False, "reason": f"rename failed: {exc}", "target": str(target)}
    return {"moved": True, "reason": "quarantined", "target": str(target)}


def verify_source(
    source_id: str,
    root: Path,
    plan: Optional[Dict[str, Dict[str, Any]]] = None,
    quarantine_root: Optional[Path] = None,
    owned_root: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Check every model dir under one source root, quarantining what fails."""
    root = Path(root)
    models = sorted(p.parent for p in root.rglob("manifest.json"))
    if limit is not None:
        models = models[:limit]
    verified: List[str] = []
    failures: List[Dict[str, Any]] = []
    missing_from_plan: List[str] = []
    seen: set = set()
    for model_dir in models:
        entry = None
        if plan is not None:
            entry = plan.get(model_dir.name)
            seen.add(model_dir.name)
            if entry is None:
                missing_from_plan.append(model_dir.name)
        problems = verify_model_dir(model_dir, entry)
        if not problems:
            verified.append(model_dir.name)
            continue
        action = {"moved": False, "reason": "quarantine not requested", "target": None}
        if quarantine_root is not None:
            action = quarantine_dir(model_dir, quarantine_root, source_id, owned_root)
        failures.append(
            {
                "model_id": model_dir.name,
                "path": str(model_dir),
                "problems": problems,
                "quarantine": action,
            }
        )
    unaccounted = sorted(set(plan or {}) - seen)
    return {
        "id": source_id,
        "root": str(root),
        "n_models": len(models),
        "n_verified": len(verified),
        "n_failed": len(failures),
        "n_plan_entries": len(plan or {}),
        "n_missing_from_plan": len(missing_from_plan),
        "n_plan_entries_not_seen": len(unaccounted),
        "failures": failures,
        "missing_from_plan": missing_from_plan[:50],
        "plan_entries_not_seen": unaccounted[:50],
    }


def verify_judge(report: Dict[str, Any]) -> Dict[str, Any]:
    """A failed model is a hole in the population; say how big and where."""
    problems: List[str] = []
    warnings: List[str] = []
    for source in report["sources"]:
        if source["n_failed"]:
            problems.append(
                f"source {source['id']}: {source['n_failed']} of {source['n_models']} models "
                "failed verification and are excluded from the ledger"
            )
        if source["n_missing_from_plan"]:
            problems.append(
                f"source {source['id']}: {source['n_missing_from_plan']} models have no "
                "entry in the pinned plan (unscored by construction)"
            )
        if source["n_plan_entries_not_seen"]:
            problems.append(
                f"source {source['id']}: {source['n_plan_entries_not_seen']} pinned plan "
                "entries have no model on disk"
            )
        unmoved = [f for f in source["failures"] if not f["quarantine"]["moved"]]
        if unmoved:
            warnings.append(
                f"source {source['id']}: {len(unmoved)} failed models were recorded but "
                "not moved (in place, or quarantine not requested)"
            )
    report["problems"] = problems
    report["warnings"] = warnings
    report["ok"] = not problems
    report["exit_code"] = 0 if report["ok"] else 7
    return report


def render_verify(report: Dict[str, Any]) -> str:
    lines = [f"CVIAF integration verify ({VERIFY_SCHEMA})", ""]
    lines.append(f"{'source':<28}{'checked':>8}{'verified':>9}{'failed':>7}{'no plan':>8}")
    for source in report["sources"]:
        lines.append(
            f"{source['id']:<28}{source['n_models']:>8}{source['n_verified']:>9}"
            f"{source['n_failed']:>7}{source['n_missing_from_plan']:>8}"
        )
    for source in report["sources"]:
        for failure in source["failures"][:20]:
            lines.append("")
            lines.append(f"  FAIL {source['id']}/{failure['model_id']}")
            for problem in failure["problems"][:3]:
                lines.append(f"    - {problem}")
            lines.append(f"    quarantine: {failure['quarantine']}")
    if report.get("problems"):
        lines.append("")
        lines.append("problems:")
        for problem in report["problems"]:
            lines.append(f"  - {problem}")
    lines.append("")
    total = sum(s["n_models"] for s in report["sources"])
    verified = sum(s["n_verified"] for s in report["sources"])
    lines.append(
        f"verdict: {'PASS' if report['ok'] else 'FAIL'} (exit {report['exit_code']}) "
        f"verified {verified}/{total}"
    )
    return "\n".join(lines)


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
    merge = sub.add_parser("merge-plans", help="assemble one byte-pinned plan")   
    merge.add_argument("--source", action="append", required=True, help="id=shard.json")
    merge.add_argument("--out", required=True, help="output directory for plan.json + shards/")
    merge.add_argument("--exclude", action="append", default=[], help="model id to refuse")
    merge.add_argument("--reference", default=None, help="reference model id, must not be in any shard")

    verify = sub.add_parser("verify", help="validate every model dir, quarantine failures")
    verify.add_argument("--source", action="append", required=True, help="id=path")
    verify.add_argument("--plan", action="append", default=[], help="id=shard.json (or a glob)")
    verify.add_argument(
        "--owned-root",
        default=str(Path.home() / "cviaf-analysis"),
        help="only models under this root may be moved; the rest are only recorded",
    )
    verify.add_argument(
        "--quarantine",
        default=str(Path.home() / "cviaf-analysis" / "quarantine"),
        help="where failing models are moved (never deleted)",
    )
    verify.add_argument("--limit", type=int, default=None, help="check at most N models per source")
    verify.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if args.command == "merge-plans":
        specs = []
        for spec in args.source:
            source_id, _, shard_path = spec.partition("=")
            if not shard_path:
                print(f"merge-plans: bad --source {spec!r} (want id=shard.json)", file=sys.stderr)
                return 8
            specs.append((source_id, Path(shard_path)))
        try:
            plan = merge_plans(
                specs, Path(args.out), excluded=args.exclude, reference=args.reference
            )
        except PlanError as exc:
            print(f"merge-plans: REFUSED: {exc}", file=sys.stderr)
            return 9
        print(render_plan(plan))
        return 0

    if args.command == "verify":
        plans: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for spec in args.plan:
            source_id, _, pattern = spec.partition("=")
            shards = [Path(p) for p in sorted(glob.glob(pattern))]
            if not shards:
                print(f"verify: no shard files matched {pattern!r}", file=sys.stderr)
                return 8
            plans[source_id] = plan_entries(shards)
        sources = []
        for spec in args.source:
            source_id, _, root = spec.partition("=")
            if not root:
                print(f"verify: bad --source {spec!r} (want id=path)", file=sys.stderr)
                return 8
            sources.append(
                verify_source(
                    source_id,
                    Path(root),
                    plan=plans.get(source_id),
                    quarantine_root=Path(args.quarantine) if args.quarantine else None,
                    owned_root=Path(args.owned_root) if args.owned_root else None,
                    limit=args.limit,
                )
            )
        report = verify_judge(
            {"schema": VERIFY_SCHEMA, "sources": sources, "quarantine": args.quarantine}
        )
        print(render_verify(report))
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return int(report["exit_code"])

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
