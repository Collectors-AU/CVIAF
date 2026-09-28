"""Manifest and registry validation — the gate that stops a silently-wrong asset.

Every downstream number in this project is a function of the models in a corpus, and
every model is described by a manifest plus one artefact file. When either is wrong the
failure is *silent*: the loader succeeds, the battery runs, and the number is garbage.

Three failures actually happened in this lane, and each has a check here:

  1. A hand-built tamper arm was saved through the base-class `save`, producing a
     ``weights.npz`` without the real-backbone marker. ``ModelArtifact.load`` correctly
     refused it — but only because a human ran it. ``validate_model_dir`` refuses it by
     construction, before any battery sees the corpus.
  2. A hand-built arm carried no ``behaviour_divergence``, so the battery crashed
     midway through a 48-model run (``KeyError: 'f1_relative_drop'``) after minutes of
     compute. The contract is checked here instead.
  3. A trainer run wrote its own ``registry.jsonl`` over the whole file, silently
     dropping 24 previously-registered models. ``validate_registry`` cross-checks the
     registry against the manifests on disk, so a registry that has lost models fails
     loudly.

The validator is deliberately dependency-free (no jsonschema) to match
``cviaf/governance/schema.py``: the deployment target is air-gapped and the dependency
list is a supply chain. JSON-Schema keyword coverage is the same documented subset, plus
the cross-file checks a generic schema cannot express.

CLI
---
    python -m cviaf.lab.manifest_schema --corpus runs/real_cifar [--json out.json]
                                        [--strict]

Exit: 0 valid, 1 problems found, 2 corpus missing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence

from cviaf.lab.poison import ATTACK_KINDS

# The keys `cviaf/lab/train.py` and `scripts/train_real_backbone.py` both promise.
REQUIRED_KEYS = (
    "lab_version", "model_id", "created_utc", "spec", "spec_digest",
    "dataset_digests", "attack_digest", "seeds", "artifact", "ground_truth",
    "metrics", "quality_flags",
)

HEX16 = re.compile(r"^[0-9a-f]{16}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
# stampfree is a lane-local kind that `poison.ATTACK_KINDS` does not carry.
EXTRA_KINDS = ("stampfree",)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _finite(obj: Any, path: str = "manifest") -> List[str]:
    """Recursively reject NaN/Inf — they serialise to invalid JSON and poison means."""
    problems: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            problems += _finite(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            problems += _finite(v, f"{path}[{i}]")
    elif isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            problems.append(f"{path} is not finite ({obj!r})")
    return problems


def validate_manifest(manifest: Dict[str, Any], model_id: Optional[str] = None,
                      artefact_dir: Optional[str] = None) -> List[str]:
    """Return a list of problems; empty means valid. Never raises."""
    problems: List[str] = []
    if not isinstance(manifest, dict):
        return ["manifest is not a JSON object"]

    for key in REQUIRED_KEYS:
        if key not in manifest:
            problems.append(f"missing required key {key!r}")

    mid = manifest.get("model_id")
    if not isinstance(mid, str) or not mid:
        problems.append("model_id must be a non-empty string")
    elif model_id is not None and mid != model_id:
        # A manifest whose id disagrees with its directory is how two different
        # artefacts end up sharing one digest in a report.
        problems.append(f"model_id {mid!r} does not match its directory {model_id!r}")

    for key in ("spec_digest", "attack_digest"):
        v = manifest.get(key)
        if not isinstance(v, str) or not HEX16.match(v):
            problems.append(f"{key} must be 16 lowercase hex chars, got {v!r}")

    dd = manifest.get("dataset_digests")
    if not isinstance(dd, dict) or not dd:
        problems.append("dataset_digests must be a non-empty object")
    else:
        for name, digest in dd.items():
            if not isinstance(digest, str) or not HEX64.match(digest):
                problems.append(
                    f"dataset_digests[{name!r}] must be 64 lowercase hex chars, got "
                    f"{str(digest)[:20]!r}")

    art = manifest.get("artifact")
    if not isinstance(art, dict):
        problems.append("artifact must be an object")
    else:
        for key in ("weights_digest", "head_digest"):
            v = art.get(key)
            if not isinstance(v, str) or not HEX64.match(v):
                problems.append(f"artifact.{key} must be 64 hex chars, got "
                                f"{str(v)[:20]!r}")

    gt = manifest.get("ground_truth")
    if not isinstance(gt, dict):
        problems.append("ground_truth must be an object")
        kind = None
    else:
        kind = gt.get("kind")
        known = set(ATTACK_KINDS) | set(EXTRA_KINDS)
        if kind not in known:
            problems.append(f"ground_truth.kind {kind!r} is not a known attack kind "
                            f"(known: {sorted(known)})")

    metrics = manifest.get("metrics")
    qf = manifest.get("quality_flags")
    if not isinstance(metrics, dict):
        problems.append("metrics must be an object")
    else:
        cq = metrics.get("clean_quality")
        if not isinstance(cq, dict):
            problems.append("metrics.clean_quality must be an object")
        else:
            for key in ("precision", "recall", "f1"):
                if key in cq and not _is_num(cq[key]):
                    problems.append(f"metrics.clean_quality.{key} must be numeric")
            if _is_num(cq.get("f1")) and not 0.0 <= float(cq["f1"]) <= 1.0:
                problems.append(f"metrics.clean_quality.f1 out of [0,1]: {cq['f1']}")
        if not isinstance(metrics.get("attack_success_rate"), dict):
            problems.append("metrics.attack_success_rate must be an object")
    if not isinstance(qf, dict):
        problems.append("quality_flags must be an object")

    # ---- the weight-space arm contract (failure 2 above)
    from cviaf.lab.poison import MODEL_ATTACK_KINDS
    if kind in MODEL_ATTACK_KINDS:
        if isinstance(qf, dict) and qf.get("is_model_attack") is not True:
            problems.append("a weight-space arm must set quality_flags."
                            "is_model_attack = true")
        bd = metrics.get("behaviour_divergence") if isinstance(metrics, dict) else None
        if not isinstance(bd, dict) or "f1_relative_drop" not in bd:
            problems.append("a weight-space arm must record "
                            "metrics.behaviour_divergence.f1_relative_drop (the "
                            "battery reads it and crashes without it)")
        elif not _is_num(bd["f1_relative_drop"]):
            problems.append("behaviour_divergence.f1_relative_drop must be numeric")
        if isinstance(qf, dict) and "model_effect_weak" not in qf:
            problems.append("a weight-space arm must record "
                            "quality_flags.model_effect_weak")

    # ---- the artefact contract (failure 1 above): cross-file, not schema-able
    if artefact_dir is not None:
        weights = os.path.join(artefact_dir, "weights.npz")
        if not os.path.isfile(weights):
            problems.append("weights.npz is missing from the model directory")
        else:
            spec = manifest.get("spec")
            if isinstance(spec, dict) and spec.get("kind") == "real_backbone":
                try:
                    from cviaf.lab.real_backbone import is_real_backbone_artifact
                    if not is_real_backbone_artifact(weights):
                        problems.append(
                            "spec.kind is 'real_backbone' but weights.npz lacks the "
                            "real-backbone _meta marker — loading it would silently "
                            "rebuild a random backbone and keep the trained head")
                except Exception as exc:            # torch absent: cannot check
                    problems.append(f"[unchecked] real-backbone marker verification "
                                    f"unavailable: {type(exc).__name__}")

    problems += _finite(manifest)
    return problems


def validate_model_dir(model_dir: str) -> List[str]:
    mpath = os.path.join(model_dir, "manifest.json")
    if not os.path.isfile(mpath):
        return [f"{model_dir}: no manifest.json"]
    try:
        with open(mpath, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception as exc:
        return [f"{model_dir}: manifest.json is not readable JSON "
                f"({type(exc).__name__}: {exc})"]
    return [f"{os.path.basename(model_dir)}: {p}" for p in
            validate_manifest(manifest, model_id=os.path.basename(model_dir),
                              artefact_dir=model_dir)]


def read_manifest(model_dir: str) -> Optional[Dict[str, Any]]:
    """The manifest dict, or None when it is missing/unreadable."""
    mpath = os.path.join(model_dir, "manifest.json")
    if not os.path.isfile(mpath):
        return None
    try:
        with open(mpath, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def is_model_manifest(manifest: Any) -> bool:
    """True when this file is a *model asset* manifest rather than a foreign artefact.

    ``runs/drift_cells`` holds drift-cell descriptors that happen to be called
    ``manifest.json``. They are not models and must not be validated as ones — but they
    must also not be silently ignored, so a foreign schema is reported once per corpus.
    """
    if not isinstance(manifest, dict):
        return False
    if "lab_version" in manifest or "model_id" in manifest:
        return True
    return False


def scan_corpus(corpus_dir: str):
    """Split a corpus directory into model dirs and foreign-schema dirs.

    ``runs/drift_cells`` holds drift-cell descriptors that happen to be called
    ``manifest.json``. They are not models and must not be validated as ones — but they
    must not be silently ignored either, so they come back as a separate bucket rather
    than as model errors.
    """
    on_disk: List[str] = []
    foreign: Dict[str, List[str]] = {}
    for name in sorted(os.listdir(corpus_dir)):
        if not os.path.isfile(os.path.join(corpus_dir, name, "manifest.json")):
            continue
        manifest = read_manifest(os.path.join(corpus_dir, name))
        if is_model_manifest(manifest) or not isinstance(manifest, dict):
            # unreadable / non-object manifests are corruption, not foreign artefacts:
            # send them down the model path so validate_model_dir reports the parse error
            # instead of the corpus quietly dropping a model.
            on_disk.append(name)
        else:
            foreign.setdefault(str(manifest.get("schema", "unreadable")), []).append(name)
    return on_disk, foreign


def validate_registry(corpus_dir: str, strict: bool = False) -> List[str]:
    """Validate every model in a corpus, and cross-check the registry against disk.

    The registry check is what catches a trainer run that overwrote the file and
    dropped models: `load_registry` prefers the registry, so a model missing from it is
    invisible to every battery even though its manifest is still on disk.
    """
    problems: List[str] = []
    if not os.path.isdir(corpus_dir):
        return [f"{corpus_dir}: not a directory"]
    on_disk, _foreign = scan_corpus(corpus_dir)
    for name in on_disk:
        problems += validate_model_dir(os.path.join(corpus_dir, name))

    reg_path = os.path.join(corpus_dir, "registry.jsonl")
    if not os.path.isfile(reg_path):
        # A corpus with no model dirs (e.g. runs/drift_cells, which holds drift-cell
        # descriptors) has no registry to check; that is not a defect.
        if on_disk:
            problems.append("registry.jsonl is missing; make_registry_from_disk() can "
                            "rebuild it")
        return problems
    registered: List[str] = []
    seen: Dict[str, int] = {}
    with open(reg_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception as exc:
                problems.append(f"registry.jsonl:{lineno} is not valid JSON "
                                f"({type(exc).__name__})")
                continue
            mid = (entry.get("manifest") or {}).get("model_id")
            if not isinstance(mid, str):
                problems.append(f"registry.jsonl:{lineno} has no manifest.model_id")
                continue
            registered.append(mid)
            seen[mid] = seen.get(mid, 0) + 1
            if entry.get("dir") and not os.path.isdir(entry["dir"]):
                problems.append(f"registry entry {mid!r} points at a missing dir "
                                f"{entry['dir']!r}")
    for mid, count in seen.items():
        if count > 1:
            problems.append(f"registry lists {mid!r} {count} times; a duplicate "
                            f"inflates every denominator it appears in")
    missing = [m for m in on_disk if m not in seen]
    if missing:
        problems.append(f"{len(missing)} model(s) on disk are absent from "
                        f"registry.jsonl: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if strict:
        extra = [m for m in registered if m not in on_disk]
        if extra:
            problems.append(f"{len(extra)} registry entries have no manifest on disk: "
                            f"{extra[:5]}")
    return problems


def make_registry_from_disk(corpus_dir: str, write: bool = True) -> List[Dict[str, Any]]:
    """Rebuild registry.jsonl deterministically from the manifests on disk.

    The manifests are the source of truth; the registry is a cache of them. This is the
    repair path when a trainer run has overwritten the cache.
    """
    entries: List[Dict[str, Any]] = []
    on_disk, _foreign = scan_corpus(corpus_dir)
    for name in on_disk:
        manifest = read_manifest(os.path.join(corpus_dir, name))
        if manifest is not None:
            entries.append({"dir": os.path.join(corpus_dir, name),
                            "manifest": manifest})
    if write:
        path = os.path.join(corpus_dir, "registry.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e, default=str) + "\n")
    return entries


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cviaf lab manifest-validate",
                                 description="validate a corpus's manifests and registry")
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--json", default=None, help="write the problem list here")
    ap.add_argument("--strict", action="store_true",
                    help="also flag registry entries with no manifest on disk")
    ap.add_argument("--rebuild-registry", action="store_true",
                    help="repair registry.jsonl from the manifests (source of truth)")
    args = ap.parse_args(argv)

    report: Dict[str, Any] = {"schema": "cviaf.manifest-validation.v1", "corpora": {}}
    total = 0
    for corpus in args.corpus:
        if not os.path.isdir(corpus):
            print(f"{corpus}: not a directory")
            return 2
        if args.rebuild_registry:
            entries = make_registry_from_disk(corpus)
            print(f"{corpus}: registry rebuilt from disk ({len(entries)} models)")
        problems = validate_registry(corpus, strict=args.strict)
        on_disk, foreign = scan_corpus(corpus)
        report["corpora"][corpus] = {
            "problems": problems,
            "n_models": len(on_disk),
            "foreign": {s: len(n) for s, n in foreign.items()},
        }
        total += len(problems)
        status = "OK" if not problems else f"{len(problems)} problem(s)"
        print(f"{corpus}: {len(on_disk)} models — {status}")
        for schema, names in sorted(foreign.items()):
            print(f"    note: {len(names)} foreign director(s) with schema {schema!r} "
                  f"(not model assets, not validated here)")
        for p in problems[:20]:
            print(f"    {p}")
        if len(problems) > 20:
            print(f"    ... and {len(problems) - 20} more")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, default=str)
        print(f"report written to {args.json}")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
