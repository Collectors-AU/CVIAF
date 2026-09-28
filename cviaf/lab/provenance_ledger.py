"""The provenance ledger: a hash-chained record of how every number was produced.

Every measurement in this lane is a function of a corpus, a command and a set of
output files, and until now that function lived only in a shell history. A report
could say "asset_tpr 0.1458" with nothing binding it to the 48 arms, the probe
digest, or the commit that produced it, and when the corpus later grew there was no
way to tell which numbers described which corpus. That is the exact failure mode
that left a stale headline in the tree for a day.

So each measurement appends one record: what ran, on which inputs, producing which
outputs, at which commit, with the sha256 of every file it names. The records are
hash-chained with the same discipline as the audit trail (``prev_hash``, canonical
JSON, domain-separated sha256), which buys three properties:

  * the ledger is append-only in a checkable sense -- editing, deleting or
    reordering any record breaks verification at the first affected index;
  * a claim can be re-derived -- ``verify(check_artefacts=True)`` re-hashes the
    files the ledger names, so an artefact that changed after it was recorded is
    caught rather than silently trusted;
  * the *absence* of a record is visible. A number with no ledger entry is not
    reproducible by construction, and the corpus lane can require one before it
    publishes.

This is deliberately not the signed checkpoint path (``cviaf/provenance``): signing
needs an operator key and answers "who attested this", while this answers "what
produced this and can I still reproduce it". The two compose -- the ledger digest is
the thing you sign.

CLI
---
    python -m cviaf.lab.provenance_ledger init --path runs/provenance_ledger.jsonl
    python -m cviaf.lab.provenance_ledger append --path L --event battery \\
        --command "python scripts/battery_model_attacks.py ..." \\
        --input corpus=runs/real_cifar --output result=runs/real_cifar/battery.json
    python -m cviaf.lab.provenance_ledger verify --path L [--check-artefacts]

Exit: 0 valid, 1 broken or missing.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

LEDGER_SCHEMA = "cviaf.provenance-ledger.v1"
GENESIS = "GENESIS"
DOMAIN = b"CVIAF provenance ledger v1\x00"
HEXDIGITS = set("0123456789abcdef")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical(obj: Any) -> bytes:
    """Canonical JSON, refusing NaN/Inf: they cannot round-trip and must not enter."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: str, chunk: int = 1 << 20) -> Dict[str, Any]:
    """Digest a file by streaming it, so a multi-GB corpus does not enter memory."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            size += len(block)
            h.update(block)
    return {"path": path, "sha256": h.hexdigest(), "bytes": size}


def tree_digest(path: str, exclude: Sequence[str] = (),
                max_bytes: int = 8 << 30) -> Dict[str, Any]:
    """Content digest of a directory tree, stable across checkouts.

    A corpus is a directory, and the corpus is exactly what changed while a stale
    headline sat in the tree -- so the ledger has to be able to name one. The digest
    is over sorted relative paths plus each file's content hash and size: no mtimes,
    no inode order, so a re-clone digests identically but any added, removed, renamed
    or edited file changes it.

    ``exclude`` holds fnmatch patterns for files that live inside the corpus but are
    not part of it -- the analysis outputs written into the corpus root. Without it a
    corpus digest would go stale the moment the next battery writes its result next to
    the models, and every real corpus change would be indistinguishable from that.
    """
    h = hashlib.sha256()
    h.update(b"TREE\x00" + canonical(list(exclude)) + b"\x00")
    files = 0
    total = 0
    stack = [path]
    while stack:
        node = stack.pop()
        for name in sorted(os.listdir(node)):
            child = os.path.join(node, name)
            rel = os.path.relpath(child, path).replace(os.sep, "/")
            if os.path.isdir(child):
                stack.append(child)
                continue
            if any(fnmatch.fnmatch(rel, pat) for pat in exclude):
                continue
            size = os.path.getsize(child)
            total += size
            if total > max_bytes:
                raise ValueError(f"refusing to digest more than {max_bytes} bytes under "
                                 f"{path}")
            h.update(b"F\x00" + rel.encode() + b"\x00" + str(size).encode() + b"\x00"
                     + bytes.fromhex(file_digest(child)["sha256"]))
            files += 1
    return {"path": path, "kind": "tree", "sha256": h.hexdigest(),
            "files": files, "bytes": total, "exclude": list(exclude)}


def describe_artifact(path: str, exclude: Sequence[str] = ()) -> Dict[str, Any]:
    """Digest a file or a directory, so a corpus and a result are recorded alike."""
    if os.path.isdir(path):
        return tree_digest(path, exclude=exclude)
    meta = file_digest(path)
    meta["kind"] = "file"
    return meta


def entry_digest(record: Mapping[str, Any]) -> str:
    """Hash of everything except ``entry_hash``, domain-separated."""
    body = {k: v for k, v in record.items() if k != "entry_hash"}
    return sha256_bytes(DOMAIN + canonical(body))


def _digest_artefacts(artefacts: Mapping[str, str],
                      base_dir: Optional[str] = None,
                      excludes: Optional[Mapping[str, Sequence[str]]] = None) -> Dict[str, Any]:
    """Hash each artefact and record both the path as given and the resolved one.

    Relative paths are the normal case on the command line, and a ledger whose paths
    only resolve from the directory they were written in cannot be verified later.
    So the absolute path is recorded next to the readable one.
    """
    out: Dict[str, Any] = {}
    base = base_dir or os.getcwd()
    excludes = excludes or {}
    for role, path in sorted(artefacts.items()):
        resolved = path if os.path.isabs(path) else os.path.join(base, path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"artefact {role!r} does not exist: {path}")
        meta = describe_artifact(resolved, exclude=excludes.get(role, ()))
        meta["path"] = path
        meta["resolved"] = os.path.abspath(resolved)
        out[role] = meta
    return out


def make_record(seq: int, prev_hash: str, event: str, command: str = "",
                payload: Optional[Mapping[str, Any]] = None,
                artefacts: Optional[Mapping[str, str]] = None,
                git_commit: Optional[str] = None, ts: Optional[str] = None,
                base_dir: Optional[str] = None,
                excludes: Optional[Mapping[str, Sequence[str]]] = None) -> Dict[str, Any]:
    """Build one chained record. Does not write it."""
    if not isinstance(event, str) or not event.strip():
        raise ValueError("event must be a non-empty string")
    record: Dict[str, Any] = {
        "seq": seq,
        "ts_utc": ts or utc_now(),
        "event": event,
        "command": command,
        "git_commit": git_commit if git_commit is not None else current_commit(),
        "artefacts": _digest_artefacts(artefacts or {}, base_dir=base_dir,
                                       excludes=excludes),
        "payload": dict(payload or {}),
        "prev_hash": prev_hash,
    }
    # allow_nan=False here is the point: a NaN in a payload would serialise to
    # invalid JSON and make every later verification meaningless.
    record["entry_hash"] = entry_digest(record)
    return record


def current_commit(cwd: Optional[str] = None) -> Optional[str]:
    """The commit the record is being written at, or None outside a repo.

    A number that cannot be attributed to a commit cannot be re-derived, so this is
    recorded rather than assumed. ``--dirty`` is not recorded; the file digests are
    what prove the inputs, and a clean tree is a weaker claim than a digest.
    """
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def read_ledger(path: str) -> List[Dict[str, Any]]:
    """All lines as parsed JSON. Raises ValueError on a malformed file."""
    with open(path, encoding="utf-8") as fh:
        raw = [line for line in fh.read().splitlines() if line.strip()]
    if not raw:
        raise ValueError(f"{path}: ledger is empty")
    out = []
    for i, line in enumerate(raw):
        try:
            out.append(json.loads(line))
        except Exception as exc:
            raise ValueError(f"{path}:{i + 1} is not valid JSON ({type(exc).__name__})") from exc
    return out


def init_ledger(path: str, note: str = "") -> Dict[str, Any]:
    """Create a ledger with a header line. Refuses to clobber an existing one."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        raise FileExistsError(f"{path} already exists; the ledger is append-only")
    header = {"schema": LEDGER_SCHEMA, "created_utc": utc_now(), "note": note,
              "hash": "sha256", "domain": DOMAIN.decode("utf-8", "replace").strip("\x00")}
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, sort_keys=True) + "\n")
    return header


def append(path: str, event: str, command: str = "",
           payload: Optional[Mapping[str, Any]] = None,
           artefacts: Optional[Mapping[str, str]] = None, ts: Optional[str] = None,
           base_dir: Optional[str] = None,
           excludes: Optional[Mapping[str, Sequence[str]]] = None) -> Dict[str, Any]:
    """Append one record, chained onto the current tail. Creates the ledger if absent."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        init_ledger(path)
    records = read_ledger(path)
    header = records[0]
    if header.get("schema") != LEDGER_SCHEMA:
        raise ValueError(f"{path}: unknown ledger schema {header.get('schema')!r}")
    entries = records[1:]
    # Refuse to extend a ledger that is already broken: appending to a broken chain
    # would launder the break by making the tail look consistent again.
    problems = verify(path)["problems"]
    if problems:
        raise ValueError(f"{path}: refusing to append to a ledger that does not "
                         f"verify: {problems[0]}")
    prev = entries[-1]["entry_hash"] if entries else GENESIS
    record = make_record(len(entries), prev, event, command=command, payload=payload,
                         artefacts=artefacts, ts=ts, base_dir=base_dir, excludes=excludes)
    with open(path, "a", encoding="utf-8") as fh:
        # Canonical bytes: the line on disk is exactly what entry_digest hashed, so an
        # offline verifier can check the raw line instead of re-parsing it.
        fh.write(canonical(record).decode() + "\n")
    return record


def verify(path: str, check_artefacts: bool = False,
           base_dir: Optional[str] = None) -> Dict[str, Any]:
    """Verify the header, the chain and (optionally) the artefacts it names.

    Returns a report rather than raising, so a caller can print all problems at once.
    """
    result: Dict[str, Any] = {"schema": LEDGER_SCHEMA, "path": path, "valid": False,
                              "n_entries": 0, "problems": [], "broken_at": None,
                              "artefacts_checked": 0, "artefacts_drifted": []}
    if not os.path.isfile(path):
        result["problems"].append(f"{path}: no such ledger")
        return result
    try:
        records = read_ledger(path)
    except ValueError as exc:
        result["problems"].append(str(exc))
        return result

    header = records[0]
    if header.get("schema") != LEDGER_SCHEMA:
        result["problems"].append(
            f"header declares schema {header.get('schema')!r}, expected {LEDGER_SCHEMA!r}")
    entries = records[1:]
    result["n_entries"] = len(entries)

    seen_seq: Dict[int, int] = {}
    prev = GENESIS
    for i, record in enumerate(entries):
        if not isinstance(record, dict):
            result["problems"].append(f"entry {i} is not an object")
            break
        seq = record.get("seq")
        if seq != i:
            # Catches a deleted, reordered or duplicated record: a gap or a repeat
            # shows up here before the hash check, with the index named.
            result["problems"].append(f"entry {i} has seq {seq!r}; expected {i}")
            result["broken_at"] = i
            break
        seen_seq[seq] = seen_seq.get(seq, 0) + 1
        if record.get("prev_hash") != prev:
            result["problems"].append(
                f"entry {i}: prev_hash {record.get('prev_hash')!r} does not link to "
                f"{prev!r}")
            result["broken_at"] = i
            break
        stored = record.get("entry_hash")
        if not isinstance(stored, str) or set(stored) - HEXDIGITS or len(stored) != 64:
            result["problems"].append(f"entry {i}: entry_hash is not a sha256 hex digest")
            result["broken_at"] = i
            break
        if entry_digest(record) != stored:
            result["problems"].append(
                f"entry {i} ({record.get('event')!r}): content does not hash to its "
                f"recorded entry_hash -- the record was edited after it was written")
            result["broken_at"] = i
            break
        try:
            canonical(record.get("payload"))
        except (ValueError, TypeError) as exc:
            result["problems"].append(f"entry {i}: payload is not canonicalisable ({exc})")
            result["broken_at"] = i
            break
        prev = stored

    if check_artefacts:
        for i, record in enumerate(entries):
            for role, meta in (record.get("artefacts") or {}).items():
                # prefer the resolved path recorded at write time; fall back to
                # resolving the readable one against base_dir
                target = meta.get("resolved") or meta.get("path")
                if base_dir and target and not os.path.isabs(target):
                    target = os.path.join(base_dir, target)
                # exists(), not isfile(): a corpus is a directory
                if not target or not os.path.exists(target):
                    result["problems"].append(
                        f"entry {i} ({record.get('event')}): artefact {role!r} is gone "
                        f"({meta.get('path')!r})")
                    continue
                result["artefacts_checked"] += 1
                try:
                    current = describe_artifact(target, exclude=meta.get("exclude") or ())
                except (ValueError, OSError) as exc:
                    result["problems"].append(
                        f"entry {i} ({record.get('event')}): artefact {role!r} could not "
                        f"be re-digested ({exc})")
                    continue
                actual = current["sha256"]
                if current["kind"] != meta.get("kind", "file"):
                    result["problems"].append(
                        f"entry {i} ({record.get('event')}): artefact {role!r} was a "
                        f"{meta.get('kind')} and is now a {current['kind']}")
                    continue
                if actual != meta.get("sha256"):
                    result["artefacts_drifted"].append(
                        {"entry": i, "event": record.get("event"), "role": role,
                         "path": meta.get("path"), "recorded": meta.get("sha256"),
                         "actual": actual})
                    result["problems"].append(
                        f"entry {i} ({record.get('event')}): artefact {role!r} "
                        f"({meta.get('path')}) changed since it was recorded "
                        f"({meta.get('sha256', '')[:12]} -> {actual[:12]})")

    result["valid"] = not result["problems"]
    return result


def digest_of_ledger(path: str) -> str:
    """The ledger's own digest: the thing you sign or publish alongside a result."""
    return file_digest(path)["sha256"]


def _parse_kv(items: Optional[Iterable[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"expected role=path, got {item!r}")
        role, value = item.split("=", 1)
        out[role.strip()] = value.strip()
    return out


def _parse_payload(items: Optional[Iterable[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"expected key=value, got {item!r}")
        key, value = item.split("=", 1)
        try:
            out[key.strip()] = json.loads(value)
        except Exception:
            out[key.strip()] = value
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cviaf lab provenance-ledger",
                                 description="append-only provenance ledger for lab results")
    sub = ap.add_subparsers(dest="command", required=True)
    p_init = sub.add_parser("init")
    p_init.add_argument("--path", required=True)
    p_init.add_argument("--note", default="")
    p_app = sub.add_parser("append")
    p_app.add_argument("--path", required=True)
    p_app.add_argument("--event", required=True)
    # dest is not `command`: the subparser's namespace carries the subcommand name in
    # `command`, and an argument whose default lands in the same dest overwrites it
    # (which silently routed `append` into the `verify` branch).
    p_app.add_argument("--command", dest="cmd", default="")
    p_app.add_argument("--input", action="append", metavar="ROLE=PATH")
    p_app.add_argument("--output", action="append", metavar="ROLE=PATH")
    p_app.add_argument("--payload", action="append", metavar="KEY=VALUE")
    p_app.add_argument("--base-dir", default=None,
                       help="directory relative artefact paths are resolved against")
    p_app.add_argument("--exclude", action="append", metavar="ROLE=PATTERN[,PATTERN]",
                       help="fnmatch patterns to exclude when digesting a directory "
                            "artefact (e.g. the analysis outputs written into a corpus)")
    p_ver = sub.add_parser("verify")
    p_ver.add_argument("--path", required=True)
    p_ver.add_argument("--check-artefacts", action="store_true")
    p_ver.add_argument("--base-dir", default=None)
    args = ap.parse_args(argv)

    try:
        if args.command == "init":
            header = init_ledger(args.path, note=args.note)
            print(f"created {args.path} ({LEDGER_SCHEMA})")
            print(f"  sha256 {sha256_bytes(json.dumps(header, sort_keys=True).encode())}")
            return 0
        if args.command == "append":
            artefacts = {f"in:{r}": p for r, p in _parse_kv(args.input).items()}
            artefacts.update({f"out:{r}": p for r, p in _parse_kv(args.output).items()})
            excludes = {}
            for role, value in _parse_kv(args.exclude).items():
                # roles may be given bare (corpus=) or as they appear in the record
                # (in:corpus=); accept both rather than making the caller guess
                key = role if role in artefacts else f"in:{role}"
                if key not in artefacts:
                    key = f"out:{role}"
                if key not in artefacts:
                    print(f"ERROR: --exclude names {role!r}, which is not an artefact "
                          f"(known: {sorted(artefacts)})")
                    return 1
                excludes[key] = [p for p in value.split(",") if p]
            record = append(args.path, args.event, command=args.cmd,
                            payload=_parse_payload(args.payload), artefacts=artefacts,
                            base_dir=args.base_dir, excludes=excludes)
            print(f"appended seq={record['seq']} event={record['event']} "
                  f"entry_hash={record['entry_hash'][:12]}")
            return 0
        report = verify(args.path, check_artefacts=args.check_artefacts,
                        base_dir=args.base_dir)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}")
        return 1

    if report["valid"]:
        print(f"{args.path}: VALID — {report['n_entries']} entries, chain intact, "
              f"{report['artefacts_checked']} artefact(s) re-hashed")
        if report["artefacts_checked"]:
            print(f"  ledger digest {digest_of_ledger(args.path)}")
        return 0
    print(f"{args.path}: INVALID — {len(report['problems'])} problem(s)")
    for p in report["problems"]:
        print(f"    {p}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
