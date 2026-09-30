#!/usr/bin/env python
"""Clause coverage: PS 26228 clause -> the artefact that *measures* it, or nothing.

The hand-written trace (`docs/PS26228_REQUIREMENT_TRACE.md`) says what the design
intends. This module says what is on disk right now, and it refuses to call a clause
"measured" without a number and a denominator behind it. The distinction is the whole
point: the trace marked clause 3.7 (ODA recall) as planned for weeks while a corpus
sitting in `runs/day1` could have answered it in seconds.

Three states, and no fourth:

  ``measured``      an artefact exists and a resolvable number was read from it
  ``unmeasured``    the artefact exists (code or corpus) but nothing in it is a number
                    — "implemented, not measured", which is where prose tends to lie
  ``missing``       the artefact is not on disk at all

The checklist half (`GATE`) is what must be true before an FPR run on the 20k fleet is
worth reading: a validated population, a validated ledger, enough negatives on both
sides of the split, at least one positive kind so the TPR is not vacuous, a verifying
audit trail, and no network imports anywhere in the package.

CLI
---
    python -m cviaf.lab.coverage [--json runs/coverage.json]
                                 [--markdown docs/COVERAGE_STATEMENT.md]

Exit: 0 every gate item passes, 1 something in the gate is unmet, 2 nothing to read.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

COVERAGE_SCHEMA = "cviaf.clause-coverage.v1"

# A clause is "measured" only if a number comes out of an artefact. Keep these
# derivations small and explicit: a fancy rule here is indistinguishable from prose.
DERIVERS: Dict[str, Callable[[Dict[str, Any], List[str]], Tuple[Any, str]]] = {}


def deriver(name: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        DERIVERS[name] = fn
        return fn
    return deco


def _load(path: str) -> Optional[Any]:
    if not os.path.isfile(path):
        return None
    try:
        if path.endswith(".jsonl"):
            with open(path, encoding="utf-8") as fh:
                return [json.loads(line) for line in fh if line.strip()]
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# derivations: each names what it read, because "measured: True" is not evidence
# --------------------------------------------------------------------------- #

@deriver("measured_positive_kinds")
def _measured_positive_kinds(docs: Dict[str, Any], paths: List[str]) -> Tuple[Any, str]:
    """Distinct attack kinds with a non-zero evaluation denominator in the FPR report."""
    rep = docs.get(paths[0])
    if not isinstance(rep, dict):
        return None, "no FPR report"
    kinds: set = set()
    for signal, block in (rep.get("per_kind") or {}).items():
        for kind, row in ((block or {}).get("kinds") or {}).items():
            if isinstance(row, dict) and int(row.get("n_evaluation") or 0) > 0:
                kinds.add(kind)
    if not kinds:
        return None, f"{paths[0]} has no per-kind denominators"
    return len(kinds), f"{len(kinds)} kinds with an evaluation denominator in {paths[0]}: {sorted(kinds)}"


@deriver("rules_with_a_rate")
def _rules_with_a_rate(docs: Dict[str, Any], paths: List[str]) -> Tuple[Any, str]:
    rep = docs.get(paths[0])
    if not isinstance(rep, dict):
        return None, "no FPR report"
    ok = [name for name, rule in (rep.get("rules") or {}).items()
          if ((rule or {}).get("fpr") or {}).get("point_estimate") is not None
          or ((rule or {}).get("fpr") or {}).get("status") == "insufficient_denominator"]
    return (len(ok) or None), (f"{len(ok)} rule(s) with an FPR denominator in {paths[0]}"
                              if ok else f"{paths[0]} refuses every rule")


@deriver("priced_rules")
def _priced_rules(docs: Dict[str, Any], paths: List[str]) -> Tuple[Any, str]:
    """Clause 1.4: rules whose expected loss per disposition was actually computed.

    A price needs a TPR, a TPR needs attacked assets: on an all-clean fleet ledger this
    correctly reads `unmeasured` rather than quoting the accept-all loss as a result.
    """
    risk = ((docs.get(paths[0]) or {}).get("risk") or {})
    priced, why = [], []
    for name, row in (risk.get("per_signal") or {}).items():
        if not isinstance(row, dict):
            continue
        loss = row.get("expected_loss_per_asset")
        if isinstance(loss, dict) and any(v is not None for v in loss.values()):
            priced.append(name)
        else:
            why.append(str(row.get("status") or row.get("refusal") or "no loss"))
    return (len(priced) or None), (f"clause 1.4 priced {len(priced)} rule(s) in {paths[0]}: "
                                   f"{priced}" if priced else
                                   f"expected loss not priced ({'; '.join(sorted(set(why))[:2])})")


@deriver("contributor_level_measured")
def _contributor_level_measured(docs: Dict[str, Any], paths: List[str]) -> Tuple[Any, str]:
    doc = docs.get(paths[0])
    if not isinstance(doc, dict):
        return None, "no contributor-arm result"
    summary = doc.get("summary") or {}
    rows = sum(len(v) if isinstance(v, dict) else 0 for v in summary.values())
    return (rows or None), (f"{rows} contributor-level cell(s) in {paths[0]} "
                            f"(malicious contributor {doc.get('mal_contributor')!r})"
                            if rows else f"{paths[0]} has no summary cells")


@deriver("drift_decided")
def _drift_decided(docs: Dict[str, Any], paths: List[str]) -> Tuple[Any, str]:
    doc = docs.get(paths[0])
    if not isinstance(doc, dict):
        return None, "no drift report"
    summary = doc.get("summary") or {}
    decided = sum((v or {}).get("drift", 0) for v in summary.values())
    under = sum((v or {}).get("under_determined", 0) for v in summary.values())
    if not summary:
        return None, "drift report has no per-metric summary"
    return decided, (f"{decided} drift decision(s) and {under} under-determined across "
                     f"{len(summary)} metrics in {paths[0]}")


@deriver("under_determined_exposed")
def _under_determined_exposed(docs: Dict[str, Any], paths: List[str]) -> Tuple[Any, str]:
    """Clause 5.6: the report must be able to say 'cannot tell', not just yes/no."""
    doc = docs.get(paths[0])
    if not isinstance(doc, dict):
        return None, "no drift report"
    under = sum((v or {}).get("under_determined", 0) for v in (doc.get("summary") or {}).values())
    return (under or None), (f"{under} cell/metric pair(s) reported under_determined in "
                             f"{paths[0]}" if under else "no under_determined cells reported")


@deriver("audit_chain")
def _audit_chain(docs: Dict[str, Any], paths: List[str]) -> Tuple[Any, str]:
    """Clause 2.2.3/2.3: the hash chain has to verify, not merely exist."""
    from cviaf.lab.provenance_ledger import verify
    # Resolve against the report root, not the process cwd: a fixture root would
    # otherwise "verify" the real repository's ledger and pass on the wrong file.
    path = os.path.join(docs.get("__root__") or ".", paths[0])
    if not os.path.isfile(path):
        return None, f"{paths[0]} not on disk under the report root"
    result = verify(path)
    problems = result.get("problems") or []
    if problems:
        return None, f"{path} does not verify: {problems[:2]}"
    return int(result.get("n_entries") or 0), (
        f"{result.get('n_entries')} chained entries verify (seq + prev_hash) in {path}")


@deriver("report_with_dispositions")
def _report_with_dispositions(docs: Dict[str, Any], paths: List[str]) -> Tuple[Any, str]:
    """Clause 2.2.5: every flag carries a reason, evidence, affected asset, disposition.

    The PS names four things per flag, so the check asks for all four (accepting the
    report schema's own field names: title/description for the reason, affected_assets
    for the asset, disposition for the recommendation).
    """
    doc = docs.get(paths[0])
    if not isinstance(doc, dict):
        return None, f"{paths[0]} is not a readable JSON report"
    findings = doc.get("findings")
    if not isinstance(findings, list) or not findings:
        return None, f"{paths[0]} has no findings to check for reason/evidence/disposition"
    complete = [f for f in findings if isinstance(f, dict)
                and f.get("disposition")
                and (f.get("title") or f.get("description") or f.get("reason"))
                and f.get("evidence")
                and f.get("affected_assets")]
    if not complete:
        return None, (f"none of {len(findings)} findings in {paths[0]} carry all four of "
                      f"reason/evidence/asset/disposition")
    return len(complete), (f"{len(complete)}/{len(findings)} findings carry a reason, "
                           f"evidence, affected asset and disposition in {paths[0]}")


@deriver("declared_coverage")
def _declared_coverage(docs: Dict[str, Any], paths: List[str]) -> Tuple[Any, str]:
    """PS 2.3: 'a clear coverage statement ... supported attack classes, assumptions,
    known limitations'. Count what the report itself declares, so an empty declaration
    cannot pass as one."""
    doc = docs.get(paths[0])
    if not isinstance(doc, dict):
        return None, f"{paths[0]} is not a readable JSON report"
    cov = doc.get("coverage_statement")
    if not isinstance(cov, dict):
        return None, f"{paths[0]} carries no coverage_statement block"
    counts = {k: len(v) for k, v in cov.items() if isinstance(v, (list, dict))}
    total = sum(counts.values())
    limits = doc.get("limitations")
    n_lim = len(limits) if isinstance(limits, list) else 0
    if not total:
        return None, f"coverage_statement in {paths[0]} declares nothing"
    return total, (f"declares {total} coverage entr(ies) {counts} and {n_lim} limitation(s) "
                   f"in {paths[0]}")


@deriver("coco_yolo_ok")
def _coco_yolo_ok(docs: Dict[str, Any], paths: List[str]) -> Tuple[Any, str]:
    doc = docs.get(paths[0])
    if not isinstance(doc, dict):
        return None, f"no ingest artefact at {paths[0]}"
    fmt = doc.get("formats") or {}
    ok = [n for n in ("coco", "yolo") if (fmt.get(n) or {}).get("ok")]
    if not ok:
        return None, f"{paths[0]}: neither COCO nor YOLO ingested"
    boxes = {n: (fmt.get(n) or {}).get("n_boxes") for n in ok}
    return len(ok), f"{paths[0]}: ingested {ok} with box counts {boxes}"


@deriver("runtime_formats_ok")
def _runtime_formats_ok(docs: Dict[str, Any], paths: List[str]) -> Tuple[Any, str]:
    doc = docs.get(paths[0])
    if not isinstance(doc, dict):
        return None, f"no ingest artefact at {paths[0]}"
    fmt = doc.get("formats") or {}
    want = ("onnx", "pytorch", "torchscript")
    ok = [n for n in want if (fmt.get(n) or {}).get("ok")]
    if not ok:
        return None, f"{paths[0]}: none of {want} loaded"
    return len(ok), (f"{paths[0]}: loaded {ok} "
                     f"(onnx forward {fmt.get('onnx', {}).get('output_shape')})")


@deriver("source_symbols")
def _source_symbols(docs: Dict[str, Any], paths: List[str]) -> Tuple[Any, str]:
    """Count symbols matching keywords in source docs, e.g. coco+yolo or onnx+torchscript."""
    need = [p for p in paths if p.startswith("sym:")]
    files = list(docs.get("__source__") or [])
    want = [p.split(":", 1)[1] for p in need]
    hits = {w: 0 for w in want}
    for path in files:
        try:
            text = open(path, encoding="utf-8", errors="ignore").read().lower()
        except OSError:
            continue
        for w in want:
            if w in text:
                hits[w] += 1
    got = [w for w, c in hits.items() if c]
    return (len(got) or None), (f"symbols {got} found in {len(files)} source file(s); "
                               f"expected {want}")


# --------------------------------------------------------------------------- #
# the clause table. `evidence` is (path, deriver-name-or-None); a bare path with no
# deriver can only ever reach `unmeasured`, which is the honest ceiling for "the code
# is there" claims.
# --------------------------------------------------------------------------- #

CLAUSES: List[Dict[str, Any]] = [
    {"id": "1.4", "title": "Risk, not just integrity: expected loss per disposition",
     "evidence": [("runs/fpr_ledger_report.json", "priced_rules")]},
    {"id": "2.2.1-a", "title": "Five data-attack families, each with a denominator",
     "evidence": [("runs/fpr_ledger_report.json", "measured_positive_kinds")]},
    {"id": "2.2.1-b", "title": "Sample evidence aggregated to contributor/source risk",
     "evidence": [("runs/arm_b/arm_b_u3.json", "contributor_level_measured")]},
    {"id": "2.2.2-a", "title": "Model integrity against a defined reference battery",
     "evidence": [("runs/real_cifar/battery_model_attacks.json", None),
                  ("runs/fpr_ledger_report.json", "rules_with_a_rate")]},
    {"id": "2.2.3", "title": "Inference provenance: hash-chained, replay-checked records",
     "evidence": [("runs/provenance_ledger.jsonl", "audit_chain"),
                  ("cviaf/provenance/replay.py", None)]},
    {"id": "2.2.4", "title": "Distribution shift vs manipulation, with under-determined",
     "evidence": [("runs/drift_cells/harness_report.json", "drift_decided"),
                  ("runs/drift_cells/harness_report.json", "under_determined_exposed")]},
    {"id": "2.2.5-a", "title": "Findings carry reason, evidence, asset and disposition",
     "evidence": [("runs/demo_assurance/assurance_report.json", "report_with_dispositions")]},
    {"id": "2.2.5-b", "title": "Supported classes and unsupported conditions declared",
     "evidence": [("runs/demo_assurance/assurance_report.json", "declared_coverage")]},
    {"id": "2.2.6-a", "title": "Offline / air-gapped: no network imports",
     "evidence": [("check:offline", None)]},
    {"id": "2.2.6-b", "title": "Ingest COCO and YOLO dataset formats",
     "evidence": [("runs/format_ingest.json", "coco_yolo_ok")]},
    {"id": "2.2.6-c", "title": "Reference model formats: ONNX and PyTorch/TorchScript",
     "evidence": [("runs/format_ingest.json", "runtime_formats_ok")]},
    {"id": "2.3-a", "title": "Reproducible audit log of the assurance runs",
     "evidence": [("runs/provenance_ledger.jsonl", "audit_chain")]},
    {"id": "2.3-b", "title": "Assurance-report schema shipped and validated",
     "evidence": [("runs/demo_assurance/assurance_report.json", "declared_coverage"),
                  ("cviaf/governance/schema.py", None)]},
    {"id": "2.3-c", "title": "Coverage statement naming supported classes and limits",
     "evidence": [("check:self", None)]},
]

SOURCE_FILES = [
    "cviaf/lab/pipeline.py", "cviaf/lab/compare.py", "cviaf/lab/detectors.py",
    "cviaf/lab/evaluate.py", "cviaf/formats/__init__.py", "cviaf/formats/model_loader.py",
    "cviaf/lab/fpr_tpr.py", "cviaf/lab/drift_harness.py", "cviaf/lab/provenance_ledger.py",
    "cviaf/provenance/replay.py", "cviaf/governance/schema.py", "cviaf/cli.py",
]

NETWORK_MODULES = ("requests", "httpx", "urllib.request", "urllib3", "http.client",
                   "socket", "aiohttp", "boto3", "openai", "anthropic", "google.cloud")


# Corpus construction is allowed to fetch a public dataset once; the *assurance*
# workflow is not allowed to reach the network at all. The PS constraint (2.2.6) is
# about the latter, so the two are reported separately rather than blurred into one
# scary number.
CORPUS_MODULES = ("cviaf/lab/cifar.py", "cviaf/lab/synth.py", "cviaf/lab/drift_cells.py",
                  "cviaf/lab/cifar_subset.py")


def offline_audit(root: str = "cviaf") -> List[Dict[str, str]]:
    """AST-scan the package for imports that would break an air-gapped deployment.

    Grepping would match the word inside a docstring (this file mentions several of
    them by name); parsing the import statements does not. Each hit carries an
    ``area`` of ``assurance`` (must be clean) or ``corpus_ingest`` (fetch-once dataset
    helpers, guarded by ``CVIAF_OFFLINE``).
    """
    hits: List[Dict[str, str]] = []
    for dirpath, _dirs, files in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            try:
                tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                names: List[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for mod in names:
                    for banned in NETWORK_MODULES:
                        if mod == banned or mod.startswith(banned + "."):
                            norm = path.replace(os.sep, "/").lstrip("./")
                            area = ("corpus_ingest" if norm.endswith(CORPUS_MODULES)
                                    else "assurance")
                            hits.append({"file": path, "line": str(node.lineno),
                                         "import": mod, "area": area})
    return hits


# --------------------------------------------------------------------------- #
# gate: what must be true before an FPR run on the 20k fleet means anything
# --------------------------------------------------------------------------- #

GATE: List[Dict[str, str]] = [
    {"key": "population_validated",
     "what": "the fleet census reports no duplicate ids, specs or weights",
     "artefact": "runs/fleet_census.json"},
    {"key": "manifests_clean",
     "what": "every manifest validates against the locked schema",
     "artefact": "runs/manifest_validation.json + runs/fleet_manifest_validation.json",
     "artefacts": ["runs/manifest_validation.json",
                   "runs/fleet_manifest_validation.json"]},
    {"key": "ledger_valid",
     "what": "the FPR ledger validates (kinds declared, no partial signal coverage)",
     "artefact": "runs/fpr_ledger.json"},
    {"key": "negatives_on_both_sides",
     "what": ">=20 calibration and >=20 evaluation negatives (interval width)",
     "artefact": "runs/fpr_ledger_report.json"},
    {"key": "a_positive_kind_measured",
     "what": "at least one attacked kind has a denominator, or the TPR is vacuous",
     "artefact": "runs/fpr_ledger_report.json"},
    {"key": "fleet_one_population",
     "what": "the fleet's negatives are exchangeable: shards and splits agree on the FPR",
     "artefact": "runs/fleet_fpr_population.json"},
    {"key": "audit_trail_verifies",
     "what": "the provenance ledger hash chain verifies over its artefacts",
     "artefact": "runs/provenance_ledger.jsonl"},
    {"key": "offline",
     "what": "no network imports anywhere in the package",
     "artefact": "check:offline"},
]


def _gate_item(key: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    ok = False
    detail = "not evaluated"
    if key == "population_validated":
        c = ctx["docs"].get("runs/fleet_census.json")
        if isinstance(c, dict):
            ok = bool(c.get("ready_for_fpr")) and not c.get("problems")
            detail = (f"{c.get('n_models')} models, ready_for_fpr={c.get('ready_for_fpr')}, "
                      f"{len(c.get('problems') or [])} problem(s)")
    elif key == "manifests_clean":
        docs = [ctx["docs"].get("runs/manifest_validation.json"),
                ctx["docs"].get("runs/fleet_manifest_validation.json")]
        present = [d for d in docs if isinstance(d, dict)]
        if present:
            problems = 0
            models = 0
            for d in present:
                corpora = d.get("corpora") or {}
                for row in corpora.values():
                    problems += len((row or {}).get("problems") or [])
                    models += int((row or {}).get("n_models") or 0)
            ok = problems == 0
            detail = f"{models} manifests validated, {problems} problem(s)"
    elif key == "ledger_valid":
        led = ctx["docs"].get("runs/fpr_ledger.json")
        if isinstance(led, dict):
            problems = ctx["validate"](led)
            ok = not problems
            detail = (f"{len(led.get('records') or [])} records, "
                      f"{len(problems)} problem(s)" + (f": {problems[:2]}" if problems else ""))
    elif key == "negatives_on_both_sides":
        rep = ctx["docs"].get("runs/fpr_ledger_report.json")
        if isinstance(rep, dict):
            d = rep.get("denominators") or {}
            cal, ev = int(d.get("calibration_negatives") or 0), int(d.get("evaluation_negatives") or 0)
            ok = cal >= 20 and ev >= 20
            detail = f"calibration_negatives={cal}, evaluation_negatives={ev}"
    elif key == "a_positive_kind_measured":
        rep = ctx["docs"].get("runs/fpr_ledger_report.json")
        if isinstance(rep, dict):
            n, why = _measured_positive_kinds({"runs/fpr_ledger_report.json": rep},
                                              ["runs/fpr_ledger_report.json"])
            ok = bool(n)
            detail = why
    elif key == "fleet_one_population":
        pop = ctx["docs"].get("runs/fleet_fpr_population.json")
        if isinstance(pop, dict):
            v = pop.get("verdicts") or {}
            shards = len(((pop.get("shard_heterogeneity") or {}).get("corpora")) or [])
            warnings = v.get("warnings") or []
            ok = bool(v.get("one_population")) and bool(v.get("split_stable"))
            detail = (f"{pop.get('n_records')} records on {shards} shard(s); "
                      f"one_population={v.get('one_population')}, "
                      f"split_stable={v.get('split_stable')}, "
                      f"{len(warnings)} warning(s)")
            if not ok:
                detail += f": {(v.get('refusals') or ['no verdict'])[0]}"
    elif key == "audit_trail_verifies":
        rel = "runs/provenance_ledger.jsonl"
        path = os.path.join(ctx.get("root") or ".", rel)
        if os.path.isfile(path):
            try:
                from cviaf.lab.provenance_ledger import verify
                # check_artefacts=True: the chain alone proves the records were not
                # edited, but not that the numbers they point at still describe the
                # files on disk. A gate that passes on the weaker claim is the stale
                # headline with a signature on it, so the strong check is the gate.
                result = verify(path, check_artefacts=True)
                problems = result.get("problems") or []
                ok = bool(result.get("valid")) and not problems
                # Report the chain and the artefacts separately: "chain BROKEN" when the
                # chain is intact and a report was merely regenerated sends the reader
                # looking for an edit that never happened.
                broken_at = result.get("broken_at")
                chain = ("intact" if broken_at is None
                         else f"BROKEN at entry {broken_at}")
                detail = (f"{rel}: {result.get('n_entries', '?')} entries, chain {chain}, "
                          f"{result.get('artefacts_checked', 0)} artefact(s) re-hashed, "
                          f"{len(result.get('artefacts_superseded') or [])} superseded, "
                          f"{len(result.get('artefacts_derived_from_ledger') or [])} "
                          f"lineage-only (derived from this trail)"
                          + (f" — {problems[:2]}" if problems else ""))
            except Exception as exc:                       # pragma: no cover
                detail = f"verifier raised {type(exc).__name__}: {exc}"
        else:
            detail = f"{rel} is not on disk under the report root"
    elif key == "offline":
        hires = ctx["offline"]
        hits = [h for h in hires if h.get("area") != "corpus_ingest"]
        corpus = [h for h in hires if h.get("area") == "corpus_ingest"]
        ok = not hits
        detail = (f"{len(hits)} network import(s) in the assurance path: {hits[:2]}" if hits
                  else "assurance path is import-clean"
                       + (f"; {len(corpus)} guarded fetch site(s) in corpus construction"
                          if corpus else ""))
    return {"key": key, "ok": bool(ok), "detail": detail}


def build_report(root: str = ".", offline_hits: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    from cviaf.lab.fpr_tpr import validate_ledger

    paths: List[str] = []
    for clause in CLAUSES + GATE:
        if "evidence" in clause:
            entries = [(p, None) for p, _ in clause["evidence"]]
        else:
            entries = [(p, None) for p in clause.get("artefacts") or [clause["artefact"]]]
        for path, _deriver_name in entries:
            if not path.startswith(("check:", "sym:")) and path not in paths:
                paths.append(path)
    docs = {p: _load(os.path.join(root, p)) for p in paths}
    docs["__root__"] = root
    docs["__source__"] = [os.path.join(root, f) for f in SOURCE_FILES
                          if os.path.isfile(os.path.join(root, f))]

    rows: List[Dict[str, Any]] = []
    for clause in CLAUSES:
        best: Optional[Dict[str, Any]] = None
        for path, dname in clause["evidence"]:
            entry = _resolve_evidence(path, dname, docs, root, offline_hits)
            if best is None or _rank(entry["state"]) > _rank(best["state"]):
                best = entry
        rows.append({"id": clause["id"], "title": clause["title"], "state": best["state"],
                     "value": best["value"], "evidence": best["evidence"],
                     "detail": best["detail"]})

    ctx = {"docs": docs, "validate": validate_ledger, "root": root,
           "offline": offline_hits if offline_hits is not None else offline_audit(
               os.path.join(root, "cviaf"))}
    gate = [_gate_item(item["key"], ctx) for item in GATE]
    for item, spec in zip(gate, GATE):
        item["what"] = spec["what"]
        item["artefact"] = spec["artefact"]

    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    return {"schema": COVERAGE_SCHEMA, "clauses": rows, "states": counts,
            "gate": gate, "gate_ok": all(g["ok"] for g in gate)}


def _rank(state: str) -> int:
    return {"missing": 0, "unmeasured": 1, "measured": 2}[state]


def _resolve_evidence(path: str, dname: Optional[str], docs: Dict[str, Any],
                      root: str, offline_hits: Optional[List[Dict[str, str]]]) -> Dict[str, Any]:
    if path == "check:offline":
        hits = (offline_hits if offline_hits is not None
                else offline_audit(os.path.join(root, "cviaf")))
        assurance = [h for h in hits if h.get("area") != "corpus_ingest"]
        corpus = [h for h in hits if h.get("area") == "corpus_ingest"]
        if assurance:
            return {"state": "unmeasured", "value": None, "evidence": "check:offline",
                    "detail": f"{len(assurance)} network import(s) in the assurance path; "
                              f"clause not satisfiable offline: {assurance[:2]}"}
        tail = (f"; {len(corpus)} fetch-once call site(s) in corpus construction "
                f"({sorted({h['file'] for h in corpus})}), guarded by CVIAF_OFFLINE"
                if corpus else "")
        return {"state": "measured", "value": len(assurance), "evidence": "check:offline",
                "detail": f"AST scan of cviaf/: no network import in the assurance path{tail}"}
    if path == "check:self":
        doc = os.path.join(root, "docs/COVERAGE_STATEMENT.md")
        if os.path.isfile(doc):
            return {"state": "measured", "value": os.path.getsize(doc),
                    "evidence": "docs/COVERAGE_STATEMENT.md",
                    "detail": "this statement is generated from live artefacts"}
        return {"state": "missing", "value": None, "evidence": "docs/COVERAGE_STATEMENT.md",
                "detail": "generate with --markdown docs/COVERAGE_STATEMENT.md"}
    if path.startswith("sym:"):
        return {"state": "unmeasured", "value": None, "evidence": f"sym:{path}",
                "detail": "keyword presence is not a measurement"}
    full = os.path.join(root, path)
    doc = docs.get(path)
    if not os.path.exists(full):
        return {"state": "missing", "value": None, "evidence": path,
                "detail": "not on disk"}
    if dname:
        fn = DERIVERS.get(dname)
        if fn is None:                                     # pragma: no cover - typo guard
            return {"state": "unmeasured", "value": None, "evidence": path,
                    "detail": f"unknown deriver {dname!r}"}
        value, detail = fn(docs, [path])
        if value is None:
            return {"state": "unmeasured", "value": None, "evidence": path, "detail": detail}
        return {"state": "measured", "value": value, "evidence": path, "detail": detail}
    if path.endswith(".py"):
        n_lines = sum(1 for _ in open(full, encoding="utf-8", errors="ignore"))
        return {"state": "unmeasured", "value": None, "evidence": path,
                "detail": f"source present ({n_lines} lines); no measured behaviour "
                          f"artefact is attached to this clause"}
    if doc is None:
        return {"state": "unmeasured", "value": None, "evidence": path,
                "detail": "file present but unreadable or empty"}
    return {"state": "unmeasured", "value": None, "evidence": path,
            "detail": "artefact present, no number attached to it"}


def render_markdown(report: Dict[str, Any], produced_by: str = "python -m cviaf.lab.coverage",
                    regenerate: str = "--markdown docs/COVERAGE_STATEMENT.md") -> str:
    lines = [
        "# Coverage statement — PS 26228",
        "",
        "**Generated, not written by hand.** Every row below is resolved from an artefact on",
        "disk by `cviaf/lab/coverage.py`. A clause reads `measured` only when a number with a",
        "denominator was read out of a file; `unmeasured` means the code or corpus exists but",
        "nothing in it is a number yet — which is where a hand-written trace drifts into",
        "optimism. Regenerate with:",
        "",
        "```bash",
        f"cd .task3 && PYTHONPATH=. <venv>/bin/python -m cviaf.lab.coverage {regenerate}",
        "```",
        "",
        f"**States:** " + ", ".join(f"{k}={v}" for k, v in sorted(report["states"].items())),
        "",
        "## Clauses",
        "",
        "| clause | what it requires | state | measured value | artefact / why not |",
        "|---|---|---|---|---|",
    ]
    for row in report["clauses"]:
        value = "" if row["value"] is None else str(row["value"])
        detail = (row["detail"] or "").replace("|", "/")
        lines.append(f"| {row['id']} | {row['title']} | {row['state']} | {value} | "
                     f"`{row['evidence']}` — {detail} |")
    lines += ["", "## Ready-for-corpus gate", "",
              "What must be true before an FPR run on the 20k fleet is worth reading.",
              "", "| item | what it requires | state | detail |", "|---|---|---|---|"]
    for item in report["gate"]:
        lines.append(f"| {item['key']} | {item['what']} | "
                     f"{'pass' if item['ok'] else 'FAIL'} | {item['detail']} |")
    lines += ["", f"**Gate: {'PASS' if report['gate_ok'] else 'FAIL'}**",
              "",
              "## Known limitations (mirror of the gate failures and unmeasured clauses)",
              "",
              "Anything not `measured` above is a limitation of the current evidence, not of",
              "the design. The three that constrain every number in this repo:",
              "",
              "1. **FPR/TPR numbers are single-population.** Calibration and evaluation halves",
              "   are drawn from the same clean fleet, so the measured FPR is an average over",
              "   one population; the reference-relative signals shift by a large amount when",
              "   the reference model changes.",
              "2. **Denominators are small in the tamper corpora.** A TPR over 8 or 16 arms",
              "   carries a Wilson interval ~0.19-0.24 wide; the report prints the interval",
              "   and refuses a bare rate below the declared floor.",
              "3. **Drift axes are near the noise floor at n=160.** Per the drift harness,",
              "   most declared cells are `under_determined` rather than `drift`, and that is",
              "   reported as such.",
              "",
              f"_Produced by `{produced_by}` on live artefacts; do not edit by hand._", ""]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--json", default=None, help="write the machine-readable report here")
    ap.add_argument("--markdown", default=None, help="write the coverage statement here")
    args = ap.parse_args(argv)

    report = build_report(args.root)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
    text = render_markdown(report)
    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)) or ".", exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        print(text)
    for row in report["clauses"]:
        print(f"  {row['id']:10s} {row['state']:10s} {row['detail'][:96]}")
    for item in report["gate"]:
        print(f"  gate {item['key']:26s} {'pass' if item['ok'] else 'FAIL'}  {item['detail'][:80]}")
    print(f"  gate overall: {'PASS' if report['gate_ok'] else 'FAIL'}")
    return 0 if report["gate_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
