"""Tests for the clause-coverage generator.

The generator's whole value is that it refuses to say "measured" without a number, so
these tests are written as the ways it could start lying: a missing artefact quietly
becoming a pass, an empty declaration counting as a declaration, a network import in
the assurance path being filed under "corpus construction" and ignored, or a gate that
passes with no evaluation negatives.

Fixtures are built in a tmp root, so the tests never read the real `runs/` tree.
"""
from __future__ import annotations

import json
import os

import pytest

from cviaf.lab.coverage import (COVERAGE_SCHEMA, build_report, offline_audit,
                                render_markdown)


def _write(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        if isinstance(payload, str):
            fh.write(payload)
        else:
            json.dump(payload, fh)


def _ledger() -> dict:
    from cviaf.lab.fpr_tpr import LEDGER_SCHEMA
    recs = []
    for i in range(30):
        recs.append({"model_id": f"clean_s{i}", "corpus": "c", "kind": "clean",
                     "is_positive": False, "split": "unassigned",
                     "scores": {"sig": 0.1 + i / 1000.0}})
    for i in range(12):
        recs.append({"model_id": f"oga_s{i}", "corpus": "c", "kind": "oga",
                     "is_positive": True, "split": "unassigned",
                     "scores": {"sig": 0.9 + i / 1000.0}})
    return {"schema": LEDGER_SCHEMA, "alpha": 0.05, "higher_is_more_anomalous": True,
            "positive_kinds": ["oga"], "negative_kinds": ["clean"],
            "provenance": {"producer": "test"}, "records": recs}


def _report() -> dict:
    kind = {"status": "measured", "n_evaluation": 6, "tp": 1, "fn": 5,
            "tpr": {"numerator": 1, "denominator": 6, "point_estimate": 0.1667,
                    "ci95_wilson": [0.03, 0.56], "exact_upper_bound_95": 0.68,
                    "status": "measured"}}
    return {
        "schema": "cviaf.fpr-tpr-report.v1",
        "denominators": {"calibration_negatives": 25, "evaluation_negatives": 25,
                         "evaluation_positives": 6},
        "rules": {"sig": {"signal": "sig", "status": "measured",
                          "tpr": {"point_estimate": 0.1667, "denominator": 6,
                                  "status": "measured"},
                          "fpr": {"point_estimate": 0.0667, "denominator": 25,
                                  "status": "measured"}}},
        "per_kind": {"sig": {"status": "measured", "threshold": 0.5,
                             "kinds": {"oga": kind}}},
        "risk": {"per_signal": {"sig": {"status": "measured",
                                        "expected_loss_per_asset":
                                            {"accept_all": 1.0, "quarantine_flagged": 0.5,
                                             "review_flagged": 0.6},
                                        "recommended_expected_loss_per_asset": 0.5}}},
        "fpr_measured": True,
    }


def _population_report() -> dict:
    return {"schema": "cviaf.fpr-population.v1", "n_records": 10,
            "verdicts": {"one_population": True, "split_stable": True,
                         "refusals": [], "warnings": []},
            "shard_heterogeneity": {"corpora": ["w1"]}}


def _assurance_report() -> dict:
    return {
        "findings": [{"finding_id": "f1", "severity": "HIGH", "title": "t",
                      "description": "d", "evidence": {"sig": 1.0},
                      "affected_assets": ["a"], "disposition": "quarantine"},
                     {"finding_id": "f2", "severity": "LOW", "title": "t2",
                      "description": "d", "evidence": {}, "affected_assets": ["b"],
                      "disposition": "accept"}],
        "coverage_statement": {"supported_attack_classes": ["oga", "oda"],
                               "unsupported_conditions": ["video input"]},
        "limitations": ["no reference battery for this architecture"],
    }


def make_root(tmp_path, *, provenance=True) -> str:
    root = str(tmp_path)
    _write(os.path.join(root, "runs/fleet_census.json"),
           {"schema": "cviaf.fleet-census.v1", "n_models": 10, "problems": [],
            "ready_for_fpr": True})
    _write(os.path.join(root, "runs/manifest_validation.json"),
           {"schema": "cviaf.manifest-validation.v1",
            "corpora": {"c": {"problems": [], "n_models": 10, "foreign": {}}}})
    _write(os.path.join(root, "runs/fleet_manifest_validation.json"),
           {"schema": "cviaf.manifest-validation.v1",
            "corpora": {"w1": {"problems": [], "n_models": 4, "foreign": {}}}})
    _write(os.path.join(root, "runs/fpr_ledger.json"), _ledger())
    _write(os.path.join(root, "runs/fpr_ledger_report.json"), _report())
    _write(os.path.join(root, "runs/fleet_fpr_population.json"), _population_report())
    _write(os.path.join(root, "runs/demo_assurance/assurance_report.json"),
           _assurance_report())
    _write(os.path.join(root, "runs/format_ingest.json"),
           {"schema": "cviaf.format-ingest.v1", "all_ok": True,
            "formats": {"coco": {"ok": True, "n_boxes": 3},
                        "yolo": {"ok": True, "n_boxes": 3},
                        "onnx": {"ok": True, "output_shape": [1, 64, 16, 16]},
                        "pytorch": {"ok": True}, "torchscript": {"ok": True}}})
    _write(os.path.join(root, "runs/arm_b/arm_b_u3.json"),
           {"experiment": "arm_b.U3", "mal_contributor": "vendor_x",
            "summary": {"clean_label": {"vendor_x": {"rate": 0.01}},
                        "oda": {"vendor_x": {"rate": 0.2}}}})
    _write(os.path.join(root, "runs/drift_cells/harness_report.json"),
           {"summary": {"mmd2_rbf": {"drift": 4, "no_drift": 0, "under_determined": 6}}})
    _write(os.path.join(root, "runs/tamper_probe.json"), {"clean_models": ["m"]})
    _write(os.path.join(root, "cviaf/governance/schema.py"), "# report schema\n")
    _write(os.path.join(root, "cviaf/provenance/replay.py"), "# replay checks\n")
    _write(os.path.join(root, "docs/COVERAGE_STATEMENT.md"), "# coverage\n")
    if provenance:
        from cviaf.lab.provenance_ledger import append, init_ledger
        led_path = os.path.join(root, "runs/provenance_ledger.jsonl")
        init_ledger(led_path, note="fixture")
        append(led_path, event="fixture", command="pytest")
    return root


# --------------------------------------------------------------------------- #
# resolution: measured means a number came out of a file
# --------------------------------------------------------------------------- #

def test_absent_artefact_is_missing_not_measured(tmp_path):
    root = make_root(tmp_path)
    os.remove(os.path.join(root, "runs/format_ingest.json"))
    report = build_report(root, offline_hits=[])
    row = {r["id"]: r for r in report["clauses"]}["2.2.6-b"]
    assert row["state"] == "missing" and row["value"] is None


def test_artefact_without_a_number_is_unmeasured(tmp_path):
    """The 2.2.5 trap: the report exists, but declares nothing to check."""
    root = make_root(tmp_path)
    _write(os.path.join(root, "runs/demo_assurance/assurance_report.json"),
           {"findings": [], "coverage_statement": {}, "limitations": []})
    report = build_report(root, offline_hits=[])
    rows = {r["id"]: r for r in report["clauses"]}
    assert rows["2.2.5-a"]["state"] == "unmeasured"
    assert rows["2.2.5-b"]["state"] == "unmeasured"


def test_a_clause_reads_measured_on_a_complete_fixture(tmp_path):
    report = build_report(make_root(tmp_path), offline_hits=[])
    assert report["schema"] == COVERAGE_SCHEMA
    assert report["states"].get("measured", 0) >= 12
    assert not [r for r in report["clauses"] if r["state"] == "missing"]


def test_findings_missing_the_disposition_are_not_counted(tmp_path):
    root = make_root(tmp_path)
    _write(os.path.join(root, "runs/demo_assurance/assurance_report.json"),
           {"findings": [{"finding_id": "f1", "severity": "HIGH", "description": "d"}],
            "coverage_statement": {"supported_attack_classes": ["oga"]},
            "limitations": []})
    report = build_report(root, offline_hits=[])
    row = {r["id"]: r for r in report["clauses"]}["2.2.5-a"]
    assert row["state"] == "unmeasured"


# --------------------------------------------------------------------------- #
# the offline audit: the assurance path is the one that must be clean
# --------------------------------------------------------------------------- #

def test_network_import_in_the_assurance_path_fails_the_clause(tmp_path):
    hits = [{"file": "cviaf/lab/pipeline.py", "line": "3", "import": "requests",
             "area": "assurance"}]
    report = build_report(make_root(tmp_path), offline_hits=hits)
    row = {r["id"]: r for r in report["clauses"]}["2.2.6-a"]
    assert row["state"] == "unmeasured" and "assurance path" in row["detail"]
    assert not [g for g in report["gate"] if g["key"] == "offline"][0]["ok"]


def test_fetch_once_in_corpus_construction_does_not_fail_the_clause(tmp_path):
    """PS 2.2.6 constrains the assessment workflow, and corpus builders may cache a
    public dataset once — but that must be visible, not hidden."""
    hits = [{"file": "cviaf/lab/cifar.py", "line": "42",
             "import": "urllib.request", "area": "corpus_ingest"}]
    report = build_report(make_root(tmp_path), offline_hits=hits)
    row = {r["id"]: r for r in report["clauses"]}["2.2.6-a"]
    assert row["state"] == "measured" and "corpus construction" in row["detail"]


def test_offline_audit_reads_imports_not_docstrings(tmp_path):
    root = tmp_path
    (root / "cviaf").mkdir()
    (root / "cviaf/x.py").write_text(
        '"""This module mentions requests and urllib.request on purpose."""\n'
        "import json\n", encoding="utf-8")
    assert offline_audit(str(root / "cviaf")) == []
    (root / "cviaf/y.py").write_text("import urllib.request\n", encoding="utf-8")
    hits = offline_audit(str(root / "cviaf"))
    assert hits and hits[0]["import"] == "urllib.request" and hits[0]["area"] == "assurance"


# --------------------------------------------------------------------------- #
# the gate: what must hold before a 20k FPR run is worth reading
# --------------------------------------------------------------------------- #

def test_gate_passes_on_a_complete_fixture(tmp_path):
    report = build_report(make_root(tmp_path), offline_hits=[])
    assert report["gate_ok"] is True
    assert all(g["ok"] for g in report["gate"])


def test_gate_fails_when_negatives_are_below_the_floor(tmp_path):
    root = make_root(tmp_path)
    rep = _report()
    rep["denominators"]["evaluation_negatives"] = 3   # below the >=20 interval floor
    _write(os.path.join(root, "runs/fpr_ledger_report.json"), rep)
    report = build_report(root, offline_hits=[])
    item = [g for g in report["gate"] if g["key"] == "negatives_on_both_sides"][0]
    assert item["ok"] is False and "evaluation_negatives=3" in item["detail"]
    assert report["gate_ok"] is False


def test_gate_fails_on_a_mislabelled_ledger(tmp_path):
    """The compare.py failure, seen from the gate: a tamper arm recorded as clean."""
    root = make_root(tmp_path)
    led = _ledger()
    led["records"][0]["kind"] = "oga"          # still is_positive False
    _write(os.path.join(root, "runs/fpr_ledger.json"), led)
    report = build_report(root, offline_hits=[])
    item = [g for g in report["gate"] if g["key"] == "ledger_valid"][0]
    assert item["ok"] is False and "mislabelled" in item["detail"]


def test_gate_fails_when_the_audit_chain_does_not_verify(tmp_path):
    root = make_root(tmp_path)
    led_path = os.path.join(root, "runs/provenance_ledger.jsonl")
    with open(led_path, "a", encoding="utf-8") as fh:      # splice a record out of chain
        fh.write(json.dumps({"seq": 99, "prev_hash": "deadbeef"}) + "\n")
    report = build_report(root, offline_hits=[])
    item = [g for g in report["gate"] if g["key"] == "audit_trail_verifies"][0]
    assert item["ok"] is False


def test_markdown_says_which_gate_items_failed(tmp_path):
    root = make_root(tmp_path)
    os.remove(os.path.join(root, "runs/fleet_census.json"))
    report = build_report(root, offline_hits=[])
    text = render_markdown(report)
    assert "Ready-for-corpus gate" in text
    assert "| population_validated |" in text and "FAIL" in text


def test_cli_exit_code_tracks_the_gate(tmp_path, capsys):
    from cviaf.lab.coverage import main
    root = make_root(tmp_path)
    out = os.path.join(root, "runs/coverage.json")
    assert main(["--root", root, "--json", out]) == 0
    os.remove(os.path.join(root, "runs/fleet_census.json"))
    assert main(["--root", root, "--json", out]) == 1
    capsys.readouterr()


def test_coverage_on_the_real_tree_when_artefacts_exist(capsys):
    if not os.path.isfile("runs/fpr_ledger_report.json"):
        pytest.skip("no FPR report built yet")
    from cviaf.lab.coverage import main
    code = main(["--root", ".", "--markdown", "docs/.coverage_test_tmp.md"])
    text = capsys.readouterr().out
    assert "gate overall" in text and code in (0, 1)
    os.remove("docs/.coverage_test_tmp.md")
