"""Tests for the doc-vs-receipt number audit.

The audit exists because "every number was checked against its receipt" is a claim that
decays the moment somebody edits a table. Each test below is the failure it prevents:

  * the figures must come from the receipts, so the check cannot pass by agreeing with a
    constant that has since changed;
  * a document that loses a canonical figure is drift, not a smaller document;
  * a superseded figure is drift *outside* the section that keeps it as history -- and
    clean inside it, because a history section that cannot name what it retired is useless;
  * a receipt that disagrees with itself about its own survivorship curve is rejected
    before the prose is even consulted.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import number_audit as audit  # noqa: E402


def test_the_figures_are_recomputed_from_the_committed_receipts():
    canon = audit.canonical_strings()
    assert canon["fpr_ctc"] == "5.23%"
    assert canon["fpr_refdiv"] == "5.18%"
    assert canon["population"] == "56,627"
    assert canon["verified"] == audit.canonical_strings()["planned"] == "56,628"
    assert canon["arms_built"] == "1,188"
    assert canon["arms_scored"] == "1,055"
    assert canon["sub_0_25"] == "47.9%"
    assert canon["sub_0_50"] == "93.7%"
    assert canon["wt_1_00"] == "92.9%"
    assert canon["bias_any"] == "5.1%"
    assert canon["unscorable_curve"] == "0 / 5 / 36 / 92"
    assert canon["pooled_refused"] == "null"


def test_a_document_that_lost_a_canonical_figure_is_drift(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("nothing canonical here\n", encoding="utf-8")
    monkeypatch.setattr(audit, "REPO", str(tmp_path))
    problems = audit.run_forward({}, checks=[("README.md", "5.23%")])
    assert len(problems) == 1
    assert "missing canonical figure '5.23%'" in problems[0]


def test_a_retired_figure_is_drift_outside_its_history_section(tmp_path, monkeypatch):
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "page.html").write_text("we catch 34.7% of attacks\n",
                                                 encoding="utf-8")
    monkeypatch.setattr(audit, "REPO", str(tmp_path))
    problems = audit.run_backward(roots=("demo",), retired={"34.7%": ("STATUS.md",)})
    assert len(problems) == 1
    assert "retired figure" in problems[0]


def test_a_retired_figure_is_clean_inside_its_history_section(tmp_path, monkeypatch):
    (tmp_path / "STATUS.md").write_text("superseded: 34.7% pooled\n", encoding="utf-8")
    monkeypatch.setattr(audit, "REPO", str(tmp_path))
    assert audit.run_backward(roots=("STATUS.md",),
                              retired={"34.7%": ("STATUS.md",)}) == []


def test_a_receipt_that_disagrees_with_itself_about_survivorship_is_rejected(tmp_path,
                                                                            monkeypatch):
    real = audit.load(audit.LADDER)
    # Claim one fewer unscorable arm at dose 0.25 than the cells imply. The prose cannot
    # be audited against a self-contradicting receipt, so the audit must refuse first.
    real["unscorable"]["substitution@0.25"] = 4
    write_receipts(tmp_path, ladder=real)
    monkeypatch.setattr(audit, "REPO", str(tmp_path))
    with pytest.raises(ValueError) as excinfo:
        audit.canonical_strings()
    assert "disagrees with itself" in str(excinfo.value)


def test_a_receipt_whose_cells_do_not_sum_to_n_arms_is_rejected(tmp_path, monkeypatch):
    real = audit.load(audit.LADDER)
    # Same class of defect, the other direction: `n_arms` inflated but the cells left
    # alone. Both halves must agree before anything downstream is trusted.
    real["n_arms"] = real["n_arms"] + 1
    write_receipts(tmp_path, ladder=real)
    monkeypatch.setattr(audit, "REPO", str(tmp_path))
    with pytest.raises(ValueError) as excinfo:
        audit.canonical_strings()
    assert "cells sum to" in str(excinfo.value)


def test_exit_code_is_three_when_a_document_drifts(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "REPO", str(tmp_path))
    monkeypatch.setattr(audit, "canonical_strings", lambda: {"x": "1"})
    monkeypatch.setattr(audit, "forward_checks", lambda canon: [("gone.md", "1")])
    assert audit.main() == 3


def test_the_repository_as_committed_is_clean():
    """The end-to-end check: the real tree, the real receipts."""
    assert audit.main() == 0, "number drift in the committed docs"


def write_receipts(root, ladder=None):
    """Copy the real receipts into ``root``, optionally with the ladder overridden."""
    for rel in (audit.MERGED_REPORT, audit.LADDER, audit.SINGLE_DOSE, audit.VERIFY):
        target = os.path.join(str(root), rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        payload = ladder if (ladder is not None and rel == audit.LADDER) else audit.load(rel)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
