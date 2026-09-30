"""Tests for multiplicity control over the per-kind table (`per_kind_family`).

Each test is a failure this lane would otherwise have published:

  * a per-kind cell quoted on its own -- four signals by seven kinds is 28 chances
    to be the largest fraction, and "the detector catches 75% of GMA arms" was a
    3/4;
  * a correction that includes cells it cannot test, which weakens it for the cells
    that can be read;
  * a retired (saturated) signal contributing 0/n cells to the family, buying the
    quotable cells a weaker correction;
  * the family size or the null being implicit, so two reports' q-values look
    comparable when they are not.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cviaf.lab.fpr_tpr import (  # noqa: E402
    binom_pvalue,
    binom_tail_ge,
    evaluate_corpus,
    render_report,
)

KINDS = ["dup_flood", "gma", "label_flip", "oda", "oga", "ood_insert", "rma"]


def _kind_ledger(spec, n_clean=200, signal_names=("s",)):  # noqa: D401
    """A ledger with EXACT per-kind evaluation counts.

    ``spec`` maps signal -> {kind: (tp, n)} and the record's split is set explicitly,
    so the eval-half cell is the one the test asked for rather than whatever a split
    seed produced (assign_splits only touches 'unassigned').
    """
    import numpy as np
    rng = np.random.default_rng(11)
    records = []
    for i in range(n_clean):
        records.append({"model_id": f"clean_{i}", "corpus": "shard0", "kind": "clean",
                        "is_positive": False,
                        "split": "calibration" if i % 2 else "evaluation",
                        "scores": {s: float(rng.uniform(0, 1)) for s in signal_names}})
    for s in signal_names:
        for kind in spec.get(s, {}):
            tp, n = spec[s][kind]
            for j in range(n):
                caught = j < tp
                records.append({
                    "model_id": f"{s}_{kind}_{j}", "corpus": "shard0", "kind": kind,
                    "is_positive": True, "split": "evaluation",
                    "scores": {name: (5.0 if (name == s and caught) else 0.5)
                               for name in signal_names}})
                # a matching calibration-half arm, so both halves carry the kind
                records.append({
                    "model_id": f"{s}_{kind}_{j}_cal", "corpus": "shard0", "kind": kind,
                    "is_positive": True, "split": "calibration",
                    "scores": {name: (5.0 if (name == s and caught) else 0.5)
                               for name in signal_names}})
    return {"schema": "cviaf.fpr-tpr-ledger.v1", "alpha": 0.05,
            "higher_is_more_anomalous": True,
            "positive_kinds": sorted({k for s in spec for k in spec[s]}),
            "negative_kinds": ["clean"],
            "provenance": {"producer": "test", "corpora": ["shard0"]},
            "records": records}


def _seven_kinds(dominant=None, tp=0, n=8):
    return {k: (tp if k == dominant else 0, n) for k in KINDS}


def test_no_per_kind_cell_reaches_a_reader_without_its_q_value():
    spec = {"s": _seven_kinds(dominant="gma", tp=3, n=8)}
    report = evaluate_corpus(_kind_ledger(spec), alpha=0.05, min_negatives=20,
                             split_seed=0)
    family = report["per_kind_family"]
    assert family["n_tested"] == 7
    assert all("q_value" in c for c in family["cells"] if c["status"] == "tested")
    text = render_report(report)
    body = text.split("recall by attack kind")[1]
    assert "gma=3/8(q=" in body.replace(" ", "") or "gma=3/8(q=" in body
    assert "q=n/a" in body or "BH across 7" in body


def test_the_family_covers_every_kind_cell_of_the_quotable_rules():
    spec = {"a": _seven_kinds(dominant="gma", tp=3), "b": _seven_kinds(tp=2)}
    report = evaluate_corpus(_kind_ledger(spec, signal_names=("a", "b")), alpha=0.05,
                             min_negatives=20, split_seed=0)
    family = report["per_kind_family"]
    seen = {(c["signal"], c["kind"]) for c in family["cells"]}
    assert seen == {("a", k) for k in KINDS} | {("b", k) for k in KINDS}
    assert family["n_tested"] + family["n_not_tested"] == family["n_cells"] == 14


def test_a_kind_that_stands_out_is_reported_only_because_it_survives_bh():
    spec = {"s": _seven_kinds(dominant="gma", tp=6, n=8)}
    report = evaluate_corpus(_kind_ledger(spec), alpha=0.05, min_negatives=20,
                             split_seed=0)
    family = report["per_kind_family"]
    survivors = {(c["signal"], c["kind"]) for c in family["survivors"]}
    assert ("s", "gma") in survivors
    cell = next(c for c in family["cells"] if c["kind"] == "gma")
    assert cell["q_value"] < 0.05
    assert cell["p_greater"] < cell["p_less"]


def test_the_largest_of_many_small_fractions_is_not_a_finding():
    """The exact failure: 3/4 of GMA looks like 75% recall until you count the cells."""
    spec = {"s": _seven_kinds(dominant="gma", tp=3, n=8)}
    report = evaluate_corpus(_kind_ledger(spec), alpha=0.05, min_negatives=20,
                             split_seed=0)
    family = report["per_kind_family"]
    assert family["n_rejected"] == 0
    assert "NOTHING survives BH" in family["verdict"]
    assert family["smallest_q"] is not None and family["smallest_q"] > 0.05


def test_cells_that_cannot_be_tested_do_not_weaken_the_correction():
    spec = {"s": {**{k: (0, 8) for k in KINDS}, "tiny": (0, 2)}}
    report = evaluate_corpus(_kind_ledger(spec, signal_names=("s",)), alpha=0.05,
                             min_negatives=20, split_seed=0)
    family = report["per_kind_family"]
    tiny = next(c for c in family["cells"] if c["kind"] == "tiny")
    assert tiny["status"] == "not_tested"
    assert "min_cells" in tiny["reason"]
    assert family["n_tested"] == 7 and family["n_not_tested"] == 1


def test_a_retired_signal_contributes_no_cells_and_does_not_widen_the_family():
    """A rule that cannot fire cannot testify about a kind, and its 0/n cells would
    only buy the readable cells a laxer correction."""
    report = evaluate_corpus(_kind_ledger({"s": _seven_kinds(tp=3)}, signal_names=("s",)),
                             alpha=0.05, min_negatives=20, split_seed=0)
    assert report["per_kind_family"]["excluded_retired_signals"] == []
    # Now the fleet-like case: a signal pinned at the corpus maximum is retired.
    spec = {"s": {k: (3, 8) for k in KINDS}}
    led = _kind_ledger(spec)
    for rec in led["records"]:
        rec["scores"]["sat"] = 1.0 if rec["kind"] == "clean" else 5.0
    report = evaluate_corpus(led, alpha=0.05, min_negatives=20, split_seed=0)
    if "sat" in report["appendix_rules"]:
        assert "sat" in report["per_kind_family"]["excluded_retired_signals"]
        assert not any(c["signal"] == "sat" for c in report["per_kind_family"]["cells"])


def test_each_cell_is_tested_against_its_own_signals_recall():
    """Two signals with different corpus levels must not share one null."""
    spec = {"a": _seven_kinds(dominant="gma", tp=6, n=8),
            "b": _seven_kinds(dominant="gma", tp=1, n=8)}
    report = evaluate_corpus(_kind_ledger(spec, signal_names=("a", "b")), alpha=0.05,
                             min_negatives=20, split_seed=0)
    cells = {(c["signal"], c["kind"]): c for c in report["per_kind_family"]["cells"]}
    assert cells[("a", "gma")]["baseline_recall"] != cells[("b", "gma")]["baseline_recall"]
    assert cells[("a", "gma")]["baseline_recall"] > cells[("b", "gma")]["baseline_recall"]


def test_the_exact_binomial_helpers_are_sane():
    assert binom_tail_ge(10, 0.5, 0) == 1.0
    assert binom_tail_ge(10, 0.5, 11) == 0.0
    assert binom_tail_ge(10, 0.5, 10) == pytest.approx(0.5 ** 10, rel=1e-6)
    assert binom_tail_ge(4, 0.5, 3) == pytest.approx(0.3125, rel=1e-9)
    assert binom_tail_ge(4, 0.5, 2) == pytest.approx(0.6875, rel=1e-9)
    assert binom_tail_ge(4, 0.5, 4) == pytest.approx(0.0625, rel=1e-9)
    assert binom_pvalue(5, 10, 0.5) == pytest.approx(1.0, abs=1e-9)
    assert binom_pvalue(10, 10, 0.5) < 0.01
    assert 0.0 <= binom_pvalue(3, 8, 0.4) <= 1.0


def test_the_two_exact_tail_implementations_agree():
    """Two 'exact' answers that drift apart is how a q-value stops meaning anything."""
    from cviaf.lab.power import _tail_ge
    for n in (5, 11, 40, 137):
        for p in (0.05, 0.2, 0.5, 0.8):
            for k in (0, n // 3, n // 2, n):
                assert binom_tail_ge(n, p, k) == pytest.approx(_tail_ge(n, p, k), abs=1e-12)
