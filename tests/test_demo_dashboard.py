"""Tests for the interactive ledger view, named after what each one prevents.

The page exists so a reviewer who will not run the pipeline can still falsify the
headline by hand. That only works if the page's arithmetic is the report's arithmetic -
so the important test here is not "does it render", it is "does it refuse when the two
disagree".
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.demo_dashboard import (  # noqa: E402
    b64_bits,
    b64_floats,
    main,
    quantile,
    seed_owner_map,
    wilson,
)

SIGNALS = ["ctc_mean_clean", "ctc_peak_clean", "ctc_q95_clean", "refdiv_mean_clean"]


def test_wilson_puts_a_zero_count_inside_a_positive_upper_bound():
    """0 alarms out of n is an upper bound, never a rate of zero with no width.

    The whole lane exists because `FPR 0.000` gets banked. Two of the four rules measure
    exactly 0/n at the committed threshold, so the interval has to be visible.
    """
    lo, hi = wilson(0, 28313)
    assert lo == 0.0
    assert 0.0 < hi < 0.001, "a zero count must still bound the interval above zero"
    lo, hi = wilson(1481, 28313)
    assert lo < 1481 / 28313 < hi


def test_quantile_matches_numpy_by_not_interpolating_differently():
    """The page recomputes the threshold; a different quantile convention is a silent drift."""
    np = pytest.importorskip("numpy")
    values = [float(v) for v in np.random.default_rng(3).random(500)]
    for q in (0.5, 0.9, 0.95, 0.99):
        assert quantile(sorted(values), q) == pytest.approx(float(np.quantile(values, q)))


def test_seed_owner_map_uses_the_plan_not_the_seed_range():
    """Two sources have overlapping seed *ranges*; only the model list decides ownership.

    mac spans 10,264-24,607 and git spans 10,150-28,385, so a range containment test
    would label one source's models as the other's. Found while building the per-source
    table; the plan lists the models, so the plan decides.
    """
    plan_dir = tempfile.mkdtemp()
    shards = os.path.join(plan_dir, "shards")
    os.makedirs(shards)
    for shard_id, seeds in (("seed_10150_28385", [10150, 10264]), ("seed_10264_24607", [24607])):
        with open(os.path.join(shards, shard_id + ".json"), "w", encoding="utf-8") as fh:
            json.dump({"shard_id": shard_id, "models": [{"seed": s} for s in seeds]}, fh)

    owner = seed_owner_map(__import__("pathlib").Path(plan_dir))

    assert owner[10264] == "seed_10150_28385", "a seed must follow the plan's model list"
    assert owner[24607] == "seed_10264_24607"


def _fixture(tmp, *, threshold_delta=0.0):
    """A tiny but structurally real ledger + plan + receipts, for the two build tests."""
    import numpy as np

    from cviaf.lab.fpr_tpr import assign_splits

    rng = np.random.default_rng(11)
    n = 40
    records = []
    for i in range(n):
        seed = 60000 + i
        records.append(
            {
                "model_id": f"clean_none_fixed_s{seed}",
                "corpus": "distributed_clean_null",
                "kind": "clean",
                "is_positive": False,
                "split": "unassigned",
                "scores": {s: float(rng.random()) for s in SIGNALS},
            }
        )
    ledger = {"schema": "cviaf.fpr-tpr-ledger.v1", "alpha": 0.05, "records": records}
    ledger_path = os.path.join(tmp, "ledger.json")
    with open(ledger_path, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh)

    assigned = assign_splits(ledger, seed=0)["records"]
    rules = {}
    for sig in SIGNALS:
        cal = sorted(r["scores"][sig] for r in assigned if r["split"] == "calibration")
        ev = [r["scores"][sig] for r in assigned if r["split"] == "evaluation"]
        thr = quantile(cal, 1.0 - 0.05)
        hits = sum(1 for v in ev if v > thr)
        rules[sig] = {
            "status": "fpr_only",
            "threshold": thr + threshold_delta,
            "fpr": {
                # The real report publishes this rounded to 4 decimals, so the fixture has
                # to as well, or the "agrees" comparison below is never exercised.
                "point_estimate": round(hits / len(ev), 4),
                "numerator": hits,
                "ci95_wilson": wilson(hits, len(ev)),
            },
        }
    report_path = os.path.join(tmp, "report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump({"schema": "cviaf.fpr-tpr-report.v1", "alpha": 0.05, "rules": rules}, fh)

    plan_dir = os.path.join(tmp, "plan")
    os.makedirs(os.path.join(plan_dir, "shards"))
    with open(os.path.join(plan_dir, "plan.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "schema": "cviaf.analysis-plan.v1",
                "corpus_count": n,
                "shards": [
                    {
                        "shard_id": "seed_60000_60039",
                        "count": n,
                        "seed_min": 60000,
                        "seed_max": 60039,
                    }
                ],
            },
            fh,
        )
    with open(os.path.join(plan_dir, "shards", "seed_60000_60039.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "shard_id": "seed_60000_60039",
                "count": n,
                "models": [{"model_id": r["model_id"], "seed": 60000 + i} for i, r in enumerate(records)],
            },
            fh,
        )

    verify_path = os.path.join(tmp, "verify.json")
    with open(verify_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"ok": True, "sources": [{"id": "lab", "n_verified": n, "n_failed": 0, "n_missing_from_plan": 0}]},
            fh,
        )
    census_path = os.path.join(tmp, "census.json")
    with open(census_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "sources": [],
                "n_collisions": 0,
                "collisions_truncated": 0,
                "collisions_fatal": [],
                "lab_seed_range": [60000, 60039],
                "gaps": [],
            },
            fh,
        )
    return ledger_path, report_path, plan_dir, verify_path, census_path


def _args(tmp, out, ledger, report, plan, verify, census):
    return [
        "--ledger", ledger,
        "--report", report,
        "--plan", plan,
        "--verify", verify,
        "--census", census,
        "--out", out,
        "--split-seed", "0",
    ]


def test_dashboard_refuses_to_publish_a_page_that_contradicts_the_report():
    """A page whose arithmetic disagrees with the signed report is worse than no page.

    Built from a real near-miss: the first version embedded float32 scores, which moves
    the 0.05 quantile and can change the hit count by one. A reader would have seen the
    dashboard disagree with the committed report and had no way to tell which was wrong.
    """
    tmp = tempfile.mkdtemp()
    ledger, report, plan, verify, census = _fixture(tmp, threshold_delta=0.01)
    out = os.path.join(tmp, "page.html")

    with pytest.raises(SystemExit) as excinfo:
        main(_args(tmp, out, ledger, report, plan, verify, census))

    assert "threshold mismatch" in str(excinfo.value)
    assert not os.path.exists(out), "a refused build must not leave a page behind"


def test_dashboard_writes_a_self_contained_page_when_the_numbers_agree():
    tmp = tempfile.mkdtemp()
    ledger, report, plan, verify, census = _fixture(tmp)
    out = os.path.join(tmp, "page.html")

    assert main(_args(tmp, out, ledger, report, plan, verify, census)) == 0
    html = open(out, encoding="utf-8").read()

    assert '"n":40' in html, "every model must be embedded, not summarised"
    assert "http://" not in html and "https://" not in html, "the page must work offline"
    assert "TPR is absent by construction" in html


def test_a_rule_whose_rate_is_right_is_not_reported_as_disagreeing_with_the_report():
    """The page recomputes the ratio; the report publishes it rounded to 4 decimals.

    Comparing 0.05230812... against a published 0.0523 at a 1e-12 tolerance printed "NO"
    in the *agrees* column of every rule that was exactly right - a page accusing the
    report of disagreeing with itself, in the one column a sceptical reader looks at
    first. Agreement is now decided at the precision the report publishes, plus an exact
    hit-count match.
    """
    tmp = tempfile.mkdtemp()
    ledger, report, plan, verify, census = _fixture(tmp)
    out = os.path.join(tmp, "page.html")

    assert main(_args(tmp, out, ledger, report, plan, verify, census)) == 0
    html = open(out, encoding="utf-8").read()

    # The blob is embedded with compact separators, so the key has no space after the colon.
    assert '"agrees":true' in html
    assert '"agrees":false' not in html, "a correct rule must not be flagged as disagreeing"


def _ladder_receipt(tmp):
    """A minimal but structurally real per-class, per-dose receipt."""
    rules = {s: {"threshold": 0.9, "degenerate": False, "status": "measured"} for s in SIGNALS}
    rules["ctc_peak_clean"] = {"threshold": 1.0, "degenerate": True,
                              "status": "degenerate_threshold_at_ceiling"}
    cells = {
        "substitution": {
            "0.25": {"n": 94, "n_behaviour_inert": 40, "median_f1_relative_drop": 0.6,
                     "rules": {s: {"caught": 45, "tpr": 45 / 94, "ci95_wilson": [0.39, 0.59],
                                    "conclusion": "measured", "n_below_floor": False}
                               for s in SIGNALS if s != "ctc_peak_clean"}},
        },
        "bias_lift": {
            "0.25": {"n": 99, "n_behaviour_inert": 99, "median_f1_relative_drop": 0.0,
                     "rules": {s: {"caught": 5, "tpr": 5 / 99, "ci95_wilson": [0.02, 0.11],
                                    "conclusion": "measured", "n_below_floor": False}
                               for s in SIGNALS if s != "ctc_peak_clean"}},
        },
    }
    path = os.path.join(tmp, "ladder.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({
            "schema": "cviaf.tpr-ladder.v1",
            "n_arms": 193,
            "kinds": ["bias_lift", "substitution"],
            "doses": [0.25],
            "units_per_kind": {"bias_lift": "absolute logit units",
                              "substitution": "fraction of hidden units zeroed"},
            "min_positives": 20,
            "rules": rules,
            "cells": cells,
            "unscorable": {"substitution@0.25": 5},
            "pooled_estimate": None,
            "pooled_refusal": "no pooled rate across attack classes or doses",
            "behaviour_metric": "f1_relative_drop; arms that did not move it are inert",
            "problems": [],
        }, fh)
    return path


def test_the_detection_panel_refuses_a_single_dose_receipt():
    """A per-dose table cannot be built from a one-dose receipt without inventing rows.

    Inventing a row is the failure this page exists to prevent, so the build refuses rather
    than drawing an empty cell that reads as "nothing found here".
    """
    tmp = tempfile.mkdtemp()
    ledger, report, plan, verify, census = _fixture(tmp)
    out = os.path.join(tmp, "page.html")
    single = os.path.join(tmp, "single.json")
    with open(single, "w", encoding="utf-8") as fh:
        json.dump({"schema": "cviaf.tpr-at-frozen.v1", "n_arms": 193, "kinds": ["substitution"],
                   "rules": {}}, fh)

    with pytest.raises(SystemExit) as excinfo:
        main(_args(tmp, out, ledger, report, plan, verify, census) + ["--tpr", single])

    assert "cviaf.tpr-ladder.v1" in str(excinfo.value)
    assert not os.path.exists(out), "a refused build must not leave a page behind"


def test_the_detection_panel_carries_every_class_and_dose_and_no_pooled_rate():
    """The panel is per-cell by construction: no pooled number is even shipped to it."""
    tmp = tempfile.mkdtemp()
    ledger, report, plan, verify, census = _fixture(tmp)
    out = os.path.join(tmp, "page.html")
    ladder = _ladder_receipt(tmp)

    assert main(_args(tmp, out, ledger, report, plan, verify, census) + ["--tpr", ladder]) == 0
    html = open(out, encoding="utf-8").read()

    assert '"bias_lift"' in html and '"substitution"' in html
    assert '"cells"' in html, "the panel is driven by cells, not by a rule-level rate"
    assert "pooled_estimate" not in html, (
        "no pooled rate may reach the page: a blank or averaged cell is what a reader "
        "quotes when the table gets awkward"
    )
    assert "pooled_refusal" in html, "the refusal has to be visible, not just absent"


def test_score_blob_round_trips_exactly():
    """float64, not float32: the threshold must survive the embed."""
    np = pytest.importorskip("numpy")
    values = np.array([0.9845092069058692, 0.1, 1.0 / 3.0])
    encoded = b64_floats(values)
    import base64

    assert np.array_equal(np.frombuffer(base64.b64decode(encoded), dtype="<f8"), values)
    flags = b64_bits([1, 0, 1, 1, 0, 0, 0, 1, 1])
    raw = np.unpackbits(np.frombuffer(base64.b64decode(flags), dtype="uint8"))[:9]
    assert list(raw) == [1, 0, 1, 1, 0, 0, 0, 1, 1]
