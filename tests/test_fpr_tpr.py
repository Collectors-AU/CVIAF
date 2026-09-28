"""Tests for the FPR/TPR harness.

Fixture-based on purpose: the harness is the gate every future measurement passes
through, so its own correctness must not depend on any corpus being on disk. The
validator tests are written as the specific failures the lane has already suffered
(16 tamper arms counted as negatives; a denominator of 8 reported as an FPR).
"""
from __future__ import annotations

import json
import os

import pytest

from cviaf.lab.fpr_tpr import (LEDGER_SCHEMA, LedgerError, assign_splits,
                              evaluate_corpus, exact_upper_bound, rate,
                              validate_ledger, wilson_interval)


def rec(mid, kind, score, split="unassigned", is_positive=None, corpus="c", **extra):
    if is_positive is None:
        is_positive = kind != "clean"
    r = {"model_id": mid, "corpus": corpus, "kind": kind, "split": split,
         "is_positive": is_positive, "scores": {"signal": score}}
    r.update(extra)
    return r


def ledger(records, positive_kinds=("attack",), negative_kinds=("clean",),
           alpha=0.05, higher=True):
    return {"schema": LEDGER_SCHEMA, "alpha": alpha,
            "higher_is_more_anomalous": higher,
            "positive_kinds": list(positive_kinds),
            "negative_kinds": list(negative_kinds),
            "provenance": {"producer": "test"},
            "records": records}


def make_population(n_neg=40, n_pos=20, sep=1.0, noise=0.1, seed=0, **kw):
    import numpy as np
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n_neg):
        recs.append(rec(f"clean_s{i}", "clean", float(rng.normal(0.0, noise))))
    for i in range(n_pos):
        recs.append(rec(f"attack_s{i}", "attack",
                        float(rng.normal(sep, noise))))
    return ledger(recs, **kw)


# --------------------------------------------------------------------------- #
# validator: the failures this lane actually had
# --------------------------------------------------------------------------- #

def test_valid_ledger_has_no_problems():
    assert validate_ledger(make_population()) == []


def test_mislabelled_arm_is_rejected_not_warned():
    """The compare.py failure: a tamper arm scored as a negative.

    16 weight-space arms were silently counted as negatives because the module's
    own attack-kind tuple did not list them. The ledger now refuses to score a
    record whose polarity contradicts its declared kind.
    """
    bad = make_population()
    bad["records"][-1]["is_positive"] = False        # attack declared as negative
    problems = validate_ledger(bad)
    assert any("mislabelled" in p for p in problems), problems


def test_undeclared_kind_is_a_hard_error_not_a_negative():
    """The compare.py bug, reproduced at the ledger level.

    There, `weight_tamper` was absent from the module's attack-kind tuple and was
    therefore silently scored as a negative, producing "n=0 attacked, 24 negative".
    A kind declared in neither list must now be refused outright.
    """
    bad = make_population()
    bad["records"][0]["kind"] = "weight_tamper"       # in neither list
    problems = validate_ledger(bad)
    assert any("declared neither" in p for p in problems), problems

    # Declaring it as a positive while the record says negative is the *other*
    # half of the same failure and must also be refused.
    bad2 = make_population(positive_kinds=("attack", "weight_tamper"))
    bad2["records"][0]["kind"] = "weight_tamper"      # is_positive is still False
    assert any("mislabelled" in p for p in validate_ledger(bad2))

    # And with the kind declared positive on both sides it validates.
    bad2["records"][0]["is_positive"] = True
    assert validate_ledger(bad2) == []


def test_a_kind_cannot_be_both_positive_and_negative():
    bad = make_population(negative_kinds=("clean", "attack"))
    assert any("declared positive AND negative" in p for p in validate_ledger(bad))


def test_negative_kinds_list_is_required():
    led = make_population()
    del led["negative_kinds"]
    assert any("negative_kinds must be" in p for p in validate_ledger(led))


def test_duplicate_asset_is_rejected():
    dupe = make_population()
    dupe["records"].append(dict(dupe["records"][0]))
    assert any("duplicate" in p for p in validate_ledger(dupe))


def test_non_finite_score_is_rejected():
    bad = make_population()
    bad["records"][3]["scores"]["signal"] = float("nan")
    assert any("not finite" in p for p in validate_ledger(bad))
    bad["records"][3]["scores"]["signal"] = float("inf")
    assert any("not finite" in p for p in validate_ledger(bad))


def test_partial_signal_coverage_is_rejected():
    """Unequal denominators between rules would make a TPR comparison meaningless."""
    bad = make_population()
    bad["records"][5]["scores"]["other"] = 0.5
    assert any("not present on every record" in p for p in validate_ledger(bad))


def test_schema_and_direction_are_required():
    bad = make_population()
    bad["schema"] = "something-else"
    assert any("schema must be" in p for p in validate_ledger(bad))
    bad2 = make_population()
    del bad2["higher_is_more_anomalous"]
    assert any("higher_is_more_anomalous" in p for p in validate_ledger(bad2))
    bad3 = make_population()
    bad3["extra_key"] = 1
    assert any("unknown top-level keys" in p for p in validate_ledger(bad3))


def test_alpha_out_of_range_is_rejected():
    bad = make_population(alpha=1.5)
    assert any("alpha" in p for p in validate_ledger(bad))


def test_every_positive_kind_present_in_the_population_is_covered():
    """A declared list that no record uses is harmless; an unused *record* kind is
    the dangerous direction and is covered above."""
    led = make_population(positive_kinds=("attack", "oda", "oga"),
                          negative_kinds=("clean", "label_flip"))
    assert validate_ledger(led) == []


# --------------------------------------------------------------------------- #
# intervals
# --------------------------------------------------------------------------- #

def test_wilson_interval_is_bounded_and_correct_at_the_extremes():
    lo, hi = wilson_interval(0, 50)
    assert lo == 0.0 and 0.0 < hi < 0.15
    lo, hi = wilson_interval(50, 50)
    assert hi == 1.0 and 0.85 < lo < 1.0
    lo, hi = wilson_interval(5, 10)
    assert lo < 0.5 < hi
    # A detector is never allowed to claim a point estimate with no uncertainty.
    assert hi - lo > 0


def test_exact_upper_bound_matches_the_known_zero_case():
    # The repo reports 0/N as its exact upper bound; 0/8 must be ~31%, not 0%.
    assert abs(exact_upper_bound(0, 8) - (1 - 0.05 ** (1 / 8))) < 1e-12
    assert 0.30 < exact_upper_bound(0, 8) < 0.32
    assert exact_upper_bound(0, 50) < exact_upper_bound(0, 8)
    assert exact_upper_bound(5, 5) == 1.0


def test_rate_refuses_below_the_denominator_floor():
    """The 0/8 "FPR" from the Task 3 battery: not a measurement, a bound."""
    r = rate(0, 8, min_n=20)
    assert r["status"] == "insufficient_denominator"
    assert r["point_estimate"] is None and r["ci95_wilson"] is None
    assert 0.30 < r["exact_upper_bound_95"] < 0.32
    assert "refusal" in r

    ok = rate(0, 50, min_n=20)
    assert ok["status"] == "measured" and ok["point_estimate"] == 0.0


# --------------------------------------------------------------------------- #
# splitting discipline
# --------------------------------------------------------------------------- #

def test_calibration_and_evaluation_are_disjoint_models():
    led = assign_splits(make_population(), seed=0)
    cal = {r["model_id"] for r in led["records"] if r["split"] == "calibration"}
    ev = {r["model_id"] for r in led["records"] if r["split"] == "evaluation"}
    assert cal and ev
    assert not (cal & ev), "a model was used for both calibration and scoring"


def test_split_keeps_both_polarities_in_both_halves():
    led = assign_splits(make_population(n_neg=40, n_pos=20), seed=0)
    for split in ("calibration", "evaluation"):
        pos = [r for r in led["records"] if r["split"] == split and r["is_positive"]]
        neg = [r for r in led["records"] if r["split"] == split and not r["is_positive"]]
        assert pos, f"no positives in {split}"
        assert neg, f"no negatives in {split}"


def test_split_is_deterministic_for_a_seed():
    a = assign_splits(make_population(), seed=7)
    b = assign_splits(make_population(), seed=7)
    assert [r["split"] for r in a["records"]] == [r["split"] for r in b["records"]]


def test_explicit_splits_are_respected():
    recs = [rec(f"clean_s{i}", "clean", 0.1, split="calibration") for i in range(6)]
    recs += [rec(f"attack_s{i}", "attack", 0.9, split="evaluation") for i in range(4)]
    report = evaluate_corpus(ledger(recs), min_negatives=5)
    assert report["denominators"] == {"calibration_negatives": 6,
                                     "evaluation_negatives": 0,
                                     "evaluation_positives": 4}


# --------------------------------------------------------------------------- #
# end-to-end behaviour
# --------------------------------------------------------------------------- #

def test_separable_population_gives_high_tpr_and_low_fpr():
    report = evaluate_corpus(make_population(sep=3.0, noise=0.1), min_negatives=10)
    r = report["rules"]["signal"]
    assert r["status"] == "measured"
    assert r["tpr"]["point_estimate"] > 0.8
    assert r["fpr"]["point_estimate"] < 0.2
    assert report["fpr_measured"] is True
    assert "FPR is measured" in report["headline"]


def test_overlapping_population_is_not_reported_as_detection():
    report = evaluate_corpus(make_population(sep=0.0, noise=1.0), min_negatives=10)
    r = report["rules"]["signal"]
    # A rule with no signal must not exceed chance by much at alpha=.05.
    assert r["tpr"]["point_estimate"] < 0.6


def test_small_clean_population_yields_bound_not_rate():
    """The Task 3 corpus (11 clean models) may only state a bound."""
    recs = [rec(f"clean_s{i}", "clean", 0.1, split="evaluation") for i in range(8)]
    recs += [rec(f"cal_s{i}", "clean", 0.1, split="calibration") for i in range(6)]
    recs += [rec(f"attack_s{i}", "attack", 0.9, split="evaluation") for i in range(6)]
    report = evaluate_corpus(ledger(recs), min_negatives=20)
    assert report["fpr_measured"] is False
    assert "NOT MEASURED" in report["headline"]
    f = report["rules"]["signal"]["fpr"]
    assert f["point_estimate"] is None and f["status"] == "insufficient_denominator"
    assert 0.0 < f["exact_upper_bound_95"] < 0.4


def test_lower_is_more_anomalous_direction_is_honoured():
    """A duplicate-distance signal is anomalous in the LOW tail."""
    recs = [rec(f"clean_s{i}", "clean", 0.5, split="calibration") for i in range(10)]
    recs += [rec(f"clean_e{i}", "clean", 0.5, split="evaluation") for i in range(25)]
    recs += [rec(f"attack_s{i}", "attack", 0.05, split="evaluation") for i in range(10)]
    report = evaluate_corpus(ledger(recs, higher=False), min_negatives=20)
    r = report["rules"]["signal"]
    assert r["tpr"]["point_estimate"] == 1.0
    assert r["fpr"]["point_estimate"] <= 0.05

    # Same ledger, wrong direction: the rule must NOT find the anomaly.
    wrong = evaluate_corpus(ledger(recs, higher=True), min_negatives=20)
    assert wrong["rules"]["signal"]["tpr"]["point_estimate"] < 0.5


def test_refuses_when_calibration_is_too_small_to_place_a_threshold():
    recs = [rec("clean_a", "clean", 0.1, split="calibration")]
    recs += [rec(f"clean_s{i}", "clean", 0.1, split="evaluation") for i in range(25)]
    recs += [rec(f"attack_s{i}", "attack", 0.9, split="evaluation") for i in range(5)]
    report = evaluate_corpus(ledger(recs), min_negatives=20)
    assert report["rules"]["signal"]["status"] == "refused"


def test_invalid_ledger_raises_and_cli_returns_3(tmp_path):
    from cviaf.lab.fpr_tpr import main
    bad = make_population()
    bad["records"][0]["is_positive"] = not bad["records"][0]["is_positive"]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    assert main(["--input", str(p), "--validate-only"]) == 3


def test_cli_writes_a_report(tmp_path):
    from cviaf.lab.fpr_tpr import main
    p = tmp_path / "led.json"
    p.write_text(json.dumps(make_population()))
    out = tmp_path / "report.json"
    assert main(["--input", str(p), "--out", str(out), "--min-negatives", "5"]) == 0
    report = json.loads(out.read_text())
    assert report["schema"] == "cviaf.fpr-tpr-report.v1"
    assert report["fpr_measured"] is True


# --------------------------------------------------------------------------- #
# recall by attack kind, and clause 3.7 (ODA recall > 0)
# --------------------------------------------------------------------------- #

def kind_ledger(oda_scores=(5.0,), other_kinds=("gma",), n_neg=30, signal="sig"):
    """A ledger with explicit splits, so the fixtures do not depend on a split seed."""
    recs = []
    for i in range(n_neg):
        recs.append(rec(f"clean_s{i}", "clean", 0.0,
                        split="calibration" if i < n_neg // 2 else "evaluation"))
    for i, s in enumerate(oda_scores):
        recs.append(rec(f"oda_s{i}", "oda", s, split="evaluation"))
    for k in other_kinds:
        recs.append(rec(f"{k}_s0", k, 5.0, split="evaluation"))
        recs.append(rec(f"{k}_s1", k, 0.0, split="evaluation"))
    return ledger(recs, positive_kinds=("oda",) + tuple(other_kinds))


def test_per_kind_recall_uses_the_corpus_threshold():
    from cviaf.lab.fpr_tpr import evaluate_corpus
    report = evaluate_corpus(kind_ledger(oda_scores=(5.0, 0.0)), min_negatives=10)
    per_kind = report["per_kind"]["signal"]
    assert per_kind["threshold"] == report["rules"]["signal"]["threshold"]
    counts = {k: (v["tp"], v["n_evaluation"]) for k, v in per_kind["kinds"].items()}
    assert counts == {"gma": (1, 2), "oda": (1, 2)}


def test_a_kind_with_no_evaluation_assets_is_not_measured_not_zero_recall():
    """"We caught none" and "we never looked" are different claims."""
    from cviaf.lab.fpr_tpr import evaluate_corpus
    led = kind_ledger(oda_scores=(5.0,))
    led["records"].append(rec("rma_cal", "rma", 5.0, split="calibration"))
    led["positive_kinds"] = list(led["positive_kinds"]) + ["rma"]
    report = evaluate_corpus(led, min_negatives=10)
    entry = report["per_kind"]["signal"]["kinds"]["rma"]
    assert entry["status"] == "not_measured" and entry["n_evaluation"] == 0
    assert "tpr" not in entry


def test_clause_3_7_is_satisfied_when_the_oda_arm_fires():
    from cviaf.lab.fpr_tpr import evaluate_corpus
    report = evaluate_corpus(kind_ledger(oda_scores=(5.0,)), min_negatives=10)
    check = report["clause_checks"]["3.7_oda_recall"]
    assert check["satisfied"] is True and check["best_recall"] == 1.0
    assert check["best_signal"] == "signal"
    assert check["recall_per_signal"]["signal"] == 1.0
    assert "optimistic" in check["caveat"]


def test_clause_3_7_is_not_satisfied_when_no_oda_arm_fires():
    from cviaf.lab.fpr_tpr import evaluate_corpus
    report = evaluate_corpus(kind_ledger(oda_scores=(0.0, 0.0)), min_negatives=10)
    check = report["clause_checks"]["3.7_oda_recall"]
    assert check["satisfied"] is False and check["best_recall"] == 0.0
    assert check["recall_per_signal"]["signal"] == 0.0


def test_the_real_report_measures_oda_recall_and_says_which_signal():
    """The on-disk report must carry the clause check, whichever way it reads."""
    path = "runs/fpr_ledger_report.json"
    if not os.path.isfile(path):
        pytest.skip("report not built yet")
    with open(path, encoding="utf-8") as fh:
        report = json.load(fh)
    check = report["clause_checks"]["3.7_oda_recall"]
    assert check["satisfied"] in (True, False)
    assert "oda" in report["per_kind"]["refdiv_mean_clean"]["kinds"]
    assert report["per_kind"]["refdiv_mean_clean"]["kinds"]["oda"]["n_evaluation"] > 0


def test_render_prints_per_kind_recall_and_the_clause():
    from cviaf.lab.fpr_tpr import evaluate_corpus, render_report
    text = render_report(evaluate_corpus(kind_ledger(oda_scores=(5.0,)),
                                         min_negatives=10))
    assert "recall by attack kind" in text
    assert "oda=1/1" in text
    assert "clause 3.7 ODA recall > 0: satisfied" in text


def test_smoke_on_the_real_corpus_when_present():
    """If a ledger produced from the on-disk corpora exists, it must validate and
    score. This is the check that the harness works on real artefacts, not just
    fixtures."""
    ledger_path = "runs/fpr_ledger.json"
    if not os.path.isfile(ledger_path):
        pytest.skip("runs/fpr_ledger.json not built yet")
    from cviaf.lab.fpr_tpr import load_ledger
    led = load_ledger(ledger_path)
    report = evaluate_corpus(led)
    assert report["denominators"]["evaluation_positives"] > 0
    for name, r in report["rules"].items():
        if r.get("status") == "refused":
            continue
        # every reported rate carries a denominator
        assert r["tpr"]["denominator"] >= 0 and r["fpr"]["denominator"] >= 0
        if r["tpr"]["point_estimate"] is not None:
            assert r["tpr"]["ci95_wilson"] is not None
