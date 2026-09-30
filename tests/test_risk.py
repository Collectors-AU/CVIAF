"""Tests for expected-loss risk (clauses 1.4, 6.4, 6.6).

The claims to protect are narrow and checkable: the posterior is the one the measured
operating characteristics actually identify (PPV/NPV), every finding carries expected
loss for all three dispositions, severity is monotone in expected loss, and changing
the loss matrix changes the disposition in a logged way.
"""
from __future__ import annotations

import json
import os

import pytest

from cviaf.lab.review import OperatorCosts
from cviaf.lab.risk import (ASSET_RISK_SCHEMA, DEFAULT_ACCEPT_BOUND, DEFAULT_PREVALENCE,
                            POLICIES, asset_risk, break_even_prevalence, disposition,
                            main, policy_expected_loss, posterior_flagged,
                            posterior_unflagged, render, rule_risk)

COSTS = OperatorCosts()


# --------------------------------------------------------------------------- #
# the posterior that is identifiable
# --------------------------------------------------------------------------- #

def test_ppv_matches_its_definition():
    # pi=.1, TPR=.9, FPR=.1 -> .09 / (.09 + .09) = .5
    assert posterior_flagged(0.9, 0.1, 0.1) == pytest.approx(0.5)


def test_a_perfect_rule_has_a_certain_posterior_on_both_sides():
    assert posterior_flagged(1.0, 0.0, 0.1) == pytest.approx(1.0)
    assert posterior_unflagged(1.0, 0.0, 0.1) == pytest.approx(0.0)


def test_a_useless_rule_is_informative_nothing():
    """TPR == FPR: flagging tells you nothing, so the posterior is the prevalence."""
    assert posterior_flagged(0.3, 0.3, 0.2) == pytest.approx(0.2)
    assert posterior_unflagged(0.3, 0.3, 0.2) == pytest.approx(0.2)


def test_prevalence_must_matter():
    low = posterior_flagged(0.5, 0.1, 0.01)
    high = posterior_flagged(0.5, 0.1, 0.5)
    assert low < high


def test_a_rule_that_never_fires_has_no_flagged_posterior():
    assert posterior_flagged(0.0, 0.0, 0.1) is None
    assert posterior_unflagged(1.0, 1.0, 0.1) is None


# --------------------------------------------------------------------------- #
# dispositions
# --------------------------------------------------------------------------- #

def test_expected_loss_is_present_for_all_three_dispositions():
    """Clause 1.4's acceptance test."""
    found = asset_risk("m0", "weight_tamper", 0.001, 0.05, 0.2, 0.07)
    assert set(found["expected_loss"]) == set(POLICIES)
    for value in found["expected_loss"].values():
        assert isinstance(value, float)


def test_quarantining_a_flagged_asset_is_preferred_when_the_posterior_is_high():
    found = asset_risk("m0", "weight_tamper", 0.001, 0.05, 0.9, 0.02,
                       prevalence=0.3)
    assert found["flagged"] is True and found["posterior"] > 0.9
    assert found["disposition"] == "quarantine"


def test_an_unflagged_asset_is_accepted_when_the_rule_is_good():
    found = asset_risk("m0", "clean", 0.9, 0.05, 0.9, 0.02, prevalence=0.1)
    assert found["flagged"] is False and found["disposition"] == "accept"


def test_accept_is_demoted_to_review_on_thin_evidence():
    """A cheap loss matrix must not license stopping the investigation."""
    # with the default matrix review is cheap enough to win on its own, so use one
    # where accepting is genuinely the cheapest action (false_accept 1)
    cheap = OperatorCosts(false_accept=1.0, false_quarantine=10.0, review=1.0)
    found = asset_risk("m0", "clean", 0.9, 0.05, 0.3, 0.3, prevalence=0.5, costs=cheap)
    assert found["posterior"] == pytest.approx(0.5)
    assert found["posterior"] > DEFAULT_ACCEPT_BOUND
    assert found["expected_loss"]["accept"] < found["expected_loss"]["quarantine"]
    assert found["disposition"] == "review" and found["demoted_from"] == "accept"
    assert "accept demoted" in found["reason"]


def test_accept_is_refused_outright_when_not_permitted():
    decision = disposition(1e-9, COSTS, accept_permitted=False)
    assert decision["disposition"] == "review"
    assert decision["demoted_from"] == "accept"
    assert len(decision["expected_loss"]) == 3


def test_an_unidentified_posterior_does_not_fabricate_losses():
    decision = disposition(None, COSTS)
    assert all(v is None for v in decision["expected_loss"].values())
    assert "not identified" in decision["reason"]


def test_review_wins_an_exact_tie():
    """The informative action is preferred, matching review.plan_review."""
    # a posterior where review and quarantine cost the same: review + e*FA == (1-q)*FQ
    # with the default matrix (FA 25, FQ 5, review 1, e 0) this needs q = 0.8
    decision = disposition(0.8, COSTS)
    assert decision["expected_loss"]["review"] == pytest.approx(
        decision["expected_loss"]["quarantine"])
    assert decision["disposition"] == "review"


def test_changing_the_loss_matrix_changes_the_disposition():
    """Clause 6.6: the disposition must follow the operator's declared costs."""
    expensive_removal = OperatorCosts(false_accept=5.0, false_quarantine=100.0)
    cheap_removal = OperatorCosts(false_accept=100.0, false_quarantine=1.0)
    q = 0.5
    assert disposition(q, cheap_removal, accept_permitted=False)["disposition"] \
        == "quarantine"
    assert disposition(q, expensive_removal, accept_permitted=False)["disposition"] \
        == "review"


def test_severity_is_monotone_in_expected_loss():
    """Clause 6.4: severity is derived from expected loss, so it must not invert.

    Severity is read off the risk EXPOSURE (what accepting the asset would cost), not
    off the cost of the best action -- quarantining overwhelming evidence is cheap, so
    "loss if acted" is deliberately not monotone in the evidence.
    """
    found = [asset_risk(f"m{rank}", "weight_tamper", p, 0.05, 0.95, 0.005,
                        prevalence=0.2)
             for rank, p in enumerate((0.9, 0.5, 0.4, 0.01))]
    exposure = [f["expected_loss"]["accept"] for f in found]
    assert exposure == sorted(exposure) and exposure[0] < exposure[-1]
    severity = {"accept": 0, "review": 1, "quarantine": 2}
    order = [severity[f["disposition"]] for f in found]
    assert order == sorted(order), [f["disposition"] for f in found]
    assert order[-1] > order[0]


# --------------------------------------------------------------------------- #
# policies
# --------------------------------------------------------------------------- #

def test_accept_all_costs_the_prevalence_times_the_catastrophe():
    losses = policy_expected_loss(0.0, 0.0, 0.1, COSTS)
    assert losses["accept_all"] == pytest.approx(0.1 * COSTS.false_accept)


def test_a_perfect_rule_beats_accept_all_at_any_prevalence():
    for pi in (0.001, 0.1, 0.5):
        losses = policy_expected_loss(1.0, 0.0, pi, COSTS)
        assert losses["quarantine_flagged"] < losses["accept_all"]


def test_a_useless_rule_ties_accept_all_and_is_reported_as_tied():
    out = rule_risk(0.0, 0.0, 0.1, COSTS)
    assert out["expected_loss_per_asset"]["accept_all"] == pytest.approx(
        out["expected_loss_per_asset"]["review_flagged"])
    assert out["recommended_policy"] == "accept_all"      # fewest interventions
    assert len(out["tied_policies"]) == 3
    assert "never fires" in out["note"]


def test_a_noisy_rule_can_be_worse_than_shipping_everything():
    """The result the lane needed: FPR 0.27 makes the rule net-harmful at pi=0.1."""
    out = rule_risk(0.056, 0.267, 0.1, COSTS)
    assert out["recommended_policy"] == "accept_all"
    assert out["break_even_prevalence_vs_accept_all"] > 0.4


def test_a_discriminating_rule_pays_for_itself_above_a_break_even_prevalence():
    out = rule_risk(0.222, 0.067, 0.1, COSTS)
    assert out["recommended_policy"] == "quarantine_flagged"
    be = out["break_even_prevalence_vs_accept_all"]
    assert 0.0 < be < 0.1
    # below the break-even the rule is not worth running, above it, it is
    assert rule_risk(0.222, 0.067, be / 2, COSTS)["recommended_policy"] == "accept_all"
    assert rule_risk(0.222, 0.067, be * 2, COSTS)["recommended_policy"] \
        == "quarantine_flagged"


def test_review_is_worth_its_cost_only_when_a_mistake_is_expensive():
    cheap_mistake = OperatorCosts(false_accept=1.0, false_quarantine=5.0, review=1.0)
    losses = policy_expected_loss(0.5, 0.1, 0.2, cheap_mistake)
    assert losses["accept_all"] < losses["review_flagged"]


def test_break_even_returns_none_when_the_policies_never_cross():
    """A policy that is worse at every prevalence has no break-even point."""
    expensive_review = OperatorCosts(review=100.0)
    assert break_even_prevalence(0.5, 0.1, expensive_review, "review_flagged",
                                 "accept_all") is None
    # and a rule that never fires does cross: at pi=1 it is exactly as bad
    assert break_even_prevalence(0.0, 0.0, COSTS, "quarantine_flagged",
                                 "accept_all") == 0.0


def test_a_bound_instead_of_a_point_estimate_is_not_priced():
    out = rule_risk(None, 0.0, 0.1, COSTS)
    assert out["status"] == "not_measured" and "bound at" in out["reason"]
    out2 = rule_risk(0.2, None, 0.1, COSTS)
    assert out2["status"] == "not_measured"


def test_totals_scale_with_the_number_of_assets():
    one = rule_risk(0.2, 0.05, 0.1, COSTS, n_assets=1)
    ten = rule_risk(0.2, 0.05, 0.1, COSTS, n_assets=10)
    assert ten["expected_total_loss"]["accept_all"] == pytest.approx(
        10 * one["expected_total_loss"]["accept_all"])


def test_schema_and_loss_matrix_are_recorded():
    out = rule_risk(0.2, 0.05)
    assert out["schema"] == ASSET_RISK_SCHEMA
    assert out["loss_matrix"] == COSTS.to_dict()
    assert out["prevalence"] == DEFAULT_PREVALENCE
    assert "decision-conditioned" in out["caveat"]


def test_render_shows_the_policies_the_break_even_and_the_note():
    text = render(rule_risk(0.0, 0.0, 0.1, COSTS))
    assert "break-even prevalence" in text and "never fires" in text
    assert "not measured" in render(rule_risk(None, None))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_cli_prices_a_rule(tmp_path, capsys):
    out = tmp_path / "risk.json"
    assert main(["--tpr", "0.222", "--fpr", "0.067", "--prevalence", "0.1",
                 "--json", str(out)]) == 0
    written = json.loads(out.read_text())
    assert written["recommended_policy"] == "quarantine_flagged"
    capsys.readouterr()


def test_cli_refuses_without_inputs(capsys):
    assert main([]) == 2
    capsys.readouterr()


def test_cli_prices_every_signal_of_a_real_report(tmp_path, capsys):
    report = "runs/fpr_ledger_report.json"
    if not os.path.isfile(report):
        pytest.skip("FPR report not built yet")
    out = tmp_path / "risk.json"
    assert main(["--report", report, "--prevalence", "0.1", "--json", str(out)]) == 0
    capsys.readouterr()
    written = json.loads(out.read_text())
    assert set(written["per_signal"]) == {"ctc_mean_clean", "ctc_peak_clean",
                                          "ctc_q95_clean", "refdiv_mean_clean"}
    priced = written["per_signal"]["refdiv_mean_clean"]
    assert priced["status"] == "measured"
    # the saturated pair is priced as a rule that never fires, not as a detector
    assert written["per_signal"]["ctc_q95_clean"]["recommended_policy"] == "accept_all"


def test_cli_rejects_an_unknown_signal(capsys):
    report = "runs/fpr_ledger_report.json"
    if not os.path.isfile(report):
        pytest.skip("FPR report not built yet")
    assert main(["--report", report, "--signal", "nope"]) == 2
    capsys.readouterr()
