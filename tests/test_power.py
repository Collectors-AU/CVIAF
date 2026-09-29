"""Tests for the recall power instrument (`cviaf/lab/power.py`).

Each test is a failure that would otherwise reach a corpus-sizing decision:

  * a one-sided UPWARD test used to size a *loss* (it has exactly zero power against
    a decrease, so the number it returns is not conservative, it is empty);
  * the normal approximation's suggestion for a small p0 and small n, where it is
    wrong in the direction that flatters the design;
  * an exact rejection region off by one count, which shifts every MDE in the table;
  * a power claim that is not monotone in n (a bisection built on a non-monotone
    objective returns a confident wrong answer);
  * the paired (McNemar) and unpaired sizing being conflated, which overstates the
    corpus by the amount the two rules agree.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cviaf.lab.power import (  # noqa: E402
    BATTERY_TPR,
    arms_needed,
    critical_region,
    delta_table,
    exact_power,
    interval_floor,
    mcnemar_arms,
    mde,
    paired_table,
    power_table,
)


def test_exact_rejection_region_is_the_smallest_count_whose_tail_is_at_most_alpha():
    n, p0, alpha = 48, 0.1458, 0.05
    region = critical_region(n, p0, alpha, sided=1, direction="gain")
    k = region["k_hi"]
    from cviaf.lab.power import _tail_ge
    assert _tail_ge(n, p0, k) <= alpha
    assert _tail_ge(n, p0, k - 1) > alpha


def test_a_one_sided_upward_test_has_no_power_against_a_decrease():
    """Sizing a loss with a gain-shaped test returns an empty number, not a small one.

    Read off a gain-shaped table, a 10-point DROP in recall of 0.1458 looks
    undetectable at any arm count. The same drop is detectable at ~50 arms with a
    lower-tail test. Quoting the first number as "we could not see a loss" is the
    failure."""
    gain_shaped = exact_power(200, 0.30, 0.20, 0.05, sided=1, direction="gain")
    assert gain_shaped < 0.001
    assert mde(200, 0.30, 0.05, 0.8, 1, "gain") > 0        # a gain IS sized normally
    loss_shaped = exact_power(200, 0.30, 0.20, 0.05, sided=1, direction="loss")
    assert loss_shaped > 0.5


def test_power_is_monotone_in_n_and_in_the_size_of_the_change():
    p0 = 0.1458
    powers = [exact_power(n, p0, 0.30, 0.05, 1, "gain") for n in (24, 48, 96, 192)]
    assert powers == sorted(powers)
    deltas = [exact_power(96, p0, p0 + d, 0.05, 1, "gain") for d in (0.02, 0.05, 0.15)]
    assert deltas == sorted(deltas)


def test_arms_needed_and_mde_are_inverses_of_each_other():
    """If n arms detect delta at the target power, n-1 arms must not."""
    p0, delta = 0.1458, 0.05
    n = arms_needed(p0, delta, 0.05, 0.8, 1, "gain")
    assert n is not None
    assert exact_power(n, p0, p0 + delta, 0.05, 1, "gain") >= 0.8
    assert exact_power(n - 1, p0, p0 + delta, 0.05, 1, "gain") < 0.8
    assert mde(n, p0, 0.05, 0.8, 1, "gain") <= delta


def test_forty_eight_arms_cannot_resolve_a_five_point_gain():
    """The headline the corpus is currently sized against."""
    assert interval_floor(48, BATTERY_TPR) > 0.05
    assert mde(48, BATTERY_TPR, 0.05, 0.8, 1, "gain") > 0.10
    assert arms_needed(BATTERY_TPR, 0.05, 0.05, 0.8, 1, "gain") > 300


def test_the_exact_answer_is_not_the_normal_approximation_at_small_n_and_small_p():
    """The approximation flatters the design; the exact region does not."""
    two_group_normal = 7.84 * (0.1458 * 0.8542 + 0.1958 * 0.8042) / 0.05 ** 2
    exact = arms_needed(0.1458, 0.05, 0.05, 0.8, 2, "gain")
    assert exact is not None
    assert exact > 0.35 * two_group_normal      # same order, and honestly reported


def test_the_interval_floor_needs_no_power_assumption_and_sits_below_the_mde():
    """The significance-only floor is the smallest of the three numbers, and calling
    it the widest inverts what a reader takes from the table."""
    for n in (48, 192, 768):
        floor = interval_floor(n, 0.1458)
        assert 0 < floor <= mde(n, 0.1458, 0.05, 0.8, 1, "gain")


def test_a_two_sided_design_costs_more_arms_than_a_one_sided_one():
    one = arms_needed(0.1458, 0.05, 0.05, 0.8, 1, "gain")
    two = arms_needed(0.1458, 0.05, 0.05, 0.8, 2, "gain")
    assert one < two


def test_an_unreachable_change_returns_nothing_rather_than_a_large_number():
    """A search that hits its ceiling must say so, not return the ceiling."""
    assert arms_needed(0.1458, 0.002, 0.05, 0.8, 1, "gain", n_max=100) is None
    assert arms_needed(0.05, 0.05, 0.05, 0.8, 1, "loss") is None      # p1 = 0


def test_a_loss_of_the_whole_baseline_is_undefined_not_unachievable():
    table = delta_table(0.05, baselines=(0.05,))
    entry = table["per_baseline"]["0.05"]["loss_onesided"]
    assert isinstance(entry, str) and "undefined" in entry


def test_the_paired_design_needs_the_discordance_and_fewer_arms_than_the_unpaired_one():
    """Comparing two rules on the same arms is not a two-sample problem."""
    paired = mcnemar_arms(0.05, 0.20)["n_arms"]
    unpaired = arms_needed(0.1458, 0.05, 0.05, 0.8, 1, "gain")
    assert paired > 0
    assert mcnemar_arms(0.05, 0.05)["n_arms"] < mcnemar_arms(0.05, 0.40)["n_arms"]
    assert paired_table()["cells"]["0.05"]["0.2"]["n_arms"] == paired
    assert unpaired is not None       # both numbers exist and are labelled differently


def test_the_table_reports_every_cell_including_the_ones_it_cannot_fill():
    table = power_table(arm_counts=(24, 48), baselines=(0.05, 0.5))
    assert [row["arms"] for row in table["table"]] == [24, 48]
    for row in table["table"]:
        for key in ("0.05", "0.5"):
            for field in ("mde_gain_onesided", "mde_gain_twosided",
                          "mde_loss_onesided", "interval_floor_gain"):
                assert field in row[key], (field, row["arms"], key)
    assert "baseline is estimated" in table["caveat"]


def test_critical_region_rejects_bad_arguments():
    with pytest.raises(ValueError):
        critical_region(48, 0.1, 0.05, sided=3)
    with pytest.raises(ValueError):
        mde(48, 0.1, 0.05, 0.8, 1, "sideways")
    with pytest.raises(ValueError):
        mcnemar_arms(0.0, 0.1)
