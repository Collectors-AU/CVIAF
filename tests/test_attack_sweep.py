"""Tests for the attack-strength sweep planner (`scripts/attack_sweep.py`).

Each test is a way the ladder would be quietly wrong:

  * a dose typo (2.5 for 0.25) that silently duplicates another arm, so a rung of the
    ladder is missing while the arm count still looks right;
  * a cell with no builder or no parameter name, which is an arm nobody can derive;
  * a plan whose arithmetic does not match its own config, which is how a compute bill
    and the arms actually built diverge;
  * a sweep sized by habit instead of by the power table, which is the whole reason
    the table exists;
  * a declared per-arm cost presented as if it had been measured.
"""
from __future__ import annotations

import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cviaf.lab.power import BATTERY_TPR, arms_needed  # noqa: E402
from scripts.attack_sweep import (  # noqa: E402
    arms_for_half_width,
    load_config,
    plan_sweep,
    render,
    validate_config,
)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "configs", "attack_sweep_pilot.json")


def config():
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_shipped_pilot_config_is_valid():
    assert validate_config(config()) == []


def test_a_dose_typo_cannot_silently_duplicate_an_arm():
    bad = config()
    bad["kinds"]["weight_tamper"]["doses"] = [0.25, 2.5]
    problems = validate_config(bad)
    assert any("outside (0, 1]" in p for p in problems), problems


def test_a_repeated_dose_is_refused_because_it_builds_the_same_arm_twice():
    bad = config()
    bad["kinds"]["substitution"]["doses"] = [0.25, 0.25]
    assert any("duplicate doses" in p for p in validate_config(bad))


def test_a_cell_without_a_builder_or_a_parameter_is_refused():
    bad = config()
    del bad["kinds"]["substitution"]["builder"]
    del bad["kinds"]["weight_tamper"]["param"]
    problems = validate_config(bad)
    assert any("builder" in p for p in problems)
    assert any("param" in p for p in problems)


def test_the_target_must_declare_both_questions_it_sizes_for():
    bad = config()
    del bad["target"]["half_width"]
    assert any("half_width" in p for p in validate_config(bad))
    bad = config()
    bad["target"]["delta"] = 0
    assert any("delta" in p for p in validate_config(bad))


def test_the_plan_arithmetic_matches_its_config():
    cfg = config()
    plan = plan_sweep(cfg)
    expected_cells = sum(len(spec["doses"]) for spec in cfg["kinds"].values())
    assert plan["n_cells"] == expected_cells
    assert plan["pilot_arms"] == expected_cells * len(cfg["seeds"]["pilot"])
    assert sum(c["pilot_arms"] for c in plan["cells"]) == plan["pilot_arms"]


def test_the_ladder_is_reported_in_dose_order_with_every_rung_named():
    plan = plan_sweep(config())
    for kind in config()["kinds"]:
        doses = [c["dose"] for c in plan["cells"] if c["kind"] == kind]
        assert doses == sorted(doses)
        assert doses == sorted(float(d) for d in config()["kinds"][kind]["doses"])
    assert all(c["builder"] and c["param"] for c in plan["cells"])
    assert "weight_tamper" in render(plan)


def test_the_sweep_is_sized_by_the_power_table_not_by_habit():
    cfg = config()
    plan = plan_sweep(cfg)
    den = plan["denominator"]
    expected = arms_needed(BATTERY_TPR, den["delta"], den["alpha"], den["power"],
                           1, "gain")
    assert den["per_cell_to_detect"] == expected
    assert den["sweep_arms_to_detect"] == expected * plan["n_cells"]
    assert den["per_cell_to_measure"] == arms_for_half_width(BATTERY_TPR,
                                                             den["half_width"])
    assert den["baseline_recall"] == pytest.approx(7 / 48, abs=1e-4)
    assert "only an estimate" in den["baseline_basis"]


def test_measuring_every_cell_costs_far_fewer_arms_than_detecting_every_cell():
    """"Size the sweep for a 5-point difference" overspends by an order of magnitude."""
    den = plan_sweep(config())["denominator"]
    assert den["sweep_arms_to_measure"] < den["sweep_arms_to_detect"] / 3
    assert "overspends" in den["reading"]


def test_the_plan_is_deterministic_and_digests_its_inputs():
    a = plan_sweep(config())
    b = plan_sweep(config())
    assert a["plan_digest"] == b["plan_digest"]
    changed = config()
    changed["kinds"]["substitution"]["doses"] = [0.25, 0.5]
    assert plan_sweep(changed)["plan_digest"] != a["plan_digest"]


def test_a_pilot_plan_does_not_claim_a_full_sweep_arm_count():
    assert plan_sweep(config(), mode="plan").get("full_arms") is None
    pilot = plan_sweep(config(), mode="pilot")
    assert pilot["pilot_arms"] < 100 and pilot.get("full_arms") is None
    full = plan_sweep(config(), mode="full")
    assert full["full_arms"] == full["denominator"]["sweep_arms_to_measure"]
    assert full["mode"] == "full"


def test_the_declared_per_arm_cost_is_named_as_unmeasured():
    plan = plan_sweep(config(), seconds_per_arm=25.0)
    assert "must be recomputed" in plan["cost_basis"]
    assert plan["pilot_minutes"] == pytest.approx(27 * 25.0 / 60.0, abs=0.2)
    assert plan["seconds_per_arm"] == 25.0


def test_the_plan_states_that_no_detector_is_trained():
    assert "no clean detector is trained" in plan_sweep(config())["no_training"]


def test_half_width_sizing_refuses_impossible_or_degenerate_targets():
    with pytest.raises(ValueError):
        arms_for_half_width(0.5, 0.6)
    with pytest.raises(ValueError):
        arms_for_half_width(0.0, 0.1)
    assert arms_for_half_width(0.5, 0.49) == 1
    # 93 arms put the Wilson half-width at 0.5 +/- 0.10 (the textbook normal answer is
    # 96; the exact interval is slightly tighter, and the exact one is the one used).
    assert arms_for_half_width(0.5, 0.1) == 93
    assert arms_for_half_width(0.5, 0.05) > arms_for_half_width(0.5, 0.1)


def test_a_loadable_config_round_trips_and_refuses_an_invalid_one(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps(config()), encoding="utf-8")
    loaded = load_config(str(path))
    assert loaded["schema"] == "cviaf.attack-sweep.v1"
    bad = copy.deepcopy(config())
    bad["seeds"]["pilot"] = []
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_config(str(path))
    assert "seeds.pilot" in str(exc.value)
