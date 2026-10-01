import pytest

from cviaf.lab.clean_battery import BatteryPlan, conditional_pvalues, coverage, measure, synthesize
import numpy as np
from cviaf.lab.synth import SceneSpec


def test_deterministic_disjoint_and_coverage():
    plan = BatteryPlan(terrains=("desert", "snow"), seasons=("summer",),
                       sensors=("clear", "noisy"), n_cal=6, n_test=5, seed=12)
    battery = synthesize(plan)
    assert len(battery) == 4
    assert coverage(plan, battery, minimum_cal=6)["covered"] == 4
    assert coverage(plan, {}, minimum_cal=6)["covered"] == 0
    assert coverage(plan, {next(iter(battery)): {"cal": battery[next(iter(battery))]["cal"]}},
                    minimum_cal=6)["covered"] == 0
    repeated = synthesize(plan)
    for key, pair in battery.items():
        assert pair["cal"].digest() == repeated[key]["cal"].digest()
        assert pair["cal"].digest() != pair["test"].digest()
        assert set(pair["cal"].contributors).isdisjoint(pair["test"].contributors)
        assert pair["cal"].spec["terrain"] == key[0]
        assert pair["cal"].spec["season"] == key[1]


def test_rejects_missing_or_duplicate_axes():
    for kwargs in ({"terrains": ()}, {"seasons": ("winter", "winter")},
                   {"sensors": ("unknown",)}, {"n_cal": 0}, {"alpha": 1.0}):
        with pytest.raises(ValueError):
            synthesize(BatteryPlan(**kwargs))


def test_measures_same_clean_tests_for_both_arms():
    plan = BatteryPlan(terrains=("desert", "forest"), seasons=("summer",),
                       sensors=("clear",), n_cal=20, n_test=20, seed=8)
    report = measure(plan, SceneSpec())
    assert report["coverage"]["covered"] == 2
    assert report["pooled"]["hand_built"]["n"] == 40
    assert report["pooled"]["synthesized"]["n"] == 40
    assert report == measure(plan, SceneSpec())
    assert report["arms"]["hand_built"].keys() == report["arms"]["synthesized"].keys()


def test_conditional_calibration_refuses_missing_and_underfilled():
    key = ("desert", "summer", "clear")
    with pytest.raises(ValueError, match="no clean calibration"):
        conditional_pvalues(key, np.array([0.0]), {})
    with pytest.raises(ValueError, match="underfilled"):
        conditional_pvalues(key, np.array([0.0]), {key: np.array([1.0])}, minimum_cal=2)
    with pytest.raises(ValueError, match="nonfinite"):
        conditional_pvalues(key, np.array([np.nan]), {key: np.array([1.0])})
    assert conditional_pvalues(key, np.array([3.0]), {key: np.array([1.0, 2.0])})[0] == pytest.approx(1/3)
