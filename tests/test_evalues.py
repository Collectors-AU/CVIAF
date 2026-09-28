import hashlib
import json
import math

import numpy as np
import pytest

from cviaf.lab.evalues import TailBet, WealthLedger
from cviaf.lab.monitor_experiment import simulate


def test_calibration_and_evalues_are_frozen_and_bounded():
    cal = np.arange(400.)
    bet = TailBet.calibrate(cal, delta=.001)
    assert 0 < bet.tail_bound < 1
    assert bet.value(-1) >= 0
    assert bet.value(401) > 1
    assert bet.calibration_digest == TailBet.calibrate(cal, delta=.001).calibration_digest
    with pytest.raises(ValueError):
        TailBet.calibrate([float('nan'), 1], delta=.001)
    with pytest.raises(ValueError):
        bet.value(float('nan'))


def test_ledger_hash_chain_and_first_crossing():
    bet = TailBet(.5, .1, .5, 'frozen', 400)
    led = WealthLedger('asset-a', {'a': bet, 'b': bet}, alpha=.05, delta=.005)
    for t in range(1, 30):
        record = led.step({'a': 1., 'b': 1.})
        assert record['t'] == t
        assert record['threshold'] == pytest.approx(1 / .045)
        body = {k: v for k, v in record.items() if k != 'record_hash'}
        assert hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest() == record['record_hash']
        assert record['previous_hash'] == ('0' * 64 if t == 1 else led.records[-2]['record_hash'])
        if led.first_alarm:
            break
    assert led.first_alarm == t
    with pytest.raises(ValueError):
        led.step({'a': 1.})
    with pytest.raises(ValueError):
        WealthLedger('asset-a', {'a': bet}, alpha=.05, delta=.05)


def test_fusion_under_perfect_detector_dependence_does_not_double_count():
    bet = TailBet(.5, .1, .5, 'c', 400)
    led = WealthLedger('a', {'a': bet, 'b': bet}, alpha=.05)
    assert led.step({'a': 1, 'b': 1})['fused_e'] == pytest.approx(bet.value(1))
    assert led.step({'a': 1, 'b': 0})['fused_e'] == pytest.approx((bet.value(1) + bet.value(0)) / 2)


def test_simulated_clean_control_retests_and_no_attack_delay():
    rng = np.random.default_rng(17)
    cal = {k: rng.uniform(size=400) for k in ('ctc', 'refdiv', 'ftc')}
    x = rng.uniform(size=1000)
    null = {k: x for k in cal}  # maximal cross-detector dependence
    out = simulate(cal, null, null, streams=200, horizon=60)
    assert out['clean_false_alarm_curve']['e'][-1] <= .05
    assert out['clean_false_alarm_curve']['repeated_fixed_bonf'][-1] > .05
    assert out['attack_detection']['e_alarm_fraction'] <= .05


def test_nonfinite_attack_abstains_without_claiming_clean():
    x = np.linspace(0, 1, 30)
    c = {k: x.copy() for k in ('ctc', 'refdiv', 'ftc')}
    a = {k: x.copy() for k in c}
    a['ctc'][0] = np.nan
    out = simulate(c, c, a, streams=3, horizon=3)
    assert out['unscorable'] is True
    assert out['nonfinite_counts']['ctc'] == 1
