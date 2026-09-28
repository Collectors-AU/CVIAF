import numpy as np
import pytest
from cviaf.lab.model_asset_rule import (DEFAULT_SIGNALS, DIAGNOSTIC_SIGNALS,
                                        decide_model_asset, stamp_response)


def test_paired_stamp_response_and_bad_input():
    assert stamp_response({'ctc': [1, 2], 'refdiv': [0, 0]},
                          {'ctc': [2, 4], 'refdiv': [1, 1]}) == {'ctc': 1.5, 'refdiv': 1.}
    with pytest.raises(ValueError):
        stamp_response({'ctc': [1], 'refdiv': [1]},
                       {'ctc': [1, 2], 'refdiv': [1]})


def test_gate_keys_on_refdiv_and_ctc_is_diagnostic_only():
    """The merged default gates on refdiv alone: ctc's stamp response is not
    exchangeable across seed ranges, so it is reported but never multiplied in."""
    assert DEFAULT_SIGNALS == ("refdiv",)
    assert DIAGNOSTIC_SIGNALS == ("ctc",)
    nulls = [{'ctc': float(i), 'refdiv': float(i)} for i in range(24)]
    x = {'ctc': 100., 'refdiv': 100.}
    # m = 1 => required is 5 + ceil(1/alpha) - 1 = 24, and p_floor = 1/20 = .05
    assert decide_model_asset(x, nulls[:23])['abstain']
    d = decide_model_asset(x, nulls)
    assert d['signals'] == ['refdiv'] and d['required_clean_models'] == 24
    assert d['asset_pvalue'] == .05 and d['reject_at_alpha']
    assert d['diagnostic_per_signal']['ctc']['diagnostic_only'] is True
    assert d['diagnostic_per_signal']['ctc']['p'] == .05
    # a ctc-only anomaly cannot reject: the gate does not see that column
    assert not decide_model_asset({'ctc': 100., 'refdiv': 0.}, nulls)['reject_at_alpha']


def test_two_signal_gate_is_still_available_explicitly():
    nulls = [{'ctc': float(i), 'refdiv': float(i)} for i in range(44)]
    x = {'ctc': 100., 'refdiv': 100.}
    two = ('ctc', 'refdiv')
    assert decide_model_asset(x, nulls[:43], signals=two)['abstain']
    d = decide_model_asset(x, nulls, signals=two)
    assert d['asset_pvalue'] == .05 and d['reject_at_alpha']
    assert d['n_clean_models'] == 44
    assert not decide_model_asset({'ctc': 22., 'refdiv': 22.}, nulls,
                                  signals=two)['reject_at_alpha']
    assert not decide_model_asset(x, nulls, signals=two, alpha=.01)['reject_at_alpha']


def test_nonfinite_abstains_instead_of_clear():
    nulls = [{'ctc': float(i), 'refdiv': 0.} for i in range(24)]
    assert decide_model_asset({'ctc': 1., 'refdiv': np.nan}, nulls)['abstain']
    # a missing diagnostic column cannot change the verdict either way
    d = decide_model_asset({'refdiv': 100.}, nulls)
    assert d['diagnostic_per_signal']['ctc']['available'] is False or d['abstain']
