import hashlib
import hmac
import json
import math

import numpy as np
import pytest

from cviaf.provenance.replay import seal_record
from cviaf.provenance.verify_replay import sample_size, verify
from cviaf.lab.evasion import Candidate, successive_halving


class Model:
    def __init__(self, altered=False):
        self.altered = altered
    def digest(self):
        return 'enrolled' if not self.altered else 'substituted'
    def predict(self, image, **cfg):
        v = float(image.flat[0]) + (0.2 if self.altered else 0)
        return {'boxes': np.array([[v, 1, v+3, 4]]),
                'scores': np.array([v]), 'labels': np.array([1])}


def corpus(n=1000, forged=50):
    images = {str(i): np.array([i / n], np.float32) for i in range(n)}
    records = [seal_record(im, Model(int(rid) < forged).predict(im), 'enrolled', {},
                           {'runtime': 'lab'}, rid, b'service')
               for rid, im in images.items()]
    return records, images


def test_replay_catches_sealed_substitution_and_binomial_bound():
    records, images = corpus()
    assert all(r['model_digest'] == 'enrolled' for r in records)
    assert all(r['output_sha256'] for r in records)
    found = sum(bool(verify(records, lambda _: Model(), images.__getitem__,
                            b'service', b'auditor', audit_seed=i.to_bytes(8, 'big'))
                     ['mismatched_record_ids']) for i in range(200))
    k = sample_size(.05, .05, 1000)
    assert k == 59
    assert found >= 190
    assert (1-.05)**k <= .05


def test_clean_and_tampered_capsules():
    records, images = corpus(40, 0)
    result = verify(records, lambda _: Model(), images.__getitem__, b'service', b'auditor',
                    q=.1, miss_probability=.05, audit_output_key=b'other')
    assert result['status'] == 'pass' and result['k'] == 29
    records[0]['output'][0][1] += 10
    with pytest.raises(ValueError, match='invalid record seal'):
        verify(records, lambda _: Model(), images.__getitem__, b'service', b'auditor')


def test_evasion_grid_recovers_lowest_budget_and_is_reproducible():
    clean = {'ctc': np.arange(100)/100, 'ftc': np.arange(100)/100}
    attack = {'ctc': np.ones(100)*1.1, 'ftc': np.ones(100)*1.2}
    grid = [Candidate('dilution', float(p), float(p)) for p in np.linspace(0, .9, 10)]
    a = successive_halving(clean, attack, grid, initial_count=40, attack_qualified=True)
    b = successive_halving(clean, attack, grid, initial_count=40, attack_qualified=True)
    for name in clean:
        assert a['detectors'][name]['curve'] == b['detectors'][name]['curve']
        assert .5 <= a['detectors'][name]['b_star_grid']['budget'] <= .61
    assert len(a['detectors']['ctc']['curve']) == 10


def test_no_evasion_claim_when_detector_is_already_blind():
    x = successive_halving({'ctc': np.arange(100)/100},
                            {'ctc': np.arange(100)/100},
                            [Candidate('dilution', .0, .0)], attack_qualified=True)['detectors']['ctc']
    assert x['status'] == 'already_below_target' and x['b_star_grid'] is None
