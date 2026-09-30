"""Frozen calibrated assurance fails closed and keeps old path independent."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
from cviaf.lab.assure_protocol import load_protocol, assess_data, review_assessment, _hash
from cviaf.lab.evaluate import load_registry, train_spec_from_manifest
from cviaf.lab.train import build_splits

CORPUS = 'runs/mvp'
PROTOCOL = Path(__file__).resolve().parents[1] / 'configs/calibration/assure-synthetic-mvp-240.json'


def test_protocol_load_clean_attacks_and_overlap(tmp_path):
    registry = load_registry(CORPUS)
    clean = next(e for e in registry if e['manifest']['model_id'] == 'clean_none_fixed_s5')
    protocol, gate, _ = load_protocol(str(PROTOCOL), CORPUS, .05, clean['manifest'])
    assert protocol['heldout']['n'] >= 59
    clean_ds = build_splits(train_spec_from_manifest(clean['manifest'])).train_poisoned
    result = assess_data(clean_ds, gate, protocol)
    assert not result['calibrated_protocol']['asset_flag']
    assert result['calibrated_protocol']['review_queue']['summary']['n_items'] == len(clean_ds)
    flip = next(e for e in registry if e['manifest']['model_id'] == 'label_flip_none_fixed_s5')
    attacked = build_splits(train_spec_from_manifest(flip['manifest'])).train_poisoned
    assert assess_data(attacked, gate, protocol)['calibrated_protocol']['label']['flagged']
    changed = deepcopy(protocol)
    changed['calibration_seeds'][0] = changed['target_scene_seeds'][0]
    changed.pop('sha256')
    changed['sha256'] = _hash(changed)
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='overlap'):
        load_protocol(str(path), CORPUS, .05, clean['manifest'])
    changed = deepcopy(protocol)
    changed['effect_floor'] = 0.0
    changed.pop('sha256')
    changed['sha256'] = _hash(changed)
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='effect floor'):
        load_protocol(str(path), CORPUS, .05, clean['manifest'])


def test_missing_calibration_is_review_not_clear():
    result = review_assessment('missing drift calibration')
    assert result['findings'][0]['disposition'] == 'review'
    assert result['calibrated_protocol']['status'] == 'unavailable'


def test_frozen_protocol_rejects_tamper_target_and_detector_mismatch(tmp_path):
    manifest = load_registry(CORPUS)[0]['manifest']
    source = json.loads(PROTOCOL.read_text())
    for field, value in [('effect_floor', 0.0), ('target_model_ids', ['wrong']),
                         ('asset_size', 241)]:
        changed = deepcopy(source)
        changed[field] = value
        changed.pop('sha256')
        changed['sha256'] = _hash(changed)
        path = tmp_path / f'{field}.json'
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            load_protocol(str(path), CORPUS, .05, manifest)
    changed = deepcopy(source)
    changed['detector_versions']['patch_local'] = 'wrong'
    changed.pop('sha256')
    changed['sha256'] = _hash(changed)
    path = tmp_path / 'detector.json'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='detector mismatch'):
        load_protocol(str(path), CORPUS, .05, manifest)
    with pytest.raises(FileNotFoundError):
        load_protocol(str(tmp_path/'missing.json'), CORPUS, .05, manifest)
    for model in ('dup_flood_none_fixed_s5', 'ood_insert_none_fixed_s5'):
        entry = next(e for e in load_registry(CORPUS) if e['manifest']['model_id'] == model)
        dataset = build_splits(train_spec_from_manifest(entry['manifest'])).train_poisoned
        protocol, gate, _ = load_protocol(str(PROTOCOL), CORPUS, .05, entry['manifest'])
        with pytest.raises(ValueError, match='asset size mismatch'):
            assess_data(dataset, gate, protocol)
