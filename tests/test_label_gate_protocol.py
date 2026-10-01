"""Synthetic label gate must be explicit, frozen and target-disjoint."""
import json
from pathlib import Path
import pytest
from cviaf.lab.label_gate_protocol import fit_synthetic, load_synthetic
from cviaf.lab.synth import SceneSpec, build_dataset
from cviaf.lab.poison import AttackSpec, inject


def test_protocol_fit_load_and_detect(tmp_path):
    corpus = 'runs/mvp'
    path = tmp_path / 'ref.json'
    fit_seeds = range(100, 104)
    cal_seeds = range(200, 239)
    record = fit_synthetic(corpus, str(path), fit_seeds, cal_seeds)
    gate, digest = load_synthetic(str(path), corpus, .05)
    assert digest == record['sha256'] and len(gate.null_counts) == 39
    scene = SceneSpec(**{**record['domain'], 'seed': 301})
    clean = build_dataset(240, scene)
    poison, _ = inject(clean, AttackSpec(kind='label_flip', rate=.1, seed=11))
    assert not gate.assess(clean)['flagged']
    assert gate.assess(poison)['flagged']
    with pytest.raises(FileExistsError):
        fit_synthetic(corpus, str(path), fit_seeds, cal_seeds)
    tampered = json.loads(path.read_text())
    tampered['null_counts'] = [0]
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match='digest'):
        load_synthetic(str(path), corpus, .05)


def test_protocol_rejects_target_seed_overlap(tmp_path):
    with pytest.raises(ValueError, match='overlap'):
        fit_synthetic('runs/mvp', str(tmp_path/'x.json'), [12], range(200, 239))


def test_protocol_rejects_bound_asset_size_change(tmp_path):
    path = tmp_path / 'ref.json'
    record = fit_synthetic('runs/mvp', str(path), range(100, 104), range(200, 239))
    record['asset_size'] += 1
    record.pop('sha256')
    from cviaf.lab.label_gate_protocol import _hash
    record['sha256'] = _hash(record)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='asset size'):
        load_synthetic(str(path), 'runs/mvp', .05)
