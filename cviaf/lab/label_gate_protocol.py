"""Opt-in, reproducible synthetic label-consistency reference protocol.

This is a lab-only generator. It does not authenticate a real supplier's scene
metadata or prove that generated references resemble operational imagery.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence
import numpy as np

from cviaf.lab.label_consistency import LabelConsistencyGate
from cviaf.lab.synth import SceneSpec, build_dataset
from cviaf.lab.evaluate import load_registry

SCHEMA = "cviaf-label-gate-synthetic-reference/1"
DEFAULT_FIT = (10000, 10001, 10002, 10003)
DEFAULT_CAL = tuple(range(11000, 11080))


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(obj):
    return hashlib.sha256(_canonical(obj).encode()).hexdigest()


def _domain(scene):
    return {k: v for k, v in scene.to_dict().items() if k != 'seed'}


def _targets(corpus):
    entries = load_registry(str(corpus))
    if not entries:
        raise ValueError('no target corpus models')
    scenes = [e['manifest']['spec']['scene'] for e in entries]
    domains = {_canonical({k: v for k, v in s.items() if k != 'seed'}) for s in scenes}
    if len(domains) != 1:
        raise ValueError('target corpus has multiple declared scene domains; fit one gate per domain')
    target_seeds = {int(s['seed']) for s in scenes}
    model_ids = [e['manifest']['model_id'] for e in entries]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError('duplicate model IDs in target registry')
    return scenes[0], target_seeds, sorted(model_ids)


def fit_synthetic(corpus: str, out: str, fit_seeds: Sequence[int] = DEFAULT_FIT,
                  calibration_seeds: Sequence[int] = DEFAULT_CAL, alpha: float = .05):
    scene, targets, ids = _targets(corpus)
    fit_seeds = tuple(map(int, fit_seeds))
    calibration_seeds = tuple(map(int, calibration_seeds))
    all_seeds = fit_seeds + calibration_seeds
    if not fit_seeds or len(calibration_seeds) < 39 or len(all_seeds) != len(set(all_seeds)):
        raise ValueError('need nonoverlapping fit seeds and >=39 distinct calibration assets')
    if set(all_seeds) & targets:
        raise ValueError('reference/calibration seeds overlap target corpus scene seeds')
    if not 0 < alpha < 1:
        raise ValueError('alpha must be in (0,1)')
    n = int(load_registry(str(corpus))[0]['manifest']['spec']['n_train'])
    if any(int(e['manifest']['spec']['n_train']) != n for e in load_registry(str(corpus))):
        raise ValueError('target corpus uses mixed asset sizes')
    domain = {k: v for k, v in scene.items() if k != 'seed'}
    def generate(seed):
        return build_dataset(n, SceneSpec(**{**domain, 'seed': seed}),
                             contributors=('trusted_synthetic_clean',))
    fit_assets = [generate(seed) for seed in fit_seeds]
    cal_assets = [generate(seed) for seed in calibration_seeds]
    gate = LabelConsistencyGate.fit(fit_assets, cal_assets, alpha=alpha)
    payload = {
        'schema': SCHEMA, 'reference_type': 'synthetic_generator_not_real_world',
        'domain': domain, 'asset_size': n, 'alpha': alpha,
        'fit_seeds': list(fit_seeds), 'calibration_seeds': list(calibration_seeds),
        'target_scene_seeds': sorted(targets), 'target_model_ids': ids,
        'fit_asset_digests': [d.digest() for d in fit_assets],
        'calibration_asset_digests': [d.digest() for d in cal_assets],
        'centroids': gate.centroids.tolist(), 'null_counts': gate.null_counts.tolist(),
        'reference_spec': gate.reference_spec,
        'n_reference': gate.n_reference,
    }
    payload['sha256'] = _hash(payload)
    destination = Path(out)
    if destination.exists():
        raise FileExistsError(f'{out} already exists; reference protocols are immutable')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return payload


def load_synthetic(path: str, corpus: str, alpha: float):
    payload = json.loads(Path(path).read_text())
    claimed = payload.pop('sha256', None)
    if payload.get('schema') != SCHEMA or claimed != _hash(payload):
        raise ValueError('label-gate protocol schema/digest mismatch')
    scene, target_seeds, ids = _targets(corpus)
    domain = {k: v for k, v in scene.items() if k != 'seed'}
    registry = load_registry(str(corpus))
    sizes = {int(e['manifest']['spec']['n_train']) for e in registry}
    if len(sizes) != 1 or payload['asset_size'] != sizes.pop():
        raise ValueError('label-gate protocol asset size mismatch')
    if (payload['domain'] != domain or payload['target_model_ids'] != ids
            or payload['target_scene_seeds'] != sorted(target_seeds)
            or float(payload['alpha']) != alpha):
        raise ValueError('label-gate protocol targets, declared domain or alpha differ')
    refs = payload['fit_seeds'] + payload['calibration_seeds']
    if (len(refs) != len(set(refs)) or set(refs) & target_seeds
            or len(payload['calibration_seeds']) < 39
            or len(payload['null_counts']) != len(payload['calibration_seeds'])
            or payload['reference_type'] != 'synthetic_generator_not_real_world'
            or len(payload['fit_asset_digests']) != len(payload['fit_seeds'])
            or len(payload['calibration_asset_digests']) != len(payload['calibration_seeds'])):
        raise ValueError('invalid or overlapping clean-reference protocol')
    c = np.asarray(payload['centroids'], dtype=float)
    null = np.asarray(payload['null_counts'], dtype=float)
    if c.ndim != 2 or c.shape[1] != 3 or len(c) < 2 or not np.isfinite(c).all() or not np.isfinite(null).all():
        raise ValueError('invalid calibration scores')
    gate = LabelConsistencyGate(c, null, int(payload['n_reference']),
                                payload['reference_spec'], alpha)
    return gate, claimed
