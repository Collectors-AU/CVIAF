"""Frozen, synthetic-only calibration for opt-in end-to-end assurance.

A checksum detects accidental alteration, not a trusted supplier identity. The
calibration unit is a whole independently seeded asset, not an image or box.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from cviaf.lab.label_gate_protocol import _targets, _hash
from cviaf.lab.synth import SceneSpec, build_dataset
from cviaf.lab.evaluate import load_registry
from cviaf.lab.label_consistency import LabelConsistencyGate
from cviaf.lab.patch_local import patch_local_scores
from cviaf.drift import standardized_wasserstein_effect
from cviaf.utils import extract_features_from_images

SCHEMA = 'cviaf-assure-synthetic-protocol/1'
FIT_SEEDS = tuple(range(30000, 30004))
CAL_SEEDS = tuple(range(31000, 31080))
HELDOUT_SEEDS = tuple(range(32000, 32059))
DRIFT_REF_SEED = 33000


def _versions():
    from cviaf.lab import label_consistency, patch_local, detectors
    from cviaf import drift
    return {name: hashlib.sha256(inspect.getsource(module).encode()).hexdigest()
            for name, module in [('label_consistency', label_consistency),
                                 ('patch_local', patch_local),
                                 ('detectors', detectors), ('drift', drift)]}


def _asset_features(ds):
    return extract_features_from_images(ds.images, method='pixel_stats', target_dim=64)


def _as_dataset(domain, size, seed):
    return build_dataset(size, SceneSpec(**{**domain, 'seed': seed}),
                         contributors=('trusted_synthetic_clean',))


def _patch_count(ds, item_floor):
    return int(np.count_nonzero(patch_local_scores(ds.images) > item_floor))


def fit_protocol(corpus: str, out: str, alpha: float = .05,
                 drift_calibration: str | None = None,
                 fit_seeds=FIT_SEEDS, cal_seeds=CAL_SEEDS,
                 heldout_seeds=HELDOUT_SEEDS, drift_ref_seed=DRIFT_REF_SEED):
    if Path(out).exists():
        raise FileExistsError('frozen protocol already exists')
    if not 0 < alpha < 1:
        raise ValueError('invalid alpha')
    scene, targets, ids = _targets(corpus)
    entries = load_registry(corpus)
    sizes = {int(e['manifest']['spec']['n_train']) for e in entries}
    if len(sizes) != 1:
        raise ValueError('mixed asset sizes')
    size = sizes.pop()
    fit_seeds, cal_seeds, heldout_seeds = map(lambda v: tuple(map(int, v)),
                                              (fit_seeds, cal_seeds, heldout_seeds))
    all_seeds = fit_seeds + cal_seeds + heldout_seeds + (int(drift_ref_seed),)
    if not fit_seeds or len(cal_seeds) < 39 or len(heldout_seeds) < 59 or len(all_seeds) != len(set(all_seeds)) or set(all_seeds) & targets:
        raise ValueError('need disjoint fit, >=39 calibration, >=59 held-out and target seeds')
    domain = {k: v for k, v in scene.items() if k != 'seed'}
    fits = [_as_dataset(domain, size, seed) for seed in fit_seeds]
    ref = _as_dataset(domain, min(size, 120), int(drift_ref_seed))
    ref_features = _asset_features(ref)
    from cviaf.lab.label_consistency import box_features
    features, labels = [], []
    for ds in fits:
        x, y, _ = box_features(ds)
        features.append(x); labels.append(y)
    features, labels = np.concatenate(features), np.concatenate(labels)
    classes = np.unique(labels)
    if len(classes) < 2 or not np.array_equal(classes, np.arange(classes[-1] + 1)):
        raise ValueError('incomplete synthetic class reference')
    centroids = np.stack([np.median(features[labels == i], axis=0) for i in classes])
    ref_spec = {k: v for k, v in fits[0].spec.items()
                if k not in ('seed', 'seed_offset', 'n', 'contributors', 'contributor_mode')}
    gate = LabelConsistencyGate(centroids, np.empty(0, int), len(fits), ref_spec, alpha)
    patch_reference = np.concatenate([patch_local_scores(x.images) for x in fits])
    item_floor = float(np.quantile(patch_reference, 1 - alpha / size))
    patch_null, effect_null, cal_digests = [], [], []
    for seed in cal_seeds:
        ds = _as_dataset(domain, size, seed)
        cal_digests.append(ds.digest())
        gate.null_counts = np.append(gate.null_counts, gate.mismatch_count(ds))
        patch_null.append(_patch_count(ds, item_floor))
        effect_null.append(standardized_wasserstein_effect(ref_features, _asset_features(ds))['value'])
        del ds
    # No arbitrary 0.2 floor. Finite-sample upper order statistic at this alpha.
    rank = int(np.ceil((len(effect_null) + 1) * (1 - alpha)))
    if rank > len(effect_null):
        raise ValueError('insufficient independent clean batches for effect floor')
    effect_floor = float(np.sort(effect_null)[rank-1])
    heldout_patch, heldout_effect, heldout_label, heldout_digests = [], [], [], []
    for seed in heldout_seeds:
        ds = _as_dataset(domain, size, seed)
        heldout_digests.append(ds.digest())
        heldout_patch.append(_patch_count(ds, item_floor))
        heldout_effect.append(standardized_wasserstein_effect(ref_features, _asset_features(ds))['value'])
        heldout_label.append(gate.assess(ds, alpha=alpha / 2)['flagged'])
        del ds
    heldout_patch_flags = [(1 + sum(n >= count for n in patch_null)) / (len(patch_null) + 1) <= alpha / 2 for count in heldout_patch]
    heldout_reject = [bool(a or b) for a, b in zip(heldout_label, heldout_patch_flags)]
    payload = {
        'schema': SCHEMA, 'reference_type': 'synthetic_generator_not_real_world',
        'alpha': alpha, 'domain': domain, 'asset_size': size,
        'target_model_ids': ids, 'target_scene_seeds': sorted(targets),
        'detector_versions': _versions(),
        'natural_drift_calibration': ({
            'sha256': hashlib.sha256(Path(drift_calibration).read_bytes()).hexdigest(),
            'n_ref': len(ref), 'n_op': size,
        } if drift_calibration else None),
        'fit_seeds': fit_seeds, 'calibration_seeds': cal_seeds,
        'heldout_seeds': heldout_seeds, 'drift_reference_seed': int(drift_ref_seed),
        'fit_digests': [x.digest() for x in fits],
        'calibration_digests': cal_digests,
        'heldout_digests': heldout_digests,
        'drift_reference_digest': ref.digest(), 'drift_reference_size': len(ref),
        'label_centroids': gate.centroids.tolist(),
        'label_null_counts': gate.null_counts.tolist(),
        'label_reference_spec': gate.reference_spec,
        'patch_item_floor': item_floor, 'patch_reference_scores': patch_reference.tolist(),
        'patch_null_counts': patch_null, 'effect_null': effect_null,
        'effect_floor': effect_floor,
        'heldout': {'n': len(heldout_seeds), 'label_flags': sum(heldout_label),
                    'patch_flags': sum(heldout_patch_flags),
                    'asset_flags': sum(heldout_reject),
                    'drift_effect_above_floor': sum(x >= effect_floor for x in heldout_effect),
                    'review_rate': sum(heldout_reject) / len(heldout_seeds),
                    'abstention_rate': 0.0,
                    'fpr_upper_95_if_zero': 1 - .05 ** (1 / len(heldout_seeds)) if not any(heldout_reject) else None,
                    'independence_caveat': 'Distinct generator seeds are not independent real suppliers.'},
    }
    if not drift_calibration:
        raise ValueError('independently generated, batch-matched natural-drift calibration required')
    if drift_calibration:
        from cviaf.drift.attribution import NaturalDriftCalibration
        cal = NaturalDriftCalibration.load(drift_calibration)
        if cal.n_ref != len(ref) or cal.n_op != size or cal.alpha != alpha:
            raise ValueError('natural-drift calibration batch sizes or alpha mismatch')
    payload['sha256'] = _hash(payload)
    path = Path(out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return payload


def load_protocol(path: str, corpus: str, alpha: float, manifest: dict):
    record = json.loads(Path(path).read_text())
    claimed = record.pop('sha256', None)
    if record.get('schema') != SCHEMA or claimed != _hash(record):
        raise ValueError('assure calibration schema/digest mismatch')
    scene, targets, ids = _targets(corpus)
    domain = {k: v for k, v in scene.items() if k != 'seed'}
    size = int(manifest['spec']['n_train'])
    if (record.get('natural_drift_calibration') is None
            or record['domain'] != domain or record['asset_size'] != size
            or record['target_model_ids'] != ids or record['target_scene_seeds'] != sorted(targets)
            or record['alpha'] != alpha or record['detector_versions'] != _versions()
            or record['reference_type'] != 'synthetic_generator_not_real_world'):
        raise ValueError('assure calibration target, domain, size, alpha or detector mismatch')
    seeds = (record['fit_seeds'] + record['calibration_seeds'] + record['heldout_seeds']
             + [record['drift_reference_seed']])
    if (len(seeds) != len(set(seeds)) or set(seeds) & targets
            or any(int(e['manifest']['spec']['n_train']) != size for e in load_registry(corpus))
            or len(record['calibration_seeds']) < 39 or len(record['heldout_seeds']) < 59
            or len(record['label_null_counts']) != len(record['calibration_seeds'])
            or len(record['patch_null_counts']) != len(record['calibration_seeds'])
            or len(record['effect_null']) != len(record['calibration_seeds'])
            or len(record['fit_digests']) != len(record['fit_seeds'])
            or len(record['calibration_digests']) != len(record['calibration_seeds'])
            or len(record['heldout_digests']) != len(record['heldout_seeds'])
            or record['drift_reference_size'] != min(size, 120)
            or record['natural_drift_calibration']['n_ref'] != min(size, 120)
            or record['natural_drift_calibration']['n_op'] != size):
        raise ValueError('assure calibration overlap or incomplete battery')
    floor = float(record['effect_floor'])
    null = np.asarray(record['effect_null'], float)
    rank = int(np.ceil((len(null) + 1) * (1 - alpha)))
    if not np.isfinite(null).all() or not np.isfinite(floor) or rank > len(null) or floor != float(np.sort(null)[rank-1]):
        raise ValueError('assure effect floor invalid')
    ref = _as_dataset(domain, record['drift_reference_size'], record['drift_reference_seed'])
    if ref.digest() != record['drift_reference_digest']:
        raise ValueError('assure reference digest mismatch')
    centroids = np.asarray(record['label_centroids'], float)
    labels = np.asarray(record['label_null_counts'], float)
    patch = np.asarray(record['patch_reference_scores'], float)
    if (not np.isfinite(centroids).all() or not np.isfinite(labels).all()
            or not np.isfinite(patch).all() or centroids.ndim != 2 or centroids.shape[1] != 3
            or len(patch) != len(record['fit_seeds']) * size):
        raise ValueError('assure detector reference invalid')
    if record['label_reference_spec'] != {k: v for k, v in ref.spec.items() if k not in ('seed', 'seed_offset', 'n', 'contributors', 'contributor_mode')}:
        raise ValueError('annotation reference scene mismatch')
    gate = LabelConsistencyGate(centroids, labels, len(record['fit_seeds']),
                                record['label_reference_spec'], alpha)
    record['sha256'] = claimed
    return record, gate, ref


def assess_data(dataset, gate, protocol):
    """Whole-asset decisions and an Arm J item queue; never call no flag ACCEPT."""
    from cviaf.lab.calibrate import benjamini_yekutieli, conformal_pvalues
    from cviaf.lab.review import plan_review
    from cviaf.core.types import Finding
    alpha = protocol['alpha']
    if len(dataset) != protocol['asset_size']:
        raise ValueError(f"asset size mismatch: {len(dataset)} != {protocol['asset_size']}")
    label = gate.assess(dataset, alpha=alpha/2)
    if label['abstained']:
        raise ValueError('annotation reference domain mismatch')
    patch_scores = patch_local_scores(dataset.images)
    ref = np.asarray(protocol['patch_reference_scores'], float)
    p_item = conformal_pvalues(ref, patch_scores)
    mask = benjamini_yekutieli(p_item, alpha / 2)
    patch_count = _patch_count(dataset, protocol['patch_item_floor'])
    null = protocol['patch_null_counts']
    patch_p = (1 + sum(n >= patch_count for n in null)) / (len(null) + 1)
    patch_flag = patch_p <= alpha / 2
    # The ranking is advisory. Force item review when the signal cannot resolve
    # an item-level decision at a 240-item BY multiplicity floor.
    accept_permitted = bool(1 / (len(ref) + 1) <= alpha / (len(dataset) * sum(1/i for i in range(1, len(dataset)+1))))
    queue = plan_review(p_item, flagged=mask, alpha=alpha/2,
                        accept_permitted=accept_permitted,
                        abstain_reason='' if accept_permitted else 'BY item-level calibration floor')
    asset_flag = label['flagged'] or patch_flag
    findings = []
    if label['flagged']:
        findings.append(Finding(module='calibrated_data_integrity',
            attack_class='label_inconsistency', severity='HIGH', disposition='review',
            title='Annotation consistency flagged',
            description='Whole-asset label mismatches exceeded independent synthetic clean references; cause is not uniquely a label flip.',
            evidence={'protocol_sha256': protocol['sha256'], **label},
            affected_assets=['training_dataset']).to_dict())
    if patch_flag:
        findings.append(Finding(module='calibrated_data_integrity',
            attack_class='trigger_injection', severity='HIGH', disposition='review',
            title='Patch-local residual flagged',
            description='Whole-asset patch-local count exceeded synthetic clean references; this is not proof of a backdoor.',
            evidence={'protocol_sha256': protocol['sha256'], 'count': patch_count,
                      'asset_p': patch_p, 'calibration_assets': len(null)},
            affected_assets=['training_dataset']).to_dict())
    return {'findings': findings, 'overall_risk': 'HIGH' if asset_flag else 'LOW',
            'calibrated_protocol': {'digest': protocol['sha256'],
                'label': label, 'patch': {'count': patch_count, 'asset_p': patch_p,
                     'flagged': patch_flag, 'calibration_assets': len(null)},
                'asset_flag': asset_flag, 'item_flags': int(mask.sum()),
                'review_queue': {'summary': queue.summary(), 'tasks': queue.queue},
                'scope': 'synthetic declared-domain only; no real-world calibration claim'}}


def review_assessment(reason: str):
    from cviaf.core.types import Finding
    return {'findings': [Finding(module='calibrated_protocol', severity='HIGH',
                disposition='review', title='Calibration unavailable or mismatched',
                description=reason, affected_assets=['training_dataset']).to_dict()],
            'overall_risk': 'HIGH',
            'calibrated_protocol': {'status': 'unavailable', 'reason': reason}}
