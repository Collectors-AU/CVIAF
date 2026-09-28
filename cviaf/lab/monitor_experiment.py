"""Reproducible U1 lab experiment on the repository's trained attack corpus.

Run: python -m cviaf.lab.monitor_experiment --corpus runs/mvp --out /tmp/u1.json
No model is trained here. Fixed-sample comparator tests each observation using the
same frozen calibration, with no correction for repeated looks. This fixed-sample comparator is intentionally uncorrected across repeated looks;
its faster delay does not reflect an equal false-alarm budget.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from cviaf.lab.calibrate import conformal_pvalues
from cviaf.lab.detectors import make_backgrounds, reference_divergence, trace_ctc, trace_ftc
from cviaf.lab.evaluate import load_registry, train_spec_from_manifest
from cviaf.lab.evalues import TailBet, WealthLedger
from cviaf.lab.poison import trigger_view
from cviaf.lab.train import ModelArtifact, build_splits

NAMES = ('ctc', 'refdiv', 'ftc')


def score_views(entry, ref_model, backgrounds):
    manifest = entry['manifest']
    kind = manifest['ground_truth']['kind']
    spec = train_spec_from_manifest(manifest)
    splits = build_splits(spec)
    model = ModelArtifact.load(entry['dir']).model
    cal = splits.cal_clean.images
    clean = splits.eval_clean.images
    if kind in ('clean', 'clean_label', 'label_flip', 'dup_flood', 'ood_insert'):
        attack = clean
    else:
        attack, _ = trigger_view(splits.eval_clean, spec.attack, 1)

    def score(images):
        return {'ctc': np.asarray(trace_ctc(model, images, backgrounds)['score']),
                'refdiv': np.asarray(reference_divergence(model, ref_model, images)['score']),
                'ftc': np.asarray(trace_ftc(model, images, decoy_class=0, stride=8)['score'])}
    return score(cal), score(clean), score(attack)


def simulate(cal, clean, attack, *, alpha=.05, delta=.005, streams=200,
             horizon=80, seed=20260928):
    if not (0 < delta < 1 and streams > 0 and horizon > 0):
        raise ValueError('invalid experiment parameters')
    # Simultaneous DKW event across predeclared detectors (union bound).
    bets = {name: TailBet.calibrate(cal[name], delta=delta / len(NAMES))
            for name in NAMES}
    clean = {k: np.asarray(v, float) for k, v in clean.items()}
    attack = {k: np.asarray(v, float) for k, v in attack.items()}
    n = len(clean[NAMES[0]])
    if n < 2 or any(len(clean[k]) != n or len(attack[k]) != len(attack[NAMES[0]])
                    for k in NAMES):
        raise ValueError('detector score arrays must align by image')
    if any(not np.all(np.isfinite(v)) for v in [*cal.values(), *clean.values()]):
        raise ValueError('nonfinite calibration or null scores invalidate the experiment')
    if any(not np.all(np.isfinite(v)) for v in attack.values()):
        return {'unscorable': True, 'reason': 'nonfinite attack detector score; no alarm delay claim',
                'nonfinite_counts': {k: int((~np.isfinite(v)).sum()) for k, v in attack.items()}}
    p_clean = np.column_stack([conformal_pvalues(cal[k], clean[k]) for k in NAMES])
    p_attack = np.column_stack([conformal_pvalues(cal[k], attack[k]) for k in NAMES])
    def fixed(p):
        return np.min(p, axis=1) * len(NAMES) <= alpha
    rng = np.random.default_rng(seed)
    # Paired bootstrap: same resampled image index drives all detector scores,
    # retaining arbitrary cross-detector dependence. Rows are iid *under this
    # simulated null*; they are not new independent empirical clean examples.
    curve_e = np.zeros(horizon, int)
    curve_fixed = np.zeros(horizon, int)
    delays = []
    attack_fixed_delays = []
    example_records = []
    for j in range(streams):
        indices = rng.integers(n, size=horizon)
        ledger = WealthLedger(f'clean-{j}', bets, alpha, delta)
        fixed_seen = False
        for t, idx in enumerate(indices):
            row = ledger.step({k: float(clean[k][idx]) for k in NAMES})
            fixed_seen |= bool(fixed(p_clean)[idx])
            curve_e[t] += row['alarm']
            curve_fixed[t] += fixed_seen
        if j == 0:
            example_records = ledger.records[:min(8, horizon)]
        indices = rng.integers(len(attack[NAMES[0]]), size=horizon)
        attacked = WealthLedger(f'attack-{j}', bets, alpha, delta)
        first_fixed = None
        for t, idx in enumerate(indices, 1):
            attacked.step({k: float(attack[k][idx]) for k in NAMES})
            if first_fixed is None and fixed(p_attack)[idx]:
                first_fixed = t
        delays.append(attacked.first_alarm)
        attack_fixed_delays.append(first_fixed)
    valid = [v for v in delays if v is not None]
    fixed_valid = [v for v in attack_fixed_delays if v is not None]
    return {
        'alpha': alpha, 'delta_total': delta, 'stream_count': streams,
        'horizon': horizon, 'n_cal': len(cal[NAMES[0]]), 'n_clean_holdout': n,
        'n_attack_view': len(attack[NAMES[0]]),
        'bets': {k: vars(v) for k, v in bets.items()},
        'clean_false_alarm_curve': {'e': (curve_e / streams).tolist(),
                                    'repeated_fixed_bonf': (curve_fixed / streams).tolist()},
        'attack_detection': {'e_alarm_fraction': len(valid) / streams,
                             'e_median_delay_detected_only': float(np.median(valid)) if valid else None,
                             'e_mean_delay_detected_only': float(np.mean(valid)) if valid else None,
                             'fixed_alarm_fraction': len(fixed_valid) / streams,
                             'fixed_median_delay_detected_only': float(np.median(fixed_valid)) if fixed_valid else None,
                             'e_delays': delays, 'fixed_delays': attack_fixed_delays},
        'clean_ledger_example': example_records,
        'limitations': 'Conditional guarantee assumes future iid null images (or predictable conditional tail bounds), frozen calibration and bets; DKW fails under unmodeled drift, correlated replays, clustered contributors or adaptive score selection. Bootstrap replicates reuse held-out images and do not constitute 200 independent clean assets. Same-image clean and triggered views measure a counterfactual probe, not a live evolving deployment. Per-asset alpha does not control alarms across many assets. Wealth can decrease; running maximum is monotone. No posterior or FDR guarantee.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='runs/mvp')
    ap.add_argument('--model', default=None, help='score one model per invocation')
    ap.add_argument('--out', required=True)
    ap.add_argument('--streams', type=int, default=200)
    ap.add_argument('--horizon', type=int, default=80)
    args = ap.parse_args()
    registry = load_registry(args.corpus)
    clean = next((r for r in registry if r['manifest']['ground_truth']['kind'] == 'clean'), None)
    if clean is None:
        raise ValueError('clean model required for enrolled reference')
    reference = ModelArtifact.load(clean['dir']).model
    backgrounds = make_backgrounds(8, seed=1)
    rows = []
    for entry in registry:
        if args.model and entry['manifest']['model_id'] != args.model:
            continue
        cal, null, attack = score_views(entry, reference, backgrounds)
        result = simulate(cal, null, attack, streams=args.streams,
                          horizon=args.horizon)
        rows.append({'model_id': entry['manifest']['model_id'],
                     'kind': entry['manifest']['ground_truth']['kind'],
                     'backdoor_weak': entry['manifest']['quality_flags'].get('backdoor_weak', False),
                     'result': result})
        print(rows[-1]['model_id'], 'unscorable' if result.get('unscorable') else
              (result['clean_false_alarm_curve']['e'][-1],
               result['clean_false_alarm_curve']['repeated_fixed_bonf'][-1],
               result['attack_detection']['e_median_delay_detected_only']), flush=True)
    output = {'base_commit': 'c17141f', 'corpus': args.corpus, 'rows': rows}
    Path(args.out).write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
