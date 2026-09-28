"""Independent clean fit/calibration, held-out multi-seed power curve."""
import json
from cviaf.lab.label_consistency import LabelConsistencyGate
from cviaf.lab.synth import SceneSpec, build_dataset
from cviaf.lab.poison import AttackSpec, inject
from cviaf.lab.baseline import cviaf_data_asset_verdict

N = 240
FIT_SEEDS = range(10000,10004)
CAL_SEEDS = range(11000,11080)
VAL_SEEDS = range(20000,20032)
RATES = [0, 1/240, .005, .01, .02, .03, .05, .10, .20]


def main():
    fit = [build_dataset(N, SceneSpec(seed=s)) for s in FIT_SEEDS]
    calibration = [build_dataset(N, SceneSpec(seed=s)) for s in CAL_SEEDS]
    gate = LabelConsistencyGate.fit(fit, calibration)
    rows = []
    for seed in VAL_SEEDS:
        ds = build_dataset(N, SceneSpec(seed=seed))
        for rate in RATES:
            attacked, truth = inject(ds, AttackSpec(kind='label_flip', seed=seed + 42, rate=rate)) if rate else (ds, None)
            verdict = cviaf_data_asset_verdict('label_flip_battery', {}, label_gate=gate, dataset=attacked)
            detail = verdict.evidence['label_consistency']
            rows.append(dict(seed=seed, rate=rate,
                             flips=len(truth.poisoned_indices) if truth else 0,
                             mismatches=detail['mismatch_images'], p_value=detail['p_value'],
                             flagged=verdict.flagged))
    curve = [dict(rate=r, flips=sorted(set(v['flips'] for v in rows if v['rate']==r)),
                  power=sum(v['flagged'] for v in rows if v['rate']==r)/len(VAL_SEEDS),
                  n=len(VAL_SEEDS)) for r in RATES]
    floor = next((v['rate'] for v in curve[1:] if v['power']>=.8), None)
    result = dict(base='c17141f0f58f4a5f6f0afce8e0f5719c61fcdd46', n=N,
                  fit_seeds=list(FIT_SEEDS), calibration_seeds=list(CAL_SEEDS),
                  validation_seeds=list(VAL_SEEDS), calibration_counts=gate.null_counts.tolist(),
                  alpha_total=.05, alpha_label=.025,
                  clean_fpr=curve[0]['power'], floor_rate=floor,
                  curve=curve, rows=rows,
                  interpretation='Synthetic same-domain colour/shape scenes only; no claims about real imagery or adversarial adaptation.')
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
