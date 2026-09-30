"""Reproduce Arm F paired fixed-batch screens, not a sequential monitor.

Run: python scripts/measure_arm_f.py > arm_f_metrics.json
The effect floor is derived on a disjoint clean calibration battery.
"""
import json
import numpy as np
from cviaf.lab.evaluate import train_spec_from_manifest
from cviaf.lab.train import build_splits
from cviaf.lab.synth import build_dataset, SceneSpec
from cviaf.lab.calibrate import conformal_pvalues, benjamini_yekutieli
from cviaf.lab.fusion import grouped_by_flags
from cviaf.drift import DistributionShiftAssessor, calibrate_effect_floor
from cviaf.utils import extract_features_from_images


def signal_pvalues(cal, observed):
    # Deterministic, correlated channels from the *same* images. Calibrate the
    # statistic on an independent clean split; do not score against itself.
    raw = {
        "mean_r": lambda x: x[:, 0], "mean_g": lambda x: x[:, 25],
        "std_r": lambda x: x[:, 1], "std_g": lambda x: x[:, 26],
        "hist_r": lambda x: x[:, 18], "hist_g": lambda x: x[:, 41],
    }
    return {name: conformal_pvalues(fn(cal), fn(observed))
            for name, fn in raw.items()}


def group_benchmark(spec, n=100):
    splits = build_splits(spec)
    cal = extract_features_from_images(splits.cal_clean.images, target_dim=64)
    groups = {"colour_means": ("mean_r", "mean_g"),
              "colour_variability": ("std_r", "std_g"),
              "histograms": ("hist_r", "hist_g")}
    rows = []
    for i in range(n):
        clean = build_dataset(spec.n_train, spec.scene,
                              seed_offset=1_100_000 + i * 71)
        obs = extract_features_from_images(clean.images, target_dim=64)
        p = signal_pvalues(cal, obs)
        raw_flags = benjamini_yekutieli(np.clip(len(p) * np.stack(list(p.values()),axis=1).min(axis=1),0,1))
        grouped_flags, _, _ = grouped_by_flags(p, groups)
        rows.append([bool(raw_flags.any()), bool(grouped_flags.any())])
    a = np.asarray(rows)
    return {"clean_batches": n, "batch_size": spec.n_train,
            "raw_bonf_then_BY_alarm_rate": round(float(a[:,0].mean()),4),
            "grouped_BY_then_BY_alarm_rate": round(float(a[:,1].mean()),4),
            "paired_counts": {"both": int(np.sum(a[:,0]&a[:,1])),
                              "raw_only": int(np.sum(a[:,0]&~a[:,1])),
                              "grouped_only": int(np.sum(~a[:,0]&a[:,1]))},
            "groups":groups}


def drift_benchmark(spec, n=100):
    splits = build_splits(spec)
    reference = extract_features_from_images(splits.cal_clean.images,target_dim=64)
    calibration = [extract_features_from_images(
        build_dataset(spec.n_cal, spec.scene, seed_offset=3_100_000 + i * 101).images,
        target_dim=64) for i in range(n)]
    policy = calibrate_effect_floor(reference, calibration, alpha=.05, minimum=.2)
    assessor = DistributionShiftAssessor(min_standardized_wasserstein=policy["floor"])
    rows=[]
    for i in range(n):
        ds=build_dataset(spec.n_cal,spec.scene,seed_offset=2_100_000 + i * 101)
        op=extract_features_from_images(ds.images,target_dim=64)
        # Paired tests: seed the existing permutation test for exact reruns.
        np.random.seed(1900+i)
        r=assessor.assess(reference,op)
        rows.append([r['effect_size']['significance_screen_passed'],r['shift_detected'],r['effect_size']['value']])
    a=np.asarray(rows)
    # Same declared screen on a large, manifest-derived operational terrain shift.
    forest = spec.scene.to_dict()
    forest["terrain"] = "forest"
    shifted = extract_features_from_images(build_dataset(
        spec.n_cal, SceneSpec(**forest), seed_offset=314159).images, target_dim=64)
    np.random.seed(2026)
    positive = assessor.assess(reference, shifted)
    return {"clean_batches":n,"batch_size":spec.n_cal,
            "forest_shift": {"before": positive["effect_size"]["significance_screen_passed"],
                             "after": positive["shift_detected"],
                             "effect": positive["effect_size"]["value"]},
            "before_alarm_rate":round(float(a[:,0].mean()),4),
            "after_alarm_rate":round(float(a[:,1].mean()),4),
            "paired_before_only":int(np.sum((a[:,0]==1)&(a[:,1]==0))),
            "effect_median":float(np.median(a[:,2])),
            "effect_floor":assessor.min_standardized_wasserstein,
            "policy":policy}


def main():
    import sys
    path=sys.argv[1] if len(sys.argv)>1 else 'runs/mvp/clean_none_fixed_s5/manifest.json'
    with open(path) as f: manifest=json.load(f)
    spec=train_spec_from_manifest(manifest)
    print(json.dumps({"base_commit":"c17141f","manifest":path,
                      "group_fusion":group_benchmark(spec),
                      "drift":drift_benchmark(spec)},indent=2))

if __name__=='__main__': main()
