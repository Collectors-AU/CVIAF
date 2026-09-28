"""Factorial stamp/backdoor controls for the synthetic detector corpus.

Run: python -m cviaf.lab.null_suite --corpus runs/day1 --out /tmp/null.json
No weights are trained or changed. The four cells use the SAME held-out images
and trigger recipe per seed/attack. Never interpret a stamp contrast as evidence
of a trained backdoor.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from cviaf.lab.baseline import asset_pvalue_from_items
from cviaf.lab.calibrate import auroc, conformal_pvalues
from cviaf.lab.detectors import make_backgrounds, reference_divergence, trace_ctc, trace_ftc
from cviaf.lab.evaluate import load_registry, train_spec_from_manifest
from cviaf.lab.poison import trigger_view
from cviaf.lab.train import ModelArtifact, attack_success_rate, build_splits

SIGNALS = ("ctc", "refdiv", "ftc", "fft")
FUSIONS = {"with_ftc": ("ctc", "refdiv", "ftc"),
           "without_ftc": ("ctc", "refdiv")}
CELLS = ("clean_unstamped", "clean_stamped", "backdoored_unstamped", "backdoored_stamped", "peer_clean_stamped")


def fft_energy(images):
    """High-frequency image energy, explicitly an image-only stamp control."""
    arr = np.asarray(images, np.float64).mean(axis=-1)
    freq = np.fft.fftshift(np.fft.fft2(arr, axes=(-2, -1)), axes=(-2, -1))
    h, w = arr.shape[1:]
    y, x = np.ogrid[:h, :w]
    high = (abs(y - h // 2) > h // 4) | (abs(x - w // 2) > w // 4)
    return np.mean(abs(freq[:, high]) ** 2, axis=1)


def score_cells(clean_model, attacked_model, reference, bare, stamped, backgrounds,
                stride, decoy_class, peer_clean=None):
    result = {}
    for name, model, images in (
        ("clean_unstamped", clean_model, bare),
        ("clean_stamped", clean_model, stamped),
        ("backdoored_unstamped", attacked_model, bare),
        ("backdoored_stamped", attacked_model, stamped),
    ):
        result[name] = {
            "ctc": trace_ctc(model, images, backgrounds)["score"],
            "refdiv": reference_divergence(model, reference, images)["score"],
            "ftc": trace_ftc(model, images, decoy_class=decoy_class,
                             stride=stride)["score"],
            "fft": fft_energy(images),
        }
    if peer_clean is not None:
        model, images = peer_clean, stamped
        result["peer_clean_stamped"] = {
            "ctc": trace_ctc(model, images, backgrounds)["score"],
            "refdiv": reference_divergence(model, reference, images)["score"],
            "ftc": trace_ftc(model, images, decoy_class=decoy_class, stride=stride)["score"],
            "fft": fft_energy(images),
        }
    return result


def pair_metric(neg, pos):
    a, b = np.asarray(neg, float), np.asarray(pos, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return {"auroc": None, "tpr_at_5fpr": None, "n_neg": len(a), "n_pos": len(b)}
    # Ties are never split at a threshold. This is the conservative attainable TPR.
    cuts = np.unique(np.concatenate([a, b]))
    thresholds = np.concatenate([[np.inf], cuts])
    tpr = max(float(np.mean(b >= t)) for t in thresholds if np.mean(a >= t) <= .05)
    return {"auroc": float(auroc(np.r_[a, b], np.r_[np.zeros(len(a)), np.ones(len(b))])),
            "tpr_at_5fpr": tpr, "n_neg": len(a), "n_pos": len(b)}


def pvalues(calibration, test):
    out = {}
    for cell in CELLS:
        out[cell] = {}
        for signal in SIGNALS:
            values = np.asarray(test[cell][signal], float)
            cal = np.asarray(calibration["clean_stamped"][signal], float)
            cal = cal[np.isfinite(cal)]
            out[cell][signal] = (np.where(np.isfinite(values),
                                        conformal_pvalues(cal, np.nan_to_num(values, nan=-np.inf)),
                                        1.0) if len(cal) >= 5 else np.ones(len(values)))
        for name, members in FUSIONS.items():
            stack = np.stack([out[cell][s] for s in members], axis=0)
            out[cell][name] = np.clip(len(members) * np.min(stack, axis=0), 0, 1)
    return out


def behavioral_probe(clean, suspect, ds, spec, seed):
    """Prediction outcomes, rather than patch pixels. Same criterion in both arms."""
    null = attack_success_rate(clean, ds, spec, seed)
    attack = attack_success_rate(suspect, ds, spec, seed)
    images, _ = trigger_view(ds, spec, seed)
    def flip(model):
        changes = 0
        for bare, stamped in zip(ds.images, images):
            a = model.predict(bare)
            b = model.predict(stamped)
            changes += int(tuple(np.bincount(a["labels"], minlength=3)) !=
                           tuple(np.bincount(b["labels"], minlength=3)))
        return changes / len(ds.images)
    return {"criterion": attack["criterion"], "clean_asr": null["asr"],
            "backdoored_asr": attack["asr"], "net_asr": attack["asr"] - null["asr"],
            "n_clean": null["n"], "n_backdoored": attack["n"],
            "clean_prediction_flip_rate": flip(clean),
            "backdoored_prediction_flip_rate": flip(suspect)}


def run(corpus, seeds=(5, 6, 7), attacks=("oga", "oda", "rma"), n_eval=24,
        n_cal=32, stride=16, background_count=3, alpha=.05):
    if not 0 < alpha < 1 or n_eval < 1 or n_cal < 20:
        raise ValueError("invalid alpha or sample sizes; n_cal must be >=20 for alpha=.05")
    registry = load_registry(corpus)
    lookup = {(e["manifest"]["spec"]["detector"]["seed"],
               e["manifest"]["ground_truth"]["kind"]): e for e in registry}
    rows = []
    for seed in seeds:
        clean_entry = lookup.get((seed, "clean"))
        if not clean_entry:
            raise ValueError(f"missing clean model for seed {seed}")
        clean = ModelArtifact.load(clean_entry["dir"]).model
        peer_entry = lookup.get((seed + 3, "clean"))
        reference_entry = lookup.get((seed + 6, "clean"))
        if not peer_entry:
            raise ValueError(f"missing peer clean model for seed {seed + 3}")
        peer = ModelArtifact.load(peer_entry["dir"]).model
        if not reference_entry:
            raise ValueError(f"missing independent clean reference for seed {seed + 6}")
        reference = ModelArtifact.load(reference_entry["dir"]).model
        backgrounds = make_backgrounds(background_count, seed=seed)
        for kind in attacks:
            entry = lookup.get((seed, kind))
            if not entry:
                raise ValueError(f"missing {kind} model for seed {seed}")
            manifest = entry["manifest"]
            suspect = ModelArtifact.load(entry["dir"]).model
            spec = train_spec_from_manifest(manifest)
            splits = build_splits(spec)
            # Evaluation and calibration are disjoint by construction. The original
            # corpus uses the same scene recipe for clean and attacked seed peers.
            if clean_entry["manifest"]["spec"]["scene"] != manifest["spec"]["scene"]:
                raise ValueError("clean/attacked scenes differ; cannot form paired control")
            ev = splits.eval_clean
            cal = splits.cal_clean
            if len(ev) < n_eval or len(cal) < n_cal:
                raise ValueError("requested sample count exceeds corpus split")
            from cviaf.lab.synth import DetectionDataset
            def first(ds, n):
                return DetectionDataset(images=ds.images[:n], boxes=ds.boxes[:n],
                                        labels=ds.labels[:n], contributors=ds.contributors[:n],
                                        batches=ds.batches[:n], spec=ds.spec)
            ev, cal = first(ev, n_eval), first(cal, n_cal)
            ev_stamped, _ = trigger_view(ev, spec.attack, seed)
            cal_stamped, _ = trigger_view(cal, spec.attack, seed + 1000)
            calibration = score_cells(clean, suspect, reference, cal.images, cal_stamped,
                                      backgrounds, stride, spec.attack.target_class, peer_clean=peer)
            test = score_cells(clean, suspect, reference, ev.images, ev_stamped,
                               backgrounds, stride, spec.attack.target_class, peer_clean=peer)
            p = pvalues(calibration, test)
            metrics = {}
            for contrast, neg, pos in (
                ("stamp_null", "clean_unstamped", "clean_stamped"),
                ("backdoor_conditional_stamped", "clean_stamped", "backdoored_stamped"),
                ("peer_clean_null_stamped", "clean_stamped", "peer_clean_stamped"),
                ("backdoor_conditional_unstamped", "clean_unstamped", "backdoored_unstamped"),
                ("naive_confound", "clean_unstamped", "backdoored_stamped")):
                metrics[contrast] = {}
                for signal in SIGNALS:
                    metrics[contrast][signal] = pair_metric(test[neg][signal], test[pos][signal])
                for fusion in FUSIONS:
                    metrics[contrast][fusion] = pair_metric(-p[neg][fusion], -p[pos][fusion])
            assets = {cell: {} for cell in CELLS}
            for cell in CELLS:
                for fusion in FUSIONS:
                    val, _ = asset_pvalue_from_items(p[cell][fusion], method="cauchy")
                    assets[cell][fusion] = {"p": val, "reject": bool(val <= alpha)}
            behavior = behavioral_probe(clean, suspect, ev, spec.attack, seed)
            row = {"seed": seed, "attack": kind, "model_id": manifest["model_id"],
                   "independent_reference_id": reference_entry["manifest"]["model_id"],
                   "peer_clean_id": peer_entry["manifest"]["model_id"],
                   "n_eval": n_eval, "n_cal": n_cal,
                   "manifest_asr": manifest["metrics"]["attack_success_rate"],
                   "manifest_backdoor_weak": manifest["quality_flags"]["backdoor_weak"],
                   "behavior": behavior, "metrics": metrics, "assets": assets}
            rows.append(row)
    summary = {}
    for kind in attacks:
        group = [r for r in rows if r["attack"] == kind]
        summary[kind] = {}
        for contrast in group[0]["metrics"]:
            summary[kind][contrast] = {
                s: {m: (float(np.mean(v)) if v else None)
                    for m in ("auroc", "tpr_at_5fpr")
                    for v in [[r["metrics"][contrast][s][m] for r in group
                               if r["metrics"][contrast][s][m] is not None]]}
                for s in (*SIGNALS, *FUSIONS)}
        summary[kind]["asset_decisions"] = {
            f: {cell: {"rejected": sum(r["assets"][cell][f]["reject"] for r in group),
                       "total": len(group)} for cell in CELLS}
            for f in FUSIONS}
    return {"schema": "cviaf.null-suite.v1", "corpus": corpus, "alpha": alpha,
            "seeds": list(seeds), "attacks": list(attacks), "n_eval": n_eval,
            "n_cal": n_cal, "ftc_stride": stride, "n_backgrounds": background_count,
            "calibration": "per seed and attack, separate clean-model stamped calibration images; independent reference model (seed+6) for all arms, separate clean peer (seed+3); no test images calibrated on themselves",
            "contrasts": {"stamp_null": "clean_stamped vs clean_unstamped: visual stamp artifact",
                          "backdoor_conditional_stamped": "backdoored_stamped vs clean_stamped: model effect given identical stamp",
                          "peer_clean_null_stamped": "independently trained clean peer vs clean model on identical stamped images: normal model variation",
                          "backdoor_conditional_unstamped": "backdoored_unstamped vs clean_unstamped: untriggered model differences (not proof of a trigger)",
                          "naive_confound": "backdoored_stamped vs clean_unstamped: confounded; do not use as backdoor evidence"},
            "limitations": ["Fixed trained corpus, no new training; models may be behaviorally weak despite manifest ASR.",
                            "Only three seeds and synthetic detector; asset p-values are experimental, not validated type-I control.",
                            "Test-set prediction-flip rates and ASR are behavioral, but do not by themselves prove training-induced causality; compare to clean-model responses.",
                            "Independent clean reference removes self-reference tautology; peer-clean contrast measures one ordinary training-difference direction, not its full distribution.",
                            "FTC target class is known in advance; FFT is pixel-only and cannot distinguish model weights.",
                            "Unstamped-backdoored means backdoored weights on bare inputs, not a trigger-free backdoor mechanism."],
            "rows": rows, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="runs/day1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[5, 6, 7])
    ap.add_argument("--attacks", nargs="+", default=["oga", "oda", "rma"])
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--n-cal", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    args = ap.parse_args()
    result = run(args.corpus, args.seeds, args.attacks, args.n_eval,
                 args.n_cal, args.stride)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, allow_nan=False)
    print(args.out)


if __name__ == "__main__":
    main()
