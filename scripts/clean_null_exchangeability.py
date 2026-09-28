"""Task 1 acceptance harness: capacity + exchangeability of the clean null corpus.

Two questions, both measured rather than asserted:

  1. DISJOINTNESS. Every model in ``runs/clean_null`` must have its own held-out
     split. Recomputed from each manifest's spec (not read back from the
     manifest's own recorded digest), then compared against every other corpus in
     ``runs/`` so no held-out image is shared with an existing population.

  2. EXCHANGEABILITY. The null the asset-level rule prices is the *paired stamp
     response* of a clean model: the mean change in each detector's score when the
     same model sees the same probe images with a predeclared trigger stamped on
     them. That is the quantity ``model_asset_rule.stamp_response`` feeds to
     ``decide_model_asset``, so the spread of that quantity over clean models IS
     the null. A new clean model is exchangeable with the existing population only
     if its response distribution matches theirs. Same reference model, same probe
     images, same trigger recipe for every model -- only the model differs.

  3. CAPACITY. ``decide_model_asset`` needs ``n_center=5`` center models plus
     ``ceil(m/alpha)-1`` rank models, i.e. 44 clean models at m=2 signals and
     alpha=.05. This script reports the conformal p-value floor ``1/(n_rank+1)``
     at each population size, which is the honest statement of what n=11 blocks.

Run: .venv/bin/python scripts/clean_null_exchangeability.py --out runs/clean_null/exchangeability.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import numpy as np
from scipy import stats as sps

from cviaf.lab.detectors import make_backgrounds, reference_divergence, trace_ctc
from cviaf.lab.detector import DetectorConfig
from cviaf.lab.evaluate import load_registry, train_spec_from_manifest
from cviaf.lab.poison import AttackSpec, trigger_view
from cviaf.lab.synth import SceneSpec
from cviaf.lab.train import ModelArtifact, TrainSpec, build_splits

CORPORA = {
    "clean_null": "runs/clean_null",
    "day1": "runs/day1",
    "mvp2": "runs/mvp2",
}
# Trusted reference model and predeclared probe recipe: identical for every model.
REFERENCE_DIR = "runs/mvp/clean_none_fixed_s5"
PROBE_SOURCE = "runs/clean_null/clean_none_fixed_s100"
SIGNALS = ("ctc", "refdiv")
N_EVAL = 24
N_BACKGROUNDS = 3
ALPHA = 0.05
SIGNAL_COUNT = len(SIGNALS)


def probe_frames():
    """(bare probe images, stamped probe images, backgrounds, reference model).

    One probe set, drawn once, used unchanged for every model in every corpus:
    responses are only comparable if the stimulus is literally identical.
    """
    with open(os.path.join(PROBE_SOURCE, "manifest.json")) as fh:
        manifest = json.load(fh)
    spec = train_spec_from_manifest(manifest)
    splits = build_splits(spec)
    ds = splits.eval_clean
    from cviaf.lab.synth import DetectionDataset
    ds = DetectionDataset(images=ds.images[:N_EVAL], boxes=ds.boxes[:N_EVAL],
                          labels=ds.labels[:N_EVAL], contributors=ds.contributors[:N_EVAL],
                          batches=ds.batches[:N_EVAL], spec=ds.spec)
    attack = AttackSpec(kind="oga", trigger="patch", trigger_loc="fixed",
                        trigger_size=10, target_class=0, rate=0.2, seed=11)
    stamped, _ = trigger_view(ds, attack, 1)
    reference = ModelArtifact.load(REFERENCE_DIR).model
    backgrounds = make_backgrounds(N_BACKGROUNDS, seed=1)
    pixel_delta = float(np.max(np.abs(stamped - ds.images)))
    return ds.images, stamped, backgrounds, reference, attack, pixel_delta


def stamp_response(model, bare, stamped, backgrounds, reference) -> Dict[str, float]:
    out = {}
    for name in SIGNALS:
        if name == "ctc":
            a = np.asarray(trace_ctc(model, bare, backgrounds)["score"], float)
            b = np.asarray(trace_ctc(model, stamped, backgrounds)["score"], float)
        else:
            a = np.asarray(reference_divergence(model, reference, bare)["score"], float)
            b = np.asarray(reference_divergence(model, reference, stamped)["score"], float)
        ok = np.isfinite(a) & np.isfinite(b)
        if not ok.any():
            raise ValueError(f"{name}: no finite paired probe scores")
        out[name] = float(np.mean(b[ok] - a[ok]))
    return out


def summarise(values: List[float]) -> Dict[str, Any]:
    a = np.asarray(values, float)
    return {"n": int(len(a)), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else None,
            "min": float(a.min()), "q25": float(np.quantile(a, .25)),
            "median": float(np.median(a)), "q75": float(np.quantile(a, .75)),
            "max": float(a.max())}


def capacity_table() -> Dict[str, Any]:
    """What clean-model count each population actually buys the asset rule."""
    required_rank = int(np.ceil(SIGNAL_COUNT / ALPHA)) - 1
    rows = {}
    for n in (11, 20, 50, 44, 45):
        n_rank = max(0, n - 5)
        floor = (1.0 / (n_rank + 1)) if n_rank > 0 else None
        rows[f"n={n}"] = {
            "n_center": min(5, n), "n_rank": n_rank,
            "conformal_p_floor": floor,
            "can_reject_at_alpha": bool(floor is not None and floor <= ALPHA),
            "abstains": bool(floor is None or floor > ALPHA),
        }
    return {"signals": list(SIGNALS), "alpha": ALPHA,
            "rule": "decide_model_asset: n_center=5 + ceil(m/alpha)-1 rank models",
            "required_clean_models": 5 + required_rank, "by_population": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/clean_null/exchangeability.json")
    ap.add_argument("--max-models", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()

    print("building shared probe frames ...", flush=True)
    bare, stamped, backgrounds, reference, attack, pixel_delta = probe_frames()
    print(f"  probe n_eval={len(bare)} stamp max|delta|={pixel_delta:.4f} "
          f"(the stamp IS a pixel artifact; this is the control's stimulus)", flush=True)

    report: Dict[str, Any] = {
        "schema": "cviaf.clean-null-acceptance/1",
        "probe": {"source": PROBE_SOURCE, "n_eval": N_EVAL,
                  "reference_model": REFERENCE_DIR,
                  "attack": {"kind": attack.kind, "trigger": attack.trigger,
                             "trigger_loc": attack.trigger_loc,
                             "trigger_size": int(attack.trigger_size), "seed": attack.seed},
                  "signals": list(SIGNALS), "n_backgrounds": N_BACKGROUNDS,
                  "stamp_max_abs_pixel_delta": pixel_delta},
        "disjointness": {}, "exchangeability": {}, "capacity": capacity_table(),
    }

    responses: Dict[str, Dict[str, Dict[str, float]]] = {}
    digests: Dict[str, set] = {}

    for corpus, path in CORPORA.items():
        if not os.path.isdir(path):
            report["disjointness"][corpus] = {"available": False}
            continue
        entries = [e for e in load_registry(path)
                   if e["manifest"]["ground_truth"]["kind"] == "clean"]
        if args.max_models:
            entries = entries[:args.max_models]
        responses[corpus] = {}
        digests[corpus] = set()
        per_model: List[Dict[str, Any]] = []
        n_bad_weights = 0
        for e in entries:
            mid = e["manifest"]["model_id"]
            mdir = e["dir"]
            have = (os.path.isfile(os.path.join(mdir, "manifest.json"))
                    and os.path.isfile(os.path.join(mdir, "weights.npz")))
            if not have:
                n_bad_weights += 1
                continue
            spec = train_spec_from_manifest(e["manifest"])
            ev_digest = build_splits(spec).eval_clean.digest()
            digests[corpus].add(ev_digest)
            try:
                r = stamp_response(ModelArtifact.load(mdir).model, bare, stamped,
                                   backgrounds, reference)
            except Exception as exc:  # a model we cannot score is reported, not dropped
                per_model.append({"model_id": mid, "error": str(exc)})
                continue
            responses[corpus][mid] = r
            per_model.append({"model_id": mid, "eval_clean_digest": ev_digest, **r})
            print(f"  {corpus:11s} {mid:26s} ctc={r['ctc']:+.5f} refdiv={r['refdiv']:+.5f}",
                  flush=True)
        n = len(entries)
        report["disjointness"][corpus] = {
            "n_clean_entries": n,
            "n_with_manifest_and_weights": n - n_bad_weights,
            "unique_eval_clean_digests": len(digests[corpus]),
            "all_unique": bool(len(digests[corpus]) == n - n_bad_weights),
            "digests_recomputed_from_spec_not_manifest": True,
            "manifests_record_dataset_digests": bool(
                all("dataset_digests" in e["manifest"] for e in entries) if entries else None),
            "dataset_digest_keys": sorted(entries[0]["manifest"]["dataset_digests"].keys())
            if entries and isinstance(entries[0]["manifest"].get("dataset_digests"), dict) else None,
        }
        report["exchangeability"][corpus] = {
            "per_signal": {s: summarise([r[s] for r in responses[corpus].values()])
                           for s in SIGNALS},
            "n_scored": len(responses[corpus]),
            "models": per_model,
        }

    # cross-corpus overlap of held-out splits
    overlap = {}
    names = [c for c in CORPORA if digests.get(c)]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap[f"{a}__{b}"] = len(digests[a] & digests[b])
    report["disjointness"]["cross_corpus_eval_overlap"] = overlap

    # exchangeability tests: new 50 vs each existing population
    new = responses.get("clean_null", {})
    tests = {}
    for other in ("day1", "mvp2"):
        old = responses.get(other, {})
        if not new or not old:
            tests[f"clean_null__{other}"] = {"available": False}
            continue
        entry: Dict[str, Any] = {"n_new": len(new), "n_existing": len(old)}
        for s in SIGNALS:
            x = np.asarray([r[s] for r in new.values()], float)
            y = np.asarray([r[s] for r in old.values()], float)
            ks = sps.ks_2samp(x, y)
            # Welch CI on the mean difference: the exchangeability claim is that
            # the two populations are the same distribution, not merely close means.
            se = float(np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)))
            diff = float(x.mean() - y.mean())
            entry[s] = {
                "mean_new": float(x.mean()), "mean_existing": float(y.mean()),
                "mean_diff": diff,
                "mean_diff_ci95": [diff - 1.96 * se, diff + 1.96 * se],
                "ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
                "sd_new": float(x.std(ddof=1)), "sd_existing": float(y.std(ddof=1)),
                "exchangeable_at_05": bool(ks.pvalue >= 0.05),
            }
        tests[f"clean_null__{other}"] = entry
    report["exchangeability"]["tests_new_vs_existing"] = tests
    report["runtime_seconds"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1, allow_nan=False)

    print("\n=== ACCEPTANCE ===")
    for c in CORPORA:
        d = report["disjointness"].get(c, {})
        if d.get("available", True) is False:
            continue
        print(f"{c:11s} clean={d['n_clean_entries']:3d} unique_eval_digests="
              f"{d['unique_eval_clean_digests']:3d} all_unique={d['all_unique']}")
    print("cross-corpus eval overlap:", overlap)
    for k, v in tests.items():
        if v.get("available", True) is False:
            continue
        for s in SIGNALS:
            print(f"{k} {s:7s} new={v[s]['mean_new']:+.5f} old={v[s]['mean_existing']:+.5f} "
                  f"KS={v[s]['ks_statistic']:.3f} p={v[s]['ks_pvalue']:.4f} "
                  f"exchangeable={v[s]['exchangeable_at_05']}")
    print("capacity:", json.dumps(report["capacity"], indent=1))
    print(f"wrote {args.out} in {report['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
