"""Measure model-attack signals against the MEASURED clean spread.

Questions answered by measurement rather than assumption:

  1. Is a tamper variant a real attack? It must be harmful, and it must move the
     model further than two honestly trained clean models differ -- because that
     difference IS the null, and anything smaller is noise dressed as capability.
  2. What is the null of each detector? Measured leave-one-out over clean models,
     and for the paired fingerprint over every ordered clean pair, so the tolerance
     is empirical rather than an artefact of how many clean models we happen to have.
  3. At which detection threshold should the fingerprint read the probe battery?
     Chosen by sweeping the NULL, not by sweeping the attacks: picking a threshold
     because it separates the attacks we built would be fitting the corpus.

Run:  .venv/bin/python scripts/tamper_probe.py            # full analysis
      .venv/bin/python scripts/tamper_probe.py --sweep    # threshold sweep only
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cviaf.lab.detectors import (PAIRED_FEATURES, behavioral_fingerprint,
                                benign_variation_scale, fingerprint_distance,
                                paired_fingerprint, standardised_deviation,
                                weight_stats)
from cviaf.lab.evaluate import load_registry, train_spec_from_manifest
from cviaf.lab.train import ModelArtifact, build_splits, detection_quality

CORPUS = os.environ.get("CVIAF_CORPUS", "runs/day1")
N_PROBE = int(os.environ.get("CVIAF_PROBE_IMAGES", "80"))
THRESH = float(os.environ.get("CVIAF_PF_THRESH", "0.5"))
SWEEP = (0.15, 0.25, 0.40, 0.50, 0.60, 0.75)


def clean_population(corpus=CORPUS, n_probe=N_PROBE):
    reg = [e for e in load_registry(corpus)
           if e["manifest"]["ground_truth"]["kind"] == "clean"]
    reg.sort(key=lambda e: e["manifest"]["model_id"])
    out = []
    for e in reg:
        spec = train_spec_from_manifest(e["manifest"])
        splits = build_splits(spec)
        art = ModelArtifact.load(e["dir"])
        out.append({
            "id": e["manifest"]["model_id"], "model": art.model,
            "stats": weight_stats(art.model),
            "images": splits.eval_clean.images[:n_probe],
            "boxes": splits.eval_clean.boxes[:n_probe],
            "labels": splits.eval_clean.labels[:n_probe],
            "quality": detection_quality(art.model, splits.eval_clean.images,
                                         splits.eval_clean.boxes, splits.eval_clean.labels),
        })
    return out


def sum_stats(x):
    x = np.asarray(x, np.float64)
    return {"mean": float(x.mean()), "sd": float(x.std()),
            "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def scale_of(fp_list):
    return np.stack(fp_list).std(axis=0) + 1e-9


def dist(fp, ref_fp, scale):
    return float(np.sqrt(np.sum(((np.asarray(fp) - np.asarray(ref_fp))
                                 / np.maximum(scale, 1e-9)) ** 2)))


def whitened_null(P):
    """Covariance of the null feature differences, with a ridge for conditioning.

    The per-feature z-distance treats the eight readings as independent, and they are
    not: a suspect that misses objects the reference saw also reports a different
    object count, so ``ref_only_rate`` and ``sus_only_rate`` move together. An L2 sum
    over correlated coordinates lets a joint shift cancel itself out -- measured on
    ``obj_2`` (which costs 47% of F1) the L2 distance stayed inside the null while the
    individual readings were plainly abnormal. The whitened (Mahalanobis) distance is
    the version that accounts for that, and its covariance is MEASURED from the clean
    pairs rather than assumed diagonal.
    """
    C = np.cov(np.asarray(P, np.float64).T)
    C = C + np.eye(C.shape[0]) * 1e-6 * float(np.trace(C) / C.shape[0])
    return np.linalg.pinv(C)


def mahal(fp, mean, inv_cov):
    d = np.asarray(fp, np.float64) - np.asarray(mean, np.float64)
    return float(np.sqrt(max(0.0, float(d @ inv_cov @ d))))


def pair_null(pop, thresh, n_probe=None):
    """Every ordered clean pair: suspect i read against reference j."""
    fps = []
    for i, pi in enumerate(pop):
        imgs = pi["images"] if n_probe is None else pi["images"][:n_probe]
        for j, pj in enumerate(pop):
            if i == j:
                continue
            fps.append(paired_fingerprint(pi["model"], pj["model"], imgs,
                                          score_thresh=thresh))
    P = np.stack(fps)
    sc = scale_of(fps)
    mean = P.mean(axis=0)
    d = np.asarray([dist(f, mean, sc) for f in fps])
    return P, sc, mean, d


def severity_ladder(model):
    out = {}
    for s in (0.2, 0.4, 0.8):
        out[f"noise_{s:g}"] = model.tamper_head(scale=s, seed=7)
    for d in (1.0, 2.0, 4.0):
        out[f"bias_{d:g}"] = model.tamper_bias(delta=d, target=1)
    for d in (0.5, 1.0, 2.0):
        out[f"obj_{d:g}"] = model.tamper_objectness(delta=d)
    for frac in (0.0625, 0.125, 0.25, 0.5):
        out[f"prune_{frac:g}"] = model.tamper_prune(frac=frac, seed=7)
    return out


def sweep(pop, n_probe=32):
    print(f"threshold sweep for the paired fingerprint "
          f"({len(pop)} clean models, {n_probe} probe images, "
          f"{len(pop)*(len(pop)-1)} ordered pairs)")
    print(f"{'thresh':>7s} {'jaccard':>8s} {'ref_only':>9s} {'sus_only':>9s} "
          f"{'iou_mat':>8s} | {'null_mean':>9s} {'null_p95':>9s} {'null_max':>9s}")
    for t in SWEEP:
        P, sc, mean, d = pair_null(pop, t, n_probe=n_probe)
        print(f"{t:7.2f} {P[:,0].mean():8.3f} {P[:,1].mean():9.3f} {P[:,2].mean():9.3f} "
              f"{P[:,3].mean():8.3f} | {d.mean():9.3f} "
              f"{np.percentile(d, 95):9.3f} {d.max():9.3f}")


def main():
    pop = clean_population()
    q = [p["quality"]["f1"] for p in pop]
    print(f"clean population: {len(pop)} models; F1 mean {np.mean(q):.3f} "
          f"sd {np.std(q):.3f} range [{min(q):.3f}, {max(q):.3f}]")
    print("  (this F1 spread IS the behavioural null: two honest clean models differ "
          "this much)")

    keys = list(pop[0]["stats"].keys())
    w_rows = []
    for i, p in enumerate(pop):
        others = [o["stats"] for j, o in enumerate(pop) if j != i]
        ref = {k: float(np.mean([o[k] for o in others if o.get(k) is not None])) for k in keys}
        sd = {k: float(np.std([o[k] for o in others if o.get(k) is not None])) for k in keys}
        r = standardised_deviation([p["stats"][k] for k in keys], [ref[k] for k in keys],
                                   [sd[k] for k in keys])
        w_rows.append(r)
    print(f"\nNULL  white-box weight deviation (leave-one-out, n={len(w_rows)})")
    for k in ("z_mean", "z_max", "z_rms"):
        s = sum_stats([r[k] for r in w_rows])
        print(f"  {k:7s} mean {s['mean']:7.3f} p95 {s['p95']:7.3f} max {s['max']:7.3f}")
    print(f"  constant-statistic violations in the null: "
          f"{sum(len(r['violated_constant_statistics']) for r in w_rows)}/{len(w_rows)}")

    ref_fps = [behavioral_fingerprint(p["model"], p["images"]) for p in pop]
    unpaired_scale = benign_variation_scale(ref_fps)
    ref_fp_mean = np.mean(np.stack(ref_fps), axis=0)

    P, pair_scale, P_mean, d_null = pair_null(pop, THRESH)
    print(f"\nNULL  paired fingerprint at threshold {THRESH} "
          f"({len(P)} ordered clean pairs)")
    print(f"  paired   mean {d_null.mean():7.3f} p95 {np.percentile(d_null, 95):7.3f} "
          f"max {d_null.max():7.3f}")
    for k, f in enumerate(PAIRED_FEATURES):
        tag = "  <- continuous belief" if k >= 5 else ""
        print(f"    {f:20s} mean {P[:,k].mean():8.4f} sd {P[:,k].std():8.4f}{tag}")

    base, ref = pop[0], pop[1]
    ref_stats = ref["stats"]
    w_scale = {k: float(np.std([p["stats"][k] for p in pop])) for k in keys}
    ref_seq_fp = np.mean(np.stack([behavioral_fingerprint(p["model"], base["images"])
                                   for p in pop if p["id"] != base["id"]]), axis=0)
    d_unpaired_null = np.asarray([fingerprint_distance(f, ref_fp_mean, unpaired_scale)
                                  for f in ref_fps])
    print(f"  unpaired mean {d_unpaired_null.mean():7.3f} p95 "
          f"{np.percentile(d_unpaired_null, 95):7.3f}   (the retired statistic)")

    rows = []
    for name, m in severity_ladder(base["model"]).items():
        qual = detection_quality(m, base["images"], base["boxes"], base["labels"])
        agg = standardised_deviation([weight_stats(m)[k] for k in keys],
                                     [ref_stats[k] for k in keys],
                                     [w_scale[k] for k in keys])
        fpp = paired_fingerprint(m, ref["model"], base["images"], score_thresh=THRESH)
        rows.append({
            "variant": name, "f1": float(qual["f1"]),
            "f1_relative_drop": float((base["quality"]["f1"] - qual["f1"])
                                      / base["quality"]["f1"]),
            "w_z_mean": agg["z_mean"], "w_z_max": agg["z_max"], "w_z_rms": agg["z_rms"],
            "w_constant_violations": len(agg["violated_constant_statistics"]),
            "paired_dist": dist(fpp, P_mean, pair_scale),
            "unpaired_dist": fingerprint_distance(
                behavioral_fingerprint(m, base["images"]), ref_seq_fp, unpaired_scale)})

    inv_cov = whitened_null(P)
    d_m_null = np.asarray([mahal(f, P_mean, inv_cov) for f in P])
    print(f"  whitened mean {d_m_null.mean():7.3f} p95 "
          f"{np.percentile(d_m_null, 95):7.3f} max {d_m_null.max():7.3f}  "
          f"(8 dims -> chi2_8 mean would be {np.sqrt(8):.2f})")
    for r in rows:
        r["whitened_dist"] = mahal(
            paired_fingerprint(_VARIANT_MODELS[r["variant"]], ref["model"],
                               base["images"], score_thresh=THRESH), P_mean, inv_cov)

    np95 = {k: float(np.percentile([r[k] for r in w_rows], 95))
            for k in ("z_mean", "z_max", "z_rms")}
    up95 = float(np.percentile(d_unpaired_null, 95))
    pp95 = float(np.percentile(d_null, 95))
    mp95 = float(np.percentile(d_m_null, 95))
    hdr = (f"\n{'variant':12s} {'F1':>6s} {'drop':>6s} | {'w_zmean':>8s} {'xnull':>6s} "
           f"{'w_zrms':>7s} {'xnull':>6s} {'const':>5s} | {'paired':>7s} {'xnull':>6s} | "
           f"{'unpair':>7s} {'xnull':>6s} | {'whiten':>7s} {'xnull':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['variant']:12s} {r['f1']:6.3f} {r['f1_relative_drop']:6.3f} | "
              f"{r['w_z_mean']:8.3f} {r['w_z_mean']/max(np95['z_mean'],1e-9):6.2f} "
              f"{r['w_z_rms']:7.3f} {r['w_z_rms']/max(np95['z_rms'],1e-9):6.2f} "
              f"{r['w_constant_violations']:5d} | {r['paired_dist']:7.3f} "
              f"{r['paired_dist']/max(pp95,1e-9):6.2f} | {r['unpaired_dist']:7.3f} "
              f"{r['unpaired_dist']/max(up95,1e-9):6.2f} | {r['whitened_dist']:7.3f} "
              f"{r['whitened_dist']/max(mp95,1e-9):6.2f}")
    print(f"\nthreshold {THRESH}; null p95: w_z_mean {np95['z_mean']:.3f}, "
          f"w_z_rms {np95['z_rms']:.3f}, paired {pp95:.3f}, unpaired {up95:.3f}")

    os.makedirs("runs", exist_ok=True)
    with open("runs/tamper_probe.json", "w") as fh:
        json.dump({"clean_models": [p["id"] for p in pop],
                   "threshold": THRESH, "n_probe_images": N_PROBE,
                   "clean_f1": {"mean": float(np.mean(q)), "sd": float(np.std(q))},
                   "null_weight": {k: sum_stats([r[k] for r in w_rows])
                                   for k in ("z_mean", "z_max", "z_rms")},
                   "null_paired_fingerprint": sum_stats(d_null),
                   "null_unpaired_fingerprint": sum_stats(d_unpaired_null),
                   "paired_feature_scales": dict(zip(PAIRED_FEATURES,
                                                     (float(v) for v in pair_scale))),
                   "null_whitened": sum_stats(d_m_null),
                   "null_p95": {**np95, "paired": pp95, "unpaired": up95,
                                "whitened": mp95},
                   "variants": rows}, fh, indent=1)
    print("wrote runs/tamper_probe.json")


if __name__ == "__main__":
    pop = clean_population(n_probe=32 if "--sweep" in sys.argv else N_PROBE)
    if "--sweep" in sys.argv:
        sweep(pop)
    else:
        main()
