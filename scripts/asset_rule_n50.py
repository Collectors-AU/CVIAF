"""Asset-level model-integrity rule re-measured at n = 50 clean null models.

Task 1 trained 50 exchangeable clean models, which is the capacity this rule needs: at
the merged default (m = 1 gate signal, alpha = .05) it requires 5 center + 19 rank
models, so the conformal p-value floor is 1/20 = 5%; at the pre-merge n = 11 the floor
was 1/7 = 14.3% and the rule could only ever abstain. This script spends that capacity:

  * clean_null leave-one-out: every one of the 50 clean models is scored as a suspect
    against the other 49 as calibration. This is the rule's own false-positive rate
    (denominator 50), which no previous run could estimate.
  * day1 attacked models: the real stamp-based attacks already in the repo, by kind.
  * stampfree arms: the Task-2 stamp-free backdoor cell.

The statistic is the paired stamp response (mean change in a detector's score when the
predeclared probe is stamped), computed with ONE shared reference model and ONE shared
probe set for every model, so responses are comparable across populations.

Run: .venv/bin/python scripts/asset_rule_n50.py --out runs/asset_rule_n50.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from cviaf.lab.detectors import make_backgrounds, reference_divergence, trace_ctc
from cviaf.lab.evaluate import load_registry, train_spec_from_manifest
from cviaf.lab.model_asset_rule import DEFAULT_SIGNALS, decide_model_asset, stamp_response
from cviaf.lab.poison import AttackSpec, trigger_view
from cviaf.lab.synth import DetectionDataset
from cviaf.lab.train import ModelArtifact, build_splits

PROBE_SOURCE = "runs/clean_null/clean_none_fixed_s100"
REFERENCE_MODEL = "runs/mvp/clean_none_fixed_s5"
CLEAN_CORPUS = "runs/clean_null"
ATTACKED_CORPUS = "runs/day1"
STAMPFREE_DIR = "runs/stampfree"
N_EVAL = 24
N_BACKGROUNDS = 3
ALPHA = .05
SIGNALS = ("refdiv", "ctc")
LEGACY_GATE = ("ctc", "refdiv")


def probe_frames():
    with open(os.path.join(PROBE_SOURCE, "manifest.json")) as fh:
        manifest = json.load(fh)
    # This call is also the union check: it rebuilds an AttackSpec from a manifest
    # written BEFORE the merge, which carried `mechanism` and no `scope`.
    spec = train_spec_from_manifest(manifest)
    ds = build_splits(spec).eval_clean
    probe = DetectionDataset(images=ds.images[:N_EVAL], boxes=ds.boxes[:N_EVAL],
                             labels=ds.labels[:N_EVAL],
                             contributors=ds.contributors[:N_EVAL],
                             batches=ds.batches[:N_EVAL], spec=ds.spec)
    attack = AttackSpec(kind="oga", trigger="patch", trigger_loc="fixed",
                        trigger_size=10, target_class=0, rate=0.2, seed=11)
    stamped, _ = trigger_view(probe, attack, 1)
    ref = ModelArtifact.load(REFERENCE_MODEL).model
    return probe.images, stamped, make_backgrounds(N_BACKGROUNDS, seed=1), ref, float(
        np.max(np.abs(stamped - probe.images)))


def response(model, bare, stamped, backgrounds, reference) -> Dict[str, float]:
    out = {}
    for name in SIGNALS:
        if name == "ctc":
            a = np.asarray(trace_ctc(model, bare, backgrounds)["score"], float)
            b = np.asarray(trace_ctc(model, stamped, backgrounds)["score"], float)
        else:
            a = np.asarray(reference_divergence(model, reference, bare)["score"], float)
            b = np.asarray(reference_divergence(model, reference, stamped)["score"], float)
        ok = np.isfinite(a) & np.isfinite(b)
        out[name] = float(np.mean(b[ok] - a[ok]))
    return out


def _cp_upper(k: int, n: int, conf: float = .95) -> float:
    from scipy.stats import beta
    return 0.0 if k == n else float(beta.ppf(conf, k + 1, n - k))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/asset_rule_n50.json")
    ap.add_argument("--max-attacked", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    bare, stamped, backgrounds, reference, stamp_delta = probe_frames()
    print(f"probe: n_eval={len(bare)} stamp max|delta|={stamp_delta:.4f} "
          f"reference={REFERENCE_MODEL}", flush=True)

    clean = [e for e in load_registry(CLEAN_CORPUS)
             if e["manifest"]["ground_truth"]["kind"] == "clean"]
    clean_resp: Dict[str, Dict[str, float]] = {}
    for e in clean:
        clean_resp[e["manifest"]["model_id"]] = response(
            ModelArtifact.load(e["dir"]).model, bare, stamped, backgrounds, reference)
    print(f"clean responses: {len(clean_resp)}", flush=True)

    attacked = [e for e in load_registry(ATTACKED_CORPUS)
                if e["manifest"]["ground_truth"]["kind"] != "clean"]
    if args.max_attacked:
        attacked = attacked[:args.max_attacked]
    att_resp: List[Dict[str, Any]] = []
    for i, e in enumerate(attacked):
        r = response(ModelArtifact.load(e["dir"]).model, bare, stamped, backgrounds, reference)
        att_resp.append({"model_id": e["manifest"]["model_id"],
                         "kind": e["manifest"]["ground_truth"]["kind"], **r})
        if (i + 1) % 20 == 0:
            print(f"  attacked scored {i+1}/{len(attacked)}", flush=True)

    sf_resp: List[Dict[str, Any]] = []
    if os.path.isdir(STAMPFREE_DIR):
        for e in load_registry(STAMPFREE_DIR):
            if e["manifest"]["ground_truth"]["kind"] != "stampfree":
                continue
            r = response(ModelArtifact.load(e["dir"]).model, bare, stamped, backgrounds, reference)
            sf_resp.append({"model_id": e["manifest"]["model_id"],
                            "kind": "stampfree",
                            "manifest_asr_present_net":
                                e["manifest"]["stampfree"]["effect"]["asr_present_net"],
                            **r})

    clean_ids = sorted(clean_resp)
    clean_rows = [{"model_id": k, **clean_resp[k]} for k in clean_ids]

    def decide(resp: Dict[str, float], calibration: List[Dict[str, Any]],
               signals=("refdiv",)) -> Dict[str, Any]:
        return decide_model_asset(resp, calibration, alpha=ALPHA, signals=signals)

    # --- clean leave-one-out: the rule's own false-positive rate at n=50 ---
    loo = []
    for k in clean_ids:
        cal = [{"model_id": j, **clean_resp[j]} for j in clean_ids if j != k]
        resp = {"model_id": k, **clean_resp[k]}
        d = decide(clean_resp[k], cal)
        l = decide(clean_resp[k], cal, signals=LEGACY_GATE)
        loo.append({"model_id": k, "refdiv_gate_p": d["asset_pvalue"],
                    "refdiv_gate_reject": d["reject_at_alpha"],
                    "legacy_gate_p": l["asset_pvalue"], "legacy_gate_reject": l["reject_at_alpha"],
                    "n_calibration": len(cal), "ctc_diagnostic_p":
                        d["diagnostic_per_signal"]["ctc"]["p"]})
    n_loo = len(loo)
    loo_reject_refdiv = sum(r["refdiv_gate_reject"] for r in loo)
    loo_reject_legacy = sum(r["legacy_gate_reject"] for r in loo)

    # --- attacked assets vs the full 50-model clean null ---
    def verdicts(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for row in rows:
            d = decide(row, clean_rows)
            l = decide(row, clean_rows, signals=LEGACY_GATE)
            out.append({**{k: v for k, v in row.items()},
                        "refdiv_gate_p": d["asset_pvalue"], "refdiv_gate_reject": d["reject_at_alpha"],
                        "legacy_gate_p": l["asset_pvalue"], "legacy_gate_reject": l["reject_at_alpha"],
                        "abstain": d["abstain"],
                        "required_clean_models": d["required_clean_models"],
                        "ctc_diagnostic_p": d.get("diagnostic_per_signal", {})
                                            .get("ctc", {}).get("p")})
        return out

    att_verdicts = verdicts(att_resp)
    sf_verdicts = verdicts(sf_resp)

    by_kind: Dict[str, Dict[str, Any]] = {}
    for kind in sorted({r["kind"] for r in att_verdicts}):
        rows = [r for r in att_verdicts if r["kind"] == kind]
        n = len(rows)
        rej = sum(r["refdiv_gate_reject"] for r in rows)
        by_kind[kind] = {
            "n": n, "refdiv_gate_rejected": rej,
            "refdiv_gate_rejected_frac": rej / n if n else None,
            "legacy_gate_rejected": sum(r["legacy_gate_reject"] for r in rows),
            "median_refdiv_p": float(np.median([r["refdiv_gate_p"] for r in rows])) if n else None,
            "rows": rows,
        }

    result = {
        "schema": "cviaf.asset-rule-n50/1",
        "alpha": ALPHA, "probe": {"source": PROBE_SOURCE, "n_eval": N_EVAL,
                                  "reference_model": REFERENCE_MODEL,
                                  "stamp_max_abs_pixel_delta": stamp_delta,
                                  "signals_computed": list(SIGNALS),
                                  "gate": list(DEFAULT_SIGNALS)},
        "capacity": {"gate_signals": list(DEFAULT_SIGNALS),
                     "required_clean_models_at_alpha": 5 + int(np.ceil(1 / ALPHA)) - 1,
                     "conformal_p_floor_at_n50": 1.0 / (50 - 5 + 1),
                     "pre_merge_n11_p_floor": 1.0 / (11 - 5 + 1)},
        "clean_responses": clean_rows,
        "clean_leave_one_out": {
            "n_suspects": n_loo, "n_calibration_each": n_loo - 1,
            "refdiv_gate_rejected": loo_reject_refdiv,
            "refdiv_gate_fpr": loo_reject_refdiv / n_loo if n_loo else None,
            "refdiv_gate_fpr_upper95": _cp_upper(loo_reject_refdiv, n_loo),
            "legacy_two_signal_rejected": loo_reject_legacy,
            "legacy_two_signal_fpr": loo_reject_legacy / n_loo if n_loo else None,
            "rows": loo,
        },
        "attacked": {"n": len(att_verdicts),
                     "refdiv_gate_rejected": sum(r["refdiv_gate_reject"] for r in att_verdicts),
                     "legacy_gate_rejected": sum(r["legacy_gate_reject"] for r in att_verdicts),
                     "by_kind": by_kind, "rows": att_verdicts},
        "stampfree": {"n": len(sf_verdicts),
                      "refdiv_gate_rejected": sum(r["refdiv_gate_reject"] for r in sf_verdicts),
                      "legacy_gate_rejected": sum(r["legacy_gate_reject"] for r in sf_verdicts),
                      "rows": sf_verdicts},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1, allow_nan=False)

    print("\n=== ASSET RULE AT n=50 ===")
    print(f"capacity: gate={result['capacity']['gate_signals']} "
          f"required={result['capacity']['required_clean_models_at_alpha']} "
          f"p-floor(n=50)={result['capacity']['conformal_p_floor_at_n50']:.4f} "
          f"(was {result['capacity']['pre_merge_n11_p_floor']:.4f} at n=11)")
    lo = result["clean_leave_one_out"]
    print(f"clean leave-one-out: {lo['refdiv_gate_rejected']}/{lo['n_suspects']} rejected "
          f"(FPR {lo['refdiv_gate_fpr']:.3f}, CP upper {lo['refdiv_gate_fpr_upper95']:.3f}); "
          f"legacy ctc+refdiv gate {lo['legacy_two_signal_rejected']}/{lo['n_suspects']}")
    print(f"attacked assets: {result['attacked']['refdiv_gate_rejected']}/{result['attacked']['n']} "
          f"rejected on the refdiv gate "
          f"({result['attacked']['legacy_gate_rejected']}/{result['attacked']['n']} legacy)")
    for kind, v in sorted(by_kind.items()):
        print(f"   {kind:16s} {v['refdiv_gate_rejected']:3d}/{v['n']:<3d} "
              f"median p={v['median_refdiv_p']:.3f}")
    # Pre-merge capacity, from the same responses: at n=11 the rule cannot reject at all.
    d11 = decide_model_asset(clean_resp[clean_ids[0]], clean_rows[:11])
    result["capacity"]["n11_abstains"] = bool(d11["abstain"])
    result["capacity"]["n11_reason"] = d11.get("reason")
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1, allow_nan=False)
    print(f"pre-merge n=11: abstain={d11['abstain']} ({d11.get('reason')})")

    sf = result["stampfree"]
    print(f"stamp-free assets: {sf['refdiv_gate_rejected']}/{sf['n']} rejected on the refdiv gate")
    for r in sf["rows"]:
        print(f"   {r['model_id']:26s} refdiv_resp={r['refdiv']:+.5f} ctc_resp={r['ctc']:+.5f} "
              f"p={r['refdiv_gate_p']:.3f} reject={r['refdiv_gate_reject']} "
              f"(measured present-net ASR {r['manifest_asr_present_net']})")
    print(f"wrote {args.out} in {result['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
