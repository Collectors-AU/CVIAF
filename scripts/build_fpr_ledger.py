#!/usr/bin/env python
"""Build an FPR/TPR score ledger from corpora already on disk.

Produces the locked `cviaf.fpr-tpr-ledger.v1` input (see `cviaf/lab/fpr_tpr.py`) so the
harness can be validated on real artefacts long before the 20k corpus lands. The
producer is deliberately dumb — it computes one score per model per signal and records
the declared ground truth. All the discipline (calibration/test separation, denominator
floors, interval arithmetic) lives in the harness, not here.

Signals (mirroring `compare.model_axis`, the canonical single-signal baseline):

  ``ctc_mean_clean``    mean over the model's held-out CLEAN images of the TRACE CTC
                        stability score. High = the model's object belief is unstable
                        under background blending.
  ``ctc_q95_clean``     95th percentile of the same scores (robust, non-saturating).
  ``ctc_peak_clean``    max of the same scores. Kept ONLY for continuity with
                        ``compare.model_axis`` — see the note below on why it is a bad
                        per-asset statistic and must not be the only signal.
  ``refdiv_mean_clean`` mean reference divergence against an independent clean
                        reference model. High = the model misses objects the reference
                        finds.

Why the peak is kept but not trusted: measured on this repo's own corpora, the
max-over-images CTC score saturates at 1.0 on *clean* models (clean_none_fixed_s104 →
1.0, s6 → 0.9997), so a per-asset rule built on the peak cannot separate clean from
tampered assets at all — which is exactly why the baseline row in `compare.json`
reports FPR 1.0 for the fixed-threshold rule. The mean and the 95th percentile are
non-degenerate and are the signals the harness should be read on.

Both are computed on CLEAN images only: that keeps the statistic well defined for
weight-space attacks, which have no test-time trigger at all.

Runtime is bounded by ``--n-eval`` (images per model) x models x backgrounds. The
synthetic corpora are numpy-only and fast; a real-backbone corpus runs a torch forward
per (image, background) and should be launched in the background.

Run:
    cd .task3 && PYTHONPATH=. <venv>/bin/python scripts/build_fpr_ledger.py \\
        --corpus runs/clean_null runs/day1 --out runs/fpr_ledger.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cviaf.lab.detectors import make_backgrounds, reference_divergence, trace_ctc
from cviaf.lab.evaluate import load_registry, train_spec_from_manifest
from cviaf.lab.poison import MODEL_ATTACK_KINDS as POISON_MODEL_ATTACKS
from cviaf.lab.train import ModelArtifact, build_splits

LEDGER_SCHEMA = "cviaf.fpr-tpr-ledger.v1"

# Ground truth. Every kind that appears in any corpus on disk must be declared, or the
# harness refuses the ledger — an undeclared kind silently becoming a negative is the
# exact defect this whole format exists to prevent.
POSITIVE_KINDS = sorted(set(POISON_MODEL_ATTACKS) |
                        {"oga", "oda", "rma", "gma", "clean_label", "label_flip",
                         "dup_flood", "ood_insert", "stampfree"})
NEGATIVE_KINDS = ["clean"]


def eval_images_for(manifest: Dict[str, Any], n_eval: int, real_eval) -> Optional[np.ndarray]:
    """Held-out clean images for one model, from the corpus's own declared split."""
    spec = train_spec_from_manifest(manifest)
    if spec is None:
        if real_eval is None:
            return None
        return real_eval.images[:n_eval]
    splits = build_splits(spec)
    return splits.eval_clean.images[:n_eval]


def summarise(imgs: np.ndarray, backgrounds: np.ndarray, model, ref_model):
    """All summary statistics for one model on its held-out clean images."""
    scores: Dict[str, float] = {}
    ctc = np.asarray(trace_ctc(model, imgs, backgrounds)["score"], np.float64)
    ctc = ctc[np.isfinite(ctc)]
    if ctc.size:
        scores["ctc_mean_clean"] = float(ctc.mean())
        scores["ctc_q95_clean"] = float(np.quantile(ctc, 0.95))
        scores["ctc_peak_clean"] = float(ctc.max())
    if ref_model is not None:
        rd = np.asarray(reference_divergence(model, ref_model, imgs)["score"],
                        np.float64)
        rd = rd[np.isfinite(rd)]
        if rd.size:
            scores["refdiv_mean_clean"] = float(rd.mean())
    return scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", nargs="+", required=True,
                    help="corpus dirs holding registry.jsonl (one ledger record each)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-eval", type=int, default=40,
                    help="held-out clean images per model per signal")
    ap.add_argument("--backgrounds", type=int, default=4)
    ap.add_argument("--max-models", type=int, default=None, help="per corpus, for a smoke run")
    ap.add_argument("--signals", nargs="+",
                    default=["ctc_mean_clean", "ctc_q95_clean", "ctc_peak_clean",
                             "refdiv_mean_clean"])
    ap.add_argument("--reference", default=None,
                    help="model DIR to use as the reference for every corpus. Required "
                         "for a multi-shard fleet: each shard would otherwise pick its "
                         "own reference and refdiv scores would not be comparable")
    ap.add_argument("--write-every", type=int, default=1,
                    help="rewrite the output ledger every N models (0 = only at the end); "
                         "a long fleet run that dies keeps its progress")
    args = ap.parse_args()

    backgrounds = make_backgrounds(args.backgrounds, seed=1)
    records: List[Dict[str, Any]] = []
    started = time.time()
    real_eval = None
    ref_model = None
    ref_id: Optional[str] = None

    def write_ledger() -> None:
        """Atomically dump what has been scored so far."""
        ledger = {
            "schema": LEDGER_SCHEMA,
            "alpha": 0.05,
            "higher_is_more_anomalous": True,
            "positive_kinds": POSITIVE_KINDS,
            "negative_kinds": NEGATIVE_KINDS,
            "provenance": {
                "producer": "scripts/build_fpr_ledger.py",
                "corpora": list(args.corpus),
                "n_eval_per_model": args.n_eval,
                "n_backgrounds": args.backgrounds,
                "signals": args.signals,
                "reference_model": ref_id,
                "reference_dir": args.reference,
                "partial": len(records) == 0 or None,
                "notes": ("scores are computed on held-out CLEAN images only, so a "
                          "weight-space tamper with no test-time trigger is still "
                          "scored; splits are left unassigned for the harness to "
                          "separate"),
            },
            "records": records,
        }
        ledger["provenance"]["partial"] = not done[0]
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(ledger, fh, indent=1)
        os.replace(tmp, args.out)

    done = [False]

    # Pick ONE cross-corpus clean reference up front and exclude it from the ledger.
    # A model cannot be scored against itself: if it were, a single missing score
    # would silently drop a signal for every record (measured on the smoke run) or,
    # worse, its own zero divergence would sit in the null as a fake clean asset.
    if args.reference:
        from cviaf.lab.train import ModelArtifact as _MA
        ref_model = _MA.load(args.reference).model
        ref_id = os.path.basename(os.path.normpath(args.reference))
        print(f"  reference model: {ref_id} from --reference (excluded by id)")
    else:
        for corpus in args.corpus:
            for e in load_registry(corpus):
                if e["manifest"]["ground_truth"]["kind"] == "clean":
                    ref_id = e["manifest"]["model_id"]
                    ref_model = ModelArtifact.load(e["dir"]).model
                    break
            if ref_model is not None:
                break
        print(f"  reference model: {ref_id} (excluded from the scored population)")

    for corpus in args.corpus:
        registry = load_registry(corpus)
        if not registry:
            print(f"  [skip] {corpus}: no models")
            continue
        if train_spec_from_manifest(registry[0]["manifest"]) is None:
            # real-backbone corpus: one shared held-out CIFAR split for every model
            from cviaf.lab.evaluate import real_backbone_eval_split
            real_eval = real_backbone_eval_split()
            print(f"  real-backbone corpus: shared eval split "
                  f"{real_eval.digest()[:16]} ({len(real_eval)} images)")

        entries = [e for e in registry
                   if e["manifest"]["model_id"] != ref_id]
        entries = entries[:args.max_models] if args.max_models else entries
        for i, e in enumerate(entries, 1):
            m = e["manifest"]
            imgs = eval_images_for(m, args.n_eval, real_eval)
            if imgs is None or len(imgs) == 0:
                print(f"  [skip] {m['model_id']}: no eval images")
                continue
            try:
                art = ModelArtifact.load(e["dir"])
            except Exception as exc:                       # malformed artefact
                print(f"  [skip] {m['model_id']}: {type(exc).__name__}: {exc}")
                continue
            scores = summarise(imgs, backgrounds, art.model, ref_model)
            scores = {k: v for k, v in scores.items() if k in args.signals}
            if not scores:
                continue
            records.append({
                "model_id": m["model_id"],
                "corpus": corpus,
                "kind": m["ground_truth"]["kind"],
                "is_positive": m["ground_truth"]["kind"] in POSITIVE_KINDS,
                "split": "unassigned",
                "scores": scores,
            })
            if i % 10 == 0 or i == len(entries):
                print(f"  {corpus} [{i}/{len(entries)}] {m['model_id']:38s} "
                      f" { {k: round(v, 4) for k, v in scores.items()} }", flush=True)
            if args.write_every and i % args.write_every == 0:
                write_ledger()

    done[0] = True

    # Every signal must exist on every record, or per-rule denominators differ.
    if records:
        common = set(records[0]["scores"])
        for r in records:
            common &= set(r["scores"])
        dropped = {s for r in records for s in r["scores"]} - common
        if dropped:
            print(f"  dropping signal(s) not present on every record: {sorted(dropped)}")
            for r in records:
                r["scores"] = {k: v for k, v in r["scores"].items() if k in common}
        records = [r for r in records if r["scores"]]

    write_ledger()
    kinds: Dict[str, int] = {}
    for r in records:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"wrote {args.out}: {len(records)} records in {time.time()-started:.0f}s")
    print(f"  kinds: {dict(sorted(kinds.items()))}")
    print(f"  next: python -m cviaf.lab.fpr_tpr --input {args.out} "
          f"--out {os.path.splitext(args.out)[0]}_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
