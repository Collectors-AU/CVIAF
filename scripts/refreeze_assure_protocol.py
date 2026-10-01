#!/usr/bin/env python
"""Re-freeze the pinned assure calibration protocol -- but only for a stale pin.

`configs/calibration/assure-synthetic-mvp-240.json` records, among other things, the
sha256 of the *source* of the detectors it was calibrated with (`_versions()` in
`cviaf/lab/assure_protocol.py`). Any later edit to those modules invalidates the
fixture, and `load_protocol` correctly refuses it -- fail-closed, which is the point.
But that leaves two red tests and a confusing error, and the failure mode of the
alternative (hand-editing the JSON) is that a *real* calibration change gets laundered
in as a re-freeze.

So this script does the re-freeze in the only case where it is safe: regenerate the
protocol from the same seeds and refuse unless every difference is either the source
hash pin itself or a floating-point difference below a declared tolerance. A changed
`effect_floor`, a moved seed, a different held-out count -- any of those means the
calibration itself moved, which is a decision for the owner, not a maintenance edit.

It also refuses to touch a dirty fixture without `--apply`, and prints what it would do.

CLI
---
    python scripts/refreeze_assure_protocol.py --corpus runs/mvp \
        [--drift-calibration configs/calibration/natural-drift-...json] \
        [--tolerance 1e-9] [--apply]

Exit: 0 re-frozen (or already current), 1 the difference is substantive, 2 could not run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_OUT = "configs/calibration/assure-synthetic-mvp-240.json"
DEFAULT_DRIFT = "configs/calibration/natural-drift-synthetic-mvp-120x240.json"

# Keys whose difference is the *reason* to re-freeze rather than a surprise: the source
# digest pin is derived from the code, and sha256 is a digest over the whole payload.
PIN_KEYS = ("detector_versions", "sha256")


def _max_abs_delta(a: Any, b: Any) -> Optional[float]:
    """Largest |a-b| over a nested structure of numbers, or None if not comparable.

    None means the two values are not the same *kind* of thing (different length, a
    string against a float, a missing key): that is substantive by definition.
    """
    # A bool is not a number here: `True == 1` in Python, so a flag that flipped between
    # runs would otherwise read as a zero delta on a numeric field.
    if isinstance(a, bool) != isinstance(b, bool):
        return None
    if isinstance(a, bool) and isinstance(b, bool):
        return 0.0 if a == b else None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    # fit_protocol keeps its seed lists as tuples in memory while the fixture stores
    # them as JSON lists: sequence *type* is not a difference, only content is.
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return None
        deltas = [_max_abs_delta(x, y) for x, y in zip(a, b)]
        if any(d is None for d in deltas):
            return None
        return max(deltas) if deltas else 0.0
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return None
        deltas = [_max_abs_delta(a[k], b[k]) for k in a]
        if any(d is None for d in deltas):
            return None
        return max(deltas) if deltas else 0.0
    if isinstance(a, str) and isinstance(b, str):
        return 0.0 if a == b else None
    return None if a != b else 0.0


def classify_diff(frozen: Dict[str, Any], refit: Dict[str, Any],
                  tolerance: float = 1e-9) -> Dict[str, Any]:
    """Split the differences into 'explains a re-freeze' and 'do not touch this'.

    Returns ``{"allowed": {...}, "substantive": {...}, "ok": bool}``. A numeric drift
    below `tolerance` is recorded as noise; anything else that is not a PIN_KEY is
    substantive and blocks the re-freeze.
    """
    allowed: Dict[str, Any] = {}
    substantive: Dict[str, Any] = {}
    for key in sorted(set(frozen) | set(refit)):
        if key not in frozen or key not in refit:
            substantive[key] = {"reason": "key present on one side only"}
            continue
        if frozen[key] == refit[key]:
            continue
        if key in PIN_KEYS:
            allowed[key] = {"reason": "source pin / derived digest"}
            continue
        delta = _max_abs_delta(frozen[key], refit[key])
        if delta is None:
            substantive[key] = {"reason": "not a numeric difference"}
        elif delta <= tolerance:
            allowed[key] = {"reason": f"floating-point noise {delta:.3g} <= {tolerance:g}",
                            "max_abs_delta": delta}
        else:
            substantive[key] = {"reason": f"numeric change {delta:.6g} > {tolerance:g}",
                                "max_abs_delta": delta}
    return {"allowed": allowed, "substantive": substantive, "ok": not substantive,
            "tolerance": tolerance}


def refreeze(out: str, corpus: str, drift_calibration: Optional[str],
             tolerance: float = 1e-9, apply: bool = False,
             probe: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
    from cviaf.lab.assure_protocol import fit_protocol

    if not os.path.isfile(out):
        return 2, {"error": f"{out} does not exist; use fit_protocol to create it first"}
    frozen = json.load(open(out, encoding="utf-8"))
    probe = probe or os.path.join(os.path.dirname(out), "_refreeze_probe.json")
    if os.path.exists(probe):
        os.remove(probe)                    # fit_protocol refuses to overwrite
    try:
        refit = fit_protocol(corpus, probe, float(frozen["alpha"]),
                             drift_calibration=drift_calibration)
    except ValueError as exc:
        return 2, {"error": f"fit_protocol refused: {exc}"}
    verdict = classify_diff(frozen, refit, tolerance)
    result = {"out": out, "probe": probe, **verdict}
    if verdict["ok"] and apply:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(refit, fh, indent=2, sort_keys=True)
            fh.write("\n")
        result["applied"] = True
    elif apply:
        result["applied"] = False
    if verdict["ok"] and os.path.exists(probe):
        os.remove(probe)
    return (0 if verdict["ok"] else 1), result


def render(result: Dict[str, Any]) -> str:
    if "error" in result:
        return f"cannot re-freeze: {result['error']}"
    lines = [f"re-freeze {result.get('out', 'fixture')} "
             f"(tolerance {result.get('tolerance', 1e-9):g})"]
    if not result["allowed"] and not result["substantive"]:
        lines.append("  already current: the refit is identical to the fixture")
    for key, why in result["allowed"].items():
        lines.append(f"  allowed      {key:24s} {why['reason']}")
    for key, why in result["substantive"].items():
        lines.append(f"  SUBSTANTIVE  {key:24s} {why['reason']}")
    if result["ok"]:
        lines.append("  verdict: safe to re-freeze (pin/digest drift only)"
                     + ("  [applied]" if result.get("applied") else
                        "  [not applied: pass --apply]"))
    else:
        lines.append("  verdict: REFUSED -- the calibration itself moved; that is a "
                     "decision, not maintenance. Compare the keys above by hand.")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--corpus", default="runs/mvp")
    ap.add_argument("--drift-calibration", default=DEFAULT_DRIFT)
    ap.add_argument("--tolerance", type=float, default=1e-9)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)
    drift = args.drift_calibration if os.path.isfile(args.drift_calibration) else None
    if drift is None:
        print(f"warning: {args.drift_calibration} not found; refitting without the "
              f"drift arbiter (the fixture records its hash, so this will be substantive)",
              file=sys.stderr)
    code, result = refreeze(args.out, args.corpus, drift, args.tolerance, args.apply)
    print(render(result))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
