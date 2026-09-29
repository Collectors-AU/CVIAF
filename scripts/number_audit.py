#!/usr/bin/env python
"""Does every number in the prose still match the receipt that produced it?

The audit in ``docs/PS26228_ALIGNMENT_MATRIX.md`` §3 was done by hand and that is not a
property, it is an afternoon. This script makes it a check.

Two directions, because drift has two shapes:

  * **forward** -- a canonical figure must appear in the document that is allowed to quote
    it. A number that quietly disappears is as much a defect as one that changes: the
    reader loses the receipt.
  * **backward** -- a *retired* figure (a superseded pooled rate, an old test count) must
    not appear outside the section that keeps it as history. This is the direction that
    catches the expensive error: the single-dose figures were superseded by the ladder,
    and the ladder refuses a pooled rate by name, so a pooled number left in the demo
    text is not a typo, it is a claim the receipt contradicts.

The figures are recomputed from the committed JSON on every run rather than hard-coded,
so this cannot pass by agreeing with a stale constant.

Exit codes: 0 clean, 3 drift (one or more checks failed), 4 a receipt is unreadable.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MERGED_REPORT = "runs/merged_fpr_tpr_report.json"
LADDER = "runs/tpr_ladder_at_frozen.json"
SINGLE_DOSE = "runs/tpr_at_frozen.json"
VERIFY = "runs/integration_verify.json"

#: (document, substring) pairs. The substring is built from the receipts by
#: :func:`canonical_strings`, so a receipt change fails the check instead of the doc.
FORWARD_CHECKS: List[Tuple[str, str]] = []

#: A retired figure is allowed only in these files, which exist to record it as history.
#: Paths are resolved from the worktree, so ``../STATUS.md`` is the parent checkout's
#: untracked working note. The dashboard is deliberately walked: a superseded number left
#: in the demo text is the one place a judge would read it as current.
RETIRED_ALLOWED_IN: Dict[str, Tuple[str, ...]] = {
    # the superseded single-dose detection pass, kept as a banner-ed historical section
    "34.7%": ("../STATUS.md", "docs/PS26228_ALIGNMENT_MATRIX.md"),
    "48.9%": ("../STATUS.md", "docs/PS26228_ALIGNMENT_MATRIX.md"),
    "193 arms, one dose of 0.25": ("../STATUS.md", "../SWOT.md"),
}

#: Where drift is looked for. ``docs/`` is the prose, ``demo/`` is what gets shown.
BACKWARD_ROOTS: Tuple[str, ...] = ("README.md", "../STATUS.md", "../SWOT.md", "docs",
                                   "demo")


def resolve(path: str) -> str:
    """Worktree-relative, but the two working notes live in the parent checkout."""
    return os.path.normpath(os.path.join(REPO, path))


def load(path: str) -> dict:
    with open(resolve(path), encoding="utf-8") as handle:
        return json.load(handle)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def canonical_strings() -> Dict[str, str]:
    """Recompute the figures the prose is required to quote, from the receipts."""
    report = load(MERGED_REPORT)
    ladder = load(LADDER)
    verify = load(VERIFY)
    rules = report["rules"]
    ctc = rules["ctc_mean_clean"]
    refdiv = rules["refdiv_mean_clean"]
    cells = ladder["cells"]

    sub = cells["substitution"]
    wt = cells["weight_tamper"]
    bias = cells["bias_lift"]
    # The receipt's `n_arms` counts scored arms only; the prose quotes both it and the
    # built total (scored + named-unscorable). Derive both, and require them to agree
    # with each other: a receipt that disagreed with itself about its own survivorship
    # curve fails here rather than in the prose.
    scored = ladder["n_arms"]
    unscorable_map = ladder["unscorable"]
    parents = sub["0.1"]["n"]
    per_dose = {d: parents - sub[str(d)]["n"] for d in (0.1, 0.25, 0.5, 1)}
    for dose, reconstructed in per_dose.items():
        declared = unscorable_map.get(f"substitution@{dose}", 0)
        if declared != reconstructed:
            raise ValueError(f"ladder receipt disagrees with itself on "
                             f"substitution@{dose}: declared {declared}, cells imply "
                             f"{reconstructed}")
    cell_total = sum(cells[k][str(d)]["n"] for k in cells for d in (0.1, 0.25, 0.5, 1))
    if cell_total != scored:
        raise ValueError(f"ladder receipt disagrees with itself: cells sum to "
                         f"{cell_total}, n_arms says {scored}")
    built = scored + sum(unscorable_map.values())
    verified = sum(src["n_verified"] for src in verify["sources"])
    planned = sum(src["n_models"] for src in verify["sources"])
    if verified != planned:
        raise ValueError(f"verify receipt disagrees with itself: {verified} verified "
                         f"against {planned} planned")

    return {
        "fpr_ctc": f"{ctc['fpr']['point_estimate'] * 100:.2f}%",
        "fpr_refdiv": f"{refdiv['fpr']['point_estimate'] * 100:.2f}%",
        "fpr_negatives": f"{report['denominators']['evaluation_negatives']:,}",
        "population": f"{report['census']['all']['_total']:,}",
        "verified": f"{verified:,}",
        "planned": f"{planned:,}",
        "arms_built": f"{built:,}",
        "arms_scored": f"{scored:,}",
        "sub_0_25": _pct(sub["0.25"]["rules"]["refdiv_mean_clean"]["tpr"]),
        "sub_0_50": _pct(sub["0.5"]["rules"]["refdiv_mean_clean"]["tpr"]),
        "wt_1_00": _pct(wt["1"]["rules"]["refdiv_mean_clean"]["tpr"]),
        "bias_any": _pct(bias["0.1"]["rules"]["refdiv_mean_clean"]["tpr"]),
        "unscorable_curve": " / ".join(f"{per_dose[d]}" for d in (0.1, 0.25, 0.5, 1)),
        "pooled_refused": "null" if ladder["pooled_estimate"] is None else "PRESENT",
    }


def forward_checks(canon: Dict[str, str]) -> List[Tuple[str, str]]:
    """The canonical figure -> the documents allowed (and required) to carry it."""
    return [
        ("README.md", canon["fpr_ctc"]),
        ("README.md", canon["fpr_refdiv"]),
        ("README.md", canon["population"]),
        ("README.md", canon["arms_built"]),
        ("README.md", canon["sub_0_25"]),
        ("docs/METHODOLOGY.md", canon["population"]),
        ("docs/MODEL_INVENTORY.md", canon["population"]),
        ("docs/PS26228_ALIGNMENT_MATRIX.md", canon["fpr_ctc"]),
        ("docs/PS26228_ALIGNMENT_MATRIX.md", canon["fpr_refdiv"]),
        ("docs/PS26228_ALIGNMENT_MATRIX.md", canon["population"]),
        ("docs/PS26228_ALIGNMENT_MATRIX.md", canon["arms_built"]),
        ("docs/PS26228_ALIGNMENT_MATRIX.md", canon["arms_scored"]),
        ("docs/PS26228_ALIGNMENT_MATRIX.md", canon["unscorable_curve"]),
        ("docs/PS26228_ALIGNMENT_MATRIX.md", canon["pooled_refused"]),
        ("../STATUS.md", canon["pooled_refused"]),
    ]


def _read(path: str) -> str:
    with open(resolve(path), encoding="utf-8") as handle:
        return handle.read()


def run_forward(canon: Dict[str, str],
                checks: List[Tuple[str, str]] | None = None) -> List[str]:
    problems: List[str] = []
    for doc, needle in (forward_checks(canon) if checks is None else checks):
        try:
            text = _read(doc)
        except OSError:
            problems.append(f"{doc}: unreadable")
            continue
        if needle not in text:
            problems.append(f"{doc}: missing canonical figure {needle!r}")
    return problems


def run_backward(roots: Tuple[str, ...] = BACKWARD_ROOTS,
                 retired: Dict[str, Tuple[str, ...]] | None = None) -> List[str]:
    """Retired figures must not survive outside the file that records them as history."""
    problems: List[str] = []
    for needle, allowed in (RETIRED_ALLOWED_IN if retired is None else retired).items():
        for doc in _walk(roots):
            if doc in allowed:
                continue
            try:
                text = _read(doc)
            except (OSError, UnicodeDecodeError):
                continue
            if needle in text:
                problems.append(f"{doc}: retired figure {needle!r} (allowed only in "
                                f"{', '.join(allowed)})")
    return problems


def _walk(roots: Tuple[str, ...]) -> List[str]:
    """Return document paths in the same spelling the allowlist uses."""
    out: List[str] = []
    for root in roots:
        full = resolve(root)
        if os.path.isfile(full):
            out.append(root)
            continue
        for dirpath, dirnames, files in os.walk(full):
            dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git")]
            for name in sorted(files):
                if name.endswith((".md", ".html")):
                    rel = os.path.relpath(os.path.join(dirpath, name), REPO)
                    out.append(rel.replace(os.sep, "/"))
    return out


def main() -> int:
    try:
        canon = canonical_strings()
    except (OSError, KeyError, ValueError) as exc:
        print(f"number-audit: cannot read a receipt: {exc}", file=sys.stderr)
        return 4
    problems = run_forward(canon) + run_backward()
    checked = len(forward_checks(canon)) + len(RETIRED_ALLOWED_IN)
    if problems:
        print(f"number-audit: DRIFT ({len(problems)} of {checked} checks failed)")
        for problem in problems:
            print(f"  - {problem}")
        return 3
    print(f"number-audit: clean ({checked} checks), figures recomputed from the receipts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
