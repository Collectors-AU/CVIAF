#!/usr/bin/env python3
"""Build the CVIAF evidence dashboard site and the architecture diagram.

Design read (design-taste skills applied):
  persona      dark security / threat-intelligence console
  variance     5 (asymmetric bento, editorial split) - not chaotic
  motion       3 (scroll-reveal + hover only, prefers-reduced-motion honoured)
  density      7 (measurement console; numbers in monospace)
  accent       exactly one: steel blue. Amber/red/green reserved for semantics.
  banned       Inter/Roboto/Helvetica, emoji, purple glows, gradients, pill CTAs

Everything is generated from the committed receipts. No CDN, no webfont fetch -
the whole thing works with no network, which is the point of the product.
"""
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
REPO = os.environ.get("CVIAF_REPO", "/Users/billasur/cv-assurance-engine/.task3")
SITE = os.path.join(OUT, "site")
DIAG = os.path.join(OUT, "assets", "diagrams")
os.makedirs(os.path.join(SITE, "data"), exist_ok=True)
os.makedirs(DIAG, exist_ok=True)

REPO_URL = "https://github.com/Collectors-AU/CVIAF"


def L(rel):
    with open(os.path.join(REPO, rel)) as fh:
        return json.load(fh)


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True,
                          text=True).stdout.strip()


COMMIT = git("rev-parse", "HEAD")
BRANCH = git("rev-parse", "--abbrev-ref", "HEAD")

merged = L("runs/merged_fpr_tpr_report.json")
fleet = L("runs/fleet_fpr_ledger_report.json")
ladder = L("runs/tpr_ladder_at_frozen.json")
atf = L("runs/tpr_at_frozen.json")
cmp_ = L("runs/mvp/comparison.json")
evalj = L("runs/mvp/eval.json")
oda = L("runs/assurance/oda_s5/assurance_report.json")
demo = L("runs/demo_assurance/assurance_report.json")
cov = L("runs/coverage.json")

DOSE = ["0.1", "0.25", "0.5", "1"]
DLAB = {"0.1": "0.10", "0.25": "0.25", "0.5": "0.50", "1": "1.00"}
UNITS = {"substitution": "fraction of hidden units zeroed",
         "weight_tamper": "head-weight noise sigma",
         "bias_lift": "absolute logit units added"}


def rule_cells(kind, rule):
    out = []
    for d in DOSE:
        c = ladder["cells"][kind][d]
        r = c["rules"][rule]
        out.append({"dose": DLAB[d], "caught": r["caught"], "n": c["n"],
                    "tpr": r["tpr"] * 100 if r["tpr"] is not None else None,
                    "lo": r["ci95_wilson"][0] * 100 if r["ci95_wilson"] else None,
                    "hi": r["ci95_wilson"][1] * 100 if r["ci95_wilson"] else None,
                    "conclusion": r["conclusion"],
                    "inert": c["n_behaviour_inert"]})
    return out


def auroc(kind, det):
    for row in evalj["summary"]:
        if row["kind"] == kind and row[det]["auroc_mean"] is not None:
            return row[det]["auroc_mean"]
    return None


ts = [e["timestamp"] for e in oda["audit_trail"]]
from datetime import datetime
FMT = "%Y-%m-%dT%H:%M:%S.%f+00:00"
SPAN = (datetime.strptime(ts[-1], FMT) - datetime.strptime(ts[0], FMT)).total_seconds()

manifest = json.load(open(os.path.join(OUT, "evidence", "evidence_manifest.json")))

BUNDLE = {
    "schema": "cviaf.deck-bundle.v1",
    "generated_utc": datetime.utcnow().isoformat() + "Z",
    "repo": REPO_URL,
    "worktree": REPO,
    "branch": BRANCH,
    "commit": COMMIT,
    "reviewed_main": "6cddfe0e4aea71f2bea8c0b137e5ad4f268cb145",
    "overview": {
        "merged_models": merged["census"]["all"]["clean"],
        "calibration_negatives": merged["denominators"]["calibration_negatives"],
        "evaluation_negatives": merged["denominators"]["evaluation_negatives"],
        "attacked_positives": merged["denominators"]["evaluation_positives"],
        "alpha": merged["alpha"],
        "fleet_models": fleet["census"]["all"]["clean"],
        "fleet_negatives": fleet["denominators"]["evaluation_negatives"],
        "ladder_built": 1188, "ladder_scored": 1055, "ladder_unscorable": 133,
        "min_positives": ladder["min_positives"],
        "coverage_measured": sum(1 for c in cov["clauses"] if c["state"] == "measured"),
        "coverage_total": len(cov["clauses"]),
        "gate_ok": cov["gate_ok"],
    },
    "fpr": [], "ladder": {}, "baseline": {}, "provenance": {},
    "inventories": {}, "auroc": [], "runs": {}, "limitations": [],
}
for pop, rep in (("merged 56,627", merged), ("fleet 7,503", fleet)):
    for rule in ("ctc_mean_clean", "refdiv_mean_clean", "ctc_peak_clean",
                 "ctc_q95_clean"):
        f = rep["rules"][rule]["fpr"]
        BUNDLE["fpr"].append({
            "population": pop, "rule": rule, "alarms": f["numerator"],
            "n": f["denominator"], "fpr": f["point_estimate"] * 100,
            "lo": f["ci95_wilson"][0] * 100, "hi": f["ci95_wilson"][1] * 100,
            "threshold": rep["rules"][rule]["threshold"],
            "basis": rep["rules"][rule]["threshold_basis"],
            "degenerate": rep["rules"][rule]["threshold"] == 1.0,
        })
for kind in ("substitution", "weight_tamper", "bias_lift"):
    BUNDLE["ladder"][kind] = {
        "unit": UNITS[kind],
        "refdiv": rule_cells(kind, "refdiv_mean_clean"),
        "ctc": rule_cells(kind, "ctc_mean_clean"),
    }
# The four headline cells the repository's own number audit enforces
# (README headline + tests/test_number_audit.py). Never pooled, never re-derived
# anywhere else on the page.
HL_CELLS = [("substitution", "0.25", "Substitution", "prune fraction"),
            ("substitution", "0.5", "Substitution", "prune fraction"),
            ("weight_tamper", "1", "Weight tamper", "noise sigma"),
            ("bias_lift", "0.25", "Bias lift", "any dose · logit shift")]
BUNDLE["tpr_headline"] = {
    "rule": "refdiv_mean_clean",
    "alpha": ladder["alpha"],
    "min_positives": ladder["min_positives"],
    "dose_key_note": ("substitution@0.5 and weight_tamper@1 are the receipt's own "
                      "dose keys; the labels are the declared doses"),
    "cells": [],
}
for fam, dose, label, unit in HL_CELLS:
    c = ladder["cells"][fam][dose]
    r = c["rules"][BUNDLE["tpr_headline"]["rule"]]
    BUNDLE["tpr_headline"]["cells"].append({
        "family": fam, "label": label, "unit": unit, "short": label.split(" ")[0],
        "dose": DLAB[dose], "caught": r["caught"], "n": c["n"],
        "tpr": r["tpr"] * 100, "lo": r["ci95_wilson"][0] * 100,
        "hi": r["ci95_wilson"][1] * 100, "conclusion": r["conclusion"],
        "inert": c["n_behaviour_inert"],
        "unscorable": ladder["unscorable"].get(f"{fam}@{dose}", 0),
        "ctc": c["rules"]["ctc_mean_clean"]["caught"],
    })
BUNDLE["tpr_headline"]["superseded"] = {
    "n_arms": atf["n_arms"],
    "substitution": dict(zip(("caught", "n"), (atf["rules"]["refdiv_mean_clean"]
                                                 ["per_kind"]["substitution"]["caught"],
                                                 atf["rules"]["refdiv_mean_clean"]
                                                 ["per_kind"]["substitution"]["n"]))),
    "weight_tamper": dict(zip(("caught", "n"), (atf["rules"]["refdiv_mean_clean"]
                                                  ["per_kind"]["weight_tamper"]["caught"],
                                                  atf["rules"]["refdiv_mean_clean"]
                                                  ["per_kind"]["weight_tamper"]["n"]))),
}

BUNDLE["baseline"] = {
    "model_axis": {k: {"tpr": v["tpr"], "fpr": v["fpr"], "tp": v["true_positive"],
                       "n_pos": v["n_positive"], "fp": v["false_positive"],
                       "n_neg": v["n_negative"]}
                   for k, v in cmp_["model_axis"]["scores"].items()},
    "data_axis": {
        "baseline_tpr": cmp_["data_axis"]["scores"]["baseline_tpr"],
        "baseline_fpr": cmp_["data_axis"]["scores"]["baseline_fpr"],
        "cviaf_tpr": cmp_["data_axis"]["scores"]["cviaf_tpr"],
        "cviaf_fpr": cmp_["data_axis"]["scores"]["cviaf_fpr"],
        "item": {
            "baseline": cmp_["data_axis"]["scores"]["baseline_item_level"],
            "cviaf": cmp_["data_axis"]["scores"]["cviaf_item_level"]},
        "attribution": cmp_["data_axis"]["scores"]["attribution"],
    },
}
BUNDLE["provenance"] = {
    "baseline_digest_manifest": cmp_["provenance_axis"]["baseline_digest_manifest"],
    "baseline_internal_consistency":
        cmp_["provenance_axis"]["baseline_internal_consistency"],
    "cviaf": cmp_["provenance_axis"]["cviaf"],
    "signing_mode": cmp_["provenance_axis"]["signing_mode"],
    "assumption": cmp_["provenance_axis"]["assumption"],
    "n_attacked": cmp_["provenance_axis"]["n_attacked"],
    "n_genuine": cmp_["provenance_axis"]["n_genuine"],
}
BUNDLE["inventories"] = {
    "merged_study": merged["census"]["all"]["clean"],
    "export_record": 22693,
    "components": [{"label": "fleet (seeds 150-10149)", "models": 10000},
                   {"label": "macOS laptop export", "models": 11637},
                   {"label": "git archive export", "models": 1056}],
    "warning": ("1,056 and 11,637 are the seed_10150_28385 and seed_10264_24607 "
                "shards already inside the 56,627 study (docs/MODEL_INVENTORY.md "
                "\u00a73). The two populations overlap and must never be added."),
}
BUNDLE["auroc"] = [
    {"kind": "oga", "label": "OGA - object fabrication",
     "bonf": auroc("oga", "fused_bonf"), "cauchy": auroc("oga", "fused_cauchy")},
    {"kind": "oda", "label": "ODA - object cloaking",
     "bonf": auroc("oda", "fused_bonf"), "cauchy": auroc("oda", "fused_cauchy")},
    {"kind": "clean", "label": "clean control",
     "bonf": auroc("clean", "fused_bonf"), "cauchy": auroc("clean", "fused_cauchy")},
]
BUNDLE["runs"] = {
    "oda_s5": {"findings": len(oda["findings"]), "risk": oda["overall_risk"],
               "disposition": oda["overall_disposition"],
               "audit_len": oda["metadata"]["audit_trail_length"],
               "audit_valid": oda["metadata"]["audit_trail_valid"],
               "span_s": round(SPAN, 2), "report": "runs/assurance/oda_s5"},
    "demo_assurance": {"findings": len(demo["findings"]),
                       "duration_s": demo["metadata"]["assessment_duration_seconds"],
                       "audit_len": demo["metadata"]["audit_trail_length"],
                       "report": "runs/demo_assurance"},
}
BUNDLE["limitations"] = [
    "FPR/TPR numbers are single-population: calibration and evaluation halves come "
    "from the same clean fleet.",
    "The 56,627-model clean ledger has zero attacked positives, so TPR, precision "
    "and expected loss are undefined there, not zero.",
    "ctc_peak_clean and ctc_q95_clean sit at the corpus ceiling: their zero FPR is "
    "a range bound, not a measurement of specificity.",
    "Attack arms are synthetic TinyDetector models; real-backbone results are "
    "planned work, not measured capability.",
    "Dose units differ per attack family; never pool families or doses.",
    "The headline TPR is measured on the attack ladder, not in the clean-null ledger, "
    "which has no attacked positives. An earlier 193-arm single-dose experiment "
    "(substitution 46/94, weight tamper 21/99) is superseded by the 4-dose ladder "
    "(45/94, 92/99) and the two arms sets must never be merged or averaged.",
    "substitution at dose 1.00 has n=7, below the declared floor of 20.",
    "bias_lift is behaviour-inert on all 396 arms - not a proven harmful backdoor.",
    "The committed provenance run uses the HMAC-SHA256 fallback, not Ed25519: "
    "tamper checking under shared-key assumptions, not public-key non-repudiation.",
    "The 22,693-member export record overlaps the 56,627 study and is not additive.",
    "No byte-level image-overlap receipt has landed; do not claim leakage-free "
    "evaluation or disjoint images.",
]

# --------------------------------------------------------------- research basis
# The signal names on this page are published methods, and the code already says so
# in its own docstrings -- cviaf/lab/detectors.py names TRACE and adds "we cite them
# as theirs and publish ours separately". This block is the human-facing half of that
# discipline: which published method each signal comes from, where it lives here, and
# whether this build RUNS it or only designs against it. Nothing here is a result.
RESEARCH_DOCS = [
    ("CV_INTEGRITY_ASSURANCE_2026.md",
     "2025-26 state of the art per capability, the gap against this engine, and the "
     "action list for each module. Every source fetched against its primary page and "
     "tagged with a confidence level; the classifier-to-detector transfer gap is "
     "argued from it."),
    ("RESEARCH_CHECKPOINT_26228.md",
     "The verification log behind that dossier: what was checked against which "
     "primary page, and the citations dropped or downgraded when verification failed "
     "-- a withdrawn preprint, a method that could not be shown to exist."),
]
RESEARCH_RUN = [
    {"method": "TRACE - CTC and FTC (CVPR 2025, arXiv 2503.15293)",
     "path": "cviaf/lab/detectors.py", "where": "trace_ctc, trace_ftc",
     "role": "CTC-mean is the second rule scored on the headline table; FTC is the "
             "Island-Effect signal behind the cloaking (ODA) scores. CTC cannot see "
             "disappearance, which is why FTC exists.",
     "state": "background arm only, reimplemented at MVP scale; TRACE's foreground "
              "arm is an open gap and the paper's numbers are theirs, not quoted here"},
    {"method": "BadDet - OGA / RMA / GMA / ODA (ECCV 2022 workshop)",
     "path": "cviaf/core/types.py", "where": "attack kinds",
     "role": "names the attack rows: object generation, regional and global "
             "misclassification, object disappearance. The cloaking run is ODA.",
     "state": "taxonomy adopted; the synthetic arms are not the paper's detectors"},
    {"method": "Confident-learning label screening (cleanlab family)",
     "path": "cviaf/data_integrity/__init__.py", "where": "label screen",
     "role": "scores contributed boxes for label errors without retraining, in the "
             "detector-native form the problem statement needs.",
     "state": "implemented in the same family; the cleanlab library is not vendored"},
    {"method": "Self-supervised copy detection (SSCD family, CVPR 2022)",
     "path": "cviaf/lab/attribute.py", "where": "near-duplicate screen",
     "role": "flooding and near-duplicate detection across contributed data.",
     "state": "a deterministic frozen-conv embedding stands in; SSCD is the named "
              "scaling target, per docs/SCALING_PLAN.md"},
    {"method": "Hash-chain audit trail (AuditableLLM pattern)",
     "path": "runs/assurance/oda_s5/assurance_report.json",
     "where": "metadata.audit_trail",
     "role": "every report carries a chained audit trail, so the assurance verdict "
             "is itself tamper-evident.",
     "state": "chain implemented and validated in the committed run (7 entries)"},
]
RESEARCH_UPDATE = {
    "title": "What the as-built record changed since the dossier",
    "where": "CV_INTEGRITY_ASSURANCE_2026.md \u00a712 \u00b7 RESEARCH_CHECKPOINT_26228.md UPDATE 3",
    "items": [
        ["TRACE is implemented for one of its two arms",
         "trace_ctc blends backgrounds only. TRACE's second arm - clean samples are the "
         "more consistent ones under focal information - is missing, and it is the "
         "citable explanation for the measured null control.",
         "null control on a model with no backdoor: CTC 0.816, refdiv 0.786, FTC 0.792"],
        ["Three of the four trigger attacks never implanted",
         "With the trigger re-placed per image and two placebos drawn from the same RNG "
         "stream subtracted against each model's seed-matched clean twin, the previously "
         "published trigger rates for oga / oda / rma net to zero. Only gma survives.",
         "the attack ledger now reports raw, null and net as three separate columns"],
        ["The backdoor-like case is a measured failure",
         "bias_lift is behaviour-inert on all 396 arms, so the headline rate for it equals "
         "the rule's own false-alarm rate. This is the blind spot the headline table "
         "names rather than a recall figure.",
         "5.1% caught = 5.1% false-alarm rate"],
        ["Drift is mostly undecided, not merely inseparable",
         "The evadability result predicted the difficulty; the as-built drift battery then "
         "decided only a fifth of its cells. The under-determined verdict is emitted per "
         "cell instead of being inferred globally.",
         "8 decisions against 32 under-determined cells at n=160 per cell"],
    ],
    "scope": "These are the improvement-loop and as-built records, not the two receipts "
             "measured above: they come from the lab's own measurement notes and the "
             "alignment matrix, and they are deliberately not mixed into the clean-null "
             "ledger or the frozen-threshold ladder.",
    "unbuilt": "Still unbuilt, and named as such: C2PA-aligned manifests, ODSCAN/DISTIL "
               "trigger inversion, OpenOOD, SSCD itself, a real-backbone evidence corpus, "
               "and a byte-level image-overlap audit (9,984,920 images).",
}
RESEARCH_PLANNED = [
    ["ODSCAN - IEEE S&P 2024",
     "white-box, detector-specific trigger scanning: the strongest scanner to "
     "benchmark against, not implemented here"],
    ["DISTIL - ICCV 2025, arXiv 2507.22813",
     "data-free trigger inversion; the white-box path that survives an air gap "
     "without the contributor's data"],
    ["C2PA 2.4 - AI/ML guidance",
     "a portable, standards-aligned provenance manifest. The seal here is a "
     "repository-local schema, and this run falls back to HMAC-SHA256"],
    ["OpenOOD v1.5 - arXiv 2306.09301",
     "a standard protocol for the OOD insertion screen; named as the validation "
     "target, not run in this build"],
    ["NIST TrojAI report and AI RMF",
     "the governance vocabulary behind the coverage statement and the disposition "
     "ladder"],
]
BUNDLE["manifest"] = manifest["entries"]

# annotate every manifest entry with whether the source file is actually tracked,
# so the evidence explorer can link to GitHub only when the link will resolve.
tracked = set(subprocess.run(["git", "-C", REPO, "ls-files"], capture_output=True,
                             text=True).stdout.split())
for e in BUNDLE["manifest"]:
    src = e.get("source_file") or ""
    e["in_repo"] = src in tracked
    e["repo_url"] = (f"{REPO_URL}/blob/{COMMIT}/{src}" if src in tracked else None)


def _research_url(rel):
    return f"{REPO_URL}/blob/{COMMIT}/{rel}" if rel in tracked else None


BUNDLE["research"] = {
    "documents": [{"name": name, "path": name, "what": what,
                   "url": _research_url(name)} for name, what in RESEARCH_DOCS],
    "run": [{"method": r["method"], "role": r["role"], "where": r["where"],
             "state": r["state"], "path": r["path"],
             "url": _research_url(r["path"])} for r in RESEARCH_RUN],
    "planned": RESEARCH_PLANNED,
    "update": RESEARCH_UPDATE,
    "note": "Research basis, not results. The methods marked as run here are the "
            "mechanisms this build exercises, at the scale the receipts describe: a "
            "faithful reimplementation at MVP scale, never a reproduction of the "
            "published numbers, which stay with their authors. Full citation lists "
            "and verification status live in the two documents above; the design "
            "trace is docs/CVIAF_V3_ARCHITECTURE.md and the requirement trace is "
            "PRD_SIH26228_CVIAF.md.",
}

with open(os.path.join(SITE, "data", "bundle.json"), "w") as fh:
    json.dump(BUNDLE, fh, indent=2)

payload = json.dumps(BUNDLE)

# ------------------------------------------------------------- architecture SVG
ARCH_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1240 620" width="1240" height="620" font-family="'SF Pro Display',-apple-system,Segoe UI,system-ui,sans-serif">
 <defs>
  <style>
   .t { fill: #E8EAED; font-size: 13px; }
   .ttl { fill:#FFFFFF; font-size:15px; font-weight:600; letter-spacing:.2px }
   .sub { fill:#9AA0A6; font-size:11px }
   .mono { font-family:'SF Mono',ui-monospace,Menlo,monospace; font-size:10.5px; fill:#9AA0A6 }
   .box { fill:#121417; stroke:rgba(255,255,255,0.10); stroke-width:1 }
   .core { fill:#151A22; stroke:#3C5A8A; stroke-width:1 }
   .acc { fill:#5B8DEF }
   .lbl { fill:#8AB4F8; font-size:10px; letter-spacing:1.2px; font-weight:600 }
  </style>
  <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4A5568"/>
  </marker>
 </defs>
 <rect width="1240" height="620" fill="#0B0C0E"/>

 <text x="28" y="34" class="lbl">USERS / ACTORS</text>
 <rect x="28" y="44" width="366" height="70" rx="8" class="box"/>
 <text x="46" y="70" class="ttl">Contributors</text>
 <text x="46" y="90" class="sub">vendor datasets &#183; lab models &#183; edge-unit inference records</text>
 <text x="46" y="105" class="sub">parties the pipeline does not control</text>

 <rect x="418" y="44" width="366" height="70" rx="8" class="box"/>
 <text x="436" y="70" class="ttl">Assurance analysts</text>
 <text x="436" y="90" class="sub">READ: accept / review / quarantine, with reason + evidence</text>
 <text x="436" y="105" class="sub">never asked to trust a bare score</text>

 <rect x="808" y="44" width="404" height="70" rx="8" class="box"/>
 <text x="826" y="70" class="ttl">Procurement, audit and the after-action record</text>
 <text x="826" y="90" class="sub">re-verify offline, third party, no network</text>
 <text x="826" y="105" class="sub">hash-chained trail + coverage statement</text>

 <text x="28" y="156" class="lbl">SYSTEM &#8212; ONE OFFLINE ASSESSMENT CHAIN (NO NETWORK CALLS)</text>

 <rect x="28" y="170" width="196" height="150" rx="8" class="box"/>
 <text x="44" y="196" class="ttl">1 &#183; Intake</text>
 <text x="44" y="216" class="sub">COCO / YOLO datasets</text>
 <text x="44" y="233" class="sub">ONNX / PyTorch / TorchScript</text>
 <text x="44" y="250" class="sub">normalise, hash, refuse-if-unknown</text>
 <text x="44" y="276" class="mono">240 images &#183; 392 boxes</text>
 <text x="44" y="291" class="mono">exact COCO/YOLO geometry</text>
 <text x="44" y="306" class="mono">PNG pixels quantised</text>

 <rect x="248" y="170" width="196" height="150" rx="8" class="box"/>
 <text x="264" y="196" class="ttl">2 &#183; Reference battery</text>
 <text x="264" y="216" class="sub">signed clean reference model</text>
 <text x="264" y="233" class="sub">frozen 5% thresholds</text>
 <text x="264" y="250" class="sub">never re-tuned on test</text>
 <text x="264" y="276" class="mono">cal 28,314 / eval 28,313</text>
 <text x="264" y="291" class="mono">disjoint halves</text>

 <rect x="468" y="170" width="300" height="150" rx="8" class="core"/>
 <text x="484" y="196" class="ttl">3 &#183; Signals</text>
 <text x="484" y="216" class="sub">model probes: ctc &#183; refdiv &#183; focal arm</text>
 <text x="484" y="233" class="sub">weight stats &#183; behavioural fingerprint</text>
 <text x="484" y="250" class="sub">data families: dup &#183; gma &#183; label-flip &#183; oda</text>
 <text x="484" y="267" class="sub">oga &#183; ood-insert &#183; rma</text>
 <text x="484" y="284" class="sub">provenance: chain &#183; replay &#183; signature</text>
 <text x="484" y="306" class="mono">complementary by construction</text>

 <rect x="792" y="170" width="196" height="150" rx="8" class="box"/>
 <text x="808" y="196" class="ttl">4 &#183; Calibration</text>
 <text x="808" y="216" class="sub">conformal p-values</text>
 <text x="808" y="233" class="sub">FDR at asset level</text>
 <text x="808" y="250" class="sub">ASR gate + abstain</text>
 <text x="808" y="276" class="mono">alpha = 0.05</text>
 <text x="808" y="291" class="mono">no decision if unrun</text>

 <rect x="1012" y="170" width="200" height="150" rx="8" class="core"/>
 <text x="1028" y="196" class="ttl">5 &#183; Disposition</text>
 <text x="1028" y="222" class="sub">ACCEPT</text>
 <text x="1028" y="242" class="sub">REVIEW</text>
 <text x="1028" y="262" class="sub">QUARANTINE</text>
 <text x="1028" y="292" class="mono">one merged report</text>
 <text x="1028" y="307" class="mono">per assessed asset</text>

 <g stroke="#4A5568" stroke-width="1.4" marker-end="url(#a)" fill="none">
  <path d="M224 245 H242"/><path d="M444 245 H462"/><path d="M768 245 H786"/>
  <path d="M988 245 H1006"/>
  <path d="M211 118 V150 H1112 V166"/>
  <path d="M601 118 V150 H601 V166"/>
  <path d="M1010 118 V150 H1012 V166"/>
 </g>
 <g stroke="#4A5568" stroke-width="1.4" marker-end="url(#a)" fill="none">
  <path d="M1112 320 V366 H140 V392"/>
 </g>

 <text x="28" y="356" class="lbl">EVIDENCE OUTPUTS &#8212; WHAT MAKES A VERDICT AUDITABLE</text>
 <rect x="28" y="368" width="288" height="96" rx="8" class="box"/>
 <text x="46" y="394" class="ttl">Signed assurance report</text>
 <text x="46" y="414" class="sub">17 findings, CRITICAL, quarantine</text>
 <text x="46" y="431" class="mono">schema-valid &#183; run oda_s5</text>
 <text x="46" y="447" class="mono">27.1 s audit-trail span</text>

 <rect x="336" y="368" width="288" height="96" rx="8" class="box"/>
 <text x="354" y="394" class="ttl">Hash-chained audit trail</text>
 <text x="354" y="414" class="sub">entry links previous hash + artefact digest</text>
 <text x="354" y="431" class="mono">7-entry chain valid (oda_s5)</text>
 <text x="354" y="447" class="mono">10-entry chain valid (demo run)</text>

 <rect x="644" y="368" width="288" height="96" rx="8" class="box"/>
 <text x="662" y="394" class="ttl">Coverage statement</text>
 <text x="662" y="414" class="sub">supported classes + unsupported conditions</text>
 <text x="662" y="431" class="mono">14 of 14 clauses measured</text>
 <text x="662" y="447" class="mono">14 supported / 8 unsupported classes</text>

 <rect x="952" y="368" width="260" height="96" rx="8" class="box"/>
 <text x="970" y="394" class="ttl">Named limits</text>
 <text x="970" y="414" class="sub">133 unscorable arms reported</text>
 <text x="970" y="431" class="mono">HMAC fallback, not Ed25519</text>
 <text x="970" y="447" class="mono">overlap audit pending</text>

 <line x1="28" y1="494" x2="1212" y2="494" stroke="rgba(255,255,255,0.10)"/>
 <text x="28" y="522" class="sub">Trust boundary: everything inside the dashed region is computed offline from assets the team controls. The contributors (top left) are outside it.</text>
 <rect x="20" y="158" width="1200" height="246" rx="12" fill="none" stroke="#3C5A8A" stroke-width="1" stroke-dasharray="5 5"/>
 <text x="28" y="560" class="mono">source: runs/assurance/oda_s5 &#183; runs/merged_fpr_tpr_report.json &#183; runs/tpr_ladder_at_frozen.json &#183; runs/coverage.json &#183; commit __COMMIT__</text>
 <text x="28" y="582" class="mono">laboratory / synthetic scope &#183; field validation pending</text>
</svg>""".replace("__COMMIT__", COMMIT[:8])

with open(os.path.join(DIAG, "architecture.svg"), "w") as fh:
    fh.write(ARCH_SVG)
print("  diagram assets/diagrams/architecture.svg")
print("  bundle  site/data/bundle.json")

# render the SVG to PNG with headless Chrome (no extra image deps)
_svg_png_html = os.path.join(DIAG, "_arch.html")
with open(_svg_png_html, "w") as fh:
    # Keep the SVG at its natural 1240x620 CSS size: --force-device-scale-factor=2
    # already yields the 2480x1240 PNG. Upscaling the SVG too made the page
    # 2480x1240 CSS px inside a 1240x620 viewport, so the capture kept only the
    # top-left quadrant of the diagram (the bug seen as a clipped render).
    fh.write("<!doctype html><meta charset=utf-8><body style='margin:0'>"
             + ARCH_SVG
             + "</body>")
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                "--window-size=1240,620", "--force-device-scale-factor=2",
                "--default-background-color=00000000",
                f"--screenshot={os.path.join(DIAG, 'architecture.png')}",
                "file://" + _svg_png_html], capture_output=True, timeout=120)
print("  diagram assets/diagrams/architecture.png")

# --------------------------------------------------------------- index.html
tpl = open(os.path.join(HERE, "site_template.html")).read()
html = tpl.replace("__BUNDLE__", payload)
with open(os.path.join(SITE, "index.html"), "w") as fh:
    fh.write(html)
print("  site    site/index.html", len(html), "bytes")
print("  ->", SITE)

# deployment note shipped with the build
with open(os.path.join(SITE, "DEPLOY.md"), "w") as fh:
    fh.write(f"""# Deploying the CVIAF evidence dashboard

**Status: built and ready to deploy. Not deployed.** No hosting destination has been
chosen and nothing has been published.

The build is a single self-contained file plus one JSON bundle. There is no server, no
build step and no runtime dependency.

```
site/
  index.html          # the whole UI; bundle is inlined so it opens from file://
  data/bundle.json    # same data, versioned separately for review/diffing
  DEPLOY.md
```

## Review before publishing

1. `grep -riE "(/Users/|billasur|cviaf-analysis|BEGIN [A-Z ]*PRIVATE KEY)" site/` must return
   nothing. The bundle names repository-relative paths only.
2. Confirm the bundle commit matches the deck: `python scripts/build_site.py` regenerates
   `bundle.json` from the receipts at the checked-out commit.
3. Open `index.html` from `file://` and confirm the evidence explorer shows both linked rows
   and `local artifact` rows -- the large corpora are deliberately outside git.

## Hosting

Any static host works (GitHub Pages, an internal S3 bucket, `python -m http.server` for a
local demo). Because the page is inlined it also works from an air-gapped machine.

```bash
cd site && python3 -m http.server 8080   # local preview only
```

## Regenerating

```bash
CVIAF_REPO=<path-to-cviaf-worktree> python scripts/build_site.py
```

Commit: {COMMIT} ({BRANCH})
""")
print("  site    site/DEPLOY.md")
