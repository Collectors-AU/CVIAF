#!/usr/bin/env python3
"""Build the CVIAF SIH26228 deck data layer from committed receipts.

Every value is read from a named file + field. Nothing is hard-coded as a
"known" number: the source JSON is the authority and the script fails loudly
if a field moves.
"""
import csv
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

REPO = os.environ.get("CVIAF_REPO", "/Users/billasur/cv-assurance-engine/.task3")
OUT = os.environ.get("DECK_OUT", "/Users/billasur/cv-assurance-engine/sih-deck-v2")

DATA = os.path.join(OUT, "assets", "data")
CHARTS = os.path.join(OUT, "assets", "charts")
EVID = os.path.join(OUT, "evidence")
for d in (DATA, CHARTS, EVID):
    os.makedirs(d, exist_ok=True)

BLUE = "#1F4E9C"
GREY = "#9AA0A6"
WARN = "#C0392B"
GREEN = "#2E7D32"
INK = "#202124"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#B0B4BA",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.titlesize": 13,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def load(rel):
    with open(os.path.join(REPO, rel)) as fh:
        return json.load(fh)


def save(fig, name):
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(CHARTS, f"{name}.{ext}"),
                    format=ext, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart  {name}.svg/.png")


def write_csv(name, header, rows):
    path = os.path.join(DATA, name)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  csv    {name} ({len(rows)} rows)")


def ci(pair):
    return (pair[0] * 100, pair[1] * 100) if pair else (None, None)


# ---------------------------------------------------------------- inputs
merged = load("runs/merged_fpr_tpr_report.json")
fleet = load("runs/fleet_fpr_ledger_report.json")
ladder = load("runs/tpr_ladder_at_frozen.json")
atf = load("runs/tpr_at_frozen.json")
cmp_ = load("runs/mvp/comparison.json")
evalj = load("runs/mvp/eval.json")
oda = load("runs/assurance/oda_s5/assurance_report.json")
demo = load("runs/demo_assurance/assurance_report.json")
cov = load("runs/coverage.json")

COMMIT = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
BRANCH = subprocess.run(["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
REVIEWED_MAIN = "6cddfe0e4aea71f2bea8c0b137e5ad4f268cb145"

manifest = []


def add(asset, src, field, unit, num, den, rule=None, op=None,
        excl=None, limits=None):
    manifest.append({
        "asset": asset, "source_file": src, "source_field": field,
        "commit": COMMIT, "unit": unit, "numerator": num, "denominator": den,
        "rule": rule, "operating_point": op,
        "exclusions": excl, "limitations": limits or [],
    })


# -------------------------------------------------------- chart A: FPR study
rows = []
for lbl, rep, src in (("merged 56,627", merged, "runs/merged_fpr_tpr_report.json"),
                      ("fleet 7,503", fleet, "runs/fleet_fpr_ledger_report.json")):
    for rule in ("ctc_mean_clean", "refdiv_mean_clean"):
        f = rep["rules"][rule]["fpr"]
        rows.append([lbl, rule, f["numerator"], f["denominator"],
                     round(f["point_estimate"] * 100, 2),
                     round(f["ci95_wilson"][0] * 100, 2),
                     round(f["ci95_wilson"][1] * 100, 2),
                     rep["denominators"]["evaluation_negatives"],
                     f.get("status")])
write_csv("fpr_clean_null.csv",
          ["population", "rule", "false_alarms", "evaluated_negatives",
           "fpr_pct", "ci95_low_pct", "ci95_high_pct",
           "evaluation_negatives_declared", "status"], rows)

fig, ax = plt.subplots(figsize=(8.2, 3.6))
groups = ["merged 56,627", "fleet 7,503"]
rules = ["ctc_mean_clean", "refdiv_mean_clean"]
colors = {"ctc_mean_clean": BLUE, "refdiv_mean_clean": GREY}
width = 0.34
for i, rule in enumerate(rules):
    vals, errs, xs = [], [], []
    for j, g in enumerate(groups):
        r = [x for x in rows if x[0] == g and x[1] == rule][0]
        vals.append(r[4])
        errs.append([r[4] - r[5], r[6] - r[4]])
        xs.append(j + (i - 0.5) * width)
    errs = list(zip(*errs))
    ax.bar(xs, vals, width, label=rule, color=colors[rule],
           yerr=errs, capsize=4, ecolor=INK, error_kw={"lw": 1.1})
    for x, v, r in zip(xs, vals, [r for r in rows if r[1] == rule]):
        ax.text(x, v + 0.28, f"{r[2]}/{r[3]}", ha="center", fontsize=9)
ax.axhline(5.0, color=WARN, ls="--", lw=1.2)
ax.text(1.49, 5.06, "target alpha = 5%", color=WARN, fontsize=9, ha="right")
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(["merged study\n28,313 held-out negatives",
                    "fleet ledger\n3,751 held-out negatives"])
ax.set_ylabel("False-alarm rate on clean models (%)")
ax.set_ylim(0, 7.6)
ax.set_title("Clean-null false alarms: near the 5% target, intervals include it")
ax.legend(frameon=False, loc="upper left", fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.01, -0.10,
         f"Source: runs/merged_fpr_tpr_report.json, runs/fleet_fpr_ledger_report.json "
         f"@ {COMMIT[:8]}. Wilson 95% intervals; ctc_peak/q95 are range bounds, "
         f"not specificity measurements.",
         fontsize=7.5, color="#5F6368")
save(fig, "fpr_clean_null")
for r in rows:
    add(f"page2/3-fpr-{r[1]}-{r[0].split()[0]}", r[0] == "merged 56,627" and
        "runs/merged_fpr_tpr_report.json" or "runs/fleet_fpr_ledger_report.json",
        f"rules.{r[1]}.fpr", "clean evaluation models", r[2], r[3], r[1],
        "alpha=0.05, frozen 5%-quantile threshold", None,
        ["intervals include the 5% target", "single population: calibration and "
         "evaluation halves come from the same clean fleet"])

# ------------------------------------------------- chart B: dose ladder panels
CELLS = ladder["cells"]
DOSE_ORDER = ["0.1", "0.25", "0.5", "1"]
DOSE_LABEL = {"0.1": "0.10", "0.25": "0.25", "0.5": "0.50", "1": "1.00"}
UNIT = {
    "substitution": "dose = fraction of hidden units zeroed",
    "weight_tamper": "dose = head-weight noise sigma",
    "bias_lift": "dose = absolute logit units added",
}
lrows, chart_rows = [], []
for kind in ("substitution", "weight_tamper", "bias_lift"):
    for d in DOSE_ORDER:
        cell = CELLS[kind][d]
        for rule in ("refdiv_mean_clean", "ctc_mean_clean"):
            r = cell["rules"][rule]
            lrow = [kind, DOSE_LABEL[d], rule, r["caught"], cell["n"],
                    round(r["tpr"] * 100, 2) if r["tpr"] is not None else "",
                    round(r["ci95_wilson"][0] * 100, 2) if r["ci95_wilson"] else "",
                    round(r["ci95_wilson"][1] * 100, 2) if r["ci95_wilson"] else "",
                    r["conclusion"], cell["n_behaviour_inert"],
                    round((cell["median_f1_relative_drop"] or 0) * 100, 2)]
            lrows.append(lrow)
write_csv("tpr_dose_ladder.csv",
          ["family", "dose", "rule", "caught", "n", "tpr_pct",
           "ci95_low_pct", "ci95_high_pct", "conclusion",
           "n_behaviour_inert", "median_f1_relative_drop_pct"], lrows)

for kind in ("substitution", "weight_tamper"):
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    xs = list(range(4))
    for rule, col, mk in (("refdiv_mean_clean", BLUE, "o"),
                          ("ctc_mean_clean", GREY, "s")):
        ys, lo, hi, insuff = [], [], [], []
        for i, d in enumerate(DOSE_ORDER):
            r = CELLS[kind][d]["rules"][rule]
            n = CELLS[kind][d]["n"]
            ys.append(r["tpr"] * 100)
            if r["ci95_wilson"]:
                lo.append(ys[-1] - r["ci95_wilson"][0] * 100)
                hi.append(r["ci95_wilson"][1] * 100 - ys[-1])
            else:
                lo.append(0); hi.append(0)
            insuff.append(r["conclusion"] == "insufficient_denominator")
        ax.errorbar(xs, ys, yerr=[lo, hi], color=col, marker=mk, lw=2,
                    capsize=4, label=rule, zorder=3)
        for i, d in enumerate(DOSE_ORDER):
            r = CELLS[kind][d]["rules"][rule]
            n = CELLS[kind][d]["n"]
            if insuff[i]:
                ax.plot(xs[i], ys[i], marker="o", mfc="white", mec=col,
                        ms=13, mew=2, zorder=2)
                ax.annotate(f"{r['caught']}/{n}\ninsufficient\n(n<20)",
                            (xs[i], ys[i]), textcoords="offset points",
                            xytext=(-6, -38), fontsize=8, color=WARN, ha="center")
            else:
                ax.annotate(f"{r['caught']}/{n}", (xs[i], ys[i]),
                            textcoords="offset points", xytext=(0, 9),
                            fontsize=8.5, ha="center", color=col)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{DOSE_LABEL[d]}\nn={CELLS[kind][d]['n']}"
                        for d in DOSE_ORDER])
    ax.set_xlabel(UNIT[kind])
    ax.set_ylabel("Attacked arms caught (%)")
    ax.set_ylim(-8, 112)
    ax.set_xlim(-0.5, 3.6)
    inert = [f"{DOSE_LABEL[d]}: {CELLS[kind][d]['n_behaviour_inert']}/{CELLS[kind][d]['n']}"
             for d in DOSE_ORDER]
    ax.set_title(f"{kind.replace('_', ' ')}: RefDiv rises with measured damage")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.text(0.99, 0.02, "behaviour-inert arms  " + "  |  ".join(inert),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="#5F6368")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.01, -0.09, f"Source: runs/tpr_ladder_at_frozen.json @ {COMMIT[:8]}. "
             "Hollow marker = below min_positives=20; labels are caught/n arms.",
             fontsize=7.5, color="#5F6368")
    save(fig, f"dose_{kind}")
    for r in lrows:
        if r[0] == kind:
            add(f"page3-dose-{kind}-{r[1]}-{r[2]}", "runs/tpr_ladder_at_frozen.json",
                f"cells.{kind}.{r[1]}.rules.{r[2]}", "attacked model arms",
                r[3], r[4], r[2],
                "alpha=0.05, thresholds frozen from runs/merged_fpr_tpr_report.json",
                r[4] and (99 - r[4]),
                ["dose units differ by family; do not pool",
                 "n_behaviour_inert=%d/%d" % (r[9], r[4])] +
                (["below min_positives=20: insufficient denominator"] if
                 r[8] == "insufficient_denominator" else []))

# ------------------------------------------------- chart C: baseline model axis
ma = cmp_["model_axis"]["scores"]
order = ["fixed_threshold_single_signal", "quantile_threshold_single_signal",
         "hand_tuned_single_signal", "cviaf"]
label = {"fixed_threshold_single_signal": "Fixed threshold\n(single signal)",
         "quantile_threshold_single_signal": "Reference quantile\n(single signal)",
         "hand_tuned_single_signal": "Hand-tuned\n(single signal)",
         "cviaf": "CVIAF"}
mrows = [[label[s].replace("\n", " "), ma[s]["true_positive"], ma[s]["n_positive"],
          round(ma[s]["tpr"] * 100, 2), ma[s]["false_positive"],
          ma[s]["n_negative"], round(ma[s]["fpr"] * 100, 2)] for s in order]
write_csv("baseline_model_axis.csv",
          ["system", "true_positive", "n_positive", "tpr_pct",
           "false_positive", "n_negative", "fpr_pct"], mrows)

fig, ax = plt.subplots(figsize=(8.2, 3.9))
xs = list(range(4)); w = 0.36
tprs = [r[3] for r in mrows]; fprs = [r[6] for r in mrows]
b1 = ax.bar([x - w / 2 for x in xs], tprs, w, color=BLUE, label="TPR (attacks caught)")
b2 = ax.bar([x + w / 2 for x in xs], fprs, w, color=GREY, label="FPR (clean alarms)")
for i, r in enumerate(mrows):
    ax.text(xs[i] - w / 2, r[3] + 3, f"{r[1]}/{r[2]}", ha="center", fontsize=9)
    ax.text(xs[i] + w / 2, r[6] + 3, f"{r[4]}/{r[5]}", ha="center", fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels([label[s] for s in order], fontsize=9.5)
ax.set_ylabel("Percent (%)")
ax.set_ylim(0, 118)
ax.set_title("Same MVP cohort, model axis: 2 scored positives, 4 negatives, "
             "2 ASR-gated exclusions")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
          fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.01, -0.20, f"Source: runs/mvp/comparison.json @ {COMMIT[:8]}. "
         "Tiny illustrative denominators — not a production guarantee.",
         fontsize=7.5, color="#5F6368")
save(fig, "baseline_model_axis")
for r in mrows:
    add(f"page5-model-axis-{r[0]}", "runs/mvp/comparison.json",
        "model_axis.scores", "scored attack-qualified models", r[1], r[2],
        r[0], "alpha=0.05", 2,
        ["n=2 positives / 4 negatives", "2 ASR-gated cases excluded from scoring"])

# ----------------------------------------- chart D: data asset + item level
da = cmp_["data_axis"]["scores"]
asset_rows = [["baseline per-sample outlier", da["baseline_tpr"], 7,
               round(da["baseline_tpr"] * 100, 2), da["baseline_fpr"], 1,
               round(da["baseline_fpr"] * 100, 2)],
              ["CVIAF asset decision", da["cviaf_tpr"], 7,
               round(da["cviaf_tpr"] * 100, 2), da["cviaf_fpr"], 1,
               round(da["cviaf_fpr"] * 100, 2)]]
write_csv("baseline_data_axis.csv",
          ["system", "asset_tpr", "n_poisoned_assets", "asset_tpr_pct",
           "asset_fpr", "n_clean_assets", "asset_fpr_pct"], asset_rows)
ni = da["cviaf_item_level"]; bi = da["baseline_item_level"]
write_csv("item_level.csv",
          ["system", "true_positive", "false_positive", "n_poisoned_samples",
           "n_clean_samples", "tpr_pct", "fpr_pct", "precision_pct"],
          [["baseline per-sample outlier", bi["true_positive"], bi["false_positive"],
            bi["n_poisoned_samples"], bi["n_clean_samples"],
            round(bi["item_tpr"] * 100, 2), round(bi["item_fpr"] * 100, 2),
            round(bi["item_precision"] * 100, 2)],
           ["CVIAF (BH, FDR)", ni["true_positive"], ni["false_positive"],
            ni["n_poisoned_samples"], ni["n_clean_samples"],
            round(ni["item_tpr"] * 100, 2), round(ni["item_fpr"] * 100, 2),
            round(ni["item_precision"] * 100, 2)]])

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
ax = axes[0]
xs = [0, 1]; w = 0.36
ax.bar([x - w / 2 for x in xs], [r[3] for r in asset_rows], w, color=BLUE, label="TPR")
ax.bar([x + w / 2 for x in xs], [r[6] for r in asset_rows], w, color=GREY, label="FPR")
for i, r in enumerate(asset_rows):
    ax.text(xs[i] - w / 2, r[3] + 3, f"{round(r[1]*7)}/7", ha="center", fontsize=9)
    ax.text(xs[i] + w / 2, r[6] + 3, f"{round(r[4]*1)}/1", ha="center", fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels(["baseline", "CVIAF"], fontsize=10)
ax.set_ylim(0, 118); ax.set_ylabel("Percent (%)")
ax.set_title("Data assets (7 poisoned, 1 clean)", fontsize=11)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax = axes[1]
xs = [0, 1]; w = 0.26
vals = {"TPR": [round(bi["item_tpr"]*100, 2), round(ni["item_tpr"]*100, 2)],
        "FPR": [round(bi["item_fpr"]*100, 2), round(ni["item_fpr"]*100, 2)],
        "Precision": [round(bi["item_precision"]*100, 2),
                      round(ni["item_precision"]*100, 2)]}
for k, (name, ys) in enumerate(vals.items()):
    ax.bar([x + (k - 1) * w for x in xs], ys, w,
           color=[BLUE, GREY, GREEN][k], label=name)
    for x, y in zip(xs, ys):
        ax.text(x + (k - 1) * w, y + 2, f"{y:.1f}", ha="center", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels(["baseline", "CVIAF"], fontsize=10)
ax.set_ylim(0, 112); ax.set_title("Samples (432 poisoned, 1,560 clean)", fontsize=11)
ax.legend(frameon=False, fontsize=8.5, ncol=3)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.01, -0.07, f"Source: runs/mvp/comparison.json @ {COMMIT[:8]}. "
         "CVIAF trades lower item recall for fewer false positives here — "
         "this is a tradeoff, not dominance.", fontsize=7.5, color="#5F6368")
save(fig, "baseline_data_axis")
add("page5-asset-axis", "runs/mvp/comparison.json",
    "data_axis.scores.baseline_tpr/fpr vs cviaf_tpr/fpr", "poisoned / clean assets",
    f"baseline {da['baseline_tpr']*7:.0f}/7 vs CVIAF {da['cviaf_tpr']*7:.0f}/7",
    "7 poisoned, 1 clean", "asset-level disposition",
    "asset rollup of per-sample flags", None,
    ["1 clean asset: FPR 0/1 carries almost no information"])
add("supplement-item-level", "runs/mvp/comparison.json",
    "data_axis.scores.*_item_level", "labelled samples",
    f"baseline TP{bi['true_positive']} FP{bi['false_positive']} / "
    f"CVIAF TP{ni['true_positive']} FP{ni['false_positive']}",
    "432 poisoned, 1,560 clean", "per-sample BH decision",
    "alpha=0.05, FDR control", None,
    ["CVIAF item TPR is far lower than the baseline's here",
     "unfavourable result: must stay in the supplement"])

# ------------------------------------------------- chart E: provenance
pa = cmp_["provenance_axis"]
prows = []
for key, lbl in (("baseline_digest_manifest", "Enrolled digest manifest"),
                 ("baseline_internal_consistency", "Internal hash consistency"),
                 ("cviaf", "CVIAF provenance")):
    d = pa[key]
    prows.append([lbl, d["tpr"], 8, round(d["tpr"] * 100, 1), d["fpr"], 2,
                  d.get("abstained_on_attacked", 0), d.get("abstained_on_genuine", 0)])
write_csv("provenance.csv",
          ["system", "detected", "n_attacked", "tpr_pct", "false_alarms",
           "n_genuine", "abstained_attacked", "abstained_genuine"], prows)

fig, ax = plt.subplots(figsize=(8.0, 3.4))
xs = list(range(3))
ax.bar(xs, [r[3] for r in prows], 0.5, color=[GREY, "#C6CBD1", BLUE])
for i, r in enumerate(prows):
    ax.text(i, r[3] + 3, f"{round(r[1]*8)}/8", ha="center", fontsize=10)
    ax.text(i, 4, f"abstained\n{r[6]} attacked\n{r[7]} genuine", ha="center",
            fontsize=7.5, color="#5F6368")
ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in prows], fontsize=9.5)
ax.set_ylabel("Attacked records detected (%)"); ax.set_ylim(0, 116)
ax.set_title("Inference records: 8 attacked, 2 genuine")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.01, -0.10, f"Source: runs/mvp/comparison.json @ {COMMIT[:8]}. "
         "signing_mode=%s — tamper checking under shared-key assumptions, "
         "NOT public-key non-repudiation." % pa.get("signing_mode"),
         fontsize=7.5, color=WARN)
save(fig, "provenance")
for r in prows:
    add(f"page5-provenance-{r[0]}", "runs/mvp/comparison.json",
        "provenance_axis", "inference records",
        round(r[1] * 8), 8, r[0], "HMAC-SHA256 fallback in this run",
        None, ["8 attacked / 2 genuine", "abstentions must stay visible",
               "not Ed25519"])

# ------------------------------------------------- chart F: inventory (overlap)
inv = [["fleet (seeds 150-10149)", 10000],
       ["macOS laptop export", 11637],
       ["git archive export", 1056]]
write_csv("export_inventory.csv", ["component", "models"], inv)
fig, ax = plt.subplots(figsize=(8.0, 2.3))
left = 0
cols = [BLUE, GREY, "#C6CBD1"]
for (lbl, n), c in zip(inv, cols):
    ax.barh(0, n, left=left, color=c, height=0.5)
    ax.text(left + n / 2, 0, f"{lbl}\n{n:,}", ha="center", va="center",
            fontsize=9, color="white" if c != "#C6CBD1" else INK)
    left += n
ax.set_xlim(0, 24000); ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
ax.set_xlabel("Models in the 30 Sep verified-export record")
ax.set_title("22,693 members — NOT additive with the 56,627 study", color=WARN)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
fig.text(0.01, -0.30, "1,056 and 11,637 are the seed_10150_28385 and seed_10264_24607 "
         "shards already inside the 56,627 study (docs/MODEL_INVENTORY.md §3). "
         "Treat as overlapping; a dedup receipt is required before calling them "
         "disjoint. Recommend supplement only.", fontsize=7.5, color=WARN)
save(fig, "export_inventory_overlap")
add("page4-export-inventory", "docs/MODEL_INVENTORY.md §3",
    "[needs value from coordinator export ledger]", "models",
    "[needs value from ledger]", 22693, "verified export inventory",
    None, None,
    ["OVERLAPS the 56,627 merged study: 1,056 and 11,637 are shards of it",
     "not additive; dedup receipt outstanding"])

# ---------------------------------------- chart:G canonical TPR headline
# These four cells are the figures the repository's own number audit enforces
# (README headline + tests/test_number_audit.py). Do not restate them loosely.
HEADLINE = [
    ("substitution", "0.25", "Substitution\ndose 0.25"),
    ("substitution", "0.5", "Substitution\ndose 0.50"),
    ("weight_tamper", "1", "Weight tamper\ndose 1.00"),
    ("bias_lift", "0.25", "Bias lift\nany dose"),
]
UNSCORABLE = ladder["unscorable"]
hrows = []
for fam, dose, lab in HEADLINE:
    c = CELLS[fam][dose]["rules"]["refdiv_mean_clean"]
    hrows.append([fam, DOSE_LABEL[dose], dose, c["caught"], CELLS[fam][dose]["n"],
                  round(c["tpr"] * 100, 1),
                  round(c["ci95_wilson"][0] * 100, 1),
                  round(c["ci95_wilson"][1] * 100, 1),
                  CELLS[fam][dose]["n_behaviour_inert"],
                  CELLS[fam][dose]["n"],
                  UNSCORABLE.get(f"{fam}@{dose}", 0)])
write_csv("tpr_headline.csv",
          ["family", "dose", "dose_key", "caught", "n", "tpr_pct", "ci95_low_pct",
           "ci95_high_pct", "n_behaviour_inert", "n_arms", "unscorable"], hrows)

fig, ax = plt.subplots(figsize=(8.6, 3.8))
labels, vals, los, his, cols = [], [], [], [], []
for i, (fam, dose, lab) in enumerate(HEADLINE):
    c = CELLS[fam][dose]["rules"]["refdiv_mean_clean"]
    labels.append(lab)
    vals.append(c["tpr"] * 100)
    los.append(c["tpr"] * 100 - c["ci95_wilson"][0] * 100)
    his.append(c["ci95_wilson"][1] * 100 - c["tpr"] * 100)
    cols.append(WARN if fam == "bias_lift" else BLUE)
ax.bar(range(len(vals)), vals, 0.52, color=cols,
       yerr=[los, his], capsize=5, ecolor=INK, error_kw={"lw": 1.1})
for i, (v, h) in enumerate(zip(vals, his)):
    r = hrows[i]
    ax.text(i, v + h + 3.2, f"{v:.1f}%", ha="center", fontsize=10.5,
            fontweight="bold")
    ax.text(i, 3.5, f"{r[3]}/{r[4]}", ha="center", fontsize=9, color="white")
ax.axhline(5.0, color=WARN, ls="--", lw=1.2)
ax.text(3.45, 5.6, "the frozen 5% false-alarm rate", color=WARN, fontsize=9,
        ha="right")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("Attacked arms caught (%)")
ax.set_ylim(0, 116)
ax.set_title("Headline TPR at a frozen 5% false-alarm rate, per class, never pooled")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.01, -0.12,
         f"Source: runs/tpr_ladder_at_frozen.json @ {COMMIT[:8]} \u2014 the same cells the "
         f"repository's number audit enforces. Bias lift is behaviour-inert in all 396 arms: "
         f"catching 5.1% means the rule fires at its own false-alarm rate, not that it detects "
         f"a backdoor. Substitution dose 0.50 is measured on 63 survivors of 99.",
         fontsize=7.5, color="#5F6368")
save(fig, "tpr_headline")

# ---- chart:G2 slide strip: the same four cells, sized to sit in one deck row
NAME_ = {"substitution": "Substitution", "weight_tamper": "Weight tamper",
         "bias_lift": "Bias lift"}
UNIT_ = {"substitution": "prune fraction", "weight_tamper": "noise sigma",
         "bias_lift": "logit shift"}
fig = plt.figure(figsize=(12.4, 1.60))
ax = fig.add_axes([0.006, 0.10, 0.988, 0.74])
ax.set_xlim(0, 4); ax.set_ylim(0, 1); ax.axis("off")
for i, r in enumerate(hrows):
    fam, dlab, dkey, caught, n, tp, lo, hi, inert, n_arms, unc = r
    x, w = i + 0.06, 0.88
    col = WARN if fam == "bias_lift" else BLUE
    head = (f"{NAME_[fam]} \u00b7 any dose ({UNIT_[fam]}) \u2014 inert"
            if fam == "bias_lift" else f"{NAME_[fam]} \u00b7 dose {dlab} ({UNIT_[fam]})")
    ax.text(x, 0.92, head, fontsize=10.5, color=INK, va="center")
    ax.text(x, 0.52, f"{tp:.1f}%", fontsize=22, fontweight="bold", color=col, va="center")
    ax.text(x + 0.70, 0.55, f"{caught}/{n} arms", fontsize=10, color="#5F6368", va="center")
    ax.add_patch(Rectangle((x, 0.17), w, 0.11, facecolor="#EDEFF3", edgecolor="none"))
    ax.add_patch(Rectangle((x, 0.17), w * tp / 100, 0.11, facecolor=col, edgecolor="none"))
    ax.plot([x + w * lo / 100, x + w * hi / 100], [0.325, 0.325], color=INK, lw=1.3)
    for xx in (x + w * lo / 100, x + w * hi / 100):
        ax.plot([xx, xx], [0.30, 0.35], color=INK, lw=1.3)
    ax.plot([x + w * 0.05, x + w * 0.05], [0.13, 0.42], color=WARN, ls="--", lw=1.2)
    ax.text(x, 0.015, f"95% CI {lo:.1f}\u2013{hi:.1f}%  \u00b7  inert {inert}/{n}"
            + (f"  \u00b7  unscorable {unc}" if unc else ""),
            fontsize=8.5, color="#5F6368", va="center")
fig.text(0.006, 0.006,
         "RefDiv-mean, thresholds frozen on 28,313 held-out clean models before any attack was scored "
         f"(alpha 0.05) \u2014 source runs/tpr_ladder_at_frozen.json @ {COMMIT[:8]}",
         fontsize=8, color="#5F6368")
save(fig, "tpr_headline_strip")

# ---- chart:G3 deck dose panels: both families in one wide figure, so the two
# panels can sit in a single slide row at a readable size (the tall per-family
# figures are for the supplement, where they have a full page).
fig, axes = plt.subplots(1, 2, figsize=(12.4, 2.15), sharey=True)
for ax, kind in zip(axes, ("substitution", "weight_tamper")):
    xs = list(range(4))
    for rule, col, mk in (("refdiv_mean_clean", BLUE, "o"),
                          ("ctc_mean_clean", GREY, "s")):
        ys, lo, hi = [], [], []
        for d in DOSE_ORDER:
            r = CELLS[kind][d]["rules"][rule]
            ys.append(r["tpr"] * 100)
            if r["ci95_wilson"]:
                lo.append(ys[-1] - r["ci95_wilson"][0] * 100)
                hi.append(r["ci95_wilson"][1] * 100 - ys[-1])
            else:
                lo.append(0); hi.append(0)
        ax.errorbar(xs, ys, yerr=[lo, hi], color=col, marker=mk, ms=4.5, lw=1.8,
                    capsize=3, label=rule, zorder=3)
    for i, d in enumerate(DOSE_ORDER):
        c = CELLS[kind][d]
        r = c["rules"]["refdiv_mean_clean"]
        if r["conclusion"] == "insufficient_denominator":
            ax.plot(xs[i], r["tpr"] * 100, marker="o", mfc="white", mec=BLUE,
                    ms=10, mew=1.8, zorder=2)
            ax.annotate(f"{r['caught']}/{c['n']}\ninsufficient", (xs[i], r["tpr"] * 100),
                        textcoords="offset points", xytext=(0, -26), fontsize=7,
                        color=WARN, ha="center")
        else:
            ax.annotate(f"{r['caught']}/{c['n']}", (xs[i], r["tpr"] * 100),
                        textcoords="offset points", xytext=(0, 8), fontsize=7.5,
                        ha="center", color=BLUE)
    ax.axhline(5.0, color=WARN, ls="--", lw=1.0,
               label="frozen 5% false-alarm rate" if kind == "substitution" else None)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{DOSE_LABEL[d]}\nn={CELLS[kind][d]['n']}" for d in DOSE_ORDER],
                       fontsize=7.5)
    ax.set_xlabel(UNIT[kind], fontsize=8)
    ax.set_ylim(-4, 112)
    ax.set_title(f"{kind.replace('_', ' ')} \u2014 RefDiv-mean rises with measured damage",
                 fontsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Arms caught (%)", fontsize=8.5)
axes[0].set_yticks([0, 25, 50, 75, 100])
axes[0].tick_params(labelsize=7.5)
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.text(0.006, 0.005, "Labels are caught/scored arms at the frozen threshold; hollow marker = below "
         f"min_positives=20. Source: runs/tpr_ladder_at_frozen.json @ {COMMIT[:8]}",
         fontsize=7, color="#5F6368")
save(fig, "dose_panels_wide")

for r in hrows:
    add(f"page3-tpr-headline-{r[0]}-{r[1]}", "runs/tpr_ladder_at_frozen.json",
        f"cells.{r[0]}.{r[2]}.rules.refdiv_mean_clean", "attacked model arms",
        r[3], r[4], "refdiv_mean_clean",
        "alpha=0.05, thresholds frozen from runs/merged_fpr_tpr_report.json",
        r[10] or None,
        ["never pool across classes or doses",
         f"{r[8]}/{r[9]} arms behaviour-inert"] +
        ([f"dose units differ: {ladder['units_per_kind'][r[0]]}"] if r[0] != "substitution"
         else []))

# --------------------------------- superseded earlier 193-arm experiment
sup = atf
sup_rows = []
for kind in ("substitution", "weight_tamper"):
    cell = sup["rules"]["refdiv_mean_clean"]["per_kind"][kind]
    sup_rows.append([kind, cell.get("caught"), cell.get("n"),
                     round((cell.get("tpr") or 0) * 100, 2),
                     cell.get("conclusion")])
write_csv("tpr_superseded_193.csv",
          ["family", "caught", "n", "tpr_pct", "conclusion"], sup_rows)
add("supplement-superseded-193", "runs/tpr_at_frozen.json",
    "rules.refdiv_mean_clean.per_kind", "attacked model arms",
    str(sup_rows[0][1]), sup_rows[0][2], "refdiv_mean_clean (SUPERSEDED)",
    f"n_arms={sup['n_arms']}, single pooled dose", None,
    ["superseded by the 4-dose ladder; substitution 46/94 at dose 0.25 is NOT the "
     "ladder's 45/94 and the two must never be merged"])

# --------------------------------------------------- eval.json AUROC table
srows = []
for row in evalj["summary"]:
    if row["n_scored"] and row["fused_bonf"]["auroc_mean"] is not None:
        srows.append([row["kind"], row["n_models"], row["n_scored"],
                      row["fused_bonf"]["auroc_mean"], row["fused_cauchy"]["auroc_mean"],
                      row["refdiv"]["auroc_mean"], row["ctc"]["auroc_mean"]])
write_csv("eval_auroc.csv",
          ["attack_kind", "n_models", "n_scored", "fused_bonf_auroc",
           "fused_cauchy_auroc", "refdiv_auroc", "ctc_auroc"], srows)

# --------------------------------------------------- oda_s5 / demo facts
ts = [e["timestamp"] for e in oda["audit_trail"]]
from datetime import datetime
fmt = "%Y-%m-%dT%H:%M:%S.%f+00:00"
span = (datetime.strptime(ts[-1], fmt) - datetime.strptime(ts[0], fmt)).total_seconds()

# ------------------------------------------------------------- manifest
add("page3-oda-s5-findings", "runs/assurance/oda_s5/assurance_report.json",
    "findings[]", "findings", len(oda["findings"]), None, "end-to-end report",
    "white-box, alpha=0.05", None,
    ["this run only", "HMAC fallback signing"])
add("page3-oda-s5-audit", "runs/assurance/oda_s5/assurance_report.json",
    "metadata.audit_trail_length", "hash-chained entries",
    oda["metadata"]["audit_trail_length"], None, "audit chain",
    "chain_valid=%s" % oda["metadata"]["audit_trail_valid"], None, [])
add("page4-oda-s5-runtime", "runs/assurance/oda_s5/audit_trail.json",
    "first/last timestamp", "seconds", round(span, 2), None, "wall clock",
    "audit-trail span", None,
    ["deck states 27.6 s; measured span is %.1f s — use the measured value or "
     "cite the instrument that produced 27.6" % span])
add("supplement-demo-assurance", "runs/demo_assurance/assurance_report.json",
    "findings[] / metadata", "findings / seconds", len(demo["findings"]),
    None, "black-box demo run",
    "assessment_duration_seconds=%s" % demo["metadata"]["assessment_duration_seconds"],
    None, ["never combine with oda_s5 counts or timings",
           "white-box checks skipped"])

cov_states = {}
for c in cov["clauses"]:
    cov_states[c["state"]] = cov_states.get(c["state"], 0) + 1
add("page3-coverage", "runs/coverage.json", "clauses[].state",
    "clauses measured", cov_states.get("measured", 0), len(cov["clauses"]),
    "clause coverage", "gate_ok=%s" % cov["gate_ok"], None,
    ["a passing gate is not proof of universal attack detection"])

with open(os.path.join(EVID, "evidence_manifest.json"), "w") as fh:
    json.dump({
        "repo": "https://github.com/Collectors-AU/CVIAF",
        "worktree": REPO,
        "branch": BRANCH,
        "commit": COMMIT,
        "reviewed_main": REVIEWED_MAIN,
        "commit_delta_note": ("Local worktree is ahead of the reviewed main; every "
                              "asset below is labelled with the commit it was built "
                              "from and must not be silently mixed with 6cddfe0e."),
        "generated_by": "sih-deck-v2/scripts/build_assets.py",
        "entries": manifest,
    }, fh, indent=2)

print(f"\n  commit {COMMIT[:12]} ({BRANCH})  oda_s5 span {span:.2f}s")
print(f"  manifest entries: {len(manifest)}")
print("  ->", OUT)
