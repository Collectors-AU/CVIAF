# Demo video — narration and shot list

Target 3:30 (a proposed starting point, not a competition limit). Read the narration lines
as written; they are timed to the shots. Every command below was checked against the CLI
before being put in this document.

Two standing rules for the recording:

- Anything on screen is **recorded evidence from a committed run**, not live processing of
  56,627 models. Say so once, early, and it stops being a question.
- Never say Army, deployment, savings, zero-overlap or Ed25519. Those are not measured here.

---

## Shot list

| # | Time | Screen | Action to perform | Narration |
|---|---|---|---|---|
| 1 | 0:00–0:15 | Deck page 1 | Hold on the title page | "Every check in a computer-vision pipeline today covers one link. Ours covers the chain — and it runs with the network off." |
| 2 | 0:15–0:38 | Dashboard hero → KPI row | Scroll slowly from hero to the six KPI tiles | "This is our working measurement dashboard. These are not claims, they are receipts: fifty-six thousand six hundred and twenty-seven clean models scored for false alarms, on a threshold chosen from a disjoint calibration half. And the first three tiles are what that buys us — so let me show you the headline." |
| 3 | 0:38–1:05 | **Detection headline** section | Point at the four bars, then the table under them | "At a false-alarm budget frozen at five percent *before* we scored a single attack, one rule catches forty-seven point nine percent of substitution attacks at moderate strength, and ninety-three point seven percent when the attack is heavy. Ninety-two point nine percent of weight tampering at full strength. And bias lift: five point one percent — which is exactly its own false-alarm rate. That one is a declared blind spot, not a detection." |
| 4 | 1:05–1:28 | Clean-null FPR section | Hover the table; scroll to the two retired rules | "Each rule sits near that five-percent target, and the confidence intervals include it — so we say 'near target', not 'below target'. Two of the four rules are retired: their threshold sits at the ceiling of the clean corpus, so a zero false-alarm rate there is a range bound, not evidence of a better detector." |
| 5 | 1:28–1:50 | Dose ladder section | Click **Substitution**, then **Weight tamper** | "The ladder behind the headline. Detection rises with damage — inside one family at a time, because the dose units differ. Substitution goes 13 of 99 to 45 of 94 to 59 of 63. The last cell is seven arms, below our declared floor of twenty, so it is drawn hollow and never quoted as a hundred percent." |
| 6 | 1:50–2:10 | Dose ladder → **Bias lift** tab, then back | Click **Bias lift**, point at the CTC line | "And here is the trap. A second rule — CTC mean — passes the same five-percent false-alarm test, and at substitution dose 0.50 it catches two of sixty-three where RefDiv catches fifty-nine. Two rules that look identical on clean data, one nearly blind. All three hundred and ninety-six bias-lift arms show zero behaviour change; we report it as a blind spot." |
| 7 | 2:10–2:38 | Baseline vs CVIAF | Scroll the grouped bars, then the table | "Against baselines we wrote ourselves on the same cohort: on the model axis we catch one of two with no false alarms, where a fixed threshold catches both and alarms on everything. But at sample level the simple baseline beats us — eighty-four percent recall against our seventeen. We show that tradeoff rather than hiding it." |
| 8 | 2:38–2:58 | Provenance section | Scroll to the bars | "On inference records, an enrolled-digest check catches four of eight and abstains on four. Our chain catches all eight with no false alarms on the genuine pair. This run is signed with HMAC, not Ed25519 — a tamper check under a shared key, not public-key non-repudiation." |
| 9 | 2:58–3:16 | Terminal | Run the verify command (below), let the output print | "Every conclusion ships with a report that re-verifies offline against the schema, a hash-chained audit trail, and a coverage statement that names what we do not cover." |
| 10 | 3:16–3:32 | Evidence explorer | Type "tpr" in the filter, click one **open ↗** link | "And every number traces back to a file, a field and a denominator at a named commit — including the headline cells. The large corpora live outside the repository, so those rows are labelled as local artefacts instead of pretending to be links." |

## Recording commands (run from the CVIAF worktree)

```bash
# 1. regenerates the site bundle and the evidence dashboard, offline
cd sih-deck-v2 && ../.venv/bin/python scripts/build_site.py

# 2. the verification shown in shot 8 (real command, real output)
cd /path/to/cviaf-worktree
PYTHONPATH=. python -m cviaf.lab verify-report runs/assurance/oda_s5/assurance_report.json
#   -> valid against https://cviaf.local/schemas/assurance-report-3.0.0.json

# 3. coverage statement (optional, if shot 8 overruns)
PYTHONPATH=. python -m cviaf.lab coverage
```

Do **not** retrain or regenerate the model population while recording. Use the committed
receipts and, if you want a live moment, the verification command above — it takes under a
second and it is genuinely executed.

## Preparation checklist

- [ ] Open `sih-deck-v2/site/index.html` from file:// once; confirm the KPI row, all four
      charts and the evidence table render (this proves the build works with no network).
- [ ] Set browser zoom to 100% and the window to 1440×900 so the type is legible on video.
- [ ] Close other tabs and apps; the dark theme hides everything else anyway.
- [ ] Deck open at page 2 in case you want the architecture diagram as a cutaway.
- [ ] Terminal font at 16pt or larger, dark profile, no shell prompt with private paths.
- [ ] Rehearse the bias-lift shot: it is the one place the narration must sound unhurried so
      the "we report it as a blind spot" line lands.
- [ ] Check the final video for any visible path containing a username.

## Source-backed captions (put on screen where marked)

| Shot | On-screen caption |
|---|---|
| 3 | `refdiv_mean_clean @ alpha 0.05 — 45/94 · 47.9% [38.1, 57.9] · 59/63 · 93.7% · 92/99 · 92.9% · bias 5/99 = its own FPR` |
| 3 | `thresholds frozen on the clean ledger first — runs/tpr_ladder_at_frozen.json` |
| 4 | `runs/merged_fpr_tpr_report.json — 28,313 held-out clean models, alpha 0.05` |
| 5 | `runs/tpr_ladder_at_frozen.json — frozen thresholds, caught/n arms` |
| 6 | `same threshold: refdiv 59/63 vs ctc 2/63 at substitution dose 0.50` · `396/396 bias-lift arms behaviour-inert` |
| 7 | `runs/mvp/comparison.json — 2 positives, 4 negatives, 2 ASR-gated excluded` |
| 8 | `signing_mode = hmac-sha256-fallback (not Ed25519)` |
| 10 | `<commit> · repository-relative paths only` |

## Fallback plan

- If the site will not open: use `deck/CVIAF_SIH26228_render.pdf` pages 3–5 as static
  visuals and keep the same narration. The headline strip and the dose panels are on page 3,
  so shot 3 and shots 5–6 become page 3, and shots 7–8 become page 5.
- If the dashboard's headline section is unreachable, the same four cells are on deck page 3;
  do not substitute the shape of the claim, only its home.
- If the terminal is unavailable: cut shot 9 and extend shot 10 by fifteen seconds.
- If you must shorten to two minutes: keep shots 2, 3, 7 and 10, and cut shots 4, 5, 8 and 9
  — in that order. Never cut shot 3 (the headline) or shot 6 (the behaviour-inert result);
  those two are the credibility of the whole video.
