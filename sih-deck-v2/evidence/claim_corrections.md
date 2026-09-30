# Claim corrections — supplied SIH26228 deck vs committed receipts

Built from worktree `2fda6eab` (branch `task3-real-backbone`). Reviewed main was
`6cddfe0e`; the local worktree is ahead, so anything newer is labelled rather than
silently mixed.

The first section audits the **supplied** deck against the receipts. The last section audits
**this rebuild** against the same receipts.

Legend: **confirmed** · **overstated** · **wrong unit** · **unverified** · **stale** ·
**mixed runs**

| # | Page | Claim as written | Verified value | Verdict | Replacement |
|---|------|------------------|----------------|---------|-------------|
| 1 | 2 | "Measured: AUROC 0.987 / 0.906" | `fused_bonf` OGA 0.9865625 (object fabrication), ODA 0.90625 (object cloaking); clean control exactly 0.500 | overstated | Name detector **and** attack: "image-level AUROC 0.9866 (fused_bonf, OGA fabrication) / 0.9063 (ODA cloaking); clean control 0.500" |
| 2 | 2 | "provenance 1.00 vs 0.50 hash-only" | CVIAF 8/8 attacked records, 0/2 genuine; enrolled digest 4/8, abstains on 2 attacked + 2 genuine | confirmed | Add the denominator and the abstentions: "8/8 vs 4/8, both 0/2 on genuine; the baseline abstains on 4 records" |
| 3 | 2 | "27.6 s offline" | `oda_s5` audit trail spans 07:44:42.211764 → 07:45:09.358053 = **27.15 s** | stale | Use **27.1 s** and name the run, or cite the instrument that produced 27.6 |
| 4 | 3 | Hand-tuned cut "0.082 / 0.000" | `runs/mvp/comparison.json` → `hand_tuned_single_signal`: TPR 0/2 = 0.00, FPR 0/4 = 0.00 | unverified | "0.00 / 0.00 — the tuned cut never fires on this cohort" |
| 5 | 3 | CVIAF "0.146 / (7/48 arms)" with data 1.00 / 0.00 | CVIAF model axis TPR **1/2 = 0.50**, FPR 0/4; data asset axis TPR **2/7 = 0.2857**, FPR 0/1 | wrong unit | Split the row: model 1/2 & 0/4; asset 2/7 & 0/1. The 7/48 receipt was not found in the repository |
| 6 | 3 | "fleet_fpr_ledger_report.json 7,503 clean models • FPR 0.048 (3,751)" | 7,503 clean models; 3,752 calibration / 3,751 held-out; ctc_mean 180/3751 = 4.80% [4.16, 5.53]; refdiv_mean 215/3751 = 5.73% [5.03, 6.52] | confirmed | Keep, and add that refdiv_mean is 5.73% in this smaller ledger — the two rules do not agree here |
| 7 | 3 | "fleet_census.json 8,064 models censused • 0 duplicate ids, specs or weights" | `runs/fleet_census.json`: n_models 8,064, 6 shards, duplicate lists empty, `ready_for_fpr: true` | confirmed | Keep unchanged |
| 8 | 3 | "coverage.json 14 clauses measured • gate PASS" | `runs/coverage.json`: 14 clauses measured, `gate_ok: true` | confirmed | Keep, plus: "a passing gate is not proof of universal attack detection" |
| 9 | 3 | REPRODUCE block: `configs/corpus_mvp2.json`, `runs/mvp2` | Present in the **outer** checkout, absent from the `.task3` worktree | unverified | State which root the commands assume, or point at `configs/corpus_mvp.json` + `runs/mvp` which exist in the worktree |
| 10 | 4 | "55,000+ trained models across 9 attack families" | 56,627 **clean-null** models in the FPR study (no attacked positives); attack arms are 1,188 built across **3** classes × 4 doses | overstated | "56,627 clean models scored for false alarms, plus 1,188 attacked variants across three tamper classes" |
| 11 | 4 | "entire 55,000-model corpus is under 2 GB" | No size receipt in the repository; model directories live outside git | unverified | Omit from the slide. If it must appear, attach a measured `du` receipt |
| 12 | 4 | "Ed25519 keys held on-box" | `runs/mvp/comparison.json` → `signing_mode: hmac-sha256-fallback`; the assumption text calls it symmetric | overstated | "Tamper-evident signing with HMAC-SHA256 in this run; Ed25519 is the intended deployment mode and is not yet measured" |
| 13 | 4 | "One H200 for 4–8 hours generates the real-detector benchmark" | No run ledger for an H200 job | unverified | Keep only as explicitly labelled planned work |
| 14 | 5 | "17 findings, CRITICAL → quarantine, 27.6 s" | oda_s5: 17 findings, CRITICAL, quarantine, 7-entry valid chain, 27.15 s span | stale (time only) | 27.1 s, run oda_s5 |
| 15 | 5 | "coverage 14 / 8 / 5" on the same slide as the 17 findings | 14 supported / 8 unsupported / 5 assumptions comes from **demo_assurance**; the 17 findings come from **oda_s5** | mixed runs | Never on one slide. Either name both runs or drop the coverage triple here |
| 16 | 5 | "Top-1 source accuracy 1.00, false accusations 0.00" | 2/2 on the two single-culprit classes (dup_flood, ood_insert); five other classes have no single culprit by construction | overstated | "top-1 2/2 on the two single-culprit classes; 0 false accusations" with the class count |
| 17 | 5 | "Asset FPR 0.00 vs 1.00" | CVIAF 0/1 vs baseline 1/1 — one clean asset | overstated | Keep the numbers but state n=1: it cannot support a rate |
| 18 | 5 | (implied) CVIAF dominates the baseline | Sample level: baseline TP 363 / FP 24 (TPR 84.03%, precision 93.80%) vs CVIAF TP 73 / FP 51 (TPR 16.90%, precision 58.87%) | overstated | Show the tradeoff explicitly; both units are already in the rebuilt page 5 |
| 19 | 6 | Empty placeholder | — | — | Filled with 10 topic links that open the specific artefact in the repository at this commit |
| 20 | all | FPR study and export inventory used in the same conversation | 1,056 and 11,637 are shards *inside* the 56,627 study (`docs/MODEL_INVENTORY.md` §3) | mixed runs | Never add the two populations; a deletion/dedup receipt is still required |

## Self-audit of this rebuild — the defects found in our own artefacts

These are ours, not the supplied deck's, and every one is fixed in place.

| Symptom | Root cause | Fix |
|---|---|---|
| **The detection headline was missing.** Neither the site nor the deck ever said how much is caught. The first thing a reader met was "0 attacked positives in the clean ledger → TPR undefined, not zero". | That caveat is correct and load-bearing, but it *led* every artefact, while the ladder's frozen-threshold TPR existed only as per-dose counts inside a table. The headline (47.9% / 93.7% / 92.9% / 5.1%) was never surfaced as a claim. | The four audited cells are now quoted together everywhere: site section `#headline` (chart + table + three notes), deck page 2 (proof bullet) and page 3 (`tpr_headline_strip` + `dose_panels_wide` + findings table), supplement §6, narration shot 4, and four manifest rows `page3-tpr-headline-*`. The zero-positive caveat is demoted, not deleted — it now says the ladder is where detection is measured. |
| Generated manifest rows carried a fabricated `exclusions` value (`n − caught`, e.g. 49 for a cell with 5 unscorable arms). | The block computed exclusions arithmetically instead of reading the receipt's `unscorable` map. | Rows now read `unscorable` from `runs/tpr_ladder_at_frozen.json`: 5 (subst 0.25), 36 (subst 0.50), 0, 0. |
| `source_field` paths pointed at dose keys that do not exist (`cells.substitution.0.5`, `cells.weight_tamper.1`). | The label formatter stripped trailing zeros off the key. | Both the receipt key (`0.5`, `1`) and the declared label (`0.50`, `1.00`) are now stored and written separately. |
| The superseded 193-arm experiment (substitution 46/94, weight tamper 21/99) sat one click from the 4-dose ladder (45/94, 92/99) with no warning. | Two arms sets, same rule name, adjacent files. | A SUPERSEDED banner on the site, a manifest row `supplement-superseded-193` and a "never merge or average" line in supplement §6 and `assets/data/tpr_superseded_193.csv`. |
| **Pictures overlapped tables** in the editable template PPTX: page 3's dose panels ran into the findings table, page 4's terminal crops into their caption, page 5's baseline charts into the before/after table. | The build only asserted footer/slide bounds, never shape-to-shape collision, and the chart PNGs are taller than assumed at their placed width. | `build_deck_template.py` now registers every content box and fails the build on any overlap; the affected pages were re-laid out (page 3 gained the headline strip and a single wide dose-panel figure, page 4 and 5 charts are placed to fit above their tables). |

## Claims removed rather than rewritten

These had no receipt at all and were dropped from the main slides:

- "9 attack families" as a headline (the scored ladder has three classes).
- "under 2 GB" corpus size.
- "universal GPU-free" support as a capability statement.
- Completed Ed25519 non-repudiation.
- Any Army/DGIS deployment, rollout, cost-saving or readiness claim.

Unresolved values stay as `[needs value from FILE/FIELD]` in the working material and are
omitted from the submission claim — most visibly the H200 benchmark ledger and the
byte-level image-overlap receipt.
