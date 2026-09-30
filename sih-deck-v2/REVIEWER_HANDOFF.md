# Reviewer handoff — CVIAF SIH26228 deck

For the final design and credibility pass. Everything below names the file to open.

## Chosen commit

| | |
|---|---|
| Repository | `https://github.com/Collectors-AU/CVIAF` |
| Worktree used for every number | job-local CVIAF worktree, branch **`task3-real-backbone`** |
| Commit | **`2fda6eab99dea3da40b69c10cd8b97bbf2fe315c`** |
| Previously reviewed main | `6cddfe0e4aea71f2bea8c0b137e5ad4f268cb145` |
| Delta | the worktree is **ahead** of the reviewed main |

Consequence for the review: evidence newer than `6cddfe0e` is labelled in the deck and in
the manifest. Nothing from the two states is silently mixed. `docs/MODEL_INVENTORY.md` is
the catalogue of what is committed where; note that the 56,627 model directories live
**outside git**, which is why some evidence-explorer rows are marked `local artifact`.

## What was produced

| Artefact | Path | Notes |
|---|---|---|
| Editable deck | `deck/CVIAF_SIH26228_template.pptx` | built **inside the supplied SIH template**; 6 slides |
| Rendered PDF | `deck/CVIAF_SIH26228_render.pdf` | 6 pages, 16:9, generated from the HTML rendering |
| Chart source | `assets/charts/*.svg` + `.png` | matplotlib from parsed JSON, not hand-drawn |
| **Detection headline** | `assets/charts/tpr_headline_strip.png` + `tpr_headline.png` | the four audited cells, as a slide strip and a bar chart |
| **Dose panels (slide)** | `assets/charts/dose_panels_wide.png` | both families in one wide figure, sized for one slide row |
| Screenshots | `assets/screenshots/*.png` | real dashboard captures + two real terminal runs |
| Architecture diagram | `assets/diagrams/architecture.svg` + `.png` | users/actors + offline chain |
| Data layer | `assets/data/*.csv` | one CSV per claim family (incl. `tpr_headline.csv`, `tpr_superseded_193.csv`) |
| Evidence manifest | `evidence/evidence_manifest.json` | 40 entries: file, field, unit, n/d |
| Claim corrections | `evidence/claim_corrections.md` | 20 entries vs the supplied deck + 5 self-audit entries |
| Supplement | `supplement/before_after_analysis.md` | full numbers + unfavourable results |
| Dashboard | `site/index.html` + `site/data/bundle.json` | self-contained, offline, ready to deploy |
| **Research basis** | repo root `CV_INTEGRITY_ASSURANCE_2026.md` + `RESEARCH_CHECKPOINT_26228.md` | the 2025–26 state-of-the-art dossier and its citation-verification log; both tracked and deployed to this commit |
| Video | `video/narration_and_storyboard.md` | narration + shot list + checklist |

Regenerate anything with:

```bash
cd sih-deck-v2
../.venv/bin/python scripts/build_assets.py        # CSVs + charts + manifest
../.venv/bin/python scripts/build_site.py          # bundle + architecture + dashboard
../.venv/bin/python scripts/build_deck.py          # HTML + PDF render
../.venv/bin/python scripts/build_deck_template.py # the editable PPTX
```

## The research basis — where the method names come from

The signal names in this evidence are published methods, and the repository carries both the
dossier and the log of how its citations were checked. Both documents are tracked at the handoff
commit, so the deck's page-6 links resolve.

| Where | What it names |
|---|---|
| `CV_INTEGRITY_ASSURANCE_2026.md` (repo root) | 2025–26 state of the art per capability, the gap against this engine, and the action list for each module. Every source fetched against its primary page and confidence-tagged. |
| `RESEARCH_CHECKPOINT_26228.md` (repo root) | The verification log behind that dossier: what was checked against which primary page, and the citations dropped or downgraded when verification failed — a withdrawn preprint, a method that could not be shown to exist. |
| Dashboard → *Research basis* | Two document cards plus a table splitting the cited methods into **run in this build** (with the code path that implements each) and **design basis, not run here**. |
| Deck page 6 | Two rows in the reference table, linked at the handoff commit, plus a foot line naming TRACE and BadDet. |
| `video/narration_and_storyboard.md` | Shot 6 names CTC as TRACE's transformation-consistency score; the caption for that shot carries the citation. |

The attribution that matters most: **CTC and FTC are TRACE (CVPR 2025, arXiv 2503.15293)** and
**the OGA / RMA / GMA / ODA attack labels are BadDet** — the AUROC rows and the CTC column are
those methods. `cviaf/lab/detectors.py` says in its own docstring that this is a faithful
reimplementation of the *mechanisms* at MVP scale, not a reproduction of the published numbers,
and the published figures are never quoted as ours. The page states the same split for the
enrichments named in the dossier but not implemented here (ODSCAN, DISTIL, C2PA 2.4, OpenOOD v1.5,
NIST TrojAI/AI RMF) rather than implying coverage.

## What changed in the deck

1. **The template is preserved.** Slide order, title page fields, blue footer bar
   (`0070C0`), the team oval, the header logo and the slide numbers are the template's own.
   Slide 7 ("IMPORTANT INSTRUCTIONS") was removed so the submission is exactly six pages.
   Two layout assertions run at build time: no added shape may cross into the footer bar or
   run off the slide, **and no two content shapes may overlap** — the build fails on either.
   (The overlap assertion was added after it caught real collisions on pages 3, 4 and 5;
   see `evidence/claim_corrections.md`, self-audit section.)
2. **Architecture diagram added** to page 2 — a user/actor view (contributors → analysts →
   procurement/audit) over the offline chain (intake → reference → signals → calibration →
   disposition), with the evidence outputs and the trust boundary drawn explicitly.
3. **Clickable references** on page 6: 12 rows, each opening the specific artefact in the
   repository **at this commit** — first the research basis and its verification log, then the
   receipts, as-built docs and reproduce guide. A foot line on that page names the published
   methods behind the metric names (TRACE for CTC/FTC, BadDet for the attack labels).
4. **Page 3 leads with the detection headline.** A four-cell strip (substitution 47.9% at
   dose 0.25 and 93.7% at 0.50, weight tamper 92.9% at 1.00, bias lift 5.1% — the
   behaviour-inert failure) sits above one wide two-family dose panel with Wilson intervals
   and caught/n labels, then a three-row findings table whose first row carries the headline
   percentages and CTC-mean at the same threshold. The intro states the headline in words, so
   the page reads as an answer even without the charts.
5. **Page 5 is a before/after page**, not a promise list, including the result that is
   unfavourable to CVIAF.

## The detection headline — where it is stated, and what it means

Every artefact now quotes the same four cells, from `runs/tpr_ladder_at_frozen.json`, rule
`refdiv_mean_clean` at alpha 0.05, thresholds frozen on the clean ledger before any attack
was scored. They are never pooled across classes or doses.

| Where | What it shows |
|---|---|
| Dashboard `site/index.html` → *Detection headline* | bar chart with Wilson intervals and the frozen 5% line, the four-row table (caught/scored, CI, CTC-mean at the same threshold, unscorable, behaviour-inert), the inert-blind-spot note, the CTC divergence note, and the SUPERSEDED banner for the earlier 193-arm set |
| Deck page 2 | one proof bullet with the dose-0.25 and dose-1.00 counts, and the CTC contrast |
| Deck page 3 | the strip, the wide dose panels, and the findings table |
| `supplement/before_after_analysis.md` §6 | the full table, the equal-calibration/unequal-sensitivity result, and the two-arms-sets warning |
| `evidence/evidence_manifest.json` | four rows `page3-tpr-headline-*` with file, field, unit, numerator, denominator, rule, operating point and exclusions |
| `assets/data/tpr_headline.csv` | the four cells as data; `tpr_dose_ladder.csv` carries all 24 ladder cells |

Read it as: **at a false-alarm budget of 5%, frozen in advance, RefDiv-mean catches 47.9% of
substitution attacks at dose 0.25 and 93.7% at dose 0.50; 92.9% of weight tamper at dose
1.00; and 5.1% of bias lift, which is behaviour-inert, so that number is the false-alarm rate
and not a detection.** Two rules pass the same 5% false-alarm test and still behave very
differently on attacks — 59/63 versus 2/63 at substitution dose 0.50.

The clean-null ledger's zero attacked positives remain stated (overview note, limitations
accordion, page 3 foot line); it is the reason detection is measured on the ladder rather
than in that ledger. Bias-lift inertness is declared in all four places.

## Open evidence gaps — do not close these in prose

1. **`[needs value from H200 run ledger]`** — real-backbone (YOLOv8 / Faster-RCNN) benchmark.
   Presented as planned work on page 4.
2. **No image-overlap receipt.** The 22,693-member export record overlaps the 56,627 study
   (`docs/MODEL_INVENTORY.md` §3: shards `seed_10150_28385` = 1,056 and `seed_10264_24607` =
   11,637). The populations are **not additive** and no disjointness or leakage-free claim
   is made anywhere.
3. **Corpus size.** No `du` receipt exists, so "under 2 GB" was removed.
4. **Ed25519.** The committed provenance run is HMAC-SHA256 fallback; the deck says so on
   page 2, page 5 and in the site.
5. **Group-aware uncertainty.** Parent models are reused across dose variants, so per-arm
   Wilson intervals are descriptive only.
6. **27.6 s → 27.1 s.** The supplied deck's runtime does not match the `oda_s5` audit-trail
   span (07:44:42.21 → 07:45:09.36 = 27.15 s). The deck now uses the measured span.
7. **Two mixed-run claims** were separated: page 5's `17 findings` (oda_s5) and
   `14 / 8 / 5 coverage` (demo_assurance) must not sit on one slide.

## Assembly limitations — read before editing

- **The PDF is not a PowerPoint export.** No LibreOffice is installed on this machine, so
  `CVIAF_SIH26228_render.pdf` is Chrome's print of the HTML rendering, not a render of the
  PPTX. The two carry the same content and numbers, but line breaks differ. Treat the PPTX
  as the submission file and the PDF as a faithful preview; if a pixel-exact PDF is needed,
  open the PPTX in PowerPoint and export.
- **The template's title font is Times New Roman bold**, inherited from the official
  template. It was deliberately not restyled. If the design pass changes it, change it in
  the template, not per slide.
- **Screenshots are captured at committed settings** through a hash-driven tab shim added to
  a *copy* of `demo/fpr_dashboard.html`; the original file is untouched. The shim only
  clicks an existing tab and adds no product behaviour.
- **The dashboard is a new build**, not the existing one. `demo/fpr_dashboard.html` is the
  product's own dashboard; the site at `site/index.html` is an evidence view built for this
  submission and is labelled as a static build, not a live engine.
- **The lab photo is still a placeholder** on page 4.

## What the design pass should look at first

1. Whether the architecture diagram on page 2 reads at presentation size — it is the newest
   and densest element.
2. Page 5's table density at 7.5pt; it is the most likely place to overrun when text is
   edited by hand.
3. Whether the terminal crops on page 4 are legible from the back of a room. If not, cut the
   coverage one and let the verification crop take the full width.
4. Every number on a slide must still match `evidence/evidence_manifest.json`; if you edit a
   figure, edit the receipt path with it.
