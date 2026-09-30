#!/usr/bin/env python3
"""Populate the real SIH 2026 template with the CVIAF deck content.

Edits a COPY of the supplied template: the official title page, page order,
blue footer bar (0070C0), team oval, header logo and slide numbers are kept as
they are. Slide 7 ("IMPORTANT INSTRUCTIONS") is dropped so the deck is six
pages, and everything the team supplies is added inside the template's own
content area.

Layout is validated before saving: no added shape may cross into the footer bar
or run off the slide.
"""
import json
import os
import copy

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
REPO = os.environ.get("CVIAF_REPO", "/Users/billasur/cv-assurance-engine/.task3")
DECK = os.path.join(OUT, "deck")
CHARTS = os.path.join(OUT, "assets", "charts")
SHOTS = os.path.join(OUT, "assets", "screenshots")
DIAG = os.path.join(OUT, "assets", "diagrams")
TEMPLATE = "/Users/billasur/Downloads/SIH2026-IDEA-Presentation-Format(2).pptx"
REPO_URL = "https://github.com/Collectors-AU/CVIAF"
os.makedirs(DECK, exist_ok=True)

B = json.load(open(os.path.join(OUT, "site", "data", "bundle.json")))
COMMIT, BRANCH = B["commit"], B["branch"]
SHORT = COMMIT[:8]

BLUE = RGBColor(0x00, 0x70, 0xC0)      # template footer blue
INK = RGBColor(0x1F, 0x24, 0x2B)
GREY = RGBColor(0x5F, 0x66, 0x70)
MUTED = RGBColor(0x8A, 0x90, 0x98)
WARN = RGBColor(0xB0, 0x3A, 0x2E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xD8, 0xDC, 0xE2)

SLIDE_W = 13.3333
SLIDE_H = 7.5
FOOTER_TOP = 6.95          # inches; blue bar starts at 6354762 EMU
LEFT = 0.42
RIGHT = 12.95
WIDTH = RIGHT - LEFT

problems = []
placed = []          # (slide_no, name, left, top, width, height) for overlap checks


def check(slide_no, name, top, height, left, width):
    if top + height > FOOTER_TOP + 0.01:
        problems.append(f"slide {slide_no}: {name} crosses the footer "
                        f"({top + height:.2f}in > {FOOTER_TOP}in)")
    if left < 0 or left + width > SLIDE_W + 0.01:
        problems.append(f"slide {slide_no}: {name} runs off the slide "
                        f"(x {left:.2f}..{left + width:.2f})")
    placed.append((slide_no, name, left, top, width, height))


def overlaps():
    """Content shapes on the same slide must not sit on top of each other.

    Only the declared boxes are compared; pictures and tables are registered
    through box()/pic()/table_shape(), so a chart that grows past its caption
    fails the build instead of shipping.
    """
    hits = []
    for i, (n1, a1, l1, t1, w1, h1) in enumerate(placed):
        for n2, a2, l2, t2, w2, h2 in placed[i + 1:]:
            if n1 != n2:
                continue
            ox = min(l1 + w1, l2 + w2) - max(l1, l2)
            oy = min(t1 + h1, t2 + h2) - max(t1, t2)
            if ox > 0.03 and oy > 0.03:
                hits.append(f"slide {n1}: {a1} overlaps {a2} "
                            f"({ox:.2f}in x {oy:.2f}in)")
    return hits


def table_shape(slide, no, name, rows, cols, l, t, w, h):
    tbl = slide.shapes.add_table(rows, cols, Inches(l), Inches(t),
                                 Inches(w), Inches(h)).table
    tbl.name = name
    check(no, name, t, h, l, w)
    return tbl


def box(slide, no, name, l, t, w, h, wrap=True):
    sh = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    sh.name = name
    tf = sh.text_frame
    tf.word_wrap = wrap
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    check(no, name, t, h, l, w)
    return sh


def write(sh, blocks, size=10.5, colour=INK, space=3.5, bold=False, font=None):
    """blocks: list of str | (str, dict) | ('bullet', str)"""
    tf = sh.text_frame
    first = True
    for item in blocks:
        opts = {}
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
            text, opts = item[0], item[1]
        elif isinstance(item, tuple):
            kind, text = item[0], item[1]
            opts = item[2] if len(item) > 2 else {}
            if kind == "bullet":
                text = "\u2022  " + text
        else:
            text = item
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        p.text = text
        p.font.size = Pt(opts.get("size", size))
        p.font.bold = opts.get("bold", bold)
        p.font.color.rgb = opts.get("colour", colour)
        p.font.name = opts.get("font", font)
        p.space_after = Pt(opts.get("space", space))
        p.line_spacing = opts.get("ls", 1.0)
        first = False
    return sh


def pic(slide, no, name, path, l, t, w=None, h=None):
    if w:
        sh = slide.shapes.add_picture(path, Inches(l), Inches(t), width=Inches(w))
    else:
        sh = slide.shapes.add_picture(path, Inches(l), Inches(t), height=Inches(h))
    sh.name = name
    check(no, name, t, sh.height / 914400, l, sh.width / 914400)
    return sh


def hline(slide, no, name, l, t, w, colour=LINE, h=0.014):
    sh = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = colour
    sh.line.fill.background(); sh.shadow.inherit = False
    sh.name = name
    return sh


def band(slide, no, name, l, t, w, h, fill=RGBColor(0xF2, 0xF5, 0xF9)):
    sh = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = LINE; sh.shadow.inherit = False
    sh.name = name
    check(no, name, t, h, l, w)
    return sh


def retag(slide, no, shape, text, size=None, colour=None, bold=None,
          align=None, anchor=None):
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    if size:
        p.font.size = Pt(size)
    if colour is not None:
        p.font.color.rgb = colour
    if bold is not None:
        p.font.bold = bold
    if align is not None:
        p.alignment = align
    if anchor is not None:
        tf.vertical_anchor = anchor
    return shape


# ---------------------------------------------------------------- build
prs = Presentation(TEMPLATE)

# drop the instructions slide so the submission is exactly six pages
xml_slides = prs.slides._sldIdLst
ids = list(xml_slides)
xml_slides.remove(ids[-1])

slides = list(prs.slides)
print("slides after trim:", len(slides))

# --- common furniture: team name on every oval -------------------------------
for s in slides:
    for sh in s.shapes:
        if sh.name.startswith("Oval"):
            retag(s, 0, sh, "Team Collectors", size=10, bold=True, colour=WHITE,
                  align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# =============================== PAGE 1 =====================================
s1 = slides[0]
for sh in list(s1.shapes):
    if sh.name == "TextBox 9":
        tf = sh.text_frame
        tf.clear()
        rows = [("Problem Statement ID", "SIH26228"),
                ("Problem Statement Title", "Trustworthy Computer Vision Integrity "
                 "Assurance for Data, Models and Inference Outputs in "
                 "Multi-Contributor Pipelines"),
                ("Theme", "Blockchain & Cybersecurity"),
                ("PS Category", "Software"),
                ("Team ID", "179631"),
                ("Team Name", "Team Collectors")]
        first = True
        for k, v in rows:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            r1 = p.add_run(); r1.text = k + "  "
            r1.font.size = Pt(12); r1.font.bold = True; r1.font.color.rgb = GREY
            r2 = p.add_run(); r2.text = v
            r2.font.size = Pt(12.5); r2.font.color.rgb = INK
            p.space_after = Pt(5)
            first = False
        p = tf.add_paragraph()
        r = p.add_run()
        r.text = ("Offline integrity assurance for contributed data, models and "
                  "inference records \u2014 measured, not asserted.")
        r.font.size = Pt(12.5); r.font.bold = True; r.font.color.rgb = BLUE
        p.space_before = Pt(12)
        p = tf.add_paragraph()
        r = p.add_run()
        r.text = (f"Evidence base: CVIAF worktree {SHORT} (branch {BRANCH}); "
                  f"reviewed main {B['reviewed_main'][:8]}. Every figure names its "
                  f"file and denominator.")
        r.font.size = Pt(8.5); r.font.color.rgb = MUTED

# =============================== PAGE 2 =====================================
s2 = slides[1]
for sh in list(s2.shapes):
    if sh.name == "Title 1":
        retag(s2, 2, sh, "CVIAF \u2014 Computer Vision Integrity Assurance Framework",
              size=24, bold=True)
    elif sh.name == "TextBox 8":
        sh._element.getparent().remove(sh._element)
    elif sh.name.startswith("Rectangle"):
        sh.fill.solid()
        sh.fill.fore_color.rgb = RGBColor(0x0B, 0x0C, 0x0E)
        sh.line.fill.background()

write(box(s2, 2, "p2-problem", LEFT, 1.28, 5.05, 1.25), [
    ("h", "Problem & insight", {"size": 11, "bold": True, "colour": BLUE, "space": 2}),
    ("bullet", "Datasets, models and inference logs arrive from parties we do not control \u2014 each can be poisoned, swapped or edited."),
    ("bullet", "Today's checks cover one link each, none covers the chain, and none works when the network is gone."),
    ("bullet", "Classifier-era defences do not transfer: phantom boxes and cloaked objects have no classifier analogue."),
], size=9.5)

write(box(s2, 2, "p2-solution", LEFT, 2.66, 5.05, 1.35), [
    ("h", "Our solution", {"size": 11, "bold": True, "colour": BLUE, "space": 2}),
    ("bullet", "One offline engine assesses data, model, inference records and drift, then merges every flag into one signed report."),
    ("bullet", "Ground truth first: a laboratory plants known attacks, so every detector is scored, not asserted."),
    ("bullet", "Every score is a conformal p-value against a signed reference battery we control; sample flags roll up per contributor."),
], size=9.5)

write(box(s2, 2, "p2-proof", LEFT, 4.12, 5.05, 2.02), [
    ("h", "Innovation & measured proof", {"size": 11, "bold": True, "colour": BLUE, "space": 2}),
    ("bullet", "Calibrated, not tuned: an explicit error guarantee, and accept is forbidden when an applicable check did not run."),
    ("bullet", f"Image-level AUROC {B['auroc'][0]['bonf']:.4f} (fused_bonf, OGA fabrication) / {B['auroc'][1]['bonf']:.4f} (ODA cloaking); clean control {B['auroc'][2]['bonf']:.3f}."),
    ("bullet", "Provenance 8/8 attacked records vs 4/8 for the enrolled-digest baseline; 0/2 genuine false alarms."),
    ("bullet", f"Detection is measured, not asserted: at the frozen 5% false-alarm threshold RefDiv-mean "
     f"catches {B['tpr_headline']['cells'][0]['caught']}/{B['tpr_headline']['cells'][0]['n']} substitution "
     f"attacks at dose 0.25 and {B['tpr_headline']['cells'][2]['caught']}/{B['tpr_headline']['cells'][2]['n']} "
     f"weight tamper at dose 1.00; CTC-mean at the same threshold catches almost none."),
    ("bullet", f"One end-to-end report: {B['runs']['oda_s5']['findings']} findings, CRITICAL \u2192 quarantine, {B['runs']['oda_s5']['span_s']:.1f} s audit-trail span."),
], size=9.5)
write(box(s2, 2, "p2-foot", LEFT, 6.20, 5.05, 0.60), [
    ("AUROC is an image-level ranking statistic on 8 laboratory models, not model-level recall. "
     "The provenance run was signed with the HMAC-SHA256 fallback, not Ed25519.", {"size": 7.5, "colour": MUTED}),
], size=7.5)

write(box(s2, 2, "p2-archlabel", 5.72, 1.24, 7.2, 0.3),
      [("Architecture \u2014 who uses it and what the chain computes offline",
        {"size": 10.5, "bold": True, "colour": BLUE})], size=10.5)
pic(s2, 2, "p2-architecture", os.path.join(DIAG, "architecture.png"),
    5.72, 1.56, w=7.2)
write(box(s2, 2, "p2-archcap", 5.72, 6.14, 7.2, 0.6), [
    ("Everything inside the dashed boundary is computed offline from assets the team controls; "
     "contributors sit outside it. Source: runs/assurance/oda_s5, runs/merged_fpr_tpr_report.json, "
     f"runs/tpr_ladder_at_frozen.json, runs/coverage.json @ {SHORT}.", {"size": 7.5, "colour": MUTED}),
], size=7.5)

# =============================== PAGE 3 =====================================
s3 = slides[2]
for sh in list(s3.shapes):
    if sh.name == "Title 1":
        retag(s3, 3, sh, "TECHNICAL APPROACH", size=24, bold=True)
    elif sh.name == "TextBox 8":
        sh._element.getparent().remove(sh._element)

write(box(s3, 3, "p3-intro", LEFT, 1.18, WIDTH, 0.34), [
    ("Every artefact entering a computer-vision pipeline \u2014 data, model, inference record \u2014 is "
     "measured against a reference we own, and every conclusion ships with its confidence and its "
     "limits. Detection is scored only at thresholds the clean ledger froze first: RefDiv-mean catches "
     f"{B['tpr_headline']['cells'][0]['tpr']:.1f}% of substitution attacks at dose 0.25 and "
     f"{B['tpr_headline']['cells'][1]['tpr']:.1f}% at dose 0.50, and "
     f"{B['tpr_headline']['cells'][2]['tpr']:.1f}% of weight tamper at dose 1.00.",
     {"size": 9.5, "colour": GREY}),
], size=9.5)

chain = ["intake \u2022 reference", "signals", "frozen calibration",
         "attack scoring", "findings \u2022 coverage"]
cw = WIDTH / len(chain)
for i, c in enumerate(chain):
    sh = band(s3, 3, f"p3-chain{i}", LEFT + i * cw, 1.58, cw - 0.08, 0.30)
    retag(s3, 3, sh, c, size=9.5, bold=True, colour=BLUE, align=PP_ALIGN.CENTER,
          anchor=MSO_ANCHOR.MIDDLE)

# The headline cells, then the dose curves they are read off. Both come from the
# same receipt; the strip is the quoted figure, the panels are the curve behind it.
write(box(s3, 3, "p3-headline-note", 1.59, 1.92, 10.19, 0.18), [
    ("HEADLINE CELLS \u2014 the figures the repository's number audit enforces",
     {"size": 8, "bold": True, "colour": BLUE}),
], size=8)
pic(s3, 3, "p3-headline-strip", os.path.join(CHARTS, "tpr_headline_strip.png"),
    1.59, 2.10, w=10.19)
pic(s3, 3, "p3-dose-panels", os.path.join(CHARTS, "dose_panels_wide.png"),
    LEFT, 3.44, w=WIDTH)

findings = [
    ("Worked: RefDiv sensitivity rises with measured damage",
     "substitution 13/99 \u2192 45/94 (47.9%) \u2192 59/63 (93.7%); weight_tamper 7/99 \u2192 92/99 (92.9%). "
     "CTC-mean catches 2\u201329% at the same threshold.",
     "thresholds frozen from the clean ledger; arms are synthetic TinyDetector models"),
    ("Did not work: bias lift stays at the false-alarm rate",
     "5/99 = 5.1% at every dose; 396/396 arms behaviour-inert (zero F1 change)",
     "inert weight change, not a proven harmful backdoor \u2014 the blind spot is declared"),
    ("Failure handled: unscorable arms are reported, not hidden",
     "1,188 built \u2192 1,055 scored; 133 unscorable, all substitution",
     "substitution dose 1.00 has n=7, below min_positives=20 \u2014 shown hollow, never as 100%"),
]
tbl = table_shape(s3, 3, "p3-findings-table", len(findings) + 1, 3,
                  LEFT, 5.68, WIDTH, 1.06)
for i, htxt in enumerate(["Finding", "Evidence", "Limit"]):
    c = tbl.cell(0, i); c.text = htxt
    c.text_frame.paragraphs[0].font.size = Pt(8)
    c.text_frame.paragraphs[0].font.bold = True
for r, row in enumerate(findings, 1):
    for i, v in enumerate(row):
        c = tbl.cell(r, i); c.text = v
        c.text_frame.paragraphs[0].font.size = Pt(7.5)
        c.text_frame.word_wrap = True

write(box(s3, 3, "p3-foot", LEFT, 6.80, WIDTH, 0.14), [
    ("Clean-null FPR ledger, 28,313 held-out negatives: refdiv_mean_clean 5.18%, ctc_mean_clean 5.23% "
     "\u2014 both intervals include the 5% target; ctc_peak/q95 sit at the corpus ceiling (degenerate, "
     "not specific). Coverage 14 of 14 clauses measured, gate PASS. A passing gate is not proof of "
     "universal detection.", {"size": 6.5, "colour": MUTED}),
], size=6.5)

# =============================== PAGE 4 =====================================
s4 = slides[3]
for sh in list(s4.shapes):
    if sh.name == "Title 1":
        retag(s4, 4, sh, "FEASIBILITY AND VIABILITY", size=24, bold=True)
    elif sh.name == "TextBox 8":
        sh._element.getparent().remove(sh._element)

rows = [
    ("Claim", "Actual evidence", "Boundary"),
    ("Runs offline, end to end",
     f"verify-report validates runs/assurance/oda_s5 against the shipped schema; "
     f"audit chain {B['runs']['oda_s5']['audit_len']} entries, valid={B['runs']['oda_s5']['audit_valid']}",
     "laboratory corpus; no network in the assurance path"),
    ("Ingests real dataset formats",
     "COCO and YOLO round trip: 240 images, 392 boxes, exact annotation geometry",
     "PNG pixels are quantised \u2014 not byte-equal"),
    ("Produces a signed, auditable report",
     f"{B['runs']['oda_s5']['findings']} findings, CRITICAL \u2192 quarantine, "
     f"{B['runs']['oda_s5']['span_s']:.1f} s; report schema-valid",
     "HMAC-SHA256 fallback in this run \u2014 not Ed25519 non-repudiation"),
    ("Named limits, not silent passes",
     "14 supported attack classes and 8 unsupported conditions declared in the report's coverage statement",
     "the demo run skips white-box checks and is kept separate (run demo_assurance)"),
    ("Planned: real-backbone benchmark and deployment-grade signing",
     "[needs value from H200 run ledger] \u2014 not measured in this repository",
     "planned work, deliberately not presented as capability"),
]
tbl4 = table_shape(s4, 4, "p4-table", len(rows), 3, LEFT, 1.26, WIDTH, 1.92)
for r, row in enumerate(rows):
    for i, v in enumerate(row):
        c = tbl4.cell(r, i); c.text = v
        p = c.text_frame.paragraphs[0]
        p.font.size = Pt(8 if r else 8.5)
        p.font.bold = (r == 0)
        c.text_frame.word_wrap = True

pic(s4, 4, "p4-verify", os.path.join(SHOTS, "verify_oda_s5_terminal.png"),
    LEFT, 3.26, w=6.1)
pic(s4, 4, "p4-coverage", os.path.join(SHOTS, "coverage_terminal.png"),
    6.72, 3.26, w=6.1)
write(box(s4, 4, "p4-shots-cap", LEFT, 5.46, WIDTH, 0.32), [
    ("Left: a real verification run against the named report, executed on this build. "
     "Right: coverage resolved from live artefacts; unscorable arms are named, not dropped. "
     "Both captions carry the run and commit they came from.", {"size": 7.5, "colour": MUTED}),
], size=7.5)
write(box(s4, 4, "p4-inventory", LEFT, 5.82, WIDTH, 0.66), [
    ("Corpus inventories are kept apart: the 22,693-member 30 September export record is NOT additive "
     "with the 56,627-model study \u2014 its 1,056 and 11,637 components are the seed_10150_28385 and "
     "seed_10264_24607 shards already inside that study (docs/MODEL_INVENTORY.md \u00a73). A byte-level "
     "image-overlap receipt is still outstanding, so no leakage-free claim is made.",
     {"size": 7.5, "colour": WARN}),
], size=7.5)
write(box(s4, 4, "p4-photo", LEFT, 6.52, WIDTH, 0.22), [
    ("[insert lab photo \u2014 AI Lab @ AU QUASAR; context only, not a benchmark]",
     {"size": 7, "colour": MUTED}),
], size=7)

# =============================== PAGE 5 =====================================
s5 = slides[4]
for sh in list(s5.shapes):
    if sh.name == "Title 1":
        retag(s5, 5, sh, "IMPACT AND BENEFITS", size=24, bold=True)
    elif sh.name == "TextBox 8":
        sh._element.getparent().remove(sh._element)

write(box(s5, 5, "p5-intro", LEFT, 1.20, WIDTH, 0.44), [
    ("Implemented baseline checks vs CVIAF on the same laboratory cases \u2014 a measured tradeoff on a "
     "small cohort, not universal superiority. Read as an engineering result: fewer asset false alarms "
     "and a richer evidence trail, paid for with lower item-level recall. No measured reduction in "
     "analyst hours, incidents, cost or readiness risk exists in this repository.",
     {"size": 9.5, "colour": GREY}),
], size=9.5)

pic(s5, 5, "p5-model", os.path.join(CHARTS, "baseline_model_axis.png"),
    LEFT, 1.70, w=6.1)
pic(s5, 5, "p5-data", os.path.join(CHARTS, "baseline_data_axis.png"),
    6.72, 1.70, w=6.1)

ba = [
    ("What changed", "Before \u2014 implemented baseline", "After \u2014 CVIAF", "Boundary"),
    ("Model axis", "fixed threshold 2/2 TPR, 4/4 FPR", "1/2 TPR, 0/4 FPR",
     "2 positives, 4 negatives, 2 ASR-gated excluded"),
    ("Inference tamper + replay", "enrolled digest 4/8; abstains 2 attacked, 2 genuine",
     "8/8, FPR 0/2", "HMAC fallback; tamper check, not proof of origin"),
    ("Data axis \u2014 assets", "7/7 TPR but 1/1 false alarm", "2/7 TPR, 0/1 false alarm",
     "fewer asset false alarms, lower recall in this run"),
    ("Data axis \u2014 samples", "TP 363 / FP 24 \u00b7 TPR 84.03% \u00b7 precision 93.80%",
     "TP 73 / FP 51 \u00b7 TPR 16.90% \u00b7 precision 58.87%",
     "unfavourable to CVIAF and retained; asset and sample units are not interchangeable"),
    ("Attribution", "hash-only: top-1 undefined, 0/8 detected",
     "top-1 2/2 on single-culprit classes",
     "2 scored classes; not universal source accuracy"),
]
tbl5 = table_shape(s5, 5, "p5-table", len(ba), 4, LEFT, 5.24, WIDTH, 1.60)
for r, row in enumerate(ba):
    for i, v in enumerate(row):
        c = tbl5.cell(r, i); c.text = v
        p = c.text_frame.paragraphs[0]
        p.font.size = Pt(7 if r else 7.5)
        p.font.bold = (r == 0)
        c.text_frame.word_wrap = True

# =============================== PAGE 6 =====================================
s6 = slides[5]
for sh in list(s6.shapes):
    if sh.name == "Title 1":
        retag(s6, 6, sh, "RESEARCH AND REFERENCES", size=24, bold=True)
    elif sh.name == "TextBox 8":
        sh._element.getparent().remove(sh._element)

write(box(s6, 6, "p6-intro", LEFT, 1.26, WIDTH, 0.4), [
    ("Every link below opens the specific artefact in the project repository, at the commit this deck "
     "was built from. Read the receipt, not this slide.", {"size": 10, "colour": GREY}),
], size=10)

refs = [
    ("Problem statement", "PS 26228 \u2014 Trustworthy Computer Vision Integrity Assurance",
     f"{REPO_URL}"),
    ("Repository at this commit", f"Collectors-AU/CVIAF @ {SHORT} (branch {BRANCH})",
     f"{REPO_URL}/tree/{COMMIT}"),
    ("Research basis \u2014 2025\u201326 state of the art",
     "CV_INTEGRITY_ASSURANCE_2026.md \u2014 per-capability SOTA, the gap, and the module action list",
     f"{REPO_URL}/blob/{COMMIT}/CV_INTEGRITY_ASSURANCE_2026.md"),
    ("Research verification log",
     "RESEARCH_CHECKPOINT_26228.md \u2014 what was verified against which primary page, and what was dropped",
     f"{REPO_URL}/blob/{COMMIT}/RESEARCH_CHECKPOINT_26228.md"),
    ("Same-cohort comparison", "runs/mvp/comparison.json \u2014 model, data and provenance axes",
     f"{REPO_URL}/blob/{COMMIT}/runs/mvp/comparison.json"),
    ("Large clean-null FPR receipt", "runs/merged_fpr_tpr_report.json \u2014 56,627 models",
     f"{REPO_URL}/blob/{COMMIT}/runs/merged_fpr_tpr_report.json"),
    ("Fleet ledger", "runs/fleet_fpr_ledger_report.json \u2014 7,503 models, 3,751 held-out",
     f"{REPO_URL}/blob/{COMMIT}/runs/fleet_fpr_ledger_report.json"),
    ("Dose-ladder receipt", "runs/tpr_ladder_at_frozen.json \u2014 1,188 built / 1,055 scored",
     f"{REPO_URL}/blob/{COMMIT}/runs/tpr_ladder_at_frozen.json"),
    ("Coverage and as-built matrix", "runs/coverage.json \u00b7 docs/COVERAGE_STATEMENT.md \u00b7 docs/PS26228_ALIGNMENT_MATRIX.md",
     f"{REPO_URL}/blob/{COMMIT}/docs/COVERAGE_STATEMENT.md"),
    ("Report example", "runs/assurance/oda_s5/assurance_report.json \u2014 schema-valid, 17 findings",
     f"{REPO_URL}/blob/{COMMIT}/runs/assurance/oda_s5/assurance_report.json"),
    ("Model inventory", "docs/MODEL_INVENTORY.md \u2014 where every model came from",
     f"{REPO_URL}/blob/{COMMIT}/docs/MODEL_INVENTORY.md"),
    ("Reproduce offline", "docs/REPRODUCE.md \u2014 no network, bundle install",
     f"{REPO_URL}/blob/{COMMIT}/docs/REPRODUCE.md"),
]
tbl6 = table_shape(s6, 6, "p6-refs", len(refs) + 1, 2, LEFT, 1.7, WIDTH, 4.5)
for i, htxt in enumerate(["Topic", "Open in repository"]):
    c = tbl6.cell(0, i); c.text = htxt
    c.text_frame.paragraphs[0].font.size = Pt(8.5)
    c.text_frame.paragraphs[0].font.bold = True
for r, (topic, label, url) in enumerate(refs, 1):
    c0 = tbl6.cell(r, 0); c0.text = topic
    c0.text_frame.paragraphs[0].font.size = Pt(8.5)
    c1 = tbl6.cell(r, 1)
    c1.text_frame.clear()
    p = c1.text_frame.paragraphs[0]
    run = p.add_run(); run.text = label
    run.font.size = Pt(8.5)
    run.hyperlink.address = url
    run.font.color.rgb = BLUE
    c1.text_frame.word_wrap = True

write(box(s6, 6, "p6-foot", LEFT, 6.3, WIDTH, 0.62), [
    ("Signal names are published methods: CTC and FTC are TRACE (CVPR 2025, arXiv 2503.15293), and the "
     "OGA / RMA / GMA / ODA attack labels are BadDet. This engine reimplements those mechanisms at MVP "
     "scale \u2014 the published numbers stay with their authors and are not quoted here.",
     {"size": 7.5, "colour": MUTED}),
    ("Evidence manifest: evidence/evidence_manifest.json \u2014 every slide number maps to a file, field and "
     "denominator. All figures are laboratory / synthetic scope; field validation is pending. "
     "docs/PS26228_REQUIREMENT_TRACE.md is historical design intent, not as-built evidence.",
     {"size": 7.5, "colour": MUTED}),
], size=7.5)

problems.extend(overlaps())

out = os.path.join(DECK, "CVIAF_SIH26228_template.pptx")
prs.save(out)
print("  pptx  ", out, os.path.getsize(out))

if problems:
    print("\nLAYOUT PROBLEMS:")
    for p in problems:
        print("  -", p)
    raise SystemExit(1)
print("  layout OK: every added shape sits inside the content area")
