# Deep Research Skill — protocol for CVIAF research

**What this is.** A repeatable protocol for continuing the research behind SIH 26228 without re-deriving it, and without the failure mode that killed the first attempt (`RESEARCH_CHECKPOINT_26228.md`: 55 sub-agents completed, 56 died on quota, the synthesis never ran and its raw output was lost).

**Use it when** you need to add a method, verify a claim, answer "is this real?", or refresh a capability area. **Do not use it** to re-search things already in `RESEARCH_CHECKPOINT_26228.md` and `CV_INTEGRITY_ASSURANCE_2026.md` §11 — those are authoritative and already verified.

---

## 0. The five rules

1. **A claim without a fetched primary source is not a finding.** It is a lead. Write it under `LEADS`, never under `FINDINGS`.
2. **Tag every claim with its verification level** (§3). Never upgrade a tag you did not earn.
3. **Failure survives.** Withdrawals, retractions, gated PDFs, and unverifiable papers stay in the ledger as *negative* results. Deleting them is how a report acquires a false citation.
4. **Network research never blocks the critical path.** If a fetch fails, record `UNVERIFIED` and keep building. The deliverable is a design; research exists to inform it.
5. **Write the checkpoint before you synthesize.** Any agent that can be killed mid-run must persist its state first. This is the rule the first attempt broke.

---

## 1. Where to actually look (and what each source is good for)

Ordered by signal-to-noise for *this* problem class. The prompt that started this work asked specifically for places where practitioners talk, not just papers — that is the right instinct, and here is how to do it deliberately rather than by vibes.

| Source | What it is genuinely good for | What it is *not* good for | How to query |
|---|---|---|---|
| **arXiv abstract + HTML** (`arxiv.org/abs/`, `/html/`) | The primary claim, the method, and critically the **Limitations** section — which is where our coverage statement comes from | Nothing; it is the ground truth for methods | fetch `/abs/` for metadata, `/html/` for limitations; **check `v2`/`v3` revisions and the submission history** — revocation and revision are where surprises hide |
| **Venue proceedings** (CVPR/ICCV/ICLR/S&P/NeurIPS/AISTATS via PMLR, openaccess.thecvf, usenix, computer.org) | Confirming a paper was actually *accepted* and not merely posted. Critical: a withdrawn arXiv paper presented as authoritative is the single most common error in this literature | — | cross-check the venue page; `openaccess.thecvf.com` may block non-browser agents (see §6), so prefer the arXiv abs/HTML as the fetchable primary and record the venue separately |
| **GitHub repos** (`github.com/<owner>/<repo>`) | Whether code exists, what it actually implements, last-commit date, open issues (the honest failure list), and the pin-able commit hash for the offline bundle | Performance numbers in READMEs | search `"<method name>" github`, then read the repo's issues for known breakage |
| **Standards bodies** (csrc.nist.gov, spec.c2pa.org, cyclonedx.org, slsa.dev, in-toto.io/specs, openssf.org, sigstore.dev) | The correct primitive to *align to* instead of inventing one. This is where the differentiators live, because almost no student team reads specs | — | search `"<standard name>" <year> specification`; fetch the capability/spec page, not the blog post |
| **Reddit** (r/MachineLearning, r/computervision, r/cybersecurity, r/hackathon, r/ExperiencedDevs) | Field reports: what breaks in production, false-positive fatigue, how practitioners distrust detection tools, and hackathon evaluation reality | Technical fact. Treat as *motivation*, never as evidence | search `site:reddit.com <topic> <failure/limitation words>` and `"<method>" reddit solved OR false positive OR production` |
| **X / Twitter** | Early-signal only: a paper dropping, an author correcting their own work, a field controversy | Anything citable | search `"<method>" X twitter thread 2026`; if a claim only exists there, mark `LEAD` |
| **Benchmark registries** (pages.nist.gov/trojai, SCLBD/backdoorbench, Jingkang50/OpenOOD, DISC2021) | External validation that does not depend on our own corpus | — | note whether a relevant **round** exists (e.g. TrojAI's object-detection round) |
| **Security write-ups** (Black Hat, MSRC, securityaffairs, The Hacker News) | Real incident evidence that the threat class is not academic (e.g. AI supply-chain compromises) | Method detail | search `AI supply chain breach <year> model poisoning incident` |

**Anti-pattern to avoid:** searching the problem statement's own phrasing. Every team does that and gets the same SEO listicles. Search the *mechanisms* (`pre-NMS distribution shift`, `conformal p-value client-level anomaly`, `no-free-lunch backdoor detection`, `model manifest hashing ONNX external data`).

---

## 2. The research loop

Run it per question, not per session. Six steps, and step 6 is not optional.

```
1. FRAME      → turn the question into a falsifiable claim you'd have to withdraw.
                Bad:  "what's new in backdoor detection?"
                Good: "is universal backdoor detection possible, and if not, what
                       assumption licenses any detection claim we make?"
2. SEED       → 3–6 parallel searches naming mechanisms, not problem statements.
                Include at least one venue/standards source and one practitioner source.
3. FETCH      → pull the primary page for the best hits. Prefer /abs/ + /html/.
                Record status, final URL, content hash, and access date.
4. TRIANGULATE→ does an independent source (venue, repo, second paper) agree?
                Disagreement is a FINDING about the state of the art, not a problem.
5. TAG        → assign a verification level (§3) and write the limitation the
                source itself admits.
6. CHECKPOINT → append to the ledger + checkpoint BEFORE writing any synthesis.
```

**Parallelism rule (this is the "use background processes" instruction, done properly).** Fan out the *seeding* step — 3–6 independent searches in one batch — but **never parallelize the tagging step**, because the failure mode of the first attempt was 55 agents producing un-tagged output that could not be reconciled. Research parallelism is safe; *synthesis* parallelism is not, unless each agent writes its own checkpoint file.

**Scaling rule.** If you do use multiple agents, each must own a **file** (`research/<area>.md`) and write to it incrementally. A synthesis step then reads files, not memory. Agents that only return text will be lost when quota kills them — that is precisely what happened at `wf_4e0c352f-abd`.

---

## 3. Verification tags (use exactly these)

| Tag | Means | May be used for |
|---|---|---|
| `[VERIFIED]` | Primary page fetched and read; metadata locked (title, authors/body, ID, date) | Any load-bearing claim |
| `[SEARCH]` | Found and consistent across ≥2 independent results; not fetched in full | Supporting claims, clearly labelled medium confidence |
| `[LEAD]` | Single unverified mention; may not exist | Nothing. Listed as a lead only |
| `[RETRACTED]` | Author withdrew / paper not accepted / source disproven | Negative results only — keep it, never cite as authority |
| `[GATED]` | Real but inaccessible (CAPTCHA, paywall, 403) | Mechanism-level citation with **no numbers** |
| `[ASSERTED]` | Our own design decision, requires no source | Design sections only, never as evidence |

**Known failure modes this taxonomy exists to catch** (all observed in this project):

- **Withdrawn-but-cited.** `AnywhereDoor` (arXiv 2503.06529) was withdrawn by its author while its code repository stayed up. `[RETRACTED]` as a source; usable only as threat-model inspiration.
- **Non-existent method.** "TRIM" surfaced in a sub-agent's output and could not be verified as a real paper. Deleted. A citation that cannot be found is worse than no citation.
- **Revision surprise.** TRACE is `v2`, revised 2026-07-30, with a different abstract framing from `v1`. Always read the submission history — a `v2` often *weakens* a `v1` claim.
- **Internal inconsistency.** TRACE reports both 42 and 63 backdoored models across revisions. Cite the metric, not the corpus count.
- **Venue overstatement.** "IARPA's TrojAI" — IARPA launched the program, **NIST ran the associated evaluations**. Say it precisely; a judge from a standards background will notice.
- **Blog-derived fact.** Model-signing details differ between OpenSSF's blog and the specification. Fetch the spec.

---

## 4. Source ledger format

Append-only, one block per source. Keep it in the checkpoint file, not in chat.

```markdown
### <short name>
- URL: <primary page actually fetched>
- Status: <HTTP status> · accessed <YYYY-MM-DD> · sha256 of extracted text: <...>
- Tag: [VERIFIED] | [SEARCH] | [LEAD] | [RETRACTED] | [GATED]
- Claim used in the design: <one sentence, the exact claim>
- **Limitation the source itself admits:** <verbatim-ish — this is the most valuable line>
- Bearing: <which capability / which design axiom it licenses>
- Action: ADOPT | BASELINE-TO-BEAT | THREAT-MODEL-ONLY | DO-NOT-CITE
```

The **"limitation the source admits"** field is what turns a survey into a coverage matrix. Pull it every single time; it is the difference between "we use TRACE" and "we use TRACE and we know it needs auxiliary public images, which we therefore vendor into the battery."

---

## 5. Refresh cadence

| Area | Re-check every | Watch for |
|---|---|---|
| OD backdoor attacks/defenses | 3 weeks | it moves fast; 2026 alone added BadDet+ (Jan), DistScan (Aug), ODPure (Sep) |
| Backdoor *theory* | 3 months | impossibility/limits results change what claims are admissible |
| ML supply-chain standards | 1 month | OMS, in-toto, C2PA, CycloneDX ML-BOM are all active |
| TrojAI / BackdoorBench / OpenOOD | before any benchmark claim | a new round can invalidate comparisons |
| Benchmarks and repos | before the offline bundle is frozen | pin commits; code rot is real |

---

## 6. Mechanical verification

`scripts/verify_sources.sh` takes a list of URLs and produces a machine-checkable ledger: HTTP status, final URL after redirects, content length, and SHA-256 of the body — so a citation's verification is reproducible months later, when a page may have changed or vanished.

```bash
scripts/verify_sources.sh docs/source_urls.txt > docs/evidence/source_ledger_$(date +%F).txt
```

Two honest notes. (1) `openaccess.thecvf.com` may return 403 to bot-blocked clients. The script sets a browser User-Agent and got a 200 when this was written, but treat a block as expected rather than a bug — if it fails, cite the arXiv page as the fetchable primary and name the venue separately. (2) A changed content hash is *information*, not an error: it means the source moved, and any claim resting on it needs re-reading. That is the point.

---

## 7. The synthesis contract

When you write up research, every adopted method must arrive with four things. If any is missing, the method is not ready to enter the design:

1. **What it detects** (attack classes, in a named taxonomy — NIST AI 100-2e2025 or BadDet).
2. **What access it needs**, and what happens when that access is absent.
3. **The limitation the authors admit**, copied from their own limitations section or abstract.
4. **The calibration plan** — which battery slice gives it a null distribution, and therefore a p-value.

A method that cannot answer (4) is a detector, not an assurance component, and belongs in the *baseline-to-beat* column rather than in the pipeline. That distinction is the whole architecture in one line.
