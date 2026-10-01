# V4 merge verification (post-merge addendum to `docs/V4_MERGE_REPORT.md`)

Line: `origin/main = 63c6ca9` (merge commit, parents `c17141f` + `8c6bc3b`).
Everything below is measured on this checkout; no number is carried over from the package text.

## Step 1 — patched base and the suite deviation

**Deviation on file location.** The three patch files are **not** in the repository root. `ls
CVIAF_V4_*.txt` in the repo returns nothing; the only copies are
`~/Downloads/CVIAF_V4_CONSOLIDATED.txt` (sha256 `4ec997e1678cfa2f`),
`~/Downloads/CVIAF_V4_LABELGATE_FOLLOWUP.txt` (sha256 `140dec374b82aab0`) and
`~/Downloads/CVIAF_V4_APPENDIX_DE.txt` (read only, not applied). The consolidated patch and the
label-gate follow-up both applied clean (`git apply`, no rejects) to a fresh `c17141f` clone.

**Deviation on the expected suite count.** Expected 152 passed / 1 xfail; **measured 153 passed /
1 xfailed**. The +1 is not a failure: the follow-up patch adds 4 tests
(`tests/test_label_gate_cli.py` 1 + `tests/test_label_gate_protocol.py` 3) on top of the
consolidated package's own recorded **149**, and 149 + 4 = 153. The package's status doc
(`docs/CONSOLIDATED_V4_STATUS.md`) records 149, so the "152" in the handover is one short of the
package's own arithmetic.

**Environment deviation (cause of an earlier apparent 21-test failure set).** Without
`cryptography`, the suite cannot collect `tests/test_attestation.py` at all and reports
`21 failed, 123 passed, 2 skipped, 1 xfailed`; every failure is
`RuntimeError: Ed25519 signing is unavailable …` from `cviaf/provenance/__init__.py:217`. With
`cryptography` installed in `.venv` the whole suite runs green.

| tree | command | result |
|---|---|---|
| patched base (`c17141f` + both patches), `cryptography` present | `pytest tests/ -q` | **153 passed, 1 xfailed** |
| patched base, `cryptography` absent | `pytest tests/ -q` | 21 failed, 123 passed, 2 skipped, 1 xfailed |
| merged (`63c6ca9` + verification docs), `cryptography` present | `pytest tests/ -q` | **164 passed, 1 xfailed, 0 skipped** |

## Step 2 — conflicts

**Zero textual conflicts.** `docs/LAB_EDIT_INVENTORY.md` predicted three (poison `AttackSpec`,
`cli.main()` subparsers ×2) and none materialised: all three hunks live in my *uncommitted* lab
edits, which are correctly excluded from this merge, so the merge was pure addition.

The predicted **semantic overlap did hit**, and was resolved as instructed: `AttackSpec` is the
**union** — my `mechanism` field (appended last, so no positional argument shifts) plus v4's
`scope`. Before the union the patched tree could not load my own committed corpora
(`TypeError: AttackSpec.__init__() got an unexpected keyword argument 'mechanism'`); after it they
load, and that is verified live rather than asserted: `scripts/asset_rule_n50.py` rebuilds its
probe spec from a **pre-merge** `runs/clean_null` manifest, and its probe line reports the stamp
delta `0.8021` instead of raising.

## Step 3 — the two mandated signal changes

* **FTC removed from the merged scored set:** `evaluate.DETECTOR_NAMES = ("ctc", "refdiv")`.
  Evidence: conditional AUROC 0.5431 at TPR@5FPR 0.0000 against a clean-peer floor of **0.0038**
  (below chance) on my stamp-free cell, and 0/3 asset decisions everywhere.
* **Asset-rule gate moved from ctc to refdiv:** `model_asset_rule.DEFAULT_SIGNALS = ("refdiv",)`,
  `DIAGNOSTIC_SIGNALS = ("ctc",)` — ctc is still computed and reported, tagged
  `diagnostic_only: True`, and is never multiplied into the verdict. Evidence: my measured ctc
  null is non-exchangeable across seed ranges (KS p = .0040 vs day1, Spearman ρ = −0.290,
  p = .0087 over 81 clean models).
* Capacity consequence: the clean-model requirement falls **44 → 24**, so the n=50 corpus's
  conformal p-floor is **1/46 = 2.17%** instead of 1/7 = 14.3% at n=11.

`FUSED_DETECTORS` was **not** touched: it is absent from the v4 patch entirely (zero matches in
the consolidated file, no `evaluate.py` diff), so the exclusion is scoped to the asset rule's
scored set, exactly as written in the merge message.

## Step 4 — measurements the n=50 corpus unlocked

Source: `runs/asset_rule_n50.json` (runtime 28.4 s), α = .05, one shared reference model
(`runs/mvp/clean_none_fixed_s5`) and one shared probe set (24 eval images) for **every** model, so
responses are comparable across populations. Unit: paired stamp response = mean change in the
gate signal when the predeclared probe is stamped.

| population | n | rejected on the refdiv gate | rate / CI |
|---|---:|---:|---|
| clean null, leave-one-out (each model vs the other 49) | 50 | **5** | FPR 0.100, CP 95% [0.033, 0.192] |
| clean null, same set under the legacy ctc+refdiv gate | 50 | 5 | 0.100 — the gate change is not what moves this |
| day1 attacked, real stamp attacks | 72 | **13** | 0.181, Fisher vs the clean rate two-sided **p = .301** |
| stamp-free arms (Task 2) | 3 | **2** | 2/3, exact CI [0.094, 0.906] |

By attack kind (n = 10–11 each; `oga` 3/11, `oda` 2/11, `dup_flood` 2/10, `gma` 2/10,
`label_flip` 2/10, `ood_insert` 1/10, `rma` 1/10; median refdiv p from 0.087 `oga` to 0.478
`rma`). At **n = 11 the rule abstains outright**: `insufficient independent clean-model
calibration assets; image count cannot lower the asset p-value floor`.

Honest reading, stated because the headline numbers invite over-reading:

* The rule's own clean false-positive rate at n=50 is **10%**, twice its nominal 5% — the exact
  binomial P(X ≥ 5 | n = 50, p = .05) = **.1036**, so 10% is not separable from nominal at this
  n. The capacity win is real but the calibration is not tight.
* 13/72 attacked is above nominal 5% (p = 0.0001 against .05) yet **not distinguishable from the
  rule's own clean rate** (p = .301). It convicts nothing yesterday's rule could not.
* The stamp-free result is the interesting one and it is **not** a detection win. Two of the
  three arms land exactly on the p-floor (p = 1/46 = .0217) and reject, including
  `stampfree_scale_s101` whose own measured net ASR is only **0.111** (6/18 present objects
  moved, absent 0.132); the weakest arm (net ASR 0.056) is not rejected (p = .087). Given the
  rule's 10% clean FPR, P(X ≥ 2 | n = 3, p = .10) = .028 — so "2/3 stamp-free arms rejected" is
  consistent with the rule firing on its own noise at this n, and I am not claiming otherwise.
  n = 3 cannot separate those hypotheses.
* The two asset procedures **disagree** and I am flagging rather than resolving it:
  `cviaf/lab/null_suite.py`'s own asset fusion abstains **0/3** on the same three arms, while
  `model_asset_rule` rejects **2/3**. They use different statistics (the suite's fusion over its
  signal columns vs the rule's conformal rank test on the paired stamp response). For the Task 4
  record this is an open inconsistency for the owning lane.

### Stamp-free cell re-run on the merged tree, **no shim**

`python -m cviaf.lab.null_suite --corpus runs/stampfree --attacks stampfree --seeds 100 101 102`
from `63c6ca9` now runs **without the `mechanism` compatibility shim** the pre-merge clone needed
— the union fix removed the requirement. Artifact: `runs/stampfree/null_suite_merged.json`
(3 assets, 24 eval / 32 calibration images per seed/attack, FTC stride 16). Numbers are identical
to the pre-merge run in `runs/stampfree/null_suite.json`:

| contrast | signal | AUROC | TPR@5FPR | peer-clean floor |
|---|---|---:|---:|---:|
| `stamp_null` (stamp present vs absent) | all six columns incl. `fft` | **0.5000** | ≤ .042 | — |
| `backdoor_conditional_unstamped` (= `_stamped` by construction) | refdiv | **0.6730** | 0.000 | 0.3359 |
| " | ctc | 0.4436 | 0.105 | 0.5041 |
| " | ftc | 0.5431 | 0.000 | 0.0038 |
| " | fft (image only) | 0.5000 | 0.042 | 0.5000 |

The stamp contrast is exactly chance in every column, so the cell is genuinely
unstamped-backdoored; refdiv separates it above the clean-peer floor while ctc sits below chance.
Asset decisions inside the suite are **0/3 rejected under both fusions** (it abstains).

## Step 5 — final suite and commit

Final suite on the merged line with `cryptography` present: **164 passed, 1 xfailed, 0 skipped**.
The merge is **one commit** (`63c6ca9`, parents `c17141f` and `8c6bc3b`) carrying both conflict
resolutions and both signal decisions in its message; it is not squashed.

Decisions taken that were not on the list, disclosed:

1. The patch files were read from `~/Downloads` because the repo root has none (deviation above).
2. Post-merge verification lands as a **separate commit** rather than amending the merge: the
   merge commit is already pushed, and this addendum's measurements did not exist when it was
   written. Amending a published merge would rewrite `main` a second time for no gain.
3. `runs/stampfree/null_suite.json` is kept alongside the new `null_suite_merged.json` so both
   the shimmed pre-merge run and the shim-free merged run stay auditable.
4. Local `main` is still at `8c6bc3b` (pre-merge) because this checkout holds uncommitted lab
   edits to files the merge also modifies; `origin/main` is the merged line. Fast-forwarding the
   working tree would require committing or discarding those edits, which Task 4 forbids.
5. `~/Downloads/CVIAF_V4_CALIBRATED_ASSURE.txt` (written 19:53) exists and `git apply --check`
   against `63c6ca9` reports it **applies clean** — ready for the next handoff, not applied here.
