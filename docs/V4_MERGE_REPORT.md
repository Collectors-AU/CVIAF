# V4 merge report (Task 4)

## Provenance and application

| step | result |
|---|---|
| base | `c17141f` ("v3"), fresh clone |
| `CVIAF_V4_CONSOLIDATED.txt` | applied clean — `git apply`, no fuzz, no rejects |
| `CVIAF_V4_LABELGATE_FOLLOWUP.txt` | applied clean |
| `CVIAF_V4_APPENDIX_DE.txt` | **not applied** (documentation only, read) |
| merge of my pushed work | `8c6bc3b` merged on top; **0 textual conflicts** |

**Deviation 1 — the patch files are not in the repository root.** `ls CVIAF_V4_*.txt` in the
repo root returns nothing; the only copies on this machine are in `~/Downloads/`
(`CVIAF_V4_CONSOLIDATED.txt` sha256 prefix `4ec997e1678cfa2f`, `CVIAF_V4_LABELGATE_FOLLOWUP.txt`
`140dec374b82aab0`). Those are the files used. If a newer package was dropped elsewhere, this
report describes the older one.

**Deviation 2 — "152 passed / 1 xfail" is not reproducible on this virtualenv, and the cause is
one missing dependency.** Verbatim:

```
__________________ ERROR collecting tests/test_attestation.py __________________
tests/test_attestation.py:10: in <module>
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
E   ModuleNotFoundError: No module named 'cryptography'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.87s
```

With that module ignored (`--ignore=tests/test_attestation.py`), the patched base gives
**21 failed, 123 passed, 2 skipped, 1 xfailed in 26.52s**, and the failures are the same missing
dependency, not logic:

```
>                   raise RuntimeError(_SYMMETRIC_REFUSAL) from e
E                   RuntimeError: Ed25519 signing is unavailable (the 'cryptography' package is not
installed, or an HMAC key was loaded). Provenance refuses to sign or load symmetric keys silently:
install cviaf[full], or pass allow_symmetric=True (CLI: --allow-symmetric). Symmetric mode gives
tamper-evidence only - anyone holding the verification key can forge a seal - and every seal and
report is tagged accordingly.
cviaf/provenance/__init__.py:217: RuntimeError
```

Failure set: `test_provenance_matrix.py` 18, `test_fail_closed.py` 1, `test_provenance_hardening.py` 1,
`test_v4_live.py` 1 — identical before and after the merge. The repo README already documents this
venv state ("the current `.venv` is in this state because `cryptography` is not installed"), so it is
environmental. **Installing `cryptography` is the one thing needed to reproduce the published
152/1-xfail; I did not install it (no installs without approval).**

## Conflicts

**Zero textual conflicts.** My inventory predicted three (B1 `poison.py` AttackSpec region,
B2/B3 `cli.main()` subparsers). They did not materialize because all three came from my
*uncommitted* lab edits — `poison.py`, `cli.py`, `evaluate.py`, `train.py`, `corpus.py` — which are
deliberately excluded from this merge and remain in the working tree of the primary checkout. The
push contains only new files plus `STATE.md`/docs, so the merge is pure addition. What did
materialize is the semantic overlap the inventory flagged as C1.

**C1 resolved as instructed (union).** `AttackSpec` now carries both fields, `mechanism` (this
tree's model-attack mechanism) and `scope` (v4's poison-scope), with `mechanism` appended at the end
so no existing positional argument shifts. Before the union, the patched tree could not load this
repo's own committed corpora: `TypeError: AttackSpec.__init__() got an unexpected keyword argument
'mechanism'` from `evaluate.train_spec_from_manifest`. After it, the pre-merge manifests load —
verified live, because `scripts/asset_rule_n50.py` rebuilds its probe spec from
`runs/clean_null/clean_none_fixed_s100/manifest.json`, a manifest written before the merge.

Note for the coordinator: the mirror case still exists for the *uncommitted* side. When those lab
edits land, `evaluate.py` (41 hunks), `train.py` (20), `corpus.py` (20), `poison.py` (15), `cli.py`
(7) and `detectors.py` (3) all merge against a patched base; `README.md` and `cli.py` are the two
that the package also edits.

## Mandated changes

**FTC removed from the merged scored detector set.** `evaluate.DETECTOR_NAMES` is now
`("ctc", "refdiv")`. The removal is documented in place with the measurements behind it: v4's own
`model_asset_rule` carried "FTC failed factorial controls; optional only", and on the stamp-free
factorial cell ftc's conditional AUROC was 0.5431 at TPR@5FPR 0.0000 while its **clean-peer floor**
was 0.0038 — below chance, meaning its ordering on like-versus-like pairs is noise — with 0/3 asset
rejections in all five cells under both fusions. `trace_ftc` stays implemented and importable; it is
simply no longer part of a verdict. The v4 null suite's own `SIGNALS` tuple is left untouched (that
is the suite's measurement grid, not the evaluate verdict set).

**Asset-rule gate switched from ctc to refdiv, ctc kept as a diagnostic column.**
`model_asset_rule.DEFAULT_SIGNALS = ("refdiv",)`, `DIAGNOSTIC_SIGNALS = ("ctc",)`,
`SIGNAL_COLUMNS = ("refdiv", "ctc")`; `decide_model_asset` reports ctc under
`diagnostic_per_signal` with `diagnostic_only: True` and never multiplies it into `asset_pvalue`.
Consequence, stated because it changes what the rule needs: m = 1, so the requirement drops from 44
clean models to **24**, and the conformal p-value floor at n = 50 becomes **1/46 = 2.17%**
(1/20 = 5% at the minimum sufficient n = 24). Tests updated: `tests/test_model_asset_rule.py` now
asserts the new default, that a ctc-only anomaly cannot reject, that a missing ctc column cannot
change a verdict, and keeps the legacy two-signal behaviour covered by passing `signals=("ctc",
"refdiv")` explicitly.

## Measurement 1 — asset rule at n = 50 (the Task 1 unlock)

`scripts/asset_rule_n50.py` → `runs/asset_rule_n50.json`. One shared reference model
(`runs/mvp/clean_none_fixed_s5`), one shared 24-image probe set (stamped with the oga patch recipe,
`max|pixel delta| = 0.8021`), 50 clean calibration assets, refdiv gate, alpha .05.

| population | n suspects | rejected | rate | note |
|---|---:|---:|---:|---|
| clean_null, leave-one-out (49 calibration each) | 50 | **5** | 0.100 | CP 95% [0.033, 0.218]; nominal 5% expects 2.5/50; binomial P(X≥5 \| p=.05) = 0.104 |
| day1 attacked (real stamp attacks, 7 kinds) | 72 | **13** | 0.181 | vs nominal 5%: P(X≥13 \| p=.05) = 0.0001 |
| stamp-free arms (Task 2) | 3 | **2** | — | see below |

Per attacked kind (rejected / n, median p): oga 3/11 (0.087), oda 2/11 (0.239), rma 1/10 (0.478),
dup_flood 2/10 (0.424), gma 2/10 (0.402), label_flip 2/10 (0.185), ood_insert 1/10 (0.304).

Two honest readings, both of them uncomfortable:

1. **Against its own clean null, the rule does not separate attacked from clean assets.**
   13/72 vs 5/50, Fisher exact p = 0.301. The attacked rate *is* above the nominal 5% (p = 0.0001),
   but it is not distinguishable from the rule's own measured false-alarm rate. With n = 50 clean
   models the rule can now *reject* — that was the unlock — and what it rejects looks like noise
   plus weak signal.
2. **The clean false-alarm rate is 10%, twice the nominal alpha** (5/50, CP [0.033, 0.218]).
   5/50 is compatible with 5% at this n (p = 0.104), so this neither confirms nor refutes the
   conformal guarantee — but the point estimate is double, and the capacity the rule bought is
   1/46 = 2.2%, so it is firing far above its own floor. The two-sided |value − clean median|
   statistic is the likely cause: a model whose stamp response is unusually *small* is rejected as
   readily as one whose response is large.

The pre-merge capacity is demonstrated in the same artifact: at n = 11 the rule returns
`abstain=True`, reason "insufficient independent clean-model calibration assets; image count cannot
lower the asset p-value floor" — it could not reject anything regardless of the evidence.

## Measurement 2 — the stamp-free cell against the same null

The three Task-2 arms produce refdiv responses of **+0.138, +0.115, +0.184** against a clean null of
**+0.2935 ± 0.0516** (range +0.147 to +0.375), and ctc responses of +0.160, +0.132, +0.062 against a
clean ctc null of +0.0828 ± 0.1129. Decisions: 2/3 rejected at p = 0.0217 (the floor), one at 0.087.

**This is not detection of a stamp-free backdoor.** The arms are rejected because they
*under-respond* to the stamp relative to the clean median — they sit 2–3.5 SD below it — and the
statistic is two-sided, so a depressed response is as "nonconforming" as an elevated one. Their
measured utility is correspondingly depressed (clean F1 0.37/0.24/0.35 against control
0.61/0.44/0.61), so the most economical reading is that the rule is flagging a *perturbed model*,
not a trigger. The signal that does see the stamp-free cell is the v4 null suite's conditional
contrast (`runs/stampfree/null_suite.json`: refdiv conditional AUROC 0.6730 against a clean-peer
floor of 0.336), which is a per-image ranking, not the asset rule's stamp-response deviation.

Consequence for the merged tree: the asset rule remains **conditional on a stamp by construction**,
and a stamp-free backdoor is invisible to it in the direction that matters. Nothing here should be
read as "the rule now catches stamp-free attacks".

## Suite counts

| tree | command | result |
|---|---|---|
| patched base (c17141f + consolidated + labelgate) | `pytest tests/ -q` | **1 collection error** — `ModuleNotFoundError: No module named 'cryptography'` (verbatim above) |
| patched base, `--ignore=tests/test_attestation.py` | `pytest tests/ -q --ignore=…` | **21 failed, 123 passed, 2 skipped, 1 xfailed** (26.52 s) |
| merged (patched base + my pushed work + mandated changes) | `pytest tests/ -q --ignore=…` | **21 failed, 134 passed, 2 skipped, 1 xfailed** (26.69 s) |

+11 passing: 6 `test_stampfree.py`, 4 `test_drift_cells.py`, and 1 net new in
`test_model_asset_rule.py`. The 21 failures are the same 21 in both runs and all trace to the missing
`cryptography` package.

## Decisions I had to make that were not in the instructions

1. **`stamp_response` keeps computing both columns by default.** The gate moved to refdiv, but the
   probe function still returns ctc (and refdiv) unless told otherwise, so existing callers keep
   their measurements and the diagnostic column is populated without a second probe pass. New
   constant `SIGNAL_COLUMNS = ("refdiv", "ctc")`.
2. **Diagnostics are fail-open.** A diagnostic column that is absent or non-finite is reported as
   `{"available": False}` and cannot change a verdict — a diagnostic must not be able to reject by
   accident, and it must not be able to hide a rejection either.
3. **The v4 null suite's `SIGNALS`/`FUSIONS` were left alone** (ftc removed from the *evaluate*
   verdict set only, per the instruction's wording). `runs/stampfree/null_suite.json` therefore
   still contains `with_ftc`/`without_ftc` columns; they are now historical measurements of a column
   that no longer gates anything.
4. **`scripts/asset_rule_n50.py` stores the 50 raw clean responses** in its artifact, so the
   n = 11-versus-n = 50 capacity claim can be re-derived from the JSON without rescoring 50 models.
5. **The merge was committed as a merge commit** (parents `c17141f` and `8c6bc3b`) rather than a
   squashed single-parent commit, so the v4 base lineage stays visible in history; the tree is the
   one commit requested.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q --ignore=tests/test_attestation.py
PYTHONPATH=. .venv/bin/python scripts/asset_rule_n50.py --out runs/asset_rule_n50.json
```
