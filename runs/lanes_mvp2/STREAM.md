# Lane stream — one ranked list of what is actually established

Generated 2026-09-28T10:02:36Z from 4 lane(s) in `runs/lanes_mvp2`.

A line here is only in the first list if it carries a measured number. Everything else is a hypothesis, and the gaps are kept below as gaps — an untested cell is not a passed cell.

## Lanes

- **corpus_eval** (ok, 327.213s) — Headline corpus table (blocked until the corpus finishes training)
- **prenms_power** (ok, 12.642s) — Pre-NMS signal: how many images before it can carry a decision
- **redteam** (ok, 11.786s) — A12: dilution, trigger geometry, evasion budget
- **trace_foreground** (ok, 10.071s) — TRACE foreground arm vs the 0.816 stamp null

## Established (has a number)

- **a12.dilution** = `0.0000` — power of the BY item-level decision under dilution, reported at the ratio where the attacker's leverage is greatest
  - context: no flags at attacked_fraction=0.50, the largest of 5 ratios that blinds the decision (alpha=0.05)
  - source: `cviaf.lab.redteam.dilution_sweep` (lane `redteam`)
- **a12.evasion.ctc** = `0.5332` — ctc: rank separation (AUROC) between clean and attacked readings, with the clean-max threshold-saturation check applied
  - context: 0/198 above clean_max=1.0000 (degenerate=True); 0 above clean_p95=1.0000; threshold_used=clean_p95
  - source: `cviaf.lab.redteam.evasion_budget` (lane `redteam`)
- **a12.evasion.fg** = `0.3792` — fg: rank separation (AUROC) between clean and attacked readings, with the clean-max threshold-saturation check applied
  - context: 12/198 above clean_max=1.3051 (degenerate=False); 22 above clean_p95=1.0824; threshold_used=clean_max
  - source: `cviaf.lab.redteam.evasion_budget` (lane `redteam`)
- **a12.trigger_geometry.ctc_null** = `0.8545` — stamping a clean model's own images moves the CTC null as a function of trigger geometry; the response is that detector's stamp contamination, not a backdoor
  - context: null AUROC over 20 geometry settings (size 4-12, fixed/on_object, opacity 1.0/0.5): min 0.5746, mean 0.8545, max 0.9100; 0.500 is the target
  - source: `cviaf.lab.redteam.trigger_geometry_sweep` (lane `redteam`)
- **corpus_eval.asset.paired_whitened** = `0.8758` — asset-axis paired_whitened: AUROC and the achievable clean-max threshold (the deployable number is 'above', not the AUROC)
  - context: TPR@5FPR 0.842; clean_max=23.0693; 1/120 attacked above it; n_clean=20
  - source: `cviaf.lab.evaluate.asset_level_detectors` (lane `corpus_eval`)
- **corpus_eval.asset.paired_z** = `0.8279` — asset-axis paired_z: AUROC and the achievable clean-max threshold (the deployable number is 'above', not the AUROC)
  - context: TPR@5FPR 0.600; clean_max=17.2856; 0/120 attacked above it; n_clean=20
  - source: `cviaf.lab.evaluate.asset_level_detectors` (lane `corpus_eval`)
- **corpus_eval.asset.pre_nms_class_js** = `0.7446` — asset-axis pre_nms_class_js: AUROC and the achievable clean-max threshold (the deployable number is 'above', not the AUROC)
  - context: TPR@5FPR 0.417; clean_max=0.0831; 47/120 attacked above it; n_clean=20
  - source: `cviaf.lab.evaluate.asset_level_detectors` (lane `corpus_eval`)
- **corpus_eval.asset.unpaired** = `0.9163` — asset-axis unpaired: AUROC and the achievable clean-max threshold (the deployable number is 'above', not the AUROC)
  - context: TPR@5FPR 0.792; clean_max=6.7493; 88/120 attacked above it; n_clean=20
  - source: `cviaf.lab.evaluate.asset_level_detectors` (lane `corpus_eval`)
- **corpus_eval.asset.weight_null_statistics_violated** = `0.7500` — asset-axis weight_null_statistics_violated: AUROC and the achievable clean-max threshold (the deployable number is 'above', not the AUROC)
  - context: TPR@5FPR 0.750; clean_max=0.0000; 60/120 attacked above it; n_clean=20
  - source: `cviaf.lab.evaluate.asset_level_detectors` (lane `corpus_eval`)
- **corpus_eval.asset.weight_z_max** = `0.8315` — asset-axis weight_z_max: AUROC and the achievable clean-max threshold (the deployable number is 'above', not the AUROC)
  - context: TPR@5FPR 0.700; clean_max=2.7454; 82/120 attacked above it; n_clean=20
  - source: `cviaf.lab.evaluate.asset_level_detectors` (lane `corpus_eval`)
- **corpus_eval.asset.weight_z_mean** = `0.8283` — asset-axis weight_z_mean: AUROC and the achievable clean-max threshold (the deployable number is 'above', not the AUROC)
  - context: TPR@5FPR 0.692; clean_max=1.3119; 81/120 attacked above it; n_clean=20
  - source: `cviaf.lab.evaluate.asset_level_detectors` (lane `corpus_eval`)
- **corpus_eval.asset.weight_z_rms** = `0.8287` — asset-axis weight_z_rms: AUROC and the achievable clean-max threshold (the deployable number is 'above', not the AUROC)
  - context: TPR@5FPR 0.692; clean_max=1.5402; 82/120 attacked above it; n_clean=20
  - source: `cviaf.lab.evaluate.asset_level_detectors` (lane `corpus_eval`)
- **corpus_eval.null.ctc** = `0.5913` — ctc on a trigger-stamped CLEAN model (the falsifiable null)
  - context: spread 0.0827 over 20 clean models, stamped with oga: mild stamp sensitivity
  - source: `cviaf.lab.evaluate.null_control` (lane `corpus_eval`)
- **corpus_eval.null.fg** = `0.3781` — fg on a trigger-stamped CLEAN model (the falsifiable null)
  - context: spread 0.1364 over 20 clean models, stamped with oga: measures the stamp, not the backdoor (INVERTED direction)
  - source: `cviaf.lab.evaluate.null_control` (lane `corpus_eval`)
- **corpus_eval.null.ftc** = `0.6500` — ftc on a trigger-stamped CLEAN model (the falsifiable null)
  - context: spread 0.1170 over 20 clean models, stamped with oga: measures the stamp, not the backdoor
  - source: `cviaf.lab.evaluate.null_control` (lane `corpus_eval`)
- **corpus_eval.null.refdiv** = `0.7517` — refdiv on a trigger-stamped CLEAN model (the falsifiable null)
  - context: spread 0.1025 over 20 clean models, stamped with oga: measures the stamp, not the backdoor
  - source: `cviaf.lab.evaluate.null_control` (lane `corpus_eval`)
- **corpus_eval.scored.gma** = `0` — scored models for gma after the ASR/effect gate
  - context: 0/3 of gma scored; mean ASR 0.222; excluded as no-effect 3
  - source: `cviaf.lab.evaluate._summarise` (lane `corpus_eval`)
- **corpus_eval.scored.oda** = `0` — scored models for oda after the ASR/effect gate
  - context: 0/3 of oda scored; mean ASR 0.027; excluded as no-effect 3
  - source: `cviaf.lab.evaluate._summarise` (lane `corpus_eval`)
- **corpus_eval.scored.oga** = `0` — scored models for oga after the ASR/effect gate
  - context: 0/3 of oga scored; mean ASR 0.412; excluded as no-effect 3
  - source: `cviaf.lab.evaluate._summarise` (lane `corpus_eval`)
- **corpus_eval.scored.rma** = `0` — scored models for rma after the ASR/effect gate
  - context: 0/3 of rma scored; mean ASR 0.000; excluded as no-effect 3
  - source: `cviaf.lab.evaluate._summarise` (lane `corpus_eval`)
- **corpus_eval.scored.substitution** = `3` — scored models for substitution after the ASR/effect gate
  - context: 3/3 of substitution scored; mean ASR 0.000; excluded as no-effect 0
  - source: `cviaf.lab.evaluate._summarise` (lane `corpus_eval`)
- **corpus_eval.scored.weight_tamper** = `3` — scored models for weight_tamper after the ASR/effect gate
  - context: 3/3 of weight_tamper scored; mean ASR 0.000; excluded as no-effect 0
  - source: `cviaf.lab.evaluate._summarise` (lane `corpus_eval`)
- **prenms_power.required_n** = `40` — Measured effect 0.0956 is cleared at n=40 images (null p95 below the effect). Below that sample size the null and the effect overlap and the signal must not be reported as a detection.
  - context: null p95 at n=10/20/40/80: 10:0.1188, 20:0.1050, 40:0.0780, 80:0.0844
  - source: `cviaf.lab.power.sample_size_calibration` (lane `prenms_power`)
- **trace_foreground.stamp_null.ctc** = `0.6220` — the background (CTC) arm re-measured on a trigger-stamped CLEAN model: the corpus's own stamp contamination for that detector
  - context: stamped with oga, n=50; 0.500 = ignores the stamp
  - source: `cviaf.lab.arms.lane_trace_foreground` (lane `trace_foreground`)
- **trace_foreground.stamp_null.fg** = `0.1392` — the foreground (focal) arm on the same stamped clean model; its DECLARED sign is high=suspicious, and the measured value says whether that sign holds
  - context: fg null AUROC 0.1392 vs ctc null AUROC 0.6220 (stamped with oga, n=50)
  - source: `cviaf.lab.arms.lane_trace_foreground` (lane `trace_foreground`)

## Hypotheses (no number yet)

_none_

## Open gaps (NOT ASSESSED)

- `[high]` **a12.dilution.blind_ratio** — at attacked_fraction=0.50 the detector flags nothing, so an attacker who floods clean items hides here
  - next: increase the calibration set / batch size, or decide at the asset level rather than the item level
- `[high]` **a12.evasion.ctc.evadable** — ctc: every attacked reading is already inside the clean null; the signal does not separate at this sample size
  - next: report NOT ASSESSED for this signal, or calibrate sample size (see cviaf.lab.power)
- `[high]` **corpus_eval.no_effect.gma** — all 3 gma models were excluded (mean ASR 0.222): the attack never implanted, so this cell is unmeasured, not clean
  - next: find a recipe that implants (see the research harvest queue) or mark the cell 'not covered' in the coverage statement
- `[high]` **corpus_eval.no_effect.oda** — all 3 oda models were excluded (mean ASR 0.027): the attack never implanted, so this cell is unmeasured, not clean
  - next: find a recipe that implants (see the research harvest queue) or mark the cell 'not covered' in the coverage statement
- `[high]` **corpus_eval.no_effect.oga** — all 3 oga models were excluded (mean ASR 0.412): the attack never implanted, so this cell is unmeasured, not clean
  - next: find a recipe that implants (see the research harvest queue) or mark the cell 'not covered' in the coverage statement
- `[high]` **corpus_eval.no_effect.rma** — all 3 rma models were excluded (mean ASR 0.000): the attack never implanted, so this cell is unmeasured, not clean
  - next: find a recipe that implants (see the research harvest queue) or mark the cell 'not covered' in the coverage statement
- `[high]` **corpus_eval.null.fg.contaminated** — fg reads 0.378 against a stamped clean model -- it is measuring the stamp, and its attacked-row numbers are inflated by that much
  - next: placebo-control before publishing; see docs/EXPERIMENT_LANES.md
- `[high]` **corpus_eval.null.ftc.contaminated** — ftc reads 0.650 against a stamped clean model -- it is measuring the stamp, and its attacked-row numbers are inflated by that much
  - next: placebo-control before publishing; see docs/EXPERIMENT_LANES.md
- `[high]` **corpus_eval.null.refdiv.contaminated** — refdiv reads 0.752 against a stamped clean model -- it is measuring the stamp, and its attacked-row numbers are inflated by that much
  - next: placebo-control before publishing; see docs/EXPERIMENT_LANES.md
- `[high]` **trace_foreground.ctc_reads_stamp** — the background arm reads 0.622 on a trigger-stamped clean model, so its attacked-row numbers are inflated by that much
  - next: subtract the measured null or mark the affected rows contaminated
- `[high]` **trace_foreground.fg_sign_unsupported** — the foreground arm reads 0.139, BELOW 0.500: the stamp makes objects MORE stable under focal change, so the declared sign (high=suspicious) is not supported
  - next: calibrate the sign on a stamped clean model before publishing any detection from this arm, or invert it and re-measure
- `[medium]` **corpus_eval.asset.paired_whitened.auroc_not_deployable** — paired_whitened reads AUROC 0.876 but only 1/120 attacked models clear the achievable clean-max threshold 23.069: the rank separation is real, the operating point is not usable
  - next: report AUROC and deployment separately; do not quote the AUROC alone
- `[low]` **trace_foreground.no_trigger_skipped** — 200 attack(s) with no test-time trigger (clean_label, dup_flood, label_flip, ood_insert, substitution, weight_tamper) are NOT ASSESSED by this lane -- a stamp cannot express them
  - next: assess data-only kinds on the data axis and artifact-level kinds (weight_tamper, substitution) on the asset axis
