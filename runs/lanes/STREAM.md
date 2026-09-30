# Lane stream — one ranked list of what is actually established

Generated 2026-09-28T09:41:16Z from 4 lane(s) in `runs/lanes`.

A line here is only in the first list if it carries a measured number. Everything else is a hypothesis, and the gaps are kept below as gaps — an untested cell is not a passed cell.

## Lanes

- **corpus_eval** (blocked, 0.01s) — Headline corpus table (blocked until the corpus finishes training)
- **prenms_power** (ok, 11.465s) — Pre-NMS signal: how many images before it can carry a decision
- **redteam** (ok, 15.107s) — A12: dilution, trigger geometry, evasion budget
- **trace_foreground** (ok, 13.517s) — TRACE foreground arm vs the 0.816 stamp null

## Established (has a number)

- **a12.dilution** = `0.0000` — power of the BY item-level decision under dilution, reported at the ratio where the attacker's leverage is greatest
  - context: no flags at attacked_fraction=0.50, the largest of 5 ratios that blinds the decision (alpha=0.05)
  - source: `cviaf.lab.redteam.dilution_sweep` (lane `redteam`)
- **a12.evasion.ctc** = `0.7495` — ctc: rank separation (AUROC) between clean and attacked readings, with the clean-max threshold-saturation check applied
  - context: 73/233 above clean_max=1.0000 (degenerate=False); 119 above clean_p95=1.0000; threshold_used=clean_max
  - source: `cviaf.lab.redteam.evasion_budget` (lane `redteam`)
- **a12.evasion.fg** = `0.4569` — fg: rank separation (AUROC) between clean and attacked readings, with the clean-max threshold-saturation check applied
  - context: 5/232 above clean_max=1.5982 (degenerate=False); 22 above clean_p95=1.3658; threshold_used=clean_max
  - source: `cviaf.lab.redteam.evasion_budget` (lane `redteam`)
- **a12.trigger_geometry.ctc_null** = `0.7407` — stamping a clean model's own images moves the CTC null as a function of trigger geometry; the response is that detector's stamp contamination, not a backdoor
  - context: null AUROC over 20 geometry settings (size 4-12, fixed/on_object, opacity 1.0/0.5): min 0.5000, mean 0.7407, max 1.0000; 0.500 is the target
  - source: `cviaf.lab.redteam.trigger_geometry_sweep` (lane `redteam`)
- **prenms_power.required_n** = `20` — Measured effect 0.1386 is cleared at n=20 images (null p95 below the effect). Below that sample size the null and the effect overlap and the signal must not be reported as a detection.
  - context: null p95 at n=10/20/40/80: 10:0.1509, 20:0.1338, 40:0.1166, 80:0.0971
  - source: `cviaf.lab.power.sample_size_calibration` (lane `prenms_power`)
- **trace_foreground.stamp_null.ctc** = `0.9208` — the background (CTC) arm re-measured on a trigger-stamped CLEAN model: the corpus's own stamp contamination for that detector
  - context: stamped with gma, n=60; 0.500 = ignores the stamp
  - source: `cviaf.lab.arms.lane_trace_foreground` (lane `trace_foreground`)
- **trace_foreground.stamp_null.fg** = `0.3391` — the foreground (focal) arm on the same stamped clean model; its DECLARED sign is high=suspicious, and the measured value says whether that sign holds
  - context: fg null AUROC 0.3391 vs ctc null AUROC 0.9208 (stamped with gma, n=60)
  - source: `cviaf.lab.arms.lane_trace_foreground` (lane `trace_foreground`)

## Hypotheses (no number yet)

_none_

## Open gaps (NOT ASSESSED)

- `[high]` **a12.dilution.blind_ratio** — at attacked_fraction=0.50 the detector flags nothing, so an attacker who floods clean items hides here
  - next: increase the calibration set / batch size, or decide at the asset level rather than the item level
- `[high]` **corpus_eval.blocked** — corpus incomplete (227/300); the asset-axis TPR@FPR table cannot be filled from a partial corpus
  - next: wait for the corpus job to finish, then run: python -m cviaf.lab eval --corpus runs/mvp2 --json runs/mvp2/eval.json
- `[high]` **trace_foreground.ctc_reads_stamp** — the background arm reads 0.921 on a trigger-stamped clean model, so its attacked-row numbers are inflated by that much
  - next: subtract the measured null or mark the affected rows contaminated
- `[high]` **trace_foreground.fg_sign_unsupported** — the foreground arm reads 0.339, BELOW 0.500: the stamp makes objects MORE stable under focal change, so the declared sign (high=suspicious) is not supported
  - next: calibrate the sign on a stamped clean model before publishing any detection from this arm, or invert it and re-measure
- `[low]` **trace_foreground.data_only_skipped** — 30 data-only attack(s) (dup_flood, label_flip, ood_insert) have no test-time trigger and are NOT ASSESSED by this lane
  - next: assess data attacks on the data axis, not with a stamp
