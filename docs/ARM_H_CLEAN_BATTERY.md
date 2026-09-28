# Arm H: synthetic clean reference synthesis

`python -m cviaf.lab.clean_battery --n-cal 120 --n-test 120 --seed 17 --out arm-h.json`

The lab enumerates each declared terrain x season x sensor cell; generates seeded,
independent clean calibration and test datasets for every cell; and reports counts,
digests, conditional conformal p-value diagnostics, and the fraction of clean test
items that would abstain under the proxy rule `p <= alpha`. The comparator is the
existing one-SceneSpec clean holdout, with the same calibration count per cell,
scored on exactly the same test datasets. Brightness is a fixed covariate score.

`conditional_pvalues` fails closed on missing, underfilled, or nonfinite calibration
cells. This is a lab primitive, not a direct replacement for operational AB-1:
a clean synthetic generator does not verify real-world data provenance, the
scene labels, contributor independence, or exchangeability under deployment
shift. In particular, do not turn a synthetic p-value into a quarantine verdict.
A real battery must additionally bind scene definitions, seeds, assets and
ownership to a signed manifest, and measure score-specific detector calibration
on trusted, contributor-disjoint operational references. Missing cells cannot be
silently pooled.

The generator is the repo's procedural 64 x 64 image model. Sensor labels map
to explicit `(noise sigma, blur radius)` pairs in `clean_battery.SENSORS`; these
are not actual camera models. Reported coverage means generated cell coverage,
not coverage of the operational domain. Per-cell results matter: pooled metrics
can hide poor conditional behavior. The finite n rank grid and test-sample
uncertainty mean no single observed rate is a mathematical guarantee.
