# Model-integrity null-control verdict (synthetic day1 corpus)

Run `python3 -m cviaf.lab.null_suite --corpus runs/day1 --out results/null_suite_day1_s5-s7.json --seeds 5 6 7 --attacks oga oda rma --n-eval 24 --n-cal 32 --stride 16` at base `c17141f0f58f4a5f6f0afce8e0f5719c61fcdd46`. The committed JSON records all per-seed values, the model IDs, counts, held-out behavioral checks, and contrast definitions. No training or model weights are altered. Clean and attacked models are paired by seed and share the same test images and exact attack stamp. A separate clean reference at seed + 6 serves both model arms; another clean peer at seed + 3 supplies a trained-model negative control. Calibration uses distinct contributor-disjoint images, stamped on a clean model. Scores are higher-is-suspicious; fusion is Bonferroni over signal p-values and Cauchy across image p-values. The report's TPR@5% FPR uses whole-score thresholds with tied scores kept together. Each cell has 24 test images, 32 calibration images, three seeds per recipe. These are a small synthetic diagnostic, not a deployment benchmark.

## Results (mean across three seeds; AUROC / attainable TPR at 5% FPR)

| Recipe | Contrast / negative arm | CTC | refdiv | FTC | FFT | fused with FTC | fused without FTC |
|---|---|---:|---:|---:|---:|---:|---:|
| OGA | stamp null: clean bare -> clean stamped | .921/.347 | .611/.042 | .500/.028 | 1.000/1.000 | .554/.028 | .622/.056 |
| OGA | backdoor conditional: clean stamped -> backdoored stamped | .506/.333 | .506/.014 | .474/.014 | .500/.042 | .582/.000 | .611/.347 |
| OGA | independent clean peer stamped vs clean stamped | .650/.333 | .552/.000 | .550/.097 | .500/.042 | .598/.000 | .676/.333 |
| ODA | stamp null: clean bare -> clean stamped | .863/.431 | .757/.125 | .635/.097 | .998/.986 | .556/.097 | .641/.097 |
| ODA | backdoor conditional: clean stamped -> backdoored stamped | .107/.000 | .532/.000 | .411/.000 | .500/.042 | .428/.000 | .466/.000 |
| ODA | independent clean peer stamped vs clean stamped | .452/.000 | .573/.000 | .549/.083 | .500/.042 | .521/.083 | .550/.000 |
| RMA | stamp null: clean bare -> clean stamped | .863/.431 | .757/.125 | .635/.097 | .998/.986 | .556/.097 | .641/.097 |
| RMA | backdoor conditional: clean stamped -> backdoored stamped | .268/.000 | .534/.000 | .407/.000 | .500/.042 | .441/.000 | .490/.000 |
| RMA | independent clean peer stamped vs clean stamped | .452/.000 | .573/.000 | .549/.083 | .500/.042 | .521/.083 | .550/.000 |

The FFT is a deliberately image-only high-frequency statistic: its 1.000 OGA stamp-null AUROC and .500 conditional AUROC demonstrate why an image artifact screen cannot claim model-backdoor specificity. CTC substantially responds to the stamp even on the clean model, yet has no robust positive conditional result. Refdiv is stamp-sensitive and its conditional AUROCs (.506-.534) do not exceed the clean-peer contrast convincingly. FTC shows no useful positive conditional result on these recipes; even its ODA score runs in the wrong direction. Neither fused variant detects a backdoor specifically at this operating point. Unstamped-backdoored is tested too in the JSON, but differences on bare inputs are ordinary model differences, not proof of a trigger.

## Asset decision and FTC

At alpha=.05, **0/3 stamped clean assets and 0/3 stamped attacked assets are rejected for each attack with FTC**, and exactly the same **0/3 and 0/3 without FTC**. Thus dropping FTC does not recover any asset-level power here. The apparent OGA conditional per-image TPR gain without FTC (.347 vs .000) is not a usable asset-level win: the independent-clean-peer null gets .333 without FTC, and no asset is rejected. **Recommendation: drop FTC from the default fusion for this synthetic configuration, but retain it as an optional diagnostic under test.** This is a no-benefit decision, not a claim that the two-signal fusion now detects backdoors. Do not ship either combination as a proven backdoor detector. A targeted FTC design on a different trigger/decoy recipe must re-pass the same controls before inclusion.

## Behavior independent of pixel statistics

Paired trigger-input attack success is recomputed on held-out images for *both* weights, using the same attack recipe and seed. Net ASR (backdoored minus clean) by seed 5/6/7: OGA **.000/1.000/.375**, ODA **.087/.037/.213**, RMA **.533/1.000/.623**. Raw ODA ASR looks high in both arms (clean .913/.900/.737; attacked 1.000/.938/.950), so these ODA models cannot be called behaviorally verified backdoors on these probes. Prediction-class-count flip rates (also stored in JSON) are generally high on both clean and attacked weights and are not discriminative. OGA seeds 5 and 7 fail or fall below the .50 net-effect floor despite old manifest ASR flags. RMA has the clearest measured net behavior, but the listed integrity signals fail to isolate it. The original manifest's raw ASR alone is a weak validity gate because the same stamp often changes clean-model predictions.

The fourth factorial cell (backdoored weights on unstamped inputs) exists and is measured. It is **not** an "unstamped backdoor trigger" or an independently trainable trigger-free implant. This convolutional synthetic backbone cannot establish a trigger-free conditional behavior merely by omitting the trigger at inference. No such cell is fabricated.

Limits: three seeds, 24 images per arm, fixed recipes, synthetic data, reference selection may influence refdiv; peer-clean arm measures only one direction of normal model variation. The empirical asset rejection counts do not validate nominal type-I error control. Next test should retrain low-null OGA/RMA recipes, use a larger independent clean-model pool and calibration, then test true held-out seeds and a learned trigger-specific behavioral probe before asserting detection.
