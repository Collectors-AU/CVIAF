# Computer vision pipeline integrity: a technical reference

This document covers four areas of CV pipeline security: training data integrity, model integrity assessment, inference provenance, and distribution shift detection. Every method listed works offline without cloud APIs.

---

## 1. Training data integrity

### 1.1 Trigger-based backdoor detection in image datasets

#### Spectral Signatures (Tran et al., 2018)

Poisoned samples leave a detectable trace in the covariance spectrum of learned feature representations. The method computes the top singular vector of the feature covariance matrix for each class and scores every sample by its projection onto that vector. Outlier scores identify poisoned samples.

- **How it works:** Run a trained (or partially trained) model on the dataset. Extract penultimate-layer features per class. Compute SVD of the centered feature matrix. Poisoned samples cluster along the top singular direction because the trigger creates a consistent feature shift.
- **Libraries:** IBM [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) v1.17+ implements `SpectralSignatureDefense`. Manual implementation needs only numpy/scipy SVD.
- **Access:** White-box. Requires model internals to extract intermediate features.
- **Compute:** Low-to-moderate. One forward pass per sample, then SVD on an N x D matrix per class. Scales linearly with dataset size.
- **Limitations:** Assumes a single dominant trigger direction. Struggles with clean-label attacks where poisoned samples don't shift features as clearly. Degrades when the poison ratio is very low (<1%).

#### Activation Clustering (Chen et al., 2019)

Clusters the activation vectors of training samples within each class. Poisoned samples form a distinct cluster separate from clean samples because they activate different feature pathways.

- **How it works:** Extract activations from the model's penultimate layer for each class. Apply dimensionality reduction (PCA or ICA), then cluster with k-means (k=2) or DBSCAN. The smaller cluster likely contains poisoned samples. Evaluate with a silhouette score to judge separability.
- **Libraries:** ART implements `ActivationDefence`. Alternatively: scikit-learn (KMeans, DBSCAN, PCA) + PyTorch/TensorFlow for feature extraction.
- **Access:** White-box.
- **Compute:** Moderate. One forward pass per sample + clustering. The clustering itself is fast; the forward passes dominate.
- **Limitations:** Assumes clean and poisoned samples are separable in activation space, which fails for sophisticated attacks like WaNet or clean-label attacks. Needs at least a moderate poison ratio (~5%+) to form a detectable cluster.

#### STRIP (Gao et al., 2019)

STRong Intentional Perturbation. At inference time, overlays random clean images onto a suspect input and measures the entropy of the resulting predictions. Clean inputs produce high-entropy (varied) predictions under perturbation. Poisoned inputs with strong triggers produce low-entropy predictions because the trigger dominates regardless of the overlay.

- **How it works:** For each test input, blend it with N random clean images (typically N=100). Feed each blended image through the model. Compute the entropy of the output probability distribution. If the average entropy across perturbations falls below a threshold, flag the input as triggered.
- **Libraries:** ART implements `STRIPDefence`. Easy to implement from scratch: just alpha-blending + forward passes + scipy.stats.entropy.
- **Access:** Black-box inference access sufficient. Only needs to query the model's output probabilities.
- **Compute:** Moderate to high at inference time. N extra forward passes per sample. Impractical for real-time inference on large models without batching.
- **Limitations:** Effective against patch-based triggers (BadNets) but weaker against blended or warping-based triggers that don't dominate the prediction as strongly. The entropy threshold requires calibration on clean data.

#### Statistical analysis for BadNets patches

BadNets (Gu et al., 2017) insert a fixed pixel pattern (patch) at a consistent location. Detection exploits the pixel-level consistency.

- **How it works:** For each class, compute per-pixel variance across all images. A fixed trigger patch creates anomalously low variance in the patch region for poisoned images. Compare variance maps across classes; the poisoned class will show distinctive low-variance regions.
- **Libraries:** Pure numpy. No specialized library needed.
- **Access:** Data-only. No model needed.
- **Compute:** Very low.
- **Limitations:** Only works for static patches at fixed locations. Misses blended triggers, random-location patches, and warping attacks entirely.

#### Blended trigger detection

Blended attacks (Chen et al., 2017) add a trigger pattern (e.g., a "Hello Kitty" watermark) at low opacity across the entire image.

- **How it works:** Frequency domain analysis. Compute the 2D FFT of images and look for consistent frequency peaks that appear only in one class. The blended pattern introduces repeatable frequency components. Cross-image spectral averaging reveals the ghost pattern.
- **Libraries:** numpy.fft, scipy.fft. For more structured analysis: opencv (cv2.dft).
- **Access:** Data-only.
- **Compute:** Low. FFT is O(n log n) per image.
- **Limitations:** Very low-opacity blends become indistinguishable from JPEG compression artifacts. Only works when the trigger has distinctive frequency content.

#### WaNet warping detection

WaNet (Nguyen & Tran, 2021) applies smooth spatial warping to images instead of adding pixel patterns.

- **How it works:** Warping distorts the regular grid of the image. Detection methods analyze the local affine transformation field or check for unnatural smooth deformations. One approach: train a small network to predict optical flow between pairs of augmented versions of the same image, and flag images whose flow fields show the characteristic warping grid pattern.
- **Libraries:** No mature off-the-shelf detector. Custom implementation using OpenCV (optical flow, grid detection) or Kornia (differentiable image transformations).
- **Access:** Data-only for frequency-based; white-box for activation-based approaches.
- **Compute:** Moderate.
- **Limitations:** WaNet and similar warping attacks remain among the hardest to detect. The warping is designed to be imperceptible and doesn't leave obvious statistical traces.

### 1.2 Label flipping and systematic mislabeling detection

#### Confident Learning / Cleanlab (Northcutt et al., 2021)

The standard method for finding label errors in datasets. Uses predicted class probabilities from any trained classifier to estimate the joint distribution of noisy and true labels.

- **How it works:** Train a classifier (or use cross-validation to avoid overfitting). For each sample, record the predicted probability vector. Estimate per-class thresholds (self-confidence). Construct the confident joint matrix Q, where Q[i][j] counts samples labeled i but predicted j with high confidence. Samples that appear in off-diagonal cells of Q are candidate label errors, ranked by a normalized margin.
- **Libraries:** [cleanlab](https://github.com/cleanlab/cleanlab) (pip install cleanlab). Works with any sklearn-compatible classifier, PyTorch, TensorFlow. Core function: `cleanlab.filter.find_label_issues()`.
- **Access:** Black-box. Only needs predicted probabilities.
- **Compute:** Low beyond the initial model training. The cleanlab analysis itself takes seconds for millions of samples.
- **Limitations:** The detector is only as good as the underlying model. If the model has itself learned the mislabeling, it won't flag those samples. Performs best with cross-validated predictions. Can produce false positives on genuinely hard-to-classify samples.

#### Cross-model agreement analysis

Train multiple architecturally different models and flag samples where models disagree with the given label consistently.

- **How it works:** Train 3-5 models (e.g., ResNet, EfficientNet, ViT) on the same dataset. For each sample, collect predictions from all models. If all models predict class B but the label says class A, it's a strong label error candidate. Rank by the number of disagreeing models.
- **Libraries:** Standard ML frameworks. No specialized library. PyTorch/TensorFlow + majority voting logic.
- **Access:** Black-box.
- **Compute:** High. Requires training multiple models.
- **Limitations:** Systematic labeling errors (where the "wrong" label follows a consistent pattern) may be learned by all models, making them invisible to this method.

#### Dataset cartography (Swayamdiha et al., 2020)

Maps training dynamics to characterize each sample: confidence (mean predicted probability of true label across epochs), variability (standard deviation of that probability), and correctness (fraction of epochs where prediction matched label).

- **How it works:** During training, log the model's softmax probability for the labeled class at each epoch. After training, compute per-sample confidence and variability. Plot in 2D. Mislabeled samples tend to cluster in the high-variability, low-confidence region. Ambiguous samples have high variability but moderate confidence. Easy samples have high confidence, low variability.
- **Libraries:** Custom implementation. Log during training with PyTorch hooks.
- **Access:** White-box. Requires access to training dynamics.
- **Compute:** Marginal overhead during training (logging softmax outputs). Analysis is lightweight.
- **Limitations:** Assumes some training epochs occur before analysis. Not suitable for pre-existing models unless you retrain. Hard vs. mislabeled samples occupy overlapping regions.

### 1.3 Near-duplicate flooding detection

When an attacker floods a dataset with many near-duplicates of specific images (possibly with subtle modifications), it biases the model toward those samples.

#### Perceptual hashing

- **How it works:** Compute a compact hash of each image that's robust to minor modifications (resizing, compression, color shifts). Compare hashes to find near-duplicates. Common algorithms: pHash (DCT-based), dHash (gradient-based), aHash (average-based), wHash (wavelet-based). Hamming distance between hashes determines similarity.
- **Libraries:** [imagededup](https://github.com/idealo/imagededup) (supports all four hash types + CNN-based dedup). [imagehash](https://github.com/JohannesBuchner/imagehash) (pip install imagehash). [imgdupes](https://github.com/knjcode/imgdupes) for CLI-based dedup.
- **Access:** Data-only.
- **Compute:** Very low. Hashing is O(1) per image (fixed-size computation after resize). Comparison is O(n^2) pairwise but can be accelerated with locality-sensitive hashing (LSH) to O(n log n).
- **Limitations:** Perceptual hashes miss semantically similar but visually different images. Conversely, heavy augmentation (rotation >10 degrees, cropping >20%) can change the hash enough to evade detection.

#### Embedding clustering

- **How it works:** Extract feature embeddings from a pretrained model (e.g., CLIP, ResNet, DINO). Cluster embeddings with HDBSCAN or DBSCAN. Anomalous clusters with many samples from few source images indicate flooding. Alternatively, compute pairwise cosine distances and flag pairs below a threshold.
- **Libraries:** CLIP (openai/clip), DINO (facebookresearch/dino), sentence-transformers for image embeddings. faiss for fast nearest-neighbor search. HDBSCAN for clustering.
- **Access:** Data-only (with a pretrained feature extractor).
- **Compute:** Moderate. One embedding extraction per image. Nearest-neighbor search is O(n log n) with faiss.
- **Limitations:** Threshold selection is tricky. Legitimate duplicates (same object, different views) might be flagged. The choice of embedding model matters: a model trained on similar data might not discriminate well.

#### Contributor-level dedup analysis

When dataset contributions come from multiple sources, analyze the distribution of duplicates per contributor.

- **How it works:** After computing near-duplicate clusters, trace each cluster back to its source/contributor. A contributor who submits many near-duplicates (possibly with small perturbations) is suspicious. Compute statistics: duplicates per contributor, diversity of contributed images (cluster count / sample count), hash similarity distribution.
- **Libraries:** pandas for aggregation, imagededup for hashing, matplotlib/seaborn for distribution visualization.
- **Access:** Data + metadata (source attribution).
- **Compute:** Low.
- **Limitations:** Requires contributor metadata, which not all datasets have. Legitimate contributors may submit batches of similar images (e.g., video frames).

### 1.4 Out-of-distribution sample detection in datasets

Detecting samples that don't belong to the intended data distribution. These could be adversarial insertions or data collection errors.

#### OpenOOD benchmark methods

The [OpenOOD](https://github.com/Jingkang50/OpenOOD) benchmark evaluates 100+ OOD detection methods. Key ones for dataset auditing:

**Mahalanobis Distance (Lee et al., 2018)**
- **How it works:** Fit a class-conditional Gaussian to the penultimate-layer features. For a new sample, compute the Mahalanobis distance to the nearest class centroid using the shared covariance. High distance = OOD.
- **Libraries:** pytorch-ood (`MahalanobisDistance` detector). Manual: scipy.spatial.distance.mahalanobis.
- **Access:** White-box.
- **Compute:** Low post-fitting. Requires one forward pass per sample + distance computation.
- **Limitations:** Assumes Gaussian feature distribution, which breaks for complex feature spaces. Sensitive to the choice of layer.

**Energy Score (Liu et al., 2020)**
- **How it works:** Use the logsumexp of the logits as an energy score. ID samples have lower energy (higher confidence); OOD samples have higher energy (more uniform logits). No retraining needed.
- **Libraries:** pytorch-ood (`EnergyBased` detector). Trivially implementable: `-torch.logsumexp(logits, dim=1)`.
- **Access:** White-box (logit access) or gray-box.
- **Compute:** Negligible beyond the forward pass.
- **Limitations:** Overconfident models can assign low energy to OOD samples. Works best when combined with temperature scaling.

**KNN-based detection (Sun et al., 2022)**
- **How it works:** Compute the k-th nearest neighbor distance in the feature space of a pretrained model. OOD samples are farther from their k-th neighbor than ID samples.
- **Libraries:** faiss for fast k-NN. pytorch-ood.
- **Access:** White-box for feature extraction.
- **Compute:** Moderate. Building the k-NN index is O(n log n).
- **Limitations:** Performance depends on the quality of the feature space. Memory-intensive for large datasets.

#### Isolation Forest / Local Outlier Factor

- **How it works:** Classical anomaly detection on extracted features. Isolation Forest isolates outliers by random splits (outliers need fewer splits). LOF compares local density around each point to that of its neighbors.
- **Libraries:** scikit-learn (`IsolationForest`, `LocalOutlierFactor`).
- **Access:** Data-only (features can come from any pretrained model).
- **Compute:** Low to moderate.
- **Limitations:** Not designed for high-dimensional raw pixel data. Works better on extracted features but then inherits the feature extractor's biases.

### 1.5 Source/contributor-level risk aggregation

Aggregate per-sample anomaly scores at the contributor level to identify systematically problematic sources.

- **How it works:** For each contributor, compute: (a) fraction of samples flagged by any detector above, (b) mean anomaly score across their samples, (c) diversity metrics (embedding spread, class distribution entropy), (d) temporal patterns (burst submissions, timing anomalies). Weight these into a composite risk score. Flag contributors whose score exceeds a threshold based on the population distribution.
- **Libraries:** pandas, scipy.stats for aggregation and statistical testing. No purpose-built library exists; this is typically custom.
- **Access:** Data + metadata.
- **Compute:** Low (operates on pre-computed per-sample scores).
- **Limitations:** Arbitrary thresholds. Small contributors get noisy estimates. Adversaries can spread poisoning across many sock-puppet accounts.

---

## 2. Model integrity assessment

### 2.1 Neural Cleanse (Wang et al., 2019)

The foundational trigger reverse-engineering method. For each possible target class, it optimizes a minimal perturbation (trigger pattern + mask) that causes all inputs to be classified as that target. If one class requires a significantly smaller trigger than others, the model is likely backdoored with that target class.

- **How it works:** For each class t, solve: minimize |mask| such that f(x * (1-mask) + pattern * mask) = t for all x. Use gradient descent on mask and pattern simultaneously. The L1 norm of the mask is the "trigger size." Compute the Median Absolute Deviation (MAD) of trigger sizes across all classes. An anomaly index > 2 indicates backdoor.
- **Libraries:** ART implements `NeuralCleanse`. The [original code](https://github.com/bolunwang/backdoor) is available. Also in BackdoorBench.
- **Access:** White-box. Requires gradient computation through the model.
- **Compute:** High. Runs one optimization per target class, each requiring many forward/backward passes. For ImageNet-scale (1000 classes), this becomes very expensive.
- **Limitations:** Assumes the trigger is a small, input-independent patch. Fails against clean-label attacks, input-aware triggers, and non-additive triggers (WaNet). The optimization can get stuck in local minima. Multiple-target attacks can spread the anomaly across classes, reducing the MAD score below threshold.

### 2.2 Activation analysis methods

#### Feature visualization

- **How it works:** Generate synthetic images that maximize the activation of specific neurons or layers using gradient ascent. If a neuron strongly activates on trigger-like patterns rather than semantic features, it's suspicious.
- **Libraries:** [lucent](https://github.com/greentfrapp/lucent) (PyTorch port of Lucid). [captum](https://github.com/pytorch/captum) for attribution. [torch-dreams](https://github.com/Mayukhdeb/torch-dreams).
- **Access:** White-box.
- **Compute:** Moderate. Each visualization requires an optimization loop (typically 512-2048 steps).
- **Limitations:** Interpretation is subjective. Not every suspicious-looking neuron represents a backdoor. Doesn't scale well to examining every neuron in large models.

#### Activation clustering for backdoor detection

A different application of the activation clustering from Section 1.1, applied to a suspect model rather than a suspect dataset.

- **How it works:** Feed a clean held-out dataset through the suspect model. Extract activations from key layers. Cluster the activations per class. A backdoored model may show bimodal activation patterns for the target class, even on clean data, because the backdoor pathway creates a distinct activation subspace.
- **Libraries:** ART (`ActivationDefence`). Manual: PyTorch hooks + scikit-learn clustering.
- **Access:** White-box.
- **Compute:** Moderate.
- **Limitations:** Assumes the backdoor creates a separable activation signature, which isn't always the case for well-optimized attacks.

#### GradCAM-based analysis

- **How it works:** For predictions on clean inputs, generate GradCAM heatmaps. A backdoored model may focus on irrelevant image regions for the target class, or show diffuse, non-semantic attention patterns.
- **Libraries:** [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam). captum.
- **Access:** White-box.
- **Compute:** Low. One backward pass per visualization.
- **Limitations:** Qualitative. Requires manual inspection or training a secondary classifier on the heatmaps. Some attacks don't produce obviously suspicious heatmaps.

### 2.3 Model fingerprinting / behavioral fingerprinting

#### Adversarial example fingerprinting

- **How it works:** Generate a set of adversarial examples near the decision boundary. A model's responses to these boundary samples create a "fingerprint" of its learned decision surface. Two independently trained models will have different fingerprints. A stolen/fine-tuned model will share similar fingerprints with its parent.
- **Libraries:** [AnaFP](https://github.com/) (analytical fingerprinting). ART's adversarial example generators (PGD, FGSM) + custom comparison logic.
- **Access:** White-box for fingerprint generation; black-box for fingerprint verification.
- **Compute:** Moderate. Generating adversarial examples requires iterative optimization.
- **Limitations:** Fine-tuning or quantization can degrade fingerprint match rates. Doesn't directly detect backdoors: it detects model lineage.

#### IPGuard (Cao et al., 2021)

- **How it works:** Selects data points near the classification boundary that are maximally sensitive to model changes. The model's predictions on these points form a fingerprint. A copied model will match the fingerprint; an independently trained one won't.
- **Libraries:** No widely maintained package. Reference implementations exist in papers.
- **Access:** Black-box for verification (only needs prediction outputs).
- **Compute:** Moderate for fingerprint generation; low for verification.
- **Limitations:** Fingerprint can be broken by model extraction attacks that smooth the decision boundary differently.

#### Conferrable adversarial examples

- **How it works:** Generate adversarial examples that specifically transfer from the source model to its derivatives but not to independently trained models. The transferability rate serves as the fingerprint match score.
- **Libraries:** ART for adversarial example generation + transfer testing.
- **Access:** Black-box for verification.
- **Compute:** Moderate.
- **Limitations:** Transfer rates are noisy. Requires careful selection of perturbation magnitude.

### 2.4 Weight-level anomaly detection

#### Weight distribution analysis

- **How it works:** Analyze the statistical distribution of model weights per layer. Backdoor injection (especially through direct weight modification) can introduce outlier weights or shift the weight distribution. Compare weight histograms, compute kurtosis, skewness, and test against expected distributions (approximately Gaussian for well-trained networks).
- **Libraries:** Pure PyTorch/numpy. scipy.stats for distribution fitting and statistical tests.
- **Access:** White-box. Full weight access needed.
- **Compute:** Very low.
- **Limitations:** Backdoors injected through training (rather than weight patching) produce natural-looking weight distributions. This catches crude attacks but misses training-time poisoning.

#### Fine-Pruning (Liu et al., 2018)

- **How it works:** Prune dormant neurons (those with low average activation on clean data) then fine-tune. The intuition: backdoor behavior often lives in neurons that aren't active for clean data. Pruning them removes the backdoor while preserving clean accuracy.
- **Libraries:** PyTorch native pruning (torch.nn.utils.prune) + custom activation monitoring. No dedicated library, but the approach is simple to implement.
- **Access:** White-box.
- **Compute:** Moderate. Requires forward passes on clean data for activation statistics, then pruning + fine-tuning.
- **Limitations:** Aggressive pruning degrades clean accuracy. Some backdoors spread their activation across many neurons, resisting pruning. The defense itself is also a remediation, not just detection.

#### ANP: Adversarial Neuron Pruning (Wu & Wang, 2021)

- **How it works:** Perturb each neuron's weights with adversarial noise and measure the effect on model output. Neurons that cause disproportionate damage when perturbed are likely backdoor-related.
- **Libraries:** Manual implementation in PyTorch. Available in [BackdoorBench](https://github.com/SCLBD/BackdoorBench).
- **Access:** White-box.
- **Compute:** Moderate to high. Requires per-neuron perturbation analysis.
- **Limitations:** Computationally expensive for large models. Some naturally important clean neurons also show high sensitivity to perturbation.

### 2.5 White-box vs. black-box assessment capabilities

| Capability | White-box | Black-box |
|---|---|---|
| Trigger reverse-engineering | Neural Cleanse, DeepInspect, K-Arm | Limited. Can attempt trigger inversion with gradient estimation (NES, SPSA) but much slower |
| Activation analysis | Full access to internal representations | Not available |
| Weight analysis | All weight-level methods | Not available |
| Fingerprinting verification | Can generate fingerprints | Can verify fingerprints (query model, compare predictions) |
| STRIP-like detection | Works | Works (only needs output probabilities) |
| MNTD | Can extract meta-features from weights | Can use query-based meta-features |
| Spectral analysis | Full feature decomposition | Not available |

In practice, white-box access is essential for thorough model audit. Black-box scenarios (e.g., API-only model access) limit you to behavioral testing, query-based fingerprinting, and output probability analysis.

### 2.6 Meta Neural Analysis and MNTD

#### MNTD: Meta Neural Trojan Detection (Xu et al., 2021)

- **How it works:** Train a "meta-classifier" that takes a neural network's properties as input and outputs whether that network is trojaned. The meta-classifier is trained on a diverse set of clean and trojaned shadow models. At test time, extract the same features from the suspect model and run the meta-classifier.
  - **Jumbo Learning variant:** Use a set of optimized "query inputs" (universal litmus patterns), feed them through the suspect model, and use the output logits as features for the meta-classifier. This works even with black-box access.
- **Libraries:** The [original implementation](https://github.com/AI-secure/Meta-Nerual-Trojan-Detection) is available. No pip-installable package.
- **Access:** White-box for weight-based features. Black-box for the query-based (jumbo learning) variant.
- **Compute:** High for meta-classifier training (train many shadow models). Low for inference (one forward pass through meta-classifier).
- **Limitations:** The meta-classifier's effectiveness depends on how well the shadow models represent real-world attack diversity. If the actual attack differs significantly from training distribution attacks, detection degrades. Achieves ~97% AUC on the datasets it was evaluated on, but generalization to novel attack types is uncertain.

#### Universal Litmus Patterns (ULP, Kolouri et al., 2020)

- **How it works:** Optimize a small set of input images ("litmus patterns") such that a trojaned model produces distinguishably different outputs on these patterns compared to a clean model. A linear classifier on the outputs can then distinguish clean from trojaned.
- **Libraries:** Reference implementation available. Typically custom PyTorch.
- **Access:** Black-box at test time (just feed patterns and observe outputs).
- **Compute:** High for pattern optimization (inner loop trains shadow models). Very low at test time.
- **Limitations:** Same shadow model diversity problem as MNTD. The litmus patterns may need to be re-optimized for different model architectures or domains.

---

## 3. Inference provenance

### 3.1 Cryptographic binding of input, model, config, output

The goal: produce a tamper-evident receipt proving that a specific output came from a specific model processing a specific input with specific configuration.

#### Hash-based binding

- **How it works:** Compute: H(input) || H(model_weights) || H(config) || H(output) and sign the concatenation. This creates a cryptographic commitment that binds all four elements. Any modification to any component invalidates the signature.
  - Model hash: SHA-256 over the serialized weight tensor bytes. For large models, hash the weight file directly.
  - Input hash: SHA-256 of the raw input bytes (before preprocessing).
  - Config hash: SHA-256 of a canonical JSON representation of inference parameters (batch size, precision, etc.).
  - Output hash: SHA-256 of the serialized output tensor.
- **Libraries:** Python `hashlib` (SHA-256, SHA-3). `cryptography` library for signing. For model-specific hashing: custom serialization.
- **Access:** Full system access (inputs, model, configuration, outputs).
- **Compute:** Negligible. SHA-256 hashing is fast even for GB-sized models.
- **Limitations:** Hash-based binding is all-or-nothing. It proves tamper detection but can't prove which component was altered. Non-deterministic inference (floating-point non-determinism across hardware) means running the same input through the same model can produce slightly different outputs, breaking exact output hash verification. Mitigation: hash the rounded or quantized output, or include a tolerance specification.

#### Model identity via weight digest

- **How it works:** Rather than hashing the full weights (which changes with any format conversion), compute a canonical digest: sort layers by name, concatenate their flattened FP32 representations, hash the result. This handles different serialization formats producing the same logical model.
- **Libraries:** PyTorch's `state_dict()` + hashlib. For ONNX models: onnx library + hashlib.
- **Compute:** Minutes for multi-GB models. Can be precomputed once.
- **Limitations:** Quantized models (INT8, FP16) need their own digests. The same architecture with different precision won't match.

### 3.2 Merkle tree / hash chain approaches for inference audit trails

#### Hash chains for sequential inference

- **How it works:** Each inference record includes the hash of the previous record, creating a chain. Record_n = {input_hash, output_hash, model_hash, config_hash, timestamp, H(Record_{n-1})}. Tampering with any record breaks the chain. This is a minimal blockchain without consensus.
- **Libraries:** Custom implementation with hashlib. sqlite3 for storage. Or use a lightweight append-only log like [immudb](https://github.com/codenotary/immudb) (though this adds a dependency).
- **Compute:** Negligible per inference.
- **Limitations:** The chain proves ordering and integrity, but the first record must be anchored to a trusted starting point. If the logging system itself is compromised, false records can be inserted at the chain's end.

#### Merkle trees for batch inference audit

- **How it works:** Group inference records into batches. Build a Merkle tree (binary hash tree) over the batch. The root hash represents the entire batch. To verify a single record, only O(log n) hashes are needed (the Merkle proof / authentication path). Useful for efficient selective auditing.
- **Libraries:** [pymerkle](https://github.com/fmerg/pymerkle). [merkletools](https://github.com/Tierion/pymerkletools). Or hand-roll with hashlib.
- **Compute:** Tree construction is O(n). Proof verification is O(log n).
- **Limitations:** Like hash chains, the root must be anchored (signed, published, or timestamped by a trusted authority). Without this anchor, the entire tree can be replaced.

### 3.3 Digital signature schemes for offline/air-gapped environments

#### Ed25519

- **How it works:** EdDSA signature scheme on Curve25519. Fast, deterministic (no random nonce needed, reducing implementation risk), compact signatures (64 bytes). Sign the inference receipt with the system's private key. Anyone with the public key can verify.
- **Libraries:** `cryptography` library (from cryptography.hazmat.primitives.asymmetric.ed25519). `nacl` (PyNaCl/libsodium). `ed25519` pure Python package.
- **Compute:** Signing: ~50 microseconds. Verification: ~100 microseconds.
- **Suitability:** Excellent for air-gapped environments. Deterministic signing means no CSPRNG required at sign time (the nonce is derived from the private key and message). Small key sizes (32-byte private, 32-byte public).
- **Limitations:** No post-quantum security. For long-term (10+ year) audit trails, consider hybrid schemes.

#### ECDSA (P-256)

- **How it works:** The standard NIST curve signature scheme. Widely supported, hardware-accelerated on many platforms.
- **Libraries:** `cryptography` library. OpenSSL bindings. `ecdsa` pure Python.
- **Compute:** Slightly slower than Ed25519 but still microseconds.
- **Limitations:** Requires a good random number generator for nonce generation (unless using RFC 6979 deterministic nonces). Implementation is more error-prone than Ed25519.

#### Hardware-backed signing (when available)

- TPM, HSM, or Secure Enclave can hold private keys that never leave hardware.
- **Libraries:** `tpm2-pytss` for TPM 2.0. `pkcs11` for HSM access.
- Not always available in air-gapped ML systems, but worth integrating when the hardware exists.

### 3.4 Replay attack prevention

#### Nonces

- **How it works:** Include a unique random nonce in each inference receipt before signing. The verifier maintains a set of seen nonces and rejects duplicates. For air-gapped systems, use UUIDv4 as the nonce.
- **Libraries:** `uuid` (stdlib), `secrets` (stdlib).
- **Limitations:** Nonce storage grows unboundedly. Mitigation: combine with timestamps and expire nonces after a window.

#### Monotonic sequence numbers

- **How it works:** Each inference gets a strictly incrementing sequence number. The verifier rejects any receipt with a sequence number <= the last verified number. Simpler than nonces, no storage growth problem.
- **Libraries:** Stdlib. Persistent counter in sqlite3 or a file.
- **Limitations:** Requires the signer maintain state. If the counter resets (system crash), gap detection is needed. Parallel inference systems need coordinated counters or per-instance sequences.

#### Timestamps

- **How it works:** Include a high-resolution timestamp. The verifier rejects receipts outside an acceptable time window and rejects duplicate timestamps.
- **Limitations:** Air-gapped systems may not have accurate clocks. Clock skew tolerance creates a replay window. Best used in combination with nonces or sequence numbers, not alone.

#### Combined approach (recommended)

Each inference receipt should include: nonce (UUIDv4) + sequence number + timestamp. The sequence number provides ordering, the nonce prevents duplicates even if the sequence counter is compromised, and the timestamp provides human-readable audit context.

### 3.5 C2PA / Content Authenticity Standards relevance

The [Coalition for Content Provenance and Authenticity (C2PA)](https://c2pa.org/) defines an open standard for attaching cryptographic provenance metadata ("Content Credentials") to media files.

#### Relevance to inference provenance

- **How it works:** C2PA embeds a "manifest" in the media file (JUMBF format for images) containing: who/what created or edited the content, what tools were used, a hash of the content, and a digital signature. The manifest can chain: if image A was processed by model B to produce image C, the manifest on C references A's manifest and B's identity.
- **Libraries:** [c2pa-python](https://github.com/contentauth/c2pa-python) (official Python SDK). [c2pa-rs](https://github.com/contentauth/c2pa-rs) (Rust, also has Python bindings). [c2patool](https://github.com/contentauth/c2patool) (CLI).
- **Applicability:** Directly relevant for computer vision pipelines that produce images (generative models, image enhancement, image classification with annotated output). Less directly applicable for non-image outputs (bounding boxes, class labels).
- **Compute:** Low. Manifest creation and signing are fast.
- **Limitations:**
  - C2PA proves provenance of the manifest creator, not truth. A malicious actor can create valid C2PA manifests for manipulated content.
  - Requires a trust model: verifiers must trust the signer's certificate chain. For offline use, pre-distributed certificate bundles work, but revocation checking is impossible without network access.
  - The standard is designed for media distribution, not general ML inference logging. Using it for that purpose requires adapting the "action" and "assertion" vocabulary.
  - Stripping metadata is trivial (screenshot, re-encode without JUMBF). C2PA protects provenance for cooperative workflows, not adversarial ones.

---

## 4. Distribution shift detection

### 4.1 Maximum Mean Discrepancy (MMD)

A kernel-based statistical test for whether two sets of samples come from the same distribution.

- **How it works:** Embed both sample sets into a Reproducing Kernel Hilbert Space (RKHS) using a kernel function (typically RBF/Gaussian). Compute the distance between the mean embeddings of the two sets. The MMD statistic is: MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)], where x~P, y~Q. A permutation test or asymptotic approximation gives a p-value for the null hypothesis that P=Q.
- **Libraries:** [alibi-detect](https://github.com/SeldonIO/alibi-detect) (`MMDDrift` detector, supports both offline and online detection). [torch-two-sample](https://github.com/josipd/torch-two-sample). Manual: sklearn.metrics.pairwise.rbf_kernel + custom MMD computation.
- **Access:** Data-only (but usually applied to feature embeddings, requiring a model for extraction).
- **Compute:** O(n^2) for the naive estimator (n = total samples). Linear-time estimators exist (block MMD) but have lower statistical power. For large datasets, subsample or use the block estimator. alibi-detect supports GPU-accelerated computation.
- **Limitations:**
  - Kernel bandwidth selection strongly affects results. The median heuristic (set bandwidth = median pairwise distance) works reasonably but isn't optimal for all distributions.
  - MMD detects that distributions differ but doesn't characterize how. A significant MMD doesn't tell you whether you have covariate shift, concept drift, or an adversarial perturbation.
  - Sensitive to sample size. Small sample sizes give low power (fail to detect real shifts). Need at least ~100-500 samples per set for reliable detection.

### 4.2 Frechet Inception Distance (FID) and variants

Originally designed to evaluate generative models, FID measures the distance between the feature distributions of two image sets.

- **How it works:** Pass both image sets through InceptionV3 (pool3 layer, 2048-dim features). Fit a multivariate Gaussian (mean + covariance) to each set's features. Compute the Frechet distance: FID = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*(C_1*C_2)^{1/2}). Lower FID = more similar distributions.
- **Libraries:** [pytorch-fid](https://github.com/mseitzer/pytorch-fid) (pip install pytorch-fid). [clean-fid](https://github.com/GaParmar/clean-fid) (fixes known implementation inconsistencies around image resizing). [torch-fidelity](https://github.com/toshas/torch-fidelity). PyTorch-Ignite has `ignite.metrics.FID`.
- **Access:** Data-only (uses a fixed pretrained InceptionV3).
- **Compute:** Moderate. One forward pass per image through InceptionV3. Covariance computation is O(n * d^2) where d=2048.
- **Limitations:**
  - Assumes features are Gaussian-distributed, which is an approximation. Real feature distributions can be multimodal.
  - Tied to InceptionV3's feature space, which was trained on ImageNet. For specialized domains (medical imaging, satellite imagery), InceptionV3 features may not capture domain-relevant variations. Consider domain-specific feature extractors.
  - Needs ~10,000+ samples per set for stable estimates. Fewer samples produce noisy FID scores.
  - FID is a single scalar. Two very different shifts can produce the same FID value. It doesn't tell you what changed.

#### Kernel Inception Distance (KID)

- **How it works:** Similar to FID but uses MMD on Inception features instead of Frechet distance. Doesn't assume Gaussian features.
- **Libraries:** torch-fidelity, clean-fid.
- **Advantages over FID:** Unbiased estimator (FID is biased for small samples). No Gaussian assumption.
- **Limitations:** Higher variance than FID for the same sample size.

### 4.3 Kolmogorov-Smirnov tests on feature distributions

A non-parametric test comparing the cumulative distribution functions of two samples.

- **How it works:** For each feature dimension (or a selected set of important features), compute the maximum absolute difference between the empirical CDFs of the reference and test distributions. The KS statistic and its p-value indicate whether the two distributions differ significantly. For multivariate data, apply per-feature and correct for multiple testing (Bonferroni or Benjamini-Hochberg).
- **Libraries:** `scipy.stats.ks_2samp` for univariate tests. alibi-detect (`KSDrift`) for automated per-feature drift detection with correction. `scipy.stats.anderson_ksamp` for k-sample Anderson-Darling (more power than KS for tail differences).
- **Access:** Data-only.
- **Compute:** O(n log n) per feature (sorting). Very fast.
- **Limitations:**
  - Univariate: the KS test is inherently one-dimensional. Applying it per-feature misses multivariate correlations. Two distributions can match marginally on every feature but differ in their joint distribution.
  - Low power for high-dimensional data. Multiple testing correction (needed when testing hundreds of features) reduces sensitivity.
  - Detects distributional differences but doesn't distinguish shift types.

### 4.4 Characterizing shift type

#### Covariate shift (P(X) changes, P(Y|X) stays)

The input distribution changes but the labeling function remains the same. Example: lighting conditions change between training and deployment.

- **Detection:** MMD or KS tests on input features (before the final classifier layer) will fire. Classifier confidence distributions on the new data remain high (the model still knows the right answer for each input, it just sees the inputs less often).
- **Libraries:** alibi-detect for tabular/feature-level drift. For images: compute FID between train and deploy sets.
- **Distinguishing feature:** Model calibration on shifted data stays reasonable. Accuracy may drop but not catastrophically.

#### Concept shift (P(Y|X) changes, P(X) stays)

The relationship between inputs and labels changes. Example: what counts as "spam" evolves while email distributions look similar.

- **Detection:** Input-level tests (MMD on features) may NOT fire because P(X) hasn't changed. Monitor prediction confidence over time; dropping confidence on data that looks familiar signals concept shift. Compare model accuracy on labeled batches (requires some ground-truth labels).
- **Libraries:** [river](https://github.com/online-ml/river) (online learning library with concept drift detectors: ADWIN, DDM, EDDM, Page-Hinkley). alibi-detect.
- **Distinguishing feature:** Feature distributions look the same, but model performance degrades.

#### Prior probability shift (P(Y) changes)

The class frequencies change. Example: seasonal variation in product demand.

- **Detection:** Monitor the distribution of predicted classes over time. Compare predicted class frequencies against training class frequencies using chi-squared tests. If P(Y) shifts but P(X|Y) doesn't, a chi-squared test on predicted labels catches it.
- **Libraries:** `scipy.stats.chisquare`, `scipy.stats.chi2_contingency`.
- **Distinguishing feature:** Per-class accuracy stays stable, but overall accuracy changes because the class mix differs.

### 4.5 Distinguishing natural drift from adversarial manipulation

This is the hardest problem. Adversarial distribution shift is designed to look like natural drift while causing targeted model failures.

#### Rate and magnitude analysis

- **How it works:** Natural drift is typically gradual and affects many features. Adversarial manipulation often affects specific features sharply or injects outliers. Monitor the rate of change of drift statistics (MMD, FID) over time. A sudden spike in drift that reverts quickly is suspicious. Natural environmental changes produce smooth trends.
- **Libraries:** scipy.signal for trend analysis. prophet or statsmodels for time-series decomposition of drift metrics.

#### Targeted vs. uniform effect

- **How it works:** Natural covariate shift affects model performance roughly uniformly across classes. Adversarial manipulation often targets specific classes. Compute per-class performance metrics over time. An attack that degrades one class while others stay stable, despite a drift that appears uniform at the input level, is a red flag.
- **Libraries:** scikit-learn classification reports per batch + custom monitoring.

#### Adversarial sample detection

- **How it works:** Apply adversarial detection methods to incoming data. Feature Squeezing (Xu et al., 2017) compresses inputs (bit-depth reduction, spatial smoothing) and checks whether the model's prediction changes. Large prediction changes indicate adversarial manipulation.
- **Libraries:** ART implements `FeatureSqueezing`, `SpatialSmoothing`, and other input transformation defenses.
- **Access:** Black-box (only needs model predictions).
- **Compute:** 2-3x the base inference cost (run input through multiple compression variants).
- **Limitations:** Adaptive adversaries can craft inputs that survive feature squeezing. This is an arms race.

#### Combining statistical and behavioral signals

A practical monitoring system should combine:
1. Input-side: MMD/FID/KS on incoming features (alibi-detect)
2. Output-side: prediction confidence distribution monitoring (custom)
3. Per-class performance: if labels are available, per-class accuracy tracking (sklearn metrics)
4. Rate analysis: is the drift sudden or gradual? (time-series analysis)
5. Anomaly detection: per-sample outlier scoring (pytorch-ood, LOF)

If (1) fires but (3) doesn't, it's likely benign covariate shift. If (3) fires on specific classes but (1) looks normal, suspect concept shift or targeted attack. If (1) spikes suddenly and (5) flags many samples, investigate adversarial manipulation.

---

## Library summary table

| Library | Primary area | Install | Key capabilities |
|---|---|---|---|
| [ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox) | 1, 2 | `pip install adversarial-robustness-toolbox` | Backdoor detection (Neural Cleanse, Spectral Signatures, STRIP, Activation Clustering), adversarial attacks/defenses, poisoning defenses |
| [BackdoorBench](https://github.com/SCLBD/BackdoorBench) | 1, 2 | Clone from GitHub | Unified benchmark for 16+ attacks and 20+ defenses. Research-oriented, not a pip package |
| [cleanlab](https://github.com/cleanlab/cleanlab) | 1 | `pip install cleanlab` | Label error detection, confident learning, data quality scoring |
| [imagededup](https://github.com/idealo/imagededup) | 1 | `pip install imagededup` | Perceptual hashing (pHash, dHash, aHash, wHash) + CNN dedup |
| [imagehash](https://github.com/JohannesBuchner/imagehash) | 1 | `pip install imagehash` | Perceptual hashing (pHash, dHash, aHash, wHash) |
| [pytorch-ood](https://github.com/kkirchheim/pytorch-ood) | 1, 4 | `pip install pytorch-ood` | OOD detection (Mahalanobis, Energy, MSP, ODIN, KNN) |
| [OpenOOD](https://github.com/Jingkang50/OpenOOD) | 1 | Clone from GitHub | Benchmark for 100+ OOD methods |
| [captum](https://github.com/pytorch/captum) | 2 | `pip install captum` | Model interpretability (GradCAM, Integrated Gradients, SHAP) |
| [lucent](https://github.com/greentfrapp/lucent) | 2 | `pip install lucent` | Feature visualization, deep dream |
| [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) | 2 | `pip install grad-cam` | GradCAM and variants for attention analysis |
| [alibi-detect](https://github.com/SeldonIO/alibi-detect) | 4 | `pip install alibi-detect` | Drift detection (MMD, KS, Chi-Squared, LSDD), outlier detection, adversarial detection |
| [pytorch-fid](https://github.com/mseitzer/pytorch-fid) | 4 | `pip install pytorch-fid` | FID computation |
| [clean-fid](https://github.com/GaParmar/clean-fid) | 4 | `pip install clean-fid` | FID with correct image resizing |
| [torch-fidelity](https://github.com/toshas/torch-fidelity) | 4 | `pip install torch-fidelity` | FID, KID, Inception Score |
| [river](https://github.com/online-ml/river) | 4 | `pip install river` | Online concept drift detection (ADWIN, DDM, EDDM) |
| [c2pa-python](https://github.com/contentauth/c2pa-python) | 3 | `pip install c2pa-python` | Content Credentials creation/verification |
| [cryptography](https://github.com/pyca/cryptography) | 3 | `pip install cryptography` | Ed25519, ECDSA, hashing, certificate handling |
| [pymerkle](https://github.com/fmerg/pymerkle) | 3 | `pip install pymerkle` | Merkle tree construction and proof verification |
| [faiss](https://github.com/facebookresearch/faiss) | 1, 4 | `pip install faiss-cpu` | Fast nearest-neighbor search for embedding analysis |
| scikit-learn | 1, 2, 4 | `pip install scikit-learn` | Clustering, PCA, Isolation Forest, LOF, classification metrics |
| scipy | 1, 2, 3, 4 | `pip install scipy` | KS tests, chi-squared, entropy, SVD, statistical functions |

---

## References

### Training data integrity
- Gu et al., "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain," 2017
- Chen et al., "Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning," 2017
- Tran et al., "Spectral Signatures in Backdoor Attacks," NeurIPS 2018
- Chen et al., "Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering," 2019
- Gao et al., "STRIP: A Defence Against Trojan Attacks on Deep Neural Networks," ACSAC 2019
- Nguyen & Tran, "WaNet: Imperceptible Warping-based Backdoor Attack," ICLR 2021
- Northcutt et al., "Confident Learning: Estimating Uncertainty in Dataset Labels," JAIR 2021

### Model integrity
- Wang et al., "Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks," IEEE S&P 2019
- Liu et al., "Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks," RAID 2018
- Xu et al., "Detecting AI Trojans Using Meta Neural Analysis," IEEE S&P 2021
- Kolouri et al., "Universal Litmus Patterns: Revealing Backdoor Attacks in CNNs," CVPR 2020
- Wu & Wang, "Adversarial Neuron Pruning Purifies Backdoored Deep Models," NeurIPS 2021
- Cao et al., "IPGuard: Protecting Intellectual Property of Deep Neural Networks via Fingerprinting the Classification Boundary," AsiaCCS 2021

### Inference provenance
- C2PA Technical Specification, https://c2pa.org/specifications/
- Bernstein et al., "High-speed high-security signatures," J. Cryptographic Engineering 2012 (Ed25519)
- Merkle, "A Digital Signature Based on a Conventional Encryption Function," CRYPTO 1987

### Distribution shift detection
- Gretton et al., "A Kernel Two-Sample Test," JMLR 2012 (MMD)
- Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium," NeurIPS 2017 (FID)
- Parmar et al., "On Aliased Resizing and Surprising Subtleties in GAN Evaluation," CVPR 2022 (clean-fid)
- Liu et al., "Energy-based Out-of-distribution Detection," NeurIPS 2020
- Lee et al., "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks," NeurIPS 2018 (Mahalanobis)
- Sun et al., "Out-of-Distribution Detection with Deep Nearest Neighbors," ICML 2022
