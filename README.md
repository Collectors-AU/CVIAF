# Trustworthy CV Integrity Assurance Framework (SIH 26228)

An extensible, air-gapped computer-vision assurance engine designed to evaluate data, model, and inference integrity across multi-contributor pipelines.

## Architecture

The framework consists of four disjoint but chronologically dependent offline verification nodes:

1. **`data_integrity.py` (Training Data Integrity)**
   - Utilizes **Spectral Signatures (Activation Clustering)**. Deep features from YOLO/COCO bounding boxes are iteratively clustered using PCA and k-Means. High silhouette scores on isolated point clouds signify poisoned "trigger" patches injected by untrusted contributors.

2. **`model_integrity.py` (Model Integrity)**
   - **White-box**: Implements **Neural Cleanse** logics—attempting to build minimal universal perturbations that trigger false positives in PyTorch/ONNX models. L1 norm anomalies flag backdoors.
   - **Black-box**: Entropy probing. Uses Randomized Smoothing and query-based boundary checks. Tightly bound confidence curves over pure noise inputs indicate subversion. Graces backward cleanly when weights are hidden.

3. **`provenance.py` (Inference Provenance)**
   - Because of air-gap constraints, we cannot use cloud KMS (Key Management Services). 
   - A localized Ed25519 HMAC root generates a Merkle-leaf or hash block tying together: `SHA3(image_file) + ONNX_digest + Preprocessing_config + Output_Tensor + Offline_Nonce`. Modifying single logging bits breaks the tamper-evident signature natively without network time protocols.

4. **`drift.py` (Distribution Shift & Anomaly Assessment)**
   - Runs offline **Mahalanobis Feature Distance** mapping against pre-configured benign distributions (using Inverse Covariance matrices).
   - Fuses with **Maximum Softmax Probability (MSP)** from classification heads. Disentangles standard operational drift (moderate covariance delta) from adversarial anomalies (high anomaly delta, extreme softmax suppression).

5. **`assessor.py` (Analyst-Facing Governance)**
   - Combines modules into a final, human-readable JSON schema dictating severity (LOW/MEDIUM/HIGH/CRITICAL), confidence scores, and action verdicts (`accept`, `review`, `quarantine`).

## Setup Notes

```bash
# Environment (Requires python 3.10+, numpy, scikit-learn)
pip install numpy scikit-learn

# Run the assessor test
export PYTHONPATH=.
python3 engine/assessor.py
```