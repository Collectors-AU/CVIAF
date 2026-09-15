# Trustworthy Computer Vision Integrity Assurance for Data, Models, and Inference Outputs

**Objective:** SIH 26228 - Indian Army (DGIS)  
**System Architecture & Empirical Design Report**

## Abstract
This report details the architecture of an offline, air-gapped computer vision framework designed to verify data integrity, model behavior, and inference provenance. Addressing vulnerabilities such as data poisoning, trigger-based backdoors, and post-hoc tampering, the system employs Spectral Signatures for dataset defense, Neural Cleanse for structural integrity, and local cryptographic seals (Ed25519) to enforce inference repudiation chains. Anomaly separation is governed by Mahalanobis distance thresholds.

## 1. Introduction
Operational computer vision pipelines often ingest components from untrusted or semi-trusted vendors. A malicious contributor bounding-box (in COCO/YOLO) or an adversarial backdoor nested inside an ONNX weight digest poses critical security threats. This framework provides an offline governance engine that acts unconditionally on input vectors to declare confidence without relying on outside API access.

## 2. Methodology
The architectural framework consists of four independent verification nodes.

### 2.1 Training-Data Integrity
Relying on Activation Clustering and Spectral Signatures, the data scanner pulls features iteratively mapping the latent space of contributed batches. Backdoor patches inherently behave as strong attractors, forming an isolated distribution cluster (k=2) when transformed via PCA. By flagging abnormally tight sub-populations, the system attributes a `risk_score` directly to contributor IDs from the metadata manifest.

### 2.2 Model Integrity
The model assessor accepts configuration modes based on privilege limitation.
- **White-box access:** Adapts trigger-inversion strategies (e.g., *Neural Cleanse*). It searches for universal perturbations requiring a low $L_1$ modification budget to manipulate the model's confidence.
- **Black-box fallback:** Since complete access is never guaranteed, the node falls back to probing the model with highly degraded or obfuscated inputs. Abnormally low entropy outputs on obfuscated data flag suspicious, hard-coded inference pathways.

### 2.3 Inference Provenance
Since cloud-based KMS structures violate the strict air-gapped constraint, provenance rests on an isolated cryptographic sealing standard. An Ed25519 or SHA-3 HMAC routine receives the `image_hash`, `model_digest`, `preprocessing_config`, and the exact `output_tensor`. Wrapped using an offline sequential nonce, changes made retrospectively to the inference record fail validation against the stored block signatures.

### 2.4 Distribution Shift vs Adversarial Anomalies
Operational drift (e.g., season changes) shifts distributions holistically. Adversarial attacks manifest as hyper-singularities. The distribution assessor merges **Maximum Softmax Probability (MSP)** from classification heads with **Mahalanobis Feature Distance** mapping. Massive Mahalanobis deviations accompanying low MSP scores trigger a `HIGH` risk metric indicating adversarial Out-of-Distribution structures, whereas moderate symmetric shifts indicate natural drift.

## 3. Results & Analyst Governance
All findings cascade into an Analyst-Facing JSON packet. Every flag yields a localized human-readable directive ranging from `LOW (accept)` to `CRITICAL (quarantine)`. A strictly reproducible audit log ensures operations can be retraced securely. 

## 4. Limitations Confirmed
The system acknowledges that black-box backdoor detection is inherently a probabilistic endeavor compared to white-box gradient inversion. False positives may emerge if training data contains natural bimodal distributions.

## References
1. Chen, B. et al. (2018). *Activation Clustering: Detecting Poisoned Training Data*. 
2. Wang, B. et al. (2019). *Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks*. IEEE S&P. 
3. SCLBD. (2023). *BackdoorBench: A Comprehensive Benchmark of Backdoor Defenses*. 
4. Lee, K. et al. (2018). *A Simple Unified Framework for Detecting Out-of-Distribution Samples*. 
