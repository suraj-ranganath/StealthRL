# StealthRL Ablation Studies

This directory contains configuration files for systematic ablation experiments.

## Overview

Ablations test the contribution of each component in StealthRL's composite reward function:

```
Full Reward = w₁·(-detector) + w₂·semantic + w₃·quality + w₄·(-fairness)
```

Each ablation isolates one component to understand its impact on:
- **Evasion effectiveness** (detector scores)
- **Semantic fidelity** (BERTScore)
- **Output quality** (perplexity, readability)
- **Fairness** (ESL vs native FPR gap)
- **Cross-detector transfer** (held-out detector performance)

---

## Available Ablations

### 1. `single_detector_fast_detectgpt.yaml`
**Tests**: Ensemble training vs single-detector training

**Hypothesis**: Training against multiple detector families should improve cross-detector transfer compared to single-detector optimization.

**Configuration**:
- Detectors: Only Fast-DetectGPT (curvature-based)
- Weights: detector=0.4, semantic=0.3, quality=0.2, fairness=0.1

**Expected Result**: Worse performance on held-out Binoculars (paired-LM) and Ghostbuster (classifier) compared to ensemble baseline.

---

### 2. `no_fairness.yaml`
**Tests**: Impact of ESL fairness penalty

**Hypothesis**: Removing fairness term will increase ESL vs native FPR gap (more bias).

**Configuration**:
- Detectors: Fast-DetectGPT + RoBERTa classifier
- Weights: detector=0.5, semantic=0.35, quality=0.15, **fairness=0.0**

**Expected Result**: Higher ESL FPR gap, potentially better evasion but at cost of increased bias against non-native writing.

---

### 3. `no_semantic.yaml`
**Tests**: Impact of semantic fidelity constraint

**Hypothesis**: Removing BERTScore/cosine similarity allows better evasion but may produce semantically divergent outputs.

**Configuration**:
- Detectors: Fast-DetectGPT + RoBERTa classifier
- Weights: detector=0.5, **semantic=0.0**, quality=0.4, fairness=0.1

**Expected Result**: Lower detector scores but also lower BERTScore (meaning drift).

---

### 4. `no_quality.yaml`
**Tests**: Impact of quality constraints

**Hypothesis**: Removing perplexity/readability constraints allows better evasion but may degrade fluency.

**Configuration**:
- Detectors: Fast-DetectGPT + RoBERTa classifier
- Weights: detector=0.5, semantic=0.4, **quality=0.0**, fairness=0.1

**Expected Result**: Lower detector scores but higher perplexity (less fluent outputs).

---

### 5. `detector_only.yaml`
**Tests**: Pure evasion without any constraints

**Hypothesis**: Optimizing only for detectability will produce degenerate outputs (low quality, semantic drift, high perplexity).

**Configuration**:
- Detectors: Fast-DetectGPT + RoBERTa classifier
- Weights: **detector=1.0**, semantic=0.0, quality=0.0, fairness=0.0

**Expected Result**: Best evasion scores but worst quality/semantic fidelity. This is the "unconstrained attacker" baseline.

---

## Running Ablations

### Run All Ablations
```bash
bash scripts/run_ablations.sh
```

This sequentially trains all 5 ablations and logs outputs to `outputs/ablations/`.

### Run Individual Ablation
```bash
python scripts/train_stealthrl.py --config configs/ablations/<ablation_name>.yaml
```

### Evaluate All Ablations
```bash
python scripts/evaluate_ablations.py \
    --ablation_dir checkpoints \
    --test_data data/processed/test.jsonl \
    --esl_data data/processed/esl_test.jsonl \
    --native_data data/processed/native_test.jsonl \
    --output_dir outputs/ablations
```

Produces:
- `ablation_results.csv` - quantitative comparison table
- `ablation_detector_scores.png` - detector evasion comparison
- `ablation_bertscore.png` - semantic fidelity comparison
- `ablation_fairness_gap.png` - ESL fairness comparison

---

## Interpretation Guide

### Pareto Frontier Analysis
Plot ablation results in 2D space to visualize trade-offs:

**Detectability vs Semantic Fidelity**:
- X-axis: Mean detector score (lower = better evasion)
- Y-axis: BERTScore F1 (higher = better meaning preservation)
- Ideal: Lower-right corner (low detectability, high fidelity)

**Detectability vs Fairness**:
- X-axis: Mean detector score
- Y-axis: FPR gap (lower = more fair)
- Ideal: Lower-left corner (low detectability, low bias)

### Expected Findings

| Ablation | Evasion | Semantic | Quality | Fairness | Transfer |
|----------|---------|----------|---------|----------|----------|
| Baseline (full) | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Single Detector | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| No Fairness | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| No Semantic | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| No Quality | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Detector Only | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐⭐ |

---

## Research Questions

These ablations help answer:

1. **Q1**: Does ensemble training improve cross-detector transfer?
   - Compare `single_detector_fast_detectgpt` vs `baseline` on held-out Binoculars

2. **Q2**: What is the trade-off between evasion and fairness?
   - Compare `no_fairness` vs `baseline` on ESL FPR gap

3. **Q3**: Are semantic and quality constraints necessary?
   - Compare `detector_only` vs `baseline` on BERTScore and perplexity

4. **Q4**: Which reward component is most important for cross-detector transfer?
   - Compare all ablations on held-out detector AUROC

5. **Q5**: Can we achieve detector-agnostic evasion strategies?
   - Analyze if ensemble-trained model learns general perturbations vs detector-specific exploits

---

## Notes

- All ablations use the same base model (Qwen 1.5B) and LoRA config for fair comparison
- Training steps (10k) and hyperparameters are kept constant
- Evaluation always uses the same test set and held-out detectors
- For publication, run each ablation with 3 random seeds and report mean ± std
