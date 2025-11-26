# StealthRL Reward Refinement Summary

**Date**: November 25, 2025
**Purpose**: Refine StealthRL reward implementation for better RL stability and learnability

---

## Changes Made

### 1. Enhanced Composite Reward (`stealthrl/rewards/composite_reward.py`)

**New Features**:
- **Z-score normalization** for detector scores (μ, σ clipping to [-3, 3])
- **Threshold-based normalization** for semantic and quality scores
- **Configurable minimum thresholds** to prevent degenerate solutions
- **AuthorMist-inspired design** with explicit references

**New Formula**:
```
R = -w₁·D' + w₂·S' + w₃·Q' - w₄·F'

Where:
  D' = z-score(detector_scores) if detector_zscore else detector_scores
  S' = clamp((S - semantic_min) / (1 - semantic_min), 0, 1)
  Q' = clamp((Q - quality_min) / (1 - quality_min), 0, 1)
  F' = per-sample ESL penalty (not global gap)
```

**New Parameters**:
- `normalize_terms: bool` - Enable/disable normalization
- `detector_zscore: bool` - Apply z-score to detectors  
- `semantic_min: float` - Minimum acceptable semantic similarity (default 0.90)
- `quality_min: float` - Minimum acceptable quality (default 0.80)

**Benefits**:
- More stable gradients for GRPO/PPO
- Prevents semantic/quality collapse below thresholds
- Better scale invariance across reward components

---

### 2. Per-Sample Fairness Proxy (`stealthrl/rewards/fairness_metrics.py`)

**New Module**: `fairness_metrics.py`

**Key Functions**:

#### `compute_fairness_proxy(detector_scores, group_labels, mode="esl_penalty")`
Returns **per-sample** fairness penalties for RL optimization:
```python
F'_i = detector_scores[i] if group_labels[i] == 1 (ESL)
     = 0 otherwise
```

**Why This Matters**:
- Global FPR gaps are not differentiable → can't be used in RL loss
- Per-sample proxy provides gradients for each training example
- Encourages policy to suppress ESL detection more aggressively

#### `compute_group_fpr_gap(predictions, labels, group_labels, threshold)`
Computes **global** ESL vs native FPR gap for **evaluation only** (not training).

**Integration**:
- Use `compute_fairness_proxy()` in training loop
- Use `compute_group_fpr_gap()` for reporting and model selection

---

### 3. Explicit Quality Normalization (`stealthrl/rewards/quality_reward.py`)

**Enhanced Formula**:
```
Q = α·(1 - minmax_norm(perplexity)) + (1-α)·minmax_norm(readability)

minmax_norm(x) = clamp(x, x_min, x_max) / (x_max - x_min)
```

**New Parameters**:
- `perplexity_min: float` - Min perplexity for normalization (default 5.0)
- `perplexity_max: float` - Max perplexity for normalization (default 80.0)
- `readability_min: float` - Min Flesch score (default 0.0)
- `readability_max: float` - Max Flesch score (default 100.0)
- `quality_balance: float` - α weight between perplexity and readability (default 0.5)

**Benefits**:
- Predictable [0, 1] range for quality scores
- Configurable bounds handle domain-specific perplexity ranges
- Explicit balance between perplexity and readability

---

### 4. KL Regularization in Trainer (`stealthrl/training/trainer.py`)

**New Features**:
- **Reference model** parameter for KL divergence computation
- **KL penalty term** added to loss: `loss = policy_loss + β·KL(π || π_ref)`
- **Default β = 0.001** (following AuthorMist)

**Implementation**:
```python
def _compute_kl_divergence(policy_logits, ref_logits):
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_probs = F.softmax(ref_logits, dim=-1)
    kl = sum(ref_probs * (log(ref_probs) - policy_log_probs))
    return kl.mean()
```

**Benefits**:
- Keeps policy close to base LM → preserves fluency
- Prevents mode collapse and reward hacking
- Follows RLHF best practices from InstructGPT, AuthorMist

**Documentation**:
Added explicit reference to AuthorMist (https://arxiv.org/abs/2503.08716) in docstrings.

---

### 5. Updated Configuration (`configs/stealthrl_small.yaml`)

**New Sections**:

```yaml
trainer:
  kl_beta: 0.001  # KL penalty coefficient

reward:
  # Component weights
  detector_weight: 1.0
  semantic_weight: 1.0
  quality_weight: 0.5
  fairness_weight: 0.2
  
  # NEW: Normalization settings
  normalize_terms: true
  detector_zscore: true
  semantic_min: 0.90
  quality_min: 0.80
  fairness_mode: "esl_penalty"

quality:
  # Explicit bounds for min-max normalization
  perplexity_min: 5.0
  perplexity_max: 80.0
  readability_min: 0.0
  readability_max: 100.0
  quality_balance: 0.5
```

---

## Mathematical Summary

### Before (Original)
```
R = -w₁·D + w₂·S + w₃·Q - w₄·F_global

Issues:
- No normalization → unstable scales
- F_global is not differentiable per-sample
- No thresholding → semantic/quality can collapse
```

### After (Refined)
```
R = -w₁·D' + w₂·S' + w₃·Q' - w₄·F'

Where:
  D' = zscore(D).clamp(-3, 3)
  S' = ((S - 0.90) / 0.10).clamp(0, 1)  
  Q' = ((Q - 0.80) / 0.20).clamp(0, 1)
  F' = D * 𝟙[group=ESL]  (per-sample)

Training Loss:
  L = -E[R] + β·KL(π || π_ref)
  
Where β = 0.001 (AuthorMist-inspired)

Benefits:
✓ Stable gradients for GRPO/PPO
✓ Prevents degenerate solutions
✓ Learnable fairness optimization
✓ Preserves fluency via KL penalty
```

---

## Impact on Research Goals

### 1. **More Stable RL Training**
- Z-score normalization prevents reward scale issues
- Thresholding prevents collapse below acceptable quality
- KL regularization prevents mode collapse

### 2. **Learnable Fairness**
- Per-sample ESL penalty provides gradients
- Policy can directly optimize to reduce ESL bias
- Global FPR gap still tracked for evaluation

### 3. **Better Transfer**
- Quality/semantic thresholds prevent overfitting to detectors
- KL penalty keeps policy close to base LM fluency
- More robust paraphrases that preserve meaning

### 4. **Follows Best Practices**
- AuthorMist: KL regularization (β=0.001)
- InstructGPT: KL penalty for RLHF
- PPO/GRPO: Normalized rewards for stability

---

## Files Modified

1. ✅ `stealthrl/rewards/composite_reward.py` (+70 lines, normalized reward)
2. ✅ `stealthrl/rewards/fairness_metrics.py` (NEW, +100 lines, per-sample proxy)
3. ✅ `stealthrl/rewards/quality_reward.py` (+40 lines, explicit normalization)
4. ✅ `stealthrl/training/trainer.py` (+50 lines, KL regularization)
5. ✅ `configs/stealthrl_small.yaml` (+20 lines, new parameters)
6. ✅ `stealthrl/rewards/__init__.py` (expose fairness_metrics)
7. ✅ `stealthrl/evaluation/metrics.py` (add global FPR gap function)

**Total**: +280 lines of refined reward logic

---

## Usage Example

```python
from stealthrl.rewards import CompositeReward, compute_fairness_proxy

# Initialize with refined parameters
reward_fn = CompositeReward(
    detector_weight=1.0,
    semantic_weight=1.0,
    quality_weight=0.5,
    fairness_weight=0.2,
    normalize_terms=True,        # NEW
    detector_zscore=True,        # NEW
    semantic_min=0.90,           # NEW
    quality_min=0.80,            # NEW
)

# In training loop:
detector_scores = detector_ensemble(paraphrases)  # [batch]
semantic_scores = bertscore(paraphrases, originals)  # [batch]
quality_scores = quality_fn(paraphrases)  # [batch]

# Compute per-sample fairness proxy (NEW)
group_labels = torch.tensor([1, 0, 1, 0, ...])  # 1=ESL, 0=native
fairness_scores = compute_fairness_proxy(detector_scores, group_labels)  # [batch]

# Compute reward
rewards = reward_fn.compute(
    detector_scores,
    semantic_scores,
    quality_scores,
    fairness_scores  # Now per-sample, not global gap!
)

# In evaluation:
from stealthrl.rewards import compute_group_fpr_gap
fpr_esl, fpr_native, gap = compute_group_fpr_gap(preds, labels, groups)
print(f"ESL FPR Gap: {gap:.3f}")
```

---

## Configuration Tuning Guide

### Conservative (High Quality)
```yaml
reward:
  semantic_min: 0.95  # Very strict semantic preservation
  quality_min: 0.85   # High quality requirement
  detector_zscore: true
  
trainer:
  kl_beta: 0.005      # Strong KL penalty
```

### Aggressive (Max Evasion)
```yaml
reward:
  semantic_min: 0.85  # Looser semantic constraint
  quality_min: 0.75   # Looser quality constraint
  detector_zscore: true
  
trainer:
  kl_beta: 0.0005     # Weak KL penalty
```

### Fairness-First
```yaml
reward:
  fairness_weight: 0.5  # Increase fairness weight
  fairness_mode: "esl_penalty"
  semantic_min: 0.90
  quality_min: 0.80
```

---

## References

1. **AuthorMist** (arXiv:2503.08716) - KL regularization (β=0.001), detector-based RL rewards
2. **InstructGPT** (arXiv:2203.02155) - KL penalty in RLHF
3. **GRPO** (DeepSeek R1) - Group Relative Policy Optimization
4. **PPO** (Schulman et al.) - Normalized rewards for stability

---

## Next Steps

1. ✅ All code implemented and configured
2. ⏳ Test with small-scale training run
3. ⏳ Tune thresholds (`semantic_min`, `quality_min`) based on validation
4. ⏳ Ablate KL beta (0.0001, 0.001, 0.01) to find optimal trade-off
5. ⏳ Compare refined vs original reward on same dataset
6. ⏳ Document improvements in technical report

**Status**: Ready for training! 🚀
