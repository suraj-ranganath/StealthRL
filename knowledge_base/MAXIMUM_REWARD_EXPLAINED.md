# Maximum Possible Reward - StealthRL

## Reward Formula

From [stealthrl/tinker/reward.py](../stealthrl/tinker/reward.py):

```
R_total = α·R_det + β·R_sem + γ·R_ppl - δ·F'
```

Where:
- `R_det = 1 - P(AI)` (detector evasion)
- `R_sem = max(0, similarity - semantic_min)` (semantic similarity)
- `R_ppl = perplexity_reward(ppl)` (fluency)
- `F' = P(AI) · 𝟙[is_ESL]` (fairness penalty)

## Component Maximums

### 1. Detector Reward (R_det)
**Raw maximum**: `R_det_raw = 1 - 0 = 1.0`
- When all detectors output P(AI) = 0 (100% human-like)

**After normalization** (if `detector_zscore=True`):
```python
R_det' = clip((R_det - μ) / (σ + ε), -3, 3)
```
- Maximum: **3.0** (3 std above mean)
- Typical: ~1-2 for very good evasion

### 2. Semantic Reward (R_sem)
**Raw maximum**: `R_sem_raw = max(0, 1.0 - 0.90) = 0.10`
- When similarity = 1.0 (perfect match) and semantic_min = 0.90

**After normalization** (if `normalize_terms=True`):
```python
R_sem' = (score / (1.0 - semantic_min)) if score >= 0 else 0
R_sem' = 0.10 / (1.0 - 0.90) = 0.10 / 0.10 = 1.0
```
- Maximum: **1.0**

### 3. Perplexity Reward (R_ppl)
**Raw maximum**: `R_ppl_raw = 1.0`
- When perplexity exactly equals target (e.g., 30.0)
- Formula: `reward = 1.0 - (distance / max_distance)`

**After normalization** (if `normalize_terms=True`):
```python
# Assumes quality_min = 0.80
R_ppl' = (score - 0.80) / (1.0 - 0.80) if score >= 0.80 else 0
R_ppl' = (1.0 - 0.80) / 0.20 = 1.0
```
- Maximum: **1.0**

### 4. Fairness Penalty (F')
**Minimum penalty**: `F' = 0.0`
- For non-ESL samples (`is_esl = False`)
- Or when P(AI) = 0 for ESL samples

## Theoretical Maximum

With **default weights** from configs:
```python
detector_weight = 1.0  (α)
semantic_weight = 1.0  (β)
perplexity_weight = 0.5  (γ)
fairness_weight = 0.2  (δ)
```

### Case 1: No Normalization (`normalize_terms=False`)
```
R_max = 1.0 * 1.0 + 1.0 * 0.10 + 0.5 * 1.0 - 0.2 * 0.0
R_max = 1.0 + 0.10 + 0.5 + 0
R_max = 1.60
```

### Case 2: With Normalization (`normalize_terms=True, detector_zscore=False`)
```
R_max = 1.0 * 1.0 + 1.0 * 1.0 + 0.5 * 1.0 - 0.2 * 0.0
R_max = 1.0 + 1.0 + 0.5 + 0
R_max = 2.50
```

### Case 3: With Z-Score Normalization (`detector_zscore=True`)
```
R_max = 1.0 * 3.0 + 1.0 * 1.0 + 0.5 * 1.0 - 0.2 * 0.0
R_max = 3.0 + 1.0 + 0.5 + 0
R_max = 4.50
```

## Practical Maximum

In practice, you'll see rewards around:

**Good performance**:
- R_det ~ 0.7-0.9 (P(AI) = 0.1-0.3)
- R_sem ~ 0.8-1.0 (similarity = 0.88-1.0)
- R_ppl ~ 0.7-1.0 (PPL = 25-35)
- F' ~ 0.0-0.1

**Expected range**: **1.0 - 2.5** for typical good outputs

**From your training log**:
- Rewards around 0.9-1.1 are typical
- The corrupted sample caused all rewards to be uniform (0.418-0.538)
- After filtering, you should see more variance and higher peaks

## Summary Table

| Configuration | Detector (α=1.0) | Semantic (β=1.0) | PPL (γ=0.5) | Fair (δ=0.2) | **Total Max** |
|---------------|------------------|------------------|-------------|--------------|---------------|
| No normalization | 1.0 | 0.10 | 0.5 | 0.0 | **1.60** |
| Normalized | 1.0 | 1.0 | 0.5 | 0.0 | **2.50** |
| Z-score | 3.0 | 1.0 | 0.5 | 0.0 | **4.50** |

**Your config likely uses** normalization without z-score, so:
### Maximum Possible Reward ≈ **2.50**

But practically, you'll see:
- **Excellent**: 1.5-2.0
- **Good**: 1.0-1.5
- **Acceptable**: 0.5-1.0
- **Poor**: < 0.5
