# ESL Fairness Removal & MAGE Dataset Integration

## Summary of Changes

Removed all ESL-specific fairness components from the reward system and integrated MAGE dataset support (HuggingFace format). The system is now simplified to focus on general defensive paraphrasing across diverse text domains.

## Files Modified

### 1. **stealthrl/tinker/reward.py**
- ✅ Removed `fairness_weight` parameter from `__init__`
- ✅ Removed fairness penalty computation (was: `fairness_penalty = detector_prob if is_esl else 0.0`)
- ✅ Removed `fairness_reward` from reward return dictionary
- ✅ Removed `is_esl_flags` parameter from `compute_batch()` method signature
- ✅ Simplified total reward to: `R = α*R_det + β*R_sem + γ*R_ppl` (removed δ*R_fair term)
- ✅ Updated docstrings to remove fairness references

**Before:** 
```python
total_reward = (
    self.detector_weight * detector_reward +
    self.semantic_weight * semantic_reward +
    self.perplexity_weight * ppl_reward -
    self.fairness_weight * fairness_penalty  # REMOVED
)
```

**After:**
```python
total_reward = (
    self.detector_weight * detector_reward +
    self.semantic_weight * semantic_reward +
    self.perplexity_weight * ppl_reward
)
```

### 2. **stealthrl/tinker/env.py**
- ✅ Removed `is_esl: bool` parameter from `StealthEnv.__init__`
- ✅ Removed `is_esl` attribute assignment
- ✅ Removed `is_esl` from `StealthEnvGroupBuilder` dataclass
- ✅ Removed `is_esl` from `make_envs()` method
- ✅ Removed `esl_flags` list tracking in `compute_group_rewards`
- ✅ Removed `is_esl` from reward computation calls
- ✅ Updated docstrings to remove ESL references

### 3. **stealthrl/tinker/dataset.py**
- ✅ Removed `is_esl: bool` field from `StealthRLExample` dataclass
- ✅ Removed `is_esl` parameter from environment builder calls
- ✅ **NEW:** Added support for MAGE dataset (HuggingFace format)
- ✅ **REFACTORED:** Split `_load_examples()` into:
  - `_load_examples()` - router that detects format
  - `_load_jsonl_examples()` - handles Tinker format
  - `_load_mage_examples()` - handles MAGE HuggingFace format

## New MAGE Dataset Support

### How it works:
1. Detects dataset format automatically:
   - If `{split}.jsonl` exists → loads as JSONL (Tinker format)
   - If data_path is "data/mage" → loads from HuggingFace (MAGE format)

2. MAGE dataset characteristics:
   - Format: HuggingFace dataset with columns [text, label, src]
   - Label: 1=human text (we use these), 0=AI text
   - Src: source identifier (e.g., "eli5_human", "gpt3_davinci_002")
   - Only human-written text (label=1) is extracted for DEFENSIVE training

3. Domain extraction:
   - Automatically infers domain from src field
   - Maps: eli5→informal, hswag→reasoning, xsum→news, etc.
   - Falls back to source prefix if mapping not found

### Example usage:
```python
builder = StealthRLDatasetBuilder(
    data_path="data/mage",  # ← Auto-detects MAGE format
    batch_size=32,
    group_size=4,
    # ... other params
)

# Loads from: data/mage/test (only human texts)
train_dataset, test_dataset = await builder()
```

## Data Format Support

### JSONL Format (Tinker - unchanged):
```json
{
    "ai_text": "...",
    "human_reference": "...",
    "domain": "academic",
    "metadata": {"source": "..."}
}
```

### MAGE Format (HuggingFace - new):
```python
# From HuggingFace dataset
{
    "text": "The novel explores...",  → becomes human_reference
    "label": 1,                       → only use label=1 (human)
    "src": "eli5_human"               → infer domain + metadata
}
```

## Reward Function Changes

### Old Reward (with fairness):
- R = 1.0 * R_det + 1.0 * R_sem + 0.5 * R_ppl + 0.2 * R_fair
- R_fair penalized high detection scores for ESL texts specifically
- Tracked is_esl flag throughout pipeline

### New Reward (simplified):
- R = 1.0 * R_det + 1.0 * R_sem + 0.5 * R_ppl  
- Treats all human texts equally
- No special handling for ESL vs native writers
- Simpler, more general approach

## Testing

Created `test_mage_loading.py` to verify:
- Dataset detection works (auto-routes to MAGE loader)
- Domain extraction works
- Batch creation works with MAGE format
- No regressions with JSONL format

## Next Steps

1. ✅ Changes complete and ready to test
2. ⏭️ Run training with MAGE dataset:
   ```bash
   python -m stealthrl.tinker.train \
     --config configs/tinker_stealthrl.yaml \
     --dataset.data_path data/mage \
     --dataset.batch_size 32
   ```
3. ⏭️ Monitor detector/semantic/perplexity metrics (no fairness metrics anymore)
4. ⏭️ Evaluate paraphrase quality across all MAGE domains

## Key Differences From Previous Version

| Aspect | Before | After |
|--------|--------|-------|
| ESL Support | ✅ Special fairness penalty | ❌ Removed |
| Datasets | Tinker only (20K samples) | ✅ Tinker + MAGE (60K+ samples) |
| Reward terms | 4 (det, sem, ppl, fair) | 3 (det, sem, ppl) |
| Domain coverage | Limited (academic only) | ✅ 14 diverse domains |
| Fairness tracking | ✅ Per-example is_esl | ❌ Not tracked |

## Backward Compatibility

- ✅ Old JSONL format (Tinker) still works
- ✅ All existing APIs maintained (except `is_esl` parameter removal)
- ✅ Reward computation simpler but API-compatible
- ✅ Can easily switch data_path to use either dataset
