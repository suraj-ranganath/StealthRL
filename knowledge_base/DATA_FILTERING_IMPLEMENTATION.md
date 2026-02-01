# Data Filtering Implementation Summary

## Problem Identified
Training logs showed "All rewards are uniform. There will be no gradient" warnings caused by corrupted samples containing gibberish text that made the model refuse to paraphrase instead of generating valid output.

### Example Corrupted Sample
```
Clearly, X is better than Y because:
1. First reason
2. Second reason
[...173 numbered items...]
Filipinsript GALAges Desifications Gatcalasio Snal only determined to try to the Nationalized...
```

## Solution Implemented

### File: `stealthrl/tinker/dataset.py`

Added comprehensive `is_valid_text()` function that filters out:

1. **Gibberish patterns** - Known corruption signatures:
   - `Filipinsript`
   - `GALAges`
   - `Desifications`
   - `Gatcalasio`
   - `usedmodified`

2. **Length issues**:
   - Too short: < 20 chars
   - Too long: > 3000 chars (likely corrupted lists)

3. **Structural problems**:
   - Excessive numbered lists (>50 items)
   - Low word count (< 5 recognizable words)
   - Low word-to-character ratio (< 40%)

4. **Generation artifacts**:
   - Repeated sentences (>3 times)

### Integration Points

The filter is applied in both data loading methods:

1. **`_load_jsonl_examples()`** - Lines 232-261
   - Validates both `ai_text` and `human_reference`
   - Skips invalid samples with debug logging

2. **`_load_mage_examples()`** - Lines 263-350
   - Validates text before creating StealthRLExample
   - Logs count of filtered samples
   - Uses fixed seed for shuffling (consistency)

## Dataset Consistency Answer

**YES** - The training dataset is consistent across runs.

The `StealthRLDatasetBuilder` uses a fixed seed (default=0) for shuffling, ensuring the same order every time. With the new filtering:
- Corrupted samples are removed consistently
- Same clean dataset on every run
- Reproducible training results

## Verification

Test script `analysis/test_filtering.py` confirms:
- ✅ Corrupted sample from training.log is **CORRECTLY REJECTED**
- ✅ Normal text passes through
- ✅ All gibberish patterns detected
- ✅ Excessive numbered lists caught

## Usage

The filtering is automatic - just load the dataset normally:

```python
builder = StealthRLDatasetBuilder(
    data_sources=['mage'],
    seed=42,  # Ensures consistency
    max_train_examples=10000,
)
dataset = builder.build()
```

Filtering stats will appear in logs:
```
INFO - Loaded 9847 AI-generated examples from MAGE dataset
INFO - Filtered out 153 invalid/corrupted samples
```

## Expected Impact

1. **No more uniform reward warnings** - Corrupted samples removed before training
2. **Stable gradients** - All samples now parseable by reward function
3. **Consistent results** - Same filtered dataset every run (with same seed)
4. **Better training** - No wasted gradient steps on garbage data

## Next Steps

After filtering is in place, you should see:
- No "All rewards are uniform" warnings
- Smoother training curves
- Better valid output rate
- More stable KL divergence
