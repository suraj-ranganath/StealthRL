# Detector Lazy Loading Implementation

## Summary
Implemented lazy loading for the reranking detector in all attack methods. The detector now only loads when `n_candidates > 1`, avoiding unnecessary memory and initialization overhead when running with single-candidate generation.

## Changes Made

### 1. Simple Paraphrase (M1)
**File:** `/Users/suraj/Desktop/StealthRL/eval/methods/simple_paraphrase.py`

- **Removed** eager detector loading from `load()` method
- **Added** lazy loading in `_attack_impl()` with check:
  ```python
  if n_candidates > 1:
      if self.rerank_detector is None:
          from ..detectors import get_detector
          logger.info(f"Loading reranking detector: {self.rerank_detector_name}")
          self.rerank_detector = get_detector(self.rerank_detector_name, device=self.device)
          self.rerank_detector.load()
  ```

### 2. StealthRL (M2)  
**File:** `/Users/suraj/Desktop/StealthRL/eval/methods/stealthrl.py`

- **Removed** eager detector loading from `load()` method
- **Added** lazy loading in `_attack_impl()` (same pattern as M1)

### 3. AuthorMist (M4)
**File:** `/Users/suraj/Desktop/StealthRL/eval/methods/authormist.py`

- **Removed** eager detector loading from both `AuthorMistOllama.load()` and `AuthorMist.load()` methods
- **Added** lazy loading in both `_attack_impl()` methods (same pattern)

### 4. Adversarial Paraphrasing (M3)
**File:** `/Users/suraj/Desktop/StealthRL/eval/methods/adversarial_paraphrasing.py`

- **No changes needed** - this method already used the detector for selection and didn't have optional reranking

## Behavior

### Before
- Detector loaded eagerly during `method.load()` call
- All runs paid initialization cost (~2-5 seconds, GPU memory)
- Detector loaded even when `n_candidates=1` (not used)

### After  
- Detector is `None` after `method.load()`
- Only loads when first attack call has `n_candidates > 1`
- Runs with `n_candidates=1` have no detector overhead
- Log message: `"Reranking detector (roberta) will load lazily if n_candidates > 1"`

## Verification

**Test:** `test_lazy_init.py`

Results:
```
✓ simple_paraphrase: rerank_detector is None after load() - lazy loading works!
✓ authormist: rerank_detector is None after load() - lazy loading works!
```

## Usage Impact

### Baseline runs (n_candidates=1)
```bash
python scripts/run_eval.py --method simple_paraphrase --n-candidates 1
```
- ✅ No detector loading
- ✅ Faster startup  
- ✅ Lower memory usage

### Best-of-N runs (n_candidates > 1)
```bash
python scripts/run_eval.py --method simple_paraphrase --n-candidates 2
```
- ✅ Detector loads on first attack call
- ✅ Reranking enabled
- ✅ Same behavior as before, just delayed initialization

## Notes

- M0 (no_attack) and M5 (homoglyph) don't use detectors (no changes)
- M3 (adversarial_paraphrasing) always uses detector for its core logic (no lazy loading)
- All reranking still uses `roberta-large-openai-detector` (same as evaluation detector)
- Lazy loading happens once per run, then detector stays loaded for subsequent attacks
