# Fast-DetectGPT Multi-Model Support

## ✅ What Was Added

Added support for 3 different model sizes in Fast-DetectGPT detector:

1. **gpt2** (500MB) - Current default, fastest
2. **gpt-neo-2.7B** (5.5GB) - Official Fast-DetectGPT default
3. **falcon-7b** (14GB) - Best accuracy per ICLR 2024 paper

## 📁 Files Modified

### 1. `stealthrl/tinker/detectors.py`
- Added `MODEL_PATHS` dictionary mapping short names to HuggingFace paths
- Updated `FastDetectGPTDetector.__init__()` to resolve model names
- Added `trust_remote_code` support for Falcon models in `load_model_cached()`
- Updated logging to show model short names

### 2. `configs/tinker_stealthrl.yaml`
- Added `fast_detectgpt_model` config option
- Added comments showing all 3 model options
- Documented model sizes and trade-offs

### 3. `test_detectors_local.py`
- Updated to test all 3 Fast-DetectGPT model variants
- Updated size estimates (10-15GB total)
- Shows all 5 detector variants in results

### 4. `CURRENT_DETECTOR_STATUS.md`
- Updated Fast-DetectGPT section with 3 model options
- Added paper reference and recommendations
- Updated configuration examples

### 5. `test_fast_detectgpt_models.py` (NEW)
- New script to compare all 3 model sizes side-by-side
- Shows discrimination scores for each model
- Provides recommendations based on use case

## 🚀 Usage

### In Config File
```yaml
detectors:
  fast_detectgpt_model: "gpt2"  # Options: "gpt2", "gpt-neo-2.7B", "falcon-7b"
```

### In Code
```python
from stealthrl.tinker.detectors import FastDetectGPTDetector, DetectorCache

cache = DetectorCache()

# Option 1: GPT-2 (fast)
detector = FastDetectGPTDetector(cache, model_name="gpt2")

# Option 2: GPT-Neo-2.7B (default in paper)
detector = FastDetectGPTDetector(cache, model_name="gpt-neo-2.7B")

# Option 3: Falcon-7B (best accuracy)
detector = FastDetectGPTDetector(cache, model_name="falcon-7b")
```

### Testing Models
```bash
# Test all Fast-DetectGPT models
python test_fast_detectgpt_models.py

# Test all detectors (including 3 Fast-DetectGPT variants)
python test_detectors_local.py
```

## 📊 Model Comparison

| Model | Size | Speed | Accuracy | Recommendation |
|-------|------|-------|----------|----------------|
| gpt2 | 500MB | Fastest | Good | Development/iteration |
| gpt-neo-2.7B | 5.5GB | Medium | Better | Balanced |
| falcon-7b | 14GB | Slower | Best | Production/submission |

## 🎯 Recommendations

**For development**: Use `gpt2`
- Fast loading (~10 seconds)
- Low memory (500MB)
- Quick iterations

**For testing**: Use `gpt-neo-2.7B`
- Official Fast-DetectGPT default
- Good accuracy
- Reasonable speed

**For final submission**: Use `falcon-7b`
- Best accuracy per ICLR 2024 paper
- State-of-the-art performance
- Worth the extra compute time

## 📖 Reference

Based on: [Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature](https://github.com/baoguangsheng/fast-detect-gpt) (ICLR 2024)

The paper states: "So far the best sampling/scoring models we found for Fast-DetectGPT are falcon-7b/falcon-7b-instruct."

## 🔧 Technical Details

### Model Paths
- `gpt2` → `gpt2`
- `gpt-neo-2.7B` → `EleutherAI/gpt-neo-2.7B`
- `falcon-7b` → `tiiuae/falcon-7b`

### Special Handling
- Falcon models require `trust_remote_code=True`
- All models use fp16 on CUDA for efficiency
- Global model caching prevents reloading
- Thread-safe loading with locks

## ⚡ Next Steps

1. **Test locally**: Run `python test_fast_detectgpt_models.py`
2. **Choose model**: Based on speed/accuracy trade-off
3. **Update config**: Set `fast_detectgpt_model` in `configs/tinker_stealthrl.yaml`
4. **Train**: Run training with selected model size
5. **Evaluate**: Compare results across model sizes
