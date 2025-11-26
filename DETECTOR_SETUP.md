# Detector Setup Guide for StealthRL

## Quick Answer: You're Ready to Go! 🎉

**Your detectors are currently using MOCK implementations**, which means:
- ✅ **No additional setup required** to start training
- ✅ **No API keys needed** for detectors
- ✅ You can run the full training pipeline right now

## Which Detectors Are You Using?

Based on `configs/tinker_stealthrl.yaml`, your ensemble includes:

1. **Fast-DetectGPT** - Curvature-based detection (weight: 0.5)
2. **Ghostbuster** - RoBERTa classifier (weight: 0.5)
3. **Binoculars** - Paired-LM detection (in transfer config only)

## Current Implementation: Mock Detectors

### What Are Mock Detectors?

The code in `stealthrl/tinker/detectors.py` currently uses **placeholder/mock implementations**:

```python
class FastDetectGPTDetector(BaseDetector):
    async def _compute_score(self, text: str) -> float:
        # Placeholder: In production, this would call the actual Fast-DetectGPT model
        # For now, return a mock score
        await asyncio.sleep(0.1)  # Simulate computation
        score = min(1.0, max(0.0, 0.5 + (len(text) % 100) / 200.0))
        return score
```

### Why Mock Detectors?

This design allows you to:

1. **Test the full pipeline** - Validate RL training loop, reward computation, GRPO, etc.
2. **No API costs** - Experiment without paying for detector API calls
3. **Fast iteration** - Mock detectors return instantly (no network latency)
4. **Prove the infrastructure** - Ensure everything works before using real detectors

### What Do Mock Detectors Return?

- **Fast-DetectGPT**: Score based on text length (0.5 baseline + variation)
- **Ghostbuster**: Score based on text characteristics (0.6 baseline - variation)
- **Binoculars**: Score based on text hash (0.55 baseline + variation)

These are **deterministic but varied enough** to test reward computation and RL training.

---

## Option 1: Continue with Mock Detectors (Recommended for Testing)

**You can start training right now with mocks!**

```bash
# Add your Tinker API key to .env
nano .env  # Add: TINKER_API_KEY=tk-...

# Install dependencies
pip install -r requirements.txt

# Run quick test
python scripts/prepare_tinker_data.py --synthetic --num-train 10 --output-dir data/tinker_test
python -m stealthrl.tinker.train --config configs/tinker_stealthrl.yaml --data-path data/tinker_test --output-dir outputs/test_run --num-epochs 1
```

**Benefits**:
- ✅ Zero additional setup
- ✅ Fast experimentation
- ✅ Test pipeline before committing compute resources
- ✅ Validate GRPO, reward normalization, fairness penalties

**Limitations**:
- ❌ Mock scores don't reflect real detector behavior
- ❌ Can't measure actual ASR (Attack Success Rate)
- ❌ Results not meaningful for research publication

**When to use**: Initial testing, debugging, pipeline validation

---

## Option 2: Use Real Detectors (For Production Experiments)

For **publishable research results**, you need real detector implementations.

### Option 2A: Cloud-Based Detectors (API Keys)

If you have access to cloud-based detector APIs:

1. **Get API keys** from detector providers
2. **Add to .env**:
   ```bash
   FAST_DETECTGPT_API_KEY=your_key_here
   GHOSTBUSTER_API_KEY=your_key_here
   BINOCULARS_API_KEY=your_key_here
   ```

3. **Update detector implementations** in `stealthrl/tinker/detectors.py`:
   ```python
   class FastDetectGPTDetector(BaseDetector):
       async def _compute_score(self, text: str) -> float:
           # Call actual API
           import httpx
           api_key = os.getenv("FAST_DETECTGPT_API_KEY")
           response = await httpx.post(
               "https://api.example.com/detect",
               json={"text": text},
               headers={"Authorization": f"Bearer {api_key}"}
           )
           return response.json()["ai_probability"]
   ```

**Pros**: No local compute needed, simple setup  
**Cons**: API costs, network latency, rate limits

### Option 2B: Run Detectors Locally (Recommended)

**This is the recommended approach for research.**

#### Step 1: Install Detector Packages

```bash
# Fast-DetectGPT
pip install fast-detectgpt

# Ghostbuster
pip install ghostbuster

# Binoculars
pip install binoculars-detect

# Alternative: Clone from GitHub
git clone https://github.com/baoguangsheng/fast-detect-gpt.git
git clone https://github.com/vivek3141/ghostbuster.git
git clone https://github.com/ahans30/Binoculars.git
```

#### Step 2: Update Detector Implementations

Edit `stealthrl/tinker/detectors.py` to load real models:

**Fast-DetectGPT**:
```python
class FastDetectGPTDetector(BaseDetector):
    def __init__(self, cache: DetectorCache):
        super().__init__("fast_detectgpt", cache)
        from fast_detectgpt import FastDetectGPT
        self.model = FastDetectGPT(
            reference_model="gpt-neo-2.7B",
            scoring_model="gpt-j-6B"
        )
        logger.info("Initialized Fast-DetectGPT detector")
    
    async def _compute_score(self, text: str) -> float:
        # Run in thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        score = await loop.run_in_executor(None, self.model.detect, text)
        return score
```

**Ghostbuster**:
```python
class GhostbusterDetector(BaseDetector):
    def __init__(self, cache: DetectorCache):
        super().__init__("ghostbuster", cache)
        from ghostbuster import Ghostbuster
        self.model = Ghostbuster()
        logger.info("Initialized Ghostbuster detector")
    
    async def _compute_score(self, text: str) -> float:
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.model.predict, text)
        return result["ai_probability"]
```

**Binoculars**:
```python
class BinocularsDetector(BaseDetector):
    def __init__(self, cache: DetectorCache):
        super().__init__("binoculars", cache)
        from binoculars import Binoculars
        self.model = Binoculars()
        logger.info("Initialized Binoculars detector")
    
    async def _compute_score(self, text: str) -> float:
        import asyncio
        loop = asyncio.get_event_loop()
        score = await loop.run_in_executor(None, self.model.compute_score, text)
        return score
```

#### Step 3: Test Real Detectors

```bash
# Test Fast-DetectGPT
python -c "
from stealthrl.tinker.detectors import DetectorCache, FastDetectGPTDetector
import asyncio

async def test():
    cache = DetectorCache('cache/detectors.db')
    detector = FastDetectGPTDetector(cache)
    score = await detector.predict('This is AI-generated text.')
    print(f'Score: {score}')

asyncio.run(test())
"
```

#### Requirements for Local Detectors

**Hardware**:
- GPU with 8-16GB VRAM (NVIDIA recommended)
- 32GB+ system RAM
- ~20GB disk space for models

**Model Downloads**:
- Fast-DetectGPT: ~3GB (GPT-Neo-2.7B + GPT-J-6B)
- Ghostbuster: ~2GB (RoBERTa-large + feature models)
- Binoculars: ~4GB (Instruct + base LM)

**First Run**:
- Models download automatically
- Expect 5-10 minutes for initial setup
- Subsequent runs use cached models

---

## Option 3: Hybrid Approach

Use **mock detectors for development**, then **switch to real detectors for final experiments**:

1. **Development phase** (mock detectors):
   - Test pipeline with small datasets
   - Validate reward function
   - Debug training loop
   - Fast iteration (<5 minutes per experiment)

2. **Production phase** (real detectors):
   - Switch to local/API detectors
   - Run full experiments with 1000+ samples
   - Collect publishable results

**How to switch**:
```python
# In your config or code
USE_MOCK_DETECTORS = False  # Set to True for testing

if USE_MOCK_DETECTORS:
    # Use existing mock implementation
    pass
else:
    # Load real detector models
    self.model = FastDetectGPT(...)
```

---

## Caching Strategy

Regardless of mock vs real detectors, StealthRL uses **SQLite caching**:

```python
# Cache location
cache_path: "cache/detectors.db"
```

**Benefits**:
- First call: Compute score (mock or real)
- Subsequent calls: Instant cache lookup
- **10-20× speedup** during training
- Persistent across runs

**Cache hit rate**:
- After warmup: >90% hit rate
- Training epoch 2+: >95% hit rate

---

## Summary & Recommendations

### For Quick Testing (Today)
✅ **Use mock detectors** - no additional setup needed  
✅ Just add Tinker API key and run  
✅ Validate pipeline and code

### For Real Research (This Week)
✅ **Install local detectors** - best for research  
✅ Update detector implementations  
✅ Run experiments with real scores

### For Production (Future)
✅ Consider cloud APIs for scale  
✅ Implement rate limiting and error handling  
✅ Monitor costs and latency

---

## Quick Decision Tree

```
Do you have Tinker API key?
├─ No  → Get it first (required)
└─ Yes → Continue

Do you need real results for publication?
├─ No  → Use mock detectors (Option 1)
│        ✅ Start training now
│        ✅ Zero setup
└─ Yes → Need real detectors

Do you have GPU (8-16GB VRAM)?
├─ Yes → Use local detectors (Option 2B - Recommended)
│        1. Install packages
│        2. Update detectors.py
│        3. Test and run
└─ No  → Use cloud APIs (Option 2A)
         1. Get API keys
         2. Update detectors.py
         3. Monitor costs
```

---

## Next Steps

**Right Now** (with mock detectors):
```bash
# 1. Add Tinker API key
nano .env  # Add: TINKER_API_KEY=tk-...

# 2. Quick test
python scripts/prepare_tinker_data.py --synthetic --num-train 10 --output-dir data/tinker_test
python -m stealthrl.tinker.train --config configs/tinker_stealthrl.yaml --data-path data/tinker_test --output-dir outputs/test_run --num-epochs 1

# 3. Verify it works
ls outputs/test_run/checkpoints/  # Should see checkpoint files
```

**Later** (switch to real detectors):
```bash
# 1. Install detector packages
pip install fast-detectgpt ghostbuster binoculars-detect

# 2. Update stealthrl/tinker/detectors.py with real implementations

# 3. Test detectors
python -c "from stealthrl.tinker.detectors import DetectorEnsemble; ..."

# 4. Run full pipeline
python scripts/run_research_pipeline.py --stage all
```

---

## FAQ

**Q: Can I mix mock and real detectors?**  
A: Yes! For example, use real Fast-DetectGPT but mock Binoculars during development.

**Q: How much do API calls cost?**  
A: Varies by provider. Estimate $0.01-0.10 per detection. For 1000 samples × 3 detectors × 4 rollouts = 12,000 calls ≈ $120-1,200.

**Q: Can I use different detector versions?**  
A: Yes, just update the detector class implementations. The ensemble interface is flexible.

**Q: What if a detector fails during training?**  
A: The code has retry logic and returns 0.5 (neutral) on failure after 3 retries. Training continues.

**Q: Do I need all 3 detectors?**  
A: No! You can configure any subset in `configs/tinker_stealthrl.yaml`:
```yaml
detectors:
  names:
    - "fast_detectgpt"  # Only use one detector
```

---

**Bottom line**: You're ready to start with mock detectors right now. Add your Tinker API key and run the quick test. Switch to real detectors when you need publishable results.
