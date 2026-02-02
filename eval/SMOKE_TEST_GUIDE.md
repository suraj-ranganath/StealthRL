# Smoke Test Guide for StealthRL Eval Pipeline

## Hardware Requirements Summary

| Component | Mac M4 Pro (48GB) | A5000 GPU (DSMLP) |
|-----------|:-----------------:|:-----------------:|
| Data Loading (MAGE/RAID/PadBen) | ✅ | ✅ |
| RoBERTa Detector (~1.4GB) | ✅ | ✅ |
| Fast-DetectGPT (GPT-2 Medium ~1.5GB) | ✅ | ✅ |
| DetectGPT (GPT-2 + T5-Large ~4GB) | ⚠️ Slow | ✅ |
| Binoculars Lightweight (GPT-2 ~2GB) | ✅ | ✅ |
| Binoculars Full (Falcon-7B ~14GB) | ❌ | ✅ |
| Ghostbuster Proxy (~1.4GB) | ✅ | ✅ |
| Methods M0/M1/M5 (no model) | ✅ | ✅ |
| StealthRL/Tinker (Qwen 3B + LoRA) | ⚠️ ~8GB RAM | ✅ |

---

## 1. Quick Smoke Test (Mac M4 Pro)

Test the pipeline end-to-end with lightweight components:

```bash
cd /Users/suraj/Desktop/StealthRL

# Test basic imports
python -c "from eval import *; print('✓ All imports work')"

# Run unit tests
python -m eval.test_eval_module

# Minimal smoke test: data + RoBERTa + simple methods
python -m eval.run \
    --datasets mage \
    --methods m0 m1 \
    --detectors roberta \
    --n-human 50 --n-ai 50 \
    --n-candidates 1 \
    --out-dir artifacts/smoke_mac \
    --run-name smoke_test
```

**Expected time:** ~2-3 minutes  
**Expected outputs:**
- `artifacts/smoke_mac/smoke_test_*/scores.parquet`
- `artifacts/smoke_mac/smoke_test_*/quality.parquet`
- `artifacts/smoke_mac/smoke_test_*/figures/`
- `artifacts/smoke_mac/smoke_test_*/tables/`

---

## 2. Extended Mac Test (More Detectors)

```bash
# Add Fast-DetectGPT (still lightweight)
python -m eval.run \
    --datasets mage \
    --methods m0 m1 m5 \
    --detectors roberta fast_detectgpt \
    --n-human 100 --n-ai 100 \
    --n-candidates 1 2 \
    --out-dir artifacts/smoke_extended \
    --run-name extended
```

**Expected time:** ~10-15 minutes

---

## 3. Full GPU Test (A5000 on DSMLP)

### Step 1: SSH to DSMLP and get GPU node

```bash
# Request A5000 node (adjust as per your DSMLP setup)
# Example: srun --partition=gpu --gres=gpu:1 --time=2:00:00 --pty bash
```

### Step 2: Setup environment

```bash
cd /path/to/StealthRL
conda activate stealthrl  # or your env name

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Step 3: Run with your Tinker checkpoint

```bash
# Full evaluation with StealthRL method
python -m eval.run \
    --datasets mage \
    --methods m0 m1 m2 \
    --detectors roberta fast_detectgpt binoculars \
    --stealthrl-checkpoint /path/to/your/tinker/checkpoint \
    --n-human 200 --n-ai 200 \
    --budget-sweep \
    --out-dir artifacts/gpu_full \
    --run-name tinker_eval
```

**Note:** Replace `/path/to/your/tinker/checkpoint` with your actual Tinker checkpoint path, e.g.:
- `outputs/tinker_full_ensemble_unoptimised/checkpoint-XXX`
- `outputs/tinker_ultrafast/checkpoint-XXX`

---

## 4. Setup Requirements You Need to Complete

### 4.1 Ghostbuster (Optional)

**Current state:** Uses "proxy mode" (a RoBERTa model) - no setup needed.

**For full Ghostbuster (requires OpenAI API):**
```bash
export OPENAI_API_KEY="sk-your-key-here"

# Then use api mode (not recommended - expensive)
# This requires modifying the detector initialization
```

**Recommendation:** Stick with proxy mode for the paper. It's a reasonable approximation.

### 4.2 Binoculars Full Model (Optional)

**Current state:** Uses lightweight GPT-2 pair by default.

**For full Falcon-7B pair (better accuracy, needs ~14GB VRAM):**
```python
# In your evaluation script, initialize with:
from eval.detectors import BinocularsDetector

detector = BinocularsDetector(
    use_lightweight=False,  # This enables Falcon-7B
    device="cuda"
)
```

Or modify the runner to accept a `--binoculars-full` flag.

### 4.3 Watermark Detector (Optional)

**Current state:** "Simulate" mode for pipeline testing only.

**For real watermark detection:** You need:
1. Access to the watermark key/seed used during generation
2. Knowledge of gamma/delta parameters from training

This is only useful if your Tinker model was trained with watermarking.

### 4.4 HuggingFace Token (If using gated models)

```bash
# Login to HuggingFace (one-time)
huggingface-cli login

# Or set token
export HF_TOKEN="hf_your_token_here"
```

**When needed:** Only for gated models like Llama, Mistral, etc. The default models (Qwen, GPT-2, RoBERTa) are public.

---

## 5. Finding Your Tinker Checkpoint

```bash
# List available checkpoints
ls -la outputs/*/checkpoint-*

# Example paths:
# outputs/tinker_full_ensemble_unoptimised/checkpoint-500
# outputs/tinker_ultrafast/checkpoint-100
```

The checkpoint should contain:
- `adapter_config.json`
- `adapter_model.safetensors` (or `.bin`)

---

## 6. Verification Checklist

After smoke test, verify these files exist:

```bash
# Check output structure
ls -la artifacts/smoke_mac/smoke_test_*/

# Should see:
# ├── scores.parquet      # Detection scores
# ├── quality.parquet     # Quality metrics
# ├── metrics.json        # Summary metrics
# ├── figures/
# │   ├── heatmap.png
# │   └── budget_sweep.png (if multiple N values)
# ├── tables/
# │   ├── main_results.md
# │   └── quality.md
# └── examples/
#     └── qualitative_examples.md
```

---

## 7. Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size or sample count
python -m eval.run ... --n-human 50 --n-ai 50
```

### "No module named 'eval'"
```bash
# Run from StealthRL root directory
cd /Users/suraj/Desktop/StealthRL
python -m eval.run ...
```

### "LoRA checkpoint not found"
```bash
# Verify checkpoint exists
ls -la /path/to/checkpoint/
# Should have adapter_config.json and adapter_model.*
```

### Detector loads slowly
First run downloads models (~2-5GB). Subsequent runs use cache.

---

## 8. Quick Reference: Method Names

| Method | Name | Description | Needs GPU |
|--------|------|-------------|-----------|
| M0 | `m0` / `no_attack` | Baseline (no modification) | ❌ |
| M1 | `m1` / `simple_paraphrase` | Simple paraphrase | ❌ |
| M2 | `m2` / `stealthrl` | Your Tinker/GRPO model | ✅ |
| M3 | `m3_roberta` | Guidance w/ RoBERTa | ✅ |
| M4 | `m4_authormist` | AuthorMist style | ✅ |
| M5 | `m5` / `homoglyph` | Unicode substitution | ❌ |

---

## 9. Recommended Test Sequence

1. **Mac (now):** Quick smoke test with M0/M1 + RoBERTa
2. **Mac (now):** Extended test with Fast-DetectGPT  
3. **DSMLP (later):** Full eval with M2 (Tinker) + all detectors
4. **DSMLP (final):** Budget sweep for paper figures

This ensures the pipeline works before committing GPU time.
