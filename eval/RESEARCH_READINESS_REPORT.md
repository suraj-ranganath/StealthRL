# Research Readiness Report for ICLR Paper

**Date:** January 30, 2026  
**Purpose:** Complete audit of eval pipeline for publication-ready results

---

## Executive Summary

Your eval pipeline has **4 critical issues** that need resolution before ICLR submission:

| Priority | Issue | Status | Action Required |
|----------|-------|--------|-----------------|
| 🔴 HIGH | Ghostbuster API mode not implemented | Proxy workaround | **Decision needed** |
| 🔴 HIGH | Binoculars uses GPT-2 not Falcon-7B | Lightweight mode | **Setup on GPU** |
| 🟡 MED | Watermark detector in simulate mode | Pipeline test only | **Remove from paper** |
| 🟢 LOW | Budget sweep needs testing | Implemented | **Test run** |

---

## 1. GHOSTBUSTER: What is Proxy Mode?

### The Problem
**Original Ghostbuster** (NAACL 2024) works by:
1. Passing text through OpenAI's `davinci-002` and `babbage-002` models
2. Extracting token-wise log probabilities
3. Computing 100+ features from these probabilities
4. Training a classifier on the features

**This requires OpenAI API calls for every single text sample.**

### Cost Estimate (OpenAI API)
- Models: `davinci-002` and `babbage-002` (legacy completion models)
- Per-sample: ~500 tokens × 2 models = ~1000 tokens
- Cost: ~$0.002-0.006 per 1K tokens for davinci-002
- **For 2000 samples: ~$20-60 in API costs + rate limiting delays**

### What "Proxy Mode" Does
Instead of calling OpenAI, it uses `openai-community/roberta-large-openai-detector` - a RoBERTa model trained for AI detection. This is **NOT the same** as Ghostbuster but serves as a reasonable classifier baseline.

### My Recommendation for Your Paper

**Option A (Recommended): Drop Ghostbuster entirely**
- Your detector panel already has 4 families: RoBERTa (classifier), Fast-DetectGPT (curvature), DetectGPT (curvature), Binoculars (zero-shot)
- Ghostbuster adds limited value since you already have a classifier (RoBERTa)
- SPEC.md marks it as "Optional (strong)"

**Option B: Use proxy mode but rename it**
- Call it "RoBERTa-Large-OpenAI" in your paper
- Cite it correctly as the OpenAI detector, not Ghostbuster
- Be honest about what it is

**Option C: Implement full Ghostbuster (not recommended)**
- Requires OpenAI API key with billing
- ~$20-60 for 2000 samples
- Significant implementation work
- Rate limiting will slow evaluation

---

## 2. BINOCULARS: Is Falcon-7B the Best?

### Yes, Falcon-7B is the Paper's Default

From the [official Binoculars paper](https://arxiv.org/abs/2401.12070) (ICML 2024):

> "This threshold is selected using **Falcon-7B and Falcon-7B-Instruct** models for scoring."

The paper achieves **>90% detection of ChatGPT at 0.01% FPR** using Falcon-7B pair.

### What Your Pipeline Currently Uses

```python
# Current (lightweight mode - DEFAULT)
PERFORMER_LIGHT = "gpt2-medium"   # 355M params
OBSERVER_LIGHT = "gpt2-large"     # 774M params

# Paper's models (full mode)
PERFORMER = "tiiuae/falcon-7b-instruct"  # 7B params
OBSERVER = "tiiuae/falcon-7b"            # 7B params
```

### Performance Difference
- **GPT-2 pair**: Lower accuracy, but same algorithmic approach
- **Falcon-7B pair**: Paper-reported accuracy (state-of-the-art)

### Setup for Falcon-7B on A5000

**Memory requirement:** ~14GB VRAM (fits on A5000's 24GB)

**How to enable:**
```bash
# In your eval command, the runner needs to be updated to pass this flag
# For now, you can modify detectors.py directly or add CLI flag
```

I need to add a CLI flag for this. See fixes below.

---

## 3. WATERMARK DETECTOR: What's the Point?

### Purpose
Watermarking is a **proactive defense** where the AI model embeds statistical patterns during generation. The detector then looks for these patterns.

### Why It's Currently Useless for You
The watermark detector only works if:
1. The text was generated with a **known watermarking algorithm**
2. You have the **exact watermark key/seed** used during generation
3. The watermark parameters (gamma, delta) match

**Your Tinker model was NOT trained with watermarking**, so:
- The watermark detector will always return ~0.5 (random)
- Including it in your paper would be meaningless

### My Recommendation

**Remove watermark from your evaluation.** It's marked as optional in SPEC.md:

> "If watermarking is too expensive, skip; do not block core eval."

The watermark family is only relevant if you're attacking watermarked text, which you're not.

---

## 4. BUDGET SWEEP: What Is It?

### Definition
Budget sweep evaluates **how many paraphrase candidates (N)** affect attack success.

```
N=1: Generate 1 candidate, use it
N=2: Generate 2 candidates, pick best (lowest detector score)
N=4: Generate 4 candidates, pick best
N=8: Generate 8 candidates, pick best
```

### Why It Matters
- Shows **compute-accuracy tradeoff**
- Papers like DIPPER and AuthorMist report N=1,2,4,8
- Important for practical deployment

### CLI Usage
```bash
# Single run
python -m eval.run --n-candidates 4

# Budget sweep (tests N=1,2,4,8)
python -m eval.run --budget-sweep

# Manual sweep
python -m eval.run --n-candidates 1 2 4 8
```

---

## 5. COMPLETE FIX LIST

### Fix 1: Add Binoculars Full Mode CLI Flag

I need to add `--binoculars-full` flag to runner.py.

### Fix 2: Remove Watermark from Default Detectors

Watermark should not be in your evaluation - it's not applicable.

### Fix 3: Rename Ghostbuster to What It Actually Is

Either drop it or rename to "RoBERTa-OpenAI-Detector".

### Fix 4: Document What Models Are Used

Your paper must accurately cite:
- RoBERTa: `openai-community/roberta-large-openai-detector`
- Fast-DetectGPT: `gpt2-medium` (scoring model)
- DetectGPT: `gpt2-medium` + `t5-large` (mask model)
- Binoculars: `tiiuae/falcon-7b` + `tiiuae/falcon-7b-instruct`

---

## 6. RECOMMENDED DETECTOR PANEL FOR PAPER

| Detector | Family | Model | VRAM | Run On |
|----------|--------|-------|------|--------|
| RoBERTa | Classifier | roberta-large-openai-detector | ~1.4GB | Mac or GPU |
| Fast-DetectGPT | Curvature | gpt2-medium | ~1.5GB | Mac or GPU |
| Binoculars | Zero-shot | **falcon-7b pair** | ~14GB | **GPU only** |

**Optional additions:**
- DetectGPT (curvature, classic) - slower but cited

**Remove:**
- Ghostbuster (redundant with RoBERTa)
- Watermark (not applicable)

---

## 7. YOUR ACTION ITEMS

### Immediate (Before GPU Run)

1. **Confirm detector panel:**
   - [ ] RoBERTa ✓
   - [ ] Fast-DetectGPT ✓
   - [ ] Binoculars (Falcon-7B) - **need to enable full mode**
   - [ ] DetectGPT (optional)
   - [ ] Ghostbuster - **decide: keep as RoBERTa-variant or drop**
   - [ ] Watermark - **remove**

2. **Find your Tinker checkpoint path:**
   ```bash
   ls -la outputs/*/checkpoint-*/adapter_config.json
   ```

3. **Test on Mac first:**
   ```bash
   python -m eval.run \
       --datasets mage \
       --methods m0 m1 \
       --detectors roberta fast_detectgpt \
       --n-human 50 --n-ai 50 \
       --out-dir artifacts/test
   ```

### On DSMLP (A5000)

4. **Full GPU run with Falcon-7B Binoculars:**
   ```bash
   python -m eval.run \
       --datasets mage \
       --methods m0 m1 m2 \
       --detectors roberta fast_detectgpt binoculars \
       --binoculars-full \
       --stealthrl-checkpoint /path/to/checkpoint \
       --n-human 1000 --n-ai 1000 \
       --budget-sweep \
       --out-dir artifacts/iclr_final \
       --run-name final
   ```

---

## 8. WHAT I NEED FROM YOU

1. **Decision on Ghostbuster:** Keep (as RoBERTa variant) or drop?

2. **Confirm checkpoint path:** Where is your best Tinker checkpoint?

3. **Sample sizes:** 
   - SPEC.md suggests 1000 human + 1000 AI
   - Is this acceptable for ICLR?

4. **Methods to include:**
   - M0 (no attack) ✓
   - M1 (simple paraphrase) ✓
   - M2 (your Tinker/StealthRL) ✓
   - M4 (AuthorMist) - do you want this comparison?
   - M5 (homoglyph) - do you want this?

---

## 9. APPLYING FIXES NOW

Let me apply the necessary code changes to make your pipeline research-ready.
