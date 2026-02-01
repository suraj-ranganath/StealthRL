# Dataset Size & API Cost Analysis - StealthRL

**Generated**: January 24, 2026  
**Purpose**: Estimate API costs for detector evaluation

---

## 📊 Dataset Overview

### Main Dataset: `tinker_full_esl40_nodup`

| Split | Samples | Total Words | Avg Words/Sample |
|-------|---------|-------------|------------------|
| Train | 8,185 | 4,813,626 | 588 |
| Test | 2,047 | 1,201,977 | 587 |
| **TOTAL** | **10,232** | **6,015,603** | **588** |

**Composition**:
- Human text: 3,077,593 words (51%)
- AI text: 2,938,010 words (49%)
- ESL representation: ~40%

---

## 💰 API Cost Analysis

### Per Single Experiment

**Experiment Scope**:
- Training: 8,185 samples × 16 GRPO rollouts = **130,960 detections**
- Evaluation: 2,047 samples × 2 (human + paraphrased) = **4,094 detections**
- **Total: 135,054 detections**
- **Total words: ~39.7 million words**

### Cost Breakdown by API

| API Provider | Pricing Model | Cost per Experiment |
|--------------|---------------|---------------------|
| **GPTZero** | $0.01 per 500 words | **$794** |
| **Originality.AI** | $0.01 per 100 words | **$3,970** ⚠️ |
| **Sapling.ai** | 100 free/day, then $25/month | $25 (takes 1,351 days!) |

**⚠️ IMPORTANT**: These costs are **PER EXPERIMENT**. For a full research project (5 experiments), multiply by 5!

---

## 📈 Full Research Project Costs

### Complete Experimental Suite (5 experiments)

1. Full ensemble training
2. Transfer evaluation  
3. Ablation: No fairness
4. Ablation: Detector-only
5. Ablation: Single detector

### Total Budget Requirements

| Scenario | Configuration | Total Cost |
|----------|---------------|------------|
| **Option A: All Open Source** ✅ | RoBERTa + Fast-DetectGPT + Binoculars | **$0** |
| **Option B: Mixed (Recommended)** | 1 API + 2 Local | **~$4,000** |
| **Option C: All API** ❌ | 2-3 APIs | **~$15,000-20,000** |

**Recommended Budget**: **$4,764** (includes 20% buffer for retries/errors)

---

## 🎯 Realistic Cost Estimates

### The REAL Numbers You Need to Know

**Problem**: The above assumes FULL dataset for EVERY experiment. In reality:

1. **Training doesn't run detectors on EVERY rollout** - you sample for reward computation
2. **Evaluation is cheaper** - you only test on small subsets (500-1000 samples)
3. **API caching** - same texts get detected multiple times

### Adjusted Estimates (Realistic)

#### Per Experiment (Optimized)

**Training Phase**:
- Sample 10% of rollouts for detector evaluation (not all 130,960)
- Use local detectors for most checks, API for final validation
- **Estimated detections: ~5,000-10,000**

**Evaluation Phase**:
- Test on 500-1000 samples (not all 2,047)
- **Estimated detections: ~1,000-2,000**

**Total per experiment: ~6,000-12,000 detections**

#### Revised API Costs

| Provider | Optimized Cost per Experiment | 5 Experiments |
|----------|-------------------------------|---------------|
| **GPTZero** | $35-70 | **$175-350** |
| **Originality.AI** | $180-350 | **$900-1,750** |

**Realistic Budget**: **$300-500** for full research project

---

## 💡 Cost-Saving Strategies

### Strategy 1: Detector Hierarchy ⭐ RECOMMENDED

```
1. Use local detectors (RoBERTa, Fast-DetectGPT) during training
2. Use API detectors (GPTZero) only for final evaluation
3. Cache all API results to avoid duplicate calls
```

**Savings**: 90-95% reduction in API costs

### Strategy 2: Sampling

```
1. Training: Compute rewards on 10% sample, not full batch
2. Evaluation: Test on 500-1000 samples, not full 2,047
3. Checkpoints: Only evaluate every 5-10 steps, not every step
```

**Savings**: 80-90% reduction in API calls

### Strategy 3: Free Tiers

```
1. Sapling.ai: 100 free requests/day
2. Use free tier for preliminary testing
3. Switch to paid only for final experiments
```

**Savings**: $0 for initial development

### Strategy 4: Mixed Approach ⭐ BEST VALUE

```
Training:
  - RoBERTa (local) - fast, cheap
  - Fast-DetectGPT (local) - theoretically grounded
  
Evaluation:
  - Above 2 (local)
  - GPTZero (API) - industry standard for comparison
```

**Cost**: ~$50-100 per experiment
**Total project**: ~$250-500

---

## 🚀 Recommended Implementation Plan

### For 1-Week Timeline (Budget: $0-100)

**Day 1**: Implement RoBERTa (local, free)
```bash
pip install transformers torch
# Use: Hello-SimpleAI/chatgpt-detector-roberta
```

**Day 2**: Add Sapling.ai (free tier)
```bash
# 100 free requests/day
# Use for evaluation only, not training
```

**Day 3-4**: Run 1-2 experiments with free detectors
- Training: RoBERTa only
- Evaluation: RoBERTa + Sapling (sampled)

**Day 5-7**: Write paper
- Acknowledge detector limitations
- Focus on framework contribution

**Total cost**: $0

---

### For Strong Publication (Budget: $300-500)

**Week 1**: Implement 3 detectors
1. RoBERTa (local)
2. Fast-DetectGPT or Binoculars (local)
3. GPTZero (API, paid)

**Week 2-3**: Run full experiments
- Training: 2 local detectors
- Evaluation: All 3 detectors on 500-1000 samples

**Week 4**: Analysis & writing

**Total cost**: $250-500

---

## 📋 Implementation Checklist

### Free Detectors ($0)
- [ ] RoBERTa-ChatGPT-Detector (HuggingFace)
- [ ] Fast-DetectGPT (GitHub)
- [ ] Binoculars (GitHub)
- [ ] Sapling.ai free tier (100/day)

### Paid Detectors (if budget allows)
- [ ] GPTZero ($10-25/month plan)
- [ ] Originality.AI (pay-as-you-go)

### Cost Optimization
- [ ] Implement detector caching (save results to avoid re-computation)
- [ ] Sample rollouts for reward (don't check every generation)
- [ ] Limit evaluation to 500-1000 samples
- [ ] Use hierarchical approach (cheap → expensive)

---

## 🎯 Bottom Line Recommendations

### If Budget = $0 (Current Situation)
**Use**: RoBERTa + Binoculars (both local, open source)
**Result**: Valid research, publishable
**Limitation**: No industry-standard detector (GPTZero)

### If Budget = $100-200
**Use**: RoBERTa + Fast-DetectGPT (local) + Sapling.ai (API, $25/month)
**Result**: Strong empirical validation
**Limitation**: Sapling less popular than GPTZero

### If Budget = $300-500 ⭐ RECOMMENDED
**Use**: RoBERTa + Fast-DetectGPT (local) + GPTZero (API)
**Result**: Best of both worlds - reproducible + industry standard
**Limitation**: None for workshop paper

### If Budget = $1,000+
**Use**: All 5 detectors (3 local + 2 API)
**Result**: Comprehensive evaluation, top-tier publication
**Limitation**: Overkill for workshop paper

---

## 🔑 Key Takeaways

1. **Your main dataset has 10,232 samples (~6M words total)**

2. **Naive API costs are VERY HIGH ($794-$3,970 per experiment)**

3. **BUT with smart sampling, costs drop to $50-100 per experiment**

4. **For 1-week timeline + $0 budget**: Use RoBERTa + Binoculars (both free)

5. **For strong publication + $300-500 budget**: Add GPTZero API for industry validation

6. **Full research project (5 experiments)**: Budget $250-500 with optimization

---

## 📞 Next Steps

1. **Immediate**: Implement RoBERTa (4-6 hours, $0)
2. **Optional**: Sign up for Sapling.ai free tier (5 min, $0)
3. **If budget allows**: Get GPTZero API key ($10/month plan)
4. **During training**: Use local detectors only
5. **During evaluation**: Sample 500-1000 texts, use all available detectors

**Don't let API costs scare you!** With smart sampling and caching, you can do this for under $100 total.
