# StealthRL Research Quickstart - DSC 291

**Fast-track guide to running the complete research pipeline for the DSC 291 project proposal.**

## 🚀 Quick Setup (5 minutes)

### 1. Install Dependencies

```bash
# Clone repository
cd /Users/suraj/Desktop/StealthRL

# Install Tinker dependencies
pip install -r requirements.txt

# Set up API key
cp .env.example .env
# Edit .env and add your TINKER_API_KEY
```

### 2. Prepare Data

```bash
# Generate synthetic dataset for testing
python scripts/prepare_tinker_data.py \
    --synthetic \
    --num-train 1000 \
    --num-test 200 \
    --output-dir data/tinker

# OR use real datasets (if available)
python scripts/prepare_tinker_data.py \
    --input-paths data/raw/*.jsonl \
    --output-dir data/tinker \
    --train-split 0.8
```

## 🎯 Running Core Research Experiments

### Option A: Automated Pipeline (Recommended)

Run the complete research pipeline automatically:

```bash
python scripts/run_research_pipeline.py --stage all
```

This will:
1. ✅ Prepare data
2. ✅ Train full ensemble model (Fast-DetectGPT + Ghostbuster + Binoculars)
3. ✅ Train transfer model (in-ensemble only)
4. ✅ Run ablation experiments
5. ✅ Comprehensive evaluation
6. ✅ Generate visualizations

**Estimated time**: 3-6 hours (depending on Tinker compute)

### Option B: Step-by-Step (For Control)

#### Step 1: Train Full Ensemble

```bash
python -m stealthrl.tinker.train \
    --config configs/tinker_stealthrl.yaml \
    --data-path data/tinker \
    --output-dir outputs/tinker_full_ensemble
```

#### Step 2: Train Transfer Model (In-Ensemble Only)

```bash
python -m stealthrl.tinker.train \
    --config configs/tinker_transfer_in_ensemble.yaml \
    --data-path data/tinker \
    --output-dir outputs/tinker_transfer_in_ensemble
```

#### Step 3: Evaluate Transfer

```bash
python scripts/evaluate_transfer.py \
    --checkpoints outputs/tinker_full_ensemble outputs/tinker_transfer_in_ensemble \
    --output-dir outputs/evaluation
```

#### Step 4: Run Ablations

```bash
# Detector-only ablation
python -m stealthrl.tinker.train \
    --config configs/ablations/detector_only.yaml \
    --data-path data/tinker \
    --output-dir outputs/ablations/detector_only

# No fairness ablation
python -m stealthrl.tinker.train \
    --config configs/ablations/no_fairness.yaml \
    --data-path data/tinker \
    --output-dir outputs/ablations/no_fairness

# Continue with other ablations...
```

#### Step 5: Generate Visualizations

```bash
python scripts/visualize_stealthbench.py \
    --results-dir outputs/evaluation \
    --ablation-dir outputs/ablation_analysis \
    --output-dir outputs/visualizations
```

## 📊 Key Research Questions & Experiments

### Question 1: Transfer to Held-Out Detectors

**Goal**: Does ensemble training generalize to unseen detector families?

**Experiment**:
- Train on Fast-DetectGPT + Ghostbuster (in-ensemble)
- Evaluate on Binoculars (held-out)
- Measure ASR transfer ratio

**Config**: `configs/tinker_transfer_in_ensemble.yaml`

**Expected Output**:
- Transfer ratio > 0.7 = Good generalization
- ASR drop < 20% = Strong cross-family transfer

### Question 2: ESL Fairness

**Goal**: Does fairness penalty reduce ESL false-positive bias?

**Experiment**:
- Compare full model vs no-fairness ablation
- Measure ESL FPR gap: FPR(ESL) - FPR(native)
- Evaluate at low-FPR thresholds (0.5%, 1.0%)

**Config**: `configs/ablations/no_fairness.yaml`

**Expected Output**:
- Fairness-aware model: ESL gap < 0.05
- No-fairness ablation: ESL gap > 0.12

### Question 3: Pareto Frontier

**Goal**: Map trade-offs between detectability, semantic fidelity, and fairness

**Experiment**:
- Train all ablations (detector-only, no-fairness, no-semantic, no-quality)
- Plot 3D Pareto frontier
- Identify optimal reward weighting

**Expected Output**:
- Full model near Pareto optimal
- Ablations show clear trade-offs

### Question 4: SICO Baseline Comparison

**Goal**: How does StealthRL compare to prompt-based paraphrasing?

**Experiment**:
```python
from stealthrl.baselines import SICOParaphraser

# Initialize SICO with GPT-3.5/GPT-4
sico = SICOParaphraser(model, config=SICOConfig(
    prompt_template="paraphrase_preserve_meaning",
    num_candidates=4
))

# Compare ASR, semantic similarity, ESL fairness
```

**Expected Output**:
- StealthRL ASR > SICO ASR (better evasion)
- StealthRL semantic sim ≈ SICO (maintain quality)
- StealthRL ESL gap < SICO (better fairness)

## 📈 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir outputs/
```

**Key metrics to monitor**:
- `reward/total`: Should increase over training
- `reward/detector`: Target > 0.5 (high evasion)
- `reward/semantic`: Target > 0.8 (preserve meaning)
- `reward/fairness`: Target < -0.1 (reduce ESL bias)
- `kl`: Keep < 0.01 (don't drift too far from reference)
- `all_negative_frac_total`: Should decrease (< 0.3)

### Debug Samples

Check `outputs/*/debug_samples.jsonl` for example paraphrases:

```bash
# View sample paraphrases
tail -n 5 outputs/tinker_full_ensemble/debug_samples.jsonl | jq
```

## 🎓 Expected Results Summary

| Metric | Base AI Text | SICO Baseline | StealthRL | Target |
|--------|--------------|---------------|-----------|--------|
| **ASR (all detectors)** | 5-10% | 30-40% | **60-70%** | >60% |
| **Semantic Similarity** | N/A | 0.87-0.90 | **0.88-0.92** | >0.85 |
| **ESL FPR Gap** | 0.15-0.20 | 0.10-0.15 | **0.03-0.07** | <0.10 |
| **TPR @ FPR=0.5%** | 85-90% | 60-70% | **40-55%** | <60% |
| **Transfer Ratio** | N/A | N/A | **0.70-0.85** | >0.70 |

## 🔬 Advanced Experiments (Stretch Goals)

### Selective Fine-Tuning

Test if unfreezing top layers improves transfer:

```bash
python -m stealthrl.tinker.train \
    --config configs/tinker_selective_finetune.yaml \
    --data-path data/tinker \
    --output-dir outputs/selective_finetune
```

### Defender Analysis

Find optimal detector ensemble for defense:

```python
from scripts.defender_analysis import test_detector_mixtures

best_mixture = test_detector_mixtures(
    stealthrl_policy=model,
    detector_pool=["fast_detectgpt", "ghostbuster", "binoculars", "gptze ro"],
    mixture_sizes=[2, 3]
)
```

## 📝 Generating Final Report

### 1. Collect Results

```bash
# Evaluation metrics
cat outputs/evaluation/transfer_evaluation.json

# Ablation analysis
cat outputs/ablation_analysis/pareto_frontier.json

# Visualizations
ls outputs/visualizations/
```

### 2. Key Deliverables

- ✅ **Transfer evaluation**: `outputs/evaluation/transfer_evaluation.json`
- ✅ **ROC curves**: `outputs/visualizations/roc_curves.pdf`
- ✅ **Pareto frontier**: `outputs/visualizations/pareto_frontier.pdf`
- ✅ **ESL fairness**: `outputs/visualizations/esl_fairness_heatmap.pdf`
- ✅ **Low-FPR metrics**: `outputs/visualizations/low_fpr_comparison.pdf`

### 3. Reporting Structure

```markdown
## Results

### 1. Multi-Detector Transfer
- In-ensemble ASR: 68%
- Held-out ASR: 51%
- Transfer ratio: 0.75 ✓ (good generalization)

### 2. ESL Fairness
- Full model ESL gap: 0.05
- No-fairness ablation gap: 0.14
- Reduction: 64% ✓

### 3. Comparison with SICO
- StealthRL ASR: 65% vs SICO: 38% (+27pp)
- Semantic similarity: 0.90 vs 0.88 (similar quality)
- ESL fairness: 0.05 vs 0.11 (StealthRL more fair)

### 4. Pareto Analysis
- Full model achieves 92% Pareto efficiency
- Detector-only: High evasion but poor semantic (73%)
- No-semantic: High quality but low evasion (68%)
```

## 🚨 Troubleshooting

### Training Hangs at "All-Negative Groups"

**Symptom**: `all_negative_frac_total > 0.8`

**Fix**:
```yaml
# In config file
all_negative:
  min_reward: 0.05  # Increase shaped signal
  downweight: 0.3   # Reduce contribution more
```

### KL Divergence Too High

**Symptom**: `kl > 0.05`

**Fix**:
```yaml
kl:
  penalty_coef: 0.01  # Increase from 0.001
  adapt: true
```

### Low Semantic Similarity

**Symptom**: `semantic_sim_mean < 0.80`

**Fix**:
```yaml
reward:
  semantic_weight: 2.0  # Increase from 1.0
```

### Poor Transfer

**Symptom**: `transfer_ratio < 0.5`

**Possible causes**:
- Overfitting to in-ensemble detectors
- Need more diverse training data
- Try curriculum learning

## 📚 Additional Resources

- **Full documentation**: `TINKER_README.md`
- **Research roadmap**: `RESEARCH_ROADMAP.md`
- **Implementation log**: `interaction_records.md`
- **Tinker docs**: https://tinker-docs.thinkingmachines.ai/

## ⏱️ Timeline Estimate

| Phase | Time | Parallelizable |
|-------|------|----------------|
| Data prep | 30 min | No |
| Training (full) | 2-3 hours | No |
| Training (transfer) | 2-3 hours | Yes (parallel) |
| Ablations (5 models) | 10-15 hours | Yes (parallel) |
| Evaluation | 1-2 hours | No |
| Visualization | 15 min | No |
| **Total** | **16-24 hours** | **Can be 6-8 hours if parallel** |

---

**Questions?** Check `RESEARCH_ROADMAP.md` for detailed implementation specs.
