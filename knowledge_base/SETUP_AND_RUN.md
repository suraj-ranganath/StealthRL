# StealthRL Setup & Run Guide

**Quick reference for setting up your Tinker API key and running your research experiments.**

---

## 🔑 Step 1: Set Up Tinker API Key

### Get Your API Key

1. Go to [Tinker Platform](https://tinker.thinkingmachines.ai/)
2. Sign in with your account
3. Navigate to **Settings** or **API Keys**
4. Copy your API key

### Add to .env File

Your `.env` file has been created at: `/Users/suraj/Desktop/StealthRL/.env`

**Edit the file and add your API key:**

```bash
# Open in your preferred editor
nano .env
# OR
code .env
# OR
open -e .env
```

**Replace this line:**
```bash
TINKER_API_KEY=your_tinker_api_key_here
```

**With your actual key:**
```bash
TINKER_API_KEY=tk-abc123xyz789...
```

Save and close the file.

### Verify Setup

```bash
# Check that your key is set
grep TINKER_API_KEY .env
```

You should see: `TINKER_API_KEY=tk-...` (with your actual key)

---

## 📦 Step 2: Install Dependencies

```bash
cd /Users/suraj/Desktop/StealthRL

# Install all Tinker dependencies
pip install -r requirements.txt
```

**Expected time**: 2-5 minutes

**If you encounter errors**, try:
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

---

## 📊 Step 3: Prepare Data

### Option A: Quick Test with Synthetic Data (Recommended First)

```bash
python scripts/prepare_tinker_data.py \
    --synthetic \
    --num-train 100 \
    --num-test 20 \
    --output-dir data/tinker
```

This creates:
- `data/tinker/train.jsonl` (100 samples)
- `data/tinker/test.jsonl` (20 samples)

**Expected time**: 10-30 seconds

### Option B: Full Dataset for Real Experiments

```bash
python scripts/prepare_tinker_data.py \
    --synthetic \
    --num-train 1000 \
    --num-test 200 \
    --output-dir data/tinker
```

**Expected time**: 1-2 minutes

### Option C: Use Your Own Data

If you have ESL/native academic writing data:

```bash
python scripts/prepare_tinker_data.py \
    --input-paths data/raw/your_data.jsonl \
    --output-dir data/tinker \
    --train-split 0.8
```

**Required JSONL format:**
```json
{
  "ai_text": "Your AI-generated text...",
  "human_reference": "Original human text...",
  "domain": "academic",
  "is_esl": false
}
```

---

## 🚀 Step 4: Run Your Research

### 🎯 FASTEST: Automated Pipeline

Run everything with one command:

```bash
python scripts/run_research_pipeline.py --stage all
```

**What it does:**
1. ✅ Validates data
2. ✅ Trains full ensemble model (all 3 detectors)
3. ✅ Trains transfer model (2 detectors only)
4. ✅ Runs 5 ablation experiments
5. ✅ Evaluates all models
6. ✅ Generates visualization plots

**Time estimates:**
- With small dataset (100 train): 2-4 hours
- With full dataset (1000 train): 6-8 hours
- Parallel training (if available): 3-5 hours

**Monitor progress:**
```bash
# The script prints progress updates
# Check outputs in real-time:
ls -lh outputs/
```

---

### 🔬 DETAILED: Step-by-Step Experiments

If you want more control, run experiments individually:

#### Experiment 1: Full Ensemble Training

Train with all 3 detectors (Fast-DetectGPT, Ghostbuster, Binoculars):

```bash
python -m stealthrl.tinker.train \
    --config configs/tinker_stealthrl.yaml \
    --data-path data/tinker \
    --output-dir outputs/tinker_full_ensemble
```

**Expected output:**
- `outputs/tinker_full_ensemble/checkpoints/` - Model checkpoints
- `outputs/tinker_full_ensemble/logs/` - Training logs
- `outputs/tinker_full_ensemble/metrics.json` - Final metrics

**Time**: 1.5-2 hours (1000 samples)

**Monitor training:**
```bash
# Check latest logs
tail -f outputs/tinker_full_ensemble/logs/training.log

# OR use TensorBoard
tensorboard --logdir outputs/tinker_full_ensemble/logs
```

#### Experiment 2: Transfer Learning (Core Research Question)

Train with only 2 detectors, evaluate on held-out 3rd:

```bash
python -m stealthrl.tinker.train \
    --config configs/tinker_transfer_in_ensemble.yaml \
    --data-path data/tinker \
    --output-dir outputs/tinker_transfer_in_ensemble
```

**This tests:** Does training on Fast-DetectGPT + Ghostbuster transfer to Binoculars?

**Expected ASR transfer ratio**: 0.70-0.85 (good transfer)

**Time**: 1.5-2 hours

#### Experiment 3: Evaluate Transfer

```bash
python scripts/evaluate_transfer.py \
    --test-data data/tinker/test.jsonl \
    --full-model outputs/tinker_full_ensemble \
    --transfer-model outputs/tinker_transfer_in_ensemble \
    --output-dir outputs/transfer_eval
```

**Outputs:**
- `outputs/transfer_eval/transfer_metrics.json`
  - ASR in-ensemble (should be high, ~60-70%)
  - ASR held-out (should be decent, ~45-55%)
  - Transfer ratio (target: >0.7)

**Time**: 5-10 minutes

#### Experiment 4: Ablation Studies

Test which reward components matter most:

```bash
# 1. Detector-only (no semantic/quality constraints)
python -m stealthrl.tinker.train \
    --config configs/ablations/detector_only.yaml \
    --data-path data/tinker \
    --output-dir outputs/ablations/detector_only

# 2. No fairness penalty
python -m stealthrl.tinker.train \
    --config configs/ablations/no_fairness.yaml \
    --data-path data/tinker \
    --output-dir outputs/ablations/no_fairness

# 3. No quality constraint
python -m stealthrl.tinker.train \
    --config configs/ablations/no_quality.yaml \
    --data-path data/tinker \
    --output-dir outputs/ablations/no_quality

# 4. No semantic constraint
python -m stealthrl.tinker.train \
    --config configs/ablations/no_semantic.yaml \
    --data-path data/tinker \
    --output-dir outputs/ablations/no_semantic

# 5. Single detector (Fast-DetectGPT only)
python -m stealthrl.tinker.train \
    --config configs/ablations/single_detector_fast_detectgpt.yaml \
    --data-path data/tinker \
    --output-dir outputs/ablations/single_detector
```

**Time per ablation**: 1-1.5 hours (total: 5-7.5 hours)

**Run in parallel** (if you have multiple Tinker compute slots):
```bash
# Launch all at once in background
for config in configs/ablations/*.yaml; do
    name=$(basename $config .yaml)
    python -m stealthrl.tinker.train \
        --config $config \
        --data-path data/tinker \
        --output-dir outputs/ablations/$name &
done
```

#### Experiment 5: Comprehensive Evaluation

```bash
python scripts/evaluate_ablations.py \
    --checkpoints outputs/ablations/* \
    --output-dir outputs/ablation_analysis
```

**Outputs:**
- `outputs/ablation_analysis/ablation_comparison.json`
- `outputs/ablation_analysis/pareto_frontier.png`

**Time**: 10-15 minutes

#### Experiment 6: Visualizations

Generate publication-quality plots:

```bash
python scripts/visualize_stealthbench.py \
    --results outputs/transfer_eval outputs/ablation_analysis \
    --output-dir outputs/figures
```

**Outputs:**
- `outputs/figures/roc_curves.png` - Multi-detector ROC
- `outputs/figures/fpr_comparison.png` - FPR@TPR95 bars
- `outputs/figures/low_fpr_comparison.png` - Academic integrity thresholds
- `outputs/figures/esl_fairness_heatmap.png` - ESL bias analysis
- `outputs/figures/pareto_frontier.png` - 3D trade-off visualization

**Time**: 2-5 minutes

---

## 🧪 ESL Fairness Evaluation (Optional)

If you have ESL/native labeled data:

### Prepare ESL/Native Splits

```bash
# This requires preprocessed ESL corpora (TOEFL11, ICNALE, etc.)
python -m stealthrl.data.esl_native_corpus
```

See `ESL_FAIRNESS_GUIDE.md` for details on data preparation.

### Run ESL Evaluation

```bash
python scripts/run_esl_eval.py \
    --eval_data data/processed/esl_native_test.jsonl \
    --stealthrl_model outputs/tinker_full_ensemble \
    --enable_bertscore \
    --bertscore_model roberta-large \
    --output_dir results/esl_native_eval
```

**Outputs:**
- `results/esl_native_eval/comparison_report.json` - Overall metrics
- `results/esl_native_eval/esl_native_grouped_metrics.json` - By ESL status
- `results/esl_native_eval/bertscore_results.json` - Semantic similarity

**Time**: 20-30 minutes (with BERTScore)

---

## 📈 Expected Results

### Core Metrics (from DSC 291 Proposal)

| Metric | Baseline (Base AI) | SFT | StealthRL (Target) |
|--------|-------------------|-----|-------------------|
| **ASR (All Detectors)** | 10-15% | 30-40% | **60-70%** |
| **Semantic Similarity** | 1.0 | 0.85-0.90 | **0.88-0.92** |
| **ESL FPR Gap** | 0.12-0.15 | 0.08-0.10 | **0.03-0.07** |
| **TPR @ FPR=0.5%** | 85-90% | 55-65% | **40-55%** |
| **Transfer Ratio** | N/A | N/A | **0.70-0.85** |

### Transfer Learning (Research Question 1)

**In-ensemble training** (Fast-DetectGPT + Ghostbuster):
- ASR in-ensemble: 60-70%
- ASR held-out (Binoculars): 45-55%
- **Transfer ratio**: 0.70-0.85 ✅ (proves cross-family generalization)

### Ablation Studies (Research Question 2)

**Expected Pareto frontier ranking:**
1. **Full model** (all rewards) - Best balance
2. **No fairness** - Slight ASR gain, worse ESL gap
3. **No quality** - Higher ASR, lower semantic similarity
4. **Detector-only** - Highest ASR, poor quality
5. **Single detector** - Overfits, poor transfer

### Fairness (Research Question 3)

**ESL FPR gap reduction:**
- Base AI: 0.12-0.15 (high bias)
- StealthRL: 0.03-0.07 (60-80% reduction) ✅

---

## 🔍 Monitoring & Debugging

### Check Training Progress

```bash
# View live logs
tail -f outputs/tinker_full_ensemble/logs/training.log

# Check metrics
cat outputs/tinker_full_ensemble/metrics.json | jq .

# List all checkpoints
ls -lh outputs/tinker_full_ensemble/checkpoints/
```

### TensorBoard Visualization

```bash
tensorboard --logdir outputs/tinker_full_ensemble/logs
# Open browser: http://localhost:6006
```

### Debug Mode

Enable verbose logging:

```bash
python -m stealthrl.tinker.train \
    --config configs/tinker_stealthrl.yaml \
    --data-path data/tinker \
    --output-dir outputs/debug \
    --debug
```

### Common Issues

#### Issue: "Tinker API key not found"

**Solution:**
```bash
# Check .env file
cat .env | grep TINKER_API_KEY

# Should show: TINKER_API_KEY=tk-...
# If blank, edit .env and add your key
```

#### Issue: "CUDA out of memory"

**Solution:** Reduce batch size in config:
```yaml
training:
  batch_size: 4  # Reduce from 8
  group_size: 4  # Reduce from 4
```

#### Issue: "Detector API timeout"

**Solution:** Detectors use caching. First run is slow, subsequent runs are fast.
```bash
# Check cache
ls -lh cache/detectors.db
```

#### Issue: "Low ASR (<30%)"

**Possible causes:**
1. Not enough training steps (increase `num_epochs`)
2. Learning rate too low (try 5e-5)
3. Reward weights imbalanced (increase `detector_weight`)
4. Bad data quality

**Debug:**
```bash
# Check reward distributions
python -m stealthrl.tinker.train \
    --config configs/tinker_stealthrl.yaml \
    --data-path data/tinker \
    --output-dir outputs/debug \
    --debug \
    --num-epochs 1  # Quick test
```

---

## 📋 Checklist for Research Completion

### Setup
- [ ] Tinker API key added to `.env`
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Data prepared (`data/tinker/train.jsonl` and `test.jsonl` exist)

### Core Experiments
- [ ] Full ensemble trained (all 3 detectors)
- [ ] Transfer model trained (2 detectors only)
- [ ] Transfer evaluation completed
- [ ] 5 ablation models trained
- [ ] Ablation analysis completed

### Evaluation & Analysis
- [ ] Visualizations generated (ROC, FPR, Pareto, etc.)
- [ ] ESL fairness evaluation (if applicable)
- [ ] BERTScore computed (if applicable)
- [ ] Results documented

### Deliverables
- [ ] Training logs saved
- [ ] Model checkpoints saved
- [ ] Metrics JSON files exported
- [ ] Publication-quality figures generated
- [ ] Final report written

---

## 🎓 Next Steps After Training

### 1. Analyze Results

```bash
# Compare all models
python scripts/compare_baselines.py \
    --models outputs/tinker_full_ensemble outputs/tinker_transfer_in_ensemble \
    --baseline-sico \
    --output-dir outputs/comparison
```

### 2. Generate Report

```bash
# Auto-generate summary report
python scripts/generate_report.py \
    --results outputs/ \
    --output report.md
```

### 3. Export for Paper

```bash
# Copy figures to paper directory
cp outputs/figures/*.png ~/my_paper/figures/

# Export tables
python scripts/export_tables.py \
    --results outputs/ \
    --format latex \
    --output ~/my_paper/tables/
```

---

## 📞 Support

### Documentation
- **Tinker Integration**: `TINKER_README.md`
- **ESL Fairness**: `ESL_FAIRNESS_GUIDE.md`
- **Research Pipeline**: `QUICKSTART.md`
- **Implementation Details**: `interaction_records.md`

### Troubleshooting
- Check logs: `outputs/*/logs/training.log`
- Enable debug mode: `--debug` flag
- Review interaction records for design decisions

### Questions?
- Review `RESEARCH_ROADMAP.md` for research plan
- Check `IMPLEMENTATION_VERIFICATION.md` for feature status
- See `NEXT_STEPS.md` for optional enhancements

---

## 🎉 Quick Success Check

After setup, run this quick test (5 minutes):

```bash
# 1. Create tiny dataset
python scripts/prepare_tinker_data.py \
    --synthetic \
    --num-train 10 \
    --num-test 5 \
    --output-dir data/tinker_test

# 2. Quick training test (1 epoch)
python -m stealthrl.tinker.train \
    --config configs/tinker_stealthrl.yaml \
    --data-path data/tinker_test \
    --output-dir outputs/test_run \
    --num-epochs 1

# 3. Check outputs
ls outputs/test_run/checkpoints/

# If you see checkpoint files, you're ready to go! 🚀
```

---

**You're all set! Start with the automated pipeline or run experiments step-by-step.**

**Good luck with your research!** 🔬
