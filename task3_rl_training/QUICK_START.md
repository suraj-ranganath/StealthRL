# TASK 3: RL Training - Quick Start Guide

**Complete execution guide for TASK 3 - Main RL Training on Tinker Platform**

---

## ✅ Progress Tracker

- [ ] Step 0: Prerequisites check (TASK 1 & TASK 2 completed)
- [ ] Step 1: Configure Tinker API key
- [ ] Step 2: Run pre-flight check
- [ ] Step 3: Test with synthetic data (optional but recommended)
- [ ] Step 4: Run full ensemble training
- [ ] Step 5: Monitor training progress
- [ ] Step 6: Verify training completion
- [ ] Step 7: Run transfer experiment (optional)
- [ ] Step 8: Run ablation studies (optional)

---

## 🎯 What is TASK 3?

**Objective**: Train StealthRL model using Reinforcement Learning on the Tinker platform to create a paraphraser that:
1. Evades multiple AI detectors (Fast-DetectGPT, Ghostbuster, Binoculars)
2. Preserves semantic meaning
3. Maintains text quality
4. Reduces ESL bias

**Training Platform**: [Tinker](https://tinker.thinkingmachines.ai/) - Remote RL training with Qwen3-4B + GRPO

**Estimated Time**:
- Setup: 15 minutes
- Training: 2-4 hours per experiment (runs on Tinker, you don't need to wait)

---

## 🚀 Step-by-Step Execution

### **Step 0: Prerequisites Check**

Make sure TASK 1 and TASK 2 are completed:

```bash
cd "/Users/nishchaymahor/Documents/Study/291 - Safety in Gen AI/StealthRL/StealthRL"

# Check TASK 1 (Detectors)
ls task1_detector_implementation/

# Check TASK 2 (Data)
ls data/tinker/
ls data/processed/
```

**Expected**:
- `task1_detector_implementation/` folder exists
- `data/tinker/train.jsonl` exists (~1200 samples)
- `data/processed/esl_native_test.jsonl` exists (~500 samples)

---

### **Step 1: Configure Tinker API Key**

#### Option A: Use Setup Script (Recommended)
```bash
bash task3_rl_training/scripts/setup_env.sh
```

Follow the prompts to enter your Tinker API key.

#### Option B: Manual Setup
```bash
# Copy example
cp .env.example .env

# Edit .env file
nano .env
# or
code .env

# Replace: TINKER_API_KEY=your_tinker_api_key_here
# With:    TINKER_API_KEY=tk-your-actual-key-here
```

**Get your API key**:
1. Go to: https://tinker.thinkingmachines.ai/
2. Sign in/register
3. Navigate to: Settings → API Keys
4. Copy your key (starts with `tk-`)

---

### **Step 2: Run Pre-Flight Check**

```bash
python task3_rl_training/scripts/preflight_check.py
```

**Expected output**:
```
==============================================================
1. Environment Setup
==============================================================

✓ Tinker API Key (.env)
  → API key found (starts with 'tk-abcd12...')
✓ Python Packages
  → All required packages installed
✓ Disk Space
  → 45.2GB available

==============================================================
2. Training Data (TASK 2)
==============================================================

✓ Training Data (data/tinker/)
  → Train: 1200 samples, Test: 300 samples
✓ ESL Evaluation Data
  → ESL evaluation splits ready

==============================================================
3. Code & Configs
==============================================================

✓ Detectors (TASK 1)
  → Detector implementations ready (TASK 1 completed)
✓ Training Configs
  → All training configs present
✓ Outputs Directory
  → outputs/ directory exists

==============================================================
Summary
==============================================================

✓ ALL CHECKS PASSED (8/8)

You're ready to start TASK 3: RL Training!
```

**If checks fail**: Fix the issues reported before proceeding.

---

### **Step 3: Test with Synthetic Data** (Optional but Recommended)

Test the training pipeline with a small synthetic dataset first:

```bash
# Generate small synthetic dataset (100 training samples)
python scripts/prepare_tinker_data.py \
    --synthetic \
    --num-train 100 \
    --num-test 20 \
    --output-dir data/tinker_test

# Run short training (1 epoch, takes ~5-10 minutes)
python -m stealthrl.tinker.train \
    --config configs/tinker_stealthrl.yaml \
    --data-path data/tinker_test \
    --run-name test_run \
    --num-epochs 1 \
    --batch-size 2

# Check if it completed
ls outputs/runs/test_run/
```

**Expected**: Training completes without errors, checkpoints saved

**If test fails**: Check logs in `outputs/runs/test_run/training.log`

---

### **Step 4: Run Full Ensemble Training**

Now run the actual training with real data:

```bash
# Full ensemble training (all 3 detectors)
python -m stealthrl.tinker.train \
    --config configs/tinker_stealthrl.yaml \
    --data-path data/tinker \
    --run-name stealthrl_full_ensemble \
    --num-epochs 3
```

**What happens**:
- Training job submitted to Tinker
- Model: Qwen3-4B with LoRA
- Detectors: Fast-DetectGPT + Ghostbuster + Binoculars
- Reward: Multi-objective (evasion + semantic + quality + fairness)
- Duration: ~2-4 hours on Tinker
- Checkpoints saved automatically

**Expected output**:
```
[INFO] Submitting training job to Tinker...
[INFO] Job ID: job-abc123xyz
[INFO] Status: RUNNING
[INFO] Monitor at: https://tinker.thinkingmachines.ai/jobs/job-abc123xyz
[INFO] Logs streaming to: outputs/runs/stealthrl_full_ensemble/training.log
```

---

### **Step 5: Monitor Training Progress**

#### Option A: Check Tinker Dashboard
Go to: https://tinker.thinkingmachines.ai/jobs/

#### Option B: Check Local Logs
```bash
# View latest logs
tail -f outputs/runs/stealthrl_full_ensemble/training.log

# Check training status
cat outputs/runs/stealthrl_full_ensemble/status.json
```

#### Option C: Use Monitoring Script
```bash
python scripts/monitor_training.py \
    --run-name stealthrl_full_ensemble
```

**Key metrics to watch**:
- `reward/total`: Should increase over time
- `reward/detector`: Target > 0.5 (high evasion)
- `reward/semantic`: Target > 0.8 (preserve meaning)
- `kl`: Keep < 0.01 (don't drift from base model)
- `all_negative_frac_total`: Should decrease < 0.3

---

### **Step 6: Verify Training Completion**

Once training finishes:

```bash
# Check outputs
ls -lh outputs/runs/stealthrl_full_ensemble/

# Expected files:
# - checkpoints/          # Model checkpoints
# - training.log          # Training logs
# - metrics.json          # Final metrics
# - debug_samples.jsonl   # Example paraphrases
# - config.yaml           # Training config used
```

**View sample paraphrases**:
```bash
# See what the model learned to generate
head -5 outputs/runs/stealthrl_full_ensemble/debug_samples.jsonl | jq
```

**Check final metrics**:
```bash
cat outputs/runs/stealthrl_full_ensemble/metrics.json | jq
```

---

### **Step 7: Run Transfer Experiment** (Optional)

Test if the model generalizes to unseen detectors:

```bash
# Train on 2 detectors only (Fast-DetectGPT + Ghostbuster)
python -m stealthrl.tinker.train \
    --config configs/tinker_transfer_in_ensemble.yaml \
    --data-path data/tinker \
    --run-name transfer_experiment \
    --num-epochs 3

# Evaluate on all 3 detectors (including held-out Binoculars)
python scripts/evaluate_transfer.py \
    --model outputs/runs/transfer_experiment \
    --test-data data/tinker/test.jsonl \
    --output-dir outputs/transfer_evaluation
```

**Goal**: Transfer ratio > 0.70 (ASR on held-out / ASR on in-ensemble)

---

### **Step 8: Run Ablation Studies** (Optional)

Run 5 ablation experiments to understand contribution of each component:

```bash
# Run all ablations in parallel (if you have credits)
bash scripts/run_ablations.sh

# Or run individually:
python -m stealthrl.tinker.train --config configs/ablations/detector_only.yaml --data-path data/tinker --run-name ablation_detector_only
python -m stealthrl.tinker.train --config configs/ablations/no_fairness.yaml --data-path data/tinker --run-name ablation_no_fairness
python -m stealthrl.tinker.train --config configs/ablations/no_semantic.yaml --data-path data/tinker --run-name ablation_no_semantic
python -m stealthrl.tinker.train --config configs/ablations/no_quality.yaml --data-path data/tinker --run-name ablation_no_quality
python -m stealthrl.tinker.train --config configs/ablations/single_detector_fast_detectgpt.yaml --data-path data/tinker --run-name ablation_single_detector
```

**Evaluate all ablations**:
```bash
python scripts/evaluate_ablations.py \
    --ablation-dir outputs/runs \
    --test-data data/tinker/test.jsonl \
    --output-dir outputs/ablation_analysis
```

**Estimated time**: 2-4 hours per ablation (10-20 hours total if sequential)

---

## 📊 Expected Results

After successful training:

| Metric | Target | Where to Check |
|--------|--------|----------------|
| **ASR (all detectors)** | 60-70% | `metrics.json` → `asr_all` |
| **Semantic Similarity** | 0.88-0.92 | `metrics.json` → `semantic_sim_mean` |
| **ESL FPR Gap** | <0.07 | `metrics.json` → `esl_fpr_gap` |
| **Transfer Ratio** | >0.70 | Transfer evaluation results |
| **Training Loss** | Decreasing | Training logs |

---

## 🐛 Troubleshooting

### Issue 1: "Invalid API key"
```bash
# Check .env file
cat .env | grep TINKER_API_KEY

# Re-run setup
bash task3_rl_training/scripts/setup_env.sh
```

### Issue 2: "Training data not found"
```bash
# Complete TASK 2 first
ls data/tinker/

# If missing, run:
python scripts/prepare_tinker_data.py \
    --input-paths data/esl/toefl11.jsonl data/native/native_academic.jsonl \
    --output-dir data/tinker
```

### Issue 3: "Detectors not working"
```bash
# Complete TASK 1 first
cd task1_detector_implementation
python test_detectors_standalone.py
```

### Issue 4: Training hangs at "All-Negative Groups"
**Symptom**: `all_negative_frac_total > 0.8`

**Fix**: Update config (edit `configs/tinker_stealthrl.yaml`):
```yaml
all_negative:
  min_reward: 0.05  # Increase from 0.01
  downweight: 0.3   # Reduce from 0.5
```

### Issue 5: Low semantic similarity
**Symptom**: `semantic_sim_mean < 0.80`

**Fix**: Increase semantic weight:
```yaml
reward:
  semantic_weight: 2.0  # Increase from 1.0
```

### Issue 6: Out of Tinker credits
- Check balance: https://tinker.thinkingmachines.ai/billing
- DSC 291 should have sponsored credits
- Contact instructor if depleted

---

## 📁 Output Structure

After training:

```
outputs/runs/stealthrl_full_ensemble/
├── checkpoints/
│   ├── checkpoint_epoch_1/
│   ├── checkpoint_epoch_2/
│   └── checkpoint_final/
├── training.log
├── metrics.json
├── debug_samples.jsonl
├── config.yaml
└── status.json
```

---

## 🎓 Next Steps After TASK 3

Once training is complete:

1. **TASK 4**: ESL Fairness Evaluation
   ```bash
   python scripts/run_esl_eval.py \
       --eval_data data/processed/esl_native_test.jsonl \
       --stealthrl_model outputs/runs/stealthrl_full_ensemble \
       --output_dir results/esl_eval
   ```

2. **Generate Visualizations**:
   ```bash
   python scripts/visualize_stealthbench.py \
       --results results/esl_eval \
       --output-dir outputs/figures
   ```

3. **Write Results**: Compile findings for paper/report

---

## 📚 Additional Resources

- **Tinker Documentation**: https://tinker-docs.thinkingmachines.ai/
- **Tinker Platform**: https://tinker.thinkingmachines.ai/
- **Training Configs**: `configs/tinker_stealthrl.yaml`
- **Knowledge Base**: `knowledge_base/TINKER_README.md`
- **Research Roadmap**: `knowledge_base/RESEARCH_ROADMAP.md`

---

## 💡 Pro Tips

1. **Start with test run** (Step 3) to catch issues early
2. **Monitor training regularly** - Check dashboard every 30 minutes
3. **Save checkpoint paths** - You'll need them for evaluation
4. **Use TensorBoard** - `tensorboard --logdir outputs/runs`
5. **Parallelize ablations** - Run multiple experiments simultaneously if you have credits
6. **Document everything** - Keep notes on hyperparameters and results

---

**Ready to train? Start with Step 0 to verify prerequisites!** 🚀
