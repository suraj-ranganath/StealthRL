# TASK 3: RL Training

**Status**: ✅ **SETUP COMPLETE** - Ready for execution

---

## 📁 Task 3 Organization

All TASK 3 materials are organized in the `task3_rl_training/` folder.

```
task3_rl_training/
├── QUICK_START.md                     # ⭐ START HERE - Step-by-step guide
├── scripts/
│   ├── preflight_check.py            # ✅ Pre-flight validation
│   ├── setup_env.sh                  # ✅ Tinker API key setup
│   └── monitor_training.py           # ✅ Training monitor
├── logs/                             # Training logs (created during training)
└── configs/                          # Custom configs (optional)
```

---

## 🎯 What is TASK 3?

**Objective**: Train StealthRL model using Reinforcement Learning on Tinker platform

**What you'll train**:
- **Base Model**: Qwen3-4B with LoRA adapters
- **Algorithm**: GRPO (Group-Relative Policy Optimization)
- **Detectors**: Fast-DetectGPT + Ghostbuster + Binoculars
- **Reward**: Multi-objective (evasion + semantic + quality + fairness)

**Expected Results**:
- ASR (Attack Success Rate): 60-70%
- Semantic Similarity: 0.88-0.92
- ESL FPR Gap: <0.07
- Transfer Ratio: >0.70

---

## 🚀 Quick Start

### **Step 1: Configure Tinker API Key**
```bash
cd /Users/nishchaymahor/Documents/Study/291\ -\ Safety\ in\ Gen\ AI/StealthRL/StealthRL

# Run setup script
bash task3_rl_training/scripts/setup_env.sh
```

**Get API key**: https://tinker.thinkingmachines.ai/ → Settings → API Keys

---

### **Step 2: Run Pre-Flight Check**
```bash
python task3_rl_training/scripts/preflight_check.py
```

**This checks**:
- ✓ Tinker API key configured
- ✓ Training data ready (TASK 2)
- ✓ Detectors working (TASK 1)
- ✓ All configs present
- ✓ Sufficient disk space

---

### **Step 3: Start Training**
```bash
# Full ensemble training
python -m stealthrl.tinker.train \
    --config configs/tinker_stealthrl.yaml \
    --data-path data/tinker \
    --run-name stealthrl_full_ensemble \
    --num-epochs 3
```

**Duration**: ~2-4 hours on Tinker (runs remotely)

---

### **Step 4: Monitor Progress**
```bash
# Real-time monitoring
python task3_rl_training/scripts/monitor_training.py \
    --run-name stealthrl_full_ensemble \
    --watch

# Or check Tinker dashboard
# https://tinker.thinkingmachines.ai/jobs/
```

---

## 📊 Training Experiments

### **Experiment 1: Full Ensemble** (Required)
Train on all 3 detectors:
```bash
python -m stealthrl.tinker.train \
    --config configs/tinker_stealthrl.yaml \
    --data-path data/tinker \
    --run-name stealthrl_full_ensemble \
    --num-epochs 3
```

**Goal**: Achieve high ASR across all detectors

---

### **Experiment 2: Transfer Learning** (Optional)
Train on 2 detectors, test on 3rd:
```bash
python -m stealthrl.tinker.train \
    --config configs/tinker_transfer_in_ensemble.yaml \
    --data-path data/tinker \
    --run-name transfer_experiment \
    --num-epochs 3
```

**Goal**: Transfer ratio > 0.70

---

### **Experiment 3: Ablation Studies** (Optional)
Test contribution of each component:
```bash
bash scripts/run_ablations.sh
```

**5 ablations**:
1. Detector-only (no semantic/quality/fairness)
2. No fairness penalty
3. No semantic similarity
4. No quality control
5. Single detector baseline

---

## 🛠️ Tools Provided

### **1. Pre-Flight Check** ✅
```bash
python task3_rl_training/scripts/preflight_check.py
```
Validates all prerequisites before training

### **2. Environment Setup** ✅
```bash
bash task3_rl_training/scripts/setup_env.sh
```
Interactive Tinker API key configuration

### **3. Training Monitor** ✅
```bash
python task3_rl_training/scripts/monitor_training.py --run-name <name> --watch
```
Real-time dashboard showing:
- Training status
- Key metrics (rewards, KL, ASR)
- Issue detection
- Recent logs

---

## 📁 Expected Outputs

After training completes:

```
outputs/runs/stealthrl_full_ensemble/
├── checkpoints/
│   ├── checkpoint_epoch_1/
│   ├── checkpoint_epoch_2/
│   └── checkpoint_final/        # ← Use this for evaluation
├── training.log                  # Full training logs
├── metrics.json                  # Final metrics
├── debug_samples.jsonl          # Example paraphrases
├── config.yaml                  # Config used
└── status.json                  # Training status
```

---

## ✅ Success Criteria

TASK 3 is complete when:

- [x] Tinker API key configured
- [x] Pre-flight check passes
- [ ] Full ensemble training completes
- [ ] Final metrics achieved:
  - ASR > 60%
  - Semantic similarity > 0.85
  - ESL FPR gap < 0.10
- [ ] Checkpoints saved
- [ ] Sample paraphrases generated

---

## 🐛 Common Issues

### **Issue**: "Invalid API key"
**Solution**: Re-run `bash task3_rl_training/scripts/setup_env.sh`

### **Issue**: "Training data not found"
**Solution**: Complete TASK 2 first
```bash
python scripts/prepare_tinker_data.py \
    --input-paths data/esl/toefl11.jsonl data/native/native_academic.jsonl \
    --output-dir data/tinker
```

### **Issue**: "Detectors not working"
**Solution**: Complete TASK 1 first
```bash
cd task1_detector_implementation
python test_detectors_standalone.py
```

### **Issue**: High all-negative fraction
**Solution**: Edit `configs/tinker_stealthrl.yaml`:
```yaml
all_negative:
  min_reward: 0.05  # Increase from 0.01
```

### **Issue**: Low semantic similarity
**Solution**: Edit `configs/tinker_stealthrl.yaml`:
```yaml
reward:
  semantic_weight: 2.0  # Increase from 1.0
```

---

## 🎓 Next Steps After TASK 3

Once training completes:

### **1. ESL Fairness Evaluation (TASK 4)**
```bash
python scripts/run_esl_eval.py \
    --eval_data data/processed/esl_native_test.jsonl \
    --stealthrl_model outputs/runs/stealthrl_full_ensemble \
    --enable_bertscore \
    --output_dir results/esl_eval
```

### **2. Generate Visualizations**
```bash
python scripts/visualize_stealthbench.py \
    --results results/esl_eval \
    --output-dir outputs/figures
```

### **3. Analyze Results**
- Check `results/esl_eval/comparison_report.json`
- Review ESL FPR gaps
- Generate paper figures

---

## 📚 Documentation

- **Quick Start**: `task3_rl_training/QUICK_START.md` ⭐
- **Tinker Guide**: `knowledge_base/TINKER_README.md`
- **Research Roadmap**: `knowledge_base/RESEARCH_ROADMAP.md`
- **Team Handoff**: `knowledge_base/TEAM_HANDOFF.md`

---

## 💡 Pro Tips

1. **Test first**: Run short training with synthetic data
2. **Monitor actively**: Check dashboard every 30 minutes
3. **Save outputs**: Document checkpoint paths
4. **Use TensorBoard**: `tensorboard --logdir outputs/runs`
5. **Parallelize**: Run ablations simultaneously if you have credits

---

## 🚦 Current Status

| Component | Status | Location |
|-----------|--------|----------|
| Pre-flight check | ✅ Ready | `task3_rl_training/scripts/preflight_check.py` |
| Environment setup | ✅ Ready | `task3_rl_training/scripts/setup_env.sh` |
| Training monitor | ✅ Ready | `task3_rl_training/scripts/monitor_training.py` |
| Quick start guide | ✅ Ready | `task3_rl_training/QUICK_START.md` |
| Tinker API key | ⏸️ Needs setup | Run `setup_env.sh` |
| Training | ⏸️ Ready to run | Follow QUICK_START.md |

---

**Ready to train? Start with: `task3_rl_training/QUICK_START.md`** 🚀
