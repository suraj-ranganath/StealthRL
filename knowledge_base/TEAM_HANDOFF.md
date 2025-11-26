# StealthRL Team Handoff Document

**Date**: November 26, 2025  
**From**: Suraj Ranganath  
**To**: Project Team (Nishchay Mahor, Sibo Zhu)  
**Course**: DSC 291 - Safety in Generative AI

---

## 🎯 Project Overview

**StealthRL** is a research project investigating whether RL can train a single paraphraser to evade multiple AI text detectors while maintaining semantic quality and reducing ESL bias. We're exploring **multi-detector ensemble training** to learn detector-agnostic strategies.

**Key Innovation**: Train ONE model against MULTIPLE detectors (vs. AuthorMist's one-model-per-detector approach).

---

## 📋 Quick Start for New Team Members

### 1. Get Set Up (30 minutes)

```bash
# Clone and enter directory
cd /Users/suraj/Desktop/StealthRL

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Get Tinker API key from https://tinker.thinkingmachines.ai/
# Add to .env file:
nano .env
# Replace: TINKER_API_KEY=your_tinker_api_key_here
# With: TINKER_API_KEY=tk-abc123xyz789...

# Quick test (5 mins)
python scripts/prepare_tinker_data.py --synthetic --num-train 10 --output-dir data/tinker_test
python -m stealthrl.tinker.train --data-path data/tinker_test --run-name test --num-epochs 1 --batch-size 2
```

### 2. Read These First

1. **README.md** - Complete overview, setup, project status
2. **knowledge_base/SETUP_AND_RUN.md** - Detailed setup instructions
3. **knowledge_base/RESEARCH_ROADMAP.md** - What's done, what's next
4. **This file** - You're reading it!

---

## ✅ What's Already Done

### Infrastructure (100% Complete)
- ✅ **Tinker integration**: Full GRPO training pipeline (~3,555 lines)
- ✅ **Multi-objective reward**: Detector + semantic + quality + fairness
- ✅ **Evaluation suite**: ASR, AUROC, FPR, ESL fairness metrics
- ✅ **Configurations**: 7 YAML files (full ensemble, transfer, 5 ablations)
- ✅ **Documentation**: 13 guides in `knowledge_base/`
- ✅ **Checkpoint system**: Remote storage on Tinker servers

### Testing
- ✅ **Pipeline validated**: End-to-end training works with synthetic data
- ✅ **Checkpoint saving**: Tested and working
- ✅ **Reward computation**: All components functional
- ✅ **GRPO algorithm**: Validated with mock detectors

**Bottom line**: The code is production-ready. We can train models right now.

---

## 🔨 What Needs to Be Done

### Critical Path (Must Complete)

#### Week 1: Setup Phase

**Task 1: Real Detectors (HIGH PRIORITY)**
- **Status**: Mock implementations exist, need real ones
- **Who**: Person with GPU access (8-16GB VRAM) - You can possibly use UCSD independant study compute too. Request here for access: 
- **Time**: 1-2 days
- **Steps**:
  1. Install detector packages: `pip install fast-detectgpt ghostbuster binoculars-detect`
  2. Update `stealthrl/tinker/detectors.py` (replace mock `_compute_score()` methods)
  3. Test: `python -c "from stealthrl.tinker.detectors import DetectorEnsemble; ..."`
  4. **Guide**: `knowledge_base/DETECTOR_SETUP.md`
- **Blocker**: Need to download models (~10GB total)

**Task 2: Dataset Curation (HIGH PRIORITY)**
- **Status**: Data pipeline ready, need real datasets
- **Who**: Person good with data processing
- **Time**: 2-3 days
- **Steps**:
  1. Get ESL corpus: TOEFL11, ICNALE, ELLIPSE
  2. Get native corpus: Academic papers, essays
  3. Convert to JSONL format (see format below)
  4. Run: `python scripts/prepare_tinker_data.py --input-paths ... --output-dir data/tinker`
  5. **Guide**: `knowledge_base/ESL_FAIRNESS_GUIDE.md`
- **Target**: 40% ESL, 60% native split

**JSONL Format Required**:
```json
{
  "ai_text": "AI-generated text here...",
  "human_reference": "Original human text...",
  "domain": "academic",
  "is_esl": true,
  "metadata": {"source": "TOEFL11"}
}
```

#### Week 2-3: Training Phase

**Task 3: Main RL Training (MEDIUM PRIORITY)**
- **Prerequisites**: Real detectors + real data
- **Time**: 2-4 hours per experiment on Tinker
- **Experiments**:
  1. Full ensemble (all 3 detectors)
  2. Transfer experiment (train on 2, test on 3rd)
  3. 5 ablations (detector-only, no-fairness, etc.)
- **Commands**: See README "Priority 3" section
- **Monitor**: Use TensorBoard or check logs

**Task 4: ESL Fairness Evaluation (MEDIUM PRIORITY)**
- **Prerequisites**: ESL-stratified data + trained model
- **Time**: 1-2 hours
- **Goal**: Measure FPR gap between ESL and native writers
- **Target**: Reduce gap from 0.15 to <0.07
- **Command**: See README "Priority 4" section

#### Week 4: Analysis Phase

**Task 5: Results & Writing**
- **All team members**: Compile results, generate figures, write paper

---

## 📊 Expected Results

| Metric | Target | How to Check |
|--------|--------|--------------|
| **ASR (all detectors)** | 60-70% | Check `asr_all` in results JSON |
| **Semantic Similarity** | 0.88-0.92 | Check `semantic_sim_mean` |
| **ESL FPR Gap** | <0.07 | Check `esl_fpr_gap` per detector |
| **Transfer Ratio** | >0.70 | ASR(held-out) / ASR(in-ensemble) |

---

## 🎯 Tasks

- Set up real detectors
- Curate ESL/native datasets
- Review docs, coordinate, test setup
- Run training experiments
- Prepare ESL evaluation
- Monitor training, debug issues
- Run ablations
- ESL evaluation
- Generate visualizations
- Results analysis, paper writing

---

## 📁 Important Files & Locations

### Configuration Files
- `configs/tinker_stealthrl.yaml` - Main training config
- `configs/tinker_transfer_in_ensemble.yaml` - Transfer experiment
- `configs/ablations/*.yaml` - 5 ablation experiments

### Key Scripts
- `scripts/prepare_tinker_data.py` - Data preparation
- `stealthrl/tinker/train.py` - Main training loop
- `scripts/run_esl_eval.py` - ESL fairness evaluation
- `scripts/visualize_stealthbench.py` - Generate plots

### Documentation (knowledge_base/)
- `SETUP_AND_RUN.md` - Complete setup guide
- `DETECTOR_SETUP.md` - How to set up real detectors
- `ESL_FAIRNESS_GUIDE.md` - ESL evaluation guide
- `RESEARCH_ROADMAP.md` - Detailed research plan
- `CHECKPOINT_GUIDE.md` - How checkpoints work

### Output Locations
- `outputs/runs/<run_name>/` - Training outputs
- `outputs/runs/<run_name>/training.log` - Training logs
- `outputs/runs/<run_name>/checkpoints/` - Model checkpoints

---

## 🚨 Common Issues & Solutions

### "Invalid API key"
- Check `.env` file has correct key (starts with `tk-`)
- No extra spaces around the key

### "File not found" during training
- Generate data first: `python scripts/prepare_tinker_data.py --synthetic ...`
- Check: `ls data/tinker/*.jsonl`

### Mock detectors giving same scores
- **This is expected!** Mock detectors are deterministic
- Need to implement real detectors (Task 1)

### Training hangs
- Check Tinker dashboard for active jobs
- Verify credits available (DSC 291 has sponsored credits)
- Restart training if network issue

### Import errors
- Make sure venv is activated: `source venv/bin/activate`
- Check: `which python` points to `./venv/bin/python`

---

## 🔗 Quick Links

**Main Documentation**:
- [README.md](README.md) - Project overview
- [REPORT.md](REPORT.md) - Comprehensive project report
- [knowledge_base/](knowledge_base/) - All detailed guides

**External Resources**:
- [Tinker Platform](https://tinker.thinkingmachines.ai/)
- [Tinker Docs](https://tinker-docs.thinkingmachines.ai/)
- [AuthorMist Paper](https://arxiv.org/abs/2503.08716) (our main reference)

**Essential References for Agents & Humans**:
- [Tinker Full Docs for LLMs](https://tinker-docs.thinkingmachines.ai/llms-full.txt) - Complete API reference for AI agents
- [Tinker Cookbook for Agents](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/AGENTS.md) - Best practices for building with Tinker
- [LoRA with RL Best Practices](https://thinkingmachines.ai/blog/lora/) - How to effectively use LoRA in RL training
- [GRPO RL Training Tips](https://github.com/zechenzhangAGI/AI-research-SKILLs/tree/main/06-post-training/grpo-rl-training) - Practical tips for GRPO algorithm

**GitHub Issues/Questions**:
- Check [interaction_records.md](interaction_records.md) for implementation history
- Search documentation in `knowledge_base/`
- Ask team in Slack/Discord

---

## 📈 Success Criteria

### Minimum Viable Results
- ✅ Training completes successfully on real data
- ✅ Model checkpoints saved
- ✅ Detector scores decrease (any amount)
- ✅ Semantic similarity >0.85

### Target Results (for paper)
- 🎯 ASR (all detectors) >60%
- 🎯 Semantic similarity 0.88-0.92
- 🎯 ESL FPR gap <0.07
- 🎯 Transfer ratio >0.70

### Stretch Goals
- 🌟 Compare against SICO baseline
- 🌟 Human evaluation study
- 🌟 Selective fine-tuning ablation

---

## 💡 Tips for Success

1. **Start with synthetic data**: Test everything with `--synthetic` flag first
2. **Use small batches**: Start with `--batch-size 2` to avoid OOM errors
3. **Monitor early**: Check training logs frequently in first few iterations
4. **Parallelize**: Detectors and data can work in parallel
5. **Document**: Keep notes on what works/doesn't work. Log everything and keep those logs organised. The training logs + output is a good reference for the standard to be maintained.
6. **Ask questions**: Better to ask than waste time debugging alone

---

## 📞 Contact

**Original Developer**: Suraj Ranganath
- Email: sranganath@ucsd.edu
- Available for questions via Slack/email

**Team Members**:
- Nishchay Mahor
- Sibo Zhu

**Course Staff**:
- Instructor: [Check Canvas]
- TAs: [Check Canvas]

---

## ✅ Pre-Flight Checklist

Before you start work, make sure:

- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Tinker API key in `.env` file
- [ ] Quick test completed successfully
- [ ] Read README.md and SETUP_AND_RUN.md
- [ ] Understand task assignments
- [ ] Have access to Tinker platform
- [ ] Know where to find documentation

---

**Good luck! The infrastructure is solid—now let's get those results! 🚀**
