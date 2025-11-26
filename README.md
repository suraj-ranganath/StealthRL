# StealthRL

**Ensemble-Guided Text Transformation for Multi-Detector Transfer and Fair Detection Robustness**

> RL-trained paraphraser for multi-detector-robust, fair text generation.

🚀 **NEW: Tinker Integration** - Now supports training on remote compute with Qwen3-4B via the [Tinker platform](https://tinker-docs.thinkingmachines.ai/). See [TINKER_README.md](knowledge_base/TINKER_README.md) for quickstart and DSC 291 deployment guide.

---

## Overview

StealthRL is a research framework that uses **Reinforcement Learning with Verifiable Rewards (RFT)** to train a single, ensemble-guided paraphraser capable of reducing AI-detector scores while preserving semantic meaning and text quality. Unlike prior approaches that train separate models per detector (e.g., AuthorMist), StealthRL investigates whether **joint training against a detector ensemble** can learn detector-agnostic transformation strategies that generalize to unseen detector families.

A core focus of this project is **fairness**: AI text detectors have been shown to produce elevated false-positive rates on writing by ESL (English as a Second Language) authors. StealthRL explicitly monitors and optimizes for shrinking the ESL vs native FPR gap, treating fairness as a first-class objective rather than an afterthought.

### Two Deployment Options

- **Local Training (Original)**: HuggingFace TRL + local GPUs (see sections below)
- **Tinker Platform (DSC 291)**: Remote compute, Qwen3-4B, GRPO enhancements → [TINKER_README.md](knowledge_base/TINKER_README.md)

---

## Motivation

AI text detectors are increasingly deployed in academic integrity and content moderation settings, but they suffer from two critical issues:

1. **Brittleness**: Detectors often fail to generalize across paraphrasing attacks, and models trained to evade one detector may not transfer to others.
2. **Bias**: Studies have documented that detectors disproportionately flag ESL writing as AI-generated, raising serious fairness concerns.

Prior work like **AuthorMist** demonstrates that RL can train paraphrasers using detector outputs as reward signals, but typically trains one model per detector and does not deeply address fairness. StealthRL extends this line of research by:

- Training a **single model** against a **multi-detector ensemble** (e.g., a classifier-style detector plus a curvature-based method) within the same RL loop.
- Evaluating **out-of-ensemble transfer** to held-out detector families (e.g., paired-LM methods like Binoculars, feature-ensemble classifiers like Ghostbuster).
- Incorporating an explicit **fairness penalty** to reduce the ESL vs native false-positive gap.

---

## Key Features

- **RL with Verifiable Rewards** via HuggingFace TRL and LoRA adapters for efficient, parameter-efficient training.
- **Multi-Detector Reward Ensemble**, combining:
  - Classifier-style detectors (e.g., Ghostbuster, RoBERTa-based classifiers)
  - Curvature-based detectors (e.g., DetectGPT, Fast-DetectGPT)
- **Out-of-Ensemble Transfer Evaluation** on held-out detector families (e.g., Binoculars, held-out Ghostbuster variants).
- **StealthBench**: A unified evaluation harness that runs multiple detectors on the same texts and outputs standardized metrics (AUROC, FPR@0.5%, FPR@1%) and comparison plots.
- **Fairness-Aware Evaluation**: Tracks ESL vs native FPR and includes a fairness term in the reward to shrink this gap.
- **Semantic and Quality Controls**: BERTScore for meaning preservation, perplexity banding, and readability metrics to prevent degenerate outputs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              StealthRL Pipeline                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐      ┌───────────────────────┐      ┌──────────────────┐ │
│   │ Input Text  │ ──▶  │  StealthRL Paraphraser │ ──▶  │ Paraphrased Text │ │
│   │ (LLM-gen)   │      │  (Base LM + LoRA)      │      │                  │ │
│   └─────────────┘      └───────────────────────┘      └────────┬─────────┘ │
│                                                                 │           │
│                        ┌────────────────────────────────────────┘           │
│                        ▼                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         Reward Pipeline                             │   │
│   ├─────────────────────────────────────────────────────────────────────┤   │
│   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │   │
│   │  │ Detector Scores │  │ Semantic Fidelity│  │ Quality Metrics    │  │   │
│   │  │ (Ensemble)      │  │ (BERTScore/Cos) │  │ (PPL, Readability) │  │   │
│   │  │ ▶ detectability │  │ ▶ meaning       │  │ ▶ fluency          │  │   │
│   │  │   penalty       │  │   preservation  │  │   constraints      │  │   │
│   │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │   │
│   │                                                                     │   │
│   │  ┌─────────────────────────────────────────────────────────────┐    │   │
│   │  │ Fairness Term: Penalize ESL vs Native FPR gap               │    │   │
│   │  └─────────────────────────────────────────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                        │                                                    │
│                        ▼                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │              RL Trainer (GRPO/PPO via TRL)                          │   │
│   │              Updates LoRA parameters                                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

The implementation is **modular**: detectors, reward terms, and base models can be swapped via configuration files without changing core training code.

---

## Repository Structure

```
stealthrl/
├── models/          # Base LM loading, LoRA adapter utilities
├── rewards/         # Composite reward computation (detectors, BERTScore, PPL, fairness)
├── detectors/       # Wrappers for Fast-DetectGPT, Ghostbuster, Binoculars, etc.
├── training/        # RL training loops (GRPO/PPO via HuggingFace TRL)
├── evaluation/      # StealthBench metrics: AUROC, FPR, BERTScore, perplexity
└── tinker/          # Tinker platform integration (env, dataset, reward, training)

scripts/
├── prepare_data.py        # Prepare human/LLM text, ESL vs native subsets
├── train_stealthrl.py     # Main RL training entry point
├── evaluate_detectors.py  # Run detector ensemble, produce CSVs
├── run_stealthbench.py    # Unified evaluation harness
└── download_datasets.sh   # Download datasets from original sources

configs/               # YAML/JSON configs for models, training, detectors
examples/              # Sample scripts and notebooks
data/                  # Small toy data (large datasets downloaded separately)
knowledge_base/        # Comprehensive documentation (guides, setup, API docs)
requirements.txt       # Python dependencies
environment.yml        # Conda environment (optional)
LICENSE
```

### 📚 Documentation

All comprehensive guides and documentation are organized in the [`knowledge_base/`](knowledge_base/) directory. **Start here**: [`knowledge_base/README.md`](knowledge_base/README.md) for a complete index and navigation guide.

**Quick Access:**
- **Getting Started**: [SETUP_AND_RUN.md](knowledge_base/SETUP_AND_RUN.md), [QUICKSTART.md](knowledge_base/QUICKSTART.md), [QUICK_START_RUNS.md](knowledge_base/QUICK_START_RUNS.md)
- **Platform**: [TINKER_README.md](knowledge_base/TINKER_README.md) - Tinker integration guide
- **Implementation**: [CHECKPOINT_GUIDE.md](knowledge_base/CHECKPOINT_GUIDE.md), [CHECKPOINT_IMPLEMENTATION.md](knowledge_base/CHECKPOINT_IMPLEMENTATION.md), [REWARD_REFINEMENT.md](knowledge_base/REWARD_REFINEMENT.md), [DETECTOR_SETUP.md](knowledge_base/DETECTOR_SETUP.md)
- **Evaluation**: [ESL_FAIRNESS_GUIDE.md](knowledge_base/ESL_FAIRNESS_GUIDE.md), [IMPLEMENTATION_VERIFICATION.md](knowledge_base/IMPLEMENTATION_VERIFICATION.md)
- **Research**: [RESEARCH_ROADMAP.md](knowledge_base/RESEARCH_ROADMAP.md), [NEXT_STEPS.md](knowledge_base/NEXT_STEPS.md)
- **Operations**: [RUN_MANAGEMENT.md](knowledge_base/RUN_MANAGEMENT.md)

---

## 🚀 Getting Started (For New Team Members)

### Prerequisites

**Required:**
- Python 3.10 or higher
- Tinker API key (from [Tinker Platform](https://tinker.thinkingmachines.ai/))
- 5-10 GB disk space for dependencies and data

**Optional (for local detector testing):**
- NVIDIA GPU with CUDA 11.8+ (8-16GB VRAM recommended)
- If no GPU: Use Tinker's remote compute (recommended for DSC 291)

### Step-by-Step Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/suraj-ranganath/StealthRL.git
cd StealthRL
```

#### 2. Create Virtual Environment

**IMPORTANT**: Always use a virtual environment to avoid dependency conflicts.

```bash
# Create virtual environment
python3 -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Activate it (Windows)
# venv\Scripts\activate
```

**Verify activation**: Your terminal prompt should now show `(venv)` at the beginning.

#### 3. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**Expected time**: 2-5 minutes. If errors occur, see [Troubleshooting](#troubleshooting) below.

#### 4. Set Up Tinker API Key

**Get your API key:**
1. Go to [Tinker Platform](https://tinker.thinkingmachines.ai/)
2. Sign in (use your UCSD credentials for DSC 291)
3. Navigate to **Settings** → **API Keys**
4. Copy your API key (starts with `tk-`)

**Add to .env file:**

```bash
# Open the .env file
nano .env
# OR: code .env (if using VS Code)
# OR: open -e .env (macOS TextEdit)
```

Find this line:
```bash
TINKER_API_KEY=your_tinker_api_key_here
```

Replace it with your actual key:
```bash
TINKER_API_KEY=tk-abc123xyz789...
```

**Save and verify:**
```bash
grep TINKER_API_KEY .env
# Should show: TINKER_API_KEY=tk-...
```

#### 5. Quick Test (5 minutes)

Verify everything works:

```bash
# Generate synthetic test data
python scripts/prepare_tinker_data.py --synthetic --num-train 10 --output-dir data/tinker_test

# Run a quick training test (1 epoch)
python -m stealthrl.tinker.train \
  --data-path data/tinker_test \
  --run-name test_run \
  --num-epochs 1 \
  --batch-size 2

# Check that it created output
ls outputs/runs/test_run/
```

**Success indicators:**
- ✅ No error messages
- ✅ `outputs/runs/test_run/` directory created
- ✅ `training.log` file shows training progress
- ✅ Checkpoint info saved

### Core Dependencies

The project uses these main packages (all auto-installed via `requirements.txt`):

- `tinker-ai` - Tinker platform API for remote compute
- `transformers` - HuggingFace models (Qwen3-4B)
- `torch` - PyTorch deep learning framework
- `sentence-transformers` - E5 embeddings for semantic similarity
- `bert-score` - BERTScore for evaluation
- `peft` - LoRA adapters for efficient fine-tuning
- `trl` - Reinforcement learning utilities

**Total install size**: ~3-4 GB

---

## 📊 Project Status: What's Done vs. What's Remaining

### ✅ COMPLETED (Ready to Use)

#### Infrastructure & Training Pipeline
- ✅ **Complete Tinker integration** (~3,555 lines of code)
  - GRPO training loop with Qwen3-4B
  - LoRA adapters (rank 16, efficient training)
  - Async reward computation
  - Checkpoint management (remote storage on Tinker)
  
- ✅ **Multi-objective reward function** with normalization
  - Detector ensemble scoring (Fast-DetectGPT, Ghostbuster, Binoculars)
  - Semantic similarity (E5 embeddings)
  - Perplexity-based quality control (GPT-2)
  - ESL fairness penalty (per-sample)
  - Z-score normalization, threshold-based gating
  - KL regularization (β=0.001, AuthorMist-inspired)

- ✅ **Training configurations**
  - Full ensemble config (`configs/tinker_stealthrl.yaml`)
  - Transfer learning config (`configs/tinker_transfer_in_ensemble.yaml`)
  - 5 ablation configs (detector-only, no-fairness, etc.)

- ✅ **Comprehensive evaluation suite**
  - ASR (Attack Success Rate) metrics
  - AUROC, F1, FPR@TPR95
  - Low-FPR metrics (FPR@0.5%, FPR@1%)
  - ESL fairness gap tracking
  - BERTScore and E5 cosine similarity

- ✅ **Pipeline testing with synthetic data**
  - ✅ Successfully tested end-to-end training
  - ✅ Checkpoint saving/loading verified
  - ✅ Reward computation working
  - ✅ GRPO algorithm validated

#### Documentation
- ✅ **13 comprehensive guides** in `knowledge_base/`
- ✅ **Setup instructions** (this file + SETUP_AND_RUN.md)
- ✅ **Research roadmap** with priorities
- ✅ **Implementation verification** report

### 🔨 IN PROGRESS / TODO (Team Tasks)

#### Priority 1: Detector Setup (HIGH - Week 1)
**Status**: Mock implementations exist, need real detectors

**What needs to be done:**
1. **Install detector packages**:
   ```bash
   pip install fast-detectgpt ghostbuster binoculars-detect
   # OR clone from GitHub repos
   ```

2. **Update detector implementations** in `stealthrl/tinker/detectors.py`:
   - Replace mock `_compute_score()` with real model calls
   - Current: Returns dummy scores based on text length
   - Needed: Load actual detector models (Fast-DetectGPT, Ghostbuster, Binoculars)
   - See [`knowledge_base/DETECTOR_SETUP.md`](knowledge_base/DETECTOR_SETUP.md) for detailed instructions

3. **Test detectors**:
   ```bash
   python -c "from stealthrl.tinker.detectors import DetectorEnsemble; ..."
   ```

**Who should do this**: Team member with GPU access (8-16GB VRAM needed)  
**Estimated time**: 1-2 days  
**Blocker**: Models need to be downloaded (~10GB total)

---

#### Priority 2: Dataset Curation (HIGH - Week 1-2)
**Status**: Data pipeline ready, need real datasets

**What needs to be done:**
1. **Curate ESL/Native corpus** for fairness evaluation:
   - **ESL sources**: TOEFL11, ICNALE, ELLIPSE
   - **Native sources**: Academic papers, essays
   - **Target split**: 40% ESL, 60% native
   - See [`knowledge_base/ESL_FAIRNESS_GUIDE.md`](knowledge_base/ESL_FAIRNESS_GUIDE.md)

2. **Prepare JSONL files** with this format:
   ```json
   {
     "ai_text": "AI-generated text here...",
     "human_reference": "Original human text...",
     "domain": "academic",
     "is_esl": true,
     "metadata": {"source": "TOEFL11"}
   }
   ```

3. **Run data preparation**:
   ```bash
   python scripts/prepare_tinker_data.py \
     --input-paths data/raw/toefl11.jsonl data/raw/icnale.jsonl \
     --output-dir data/tinker \
     --train-split 0.8
   ```

**Who should do this**: Team member comfortable with data processing  
**Estimated time**: 2-3 days (includes data collection + preprocessing)  
**Resources needed**: Access to TOEFL11 corpus (may require permissions)

---

#### Priority 3: Main RL Training (MEDIUM - Week 2-3)
**Status**: Infrastructure ready, waiting on detectors + data

**What needs to be done:**
1. **Full ensemble training** (all 3 detectors):
   ```bash
   python -m stealthrl.tinker.train \
     --config configs/tinker_stealthrl.yaml \
     --data-path data/tinker \
     --run-name full_ensemble \
     --num-epochs 3
   ```
   - **Expected time**: 2-4 hours on Tinker
   - **Checkpoint**: Saved to `outputs/runs/full_ensemble/`

2. **Transfer learning experiment**:
   ```bash
   python -m stealthrl.tinker.train \
     --config configs/tinker_transfer_in_ensemble.yaml \
     --data-path data/tinker \
     --run-name transfer_experiment
   ```
   - **Goal**: Train on Fast-DetectGPT + Ghostbuster only
   - **Evaluate**: Test on held-out Binoculars

3. **Ablation studies** (5 experiments):
   ```bash
   bash scripts/run_ablations.sh
   ```
   - **Expected time**: 10-15 hours total (can parallelize)

**Who should do this**: Team member monitoring training  
**Prerequisites**: Detectors working + real data ready  
**Compute**: Tinker credits (DSC 291 sponsored)

---

#### Priority 4: ESL Fairness Evaluation (MEDIUM - Week 3)
**Status**: Evaluation code ready, need ESL-stratified data

**What needs to be done:**
1. **Run ESL evaluation pipeline**:
   ```bash
   python scripts/run_esl_eval.py \
     --eval_data data/processed/esl_native_test.jsonl \
     --stealthrl_model outputs/runs/full_ensemble \
     --enable_bertscore \
     --output_dir results/esl_eval
   ```

2. **Analyze fairness metrics**:
   - FPR gap: FPR(ESL) - FPR(native) per detector
   - Target: Reduce gap from 0.15 to <0.07
   - BERTScore by group (ESL vs native)

3. **Generate visualizations**:
   ```bash
   python scripts/visualize_stealthbench.py \
     --results results/esl_eval \
     --output-dir outputs/figures
   ```

**Who should do this**: Team member with data analysis experience  
**Prerequisites**: ESL-stratified dataset + trained model  
**Deliverables**: Fairness report + heatmap visualizations

---

#### Priority 5: Results & Paper Writing (LOW - Week 4)
**What needs to be done:**
1. Compile all experimental results
2. Generate publication-ready figures
3. Write results section for paper/report
4. Compare against SICO baseline (if time permits)

**Who should do this**: All team members  
**Prerequisites**: All experiments completed  

---

### 📋 Task Assignment Recommendations

**Week 1:**
- **Person A**: Set up real detectors + test locally
- **Person B**: Curate ESL/native datasets
- **Person C**: Review documentation + set up environment

**Week 2:**
- **Person A**: Run main training experiments
- **Person B**: Prepare ESL evaluation pipeline
- **Person C**: Monitor training + debug issues

**Week 3:**
- **Person A**: Run ablation studies
- **Person B**: Run ESL fairness evaluation
- **Person C**: Generate visualizations

**Week 4:**
- **All**: Results analysis + paper writing

---

## Quickstart

### Run a Trained StealthRL Paraphraser

If you have a trained LoRA adapter, you can paraphrase text with:

```bash
python examples/paraphrase_example.py \
    --input "The quick brown fox jumps over the lazy dog." \
    --model_path checkpoints/stealthrl-lora \
    --output_path outputs/stealthrl_samples.jsonl
```

### Minimal Python Usage

```python
from stealthrl.models import load_stealthrl_model

model, tokenizer = load_stealthrl_model(
    base_model="Qwen/Qwen2.5-1.5B-Instruct",
    lora_path="checkpoints/stealthrl-lora"
)

input_text = "AI-generated text often exhibits certain statistical patterns."
paraphrased = model.paraphrase(input_text)
print(paraphrased)
```

### Compare Detector Scores

```bash
python examples/compare_detectors.py \
    --original_text "Your input text here..." \
    --paraphrased_text "Paraphrased version..." \
    --detectors fast-detectgpt ghostbuster binoculars
```

---

## Training StealthRL

Training is built on **HuggingFace TRL** with GRPO/PPO and LoRA adapters.

### Training Configuration

The reward function combines multiple terms:

| Term | Description |
|------|-------------|
| **Detector Ensemble** | Normalized scores from multiple detectors (lower = less detectable) |
| **Semantic Fidelity** | BERTScore / cosine similarity vs original (higher = better meaning preservation) |
| **Quality** | Perplexity bands, readability scores (constrain fluency) |
| **Fairness** | Penalty if ESL vs native FPR gap is large |

### Run Training

```bash
python scripts/train_stealthrl.py --config configs/stealthrl_small.yaml
```

### Example Config (`configs/stealthrl_small.yaml`)

```yaml
model:
  base_model: "Qwen/Qwen2.5-1.5B-Instruct"
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05

training:
  algorithm: "grpo"  # or "ppo"
  learning_rate: 1e-5
  batch_size: 8
  gradient_accumulation_steps: 4
  max_steps: 10000
  warmup_steps: 500

reward:
  detectors:
    - "fast-detectgpt"
    - "roberta-base-openai-detector"
  detector_weight: 0.4
  semantic_weight: 0.3
  quality_weight: 0.2
  fairness_weight: 0.1

data:
  train_dataset: "data/train.jsonl"
  eval_dataset: "data/eval.jsonl"
```

---

## Evaluation & StealthBench

**StealthBench** is a unified evaluation harness designed for reproducible, standardized comparisons.

### Features

- Runs multiple detectors on common text sets (before/after paraphrasing)
- Computes AUROC, FPR@0.5%, FPR@1%, and other metrics
- Outputs standardized CSVs and comparison plots
- Easily extensible: plug in new detectors and datasets

### Run StealthBench

```bash
python scripts/run_stealthbench.py --config configs/stealthbench.yaml
```

### Example Output

```
┌─────────────────┬─────────┬──────────┬──────────┬─────────────┐
│ Detector        │ AUROC   │ FPR@0.5% │ FPR@1%   │ ESL FPR Gap │
├─────────────────┼─────────┼──────────┼──────────┼─────────────┤
│ Fast-DetectGPT  │ 0.72    │ 0.08     │ 0.15     │ -0.03       │
│ Ghostbuster     │ 0.68    │ 0.12     │ 0.21     │ -0.05       │
│ Binoculars      │ 0.75    │ 0.06     │ 0.11     │ -0.02       │
└─────────────────┴─────────┴──────────┴──────────┴─────────────┘
```

Results are saved to `outputs/stealthbench_results/`.

### ESL/Native Fairness Evaluation

StealthRL includes specialized infrastructure for evaluating fairness across ESL (English as Second Language) and native writers:

**Features**:
- Unified corpus loader for TOEFL11, ICNALE, ELLIPSE, and native academic writing
- Stratified test splits with ~40% ESL / 60% native ratio
- Grouped metrics by ESL status (ASR, detector probabilities, semantic similarity)
- BERTScore F1 in addition to E5 cosine similarity for semantic fidelity
- Per-sample detailed logging for analysis

**Data Preparation**:
```bash
# Prepare ESL/native evaluation splits (requires preprocessed JSONL files)
python -m stealthrl.data.esl_native_corpus
```

Expected directory structure:
```
data/
├── esl/
│   ├── toefl11.jsonl          # ESL essays from TOEFL11 corpus
│   ├── icnale_written.jsonl   # ESL academic writing (ICNALE)
│   └── ellipse.jsonl          # ESL formative writing (ELLIPSE)
├── native/
│   └── native_academic.jsonl  # Native English academic writing
└── processed/
    ├── esl_native_dev.jsonl   # Dev split (auto-generated)
    └── esl_native_test.jsonl  # Test split (auto-generated)
```

**Run ESL Fairness Evaluation**:
```bash
python scripts/run_esl_eval.py \
    --eval_data data/processed/esl_native_test.jsonl \
    --stealthrl_model outputs/stealthrl_policy \
    --enable_bertscore \
    --bertscore_model roberta-large \
    --output_dir results/esl_native_eval
```

**Outputs**:
- `comparison_report.json` - Overall metrics across all models
- `esl_native_grouped_metrics.json` - Metrics by ESL status (overall, esl, native)
- `bertscore_results.json` - BERTScore F1 by model and group
- `bertscore_esl_native.jsonl` - Per-sample detailed results

**BERTScore for Semantic Similarity**:

In addition to E5 cosine similarity, StealthRL supports **BERTScore** for token-level semantic alignment:

```bash
# Install BERTScore
pip install bert-score

# Enable in config
# configs/tinker_stealthrl.yaml:
#   reward:
#     bertscore:
#       enabled: true
#       model_type: "roberta-large"  # or "microsoft/deberta-base" for faster eval
#       batch_size: 16
```

BERTScore provides complementary semantic similarity metrics:
- **E5 cosine**: Sentence-level embedding similarity (fast, 0-1 scale)
- **BERTScore F1**: Token-level BERT alignment (slower, more granular, 0-1 scale)

For large-scale evaluation, we recommend `microsoft/deberta-base` (2-3x faster than `roberta-large`).

---

````## Datasets

### Supported Dataset Sources

| Dataset | Purpose | Source |
|---------|---------|--------|
| **DetectRL** | Real-world detection benchmark | [GitHub](https://github.com/NLP2CT/DetectRL) |
| **ai-detection-paraphrases** | Paraphrase evasion benchmark | [GitHub](https://github.com/martiansideofthemoon/ai-detection-paraphrases) |
| **Ghostbuster data** | Human vs AI text pairs | [GitHub](https://github.com/vivek3141/ghostbuster) |
| **ChatGPT-Detector-Bias** | ESL vs native writing for fairness | [GitHub](https://github.com/Weixin-Liang/ChatGPT-Detector-Bias) |
| **Human Detectors** | Human judgment alignment data | [GitHub](https://github.com/jenna-russell/human_detectors) |

### Downloading Datasets

```bash
# Download datasets from original sources
bash scripts/download_datasets.sh
```

**Note**: Large datasets are **not** stored in this repository. The download script fetches them from original sources with proper attribution. Please respect the original licenses.

---

## Fairness & Responsible Use

> **This project is for research and evaluation purposes only.**

### Research Intent

StealthRL is designed to study and improve the robustness and fairness of AI text detectors. It is **not** intended to help users cheat, bypass academic integrity systems, or evade legitimate content moderation.

### What We Release

- ✅ Evaluation harness code (StealthBench)
- ✅ Training configurations and scripts
- ✅ Aggregate experimental results
- ❌ **Evasion-tuned model weights are NOT released**

### Intended Use Cases

- Studying detector vulnerabilities to improve robustness
- Measuring and mitigating ESL vs native bias in detectors
- Benchmarking new detection methods against adversarial paraphrasing
- Academic research on AI-generated text detection

### Ethical Considerations

We encourage researchers to use StealthBench to:
- Identify and document bias in existing detectors
- Develop more robust and fair detection methods
- Advance understanding of the detector-evader arms race

We discourage any use that would:
- Facilitate academic dishonesty
- Undermine legitimate content moderation
- Cause harm to individuals or institutions

---

## Limitations & Future Work

### Current Limitations

- **Detector Coverage**: Evaluations use a limited set of detectors; results may not generalize to all detection methods.
- **Fairness Scope**: ESL vs native English is one dimension of fairness; other dimensions (disability, neurodiversity, dialect variation) are not yet addressed.
- **Overfitting Risk**: Even with LoRA, models may overfit to in-ensemble detectors rather than learning truly general strategies.
- **Domain Specificity**: Training and evaluation focus on certain text domains (e.g., essays, news); transfer to other domains is untested.

### Future Directions

- Expand detector ensemble to include more diverse detection families
- Investigate multi-dimensional fairness metrics
- Explore defender-side analysis: which detector mixtures are most robust?
- Full fine-tuning ablations to probe capacity vs generalization tradeoffs

---

## Citation

If you use StealthRL or StealthBench in your research, please cite:

```bibtex
@misc{stealthrl2025,
  title={StealthRL: Ensemble-Guided Text Transformation for Multi-Detector Transfer and Fair Detection Robustness},
  author={Ranganath, Suraj and Mahor, Nishchay and Zhu, Sibo},
  year={2025},
  howpublished={\url{https://github.com/your-org/stealthrl}},
  note={University of California, San Diego - DSC 291: Safety in Generative AI}
}
```

---

## References & Prior Work

### Primary Reference

- **AuthorMist** - Reinforcement learning to evade AI detectors  
  Paper: `https://arxiv.org/abs/2503.08716` | PDF: `https://arxiv.org/pdf/2503.08716`

### Core Libraries

- **HuggingFace TRL** - Transformer Reinforcement Learning (SFT, GRPO, DPO, PPO)  
  Docs: `https://huggingface.co/docs/trl` | GitHub: `https://github.com/huggingface/trl`

### Detection Methods

- **DetectGPT** - Zero-shot machine-generated text detection using probability curvature  
  GitHub: `https://github.com/eric-mitchell/detect-gpt`

- **Fast-DetectGPT** - Efficient curvature-based detection  
  GitHub: `https://github.com/baoguangsheng/fast-detect-gpt`

- **Ghostbuster** - Feature-ensemble AI-text detector (NAACL 2024)  
  Paper: `https://aclanthology.org/2024.naacl-long.95/` | GitHub: `https://github.com/vivek3141/ghostbuster`

- **Binoculars** - Paired-LM zero-shot detection (ICML 2024)  
  Paper: `https://arxiv.org/abs/2401.12070` | GitHub: `https://github.com/ahans30/Binoculars`

### Evasion Methods & Benchmarks

- **DIPPER / ai-detection-paraphrases** - Paraphrase-based evasion benchmark (NeurIPS 2023)  
  GitHub: `https://github.com/martiansideofthemoon/ai-detection-paraphrases`

- **SICO** - Substitution-based In-Context Optimization for evading detectors  
  GitHub: `https://github.com/ColinLu50/Evade-GPT-Detector`

- **DetectRL** - Real-world detection benchmark  
  Paper: `https://arxiv.org/abs/2410.23746` | GitHub: `https://github.com/NLP2CT/DetectRL`

### Fairness & Bias

- **ChatGPT-Detector-Bias** - ESL vs native bias analysis and datasets  
  Paper: `https://pmc.ncbi.nlm.nih.gov/articles/PMC10382961/` | GitHub: `https://github.com/Weixin-Liang/ChatGPT-Detector-Bias`

### Human Evaluation

- **Human Detectors** - Comparing human vs automated detection  
  Paper: `https://arxiv.org/abs/2501.15654` | GitHub: `https://github.com/jenna-russell/human_detectors`

### Surveys & Resources

- **LLM-generated-Text-Detection** - Survey and curated resources  
  GitHub: `https://github.com/NLP2CT/LLM-generated-Text-Detection`

- **Awesome LLM-generated Text Detection** - Curated list  
  GitHub: `https://github.com/datamllab/awsome-LLM-generated-text-detection`

- **Awesome Machine-Generated Text** - Comprehensive resource list  
  GitHub: `https://github.com/ICTMCG/Awesome-Machine-Generated-Text`

---

## 🐛 Troubleshooting

### Environment Setup Issues

**Problem**: `pip install -r requirements.txt` fails
- **Solution 1**: Upgrade pip: `pip install --upgrade pip`
- **Solution 2**: Install with no cache: `pip install -r requirements.txt --no-cache-dir`
- **Solution 3**: Check Python version: `python --version` (must be 3.10+)

**Problem**: Virtual environment not activating
- **macOS/Linux**: Use `source venv/bin/activate` (not just `venv/bin/activate`)
- **Windows**: Use `venv\Scripts\activate`
- **Check**: Terminal prompt should show `(venv)` prefix

**Problem**: Import errors after installation
- **Solution**: Make sure venv is activated
- **Verify**: `which python` should point to `./venv/bin/python`
- **Reinstall**: Deactivate venv, delete `venv/` folder, recreate from scratch

### Tinker API Issues

**Problem**: "Invalid API key" error
- **Check**: Key starts with `tk-` and has no extra spaces
- **Verify**: `grep TINKER_API_KEY .env` shows correct key
- **Test**: Try logging into Tinker dashboard with same credentials

**Problem**: Training hangs or times out
- **Check**: Tinker credits available (DSC 291 students should have sponsored credits)
- **Monitor**: Check Tinker dashboard for active jobs
- **Retry**: Sometimes network issues cause hangs, restart training

### Training Issues

**Problem**: "All-negative groups" fraction is very high (>0.8)
- **Cause**: Reward function returning negative values for all attempts
- **Solution**: Increase `all_negative_min_reward` in config:
  ```yaml
  all_negative:
    min_reward: 0.05  # Increase from 0.01
    downweight: 0.3   # Reduce from 0.5
  ```

**Problem**: KL divergence too high (>0.05)
- **Solution**: Increase KL penalty coefficient:
  ```yaml
  kl:
    penalty_coef: 0.01  # Increase from 0.001
  ```

**Problem**: Semantic similarity too low (<0.80)
- **Solution**: Increase semantic weight:
  ```yaml
  reward:
    semantic_weight: 2.0  # Increase from 1.0
  ```

### Data Issues

**Problem**: "File not found" when running training
- **Check**: Data directory exists: `ls data/tinker/`
- **Verify**: JSONL files present: `ls data/tinker/*.jsonl`
- **Regenerate**: Run `python scripts/prepare_tinker_data.py --synthetic ...` again

**Problem**: JSONL format errors
- **Validate**: Each line must be valid JSON with required fields
- **Required fields**: `ai_text`, `human_reference`, `domain`, `is_esl`
- **Check**: `head -1 data/tinker/train.jsonl | jq` should parse successfully

### Detector Issues

**Problem**: Mock detectors returning same scores
- **Expected**: This is normal! Mock detectors use deterministic formulas
- **Solution**: Implement real detectors (see [`knowledge_base/DETECTOR_SETUP.md`](knowledge_base/DETECTOR_SETUP.md))

**Problem**: Real detector out of memory
- **Solution 1**: Reduce batch size in detector config
- **Solution 2**: Use CPU instead of GPU for detectors
- **Solution 3**: Use smaller detector models (e.g., RoBERTa-base instead of large)

### Getting Help

1. **Check logs**: `tail -50 outputs/runs/<run_name>/training.log`
2. **Review documentation**: See [`knowledge_base/`](knowledge_base/) for detailed guides
3. **Search interaction records**: [`interaction_records.md`](interaction_records.md) has detailed implementation history
4. **Ask team**: Post in team Slack/Discord with:
   - Error message (full traceback)
   - Command you ran
   - What you've tried already

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgments

This project was developed as part of **DSC 291: Safety in Generative AI** at the University of California, San Diego.

We thank the authors of AuthorMist, DetectGPT, Ghostbuster, Binoculars, and other foundational works that made this research possible.

---

**Questions or feedback?** Open an issue or reach out to the maintainers.
