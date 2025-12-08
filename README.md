# StealthRL

**Ensemble-Guided Text Transformation for Multi-Detector Transfer and Fair Detection Robustness**

> RL-trained paraphraser for multi-detector-robust, fair text generation.

🚀 **NEW: Tinker Integration** - Now supports training on remote compute with Qwen3-4B via the [Tinker platform](https://tinker-docs.thinkingmachines.ai/). See [TINKER_README.md](knowledge_base/TINKER_README.md) for quickstart and DSC 291 deployment guide.

---

## Overview

StealthRL is a research framework that uses **Reinforcement Learning with Verifiable Rewards (RFT)** to train a single, ensemble-guided paraphraser capable of reducing AI-detector scores while preserving semantic meaning and text quality. Unlike prior approaches that train separate models per detector (e.g., AuthorMist), StealthRL investigates whether **joint training against a detector ensemble** can learn detector-agnostic transformation strategies that generalize to unseen detector families.

A core focus of this project is **fairness**: AI text detectors have been shown to produce elevated false-positive rates on writing by ESL (English as a Second Language) authors. StealthRL explicitly monitors and optimizes for shrinking the ESL vs native FPR gap, treating fairness as a first-class objective rather than an afterthought.

### Current Status: Ultra-Fast Proof-of-Concept Complete ✅

**Latest Run** (December 7, 2025): Completed successful ultra-fast training as proof-of-concept:
- **Configuration**: `configs/tinker_stealthrl_ultrafast.yaml`
- **Training Time**: ~2 hours (50 steps)
- **Dataset**: 800 randomly sampled training examples, 1 epoch
- **Key Hyperparameters**:
  - Model: Qwen/Qwen3-4B-Instruct-2507 with LoRA rank 16
  - Learning rate: 5e-5 (LoRA RL optimized)
  - Batch size: 16, Group size: 8 (GRPO)
  - Temperature: 0.8 (constant)
  - KL penalty: 0.03 (adaptive target 4.0)
  - LR scheduler: Cosine with 10% warmup
  - Detector: Fast-DetectGPT only (speed)
  - Semantic: E5-small-v2 (3x faster)
- **Results**:
  - ✅ No model collapse (parse success 85.9% → 99.2%)
  - ✅ 22% detector evasion improvement (best checkpoint: 45.8% detection probability vs 58.7% baseline)
  - ✅ Quality preserved (98.6% semantic similarity maintained)
  - ✅ Stable KL divergence (<0.4, target <4.0)
  - ✅ 9 Pareto-optimal checkpoints identified (2D trade-off)
  - ✅ 25 Pareto-optimal checkpoints identified (3D trade-off)
- **Visualizations**: Comprehensive suite generated including training curves, Pareto frontiers, reward decomposition, stability metrics
- **Output**: `outputs/tinker_ultrafast/run_20251207_212110/`

**Next**: Full production training run with:
- **Dataset**: 20,000+ samples from DetectRL, ChatGPT-Bias, Ghostbuster datasets
- **ESL Split**: 40% ESL (TOEFL11, ICNALE, ELLIPSE) / 60% native academic
- **Epochs**: 3 (vs 1 in ultrafast)
- **Detectors**: Full ensemble (Fast-DetectGPT, Ghostbuster, Binoculars)
- **Duration**: ~6-8 hours estimated
- **Configuration**: See `knowledge_base/FINAL_RUN_HYPERPARAMETERS.md` for optimized settings

### Training Platform

**For this project (DSC 291), all RL training is conducted on the [Tinker platform](https://tinker.thinkingmachines.ai/)** using remote compute with Qwen3-4B and GRPO. This provides:
- 🖥️ **Remote GPU access** - No local GPU required
- ⚡ **Optimized for RL** - GRPO algorithm with LoRA fine-tuning
- 💰 **Sponsored compute credits** - Provided through DSC 291 course
- 📊 **Built-in monitoring** - TensorBoard integration and checkpoint management

See [TINKER_README.md](knowledge_base/TINKER_README.md) for the complete Tinker quickstart guide.

> **Note**: The original codebase supported local training with HuggingFace TRL, but for reproducibility and resource efficiency in this course, we standardized on Tinker.

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
StealthRL/
├── stealthrl/          # Core package
│   ├── models/          # Base LM loading, LoRA adapter utilities
│   ├── rewards/         # Composite reward computation (detectors, BERTScore, PPL, fairness)
│   ├── detectors/       # Wrappers for Fast-DetectGPT, Ghostbuster, Binoculars, etc.
│   ├── training/        # RL training loops (GRPO/PPO via HuggingFace TRL)
│   ├── evaluation/      # StealthBench metrics: AUROC, FPR, BERTScore, perplexity
│   ├── data/            # Data loading utilities (esl_native_corpus, etc.)
│   ├── baselines/       # SICO and other baseline methods
│   ├── metrics/         # BERTScore and other metrics
│   └── tinker/          # Tinker platform integration (env, dataset, reward, training)
│
├── scripts/             # Execution scripts
│   ├── prepare_data.py           # Prepare human/LLM text, ESL vs native subsets
│   ├── prepare_tinker_data.py    # Prepare Tinker-format training data
│   ├── train_stealthrl.py        # Main RL training entry point
│   ├── train_ultrafast.py        # Ultra-fast training script (1-2 hour runs)
│   ├── evaluate_detectors.py     # Run detector ensemble, produce CSVs
│   ├── run_stealthbench.py       # Unified evaluation harness
│   ├── run_esl_eval.py           # ESL fairness evaluation
│   ├── evaluate_transfer.py      # Transfer evaluation script
│   ├── compare_baselines.py      # Baseline comparison
│   ├── visualize_training_results.py  # Comprehensive visualization suite (NEW)
│   ├── visualize_stealthbench.py # StealthBench visualization
│   ├── test_detectors*.py        # Detector testing utilities
│   ├── convert_chatgpt_bias*.py  # Data conversion tools
│   ├── validate_datasets.py      # Data validation
│   ├── download_datasets.sh      # Download datasets from original sources
│   ├── download_esl_datasets.sh  # Download ESL datasets
│   └── run_research_pipeline.py  # Automated all-in-one runner
│
├── configs/             # YAML/JSON configs for models, training, detectors
│   ├── stealthbench.yaml
│   ├── stealthrl_small.yaml
│   ├── tinker_stealthrl.yaml              # Full ensemble config
│   ├── tinker_stealthrl_ultrafast.yaml    # Ultra-fast config (1-2 hrs, proof-of-concept)
│   ├── tinker_transfer_in_ensemble.yaml   # Transfer learning config
│   └── ablations/       # Ablation study configurations
│       ├── detector_only.yaml
│       ├── no_fairness.yaml
│       ├── no_quality.yaml
│       ├── no_semantic.yaml
│       └── single_detector_fast_detectgpt.yaml
│
├── data/                # Data directory
│   ├── raw/             # Original downloaded datasets
│   ├── processed/       # Processed training/test splits
│   ├── esl/             # ESL corpora (TOEFL11, ICNALE, ELLIPSE)
│   ├── native/          # Native English writing
│   ├── tinker_large/    # Full curated dataset (4,625 train, 1,157 test)
│   └── README.md
│
├── knowledge_base/      # Comprehensive documentation
│   ├── README.md        # Documentation index
│   ├── QUICKSTART.md    # Fast-track guide
│   ├── SETUP_AND_RUN.md # Complete setup guide
│   ├── TINKER_README.md # Tinker platform integration
│   ├── ULTRAFAST_TRAINING_GUIDE.md  # Ultra-fast training guide (NEW)
│   ├── FINAL_RUN_HYPERPARAMETERS.md # Optimized hyperparameters for production run
│   ├── DETECTOR_SETUP.md            # Detector implementation guide
│   ├── ESL_FAIRNESS_GUIDE.md        # ESL evaluation guide
│   ├── CHECKPOINT_GUIDE.md          # Checkpoint management
│   ├── RESEARCH_ROADMAP.md          # Research plan
│   ├── PRESENTATION_GUIDE.md        # Presentation outline & future work (NEW)
│   ├── task1/           # Task 1: Detector implementation docs
│   └── task2/           # Task 2: Dataset curation docs
│
├── outputs/             # Training outputs
│   ├── runs/            # Training run directories
│   ├── tinker_ultrafast/         # Ultra-fast run (proof-of-concept)
│   │   └── run_20251207_212110/  # Latest ultrafast run (50 steps, ~2 hrs)
│   │       ├── metrics.jsonl
│   │       ├── training.log
│   │       ├── tensorboard/
│   │       ├── checkpoints/
│   │       ├── train_iteration_*.html  # 50 iteration reports
│   │       └── visualizations/         # Comprehensive visualizations (NEW)
│   │           ├── training_curves.png/pdf
│   │           ├── pareto_frontiers.png/pdf
│   │           ├── reward_decomposition.png/pdf
│   │           ├── stability_metrics.png/pdf
│   │           └── training_summary.csv/txt
│   └── tinker_full_ensemble/     # Full ensemble runs (planned)
│
├── examples/            # Sample scripts and notebooks
├── requirements.txt     # Python dependencies
├── environment.yml      # Conda environment (optional)
├── README.md            # This file
├── REPORT.md            # Comprehensive project report
├── PRESENTATION_GUIDE.md # Presentation outline & future work (NEW)
├── interaction_records.md # Development history log
└── LICENSE
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

**Note on GPU Requirements:**
- **RL Training**: Runs on Tinker's remote GPUs (no local GPU needed)
- **Local Detector Testing**: Optional, requires NVIDIA GPU with 8-16GB VRAM
- For this project, all training happens on Tinker - local GPU is only useful for testing detector implementations

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

## 📊 Project Status: Ultra-Fast Proof-of-Concept Complete

### ✅ COMPLETED (Ready to Use)

#### Ultra-Fast Training Run (December 7, 2025)
- ✅ **Successful 50-step training** (~2 hours)
  - Model: Qwen3-4B + LoRA rank 16
  - Dataset: 800 samples (randomly sampled), 1 epoch
  - Configuration: `configs/tinker_stealthrl_ultrafast.yaml`
  - No model collapse (parse success 85.9% → 99.2%)
  - Stable KL divergence (<0.4, target <4.0)
  
- ✅ **RL Best Practices Validated**
  - Learning rate: 5e-5 (LoRA RL optimized)
  - Batch size: 16, Group size: 8 (GRPO)
  - LR scheduler: Cosine with 10% warmup
  - KL penalty: Adaptive (0.03, target 4.0)
  - Temperature: 0.8 (constant for RL stability)
  
- ✅ **Strong Results Achieved**
  - 22% detector evasion improvement (45.8% detection prob vs 58.7% baseline)
  - 98.6% semantic similarity maintained
  - Perplexity 30.1 (near-perfect target of 30)
  - 9 Pareto-optimal checkpoints (2D: stealth×quality)
  - 25 Pareto-optimal checkpoints (3D: stealth×quality×naturalness)

- ✅ **Comprehensive Visualization Suite** (`scripts/visualize_training_results.py`)
  - Training curves (6 subplots): rewards, detector, semantic, perplexity, KL, parse success
  - Pareto frontier analysis: 2D and 3D trade-off plots with optimal points highlighted
  - Reward decomposition: stacked area, trajectories, detector histogram, correlation heatmap
  - Stability metrics: entropy, LR schedule, token stats, timing
  - Summary statistics: CSV/TXT with initial/final/best/mean metrics
  - All plots in publication-quality PNG (300 DPI) + PDF

- ✅ **Presentation Materials Ready**
  - Comprehensive presentation guide (`PRESENTATION_GUIDE.md`)
  - 13 detailed future extension ideas
  - Demo plan, Q&A preparation, backup slides
  - All visualizations ready in `outputs/tinker_ultrafast/run_20251207_212110/visualizations/`

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
- ✅ **Task 1 completion** (see `knowledge_base/task1/` for docs, `scripts/test_detectors*.py` for tests)
- ✅ **Task 2 setup** (see `knowledge_base/task2/` for docs, `scripts/` for conversion scripts)

### 🔨 NEXT: Full Production Training Run

#### Configuration Details
**Dataset** (data/tinker_large/):
- **Size**: 20,000+ samples (4,625 train, 1,157 test currently; expandable)
- **Sources**: DetectRL, ChatGPT-Detector-Bias, Ghostbuster datasets
- **ESL Split**: 40% ESL (TOEFL11, ICNALE, ELLIPSE) / 60% native academic writing
- **Domains**: Academic essays, news articles, creative writing

**Training Hyperparameters** (see `knowledge_base/FINAL_RUN_HYPERPARAMETERS.md`):
- **Model**: Qwen/Qwen3-4B-Instruct-2507
- **LoRA**: rank 32 (optimal for RL), alpha 32, all layers including MLP
- **Learning Rate**: 2.8e-4 (10x FullFT rule for LoRA)
- **Batch Size**: 4 (LoRA optimal), Group Size: 8 (GRPO)
- **Epochs**: 3 (vs 1 in ultrafast)
- **Detectors**: Full ensemble (Fast-DetectGPT, Ghostbuster, Binoculars)
- **Semantic Model**: E5-large (vs e5-small-v2 in ultrafast)
- **Temperature**: 1.0 constant (no decay for RL exploration)
- **KL Penalty**: 0.01 with fixed coefficient
- **Advantage Clip**: 10.0 (increased from ultrafast 5.0)

**Expected Outcomes**:
- ASR (Attack Success Rate): 60-70% across ensemble
- Semantic similarity: >88%
- ESL FPR gap: <0.07 (50-80% reduction from baseline ~0.15)
- Training time: 6-8 hours (Tinker hosted)
- Checkpoints: Every 500 steps + final

**Research Questions** (to be answered):
1. **Transfer**: Does training on 2-detector ensemble (Fast-DetectGPT + Ghostbuster) transfer to held-out Binoculars?
   - Target transfer ratio: >0.7 (ASR_held_out / ASR_in_ensemble)
2. **Ablations**: Which reward components matter most? (5 experiments: detector-only, no-fairness, no-quality, no-semantic, single-detector)
3. **Fairness**: Can we reduce ESL vs native FPR gap by 50-80%?

**Next Steps**:
1. ⌛ Execute full training: `python scripts/train_stealthrl.py --config configs/tinker_stealthrl.yaml`
2. ⌛ Run transfer experiment: `python scripts/train_stealthrl.py --config configs/tinker_transfer_in_ensemble.yaml`
3. ⌛ Run ablation studies: `bash scripts/run_ablations.sh`
4. ⌛ Comprehensive evaluation: `python scripts/run_esl_eval.py`
5. ⌛ Generate final visualizations: `python scripts/visualize_training_results.py` + `scripts/visualize_stealthbench.py`

---

### 🔨 IN PROGRESS / TODO (Team Tasks)

#### ~~Priority 1: Detector Setup (HIGH - Week 1)~~ ✅ **COMPLETED**
**Status**: ✅ Real detectors implemented and tested

**What was completed:**
1. ✅ Installed detector dependencies (transformers, torch, sentence-transformers)
2. ✅ Implemented FastDetectGPT (GPT-2 based curvature detection)
3. ✅ Implemented Ghostbuster (RoBERTa classifier)
4. ✅ Implemented Binoculars (paired language models)
5. ✅ Implemented semantic similarity (E5 embeddings)
6. ✅ Implemented perplexity computation (GPT-2)
7. ✅ Tested all detectors successfully
8. ✅ Verified caching works

**Documentation**: See `knowledge_base/task1/` folder for complete details

**Quick test**:
```bash
python scripts/test_detectors_standalone.py
```

---

#### Priority 2: Dataset Curation (HIGH - Week 1-2)
**Status**: ✅ **SETUP COMPLETE** - Scripts ready, execution in progress

**What has been completed:**
1. ✅ Created dataset curation tooling (docs in `knowledge_base/task2/`)
2. ✅ Conversion script for ChatGPT-Detector-Bias data (`scripts/convert_chatgpt_bias_data.py`)
3. ✅ Validation script for data quality checks (`scripts/validate_datasets.py`)
4. ✅ Step-by-step execution guide (see `knowledge_base/task2/QUICK_START.md`)
5. ✅ Integration with existing data pipeline

**What needs to be executed:**
1. **Download ChatGPT-Detector-Bias dataset** (primary ESL/native source):
   ```bash
   bash scripts/download_datasets.sh
   ```

2. **Convert to JSONL format** using provided script:
   ```bash
   python scripts/convert_chatgpt_bias_data.py \
     --input data/raw/ChatGPT-Detector-Bias \
     --output-esl data/esl/toefl11.jsonl \
     --output-native data/native/native_academic.jsonl
   ```

3. **Validate and generate splits**:
   ```bash
   python scripts/validate_datasets.py \
     --esl-data data/esl/toefl11.jsonl \
     --native-data data/native/native_academic.jsonl

   python -m stealthrl.data.esl_native_corpus
   python scripts/prepare_tinker_data.py \
     --input-paths data/esl/toefl11.jsonl data/native/native_academic.jsonl \
     --output-dir data/tinker
   ```

**Documentation**:
- See [`knowledge_base/TASK2_README.md`](knowledge_base/TASK2_README.md) for overview
- See [`knowledge_base/task2/QUICK_START.md`](knowledge_base/task2/QUICK_START.md) for step-by-step guide
- See [`scripts/`](scripts/) folder for all conversion and validation tools

**Target**: 1000-2000 samples (40% ESL from TOEFL, 60% native academic)
**Estimated time**: 1-2 hours (mostly download time)

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
python scripts/paraphrase_example.py \
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

## 📚 Additional Resources

### For AI Agents & Builders

If you're building on this project or using AI agents to extend it, these resources are invaluable:

- **[Tinker Full Docs for LLMs](https://tinker-docs.thinkingmachines.ai/llms-full.txt)** - Complete API reference optimized for AI agents
- **[Tinker Cookbook for Agents](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/AGENTS.md)** - Best practices and patterns for building with Tinker
- **[LoRA with RL Best Practices](https://thinkingmachines.ai/blog/lora/)** - How to effectively combine LoRA with RL training
- **[GRPO RL Training Tips](https://github.com/zechenzhangAGI/AI-research-SKILLs/tree/main/06-post-training/grpo-rl-training)** - Practical tips for the GRPO algorithm we use

These were instrumental in building StealthRL's training pipeline and will help you understand:
- How to structure RL training jobs on Tinker
- Best practices for reward shaping and hyperparameter tuning
- Common pitfalls and how to avoid them
- How to debug training issues

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgments

This project was developed as part of **DSC 291: Safety in Generative AI** at the University of California, San Diego.

We thank the authors of AuthorMist, DetectGPT, Ghostbuster, Binoculars, and other foundational works that made this research possible.

---

**Questions or feedback?** Open an issue or reach out to the maintainers.
