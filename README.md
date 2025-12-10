# StealthRL

**Ensemble-Guided Text Transformation for Multi-Detector Transfer and Fair Detection Robustness**

> Reinforcement learning-based paraphraser for multi-detector-robust, fairness-aware adversarial text generation using GRPO (Group Relative Policy Optimization).

🎓 **Course**: DSC 291 Safety in Generative AI, UC San Diego  
🖥️ **Platform**: [Tinker](https://tinker.thinkingmachines.ai/) remote compute with Qwen3-4B + LoRA  
📊 **Algorithm**: GRPO (Group Relative Policy Optimization)  
📖 **Documentation**: [`knowledge_base/`](knowledge_base/) | [`report/report.pdf`](report/report.pdf)

---

## Architecture

![StealthRL Pipeline Overview](report/StealthRL_Methodology.png)

*Complete experimental pipeline from data foundation through training, evaluation, transfer learning, ablations, baseline comparisons, and results analysis.*

The implementation is **modular**: detectors, reward terms, and base models can be swapped via configuration files without changing core training code.

---

## Table of Contents

- [Architecture](#architecture)
- [Overview](#overview)
  - [Research Questions](#research-questions)
  - [Novel Contributions](#novel-contributions)
- [Project Status](#project-status-ultra-fast-proof-of-concept-complete-)
  - [Latest Training Run](#latest-training-run-december-7-2025)
  - [Next: Full Production Training](#next-full-production-training)
- [Motivation](#motivation)
- [Key Features](#key-features)
  - [Training System](#training-system)
  - [Multi-Detector Ensemble](#multi-detector-ensemble)
  - [Quality Controls](#quality-controls)
  - [Evaluation Framework](#evaluation-framework)
  - [Outputs & Visualization](#outputs--visualization)
- [Repository Structure](#repository-structure)
- [Getting Started](#-getting-started-for-new-team-members)
  - [Prerequisites](#prerequisites)
  - [Step-by-Step Setup](#step-by-step-setup)
  - [Core Dependencies](#core-dependencies)
- [How to Run StealthRL](#-how-to-run-stealthrl)
  - [1. Training](#1-training)
  - [2. Evaluation](#2-evaluation)
  - [3. Visualization](#3-visualization)
  - [4. Using Trained Models for Paraphrasing](#4-using-trained-models-for-paraphrasing)
  - [5. Running Ablation Studies](#5-running-ablation-studies)
  - [6. Monitoring Training](#6-monitoring-training)
  - [Common Training Commands Summary](#common-training-commands-summary)
- [Configuration Files](#-configuration-files)
  - [Available Configurations](#available-configurations)
  - [Configuration Structure](#configuration-structure)
  - [How to Modify Configurations](#how-to-modify-configurations)
  - [Creating Custom Configurations](#creating-custom-configurations)
- [Output Files Generated](#-output-files-generated)
  - [Directory Structure](#directory-structure)
  - [File Descriptions](#file-descriptions)
  - [Evaluation Output Files](#evaluation-output-files)
  - [Disk Space Requirements](#disk-space-requirements)
- [Datasets](#-datasets)
  - [Completed Datasets](#-completed-ready-to-use)
  - [Next: Full Production Training Run](#-next-full-production-training-run)
  - [In Progress / TODO](#-in-progress--todo-team-tasks)
  - [Task Assignment Recommendations](#-task-assignment-recommendations)
- [Quickstart](#quickstart)
  - [Run a Trained StealthRL Paraphraser](#run-a-trained-stealthrl-paraphraser)
  - [Minimal Python Usage](#minimal-python-usage)
  - [Compare Detector Scores](#compare-detector-scores)
- [Training StealthRL](#training-stealthrl)
  - [Training Configuration](#training-configuration)
  - [Run Training](#run-training)
  - [Example Config](#example-config-configsstealthrl_smallyaml)
- [Evaluation & StealthBench](#evaluation--stealthbench)
  - [Features](#features)
  - [Run StealthBench](#run-stealthbench)
  - [Example Output](#example-output)
  - [ESL/Native Fairness Evaluation](#eslnative-fairness-evaluation)
  - [Supported Dataset Sources](#supported-dataset-sources)
  - [Downloading Datasets](#downloading-datasets)
- [Fairness & Responsible Use](#fairness--responsible-use)
  - [Research Intent](#research-intent)
  - [What We Release](#what-we-release)
  - [Intended Use Cases](#intended-use-cases)
  - [Ethical Considerations](#ethical-considerations)
- [Limitations & Future Work](#limitations--future-work)
  - [Current Limitations](#current-limitations)
  - [Future Directions](#future-directions)
- [Citation](#citation)
- [References & Prior Work](#references--prior-work)
- [Troubleshooting](#-troubleshooting)
- [Additional Resources](#-additional-resources)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

StealthRL is a comprehensive research framework that uses **Group Relative Policy Optimization (GRPO)** to train paraphrasers that evade AI text detection while preserving semantic meaning, naturalness, and fairness. Unlike prior work that trains separate models per detector (AuthorMist) or uses simple rule-based approaches (DIPPER, SICO), StealthRL addresses three key research questions:

### Research Questions

1. **Multi-Detector Generalization**: Can joint training against a detector ensemble learn detector-agnostic strategies that transfer to unseen detector families?
   - Training on 2 detector families (Fast-DetectGPT + Ghostbuster)
   - Evaluating transfer to held-out family (Binoculars)
   - Target: Transfer ratio >0.7 (ASR_held_out / ASR_in_ensemble)

2. **Fairness-Aware Adversarial Training**: Can we reduce detector bias against ESL (English as Second Language) writers?
   - Explicit ESL fairness penalty in reward function
   - Target: Reduce FPR gap from ~0.15 to <0.07 (50-80% reduction)
   - 40% ESL / 60% native training split

3. **Multi-Objective Optimization**: How do stealth, quality, and naturalness trade off?
   - Pareto frontier analysis identifying optimal checkpoints
   - Multiple checkpoints for different use cases (high stealth vs high quality vs balanced)
   - 9+ Pareto-optimal models from single training run

### Novel Contributions

1. **Generalizable Multi-Detector Framework**: First system to train on local open-source detectors (vs AuthorMist's API-dependent approach), supporting 3 detector families
2. **Fairness-Aware Adversarial Training**: First explicit ESL penalty term in adversarial NLP, targeting FPR gap <0.07
3. **Open-Source Release**: Full codebase, configs, and 9 Pareto-optimal checkpoints (vs proprietary competitors)
4. **Plug-and-Play Training Harness**: Zero-code YAML configuration for custom models, detectors, and rewards
5. **Multi-Objective Pareto Optimization**: First application to adversarial text with automated optimal checkpoint identification
6. **First GRPO for Adversarial NLP**: Simpler than PPO (no value function), more stable than supervised learning

---

## Project Status: Ultra-Fast Proof-of-Concept Complete ✅

### Latest Training Run (December 7, 2025)

**Ultra-Fast Configuration** (`configs/tinker_stealthrl_ultrafast.yaml`):
- **Training Time**: ~3.5 hours (50 steps, 1 epoch)
- **Dataset**: 800 training samples, 150 test samples
- **Model**: Qwen/Qwen3-4B-Instruct-2507 with LoRA rank 16
- **Algorithm**: GRPO (Group Relative Policy Optimization)

**Key Hyperparameters**:
```yaml
Model: Qwen/Qwen3-4B-Instruct-2507
LoRA: rank 16, alpha 16, dropout 0.05
Learning Rate: 5e-5 (LoRA RL optimized)
LR Scheduler: Cosine with 10% warmup
Batch Size: 16 prompts
Group Size: 8 rollouts per prompt (GRPO)
Temperature: 0.8 constant
KL Penalty: 0.03 with adaptive target 4.0
Epochs: 1
Detectors: Fast-DetectGPT only (speed optimization)
Semantic: E5-small-v2 (3x faster than e5-large)
```

**Training Results**:
- ✅ **No Model Collapse**: Parse success improved from 85.9% → 99.2%
- ✅ **22% Detector Evasion Improvement**: Best checkpoint (step 22) achieved 45.8% detection probability vs 58.7% baseline
- ✅ **Quality Preserved**: 98.6% average semantic similarity (never dropped below 94%)
- ✅ **Stable Training**: KL divergence stayed <0.4 (target <4.0), peak 3.06 at step 22
- ✅ **Controlled Perplexity**: Final 30.1 (very close to target 30), range 23-86
- ✅ **9 Pareto-Optimal Checkpoints** (2D: stealth × quality trade-off)
- ✅ **26 Pareto-Optimal Checkpoints** (3D: stealth × quality × naturalness)

**Optimal Checkpoints by Use Case**:
| Use Case | Step | Detection Prob | Semantic Sim | Perplexity | Total Reward |
|----------|------|----------------|--------------|------------|--------------|
| **High Stealth** | 22 | 45.8% (best) | 94.4% | 85.8 | +2.51 |
| **High Quality** | 23 | 57.7% | 99.5% (best) | 62.4 | +0.82 |
| **Balanced** | 49 | 57.9% | 98.6% | 30.1 (best) | +0.85 |

**Output Location**: `outputs/tinker_ultrafast/run_20251207_212110/`

**Visualizations Generated**: See [`outputs/tinker_ultrafast/run_20251207_212110/visualizations/`](outputs/tinker_ultrafast/run_20251207_212110/visualizations/)
- `training_curves.png/pdf`: 6-panel training progression
- `pareto_frontiers.png/pdf`: 2D and 3D trade-off analysis
- `reward_decomposition.png/pdf`: Component analysis and correlations
- `stability_metrics.png/pdf`: Convergence and timing metrics
- `training_summary.csv/txt`: Complete statistics

**Key Training Insights**:
1. **Learning Rate is Critical**: Initial 2.8e-4 caused collapse, 5e-5 ensured stability
2. **Group Size Matters**: GRPO needs group_size ≥8 for variance; <4 leads to uniform rewards
3. **KL Monitoring**: Spike to 3.06 at step 22 shows exploration boundary (high stealth correlates with higher drift)
4. **Perplexity Trade-Off**: Pushing stealth boundaries (step 22) increases perplexity to 85.8, but balanced checkpoint (step 49) maintains 30.1

### Next: Full Production Training

**Configuration** (`configs/tinker_stealthrl.yaml`):
- **Dataset**: 20,000+ samples (vs 800), 3 epochs (vs 1)
- **ESL Split**: 40% ESL / 60% native (proper fairness evaluation)
- **Detectors**: Full 3-detector ensemble (Fast-DetectGPT + Ghostbuster + Binoculars)
- **LoRA**: rank 32 (vs 16), optimal for RL
- **Learning Rate**: 2.8e-4 (vs 5e-5), 10× FullFT rule for LoRA
- **Batch Size**: 4 (vs 16), optimal for LoRA
- **Semantic**: E5-large (vs e5-small-v2), better quality
- **Duration**: ~6-8 hours (vs 3.5 hours)

**Expected Improvements**:
- ASR: 60-70% (vs 22% current, 3× better)
- Semantic similarity: >88% maintained
- ESL FPR gap: <0.07 (50-80% reduction from baseline ~0.15)
- Transfer ratio: >0.7 to held-out detector

**Planned Experiments**:
1. ⌛ **Full ensemble training**: Train on 3 detectors with 20K samples, 3 epochs
2. ⌛ **Transfer evaluation**: Train on Fast-DetectGPT + Ghostbuster, test on held-out Binoculars
3. ⌛ **Ablation studies**: 5 experiments testing necessity of each reward component
4. ⌛ **ESL fairness evaluation**: Comprehensive analysis with 40/60 ESL/native split
5. ⌛ **Baseline comparisons**: Compare against DIPPER, SICO, Pegasus, BART

**Documentation**: See [`knowledge_base/FINAL_RUN_HYPERPARAMETERS.md`](knowledge_base/FINAL_RUN_HYPERPARAMETERS.md) for optimized settings and [`knowledge_base/PRESENTATION_GUIDE.md`](knowledge_base/PRESENTATION_GUIDE.md) for comprehensive results analysis and 20+ future research directions.

---

## Motivation

AI text detectors are increasingly deployed in academic integrity and content moderation, but suffer from critical issues:

1. **Brittleness**: Detectors fail to generalize across paraphrasing attacks; models trained to evade one detector don't transfer to others
2. **Bias**: Detectors disproportionately flag ESL writing as AI-generated (FPR gap ~0.15), raising fairness concerns
3. **Lack of Robustness Testing**: Few frameworks exist for systematically evaluating detector weaknesses

StealthRL extends prior work (AuthorMist) by:
- Training a **single model** against a **multi-detector ensemble** (3 detector families in one RL loop)
- Evaluating **transfer to held-out detectors** (train on Fast-DetectGPT + Ghostbuster, test on Binoculars)
- Incorporating **explicit fairness penalty** to reduce ESL vs native FPR gap by 50-80%
- Providing **Pareto-optimal checkpoints** for different stealth-quality-naturalness trade-offs

---

## Key Features

### Training System
- **GRPO Algorithm**: Group Relative Policy Optimization (simpler than PPO, no value function)
- **LoRA Fine-Tuning**: Parameter-efficient training with rank 16-32 adapters
- **Multi-Objective Reward**: 4 components with z-score normalization
  ```
  R_total = 1.0·R_detector + 1.0·R_semantic + 0.5·R_perplexity - 0.2·R_fairness
  ```
- **Adaptive KL Penalty**: Prevents model drift from base model (β=0.01-0.03)
- **Tinker Platform Integration**: Remote GPU training, no local GPU needed

### Multi-Detector Ensemble
Three detector families supported:
1. **Fast-DetectGPT** (Curvature-based): Log-probability curvature using GPT-2-medium
2. **Ghostbuster** (Classifier-based): RoBERTa with 100+ weak features
3. **Binoculars** (Paired-LM): Compares instruction-tuned vs base model probabilities

### Quality Controls
- **Semantic Similarity**: E5 embeddings (cosine >0.85) + BERTScore F1
- **Naturalness**: GPT-2 perplexity banding (target ~30, range 5-80)
- **Parse Success**: Monitors output format validity (target >95%)
- **Fairness Penalty**: Per-sample ESL penalty = `detector_prob × 𝟙[ESL]`

### Evaluation Framework
- **StealthBench**: Unified multi-detector evaluation harness
- **ESL Fairness Analysis**: Stratified metrics by ESL status
- **Pareto Frontier Identification**: Automated optimal checkpoint selection
- **Transfer Evaluation**: Train on 2 detectors, test on 3rd held-out
- **Ablation Studies**: 5 configs testing necessity of each reward component

### Outputs & Visualization
- **Comprehensive Training Visualizations**: 8 publication-quality plots (PNG 300 DPI + PDF)
  - Training curves (6 metrics over time)
  - Pareto frontiers (2D and 3D trade-off analysis)
  - Reward decomposition (stacked area, trajectories, correlations)
  - Stability metrics (entropy, LR schedule, token stats, timing)
- **Checkpoint Management**: Automatic saving every 100-500 steps
- **Detailed Logging**: metrics.jsonl, training.log, TensorBoard integration
- **Summary Statistics**: CSV/TXT with initial/final/best/mean/std for all metrics

---

## Repository Structure

```
StealthRL/
├── stealthrl/              # Core package (~6k lines of Python)
│   ├── __init__.py
│   ├── models/             # Base LM loading, LoRA adapter utilities
│   ├── rewards/            # Composite reward computation (detectors, BERTScore, PPL, fairness)
│   ├── detectors/          # Wrappers for Fast-DetectGPT, Ghostbuster, Binoculars
│   ├── training/           # RL training loops (GRPO via HuggingFace TRL)
│   ├── evaluation/         # StealthBench metrics: AUROC, FPR, BERTScore, perplexity
│   ├── data/               # Data loading utilities (esl_native_corpus, etc.)
│   ├── baselines/          # SICO and other baseline methods
│   ├── metrics/            # BERTScore and other metrics
│   └── tinker/             # Tinker platform integration (env, dataset, reward, training)
│
├── scripts/                # Execution scripts (~35 Python scripts, 2 shell scripts)
│   ├── prepare_data.py              # Prepare human/LLM text, ESL vs native subsets
│   ├── prepare_tinker_data.py       # Prepare Tinker-format training data
│   ├── train_stealthrl.py           # Main RL training entry point
│   ├── train_ultrafast.py           # Ultra-fast training script (3-4 hour runs)
│   ├── run_ultrafast_training.py    # Ultra-fast training launcher
│   ├── run_ultrafast_direct.sh      # Direct ultra-fast training shell script
│   ├── evaluate_detectors.py        # Run detector ensemble, produce CSVs
│   ├── run_stealthbench.py          # Unified evaluation harness
│   ├── run_esl_eval.py              # ESL fairness evaluation
│   ├── evaluate_transfer.py         # Transfer evaluation script
│   ├── evaluate_ablations.py        # Ablation study evaluation
│   ├── run_ablations.sh             # Run all ablation studies
│   ├── compare_baselines.py         # Baseline comparison
│   ├── compare_detectors.py         # Detector comparison utilities
│   ├── visualize_training_results.py # Comprehensive visualization suite
│   ├── visualize_stealthbench.py    # StealthBench visualization
│   ├── paraphrase_example.py        # Example paraphrasing script
│   ├── export_model.py              # Model export utilities
│   ├── monitor_training.py          # Training monitoring
│   ├── monitor_runs.py              # Run monitoring
│   ├── track_runs.py                # Run tracking
│   ├── test_detectors*.py           # Detector testing utilities (4 scripts)
│   ├── test_all_fixes.py            # Test all fixes
│   ├── test_semantic_fix.py         # Test semantic similarity fixes
│   ├── convert_chatgpt_bias*.py     # Data conversion tools (2 scripts)
│   ├── validate_datasets.py         # Data validation
│   ├── extract_all_datasets.py      # Dataset extraction
│   ├── extract_detectrl_data.py     # DetectRL dataset extraction
│   ├── download_datasets.sh         # Download datasets from original sources
│   ├── download_esl_datasets.sh     # Download ESL datasets
│   ├── migrate_old_runs.py          # Migrate old training runs
│   ├── cancel_tinker_runs.py        # Cancel Tinker runs
│   └── run_research_pipeline.py     # Automated all-in-one runner
│
├── configs/                # YAML configs for models, training, detectors
│   ├── stealthbench.yaml            # StealthBench evaluation config
│   ├── stealthrl_small.yaml         # Small-scale training config
│   ├── tinker_stealthrl.yaml        # Full ensemble config (production)
│   ├── tinker_stealthrl_ultrafast.yaml  # Ultra-fast config (3-4 hrs, proof-of-concept)
│   ├── tinker_transfer_in_ensemble.yaml # Transfer learning config
│   └── ablations/                   # Ablation study configurations (5 configs)
│       ├── detector_only.yaml
│       ├── no_fairness.yaml
│       ├── no_quality.yaml
│       ├── no_semantic.yaml
│       └── single_detector_fast_detectgpt.yaml
│
├── data/                   # Data directory
│   ├── README.md
│   ├── raw/                # Original downloaded datasets
│   ├── processed/          # Processed training/test splits
│   ├── esl/                # ESL corpora (TOEFL11, ICNALE, ELLIPSE)
│   ├── native/             # Native English writing (DetectRL)
│   └── tinker_large/       # Full curated dataset (20k train, 700 test)
│
├── knowledge_base/         # Comprehensive documentation (~25 markdown files)
│   ├── README.md           # Documentation index
│   ├── QUICKSTART.md       # Fast-track guide
│   ├── SETUP_AND_RUN.md    # Complete setup guide
│   ├── QUICK_START_RUNS.md # Quick start commands
│   ├── TINKER_README.md    # Tinker platform integration
│   ├── ULTRAFAST_TRAINING_GUIDE.md      # Ultra-fast training guide
│   ├── ULTRAFAST_VS_FULL_COMPARISON.md  # Ultra-fast vs full comparison
│   ├── FINAL_RUN_HYPERPARAMETERS.md     # Optimized hyperparameters
│   ├── DETECTOR_SETUP.md                # Detector implementation guide
│   ├── DETECTOR_FIXES_IMPLEMENTATION.md # Detector fixes
│   ├── ESL_FAIRNESS_GUIDE.md            # ESL evaluation guide
│   ├── CHECKPOINT_GUIDE.md              # Checkpoint management
│   ├── CHECKPOINT_IMPLEMENTATION.md     # Checkpoint implementation
│   ├── REWARD_REFINEMENT.md             # Reward function design
│   ├── RESEARCH_ROADMAP.md              # Research plan
│   ├── NEXT_STEPS.md                    # Next steps
│   ├── PRESENTATION_GUIDE.md            # Presentation outline
│   ├── TEAM_HANDOFF.md                  # Team handoff guide
│   ├── RUN_MANAGEMENT.md                # Run management
│   ├── IMPLEMENTATION_VERIFICATION.md   # Implementation verification
│   ├── REORGANIZATION_SUMMARY.md        # Code reorganization summary
│   ├── DATA_CURATION_ANALYSIS.md        # Data curation analysis
│   ├── DATA_DOWNLOAD_SUMMARY.md         # Data download summary
│   ├── TINKER_IMPORT_ISSUE_RESOLUTION.md # Tinker import fixes
│   ├── LATEX_FIGURES_README.md          # LaTeX figures guide
│   ├── task1/              # Task 1: Detector implementation docs
│   │   ├── README.md
│   │   └── TASK1_README.md
│   ├── task2/              # Task 2: Dataset curation docs
│   │   ├── README.md
│   │   └── TASK2_README.md
│   └── TASK3_*.md          # Task 3 documentation (pipeline readiness)
│
├── outputs/                # Training outputs
│   ├── runs/               # Training run directories
│   ├── tinker_ultrafast/   # Ultra-fast runs (proof-of-concept)
│   │   └── run_20251207_212110/  # Latest ultrafast run (50 steps, 3.5 hrs)
│   │       ├── metrics.jsonl
│   │       ├── training.log
│   │       ├── config.yaml
│   │       ├── tensorboard/
│   │       ├── checkpoints/        # 50 GRPO checkpoints
│   │       ├── train_iteration_*.html  # 50 iteration reports
│   │       └── visualizations/     # Comprehensive visualizations
│   │           ├── training_curves.png/pdf
│   │           ├── pareto_frontiers.png/pdf
│   │           ├── reward_decomposition.png/pdf
│   │           ├── stability_metrics.png/pdf
│   │           ├── stealthrl_pipeline.png
│   │           └── training_summary.csv/txt
│   ├── tinker_full_ensemble/          # Full ensemble runs (planned)
│   │   └── run_20251207_165448/
│   └── tinker_full_ensemble_unoptimised/  # Unoptimized full runs
│
├── report/                 # Academic paper and figures
│   ├── report.tex          # LaTeX source (16+ pages)
│   ├── report.pdf          # Compiled PDF (~8 MB)
│   ├── REPORT.md           # Markdown version
│   ├── StealthRL_Methodology.png     # Pipeline overview diagram
│   └── StealthRL Pipeline.jpg        # Training loop diagram
│
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment (optional)
├── .env.example            # Example environment variables
├── .gitignore
├── LICENSE                 # MIT License
├── README.md               # This file
├── interaction_records.md  # Development history log
└── venv/                   # Python virtual environment (not tracked)
```

### 📚 Documentation

All comprehensive guides and documentation are organized in the [`knowledge_base/`](knowledge_base/) directory. **Start here**: [`knowledge_base/README.md`](knowledge_base/README.md) for a complete index and navigation guide.

**Quick Access:**
- **Getting Started**: [SETUP_AND_RUN.md](knowledge_base/SETUP_AND_RUN.md), [QUICKSTART.md](knowledge_base/QUICKSTART.md), [QUICK_START_RUNS.md](knowledge_base/QUICK_START_RUNS.md)
- **Training**: [ULTRAFAST_TRAINING_GUIDE.md](knowledge_base/ULTRAFAST_TRAINING_GUIDE.md), [FINAL_RUN_HYPERPARAMETERS.md](knowledge_base/FINAL_RUN_HYPERPARAMETERS.md), [ULTRAFAST_VS_FULL_COMPARISON.md](knowledge_base/ULTRAFAST_VS_FULL_COMPARISON.md)
- **Platform**: [TINKER_README.md](knowledge_base/TINKER_README.md) - Tinker integration guide
- **Implementation**: [CHECKPOINT_GUIDE.md](knowledge_base/CHECKPOINT_GUIDE.md), [CHECKPOINT_IMPLEMENTATION.md](knowledge_base/CHECKPOINT_IMPLEMENTATION.md), [REWARD_REFINEMENT.md](knowledge_base/REWARD_REFINEMENT.md), [DETECTOR_SETUP.md](knowledge_base/DETECTOR_SETUP.md)
- **Evaluation**: [ESL_FAIRNESS_GUIDE.md](knowledge_base/ESL_FAIRNESS_GUIDE.md), [IMPLEMENTATION_VERIFICATION.md](knowledge_base/IMPLEMENTATION_VERIFICATION.md)
- **Research**: [RESEARCH_ROADMAP.md](knowledge_base/RESEARCH_ROADMAP.md), [NEXT_STEPS.md](knowledge_base/NEXT_STEPS.md), [PRESENTATION_GUIDE.md](knowledge_base/PRESENTATION_GUIDE.md)
- **Operations**: [RUN_MANAGEMENT.md](knowledge_base/RUN_MANAGEMENT.md), [TEAM_HANDOFF.md](knowledge_base/TEAM_HANDOFF.md)

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

## 🚀 How to Run StealthRL

This section explains how to use StealthRL for training, evaluation, and visualization.

### 1. Training

StealthRL supports three training modes based on your needs:

#### Option A: Ultra-Fast Training (2-4 hours, proof-of-concept)

**Use case**: Quick iteration, testing pipeline, proof-of-concept results

```bash
# Using the ultrafast config
python -m stealthrl.tinker.train \
  --config configs/tinker_stealthrl_ultrafast.yaml \
  --data-path data/tinker_large \
  --run-name my_ultrafast_run
```

**What this does**:
- Trains for 1 epoch on 800 samples (50 steps)
- Uses single detector (Fast-DetectGPT) for speed
- LoRA rank 16, LR 5e-5, batch size 16
- Outputs to `outputs/runs/my_ultrafast_run/`
- Expected time: ~3.5 hours on Tinker

**Expected results**:
- 20-25% detector evasion improvement
- 98%+ semantic similarity
- Parse success >95%
- 9+ Pareto-optimal checkpoints

#### Option B: Full Production Training (6-8 hours, research quality)

**Use case**: Publication-ready results, maximum performance

```bash
# Using the full ensemble config
python -m stealthrl.tinker.train \
  --config configs/tinker_stealthrl.yaml \
  --data-path data/tinker_large \
  --run-name full_ensemble_run
```

**What this does**:
- Trains for 3 epochs on 20,000+ samples
- Uses 3-detector ensemble (Fast-DetectGPT + Ghostbuster + Binoculars)
- LoRA rank 32, LR 2.8e-4, batch size 4
- Proper 40% ESL / 60% native split
- Outputs to `outputs/runs/full_ensemble_run/`
- Expected time: ~6-8 hours on Tinker

**Expected results**:
- 60-70% ASR across ensemble (3× better than ultrafast)
- 88%+ semantic similarity
- ESL FPR gap <0.07
- Transfer ratio >0.7 to held-out detector

#### Option C: Transfer Learning Experiment (3-4 hours)

**Use case**: Testing generalization to unseen detectors

```bash
# Train on 2 detectors, evaluate on 3rd held-out
python -m stealthrl.tinker.train \
  --config configs/tinker_transfer_in_ensemble.yaml \
  --data-path data/tinker_large \
  --run-name transfer_experiment
```

**What this does**:
- Trains ONLY on Fast-DetectGPT + Ghostbuster (2 families)
- Evaluates transfer to held-out Binoculars (paired-LM family)
- Computes transfer ratio: ASR_binoculars / ASR_in_ensemble
- Target transfer ratio: >0.7 indicates good generalization

**Configuration file locations**:
- `configs/tinker_stealthrl_ultrafast.yaml` - Ultra-fast (recommended for testing)
- `configs/tinker_stealthrl.yaml` - Full production (recommended for research)
- `configs/tinker_transfer_in_ensemble.yaml` - Transfer experiment
- `configs/ablations/*.yaml` - 5 ablation studies

### 2. Evaluation

#### Run Comprehensive Detector Evaluation

```bash
# Evaluate trained model on test set
python scripts/evaluate_detectors.py \
  --model-path outputs/runs/full_ensemble_run \
  --test-data data/tinker_large/test.jsonl \
  --detectors fast_detectgpt ghostbuster binoculars \
  --output-dir results/detector_eval
```

**Outputs**:
- `detector_scores.csv` - Per-sample detector probabilities
- `metrics_summary.json` - AUROC, FPR@TPR95, ASR for each detector
- `comparison_table.txt` - Formatted results table

#### Run ESL Fairness Evaluation

```bash
# Evaluate fairness metrics by ESL status
python scripts/run_esl_eval.py \
  --eval-data data/processed/esl_native_test.jsonl \
  --stealthrl-model outputs/runs/full_ensemble_run \
  --enable-bertscore \
  --output-dir results/esl_fairness
```

**Outputs**:
- `esl_native_grouped_metrics.json` - Metrics split by ESL/native
- `comparison_report.json` - Overall summary
- `bertscore_results.json` - Semantic similarity by group
- `fairness_gap_analysis.txt` - FPR gap reduction analysis

**Key metrics to check**:
- **FPR Gap**: `FPR(ESL) - FPR(native)` should be <0.07 (baseline ~0.15)
- **ASR by Group**: Attack success rate for ESL vs native samples
- **Semantic Sim**: BERTScore F1 should be >0.88 for both groups

#### Run StealthBench (Unified Multi-Detector Benchmark)

```bash
# Run all detectors on same samples for fair comparison
python scripts/run_stealthbench.py \
  --config configs/stealthbench.yaml \
  --output-dir results/stealthbench
```

**Outputs**:
- Standardized AUROC, FPR@0.5%, FPR@1% for all detectors
- Comparison plots showing detector performance
- ESL fairness metrics integrated

#### Run Transfer Evaluation

```bash
# Measure transfer to held-out detector
python scripts/evaluate_transfer.py \
  --model-path outputs/runs/transfer_experiment \
  --in-ensemble fast_detectgpt ghostbuster \
  --held-out binoculars \
  --test-data data/tinker_large/test.jsonl \
  --output-dir results/transfer_eval
```

**Outputs**:
- Transfer ratio calculation
- ASR comparison (in-ensemble vs held-out)
- Analysis of which strategies transfer

### 3. Visualization

#### Generate Training Visualizations

```bash
# Create comprehensive training analysis plots
python scripts/visualize_training_results.py \
  --run-dir outputs/runs/full_ensemble_run \
  --output-dir outputs/runs/full_ensemble_run/visualizations
```

**Generates 8 plots** (PNG 300 DPI + PDF):

1. **training_curves.png**: 6-panel progression
   - Total reward over time
   - Detector evasion (lower = better)
   - Semantic similarity (higher = better)
   - Perplexity (target ~30)
   - KL divergence (monitors drift)
   - Parse success rate

2. **pareto_frontiers.png**: Multi-objective trade-offs
   - 2D plot: Stealth vs Quality (9 red stars = Pareto-optimal)
   - 3D plot: Stealth × Quality × Naturalness (26 blue diamonds)
   - Shows which checkpoints are optimal for different use cases

3. **reward_decomposition.png**: Component analysis
   - Stacked area chart showing reward contributions
   - Individual component trajectories
   - Detector probability distribution (histogram)
   - Correlation heatmap between metrics

4. **stability_metrics.png**: Convergence analysis
   - Entropy (exploration level)
   - Learning rate schedule
   - Token generation statistics
   - Iteration timing

**Summary statistics** also generated:
- `training_summary.csv` - All metrics in spreadsheet format
- `training_summary.txt` - Human-readable summary

#### Visualize StealthBench Results

```bash
# Create detector comparison visualizations
python scripts/visualize_stealthbench.py \
  --results results/stealthbench \
  --output-dir outputs/figures
```

**Generates**:
- ROC curves for all detectors
- FPR comparison bar charts
- ESL vs native fairness heatmaps

### 4. Using Trained Models for Paraphrasing

Once training is complete, you can use the model to paraphrase new texts:

```bash
# Paraphrase a single text
python scripts/paraphrase_example.py \
  --input "Your AI-generated text here..." \
  --model-path outputs/runs/full_ensemble_run \
  --checkpoint step_22  # Or step_49 for balanced, step_23 for quality
  --output-path outputs/paraphrased.txt
```

**Python API usage**:

```python
from stealthrl.models import load_stealthrl_model

# Load trained model
model, tokenizer = load_stealthrl_model(
    base_model="Qwen/Qwen3-4B-Instruct-2507",
    lora_path="outputs/runs/full_ensemble_run/checkpoints/step_22"
)

# Paraphrase text
input_text = "AI detectors use statistical patterns to identify generated text."
paraphrased = model.paraphrase(input_text, max_tokens=400, temperature=0.8)
print(paraphrased)

# Check detector scores
from stealthrl.detectors import FastDetectGPTDetector
detector = FastDetectGPTDetector()
original_score = detector.detect(input_text).item()
paraphrased_score = detector.detect(paraphrased).item()
print(f"Original: {original_score:.3f} → Paraphrased: {paraphrased_score:.3f}")
print(f"Improvement: {(original_score - paraphrased_score) / original_score * 100:.1f}%")
```

### 5. Running Ablation Studies

Test necessity of each reward component:

```bash
# Run all 5 ablation experiments
bash scripts/run_ablations.sh
```

This runs 5 experiments in parallel (if resources allow):
1. **Detector-only**: Remove semantic/quality/fairness → degenerate outputs expected
2. **No fairness**: Remove ESL penalty → higher ESL FPR gap expected
3. **No quality**: Remove perplexity reward → unnatural fluency expected
4. **No semantic**: Remove similarity constraint → semantic drift expected
5. **Single detector**: Only Fast-DetectGPT → poor transfer expected

**Expected time**: 10-15 hours total (2-3 hours each)

**Outputs**: Each experiment creates its own directory in `outputs/runs/ablation_*/`

### 6. Monitoring Training

While training is running, monitor progress:

**Option 1: Check logs**
```bash
# View training log
tail -f outputs/runs/full_ensemble_run/training.log

# View last 50 lines
tail -50 outputs/runs/full_ensemble_run/training.log
```

**Option 2: Check metrics**
```bash
# View metrics in real-time
tail -f outputs/runs/full_ensemble_run/metrics.jsonl | jq
```

**Option 3: TensorBoard** (if enabled)
```bash
tensorboard --logdir outputs/runs/full_ensemble_run/tensorboard --port 6006
# Then open http://localhost:6006 in browser
```

**Option 4: Tinker Dashboard**
- Go to [Tinker Platform](https://tinker.thinkingmachines.ai/)
- Navigate to "Jobs" → find your run
- View live metrics, logs, resource usage

### Common Training Commands Summary

```bash
# Quick test (10 samples, 2 steps)
python scripts/prepare_tinker_data.py --synthetic --num-train 10 --output-dir data/test
python -m stealthrl.tinker.train --data-path data/test --run-name test --num-epochs 1 --batch-size 2

# Ultra-fast proof-of-concept (3.5 hours)
python -m stealthrl.tinker.train --config configs/tinker_stealthrl_ultrafast.yaml --data-path data/tinker_large --run-name ultrafast

# Full production training (6-8 hours)
python -m stealthrl.tinker.train --config configs/tinker_stealthrl.yaml --data-path data/tinker_large --run-name production

# Transfer experiment (3-4 hours)
python -m stealthrl.tinker.train --config configs/tinker_transfer_in_ensemble.yaml --data-path data/tinker_large --run-name transfer

# Evaluate trained model
python scripts/evaluate_detectors.py --model-path outputs/runs/production --test-data data/tinker_large/test.jsonl

# Generate visualizations
python scripts/visualize_training_results.py --run-dir outputs/runs/production

# ESL fairness evaluation
python scripts/run_esl_eval.py --eval-data data/processed/esl_native_test.jsonl --stealthrl-model outputs/runs/production
```

---

## 📁 Configuration Files

StealthRL uses YAML configuration files for all training settings. This enables zero-code experimentation—just modify YAML files to change hyperparameters, detectors, or reward weights.

### Available Configurations

| Config File | Purpose | Training Time | Dataset Size | Detectors |
|-------------|---------|---------------|--------------|-----------|
| `tinker_stealthrl_ultrafast.yaml` | Quick proof-of-concept | ~3.5 hours | 800 samples, 1 epoch | Fast-DetectGPT only |
| `tinker_stealthrl.yaml` | Full production training | ~6-8 hours | 20K samples, 3 epochs | 3-detector ensemble |
| `tinker_transfer_in_ensemble.yaml` | Transfer experiment | ~3-4 hours | 20K samples, 2 epochs | 2 in-ensemble + 1 held-out |
| `stealthrl_small.yaml` | Local training (legacy) | Varies | Custom | Configurable |
| `stealthbench.yaml` | Evaluation only | N/A | Test set | All available |

**Ablation configs** (`configs/ablations/`):
- `detector_only.yaml` - Remove semantic/quality/fairness constraints
- `no_fairness.yaml` - Remove ESL penalty term
- `no_quality.yaml` - Remove perplexity reward
- `no_semantic.yaml` - Remove similarity constraint
- `single_detector_fast_detectgpt.yaml` - Single detector (no ensemble)

### Configuration Structure

All configs follow this structure:

```yaml
# Model settings
model:
  name: "Qwen/Qwen3-4B-Instruct-2507"  # Base model from HuggingFace
  renderer: "qwen3"  # Chat template format

# LoRA settings (parameter-efficient fine-tuning)
lora:
  rank: 16  # Adapter rank (16 for speed, 32 for quality)
  alpha: 16  # Scaling factor (usually = rank)
  dropout: 0.05  # Dropout for regularization
  target_modules: null  # null = all linear layers

# Training hyperparameters
training:
  learning_rate: 5e-5  # Critical: 5e-5 for stability, 2.8e-4 for performance
  lr_scheduler_type: "cosine"  # Gradual decay
  warmup_ratio: 0.1  # Warm up first 10% of training
  batch_size: 16  # Number of prompts per batch
  group_size: 8  # GRPO group size (rollouts per prompt)
  num_epochs: 1  # How many passes through data
  max_tokens: 400  # Max tokens per generation

# Sampling settings (for generation during RL)
sampling:
  temperature: 0.8  # Sampling temperature (0.8 for stability, 1.0 for exploration)
  temperature_schedule: "constant"  # "constant" or "linear_decay"
  top_p: 0.9  # Nucleus sampling threshold
  do_sample: true  # Use sampling (not greedy)

# GRPO settings (Group Relative Policy Optimization)
grpo:
  normalize_advantages: true  # Normalize advantages per group
  advantage_clip: 5.0  # Clip advantages to [-5, 5]
  reward_clip: 10.0  # Clip rewards to prevent outliers
  remove_constant_reward_groups: true  # Skip groups with identical rewards

# KL divergence penalty (prevents drift from base model)
kl:
  penalty_coef: 0.03  # KL penalty strength (higher = less drift)
  target: 4.0  # Adaptive KL target (if exceeded, increase penalty)
  adapt_rate: 0.1  # How fast to adapt penalty

# Dataset configuration
dataset:
  path: "data/tinker_large"  # Path to JSONL files
  max_train_examples: 800  # Limit training samples (null = use all)
  max_test_examples: 150  # Limit test samples
  split: "train"  # Which split to use
  seed: 42  # Random seed for reproducibility

# Reward configuration (multi-objective)
reward:
  detector_weight: 1.0  # Detector evasion importance
  semantic_weight: 1.0  # Semantic similarity importance
  perplexity_weight: 0.5  # Naturalness importance
  fairness_weight: 0.2  # ESL fairness importance
  
  detectors:  # Which detectors to use
    names:
      - "fast_detectgpt"  # Fast curvature-based detector
      # - "ghostbuster"  # RoBERTa classifier (uncomment for ensemble)
      # - "binoculars"  # Paired-LM detector (uncomment for ensemble)
    weights:  # Per-detector weights (if ensemble)
      fast_detectgpt: 1.0
    device: "cpu"  # "cpu" or "cuda"
  
  semantic:  # Semantic similarity settings
    model_name: "intfloat/e5-small-v2"  # E5 model (e5-large for better quality)
    threshold: 0.85  # Minimum acceptable similarity
    device: "cpu"
  
  perplexity:  # Naturalness settings
    model_name: "gpt2"  # Perplexity model
    ppl_min: 5.0  # Min acceptable perplexity
    ppl_max: 80.0  # Max acceptable perplexity
    ppl_target: 30.0  # Optimal perplexity (natural human text)
    device: "cpu"
  
  fairness:  # ESL fairness settings
    enabled: true  # Whether to apply fairness penalty
    penalty_strength: 0.2  # Fairness penalty weight

# Logging configuration
logging:
  log_path: "outputs/tinker_ultrafast"  # Where to save outputs
  log_interval: 10  # Log every N steps
  eval_interval: 50  # Evaluate every N steps
  save_interval: 100  # Save checkpoint every N steps
  num_groups_to_log: 2  # How many example groups to log
  debug_mode: false  # Enable verbose debugging
```

### How to Modify Configurations

**Example 1: Change detector ensemble**

```yaml
# Edit configs/tinker_stealthrl.yaml
reward:
  detectors:
    names:
      - "fast_detectgpt"
      - "ghostbuster"  # Add Ghostbuster
      - "binoculars"   # Add Binoculars
    weights:
      fast_detectgpt: 0.33
      ghostbuster: 0.33
      binoculars: 0.34
```

**Example 2: Adjust reward weights**

```yaml
# Prioritize stealth over quality
reward:
  detector_weight: 2.0  # Increase from 1.0
  semantic_weight: 0.8  # Decrease from 1.0
  perplexity_weight: 0.3  # Decrease from 0.5
  fairness_weight: 0.2  # Keep same
```

**Example 3: Change learning rate and epochs**

```yaml
training:
  learning_rate: 2.8e-4  # Increase for faster convergence
  num_epochs: 3  # More epochs for better results
```

**Example 4: Use different base model**

```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"  # Larger model
  renderer: "qwen"

lora:
  rank: 32  # Increase rank for larger model
```

### Creating Custom Configurations

Copy an existing config and modify:

```bash
# Copy ultrafast config
cp configs/tinker_stealthrl_ultrafast.yaml configs/my_custom_config.yaml

# Edit as needed
nano configs/my_custom_config.yaml

# Run with custom config
python -m stealthrl.tinker.train \
  --config configs/my_custom_config.yaml \
  --data-path data/tinker_large \
  --run-name my_custom_run
```

---

## 📊 Output Files Generated

StealthRL generates comprehensive outputs organized by run name.

### Directory Structure

```
outputs/
└── runs/
    └── <run_name>/  # e.g., "full_ensemble_run"
        ├── config.yaml  # Copy of config used for this run
        ├── training.log  # Human-readable training log
        ├── metrics.jsonl  # One JSON object per training step
        ├── run_metadata.json  # Run info (start time, config hash, etc.)
        │
        ├── checkpoints/  # Model checkpoints
        │   ├── step_100/
        │   ├── step_200/
        │   ├── ...
        │   └── final/  # Final checkpoint
        │
        ├── tensorboard/  # TensorBoard logs (if enabled)
        │   └── events.out.tfevents.*
        │
        ├── visualizations/  # Generated plots (run visualize_training_results.py)
        │   ├── training_curves.png
        │   ├── training_curves.pdf
        │   ├── pareto_frontiers.png
        │   ├── pareto_frontiers.pdf
        │   ├── reward_decomposition.png
        │   ├── reward_decomposition.pdf
        │   ├── stability_metrics.png
        │   ├── stability_metrics.pdf
        │   ├── training_summary.csv
        │   └── training_summary.txt
        │
        └── train_iteration_*.html  # Per-iteration detailed reports
            ├── train_iteration_0000.html
            ├── train_iteration_0001.html
            └── ...
```

### File Descriptions

#### `training.log`

Human-readable log with timestamped events:

```
2025-12-07 21:21:10 - INFO - Starting GRPO training
2025-12-07 21:21:10 - INFO - Model: Qwen/Qwen3-4B-Instruct-2507
2025-12-07 21:21:10 - INFO - LoRA rank: 16
2025-12-07 21:25:32 - INFO - Step 1/50 | Reward: 0.678 | Detector: -0.092 | Semantic: 0.9856 | PPL: 28.0
2025-12-07 21:29:54 - INFO - Step 2/50 | Reward: 0.721 | Detector: -0.045 | Semantic: 0.9871 | PPL: 29.3
...
2025-12-07 23:52:18 - INFO - Training complete! Best checkpoint: step_22
```

#### `metrics.jsonl`

One JSON object per step (newline-delimited):

```json
{"step": 1, "total_reward": 0.678, "detector_reward": -0.092, "semantic_reward": 0.9856, "perplexity_reward": 0.450, "fairness_penalty": 0.023, "detector_prob": 0.587, "semantic_sim": 0.9856, "perplexity": 28.0, "kl_div": 0.0099, "parse_success": 0.859, "entropy": 1.234, "lr": 5e-06}
{"step": 2, "total_reward": 0.721, "detector_reward": -0.045, "semantic_reward": 0.9871, "perplexity_reward": 0.465, "fairness_penalty": 0.019, "detector_prob": 0.571, "semantic_sim": 0.9871, "perplexity": 29.3, "kl_div": 0.0124, "parse_success": 0.875, "entropy": 1.198, "lr": 1.5e-05}
...
```

**Key metrics**:
- `total_reward`: Sum of all reward components (higher = better, can be negative)
- `detector_reward`: Z-score normalized detector evasion (higher = better evasion)
- `detector_prob`: Raw detector probability (lower = better, 0-1 scale)
- `semantic_reward`: Semantic similarity reward
- `semantic_sim`: Raw cosine similarity (0-1 scale, target >0.85)
- `perplexity_reward`: Naturalness reward
- `perplexity`: Raw perplexity value (target ~30)
- `fairness_penalty`: ESL fairness penalty (subtracted from total)
- `kl_div`: KL divergence from base model (target <4.0)
- `parse_success`: Fraction of valid outputs (target >0.95)
- `entropy`: Policy entropy (exploration measure)
- `lr`: Current learning rate

#### `run_metadata.json`

Run configuration and metadata:

```json
{
  "run_name": "full_ensemble_run",
  "start_time": "2025-12-07T21:21:10",
  "end_time": "2025-12-07T23:52:18",
  "duration_seconds": 9068,
  "config_hash": "abc123...",
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "lora_rank": 16,
  "learning_rate": 5e-05,
  "batch_size": 16,
  "group_size": 8,
  "num_epochs": 1,
  "total_steps": 50,
  "dataset_size": 800,
  "detectors": ["fast_detectgpt"],
  "best_checkpoint": "step_22",
  "final_metrics": {
    "total_reward": 0.854,
    "detector_prob": 0.579,
    "semantic_sim": 0.9865,
    "perplexity": 30.1
  }
}
```

#### `checkpoints/step_*/`

Each checkpoint directory contains:
- `adapter_model.bin` - LoRA adapter weights
- `adapter_config.json` - LoRA configuration
- `tokenizer_config.json` - Tokenizer settings
- `special_tokens_map.json` - Special tokens

**Loading a checkpoint**:

```python
from stealthrl.models import load_stealthrl_model

model, tokenizer = load_stealthrl_model(
    base_model="Qwen/Qwen3-4B-Instruct-2507",
    lora_path="outputs/runs/full_ensemble_run/checkpoints/step_22"
)
```

#### `visualizations/training_summary.csv`

Spreadsheet format for all metrics:

```csv
metric,initial,final,best,mean,std,min,max
total_reward,0.678,0.854,2.508,0.912,0.445,-0.234,2.508
detector_reward,-0.092,-0.223,2.069,0.145,0.623,-1.024,2.069
detector_prob,0.587,0.579,0.458,0.562,0.034,0.458,0.602
semantic_sim,0.9856,0.9865,0.9952,0.9879,0.0054,0.9442,0.9952
perplexity,28.0,30.1,23.5,35.4,14.2,23.5,85.8
kl_div,0.0099,0.2479,3.0643,0.3145,0.5234,0.0099,3.0643
parse_success,0.859,0.992,1.000,0.951,0.048,0.812,1.000
```

#### `train_iteration_*.html`

Detailed HTML reports for each iteration showing:
- Prompt text
- All 8 generated paraphrases (group)
- Individual rewards for each paraphrase
- Group statistics (mean, std, advantages)
- Policy update information

Useful for debugging and understanding what the model is learning.

### Evaluation Output Files

When running `evaluate_detectors.py`:

```
results/detector_eval/
├── detector_scores.csv  # Per-sample scores for each detector
├── metrics_summary.json  # AUROC, FPR@TPR95, ASR
└── comparison_table.txt  # Formatted results table
```

When running `run_esl_eval.py`:

```
results/esl_fairness/
├── esl_native_grouped_metrics.json  # Metrics split by ESL/native
├── comparison_report.json  # Overall summary
├── bertscore_results.json  # Semantic similarity by group
├── fairness_gap_analysis.txt  # FPR gap reduction
└── bertscore_esl_native.jsonl  # Per-sample details
```

### Disk Space Requirements

- **Ultra-fast run**: ~500 MB (50 checkpoints × ~10 MB each)
- **Full production run**: ~2-3 GB (150-300 checkpoints × ~10 MB each)
- **Visualizations**: ~10-20 MB per run (8 plots in PNG + PDF)
- **Logs and metrics**: ~50-100 MB per run

**Total for complete project**: ~5-10 GB including all runs, data, and visualizations

---

## 📚 Datasets

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
