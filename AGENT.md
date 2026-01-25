# AGENT.md - StealthRL Project Reference Guide

**Last Updated**: January 24, 2026  
**Version**: 1.0  
**Purpose**: Comprehensive reference for AI agents and developers working with the StealthRL codebase

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Reference](#quick-reference)
3. [Repository Structure](#repository-structure)
4. [Core Components](#core-components)
5. [Training Pipeline](#training-pipeline)
6. [Evaluation System](#evaluation-system)
7. [Configuration System](#configuration-system)
8. [Data Pipeline](#data-pipeline)
9. [Key Scripts](#key-scripts)
10. [Development Workflow](#development-workflow)
11. [Common Tasks](#common-tasks)
12. [Troubleshooting](#troubleshooting)
13. [Research Context](#research-context)

---

## Project Overview

### What is StealthRL?

**StealthRL** is a reinforcement learning framework that trains language models to evade multiple AI text detectors simultaneously while maintaining semantic quality and ESL (English as Second Language) fairness.

**Key Innovation**: Train ONE model against MULTIPLE detectors (vs. AuthorMist's one-model-per-detector approach).

### Core Technology Stack

- **Base Model**: Qwen/Qwen3-4B-Instruct-2507
- **Training**: GRPO (Group Relative Policy Optimization) via Tinker platform
- **Adapter**: LoRA (Low-Rank Adaptation) - rank=32, alpha=32
- **Detectors**: Fast-DetectGPT, Ghostbuster, Binoculars
- **Platform**: Tinker remote compute (https://tinker.thinkingmachines.ai/)

### Project Status (December 7, 2025)

✅ **Ultra-Fast Proof-of-Concept**: COMPLETE (22% evasion improvement, 98.6% semantic similarity)  
⌛ **Full Production Run**: READY FOR EXECUTION (6-8 hours on Tinker)  
📊 **Research Questions**: Infrastructure complete, awaiting full-scale experiments

---

## Quick Reference

### Essential Commands

```bash
# Setup environment
pip install -r requirements.txt
export TINKER_API_KEY="your_key_here"  # or add to .env

# Prepare data (synthetic test)
python scripts/prepare_tinker_data.py \
  --synthetic \
  --num-train 1000 \
  --num-test 200 \
  --output-dir data/tinker_test

# Train StealthRL
python -m stealthrl.tinker.train \
  --config configs/tinker_stealthrl.yaml \
  --data-path data/tinker_full_esl40_nodup \
  --run-name my_experiment \
  --num-epochs 3 \
  --batch-size 64

# Evaluate model
python scripts/run_stealthbench.py \
  --checkpoint outputs/runs/my_experiment/final.ckpt \
  --data-path data/tinker_full_esl40_nodup \
  --output-dir outputs/evaluation

# Fairness analysis
python scripts/analyze_detector_fairness.py \
  --data-path data/tinker_full_esl40_nodup \
  --split test \
  --max-samples 500 \
  --output-json outputs/fairness/results.json
```

### Key File Locations

- **Main Training Script**: `stealthrl/tinker/train.py` (1040 lines)
- **Detector Implementations**: `stealthrl/detectors/*.py`
- **Tinker Integration**: `stealthrl/tinker/*.py` (11 modules)
- **Configuration Files**: `configs/*.yaml`
- **Documentation**: `knowledge_base/*.md` (25+ guides)
- **Evaluation Scripts**: `scripts/evaluate_*.py`

---

## Repository Structure

```
StealthRL/
├── stealthrl/                    # Core Python package
│   ├── tinker/                   # Tinker platform integration (GRPO training)
│   │   ├── train.py             # Main training loop (1040 lines)
│   │   ├── reward.py            # Multi-objective reward computation
│   │   ├── detectors.py         # Detector ensemble management
│   │   ├── semantic.py          # Semantic similarity (E5 embeddings)
│   │   ├── perplexity.py        # Quality control (GPT-2 perplexity)
│   │   ├── evaluation.py        # ASR, AUROC, F1 metrics
│   │   ├── dataset.py           # Data loading and formatting
│   │   ├── inference.py         # Model inference utilities
│   │   └── env.py              # Environment variables
│   │
│   ├── detectors/               # AI detector implementations
│   │   ├── base_detector.py    # Abstract base class
│   │   ├── fast_detectgpt.py   # Curvature-based detection
│   │   ├── ghostbuster.py      # RoBERTa classifier
│   │   └── binoculars.py       # Paired-LM detection
│   │
│   └── baselines/               # Baseline comparisons
│       ├── authormist.py       # AuthorMist baseline
│       └── paraphrase_baselines.py
│
├── configs/                     # Training configurations
│   ├── tinker_stealthrl.yaml   # Full ensemble training
│   ├── tinker_stealthrl_ultrafast.yaml  # Quick proof-of-concept
│   ├── tinker_transfer_in_ensemble.yaml  # Transfer evaluation
│   ├── stealthrl_small.yaml    # Small-scale testing
│   └── ablations/              # Ablation study configs
│       ├── detector_only.yaml
│       ├── no_fairness.yaml
│       ├── no_quality.yaml
│       ├── no_semantic.yaml
│       └── single_detector.yaml
│
├── scripts/                     # Utility scripts (40+ files)
│   ├── prepare_tinker_data.py  # Data preparation
│   ├── build_full_dataset.py   # Combine multiple datasets
│   ├── run_stealthbench.py     # Comprehensive evaluation
│   ├── run_research_pipeline.py # Automated research pipeline
│   ├── evaluate_transfer.py    # Transfer evaluation
│   ├── analyze_detector_fairness.py  # ESL fairness analysis
│   ├── visualize_training_results.py  # Generate plots
│   ├── monitor_training.py     # Real-time training monitor
│   └── paraphrase_example.py   # Simple inference demo
│
├── data/                        # Datasets
│   ├── raw/                    # Raw downloaded datasets
│   ├── tinker/                 # Processed training data
│   ├── tinker_full_esl40_nodup/ # Full dataset (40% ESL)
│   ├── esl/                    # ESL-specific corpora
│   └── native/                 # Native English corpora
│
├── outputs/                     # Training outputs
│   ├── runs/                   # Training run artifacts
│   │   └── {run_name}/
│   │       ├── checkpoints/    # Model checkpoints
│   │       ├── logs/          # Training logs
│   │       ├── visualizations/ # Training curves, Pareto plots
│   │       └── config.yaml    # Training configuration
│   └── fairness/              # Fairness evaluation results
│
├── knowledge_base/             # Documentation (25+ guides)
│   ├── README.md              # Documentation index
│   ├── QUICKSTART.md          # Fast-track guide
│   ├── TEAM_HANDOFF.md        # Team onboarding
│   ├── SETUP_AND_RUN.md       # Complete setup guide
│   ├── DETECTOR_SETUP.md      # Detector configuration
│   ├── ESL_FAIRNESS_GUIDE.md  # Fairness evaluation
│   ├── RESEARCH_ROADMAP.md    # Research plan
│   ├── FINAL_RUN_HYPERPARAMETERS.md  # Optimized hyperparams
│   ├── CHECKPOINT_GUIDE.md    # Checkpoint management
│   ├── REWARD_REFINEMENT.md   # Reward function design
│   └── DATA_CURATION_ANALYSIS.md  # Data requirements
│
├── report/                     # Research report and paper
│   ├── REPORT.md              # Main project report
│   ├── report.tex             # LaTeX source
│   └── StealthRL_Methodology.png  # Architecture diagram
│
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment
├── README.md                   # Main project README
├── atharv_readme.md           # Quick command reference
└── AGENT.md                   # This file

```

---

## Core Components

### 1. Training System (`stealthrl/tinker/`)

#### `train.py` - Main Training Loop
**Purpose**: Implements GRPO training on Tinker platform  
**Key Classes**:
- `StealthRLConfig`: Training configuration dataclass
- `StealthRLTrainer`: Main training orchestrator
- `TensorBoardLogger`: Dual JSON + TensorBoard logging

**Key Functions**:
```python
async def train(
    config: StealthRLConfig,
    data_dir: str,
    output_dir: str,
    run_name: str,
    num_epochs: int = 2,
    checkpoint_interval: int = 10
) -> None
```

**Training Loop**:
1. Load dataset from JSONL files
2. Initialize Tinker training client with LoRA config
3. For each batch:
   - Generate K rollouts per prompt (group_size=4-16)
   - Compute multi-objective reward for each rollout
   - Calculate group-normalized advantages
   - Perform GRPO update with KL penalty
4. Save checkpoints at intervals
5. Generate visualizations and metrics

#### `reward.py` - Multi-Objective Reward
**Purpose**: Compute combined reward from multiple objectives

**Reward Components**:
```python
R_total = w_d × R_detector + w_s × R_semantic + w_q × R_quality + R_fair

where:
  R_detector = -Σ(w_i × P_detector_i)  # Lower detection probability
  R_semantic = BERTScore or E5 similarity  # Maintain meaning
  R_quality = -perplexity_penalty  # Maintain fluency
  R_fair = -0.2 × P_detector × 𝟙[ESL]  # Reduce ESL bias
```

**Key Function**:
```python
async def compute_reward(
    text: str,
    reference: str,
    detector_scores: Dict[str, float],
    is_esl: bool,
    config: RewardConfig
) -> Dict[str, float]
```

#### `detectors.py` - Detector Ensemble
**Purpose**: Manage multiple AI text detectors

**Supported Detectors**:
- **Fast-DetectGPT**: Curvature-based (log-likelihood perturbations)
- **Ghostbuster**: RoBERTa classifier trained on AI detection
- **Binoculars**: Paired-LM (compare two models' probabilities)

**Key Class**:
```python
class DetectorEnsemble:
    def __init__(self, detector_configs: List[Dict], weights: List[float])
    
    async def compute_ensemble_score(self, text: str) -> Dict[str, float]:
        """Returns {detector_name: score} for all detectors."""
```

**Note**: Current implementation uses mock detectors for testing. Replace `_compute_score()` methods with real detector API calls for production.

#### `semantic.py` - Semantic Similarity
**Purpose**: Measure semantic preservation

**Methods**:
- **E5 Embeddings**: `intfloat/e5-large-v2` sentence embeddings
- **BERTScore**: Token-level F1 with contextual embeddings

**Key Function**:
```python
def compute_semantic_similarity(
    text1: str,
    text2: str,
    method: str = "e5"  # or "bertscore"
) -> float:
    """Returns similarity score [0, 1]."""
```

#### `perplexity.py` - Quality Control
**Purpose**: Detect unnatural or low-quality text

**Implementation**:
- Uses GPT-2 (124M) for perplexity scoring
- Perplexity bands: [20, 100] = acceptable, >100 = penalty
- Prevents mode collapse and gibberish generation

**Key Function**:
```python
def compute_perplexity_penalty(text: str, max_ppl: float = 100.0) -> float:
    """Returns penalty [0, 1] if perplexity > max_ppl."""
```

#### `evaluation.py` - Metrics Computation
**Purpose**: Compute evaluation metrics

**Metrics**:
- **ASR (Attack Success Rate)**: Fraction below detection threshold (0.5)
- **AUROC**: Area Under ROC Curve
- **F1**: Harmonic mean of precision/recall
- **FPR/FNR**: False positive/negative rates
- **ESL Gap**: FPR_esl - FPR_native

**Key Function**:
```python
def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]
```

### 2. Detector Implementations (`stealthrl/detectors/`)

#### Base Structure
All detectors inherit from `BaseDetector`:

```python
class BaseDetector(ABC):
    @abstractmethod
    def load_model(self) -> None:
        """Load detector model."""
    
    @abstractmethod
    def detect(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Run detection on texts, return scores [0, 1]."""
```

#### Fast-DetectGPT (`fast_detectgpt.py`)
**Method**: Log-likelihood curvature  
**Model**: GPT-2 base model  
**Key Insight**: AI text has lower curvature than human text

**Implementation**:
```python
class FastDetectGPTDetector(BaseDetector):
    def detect(self, texts: List[str]) -> torch.Tensor:
        # 1. Compute log-likelihood
        # 2. Add small perturbations
        # 3. Measure curvature
        # 4. Return detection score
```

#### Ghostbuster (`ghostbuster.py`)
**Method**: Fine-tuned classifier  
**Model**: RoBERTa-based sequence classifier  
**Training**: Trained on human vs AI text pairs

**Implementation**:
```python
class GhostbusterDetector(BaseDetector):
    def __init__(self, model_name: str = "roberta-base-openai-detector"):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
```

#### Binoculars (`binoculars.py`)
**Method**: Paired language model comparison  
**Models**: Compare two different LMs (e.g., GPT-2 vs GPT-Neo)  
**Key Insight**: AI text shows similar probabilities across models

---

## Training Pipeline

### Configuration Flow

```
configs/*.yaml
    ↓
StealthRLConfig dataclass
    ↓
StealthRLTrainer initialization
    ↓
Tinker TrainingClient
    ↓
GRPO training loop
    ↓
outputs/runs/{run_name}/
```

### Training Workflow

#### 1. Data Loading
```python
# From dataset.py
dataset = load_jsonl_dataset(data_dir, split="train")
# Expected format:
# {
#   "ai_text": "AI-generated text...",
#   "human_reference": "Original human text...",
#   "domain": "academic",
#   "is_esl": true,
#   "metadata": {...}
# }
```

#### 2. Model Initialization
```python
# LoRA configuration
lora_config = {
    "rank": 32,              # Optimal for RL (Thinking Machines research)
    "alpha": 32,             # Standard scaling
    "dropout": 0.05,
    "target_modules": None   # All layers including MLP
}

# Initialize Tinker training client
training_client = tinker.TrainingClient(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    lora_config=lora_config
)
```

#### 3. GRPO Training Loop
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Generate K rollouts per prompt
        prompts = batch["ai_text"]
        references = batch["human_reference"]
        is_esl = batch["is_esl"]
        
        # Generate group_size rollouts per prompt
        rollouts = await training_client.generate_batch(
            prompts,
            temperature=1.0,
            top_p=0.95,
            max_tokens=512,
            num_samples=group_size  # e.g., 4-16
        )
        
        # Compute rewards
        rewards = []
        for rollout, ref, esl in zip(rollouts, references, is_esl):
            reward = await compute_reward(
                rollout.text,
                ref,
                detector_ensemble,
                esl
            )
            rewards.append(reward)
        
        # Group-normalize advantages
        advantages = normalize_advantages_by_group(rewards, group_size)
        
        # GRPO update
        loss = compute_grpo_loss(rollouts, advantages, config.kl_penalty)
        await training_client.optim_step(loss)
        
        # Log metrics
        logger.log_metrics({
            "reward_mean": np.mean(rewards),
            "advantage_mean": np.mean(advantages),
            "kl_divergence": kl_div,
            "detector_score_mean": detector_scores_mean
        }, step=global_step)
```

#### 4. Checkpoint Saving
```python
# Save every N steps
if global_step % checkpoint_interval == 0:
    checkpoint_id = await training_client.save_checkpoint()
    
    # Save metadata locally
    checkpoint_info = {
        "step": global_step,
        "epoch": epoch,
        "metrics": latest_metrics,
        "tinker_checkpoint_id": checkpoint_id
    }
    
    with open(f"{output_dir}/checkpoint_{global_step}.json", "w") as f:
        json.dump(checkpoint_info, f)
```

### Hyperparameters (Optimized)

**Critical Settings** (from `knowledge_base/FINAL_RUN_HYPERPARAMETERS.md`):

```yaml
# LoRA Configuration
lora:
  rank: 32              # Optimal for RL (don't go higher)
  alpha: 32             # Keep at 32 (don't scale with rank)
  target_modules: null  # All layers INCLUDING MLP (critical!)

# Learning Rate
training:
  learning_rate: 2.8e-4  # 10x FullFT rule (LoRA needs higher LR)

# Batch Sizes
training:
  batch_size: 4-16       # Small batches (LoRA penalty at large batches)
  group_size: 4-16       # GRPO rollouts per prompt (8 is sweet spot)

# Temperature
sampling:
  temperature: 1.0              # Keep at 1.0 for RL
  temperature_schedule: constant # Don't decay (hurts exploration)
  top_p: 0.95                   # Higher for diversity

# GRPO
grpo:
  normalize_advantages: true     # Critical for GRPO
  advantage_clip: 10.0          # Increase from 5.0
  remove_constant_reward_groups: true

# KL Penalty
kl:
  penalty_coef: 0.01  # Prevent drift (0.001 too small)
```

**Justification**: Based on Thinking Machines LoRA research showing:
- Rank=32 matches full fine-tuning performance
- LoRA requires 10x higher LR than full FT
- Small batches (<32) perform better with LoRA
- MLP layers contain 70%+ of parameters (must include)

---

## Evaluation System

### StealthBench (`scripts/run_stealthbench.py`)

**Purpose**: Comprehensive multi-detector evaluation harness

**Features**:
- Multi-detector evaluation (all 3 detectors)
- ESL-stratified analysis
- Transfer evaluation (held-out detectors)
- Quality metrics (semantic, perplexity, readability)
- Visualization generation

**Usage**:
```bash
python scripts/run_stealthbench.py \
  --checkpoint outputs/runs/my_run/checkpoint_50.json \
  --data-path data/tinker_full_esl40_nodup \
  --split test \
  --output-dir outputs/evaluation/my_run \
  --max-samples 1000
```

**Output**:
```
outputs/evaluation/my_run/
├── metrics.json              # All metrics
├── detector_scores.csv       # Per-sample detector scores
├── esl_stratified.json       # ESL vs native breakdown
├── pareto_checkpoints.json   # Optimal checkpoints
└── visualizations/
    ├── roc_curves.png
    ├── detector_comparison.png
    ├── esl_fairness.png
    └── quality_vs_stealth.png
```

### Fairness Evaluation (`scripts/analyze_detector_fairness.py`)

**Purpose**: Measure ESL bias in detectors

**Metrics**:
- FPR (False Positive Rate) per group
- ESL Gap: FPR_esl - FPR_native
- Threshold sensitivity analysis

**Usage**:
```bash
python scripts/analyze_detector_fairness.py \
  --data-path data/tinker_full_esl40_nodup \
  --split test \
  --text-field human_reference \
  --max-samples 500 \
  --detectors fast_detectgpt ghostbuster binoculars \
  --thresholds 0.3,0.5,0.7 \
  --output-json outputs/fairness/results.json
```

**Output**:
```json
{
  "fast_detectgpt": {
    "fpr_esl": 0.23,
    "fpr_native": 0.15,
    "esl_gap": 0.08,
    "threshold": 0.5
  },
  ...
}
```

### Transfer Evaluation (`scripts/evaluate_transfer.py`)

**Purpose**: Measure generalization to held-out detectors

**Approach**:
1. Train on in-ensemble detectors (e.g., Fast-DetectGPT + Ghostbuster)
2. Evaluate on both in-ensemble and held-out (e.g., Binoculars)
3. Compute transfer ratio: ASR_held_out / ASR_in_ensemble

**Usage**:
```bash
python scripts/evaluate_transfer.py \
  --checkpoint outputs/runs/transfer_model/final.ckpt \
  --data-path data/tinker_full_esl40_nodup \
  --in-ensemble fast_detectgpt ghostbuster \
  --held-out binoculars \
  --output-dir outputs/transfer_eval
```

**Target**: Transfer ratio > 0.7 (70% of in-ensemble performance)

---

## Configuration System

### YAML Configuration Files

All training runs use YAML configs in `configs/` directory.

#### Structure

```yaml
# Model settings
model:
  name: "Qwen/Qwen3-4B-Instruct-2507"
  renderer: "qwen3"

# LoRA settings
lora:
  rank: 32
  alpha: 32
  dropout: 0.05
  target_modules: null  # All layers

# Training hyperparameters
training:
  learning_rate: 2.8e-4
  batch_size: 16
  group_size: 4
  num_epochs: 2
  max_tokens: 512

# Sampling settings
sampling:
  temperature: 1.0
  temperature_schedule: "constant"
  top_p: 0.95

# GRPO settings
grpo:
  normalize_advantages: true
  advantage_clip: 10.0
  reward_clip: null
  remove_constant_reward_groups: true

# KL penalty
kl:
  penalty_coef: 0.01
  target: null

# Reward weights
reward:
  detector_weight: 1.0
  semantic_weight: 0.3
  quality_weight: 0.1
  fairness_penalty: 0.2

# Detector ensemble
detectors:
  - name: "fast_detectgpt"
    weight: 0.5
    enabled: true
  - name: "ghostbuster"
    weight: 0.5
    enabled: true
  - name: "binoculars"
    weight: 0.0  # Disabled (or use for transfer)
    enabled: false

# Evaluation settings
evaluation:
  compute_interval: 10  # Steps between evaluations
  metrics:
    - "asr"
    - "auroc"
    - "f1"
    - "semantic_similarity"
    - "perplexity"
```

#### Available Configurations

1. **`tinker_stealthrl.yaml`** - Full production training
   - 3-detector ensemble
   - Full reward function
   - Optimized hyperparameters

2. **`tinker_stealthrl_ultrafast.yaml`** - Quick proof-of-concept
   - 50 steps, 800 samples, 1 epoch
   - Faster convergence for testing
   - Same architecture, scaled down

3. **`tinker_transfer_in_ensemble.yaml`** - Transfer evaluation
   - Train on 2 detectors only
   - Evaluate on all 3 (measure transfer)

4. **`stealthrl_small.yaml`** - Small-scale testing
   - Minimal resources
   - Quick validation

5. **Ablation Configs** (`configs/ablations/`)
   - `detector_only.yaml`: No semantic/quality penalties
   - `no_fairness.yaml`: No ESL fairness term
   - `no_quality.yaml`: No perplexity penalty
   - `no_semantic.yaml`: No semantic similarity
   - `single_detector.yaml`: Only one detector

### Creating Custom Configurations

```bash
# Copy base config
cp configs/tinker_stealthrl.yaml configs/my_experiment.yaml

# Edit settings
nano configs/my_experiment.yaml

# Run with custom config
python -m stealthrl.tinker.train \
  --config configs/my_experiment.yaml \
  --data-path data/tinker_full_esl40_nodup \
  --run-name my_experiment
```

---

## Data Pipeline

### Data Format

**Required JSONL Format**:
```json
{
  "ai_text": "AI-generated paraphrase or text",
  "human_reference": "Original human-written text",
  "domain": "academic",
  "is_esl": true,
  "metadata": {
    "source": "toefl11",
    "native_language": "chinese",
    "proficiency": "intermediate"
  }
}
```

**Fields**:
- `ai_text`: Input for training (what model sees)
- `human_reference`: Ground truth for semantic similarity
- `domain`: Text domain (academic, news, creative, etc.)
- `is_esl`: Boolean flag for ESL fairness computation
- `metadata`: Optional additional information

### Data Preparation

#### Synthetic Data (for testing)
```bash
python scripts/prepare_tinker_data.py \
  --synthetic \
  --num-train 1000 \
  --num-test 200 \
  --output-dir data/tinker_test
```

#### Real Datasets
```bash
# Download datasets
bash scripts/download_datasets.sh

# Build full dataset (40% ESL, 60% native)
python scripts/build_full_dataset.py \
  --raw-dir data/raw \
  --output-dir data/tinker_full_esl40_nodup \
  --esl-percent 40 \
  --detectrl-max 20000
```

### Dataset Structure

```
data/
├── raw/                        # Raw downloaded datasets
│   ├── toefl11/               # ESL corpus
│   ├── icnale/                # ESL corpus
│   ├── detectrl/              # AI detection datasets
│   └── academic_papers/       # Native corpus
│
├── tinker_full_esl40_nodup/   # Processed dataset
│   ├── train.jsonl            # Training split
│   ├── test.jsonl             # Test split
│   └── metadata.json          # Dataset statistics
│
└── tinker_test/               # Synthetic test data
    ├── train.jsonl
    └── test.jsonl
```

### ESL Data Sources

**Recommended ESL Corpora**:
1. **TOEFL11**: ESL student essays (11 native languages)
2. **ICNALE**: Asian learners' English corpus
3. **ELLIPSE**: Educational learner corpus

**Target Distribution**: 40% ESL, 60% native (reflects academic settings)

---

## Key Scripts

### Training Scripts

#### `stealthrl/tinker/train.py`
**Main training loop with GRPO**

```bash
python -m stealthrl.tinker.train \
  --config configs/tinker_stealthrl.yaml \
  --data-path data/tinker_full_esl40_nodup \
  --run-name my_experiment \
  --num-epochs 3 \
  --batch-size 64 \
  --group-size 16
```

### Data Scripts

#### `scripts/prepare_tinker_data.py`
**Prepare training data**

```bash
# Synthetic data
python scripts/prepare_tinker_data.py \
  --synthetic \
  --num-train 1000 \
  --output-dir data/tinker_test

# Real data
python scripts/prepare_tinker_data.py \
  --input-paths data/raw/*.jsonl \
  --output-dir data/tinker \
  --train-split 0.8
```

#### `scripts/build_full_dataset.py`
**Combine multiple datasets**

```bash
python scripts/build_full_dataset.py \
  --raw-dir data/raw \
  --output-dir data/tinker_full_esl40_nodup \
  --esl-percent 40 \
  --detectrl-max 20000
```

### Evaluation Scripts

#### `scripts/run_stealthbench.py`
**Comprehensive evaluation**

```bash
python scripts/run_stealthbench.py \
  --checkpoint outputs/runs/my_run/checkpoint_50.json \
  --data-path data/tinker_full_esl40_nodup \
  --output-dir outputs/evaluation
```

#### `scripts/analyze_detector_fairness.py`
**ESL fairness analysis**

```bash
python scripts/analyze_detector_fairness.py \
  --data-path data/tinker_full_esl40_nodup \
  --split test \
  --max-samples 500 \
  --output-json outputs/fairness/results.json
```

#### `scripts/evaluate_transfer.py`
**Transfer evaluation**

```bash
python scripts/evaluate_transfer.py \
  --checkpoint outputs/runs/transfer/final.ckpt \
  --in-ensemble fast_detectgpt ghostbuster \
  --held-out binoculars \
  --output-dir outputs/transfer_eval
```

### Visualization Scripts

#### `scripts/visualize_training_results.py`
**Generate training visualizations**

```bash
python scripts/visualize_training_results.py \
  --run-dir outputs/runs/my_experiment \
  --output-dir outputs/runs/my_experiment/visualizations
```

**Generates**:
- Training curves (reward, KL, detector scores)
- Pareto frontiers (quality vs stealth trade-offs)
- Reward decomposition (per-component contributions)
- Stability metrics (variance over time)

### Monitoring Scripts

#### `scripts/monitor_training.py`
**Real-time training monitor**

```bash
python scripts/monitor_training.py \
  --run-dir outputs/runs/my_experiment \
  --refresh-interval 30
```

### Utility Scripts

#### `scripts/paraphrase_example.py`
**Simple inference demo**

```bash
python scripts/paraphrase_example.py \
  --checkpoint outputs/runs/my_run/checkpoint_50.json \
  --text "Your input text here" \
  --output paraphrased.txt
```

#### `scripts/compare_detectors.py`
**Compare detector scores**

```bash
python scripts/compare_detectors.py \
  --text "Input text" \
  --detectors fast_detectgpt ghostbuster binoculars
```

---

## Development Workflow

### Setup New Development Environment

```bash
# Clone repository
git clone https://github.com/your-org/StealthRL.git
cd StealthRL

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
nano .env  # Add TINKER_API_KEY=your_key_here

# Quick validation
python -c "import stealthrl; print('Import successful')"
python scripts/prepare_tinker_data.py --synthetic --num-train 10 --output-dir data/test
```

### Testing Changes

```bash
# Test data pipeline
python scripts/prepare_tinker_data.py \
  --synthetic --num-train 100 --output-dir data/test

# Test training (1 epoch)
python -m stealthrl.tinker.train \
  --config configs/stealthrl_small.yaml \
  --data-path data/test \
  --run-name test_run \
  --num-epochs 1 \
  --batch-size 4

# Test evaluation
python scripts/run_stealthbench.py \
  --checkpoint outputs/runs/test_run/final.ckpt \
  --data-path data/test \
  --output-dir outputs/test_eval \
  --max-samples 50
```

### Adding New Detectors

1. **Create detector class** in `stealthrl/detectors/new_detector.py`:
```python
from .base_detector import BaseDetector
import torch

class NewDetector(BaseDetector):
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__(model_name, device)
        
    def load_model(self):
        # Load your detector model
        pass
    
    def detect(self, texts: Union[str, List[str]]) -> torch.Tensor:
        # Implement detection logic
        # Return scores [0, 1] where 1 = AI-generated
        pass
```

2. **Register in detector ensemble** (`stealthrl/tinker/detectors.py`):
```python
from stealthrl.detectors.new_detector import NewDetector

DETECTOR_REGISTRY = {
    "fast_detectgpt": FastDetectGPTDetector,
    "ghostbuster": GhostbusterDetector,
    "binoculars": BinocularsDetector,
    "new_detector": NewDetector  # Add here
}
```

3. **Add to config** (`configs/tinker_stealthrl.yaml`):
```yaml
detectors:
  - name: "new_detector"
    weight: 0.33
    enabled: true
```

4. **Test**:
```bash
python scripts/compare_detectors.py \
  --text "Test text" \
  --detectors new_detector
```

### Adding New Reward Components

1. **Implement reward function** in `stealthrl/tinker/reward.py`:
```python
async def compute_new_reward(
    text: str,
    reference: str,
    config: RewardConfig
) -> float:
    """Compute new reward component."""
    # Your logic here
    return score  # [0, 1] or [-1, 1]
```

2. **Add to reward computation**:
```python
async def compute_reward(...):
    # Existing components
    detector_reward = ...
    semantic_reward = ...
    quality_reward = ...
    
    # New component
    new_reward = await compute_new_reward(text, reference, config)
    
    # Combine
    total_reward = (
        config.detector_weight * detector_reward +
        config.semantic_weight * semantic_reward +
        config.quality_weight * quality_reward +
        config.new_weight * new_reward  # Add weight to config
    )
    
    return {
        "total": total_reward,
        "detector": detector_reward,
        "semantic": semantic_reward,
        "quality": quality_reward,
        "new_component": new_reward  # Log separately
    }
```

3. **Update config dataclass**:
```python
@dataclass
class RewardConfig:
    detector_weight: float = 1.0
    semantic_weight: float = 0.3
    quality_weight: float = 0.1
    new_weight: float = 0.2  # Add new weight
```

4. **Test**:
```bash
# Create ablation config without new component
cp configs/tinker_stealthrl.yaml configs/ablations/no_new_component.yaml
# Set new_weight: 0.0

# Compare with and without
python scripts/evaluate_ablations.py \
  --configs configs/tinker_stealthrl.yaml configs/ablations/no_new_component.yaml
```

---

## Common Tasks

### Task 1: Run Quick Proof-of-Concept

```bash
# 1. Prepare small dataset
python scripts/prepare_tinker_data.py \
  --synthetic --num-train 800 --num-test 200 \
  --output-dir data/poc

# 2. Train ultrafast (50 steps, ~3.5 hours)
python -m stealthrl.tinker.train \
  --config configs/tinker_stealthrl_ultrafast.yaml \
  --data-path data/poc \
  --run-name poc_run \
  --num-epochs 1

# 3. Visualize results
python scripts/visualize_training_results.py \
  --run-dir outputs/runs/poc_run \
  --output-dir outputs/runs/poc_run/visualizations

# 4. Check Pareto-optimal checkpoints
cat outputs/runs/poc_run/pareto_checkpoints.json
```

### Task 2: Full Production Training

```bash
# 1. Build full dataset (if not done)
python scripts/build_full_dataset.py \
  --raw-dir data/raw \
  --output-dir data/tinker_full_esl40_nodup \
  --esl-percent 40 \
  --detectrl-max 20000

# 2. Train (6-8 hours)
python -m stealthrl.tinker.train \
  --config configs/tinker_stealthrl.yaml \
  --data-path data/tinker_full_esl40_nodup \
  --run-name production_run_v1 \
  --num-epochs 3 \
  --batch-size 64 \
  --group-size 16

# 3. Monitor in real-time (separate terminal)
python scripts/monitor_training.py \
  --run-dir outputs/runs/production_run_v1 \
  --refresh-interval 30

# 4. Comprehensive evaluation
python scripts/run_stealthbench.py \
  --checkpoint outputs/runs/production_run_v1/final.ckpt \
  --data-path data/tinker_full_esl40_nodup \
  --output-dir outputs/evaluation/production_v1
```

### Task 3: Transfer Evaluation

```bash
# 1. Train on 2 detectors
python -m stealthrl.tinker.train \
  --config configs/tinker_transfer_in_ensemble.yaml \
  --data-path data/tinker_full_esl40_nodup \
  --run-name transfer_experiment

# 2. Evaluate transfer
python scripts/evaluate_transfer.py \
  --checkpoint outputs/runs/transfer_experiment/final.ckpt \
  --data-path data/tinker_full_esl40_nodup \
  --in-ensemble fast_detectgpt ghostbuster \
  --held-out binoculars \
  --output-dir outputs/transfer_eval

# 3. Check transfer ratio
python -c "
import json
with open('outputs/transfer_eval/metrics.json') as f:
    metrics = json.load(f)
    asr_in = metrics['asr_in_ensemble']
    asr_out = metrics['asr_held_out']
    ratio = asr_out / asr_in
    print(f'Transfer Ratio: {ratio:.3f} (target: >0.7)')
"
```

### Task 4: ESL Fairness Analysis

```bash
# 1. Analyze baseline (before training)
python scripts/analyze_detector_fairness.py \
  --data-path data/tinker_full_esl40_nodup \
  --split test \
  --text-field human_reference \
  --max-samples 500 \
  --output-json outputs/fairness/baseline.json

# 2. Train with fairness penalty
python -m stealthrl.tinker.train \
  --config configs/tinker_stealthrl.yaml \
  --data-path data/tinker_full_esl40_nodup \
  --run-name fairness_experiment

# 3. Analyze after training
python scripts/analyze_fairness_after_training.py \
  --checkpoint outputs/runs/fairness_experiment/final.ckpt \
  --data-path data/tinker_full_esl40_nodup \
  --baseline-json outputs/fairness/baseline.json \
  --output-json outputs/fairness/after_training.json

# 4. Compare gaps
python -c "
import json
with open('outputs/fairness/baseline.json') as f:
    baseline = json.load(f)
with open('outputs/fairness/after_training.json') as f:
    after = json.load(f)

for detector in baseline:
    gap_before = baseline[detector]['esl_gap']
    gap_after = after[detector]['esl_gap']
    improvement = (gap_before - gap_after) / gap_before * 100
    print(f'{detector}: {gap_before:.3f} → {gap_after:.3f} ({improvement:.1f}% improvement)')
"
```

### Task 5: Ablation Studies

```bash
# 1. Run all ablations (parallel if multiple GPUs)
bash scripts/run_ablations.sh

# Or individually:
for config in configs/ablations/*.yaml; do
    name=$(basename $config .yaml)
    python -m stealthrl.tinker.train \
      --config $config \
      --data-path data/tinker_full_esl40_nodup \
      --run-name ablation_$name \
      --num-epochs 2
done

# 2. Evaluate all ablations
python scripts/evaluate_ablations.py \
  --run-dirs outputs/runs/ablation_* \
  --output-dir outputs/ablation_analysis

# 3. Generate comparison plots
python scripts/visualize_ablations.py \
  --ablation-dir outputs/ablation_analysis \
  --output-dir outputs/ablation_analysis/plots
```

### Task 6: Export Model for Inference

```bash
# 1. Export checkpoint
python scripts/export_model.py \
  --checkpoint outputs/runs/my_run/checkpoint_50.json \
  --output-dir models/exported/my_run_step50

# 2. Test inference
python scripts/paraphrase_example.py \
  --model-dir models/exported/my_run_step50 \
  --text "Your input text here" \
  --temperature 0.8 \
  --max-tokens 512

# 3. Batch inference
python scripts/paraphrase_example.py \
  --model-dir models/exported/my_run_step50 \
  --input-file inputs.txt \
  --output-file outputs.txt \
  --batch-size 16
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'stealthrl'
# Fix: Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/StealthRL"
```

#### 2. Tinker API Errors
```bash
# Error: TinkerAPIError: Invalid API key
# Fix: Check .env file
cat .env | grep TINKER_API_KEY

# Error: TinkerAPIError: Rate limit exceeded
# Fix: Wait or upgrade Tinker plan

# Error: TinkerAPIError: Model not found
# Fix: Check model name in config
# Valid: "Qwen/Qwen3-4B-Instruct-2507"
```

#### 3. CUDA Out of Memory
```bash
# Error: RuntimeError: CUDA out of memory
# Fix: Reduce batch size
python -m stealthrl.tinker.train \
  --config configs/tinker_stealthrl.yaml \
  --batch-size 4  # Reduce from 16

# Or use CPU (slower)
python -m stealthrl.tinker.train \
  --config configs/tinker_stealthrl.yaml \
  --device cpu
```

#### 4. Data Loading Errors
```bash
# Error: FileNotFoundError: data/tinker/train.jsonl
# Fix: Prepare data first
python scripts/prepare_tinker_data.py \
  --synthetic --num-train 1000 --output-dir data/tinker

# Error: JSONDecodeError
# Fix: Validate JSONL format
python -c "
import jsonlines
with jsonlines.open('data/tinker/train.jsonl') as f:
    for i, line in enumerate(f):
        try:
            assert 'ai_text' in line
            assert 'human_reference' in line
        except AssertionError:
            print(f'Invalid format at line {i+1}')
"
```

#### 5. Checkpoint Loading Errors
```bash
# Error: FileNotFoundError: checkpoint not found
# Fix: Check checkpoint path
ls outputs/runs/my_run/checkpoints/

# Error: TinkerAPIError: Checkpoint expired
# Fix: Checkpoints expire after 30 days on Tinker free tier
# Solution: Re-train or use local checkpoints
```

#### 6. Detector Errors
```bash
# Error: DetectorError: Model not loaded
# Fix: Load detector first
python -c "
from stealthrl.detectors.fast_detectgpt import FastDetectGPTDetector
detector = FastDetectGPTDetector()
detector.load_model()
score = detector.detect('Test text')
print(f'Score: {score}')
"

# Error: RuntimeError: Expected CUDA device
# Fix: Use CPU for detectors if no GPU
detector = FastDetectGPTDetector(device='cpu')
```

### Debugging Tips

#### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('stealthrl')
logger.setLevel(logging.DEBUG)
```

#### Check Training Progress
```bash
# View latest logs
tail -f outputs/runs/my_run/logs/train.log

# Check TensorBoard
tensorboard --logdir outputs/runs/my_run/tensorboard

# Monitor metrics
watch -n 30 "tail -20 outputs/runs/my_run/logs/metrics.jsonl"
```

#### Validate Data Quality
```bash
python scripts/validate_datasets.py \
  --data-path data/tinker_full_esl40_nodup \
  --split train \
  --max-samples 1000
```

#### Profile Performance
```bash
# Profile training step
python -m cProfile -o profile.stats -m stealthrl.tinker.train \
  --config configs/stealthrl_small.yaml \
  --data-path data/test \
  --num-epochs 1

# View results
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"
```

---

## Research Context

### Research Questions

**RQ1: Multi-Detector Generalization**
- **Question**: Does ensemble training transfer to unseen detector families?
- **Approach**: Train on 2 detectors, test on held-out 3rd
- **Metric**: Transfer ratio = ASR_held_out / ASR_in_ensemble
- **Target**: >0.7 (70% transfer)
- **Status**: Infrastructure ready, awaiting full production run

**RQ2: Reward Component Necessity**
- **Question**: Which reward components are critical?
- **Approach**: 5 ablation studies (remove each component)
- **Configs**: `configs/ablations/*.yaml`
- **Status**: Configs ready, awaiting execution

**RQ3: Fairness in Adversarial RL**
- **Question**: Can explicit ESL penalty reduce FPR gaps?
- **Approach**: Compare FPR(ESL) - FPR(native) with/without fairness term
- **Target**: Gap <0.07 (vs baseline ~0.15)
- **Status**: Dataset ready (40% ESL), evaluation framework complete

### Novel Contributions

1. **Generalizable Multi-Detector Framework**: First system using locally hosted open-source detectors (vs AuthorMist's API dependency)

2. **Fairness-Aware Adversarial Training**: First explicit ESL penalty in adversarial NLP

3. **Open-Source Release**: Full codebase + 9 Pareto-optimal checkpoints + configs

4. **First GRPO Application to Adversarial NLP**: Group-based RL for text transformation

5. **Multi-Objective Pareto Optimization**: Automated identification of optimal checkpoints

6. **Comprehensive Evaluation Infrastructure**: StealthBench + ESL fairness + transfer evaluation

### Key Results (Ultra-Fast Proof-of-Concept)

- **22% detector evasion improvement** (58.7% → 45.8%)
- **98.6% semantic similarity maintained** (never below 94%)
- **No model collapse** (parse success: 85.9% → 99.2%)
- **Stable KL divergence** (<0.4, target <4.0)
- **9 Pareto-optimal checkpoints** identified

### Future Work

1. **Additional Detectors**: Integrate GPTZero, OpenAI detector, Winston AI
2. **Larger Models**: Scale to Llama-3.1-70B, Qwen-72B
3. **Multi-lingual Fairness**: Extend ESL analysis to specific language groups
4. **Curriculum Learning**: Progressive difficulty in detector ensemble
5. **Human Evaluation**: Turing test validation
6. **Real-World Deployment**: API service for paraphrasing

---

## Appendix: File Index

### Python Modules

#### Core Training (`stealthrl/tinker/`)
- `train.py` (1040 lines) - Main training loop
- `reward.py` (400 lines) - Multi-objective reward
- `detectors.py` (350 lines) - Detector ensemble
- `semantic.py` (200 lines) - Semantic similarity
- `perplexity.py` (150 lines) - Quality control
- `evaluation.py` (300 lines) - Metrics computation
- `dataset.py` (250 lines) - Data loading
- `inference.py` (200 lines) - Model inference
- `env.py` (50 lines) - Environment variables

#### Detectors (`stealthrl/detectors/`)
- `base_detector.py` (100 lines) - Abstract base
- `fast_detectgpt.py` (250 lines) - Curvature detection
- `ghostbuster.py` (200 lines) - Classifier detection
- `binoculars.py` (200 lines) - Paired-LM detection

#### Baselines (`stealthrl/baselines/`)
- `authormist.py` - AuthorMist reimplementation
- `paraphrase_baselines.py` - Simple baselines

### Scripts (`scripts/`)

#### Training
- `train_stealthrl.py` - Legacy training script
- `run_research_pipeline.py` - Automated pipeline
- `run_ultrafast_training.py` - Quick experiments
- `monitor_training.py` - Real-time monitor

#### Data
- `prepare_tinker_data.py` - Data preparation
- `build_full_dataset.py` - Combine datasets
- `extract_detectrl_data.py` - Extract DetectRL
- `convert_chatgpt_bias_data.py` - Convert ChatGPT bias data
- `validate_datasets.py` - Data validation

#### Evaluation
- `run_stealthbench.py` - Comprehensive evaluation
- `evaluate_transfer.py` - Transfer evaluation
- `evaluate_ablations.py` - Ablation analysis
- `analyze_detector_fairness.py` - ESL fairness
- `analyze_fairness_after_training.py` - Post-training fairness
- `compare_detectors.py` - Detector comparison
- `compare_baselines.py` - Baseline comparison

#### Visualization
- `visualize_training_results.py` - Training plots
- `visualize_stealthbench.py` - Evaluation plots

#### Utilities
- `paraphrase_example.py` - Simple inference
- `export_model.py` - Export checkpoints
- `test_detectors.py` - Test detector implementations
- `cancel_tinker_runs.py` - Cancel Tinker jobs

### Documentation (`knowledge_base/`)

#### Onboarding
- `README.md` - Documentation index
- `TEAM_HANDOFF.md` - Team onboarding
- `QUICKSTART.md` - Fast-track guide
- `SETUP_AND_RUN.md` - Complete setup

#### Technical
- `DETECTOR_SETUP.md` - Detector configuration
- `ESL_FAIRNESS_GUIDE.md` - Fairness evaluation
- `CHECKPOINT_GUIDE.md` - Checkpoint management
- `CHECKPOINT_IMPLEMENTATION.md` - Checkpoint technical details
- `REWARD_REFINEMENT.md` - Reward function design
- `FINAL_RUN_HYPERPARAMETERS.md` - Optimized hyperparams

#### Research
- `RESEARCH_ROADMAP.md` - Research plan
- `NEXT_STEPS.md` - TODOs
- `DATA_CURATION_ANALYSIS.md` - Data requirements
- `ULTRAFAST_VS_FULL_COMPARISON.md` - Training comparison

#### Operations
- `RUN_MANAGEMENT.md` - Training run management
- `TINKER_README.md` - Tinker platform guide
- `IMPLEMENTATION_VERIFICATION.md` - Verification checklist

#### Tasks
- `TASK1_README.md` - Detector implementation task
- `TASK2_README.md` - Data curation task
- `TASK3_READINESS_REPORT.md` - Training readiness
- `TASK3_PIPELINE_READINESS.md` - Pipeline validation

---

## Quick Command Reference

```bash
# Setup
pip install -r requirements.txt
export TINKER_API_KEY="your_key"

# Data
python scripts/build_full_dataset.py --raw-dir data/raw --output-dir data/tinker_full_esl40_nodup --esl-percent 40

# Train
python -m stealthrl.tinker.train --config configs/tinker_stealthrl.yaml --data-path data/tinker_full_esl40_nodup --run-name my_run

# Evaluate
python scripts/run_stealthbench.py --checkpoint outputs/runs/my_run/final.ckpt --data-path data/tinker_full_esl40_nodup

# Fairness
python scripts/analyze_detector_fairness.py --data-path data/tinker_full_esl40_nodup --output-json outputs/fairness/results.json

# Visualize
python scripts/visualize_training_results.py --run-dir outputs/runs/my_run

# Inference
python scripts/paraphrase_example.py --checkpoint outputs/runs/my_run/final.ckpt --text "Your text"
```

---

## Contact & Resources

- **Documentation**: `knowledge_base/README.md`
- **Main README**: `README.md`
- **Report**: `report/REPORT.md`
- **Quick Commands**: `atharv_readme.md`
- **Tinker Platform**: https://tinker.thinkingmachines.ai/

---

**End of AGENT.md**
