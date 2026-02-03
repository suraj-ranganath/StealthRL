# StealthRL Research Evaluation Pipeline

## Comprehensive Documentation

This document provides a complete description of the StealthRL evaluation pipeline for comparing AI text detection evasion methods.

**Last Updated**: January 31, 2026

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Input Data](#input-data)
3. [Attack Methods](#attack-methods)
4. [Detectors](#detectors)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Output Structure](#output-structure)
7. [Generated Figures](#generated-figures)
8. [Generated Tables](#generated-tables)
9. [Running the Pipeline](#running-the-pipeline)
10. [Tradeoffs for M4 Mac](#tradeoffs-for-m4-mac)
11. [Key Bug Fixes and Implementation Notes](#key-bug-fixes-and-implementation-notes)

**Appendices:**
- [Appendix A: RAID Dataset](#appendix-a1-raid-dataset-optional-multi-domain-generalization)
- [Appendix B: Ablation Studies](#appendix-b-ablation-studies)
- [Appendix C: Checkpoint Comparison](#appendix-c-checkpoint-comparison) ⭐ **NEW**
- [Appendix D: PadBen Task Configurations](#appendix-d-padben-task-configurations)
- [Appendix E: Color Scheme](#appendix-e-color-scheme)

---

## 1. Pipeline Overview

The evaluation pipeline benchmarks **StealthRL** (our RL-trained paraphrasing method) against baseline attack methods on AI text detection. The pipeline:

1. **Loads evaluation datasets** (MAGE, RAID, or PadBen)
2. **Applies attack methods** (M0-M5) to AI-generated text
3. **Scores attacked text** with multiple detectors
4. **Calibrates detection thresholds** on human samples (at 1% FPR)
5. **Computes evaluation metrics** with bootstrap confidence intervals
6. **Computes text quality metrics** (similarity, perplexity, edit rate)
7. **Generates paper-ready figures and tables**
8. **Saves all artifacts** to structured output directory

### Pipeline Flow

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│ Dataset Loading │ -> │ Threshold    │ -> │ Attack Methods  │ -> │ Detection    │
│ (MAGE/RAID/     │    │ Calibration  │    │ (M0-M5)         │    │ Scoring      │
│  PadBen)        │    │ (on human)   │    │                 │    │              │
└─────────────────┘    └──────────────┘    └─────────────────┘    └──────────────┘
                                                                          │
                                                                          ▼
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│ Save Artifacts  │ <- │ Generate     │ <- │ Quality Metrics │ <- │ Compute      │
│ (CSV, Parquet,  │    │ Figures &    │    │ (sim, ppl,      │    │ Metrics      │
│  JSON)          │    │ Tables       │    │  edit rate)     │    │ (AUROC, etc) │
└─────────────────┘    └──────────────┘    └─────────────────┘    └──────────────┘
```

### Core Pipeline Implementation

The pipeline is implemented in `eval/runner.py` as the `EvalRunner` class with these key methods:

| Method | Description |
|--------|-------------|
| `load_datasets()` | Load evaluation datasets with balanced sampling |
| `load_detectors()` | Initialize detector panel with lazy loading |
| `load_methods()` | Initialize attack methods |
| `calibrate_detector_thresholds()` | Calibrate thresholds on human samples at 1% FPR |
| `run_attacks()` | Apply attack methods to AI samples |
| `score_outputs()` | Score all outputs with detector panel |
| `compute_all_metrics()` | Compute AUROC, TPR@1%FPR, ASR with bootstrap CIs |
| `compute_quality_metrics()` | Compute similarity, perplexity, edit metrics |
| `save_all_artifacts()` | Save all outputs to disk |

---

## 2. Input Data

### Datasets Supported

| Dataset | Source | Description | Role in Pipeline |
|---------|--------|-------------|------------------|
| **MAGE** | `yaful/DeepfakeTextDetect` | AI-generated text detection benchmark | **Primary benchmark** (default) |
| **RAID** | `liamdugan/raid` | Robustness of AI Detectors (~11GB) | **Optional** - multi-domain generalization |
| **PadBen** | `JonathanZha/PADBen` | Paraphrase attack benchmark (5 tasks) | **Ablation studies** - paraphrase-specific |

### MAGE Dataset (Primary)

**CRITICAL**: MAGE label encoding was verified and fixed:
- `label=1` → **Human-written text**
- `label=0` → **AI-generated text**

This was verified by examining the `src` field:
- `src` values like `"gpt2-xl"`, `"davinci"`, `"gpt-3.5-turbo"` → AI-generated
- `src` values like `"human"`, `null` → Human-written

**Why MAGE is the Primary Dataset**:
1. **Standard benchmark**: Most cited in AI detection literature
2. **Lightweight**: Fast to download and process
3. **Training alignment**: StealthRL was developed/tuned on MAGE-like distributions
4. **Balanced**: Clean human/AI splits with domain labels

### Why RAID is Optional (Not Default)

| Reason | Details |
|--------|---------|
| **Size** | ~11GB download, requires streaming or subset |
| **Hidden test labels** | Official test split has no labels; must use `train` split |
| **Complexity** | 11 domains, 11 generators, adversarial variants |
| **Use case** | Best for multi-domain generalization studies (not core experiments) |

**When to use RAID**: Testing cross-domain transfer (e.g., "Does StealthRL trained on news generalize to code/recipes?")

### Why PadBen is for Ablation Studies

PadBen is specifically designed for **paraphrase attack evaluation** with 5 distinct tasks that test different aspects:

| Task | Config | What It Tests | Relevance to StealthRL |
|------|--------|---------------|------------------------|
| **Task 1** | `exhaustive-task1` | Human vs LLM paraphrases | Can detectors tell WHO paraphrased? |
| **Task 2** | `exhaustive-task2` | Human vs LLM-generated (default) | Standard detection (same as MAGE) |
| **Task 3** | `exhaustive-task3` | Detecting paraphrase depth | Does multi-pass paraphrasing help evasion? |
| **Task 4** | `exhaustive-task4` | 1st vs 3rd iteration paraphrase | Diminishing returns of iterative attacks? |
| **Task 5** | `exhaustive-task5` | Human vs heavily paraphrased AI | **Most relevant**: Tests deep paraphrase attacks |

#### Recommended Ablation Studies with PadBen

1. **Transfer evaluation**: Does StealthRL (trained on MAGE) generalize to PadBen?
   ```bash
   --datasets mage padben --methods m0 m2
   ```

2. **Paraphrase depth analysis** (Task 3 vs Task 5): How much does iterative paraphrasing help?
   ```bash
   # Compare detection on 1st-pass vs 3rd-pass paraphrased text
   python -m eval.runner_enhanced --datasets padben --padben-config exhaustive-task5
   ```

3. **Method comparison on paraphrase-specific benchmark**: Is StealthRL better than simple paraphrasing (M1) on PadBen?

### Sample Schema

Each evaluation sample has the following structure:

```python
@dataclass
class EvalSample:
    id: str                          # Unique sample identifier
    label: Literal["human", "ai"]    # Ground truth label
    domain: str                      # Content domain (e.g., "news", "wiki")
    generator: Optional[str]         # AI model that generated the text
    text: str                        # The actual text content
    metadata: Dict[str, Any]         # Additional metadata
```

### Data Preprocessing

1. **Length filtering**: Samples with 100-500 tokens (approx. via word count × 1.3)
2. **Balancing**: Equal number of human and AI samples
3. **Shuffling**: Random seed for reproducibility

### Example Command - Data Loading

```bash
# Default: 1000 human + 1000 AI samples from MAGE
python -m eval.runner_enhanced --datasets mage --n-human 1000 --n-ai 1000
```

---

## 3. Attack Methods

### Method Registry

| Method ID | Name | Description | Implementation Class |
|-----------|------|-------------|---------------------|
| **M0** | No Attack | Original AI text (baseline) | `NoAttack` |
| **M1** | Simple Paraphrase | Ollama qwen3:4b paraphrasing | `SimpleParaphrase` |
| **M2** | StealthRL | RL-trained policy via Tinker API | `StealthRLTinker` |
| **M3** | Adversarial Paraphrasing | Detector-guided candidate selection | `AdversarialParaphrasing` |
| **M4** | AuthorMist | Style transfer via Ollama | `AuthorMistOllama` |
| **M5** | Homoglyph | SilverSpeak character substitution | `HomoglyphAttack` |

### Method Aliases

The method registry supports multiple names for each method:

```python
METHOD_REGISTRY = {
    "m0": NoAttack,
    "no_attack": NoAttack,
    "m1": SimpleParaphrase,
    "simple_paraphrase": SimpleParaphrase,
    "m2": StealthRLTinker,
    "stealthrl": StealthRLTinker,
    "m2_local": StealthRLAttack,  # Local PEFT version
    "m3": AdversarialParaphrasing,
    "adversarial_paraphrasing": AdversarialParaphrasing,
    "m3_ensemble": AdversarialParaphrasingEnsemble,
    "m4": AuthorMistOllama,
    "authormist": AuthorMistOllama,
    "m4_hf": AuthorMist,  # HuggingFace version
    "m5": HomoglyphAttack,
    "homoglyph": HomoglyphAttack,
}
```

### Guidance Variants (for Transfer Ablation Study)

```python
GUIDANCE_VARIANTS = {
    "m3_roberta": ("roberta", "Guided by RoBERTa detector"),
    "m3_fastdetect": ("fast_detectgpt", "Guided by Fast-DetectGPT"),
    "m3_ensemble": ("ensemble", "Guided by ensemble of detectors"),
}
```

### Method Details

#### M0: No Attack (Baseline)
- **Input**: Original AI-generated text
- **Output**: Same text unchanged
- **Purpose**: Establishes baseline detection rates

#### M1: Simple Paraphrase
- **Backend**: Ollama (`qwen3:4b-instruct`)
- **Prompt**: "Please paraphrase the following text while maintaining its meaning and style..."
- **Parameters**: temperature=0.9, top_p=0.95, max_tokens=512
- **Reranking**: Supports optional detector-guided reranking with lazy loading

#### M2: StealthRL (Our Method)
- **Backend**: Tinker cloud API (OpenAI-compatible endpoint)
- **Model**: RL fine-tuned Qwen3-4B via GRPO
- **Checkpoint**: JSON file with model_id and base_model info
- **Batched Inference**: Uses `num_samples` parameter for batched generation in single API call
- **Inference URL**: `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1`

#### M3: Adversarial Paraphrasing
- **Approach**: Generate N candidates using M1, select one with lowest detector score
- **Guidance**: Can be guided by specific detector (RoBERTa, Fast-DetectGPT, or ensemble)
- **Lazy Loading**: Guidance detector is loaded lazily only when needed
- **Variants**: `m3_roberta`, `m3_fastdetect`, `m3_ensemble`

#### M4: AuthorMist
- **Backend**: Ollama with GGUF quantized model
- **Model**: `authormist-originality.i1-Q4_K_M.gguf` (1.8GB)
- **Parameters**: temperature=0.7, top_p=0.9, num_predict=512
- **Paper**: ArXiv 2401.12070

#### M5: Homoglyph (SilverSpeak)
- **Library**: `silverspeak` package
- **Method**: Character substitution with visually similar Unicode characters
- **Default Rate**: 10% of characters substituted
- **Paper**: ArXiv 2406.11239

### Attack Output Schema

```python
@dataclass
class AttackOutput:
    text: str                           # Attacked/paraphrased text
    metadata: Dict[str, Any]            # Attack metadata
    all_candidates: List[str]           # All generated candidates (if N>1)
    candidate_scores: List[float]       # Detector scores for each candidate
    original_text: Optional[str]        # Original input text
```

---

## 4. Detectors

### Detector Registry

| Detector | Type | Model(s) | Memory | Purpose |
|----------|------|----------|--------|---------|
| **RoBERTa** | Trained Classifier | `openai-community/roberta-large-openai-detector` | ~1.5GB | Standard trained detector |
| **Fast-DetectGPT** | Zero-shot Statistical | `EleutherAI/gpt-neo-2.7B` | ~5.5GB | Sampling-based detection (paper standard) |
| **Binoculars** | Zero-shot Statistical | `gpt2-medium` + `gpt2-large` (lightweight) | ~4.5GB | Cross-perplexity detection |

### Detector Score Convention

**IMPORTANT**: All detectors output scores where **higher = more likely AI-generated**.

This is enforced via `DETECTOR_CONVENTIONS` in `eval/detectors.py`:

```python
DETECTOR_CONVENTIONS = {
    "roberta": {"higher_is_ai": True},
    "fast_detectgpt": {"higher_is_ai": True},
    "binoculars": {"higher_is_ai": True},
    "detectgpt": {"higher_is_ai": True},
    "ghostbuster": {"higher_is_ai": True},
}
```

### Detector Details

#### RoBERTa OpenAI Detector
- **Architecture**: RoBERTa-large fine-tuned for AI detection
- **Output**: Probability score [0, 1] where higher = more AI
- **HuggingFace**: `openai-community/roberta-large-openai-detector`

#### Fast-DetectGPT
- **Paper**: ArXiv 2310.05130
- **Method**: Conditional probability curvature analysis
- **Sampling**: Perturbation-based scoring
- **Model**: GPT-Neo 2.7B (standard model from paper)
- **Note**: This is the paper-standard implementation

#### Binoculars
- **Paper**: ArXiv 2401.12070
- **Formula**: `score = perplexity / cross_perplexity`
- **Lightweight Mode** (default for M4 Mac):
  - Performer: `gpt2-medium` 
  - Observer: `gpt2-large`
- **Full Mode** (paper-grade accuracy):
  - Performer: `falcon-7b-instruct`
  - Observer: `falcon-7b`
- **Score Inversion**: Raw Binoculars score is inverted (lower = more AI), but the wrapper normalizes this

### Detector Output Schema

```python
@dataclass
class DetectorResult:
    score: float               # Normalized score (higher = more AI)
    raw_score: Optional[float] # Original detector output
    metadata: Dict[str, Any]   # Additional info (perplexity, etc.)
```

---

## 5. Evaluation Metrics

### Primary Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **AUROC** | Area under ROC curve | Overall discrimination ability (lower = better attack) |
| **TPR@1%FPR** | True Positive Rate at 1% False Positive Rate | Detection rate at low false alarm (lower = better attack) |
| **ASR** | `1 - TPR@1%FPR` | Attack Success Rate (higher = better attack) |

### Quality Metrics

| Metric | Column | Range | Description |
|--------|--------|-------|-------------|
| **Similarity** | `sim_e5` | [0, 1] | Semantic similarity via BGE-M3/Ollama (higher = better) |
| **Perplexity** | `ppl_score` | [0, ∞) | Fluency measure (lower = more fluent) |
| **Edit Rate** | `edit_rate` | [0, 1] | Character-level edit distance ratio |
| **Token Change** | `token_change_rate` | [0, 1] | Token-level change rate |
| **ChrF** | `chrf` | [0, 1] | Character n-gram F-score |
| **Length Ratio** | `len_ratio` | (0, ∞) | Output length / Input length |
| **GPT Quality** | `quality_rating` | [1, 5] | GPT-based Likert rating of overall paraphrase quality (optional) |
| **GPT Similarity** | `similarity_rating` | [1, 5] | GPT-based Likert rating of semantic equivalence (optional) |

### Semantic Similarity Details

The pipeline uses **BGE-M3** (via Ollama) for semantic similarity, with E5-base-v2 as fallback:

| Model | Source | Embedding Dim | Accuracy | Memory |
|-------|--------|---------------|----------|--------|
| **BGE-M3** (default) | Ollama `bge-m3:latest` | 1024 | State-of-the-art | 1.2GB (Ollama) |
| E5-base-v2 (fallback) | HuggingFace | 768 | Good | ~440MB |

**Similarity Interpretation:**
| sim_e5 | Quality |
|--------|---------|
| **0.95+** | Near-identical meaning |
| **0.85-0.95** | Good paraphrase |
| **0.70-0.85** | Acceptable, some drift |
| **<0.70** | Significant meaning change |

### Threshold Calibration

Thresholds are calibrated on **human samples only** to achieve target FPR (default: 1%):

```python
# For higher_is_ai detectors:
threshold = np.quantile(human_scores, 1 - target_fpr)

# TPR is then computed on AI samples:
tpr = np.mean(ai_scores > threshold)
```

### Bootstrap Confidence Intervals

All metrics include 95% confidence intervals computed via:
- **N bootstrap samples**: 500 (default, configurable via `--n-bootstrap`)
- **Resampling**: With replacement
- **CI bounds**: 2.5th and 97.5th percentiles

### DetectorMetrics Dataclass

```python
@dataclass
class DetectorMetrics:
    detector: str
    method: str
    dataset: str
    
    # Core metrics with CIs
    auroc: float
    auroc_ci_low: float
    auroc_ci_high: float
    
    tpr_at_1fpr: float
    tpr_at_1fpr_ci_low: float
    tpr_at_1fpr_ci_high: float
    
    asr: float
    asr_ci_low: float
    asr_ci_high: float
    
    # Calibration
    threshold_1fpr: float
    
    # Sample counts
    n_human: int
    n_ai: int
```

---

## 6. Output Structure

Each evaluation run creates a timestamped directory:

```
outputs/eval_runs/{run_name}_{YYYYMMDD}_{HHMMSS}/
├── eval_{timestamp}.log      # Detailed execution log
├── scores.parquet            # Detection scores (primary format)
├── scores.csv                # Detection scores (CSV format)
├── quality.parquet           # Quality metrics (primary format)
├── quality.csv               # Quality metrics (CSV format)
├── quality_gpt.parquet       # GPT-based quality ratings (optional)
├── quality_gpt.csv           # GPT-based quality ratings (optional)
├── metrics.json              # Aggregated metrics with bootstrap CIs
├── thresholds.json           # Calibrated detection thresholds
├── raw_outputs.json          # All attack outputs
├── figures/                  # Generated figures (PNG)
│   ├── fig_heatmap_tpr.png
│   ├── fig_heatmap_auroc.png
│   ├── fig_auroc_bars.png
│   ├── fig_auroc_radar.png
│   ├── fig_method_comparison.png
│   ├── fig_tradeoff.png
│   ├── fig_score_distributions.png
│   ├── fig_human_ai_separation_{detector}.png
│   ├── fig_quality_vs_evasion.png
│   ├── fig_asr_comparison.png
│   └── fig_perplexity_vs_similarity.png
└── tables/                   # Generated tables (MD, CSV)
```

**Default Output Path**: `outputs/eval_runs/` (changed from `artifacts/`)

### Scores DataFrame Schema (`scores.csv` / `scores.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | str | Unique sample identifier (e.g., `mage_test_4774`) |
| `dataset` | str | Source dataset name (`mage`, `raid`, `padben`) |
| `method` | str | Attack method applied (`m0`, `m1`, `m2`, etc.) |
| `label` | str | Ground truth (`"human"` or `"ai"`) |
| `setting` | str | Candidate budget setting (e.g., `"N=1"`) |
| `text_in` | str | Original input text |
| `text_out` | str | Attacked/paraphrased output text |
| `metadata` | dict | Method-specific metadata |
| `detector_name` | str | Detector name (`roberta`, `fast_detectgpt`, `binoculars`) |
| `detector_score` | float | Detection score (higher = more likely AI) |

### Quality DataFrame Schema (`quality.csv` / `quality.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | str | Unique sample identifier |
| `method` | str | Attack method applied |
| `setting` | str | Run setting (e.g., `"default"`) |
| `sim_e5` | float | Semantic similarity (0-1, higher = better) |
| `ppl_score` | float | Perplexity (lower = more fluent) |
| `edit_rate` | float | Character-level edit distance ratio (0-1) |
| `token_change_rate` | float | Token-level change rate (optional) |
| `chrf` | float | Character n-gram F-score (optional) |
| `len_ratio` | float | Length ratio: output/input |
| `len_tokens_in` | int | Input token count |
| `len_tokens_out` | int | Output token count |
| `valid` | bool | Whether output is valid |
| `fail_reason` | str | Failure reason if invalid |

### Metrics JSON Schema (`metrics.json`)

```json
{
  "metrics": [
    {
      "detector": "roberta",
      "method": "m2",
      "dataset": "mage",
      "auroc": 0.5496,
      "auroc_ci_low": 0.4357,
      "auroc_ci_high": 0.6598,
      "tpr_at_1fpr": 0.08,
      "tpr_at_1fpr_ci_low": 0.0175,
      "tpr_at_1fpr_ci_high": 0.1556,
      "asr": 0.92,
      "asr_ci_low": 0.8444,
      "asr_ci_high": 0.9825,
      "threshold_1fpr": 0.5741,
      "n_human": 50,
      "n_ai": 50
    }
  ],
  "thresholds": {
    "roberta": 0.5741,
    "fast_detectgpt": 0.1234,
    "binoculars": 0.5678
  },
  "config": {
    "seed": 42,
    "n_bootstrap": 500,
    "datasets": ["mage"],
    "methods": ["m0", "m1", "m2", "m3", "m4"],
    "detectors": ["roberta", "fast_detectgpt", "binoculars"]
  }
}
```

---

## 7. Generated Figures

The pipeline generates **13 figures** automatically via `generate_all_plots()`:

### Figure List

| Figure | Filename | Description |
|--------|----------|-------------|
| **1. TPR Heatmap** | `fig_heatmap_tpr.png` | Detector × Method heatmap showing TPR@1%FPR |
| **2. AUROC Heatmap** | `fig_heatmap_auroc.png` | Detector × Method heatmap showing AUROC |
| **3. AUROC Bar Chart** | `fig_auroc_bars.png` | Bar chart comparing AUROC across detectors |
| **4. AUROC Radar Chart** | `fig_auroc_radar.png` | Radar plot of attack effectiveness |
| **5. Method Comparison** | `fig_method_comparison.png` | 4-panel comprehensive method comparison |
| **6. Tradeoff Plot** | `fig_tradeoff.png` | Quality (sim_e5) vs Evasion (TPR) Pareto frontier |
| **7. Score Distributions** | `fig_score_distributions.png` | Detection score histograms by method |
| **8-10. Human-AI Separation** | `fig_human_ai_separation_{detector}.png` | Per-detector score distributions |
| **11. Quality vs Evasion** | `fig_quality_vs_evasion.png` | Scatter plot: similarity vs ASR |
| **12. ASR Comparison** | `fig_asr_comparison.png` | Bar chart comparing ASR across methods |
| **13. Perplexity vs Similarity** | `fig_perplexity_vs_similarity.png` | Quality tradeoff scatter plot |

### Figure Details

#### Figure 1: TPR Heatmap (`fig_heatmap_tpr.png`)

**Description**: Detector × Method heatmap showing True Positive Rate at 1% FPR

**Interpretation**: 
- **Lower values (green)** = better attack (more evasion)
- **Higher values (red)** = worse attack (more detection)

#### Figure 6: Tradeoff Plot (`fig_tradeoff.png`)

**Description**: Pareto frontier showing quality vs evasion tradeoff

**Axes**:
- **X-axis**: Semantic Similarity (E5) - higher is better quality
- **Y-axis**: Mean TPR@1%FPR - lower is better evasion

**Interpretation**: Methods in the **bottom-right** corner achieve the best tradeoff (high quality, low detection)

#### Figure 7: Score Distributions (`fig_score_distributions.png`)

**Description**: Histograms of detection scores grouped by attack method for each detector

**Purpose**: Visualize how different attacks shift the detector score distributions

#### Figures 8-10: Human-AI Separation (`fig_human_ai_separation_{detector}.png`)

**Description**: Overlapping histograms showing detector score distributions for human vs AI text per detector

**Purpose**: Shows how well detectors separate human from AI text for each attack method

**Implementation Note**: Uses `_create_human_ai_separation_from_df()` helper to work with scores DataFrame

#### Figure 11: Quality vs Evasion (`fig_quality_vs_evasion.png`)

**Description**: Scatter plot with similarity on X-axis and ASR (Attack Success Rate) on Y-axis

**Interpretation**: Ideal methods appear in **top-right** (high quality, high ASR)

#### Figure 13: Perplexity vs Similarity (`fig_perplexity_vs_similarity.png`)

**Description**: Scatter plot showing the relationship between fluency (perplexity) and semantic similarity

**Interpretation**: Lower perplexity = more fluent; Higher similarity = better meaning preservation

---

## 8. Generated Tables

### Table 1: Main Results (`main_results.md`)

| Method | roberta AUC | roberta TPR | roberta ASR | ... | Mean TPR | Mean ASR |
|--------|-------------|-------------|-------------|-----|----------|----------|
| no_attack | 0.XXX | 0.XXX | 0.XXX | ... | 0.XXX | 0.XXX |
| stealthrl | 0.XXX | 0.XXX | 0.XXX | ... | 0.XXX | 0.XXX |
| ... | ... | ... | ... | ... | ... | ... |

### Table 2: Transfer Matrix (`transfer_matrix.md`)

Shows TPR@1%FPR for each detector × method combination:

| Detector | no_attack | simple_paraphrase | stealthrl | authormist | homoglyph |
|----------|-----------|-------------------|-----------|------------|-----------|
| roberta | X.XX | X.XX | X.XX | X.XX | X.XX |
| fast_detectgpt | X.XX | X.XX | X.XX | X.XX | X.XX |
| binoculars | X.XX | X.XX | X.XX | X.XX | X.XX |

### Table 3: Quality Metrics (`quality.md`)

| Method | N | Similarity (E5) | Perplexity | Edit Rate |
|--------|---|-----------------|------------|-----------|
| no_attack | 1 | 1.000 | XX.X | 0.000 |
| stealthrl | 1 | 0.XXX | XX.X | 0.XXX |
| ... | ... | ... | ... | ... |

### Table 4: Sanitization Results (`sanitize.md`)

| Detector | Before Sanitize | After Sanitize | Δ TPR |
|----------|-----------------|----------------|-------|
| roberta | X.XX | X.XX | +X.XX |
| ... | ... | ... | ... |

---

## 9. Running the Pipeline

### Prerequisites

```bash
# Install dependencies
pip install torch transformers datasets pandas numpy scikit-learn matplotlib seaborn
pip install tinker tinker-cookbook  # For StealthRL
pip install silverspeak            # For homoglyph attacks

# Set Tinker API key
export TINKER_API_KEY="your-api-key"

# Ensure Ollama is running with required models
ollama pull qwen3:4b-instruct
ollama create authormist -f models/authormist/Modelfile
```

### Basic Usage

```bash
# Quick test (50 samples each, minimal methods)
python scripts/run_eval.py \
  --run-name quick_test \
  --methods m0 m1 m2 \
  --stealthrl-checkpoint checkpoints/atharv_checkpoint_1.json \
  --detectors roberta fast_detectgpt binoculars \
  --n-human 50 --n-ai 50 \
  --datasets mage
```

### Full Research Run

```bash
# Full evaluation (1000 samples)
python scripts/run_eval.py \
  --run-name full_research \
  --methods m0 m1 m2 m3 m4 \
  --stealthrl-checkpoint checkpoints/atharv_checkpoint_1.json \
  --detectors roberta fast_detectgpt binoculars \
  --n-human 1000 --n-ai 1000 \
  --n-candidates 4 \
  --datasets mage
```

### Optional GPT Quality Evaluation

```bash
# GPT-based quality evaluation for StealthRL only (requires OpenAI API key)
export OPENAI_API_KEY="your-openai-key"
python scripts/run_eval.py \
  --methods m2 \
  --stealthrl-checkpoint checkpoints/atharv_checkpoint_1.json \
  --detectors roberta fast_detectgpt binoculars \
  --n-human 1000 --n-ai 1000 \
  --gpt-quality --gpt-quality-max-per-method 200 \
  --gpt-quality-model gpt-5-mini
```

### Fast StealthRL (M2) Tinker Evals

```bash
# Concurrent Tinker sampling with resumable cache
python scripts/run_eval.py \
  --methods m2 \
  --stealthrl-checkpoint checkpoints/atharv_checkpoint_1.json \
  --detectors roberta fast_detectgpt binoculars \
  --n-human 1000 --n-ai 1000 \
  --tinker-concurrency 64 \
  --tinker-chunk-size 256 \
  --tinker-resume
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--run-name` | "eval" | Name for output directory |
| `--out-dir` | `"outputs/eval_runs"` | Base output directory |
| `--datasets` | `["mage"]` | Datasets to evaluate |
| `--methods` | `["m0", "m1"]` | Attack methods |
| `--detectors` | `["roberta", "fast_detectgpt"]` | Detectors to use |
| `--stealthrl-checkpoint` | None | Path to Tinker checkpoint JSON (required for M2) |
| `--n-human` | 1000 | Number of human samples per dataset |
| `--n-ai` | 1000 | Number of AI samples per dataset |
| `--n-candidates` | 2 | Candidates per sample (batched API call) |
| `--cache-dir` | `"cache"` | HuggingFace cache directory |
| `--device` | Auto | Device (`cuda`/`mps`/`cpu`) |
| `--seed` | 42 | Random seed |
| `--n-bootstrap` | 500 | Bootstrap samples for CI |
| `--log-level` | `"INFO"` | Logging level |
| `--quick` | False | Quick test mode (overrides defaults) |
| `--gpt-quality` | False | Enable GPT-based quality evaluation |
| `--gpt-quality-max-per-method` | 200 | Max samples per method for GPT judging |
| `--gpt-quality-model` | `"gpt-5-mini"` | GPT model for judging |
| `--gpt-quality-methods` | None | Methods to judge (default: `m2`/`stealthrl`) |
| `--openai-api-key` | None | API key override (else uses `OPENAI_API_KEY`) |
| `--gpt-quality-no-cache` | False | Disable cached GPT judgments |
| `--tinker-concurrency` | 64 | Max concurrent Tinker requests for M2 |
| `--tinker-chunk-size` | 256 | Chunk size for concurrent M2 sampling |
| `--tinker-max-retries` | 2 | Retry attempts per M2 request |
| `--tinker-backoff-s` | 0.5 | Base backoff seconds for retries |
| `--tinker-resume` | False | Enable resumable M2 cache |
| `--tinker-resume-path` | None | Custom resume cache path |

**Note**: The `--n-values` (budget sweep) and `--no-sanitize` arguments are deprecated. The current pipeline uses a fixed `--n-candidates` value.

### Integration Test

```bash
# Run integration tests to verify setup
python scripts/test_integrations.py
```

---

## 10. Tradeoffs for M4 Mac

The following tradeoffs were made to run on M4 MacBook (18GB unified memory):

### Detector Tradeoffs

| Component | Paper Default | Current Setup | Impact |
|-----------|---------------|---------------|--------|
| **Binoculars Performer** | Falcon-7B-instruct (~14GB) | GPT-2 medium (~1.5GB) | Slightly lower accuracy |
| **Binoculars Observer** | Falcon-7B (~14GB) | GPT-2 large (~3GB) | Slightly lower accuracy |
| **Fast-DetectGPT** | GPT-Neo 2.7B | GPT-Neo 2.7B (~5.5GB) | ✅ Paper standard (full accuracy) |

### Method Tradeoffs

| Component | Full Setup | M4 Mac Setup | Impact |
|-----------|------------|--------------|--------|
| **StealthRL** | Local PEFT (~8GB) | Tinker cloud API | Network latency (~5s/request) |
| **AuthorMist** | Full HF weights (~8GB) | Q4_K_M GGUF (~1.8GB) | Slight quality loss |
| **M1 Paraphrase** | Cloud API | Local Ollama | Faster, no API cost |

### Memory Footprint

| Detector | Memory Required |
|----------|-----------------|
| RoBERTa | ~1.5GB |
| Fast-DetectGPT (GPT-Neo 2.7B) | ~5.5GB |
| Binoculars (lightweight) | ~4.5GB |
| **Total (all loaded)** | **~11.5GB** |

---

## 11. Key Bug Fixes and Implementation Notes

This section documents critical bugs that were discovered and fixed during development.

### 11.1 MAGE Dataset Label Inversion Bug (CRITICAL)

**Issue**: MAGE dataset labels were being interpreted incorrectly.

**Root Cause**: The `label` field in MAGE has the opposite encoding from intuition:
- `label=1` → Human-written text
- `label=0` → AI-generated text

**Verification Method**: Examined the `src` field patterns:
```python
# AI samples have src like: "gpt2-xl", "davinci", "gpt-3.5-turbo"
# Human samples have src like: "human" or null/empty
```

**Fix Location**: `eval/data.py`, line 187:
```python
label = "human" if item.get("label", 0) == 1 else "ai"
```

**Impact**: Without this fix, all metrics were inverted (high AUROC became low, etc.)

### 11.2 Score Distribution Plot Column Names

**Issue**: `KeyError: 'detector'` in `create_score_distribution_plot()`

**Root Cause**: The scores DataFrame uses `detector_name` and `detector_score` as column names, not `detector` and `score`.

**Fix**: Updated default parameters in `create_score_distribution_plot()`:
```python
def create_score_distribution_plot(
    scores_df: pd.DataFrame,
    detector_col: str = "detector_name",  # Fixed from "detector"
    method_col: str = "method",
    score_col: str = "detector_score",    # Fixed from "score"
    ...
)
```

### 11.3 Detector Score Convention

**Convention**: All detectors output scores where **higher = more likely AI-generated**.

This is enforced in `eval/detectors.py`:
```python
DETECTOR_CONVENTIONS = {
    "roberta": {"higher_is_ai": True},
    "fast_detectgpt": {"higher_is_ai": True},
    "binoculars": {"higher_is_ai": True},
    ...
}
```

**Note**: Binoculars raw score is inverted (lower = more AI), but is normalized by the detector wrapper.

### 11.4 Tinker API Batched Inference

**Feature**: StealthRL (M2) now supports batched inference via `num_samples` parameter.

**Implementation**: `eval/methods/stealthrl.py` uses Tinker's `SamplingClient.sample()` with:
```python
results = sampling_client.sample(
    prompt=formatted_prompt,
    num_samples=n_candidates,  # Batched in single API call
    ...
)
```

This eliminates N sequential API calls, reducing latency significantly.

### 11.5 Output Directory Change

**Change**: Default output directory changed from `artifacts/` to `outputs/eval_runs/`

**Reason**: Better organization and separation from other output types.

**CLI Flag**: `--out-dir outputs/eval_runs` (now the default)

---

## Appendix A: Tinker Checkpoint Format

```json
{
  "model_id": "a138335f-6bba-51a7-835f-1dc24461e86a:train:0",
  "base_model": "Qwen/Qwen3-4B-Instruct-2507",
  "lora_rank": 32,
  "checkpoints": {
    "final_state": "tinker://a138335f-.../weights/final",
    "sampler_weights": "tinker://a138335f-.../sampler_weights/final_sampler"
  }
}
```

## Appendix A.1: RAID Dataset (Optional Multi-Domain Generalization)

RAID is a comprehensive benchmark with **11 domains** and **11 generators**. It's optional due to its size (~11GB).

### RAID Domains
| Domain | Description | Good for Testing |
|--------|-------------|------------------|
| `news` | News articles | Standard NLP |
| `wiki` | Wikipedia text | Encyclopedic style |
| `abstracts` | Academic abstracts | Technical writing |
| `books` | Book excerpts | Narrative prose |
| `code` | Programming code | Non-natural language |
| `recipes` | Cooking recipes | Procedural text |
| `reviews` | Product reviews | Informal/opinionated |
| `reddit` | Reddit posts | Social media style |
| `poetry` | Poetry | Creative writing |
| `czech` | Czech language | Cross-lingual |
| `german` | German language | Cross-lingual |

### RAID Generators
ChatGPT, GPT-4, GPT-3, GPT-2, Llama-2-Chat, Mistral-7B, Mistral-7B-Chat, MPT-30B, MPT-30B-Chat, Cohere, Cohere-Chat

### When to Use RAID
```bash
# Cross-domain generalization test
python -m eval.runner_enhanced \
  --run-name raid_generalization \
  --methods m0 m2 \
  --datasets raid \
  --n-human 200 --n-ai 200  # Keep small due to size
```

**Note**: RAID test split has hidden labels. Use `train` split (this is the default).
```

---

## Appendix B: Ablation Studies

The ablation studies are run via a dedicated script `scripts/run_ablations.py` that systematically tests different aspects of the attack methods. These ablations help answer key research questions about method robustness, transfer, and tradeoffs.

### Running All Ablations

```bash
# Run all ablations with 100 samples per class
python scripts/run_ablations.py \
    --stealthrl-checkpoint checkpoints/atharv_checkpoint_1.json \
    --n-human 100 --n-ai 100 \
    --out-dir outputs/ablations

# Run specific ablations only
python scripts/run_ablations.py \
    --ablations guidance budget sanitize padben \
    --stealthrl-checkpoint checkpoints/atharv_checkpoint_1.json \
    --n-human 100 --n-ai 100
```

### CLI Arguments for run_ablations.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--ablations` | `["guidance", "budget", "homoglyph", "sanitize", "padben"]` | Which ablations to run |
| `--n-human` | 100 | Number of human samples |
| `--n-ai` | 100 | Number of AI samples |
| `--stealthrl-checkpoint` | None | Path to StealthRL checkpoint (required for M2) |
| `--detectors` | `["roberta", "fast_detectgpt", "binoculars"]` | Detectors to evaluate |
| `--out-dir` | `"outputs/ablations"` | Output directory |
| `--device` | Auto | Device (`cuda`/`mps`/`cpu`) |
| `--seed` | 42 | Random seed |
| `--n-bootstrap` | 500 | Bootstrap samples for CI |

### Ablation Output Structure

```
outputs/ablations/ablations_{timestamp}/
├── ablation_summary.json          # Summary of all ablations run
├── ablations_{timestamp}.log      # Detailed log
├── guidance_transfer/
│   ├── guidance_transfer_results.csv
│   ├── guidance_transfer_matrix.csv
│   └── fig_guidance_heatmap.png
├── budget_sweep/
│   ├── budget_sweep_results.csv
│   ├── budget_sweep_aggregated.csv
│   └── fig_budget_sweep.png
├── homoglyph_sweep/
│   ├── homoglyph_sweep_results.csv
│   └── fig_homoglyph_sweep.png
├── sanitize/
│   ├── sanitize_results.csv
│   ├── table_sanitize.md
│   └── fig_sanitize.png
└── padben_transfer/
    ├── padben_results.csv
    └── fig_padben_heatmap.png
```

---

### Ablation 7.2: Guidance Transfer

**Research Question**: Does the detector used for guidance during candidate selection transfer to other detectors at evaluation time?

**Motivation**: Adversarial Paraphrasing (Sadasivan et al.) showed that selection can be guided by one detector but evaluated on another. This ablation tests whether guidance detector choice matters for cross-detector robustness.

**Method**: For M3 (Adversarial Paraphrasing), we test three guidance strategies:
1. **m3_roberta**: Select candidate minimizing RoBERTa score
2. **m3_fastdetect**: Select candidate minimizing Fast-DetectGPT score  
3. **m3_ensemble**: Select candidate minimizing ensemble mean score

Each variant is then evaluated against **all three detectors**.

**Output**: A 3×3 heatmap showing TPR@1%FPR for each (guidance_detector, eval_detector) pair.

**Expected Insights**:
- Diagonal entries (matching guidance/eval) should have lowest TPR (best evasion)
- Off-diagonal entries show transfer degradation
- Ensemble guidance may provide more robust transfer

```bash
# Run guidance transfer ablation only
python scripts/run_ablations.py \
    --ablations guidance \
    --n-human 100 --n-ai 100 \
    --out-dir outputs/ablations
```

---

### Ablation 8.1: Budget Sweep

**Research Question**: How does attack performance scale with the number of candidates N?

**Motivation**: Methods like M1, M2, and M3 can generate multiple candidates and select the best. More candidates should improve evasion but at higher compute cost.

**Method**: For M1 (Simple Paraphrase) and M2 (StealthRL):
- Test N ∈ {1, 2, 4, 8} candidates
- Generate N candidates per sample
- Select best candidate by detector score (M3) or random (M1, M2)
- Plot mean TPR@1%FPR and mean similarity vs N

**Output**: 
- `fig_budget_sweep.png`: Two-panel plot (TPR vs N, Similarity vs N)
- `budget_sweep_aggregated.csv`: Mean metrics across detectors

**Expected Insights**:
- TPR should decrease (better evasion) as N increases
- Similarity may decrease slightly (quality tradeoff)
- Diminishing returns expected at higher N
- Helps justify paper's choice of N value

```bash
# Run budget sweep ablation only
python scripts/run_ablations.py \
    --ablations budget \
    --stealthrl-checkpoint checkpoints/atharv_checkpoint_1.json \
    --n-human 100 --n-ai 100
```

---

### Ablation 8.3: Homoglyph Sweep

**Research Question**: How does homoglyph substitution rate affect detection and readability?

**Motivation**: The SilverSpeak paper showed that small character substitutions can evade detectors. This ablation maps the rate-vs-detection tradeoff.

**Method**: For M5 (Homoglyph Attack):
- Test substitution rates p ∈ {0.1%, 0.5%, 1%, 2%}
- Apply homoglyph attack at each rate
- Measure TPR@1%FPR for each detector

**Output**:
- `fig_homoglyph_sweep.png`: Line plot of rate vs TPR per detector
- `homoglyph_sweep_results.csv`: Full results

**Expected Insights**:
- Higher rates → lower TPR (better evasion)
- But higher rates → more visible artifacts
- Different detectors may have different sensitivity
- Helps justify the default 10% rate or recommend alternatives

```bash
# Run homoglyph sweep ablation only
python scripts/run_ablations.py \
    --ablations homoglyph \
    --n-human 100 --n-ai 100
```

---

### Ablation 9: Sanitize Defense

**Research Question**: Can simple text sanitization (Unicode normalization) recover detection ability?

**Motivation**: Homoglyph attacks rely on Unicode trickery. A defense that normalizes text (NFKC) and removes zero-width characters might neutralize such attacks. This tests whether semantic attacks (M2) are more robust than character-level attacks (M5).

**Method**: 
1. Apply attack method (M5 homoglyph, M2 StealthRL)
2. Apply sanitization: Unicode NFKC normalization + zero-width removal + homoglyph mapping
3. Re-score with all detectors
4. Compare TPR before vs after sanitization

**Output**:
- `fig_sanitize.png`: Bar chart comparing before/after TPR
- `table_sanitize.md`: Markdown table with Δ TPR
- `sanitize_results.csv`: Full results

**Expected Insights**:
- M5 (homoglyph): Large TPR increase after sanitization (defense works)
- M2 (StealthRL): Minimal TPR change (attack is semantic, not character-level)
- Demonstrates that StealthRL is robust to simple defenses

```bash
# Run sanitize ablation only
python scripts/run_ablations.py \
    --ablations sanitize \
    --stealthrl-checkpoint checkpoints/atharv_checkpoint_1.json \
    --n-human 100 --n-ai 100
```

---

### Ablation: Cross-Dataset Transfer (PadBen)

**Research Question**: Does StealthRL generalize from MAGE to a different benchmark (PadBen)?

**Motivation**: A method tuned on one dataset might overfit. Testing on PadBen (a paraphrase-focused benchmark) validates generalization.

**Method**:
1. Load PadBen dataset (different distribution from MAGE)
2. Apply M0 (baseline), M1 (simple), M2 (StealthRL)
3. Evaluate all detectors on PadBen
4. Compare to MAGE results

**Output**:
- `fig_padben_heatmap.png`: Detector × Method heatmap on PadBen
- `padben_results.csv`: Full metrics

**Expected Insights**:
- If M2 maintains low TPR on PadBen, the RL policy learned generalizable evasion
- Compare MAGE vs PadBen performance to quantify transfer gap

```bash
# Run PadBen transfer ablation only
python scripts/run_ablations.py \
    --ablations padben \
    --stealthrl-checkpoint checkpoints/atharv_checkpoint_1.json \
    --n-human 100 --n-ai 100
```

---

### Full Ablation Run (Recommended)

For a complete ablation study suitable for paper submission:

```bash
python scripts/run_ablations.py \
    --ablations all \
    --stealthrl-checkpoint checkpoints/atharv_checkpoint_1.json \
    --n-human 200 --n-ai 200 \
    --n-bootstrap 1000 \
    --out-dir outputs/ablations
```

**Estimated Runtime** (M4 MacBook, 200 samples):
- Guidance Transfer: ~20 min
- Budget Sweep: ~30 min  
- Homoglyph Sweep: ~10 min
- Sanitize: ~15 min
- PadBen Transfer: ~20 min
- **Total**: ~95 min

---

## Appendix C: Checkpoint Comparison

Compare multiple StealthRL checkpoints against each other using `scripts/compare_checkpoints.py`. This is useful for:
- Comparing training runs with different hyperparameters
- Tracking training progress at different steps
- Ablating architecture or reward function choices
- Selecting the best checkpoint for final evaluation

### Running Checkpoint Comparison

```bash
# Compare two checkpoints
python scripts/compare_checkpoints.py \
    --checkpoints checkpoints/run1.json checkpoints/run2.json \
    --n-human 100 --n-ai 100

# Compare with custom display names
python scripts/compare_checkpoints.py \
    --checkpoints checkpoints/early.json checkpoints/late.json \
    --names "Early (5k steps)" "Late (20k steps)" \
    --n-human 100 --n-ai 100

# Quick comparison (fewer samples, single detector)
python scripts/compare_checkpoints.py \
    --checkpoints checkpoints/*.json \
    --quick

# Compare with specific detectors
python scripts/compare_checkpoints.py \
    --checkpoints cp1.json cp2.json cp3.json \
    --detectors roberta fast_detectgpt \
    --n-human 200 --n-ai 200
```

### CLI Arguments for compare_checkpoints.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoints` | **Required** | Paths to checkpoint JSON files to compare |
| `--names` | Filename stems | Display names for each checkpoint |
| `--dataset` | `"mage"` | Dataset to use for evaluation |
| `--n-human` | 100 | Number of human samples |
| `--n-ai` | 100 | Number of AI samples |
| `--n-candidates` | 1 | Candidates per sample |
| `--detectors` | `["roberta", "fast_detectgpt", "binoculars"]` | Detectors to evaluate |
| `--out-dir` | `"outputs/checkpoint_comparison"` | Output directory |
| `--quick` | False | Quick mode (20 samples, 1 detector) |
| `--device` | Auto | Device (`cuda`/`mps`/`cpu`) |
| `--seed` | 42 | Random seed |
| `--n-bootstrap` | 500 | Bootstrap samples for CI |

### Checkpoint Comparison Output Structure

```
outputs/checkpoint_comparison/comparison_{timestamp}/
├── comparison_summary.json         # Metadata about the comparison
├── compare_{timestamp}.log         # Detailed log
├── checkpoint_comparison.csv       # Full results table
├── comparison_table.md             # Markdown summary table
└── fig_checkpoint_comparison.png   # Bar chart comparing ASR and AUROC
```

### Output Metrics

For each (checkpoint, detector) pair, the comparison reports:

| Metric | Description | Better For Attack |
|--------|-------------|-------------------|
| `auroc` | Area under ROC curve | Lower |
| `tpr_at_1fpr` | True positive rate at 1% FPR | Lower |
| `asr` | Attack success rate (1 - tpr_at_1fpr) | Higher |
| `mean_similarity` | Semantic similarity to original | Higher |

### Example Output

```
======================================================================
COMPARISON SUMMARY
======================================================================

roberta:
  Best checkpoint: late_20k
  ASR: 95.2%
  AUROC: 0.312

fast_detectgpt:
  Best checkpoint: late_20k
  ASR: 98.1%
  AUROC: 0.087

binoculars:
  Best checkpoint: late_20k
  ASR: 97.5%
  AUROC: 0.124

======================================================================
OVERALL BEST: late_20k
Mean ASR across detectors: 96.9%
======================================================================
```

---

## Appendix D: PadBen Task Configurations

PadBen provides 5 tasks designed to probe different aspects of AI text detection under paraphrase attacks:

| Config | Task | Human Class | AI Class | Research Question |
|--------|------|-------------|----------|-------------------|
| `exhaustive-task1` | Paraphrase Source | Human paraphrases | LLM paraphrases | Can detectors identify WHO did the paraphrasing? |
| `exhaustive-task2` | Text Authorship | Human-written | LLM-generated | Standard detection (same as MAGE) |
| `exhaustive-task3` | AI Laundering | 1st-pass paraphrase | 3rd-pass paraphrase | Can detectors distinguish paraphrase depth? |
| `exhaustive-task4` | Iterative Depth | 1st iteration | 3rd iteration | Ablation: Does more paraphrasing help? |
| `exhaustive-task5` | Deep Attack | Human-written | 3rd-iteration paraphrase | **Key test**: Best-case paraphrase evasion |

## Appendix E: Color Scheme

Consistent colors for paper figures:

```python
COLORS = {
    "no_attack": "#0072B2",          # Blue
    "simple_paraphrase": "#E69F00",  # Orange
    "stealthrl": "#009E73",          # Green (ours)
    "authormist": "#CC79A7",         # Pink
    "homoglyph": "#56B4E9",          # Light blue
}
```

---

## References

1. **MAGE**: Pu et al., "Deepfake Text Detection" (yaful/DeepfakeTextDetect)
2. **PadBen**: Zha et al., "PADBen: Paraphrase and AI-Generated Text Detection Benchmark"
3. **Fast-DetectGPT**: Bao et al., ArXiv 2310.05130
4. **Binoculars**: Hans et al., ArXiv 2401.12070
5. **SilverSpeak**: ArXiv 2406.11239
6. **AuthorMist**: HuggingFace model card
7. **Tinker**: https://tinker-docs.thinkingmachines.ai/
