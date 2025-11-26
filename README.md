# StealthRL

**Ensemble-Guided Text Transformation for Multi-Detector Transfer and Fair Detection Robustness**

> RL-trained paraphraser for multi-detector-robust, fair text generation.

🚀 **NEW: Tinker Integration** - Now supports training on remote compute with Qwen3-4B via the [Tinker platform](https://tinker-docs.thinkingmachines.ai/). See [TINKER_README.md](TINKER_README.md) for quickstart and DSC 291 deployment guide.

---

## Overview

StealthRL is a research framework that uses **Reinforcement Learning with Verifiable Rewards (RFT)** to train a single, ensemble-guided paraphraser capable of reducing AI-detector scores while preserving semantic meaning and text quality. Unlike prior approaches that train separate models per detector (e.g., AuthorMist), StealthRL investigates whether **joint training against a detector ensemble** can learn detector-agnostic transformation strategies that generalize to unseen detector families.

A core focus of this project is **fairness**: AI text detectors have been shown to produce elevated false-positive rates on writing by ESL (English as a Second Language) authors. StealthRL explicitly monitors and optimizes for shrinking the ESL vs native FPR gap, treating fairness as a first-class objective rather than an afterthought.

### Two Deployment Options

- **Local Training (Original)**: HuggingFace TRL + local GPUs (see sections below)
- **Tinker Platform (DSC 291)**: Remote compute, Qwen3-4B, GRPO enhancements → [TINKER_README.md](TINKER_README.md)

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
└── evaluation/      # StealthBench metrics: AUROC, FPR, BERTScore, perplexity

scripts/
├── prepare_data.py        # Prepare human/LLM text, ESL vs native subsets
├── train_stealthrl.py     # Main RL training entry point
├── evaluate_detectors.py  # Run detector ensemble, produce CSVs
├── run_stealthbench.py    # Unified evaluation harness
└── download_datasets.sh   # Download datasets from original sources

configs/               # YAML/JSON configs for models, training, detectors
examples/              # Sample scripts and notebooks
data/                  # Small toy data (large datasets downloaded separately)
requirements.txt       # Python dependencies
environment.yml        # Conda environment (optional)
LICENSE
```

---

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (recommended: A100, RTX 3090/4090, or similar)
- Access to HuggingFace Hub for base models

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/stealthrl.git
cd stealthrl

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Core Dependencies

- `transformers` - HuggingFace Transformers for base LMs
- `trl` - Transformer Reinforcement Learning (GRPO, PPO, DPO)
- `peft` - Parameter-Efficient Fine-Tuning (LoRA)
- `accelerate` - Distributed training utilities
- `datasets` - HuggingFace Datasets
- `bert-score` - Semantic similarity metric
- `torch` - PyTorch backend

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

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgments

This project was developed as part of **DSC 291: Safety in Generative AI** at the University of California, San Diego.

We thank the authors of AuthorMist, DetectGPT, Ghostbuster, Binoculars, and other foundational works that made this research possible.

---

**Questions or feedback?** Open an issue or reach out to the maintainers.
