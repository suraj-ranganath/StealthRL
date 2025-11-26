# ESL Fairness Evaluation Guide

Quick reference for using the ESL/native fairness evaluation infrastructure.

---

## Overview

StealthRL includes specialized tools for evaluating fairness across ESL (English as Second Language) and native English writers:

- **Unified corpus loader** for multiple ESL datasets (TOEFL11, ICNALE, ELLIPSE)
- **Stratified splits** with 40% ESL / 60% native ratio
- **Grouped evaluation** by ESL status (ASR, detector probs, semantic similarity)
- **BERTScore F1** in addition to E5 cosine similarity

---

## Data Preparation

### Step 1: Prepare ESL/Native JSONL Files

Create JSONL files with the required schema:

```json
{
  "id": "toefl11_001",
  "text": "The essay text goes here...",
  "source": "TOEFL11",
  "is_esl": true,
  "proficiency_level": "medium",
  "prompt_id": "P1"
}
```

**Required fields**: `id`, `text`, `source`, `is_esl`
**Optional fields**: `proficiency_level`, `prompt_id`

**Expected locations**:
```
data/
├── esl/
│   ├── toefl11.jsonl
│   ├── icnale_written.jsonl
│   └── ellipse.jsonl
└── native/
    └── native_academic.jsonl
```

### Step 2: Build Evaluation Splits

```bash
python -m stealthrl.data.esl_native_corpus
```

This creates:
- `data/processed/esl_native_dev.jsonl` (200 samples by default)
- `data/processed/esl_native_test.jsonl` (500 samples by default)

**Customization**:
```python
from pathlib import Path
from stealthrl.data.esl_native_corpus import build_esl_native_eval_split

dev_records, test_records = build_esl_native_eval_split(
    data_dir=Path("data"),
    output_dir=Path("data/processed"),
    dev_size=200,
    test_size=500,
    esl_ratio=0.4,  # 40% ESL
    seed=42,
)
```

---

## Running ESL Fairness Evaluation

### Basic Usage

```bash
python scripts/run_esl_eval.py \
    --eval_data data/processed/esl_native_test.jsonl \
    --stealthrl_model outputs/stealthrl_policy \
    --output_dir results/esl_native_eval
```

### With BERTScore

```bash
python scripts/run_esl_eval.py \
    --eval_data data/processed/esl_native_test.jsonl \
    --stealthrl_model outputs/stealthrl_policy \
    --enable_bertscore \
    --bertscore_model roberta-large \
    --bertscore_batch_size 16 \
    --output_dir results/esl_native_eval
```

**BERTScore model options**:
- `roberta-large` (best quality, slower)
- `microsoft/deberta-base` (2-3x faster, good quality)
- `bert-base-uncased` (fastest, lower quality)

---

## Output Files

### 1. `comparison_report.json`

Overall metrics across all models (base, SFT, StealthRL):

```json
{
  "base_metrics": {
    "asr_all": 0.15,
    "asr_any": 0.45,
    "semantic_sim_mean": 0.88,
    "esl_fpr_gap": {
      "fast_detectgpt": 0.08,
      "ghostbuster": 0.12
    }
  },
  "stealthrl_metrics": { ... },
  "improvements": {
    "asr_improvement_base": 0.45,
    "fairness_improvement_base": -0.05
  }
}
```

### 2. `esl_native_grouped_metrics.json`

Metrics grouped by ESL status:

```json
{
  "overall_base": {
    "asr_all": 0.15,
    "per_detector": {
      "fast_detectgpt": {
        "mean_prob": 0.72,
        "std_prob": 0.18
      }
    },
    "semantic_similarity": {
      "mean": 0.88,
      "std": 0.05
    },
    "n_samples": 500
  },
  "esl_base": { ... },
  "native_base": { ... },
  "overall_stealthrl": { ... },
  "esl_stealthrl": { ... },
  "native_stealthrl": { ... }
}
```

### 3. `bertscore_results.json`

BERTScore F1 by model and group:

```json
{
  "base": {
    "overall": {
      "mean_f1": 0.92,
      "median_f1": 0.93,
      "std_f1": 0.04
    },
    "esl": { ... },
    "native": { ... }
  },
  "stealthrl": { ... }
}
```

### 4. `bertscore_esl_native.jsonl`

Per-sample detailed results:

```json
{"id": "toefl11_001", "is_esl": true, "source": "TOEFL11", "system": "base", "bertscore_f1": 0.91, "e5_cosine": 0.89}
{"id": "toefl11_001", "is_esl": true, "source": "TOEFL11", "system": "stealthrl", "bertscore_f1": 0.93, "e5_cosine": 0.91}
...
```

---

## Python API

### Load ESL/Native Data

```python
from pathlib import Path
from stealthrl.data.esl_native_corpus import load_esl_native_jsonl

# Load a single file
records = load_esl_native_jsonl(Path("data/esl/toefl11.jsonl"))

# Access fields
for rec in records:
    print(f"{rec.id}: is_esl={rec.is_esl}, source={rec.source}")
    print(f"Text: {rec.text[:100]}...")
```

### Compute BERTScore

```python
from stealthrl.metrics.bertscore_metrics import compute_bertscore, BERTScoreConfig

# Configure BERTScore
config = BERTScoreConfig(
    enabled=True,
    model_type="roberta-large",
    batch_size=16,
)

# Compute scores
outputs = ["Paraphrased text 1", "Paraphrased text 2"]
references = ["Original text 1", "Original text 2"]

result = compute_bertscore(outputs, references, config)

print(f"Mean F1: {result['mean_f1']:.4f}")
print(f"Per-sample F1: {result['per_sample_f1']}")
```

### Grouped BERTScore

```python
from stealthrl.metrics.bertscore_metrics import compute_bertscore_grouped

groups = ["esl", "esl", "native", "native"]

grouped_results = compute_bertscore_grouped(
    outputs=outputs,
    references=references,
    groups=groups,
    config=config,
)

print(f"ESL mean F1: {grouped_results['esl']['mean_f1']:.4f}")
print(f"Native mean F1: {grouped_results['native']['mean_f1']:.4f}")
```

---

## Configuration

### Enable BERTScore in Training Config

Edit `configs/tinker_stealthrl.yaml`:

```yaml
reward:
  # ... other reward settings ...
  
  # BERTScore (optional, for evaluation only)
  bertscore:
    enabled: true
    model_type: "roberta-large"
    batch_size: 16
    num_layers: null  # Auto-select best layer
```

**Note**: BERTScore is for **evaluation only** and is not used in the training reward (too slow for online RL).

---

## Expected Data Sources

### ESL Corpora

1. **TOEFL11** - Educational Testing Service
   - ESL essays from test takers
   - 11 native language backgrounds
   - Academic writing domain

2. **ICNALE** - International Corpus Network of Asian Learners of English
   - Written essays from Asian learners
   - Multiple proficiency levels
   - Academic topics

3. **ELLIPSE** - English Language Learner Insight, Proficiency and Skills Evaluation
   - Formative writing from K-12 ESL students
   - Multiple writing tasks

### Native Corpora

1. **Native Academic Writing**
   - College-level essays
   - Research paper excerpts
   - Academic blog posts

---

## Troubleshooting

### Import Error: bert-score not installed

```bash
pip install bert-score>=0.3.13
```

### Missing Data Files

Ensure JSONL files exist at expected paths:
```bash
ls data/esl/*.jsonl
ls data/native/*.jsonl
```

If missing, check preprocessing steps or data download instructions.

### BERTScore Too Slow

Try a faster model:
```bash
--bertscore_model microsoft/deberta-base
```

Or increase batch size (if memory allows):
```bash
--bertscore_batch_size 32
```

### Imbalanced Splits

If your splits are not 40/60, check:
1. Do you have enough ESL and native samples?
2. Are `is_esl` flags correct in JSONL?
3. Adjust `esl_ratio` parameter in `build_esl_native_eval_split()`

---

## Citation

If you use the ESL fairness evaluation infrastructure:

```bibtex
@misc{stealthrl2025,
  title={StealthRL: Ensemble-Guided Text Transformation for Multi-Detector Transfer and Fair Detection Robustness},
  author={Ranganath, Suraj and Mahor, Nishchay and Zhu, Sibo},
  year={2025},
  note={University of California, San Diego}
}
```

---

## References

- **ChatGPT-Detector-Bias**: Weixin Liang et al., "GPT detectors are biased against non-native English writers"
- **TOEFL11**: Educational Testing Service
- **ICNALE**: Ishikawa (2013)
- **ELLIPSE**: Kaggle / ETS
- **BERTScore**: Zhang et al. (2020), "BERTScore: Evaluating Text Generation with BERT"
