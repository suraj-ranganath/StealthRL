# StealthRL Evaluation Pipeline

Comprehensive evaluation suite for StealthRL paper submission.

## Quick Start

```bash
# Quick test (small dataset)
python scripts/run_eval.py --quick

# Full evaluation with MAGE
python scripts/run_eval.py --datasets mage --n-human 1000 --n-ai 1000

# With StealthRL checkpoint
python scripts/run_eval.py --stealthrl-checkpoint outputs/runs/best_checkpoint --methods m0 m1 m2

# With GPT-based quality evaluation (requires OPENAI_API_KEY)
python scripts/run_eval.py --methods m2 --gpt-quality --gpt-quality-max-per-method 200

# All methods and detectors
python scripts/run_eval.py \
    --datasets mage raid \
    --methods m0 m1 m2 m3 m4 m5 \
    --detectors roberta fast_detectgpt detectgpt binoculars

# Reuse exact samples from a prior run (ensures identical IDs)
python scripts/run_eval.py \
    --datasets mage \
    --methods m2 \
    --reuse-samples-from outputs/eval_runs/mage_no_m2
```

## CLI Usage

```bash
python -m eval \
    --datasets mage raid_slice \
    --methods m0 m1 m2 m3 m4 m5 \
    --detectors roberta fast_detectgpt detectgpt binoculars \
    --n-candidates 1 2 4 8 \
    --out-dir artifacts/
```

## Output Structure

```
artifacts/
├── scores.parquet          # Per-sample detector scores
├── scores.csv              # Same in CSV format
├── quality.parquet         # Per-sample quality metrics
├── quality_gpt.parquet     # GPT-based quality ratings (optional)
├── metrics.json            # Aggregated metrics with CIs
├── thresholds.json         # Calibrated detector thresholds
├── raw_outputs.json        # Raw attack outputs
├── figures/
│   ├── fig_heatmap_t1f.png # Detector × Method heatmap
│   ├── fig_tradeoff.png    # Evasion-quality tradeoff
│   └── fig_budget_sweep.png # Candidate budget sweep
└── tables/
    ├── table_main_mage.md  # Main results table
    ├── table_transfer.md   # Transfer matrix
    └── table_quality.md    # Quality metrics table
```

## Methods

| ID | Name | Description |
|----|------|-------------|
| M0 | `no_attack` | Original AI text (baseline) |
| M1 | `simple_paraphrase` | Base LM paraphrase without RL |
| M2 | `stealthrl` | RL-trained policy (requires checkpoint) |
| M3 | `adversarial_paraphrasing` | Detector-guided selection |
| M4 | `authormist` | AuthorMist HF model |
| M5 | `homoglyph` | Homoglyph substitution attack |

## Detectors

| Name | Family | Model |
|------|--------|-------|
| `roberta` | Classifier | `openai-community/roberta-large-openai-detector` |
| `fast_detectgpt` | Curvature | GPT-2 medium sampling discrepancy |
| `detectgpt` | Curvature | Original perturbation-based |
| `binoculars` | Zero-shot | Paired LM cross-entropy |
| `ghostbuster` | Feature | RoBERTa with weak features |
| `mage` | Longformer | `yaful/MAGE` |

## Metrics

### Detector Metrics (per detector, per method)
- **AUROC**: Area under ROC curve
- **TPR@1%FPR**: True positive rate at 1% false positive rate
- **ASR**: Attack success rate (1 - TPR@1%FPR)

### Quality Metrics (per sample)
- **sim_e5**: E5 cosine similarity
- **ppl_score**: Perplexity (GPT-2)
- **edit_rate**: Character edit distance ratio
 - **quality_rating**: GPT Likert quality rating (1-5, optional)
 - **similarity_rating**: GPT semantic similarity rating (1-5, optional)

## Python API

```python
from eval import EvalRunner, load_eval_dataset, get_detector, get_method

# Load dataset
dataset = load_eval_dataset("mage", n_human=500, n_ai=500)

# Load detector
detector = get_detector("roberta")
detector.load()
scores = detector.get_scores(["AI generated text..."])

# Load method
method = get_method("simple_paraphrase")
result = method.attack("AI text to paraphrase", n_candidates=4)
print(result.text)  # Paraphrased text
print(result.metadata)  # Attack metadata

# Full pipeline
runner = EvalRunner(output_dir="artifacts")
runner.run(
    datasets=["mage"],
    methods=["m0", "m1"],
    detectors=["roberta", "fast_detectgpt"],
)
```

## Adding New Methods

1. Create `eval/methods/your_method.py`
2. Inherit from `BaseAttackMethod`
3. Implement `load()` and `attack()` methods
4. Register in `eval/methods/__init__.py`

```python
from .base import BaseAttackMethod, AttackOutput

class YourMethod(BaseAttackMethod):
    def __init__(self, **kwargs):
        super().__init__(name="your_method")
    
    def load(self):
        # Load models
        self._loaded = True
    
    def attack(self, text: str, n_candidates: int = 1, **kwargs) -> AttackOutput:
        # Generate attack
        return AttackOutput(text=attacked_text, metadata={})
```

## Adding New Detectors

1. Create detector class inheriting from `BaseEvalDetector`
2. Implement `load()` and `_detect_single()` methods
3. Register in `DETECTOR_REGISTRY`

```python
from .detectors import BaseEvalDetector, DetectorResult

class YourDetector(BaseEvalDetector):
    def __init__(self, **kwargs):
        super().__init__(name="your_detector", **kwargs)
    
    def load(self):
        # Load model
        self._loaded = True
    
    def _detect_single(self, text: str) -> DetectorResult:
        score = ...  # Compute AI probability [0, 1]
        return DetectorResult(score=score, raw_score=score, metadata={})
```

## References

### Benchmarks
- [MAGE](https://huggingface.co/datasets/yaful/MAGE)
- [RAID](https://github.com/liamdugan/raid)
- [PadBen](https://huggingface.co/datasets/JonathanZha/PADBen)

### Detectors
- [RoBERTa OpenAI Detector](https://huggingface.co/openai-community/roberta-large-openai-detector)
- [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt)
- [Binoculars](https://github.com/ahans30/Binoculars)

### Baselines
- [Adversarial Paraphrasing](https://arxiv.org/abs/2506.07001)
- [AuthorMist](https://huggingface.co/authormist/authormist-originality)
- [Homoglyph Attack](https://arxiv.org/abs/2406.11239)
