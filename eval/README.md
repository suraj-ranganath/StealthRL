# Evaluation

This directory contains the evaluation harness used to compare attack methods and detectors under consistent datasets and metrics. The runner produces standardized artifacts for tables and figures.

**Quick Start**
```bash
python scripts/run_eval.py --datasets mage --methods m0 m1 --detectors roberta fast_detectgpt
```

**Common Outputs**
```
artifacts/
├── scores.parquet
├── scores.csv
├── quality.parquet
├── metrics.json
├── thresholds.json
├── raw_outputs.json
├── figures/
└── tables/
```

**Methods**
- `m0` no attack baseline
- `m1` simple paraphrase
- `m2` StealthRL policy (requires checkpoint)
- `m3` detector-guided paraphrasing
- `m4` AuthorMist baseline
- `m5` homoglyph baseline

**Detectors**
- `roberta` classifier baseline
- `fast_detectgpt` curvature-based detector
- `detectgpt` perturbation-based detector
- `binoculars` paired language model detector
- `ghostbuster` feature-ensemble detector
- `mage` longformer-based detector

**Python API**
```python
from eval import EvalRunner

runner = EvalRunner(output_dir="artifacts")
runner.run(
    datasets=["mage"],
    methods=["m0", "m1"],
    detectors=["roberta", "fast_detectgpt"],
)
```

**Extending**
- Add new methods in `eval/methods/`
- Add new detectors in `eval/detectors.py`
- Register new datasets in `eval/data.py`
