Combined dataset build
======================

Default 40% ESL target with no duplicates:

```bash
python scripts/build_full_dataset.py \
  --raw-dir data/raw \
  --output-dir data/tinker_full_esl40_nodup \
  --esl-percent 40 \
  --detectrl-max 20000
```

Optional: specify DetectRL task directories (defaults to Task1 Task2):

```bash
python scripts/build_full_dataset.py \
  --raw-dir data/raw \
  --output-dir data/tinker_full_esl40_nodup \
  --esl-percent 40 \
  --detectrl-max 20000 \
  --detectrl-tasks Task1 Task2
```

Last training command used
==========================

```bash
python -m stealthrl.tinker.train \
    --config configs/tinker_stealthrl.yaml \
    --data-path data/tinker_full_esl40_nodup \
    --run-name esl40_full_async_overlap \
    --num-epochs 5 \
    --batch-size 64 \
    --group-size 16 \
    --training-mode async \
    --async-groups-per-batch 64 \
    --async-max-steps-off-policy 2
```

Fairness analysis (ESL vs native)
=================================

```bash
python scripts/analyze_detector_fairness.py \
  --data-path data/tinker_full_esl40_nodup \
  --split train \
  --text-field human_reference \
  --batch-size 16 \
  --max-samples 500 \
  --output-json outputs/fairness/esl_vs_native.json
```

Optional cache to speed up repeat runs:

```bash
python scripts/analyze_detector_fairness.py \
  --data-path data/tinker_full_esl40_nodup \
  --split train \
  --text-field human_reference \
  --batch-size 16 \
  --max-samples 500 \
  --cache-path outputs/fairness/detector_cache.sqlite \
  --output-json outputs/fairness/esl_vs_native.json
```

Threshold sweep (Fast-DetectGPT only)
=====================================

```bash
python -m scripts.analyze_detector_fairness \
  --data-path data/tinker_full_esl40_nodup \
  --split test \
  --text-field human_reference \
  --batch-size 16 \
  --max-samples 500 \
  --detectors fast_detectgpt \
  --thresholds 0.3,0.5,0.7 \
  --output-json outputs/fairness/esl_vs_native_fast_detectgpt_thresholds.json
```

Visualization summary (esl_fast_detect_gpt)
===========================================

Run directory:
`outputs/runs/data/esl_fast_detect_gpt`

Artifacts:
- `outputs/runs/data/esl_fast_detect_gpt/visualizations/training_curves.png`
- `outputs/runs/data/esl_fast_detect_gpt/visualizations/pareto_frontiers.png`
- `outputs/runs/data/esl_fast_detect_gpt/visualizations/reward_decomposition.png`
- `outputs/runs/data/esl_fast_detect_gpt/visualizations/stability_metrics.png`
- `outputs/runs/data/esl_fast_detect_gpt/visualizations/training_summary.csv`
- `outputs/runs/data/esl_fast_detect_gpt/visualizations/training_summary.txt`

Key metrics (112 steps):
- Total reward: 0.4038 → 0.5565 (best 2.2486, mean 0.3999)
- Detector evasion: -0.2056 → -0.1845 (best 1.8287)
- Semantic similarity: 0.8606 → 0.8945 (best 0.9682)
- Perplexity: 23.8424 → 29.9861 (best 18.4490, mean 49.9081)
- KL divergence: 0.0009 → 0.3492 (mean 1.1654)
- Parse success: 0.8828 → 0.9189 (mean 0.9520)
- Detector prob: 0.5919 → 0.5644 (best 0.3805)
