# StealthRL

Ensemble-guided GRPO training to paraphrase text so it evades multiple detectors while preserving meaning and ESL fairness. Runs on Tinker (Qwen/Qwen3‑4B + LoRA).

## Quick start
```bash
pip install -r requirements.txt
export TINKER_API_KEY=...        # required for Tinker
```

### Build the full dataset (40% ESL, deduped)
```bash
python scripts/build_full_dataset.py \
  --raw-dir data/raw \
  --output-dir data/tinker_full_esl40_nodup \
  --esl-percent 40 \
  --detectrl-max 20000
```

### Train (latest command used)
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

### Fairness / detector analysis
- ESL vs native:
```bash
python scripts/analyze_detector_fairness.py \
  --data-path data/tinker_full_esl40_nodup \
  --split train \
  --text-field human_reference \
  --batch-size 16 \
  --max-samples 500 \
  --output-json outputs/fairness/esl_vs_native.json
```
- With detector cache (faster repeats):
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
- Threshold sweep (Fast‑DetectGPT):
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

### Visualization summary
Run dir: `outputs/runs/data/esl_fast_detect_gpt`

Artifacts:
- training_curves.png / pareto_frontiers.png / reward_decomposition.png / stability_metrics.png
- training_summary.csv / training_summary.txt

Key metrics (112 steps):
- Total reward: 0.4038 → 0.5565 (best 2.2486, mean 0.3999)
- Detector evasion: -0.2056 → -0.1845 (best 1.8287)
- Semantic similarity: 0.8606 → 0.8945 (best 0.9682)
- Perplexity: 23.8424 → 29.9861 (best 18.4490, mean 49.9081)
- KL divergence: 0.0009 → 0.3492 (mean 1.1654)
- Parse success: 0.8828 → 0.9189 (mean 0.9520)
- Detector prob: 0.5919 → 0.5644 (best 0.3805)

## Prior results (Dec 7, 2025 ultra-fast run)
Config: `configs/tinker_stealthrl_ultrafast.yaml`, 800 train / 150 test, 1 epoch, Qwen3‑4B LoRA rank 16.

- Detector evasion improvement: 58.7% → 45.8% det. prob (best checkpoint)
- Semantic similarity: 98.6% avg (min 94%)
- KL < 0.4 throughout; parse success 85.9% → 99.2%
- 9 Pareto‑optimal checkpoints (stealth/quality trade‑offs)

## Project layout (trimmed)
- `stealthrl/` – training, reward, detectors, dataset, env
- `configs/` – YAML configs (full, small, ablations)
- `scripts/` – data prep, evaluation, visualization
- `outputs/` – runs, metrics, visualizations (ignored in git)
- `knowledge_base/` – detailed docs

## Notes
- Batched sampling (single `sample_async` per group) is enabled by default via config (`sampling.batched_sampling: true`).
- For async/stream training, set `parallel.mode` in your config (`async` or `stream_minibatch`).

