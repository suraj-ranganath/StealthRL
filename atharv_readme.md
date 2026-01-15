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
