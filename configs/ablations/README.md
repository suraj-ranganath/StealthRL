# Ablation Configurations

This directory contains ablation configurations that selectively remove or modify reward components to study their impact.

**What These Ablations Cover**
- Single-detector training versus ensemble training
- Removing fairness, semantic, or quality rewards
- Detector-only optimization without constraints

**Run All Ablations**
```bash
bash scripts/run_ablations.sh
```

**Run One Ablation**
```bash
python scripts/train_stealthrl.py --config configs/ablations/<ablation_name>.yaml
```

**Evaluate Ablations**
```bash
python scripts/evaluate_ablations.py \
  --ablation_dir checkpoints \
  --test_data data/processed/test.jsonl \
  --esl_data data/processed/esl_test.jsonl \
  --native_data data/processed/native_test.jsonl \
  --output_dir outputs/ablations
```

**Notes**
- Each YAML file documents its intended change.
- Use the same evaluation settings across ablations for comparability.
