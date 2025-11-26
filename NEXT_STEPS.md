# Next Steps for StealthRL

This document outlines the immediate next steps to get StealthRL up and running.

---

## ✅ Completed

- [x] Complete README.md with comprehensive documentation
- [x] Full project structure created
- [x] All core modules fully implemented (rewards, detectors, training, evaluation)
- [x] Data preparation and download scripts implemented
- [x] Configuration files created
- [x] Example scripts provided
- [x] Dependencies specified in requirements.txt

---

## 🚀 Immediate Next Steps

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate stealthrl
```

**Note**: Some packages like `trl` may require specific versions. Check for compatibility.

### 2. Download Datasets

```bash
# Make the script executable
chmod +x scripts/download_datasets.sh

# Download all datasets
bash scripts/download_datasets.sh
```

This will download:
- DetectRL benchmark
- ai-detection-paraphrases (DIPPER)
- ChatGPT-Detector-Bias (ESL vs native data)
- Ghostbuster data
- Human Detectors data

**Estimated time**: 5-10 minutes depending on connection
**Disk space**: ~2-3 GB

### 3. Prepare Data

```bash
python scripts/prepare_data.py \
    --input_dir data/raw/DetectRL \
    --output_dir data/processed \
    --train_split 0.8
```

This will:
- Load human and AI texts from raw datasets
- Split into train/eval/test sets
- Separate ESL and native subsets for fairness analysis
- Save in standardized JSONL format

### 4. Test Individual Components

#### Test a Detector

```python
from stealthrl.detectors import FastDetectGPTDetector

detector = FastDetectGPTDetector(device="cuda")
texts = [
    "This is a human-written text with natural flow.",
    "The implementation of the algorithm is quite straightforward and efficient."
]
scores = detector.detect(texts)
print(f"Detection scores: {scores}")
```

#### Test Reward Computation

```python
from stealthrl.rewards import SemanticFidelityReward

reward = SemanticFidelityReward(metric="bertscore")
original = ["The cat sat on the mat."]
paraphrased = ["A feline rested upon the rug."]
score = reward.compute(original, paraphrased)
print(f"Semantic similarity: {score}")
```

#### Test StealthBench

```python
from stealthrl.evaluation import StealthBench

bench = StealthBench(
    detectors=["fast-detectgpt", "ghostbuster"],
    output_dir="outputs/test_results"
)

# Load some test data
human_texts = ["Sample human text..."]
ai_texts = ["Sample AI text..."]

results = bench.run(human_texts, ai_texts)
bench.save_results(results)
bench.generate_plots(results)
```

### 5. Run Baseline Evaluation

Before training, evaluate baseline detectors on the datasets:

```bash
python scripts/run_stealthbench.py \
    --config configs/stealthbench.yaml \
    --output_dir outputs/baseline_evaluation
```

This will:
- Run all detectors on human vs AI texts
- Compute AUROC, FPR@TPR metrics
- Measure ESL vs native fairness
- Generate comparison plots

### 6. Train StealthRL (When Ready)

```bash
python scripts/train_stealthrl.py \
    --config configs/stealthrl_small.yaml \
    --output_dir checkpoints/stealthrl_v1
```

**Note**: Training requires significant GPU resources. Start with a small model (Qwen 1.5B) and limited steps for testing.

---

## 🧪 Testing & Validation

### Unit Tests to Create

1. **Test Detectors**:
   - Verify each detector loads correctly
   - Test detection on known human/AI samples
   - Validate output score ranges (0-1)

2. **Test Rewards**:
   - Test semantic reward with identical and very different texts
   - Test quality reward on high/low quality samples
   - Test fairness reward on ESL vs native data

3. **Test Evaluation**:
   - Test metrics computation with known data
   - Verify StealthBench aggregates results correctly
   - Check plot generation

4. **Test Training**:
   - Test LoRA setup
   - Test data loading
   - Test reward computation in training loop

### Integration Test

Create `tests/test_integration.py`:

```python
def test_full_pipeline():
    """Test the complete StealthRL pipeline."""
    # 1. Load data
    # 2. Initialize detectors
    # 3. Compute rewards
    # 4. Run evaluation
    # 5. Verify outputs
    pass
```

---

## 📊 Initial Experiments

### Experiment 0: Run Ablation Studies (RECOMMENDED)

**Goal**: Understand contribution of each reward component

StealthRL provides pre-configured ablation experiments to systematically test the impact of each reward term:

**Ablations Included**:
1. **Single Detector** (vs ensemble): `configs/ablations/single_detector_fast_detectgpt.yaml`
2. **No Fairness**: `configs/ablations/no_fairness.yaml` - removes ESL penalty
3. **No Semantic Fidelity**: `configs/ablations/no_semantic.yaml` - removes BERTScore
4. **No Quality**: `configs/ablations/no_quality.yaml` - removes perplexity/readability
5. **Detector Only**: `configs/ablations/detector_only.yaml` - pure evasion, no constraints

**Run All Ablations**:
```bash
bash scripts/run_ablations.sh
```

**Evaluate All Ablations**:
```bash
python scripts/evaluate_ablations.py \
    --ablation_dir checkpoints \
    --test_data data/processed/test.jsonl \
    --esl_data data/processed/esl_test.jsonl \
    --native_data data/processed/native_test.jsonl \
    --output_dir outputs/ablations
```

**Outputs**:
- `ablation_results.csv` - comprehensive comparison table
- `ablation_detector_scores.png` - detector evasion comparison
- `ablation_bertscore.png` - semantic fidelity comparison
- `ablation_fairness_gap.png` - ESL fairness comparison

**Expected Insights**:
- Single-detector training should show worse cross-detector transfer to Binoculars
- Removing fairness penalty should increase ESL FPR gap
- Removing semantic/quality terms may improve evasion but degrade output quality
- Detector-only should achieve best evasion but produce degenerate outputs

**Expected Time**: 2-4 days for all ablations (can parallelize)

### Experiment 1: Baseline Detector Evaluation

**Goal**: Understand baseline detector performance

**Steps**:
1. Evaluate all detectors on test set
2. Compute AUROC, FPR@0.5%, FPR@1%
3. Measure ESL vs native FPR gap
4. Identify which detectors are most biased

**Expected Time**: 1-2 hours

### Experiment 2: Single Detector Training

**Goal**: Verify training pipeline works

**Steps**:
1. Train with just Fast-DetectGPT as reward
2. Use small model (1.5B params) with LoRA
3. Train for 1000 steps
4. Evaluate on test set
5. Check if detector scores decrease

**Expected Time**: 4-8 hours on single GPU

### Experiment 3: Ensemble Training

**Goal**: Test full StealthRL with detector ensemble

**Steps**:
1. Train with Fast-DetectGPT + Ghostbuster ensemble
2. Include semantic fidelity and quality terms
3. Train for 5000-10000 steps
4. Evaluate on all detectors (including held-out Binoculars)
5. Measure cross-detector transfer

**Expected Time**: 12-24 hours on single GPU

### Experiment 4: Fairness Analysis

**Goal**: Measure and optimize ESL fairness

**Steps**:
1. Include fairness penalty in reward
2. Train with varying fairness weights (0.0, 0.1, 0.2)
3. Evaluate ESL vs native FPR gap before/after
4. Compare trade-off: detectability vs fairness

**Expected Time**: 8-16 hours

### Experiment 5: Baseline Comparison (DIPPER, SICO)

**Goal**: Benchmark StealthRL against prior evasion methods

StealthRL includes a comprehensive baseline comparison script:

```bash
python scripts/compare_baselines.py \
    --input_file data/processed/test.jsonl \
    --stealthrl_model checkpoints/stealthrl-small \
    --run_dipper \
    --run_sico \
    --output_csv outputs/baseline_comparison.csv
```

**Baselines Compared**:
- **Original** (no paraphrasing) - upper bound on detectability
- **DIPPER** (paraphrase-based evasion, NeurIPS'23) - requires installation
- **SICO** (prompt-based evasion, TMLR'24) - requires installation
- **StealthRL** (RL-based ensemble evasion) - your trained model

**Metrics**:
- Mean detector scores across all detectors (evasion effectiveness)
- BERTScore F1 (semantic fidelity)
- Perplexity (output quality)

**Installation Requirements**:
```bash
# DIPPER
pip install dipper-paraphrases
# Or: https://github.com/martiansideofthemoon/ai-detection-paraphrases

# SICO
# Clone: https://github.com/ColinLu50/Evade-GPT-Detector
```

**Note**: Script gracefully handles missing baselines (warns and skips).

**Expected Results**:
- StealthRL should show lower detector scores than DIPPER
- Better cross-detector transfer than single-detector baselines
- Higher semantic fidelity than unconstrained evasion methods

**Expected Time**: 2-4 hours (if baselines installed)

---

## 🐛 Potential Issues & Solutions

### Issue 1: GPU Memory

**Problem**: Model + detectors may exceed GPU memory

**Solutions**:
- Use smaller base model (e.g., 1.5B instead of 7B)
- Use gradient checkpointing
- Reduce batch size
- Offload detectors to CPU during training

### Issue 2: Slow Detector Inference

**Problem**: Running multiple detectors in the reward loop is slow

**Solutions**:
- Cache detector scores for validation set
- Use faster approximations during training
- Implement batched detector inference
- Use lighter detector models

### Issue 3: Data Format Mismatches

**Problem**: Downloaded datasets have different formats

**Solutions**:
- Implement robust data loading in `prepare_data.py`
- Handle JSONL, JSON, CSV, TXT formats
- Standardize on single output format
- Add data validation checks

### Issue 4: Reward Scale Imbalance

**Problem**: Different reward terms have different scales

**Solutions**:
- Normalize each reward component to 0-1
- Use adaptive reward weighting
- Log reward statistics during training
- Tune reward weights empirically

---

## 📝 Documentation Tasks

### Code Documentation

- [ ] Add usage examples to each module docstring
- [ ] Document expected input/output formats
- [ ] Add type hints consistently
- [ ] Document configuration options

### Tutorial Notebooks

- [ ] `notebooks/01_data_preparation.ipynb`
- [ ] `notebooks/02_detector_evaluation.ipynb`
- [ ] `notebooks/03_training_stealthrl.ipynb`
- [ ] `notebooks/04_analyzing_results.ipynb`
- [ ] `notebooks/05_fairness_analysis.ipynb`

### Experimental Documentation

- [ ] Document baseline detector results
- [ ] Document training hyperparameters
- [ ] Track experiment configurations
- [ ] Record observations and insights
- [ ] Create results visualization dashboards

---

## 🎯 Research Milestones

### Short-term (1-2 weeks)

- [ ] Complete baseline evaluation of all detectors
- [ ] Train first StealthRL model successfully
- [ ] Demonstrate detector score reduction
- [ ] Measure semantic preservation (BERTScore)

### Medium-term (3-4 weeks)

- [ ] Train ensemble-based StealthRL
- [ ] Evaluate cross-detector transfer
- [ ] Complete fairness analysis (ESL vs native)
- [ ] Run ablation studies
- [ ] Compare against SICO baseline

### Long-term (1-2 months)

- [ ] Optimize for best performance
- [ ] Explore different detector combinations
- [ ] Study defender perspective
- [ ] Write technical report/paper
- [ ] Prepare for publication

---

## 🤝 Collaboration & Sharing

### Code Sharing

- Consider creating GitHub repository (with caution about responsible disclosure)
- Share evaluation harness (StealthBench) as standalone tool
- Release anonymized experimental results
- **Do NOT release evasion-tuned model weights**

### Community Engagement

- Share StealthBench results with detector developers
- Contribute to improving detector robustness
- Participate in AI safety discussions
- Emphasize responsible research practices

---

## 📚 Additional Resources

### Related Papers to Review

- AuthorMist (arXiv:2503.08716) - Primary inspiration
- DetectGPT original paper - Understanding curvature-based detection
- Ghostbuster (NAACL'24) - Feature ensemble methods
- Binoculars (ICML'24) - Paired-LM detection
- DIPPER (NeurIPS'23) - Paraphrase evasion
- SICO - Prompt-based evasion

### Relevant Codebases

- HuggingFace TRL documentation: https://huggingface.co/docs/trl
- PEFT (LoRA) documentation: https://huggingface.co/docs/peft
- DetectGPT implementation: https://github.com/eric-mitchell/detect-gpt
- Fast-DetectGPT: https://github.com/baoguangsheng/fast-detect-gpt

---

## ✉️ Questions or Issues?

If you encounter problems:

1. Check `interaction_records.md` for implementation details
2. Review error messages carefully
3. Check dependency versions
4. Review the code comments and docstrings
5. Consult the referenced papers and codebases

---

**Last Updated**: November 25, 2025  
**Status**: All implementations complete, ready for testing and experimentation
