# StealthRL: Tinker Integration Guide

## 🎯 Overview

StealthRL has been fully integrated with the **Tinker platform** for RL training, sponsored by DSC 291 - Safety in Generative AI. This integration provides:

- **Remote compute**: Train on `Qwen/Qwen3-4B-Instruct-2507` without local GPU requirements
- **GRPO training**: Group-based RL with reward centering and variance reduction
- **LoRA adapters**: Efficient fine-tuning (8-16 rank, 20-100× LR scaling)
- **Multi-objective rewards**: Detector evasion, semantic fidelity, fluency, ESL fairness
- **Production features**: SQLite detector caching, KL regularization, curriculum learning

## 🏗️ Architecture

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Preparation                         │
│  (AI texts + human references + domain + ESL flags)         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              StealthRLDatasetBuilder                         │
│  - Load train/test JSONL                                    │
│  - Initialize reward function                               │
│  - Create environment groups                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               StealthRL Environment (RL)                     │
│                                                              │
│  ┌──────────────────────────────────────────────┐          │
│  │  Prompt: "Paraphrase: {ai_text}"             │          │
│  └────────────┬─────────────────────────────────┘          │
│               │                                             │
│               ▼                                             │
│  ┌──────────────────────────────────────────────┐          │
│  │  Policy π (Qwen3-4B + LoRA)                  │          │
│  │  Generates: paraphrase_text                   │          │
│  └────────────┬─────────────────────────────────┘          │
│               │                                             │
│               ▼                                             │
│  ┌──────────────────────────────────────────────┐          │
│  │  Reward = R_det + R_sem + R_ppl - R_fair     │          │
│  │  - R_det: Detector ensemble (Fast-DetectGPT,│          │
│  │           Ghostbuster, Binoculars)           │          │
│  │  - R_sem: Semantic similarity (E5)           │          │
│  │  - R_ppl: Perplexity (GPT-2, target band)   │          │
│  │  - R_fair: ESL penalty (per-sample)          │          │
│  └────────────┬─────────────────────────────────┘          │
└───────────────┼─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│               GRPO Training Loop                             │
│  - Group-based reward centering (variance reduction)        │
│  - KL penalty: L = -E[R] + β*KL(π || π_ref)                │
│  - Advantage normalization & clipping                       │
│  - All-negative group handling                              │
│  - Curriculum learning (optional)                           │
│  - Temperature scheduling                                   │
└─────────────────────────────────────────────────────────────┘
```

### Reward Function

**Formula**:
```
R_total = α*R_det + β*R_sem + γ*R_ppl - δ*F'

Where:
  R_det = 1 - P(AI)  (weighted detector ensemble)
  R_sem = max(0, similarity - 0.90)  (E5 cosine)
  R_ppl = reward(perplexity, target=30)  (GPT-2)
  F' = P(AI) * 𝟙[is_ESL]  (per-sample fairness)

Normalization (Session 4 refinements):
  R_det' = zscore(R_det).clamp(-3, 3)
  R_sem' = (R_sem - 0.90) / 0.10 if R_sem ≥ 0.90 else 0
  R_ppl' = (R_ppl - 0.80) / 0.20 if R_ppl ≥ 0.80 else 0
```

**Default Weights**:
- α = 1.0 (detector evasion)
- β = 1.0 (semantic similarity)
- γ = 0.5 (perplexity/fluency)
- δ = 0.2 (ESL fairness)

## 🚀 Quickstart

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-org/StealthRL.git
cd StealthRL

# Install Tinker dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your TINKER_API_KEY
```

### 2. Prepare Data

```bash
# Option A: Create synthetic data for testing
python scripts/prepare_tinker_data.py \\
    --synthetic \\
    --num-train 100 \\
    --num-test 20 \\
    --output-dir data/tinker

# Option B: Prepare from existing datasets
python scripts/prepare_tinker_data.py \\
    --input-paths data/raw/ai_texts.jsonl data/raw/human_texts.jsonl \\
    --output-dir data/tinker \\
    --train-split 0.8
```

**Expected data format** (`train.jsonl` / `test.jsonl`):
```json
{
  "ai_text": "The implementation of neural networks requires careful consideration...",
  "human_reference": "Building neural networks demands thoughtful selection...",
  "domain": "academic",
  "is_esl": false,
  "metadata": {"model_family": "gpt"}
}
```

### 3. Configure Training

Edit `configs/tinker_stealthrl.yaml`:

```yaml
# Key hyperparameters
model:
  name: "Qwen/Qwen3-4B-Instruct-2507"

lora:
  rank: 16  # 8-16 for RL

training:
  learning_rate: 1.0e-5  # Will be scaled 20-100× for LoRA
  batch_size: 8
  group_size: 4  # GRPO rollouts per prompt
  num_epochs: 3

kl:
  penalty_coef: 0.001  # AuthorMist β

reward:
  detector_weight: 1.0
  semantic_weight: 1.0
  perplexity_weight: 0.5
  fairness_weight: 0.2
```

### 4. Train on Tinker

```bash
# Export API key
export TINKER_API_KEY=your_tinker_api_key

# Run training
python -m stealthrl.tinker.train \\
    --config configs/tinker_stealthrl.yaml \\
    --data-path data/tinker \\
    --output-dir outputs/tinker
```

**Training metrics logged to**:
- `outputs/tinker/metrics.jsonl` - Step-level metrics (reward, KL, etc.)
- `outputs/tinker/debug_samples.jsonl` - Sample paraphrases with rewards
- TensorBoard: `tensorboard --logdir outputs/tinker`

### 5. Inference

#### Basic Inference

```python
from stealthrl.tinker import StealthEnv, TinkerCompositeReward
import tinker

# Load trained model
client = tinker.SamplingClient(api_key="your_api_key")
model = client.load_model("path/to/checkpoint")

# Create reward function (for evaluation)
reward_fn = TinkerCompositeReward(
    detector_weight=1.0,
    semantic_weight=1.0,
    perplexity_weight=0.5,
    fairness_weight=0.2,
)

# Paraphrase text
ai_text = "The implementation of neural networks..."
paraphrase = model.generate(
    f"Please paraphrase: {ai_text}",
    max_tokens=512
)

# Evaluate paraphrase
result = await reward_fn.compute(
    original_text=ai_text,
    paraphrase_text=paraphrase,
    human_reference="...",
    domain="academic",
    is_esl=False,
)

print(f"Total reward: {result['total_reward']:.3f}")
print(f"Detector prob: {result['detector_prob']:.3f}")
print(f"Semantic sim: {result['semantic_sim']:.3f}")
```

#### Chunking Inference (for Long Texts)

For texts > 512 tokens, use the chunking pipeline:

```python
from stealthrl.tinker import ChunkingInference

# Initialize chunking inference
chunker = ChunkingInference(
    model=model,
    reward_fn=reward_fn,
    chunk_size=512,
    overlap=50,
    num_candidates=4,  # Generate 4 candidates per chunk
    temperature=0.9,
)

# Paraphrase long text
long_text = "..." * 1000  # Long AI-generated text
result = await chunker.paraphrase(
    text=long_text,
    human_reference="...",
    domain="academic",
    is_esl=False,
)

print(f"Original: {len(long_text)} chars")
print(f"Paraphrase: {len(result['paraphrase'])} chars")
print(f"Chunks processed: {result['num_chunks']}")
print(f"Avg reward: {result['avg_reward']:.3f}")
print(f"Avg detector prob: {result['avg_detector_prob']:.3f}")
```

**Chunking strategy**:
1. Split text into 512-token chunks with 50-token overlap
2. Generate N candidates per chunk (default: 4)
3. Select best candidate based on reward score
4. Merge chunks with overlap handling (prefer sentence boundaries)

### 6. Evaluation

Run comprehensive evaluation comparing base AI text, SFT baseline, and StealthRL:

```python
from stealthrl.tinker import EvaluationSuite, EvaluationExample

# Initialize evaluation suite
eval_suite = EvaluationSuite(
    detector_ensemble=detector_ensemble,
    semantic_similarity=semantic_similarity,
    detector_names=["fast_detectgpt", "ghostbuster", "binoculars"],
    output_dir="outputs/evaluation",
)

# Prepare evaluation examples
examples = [
    EvaluationExample(
        ai_text="...",
        human_reference="...",
        domain="academic",
        is_esl=False,
        metadata={"model_family": "gpt"},
    ),
    # ... more examples
]

# Run evaluation (generates paraphrases and scores)
examples = await eval_suite.evaluate_examples(
    examples=examples,
    base_model=base_ai_model,
    sft_model=sft_baseline,  # Optional
    stealthrl_model=stealthrl_policy,
)

# Generate comparison report
report = eval_suite.generate_comparison_report(examples, has_sft=True)

# Print summary
eval_suite.print_summary(report)

# Save to JSON
eval_suite.save_report(report, examples, filename="results.json")
```

**Metrics computed**:
- **ASR (Attack Success Rate)**: Fraction evading all detectors
- **AUROC**: Area under ROC curve per detector
- **F1**: F1 score per detector at threshold 0.5
- **Semantic similarity**: Mean, std, min across paraphrases
- **ESL fairness gap**: FPR(ESL) - FPR(native) per detector
- **Detector probabilities**: Average P(AI) for base, SFT, StealthRL

## 📊 Monitoring Training

### Key Metrics

Monitor these in `metrics.jsonl` or TensorBoard:

- **reward/total**: Overall reward (target: increase)
- **reward/detector**: Detector evasion (target: high)
- **reward/semantic**: Semantic similarity (target: high)
- **reward/perplexity**: Fluency (target: stable around 1.0)
- **reward/fairness**: ESL penalty (target: low negative)
- **kl**: KL divergence from reference (target: < 0.01)
- **all_negative_frac_total**: Fraction of all-negative groups (target: decrease)
- **temperature**: Current sampling temperature
- **kl_penalty_coef**: Adaptive KL coefficient

### Debugging

**All-negative groups too high (>50%)**:
- Detectors too strong or rewards too sparse
- Lower `detector_weight` or increase `semantic_weight`
- Enable curriculum learning: start with easier examples

**KL too high (>0.1)**:
- Increase `kl.penalty_coef` from 0.001 to 0.01
- Or enable adaptive KL: set `kl.target = 0.01`

**Semantic similarity low (<0.85)**:
- Increase `semantic_weight` from 1.0 to 2.0
- Check paraphrase quality in `debug_samples.jsonl`

**Training unstable**:
- Reduce `temperature` from 1.0 to 0.9
- Increase `advantage_clip` from 5.0 to 10.0
- Enable `temperature_schedule: decay`

## 🔬 Advanced Features

### 1. Curriculum Learning

Start training on easier examples (high baseline rewards), gradually introduce harder ones:

```yaml
curriculum:
  enabled: true
  start_quantile: 0.7  # Top 70% easiest
  end_quantile: 0.0  # All examples
  steps: 1000  # Transition steps
```

### 2. Adaptive KL Penalty

Automatically adjust KL coefficient to maintain target divergence:

```yaml
kl:
  penalty_coef: 0.001  # Initial β
  target: 0.01  # Target KL divergence
  adapt_rate: 0.1  # Adaptation speed
```

### 3. Temperature Scheduling

Decay exploration over training:

```yaml
sampling:
  temperature: 1.0
  temperature_schedule: "decay"
  temperature_decay: 0.95  # Multiply by 0.95 each epoch
```

### 4. Custom Detector Ensemble

Add/remove detectors or adjust weights:

```yaml
reward:
  detectors:
    names:
      - "fast_detectgpt"
      - "ghostbuster"
      - "binoculars"
    weights:
      fast_detectgpt: 0.4
      ghostbuster: 0.3
      binoculars: 0.3
    cache_path: "cache/detectors.db"
```

### 5. All-Negative Group Handling

Prevent training collapse when all candidates get negative rewards:

```yaml
all_negative:
  min_reward: 0.01  # Small shaped signal
  downweight: 0.5  # Reduce contribution
```

## 📈 Comparison: TRL vs Tinker

| Feature | Original (HuggingFace TRL) | Tinker Integration |
|---------|----------------------------|-------------------|
| **Compute** | Local GPUs required | Remote (Tinker cloud) |
| **Model** | Qwen 1.5B | Qwen3-4B (2×  larger) |
| **RL Algorithm** | GRPO/PPO (TRL) | GRPO (Tinker native) |
| **LoRA** | Manual LR scaling | Automatic via `get_lora_lr_over_full_finetune_lr` |
| **Reward Compute** | Synchronous | Async (non-blocking) |
| **Caching** | None | SQLite (persistent) |
| **KL Penalty** | Manual | Integrated with adaptive option |
| **GRPO Enhancements** | Basic | Curriculum, temp schedule, all-negative handling |
| **Logging** | Custom | ML logger (JSONL) + TensorBoard |
| **Config** | Python args | YAML + `@chz.chz` decorators |

**Migration Benefits**:
- ✅ No GPU setup required (Tinker handles compute)
- ✅ Larger model (Qwen3-4B vs 1.5B)
- ✅ Automatic LoRA LR scaling
- ✅ Persistent detector caching (avoid re-computation)
- ✅ Production-ready GRPO features
- ✅ Cleaner async architecture

## 🛠️ Troubleshooting

### API Key Issues

```bash
# Verify API key is set
echo $TINKER_API_KEY

# Test connection
python -c "import tinker; print(tinker.ServiceClient(api_key='$TINKER_API_KEY').ping())"
```

### Cache Errors

```bash
# Clear detector cache if corrupted
rm cache/detectors.db

# Check cache path in config
grep cache_path configs/tinker_stealthrl.yaml
```

### Import Errors

```bash
# Verify tinker-cookbook installed
pip show tinker-cookbook

# Reinstall if needed
pip install --upgrade tinker-ai tinker-cookbook
```

### Memory Issues

Reduce batch size or group size:

```yaml
training:
  batch_size: 4  # Reduce from 8
  group_size: 2  # Reduce from 4
```

## 📚 Additional Resources

- **Tinker Documentation**: https://tinker-docs.thinkingmachines.ai/
- **LoRA Primer**: https://tinker-docs.thinkingmachines.ai/lora-primer
- **RL Guide**: https://tinker-docs.thinkingmachines.ai/rl
- **GRPO Paper**: [Group Relative Policy Optimization](https://arxiv.org/abs/...)
- **AuthorMist**: [arXiv:2503.08716](https://arxiv.org/abs/2503.08716) (KL penalty reference)

## 🎓 DSC 291 Context

This Tinker integration was developed for **DSC 291: Safety in Generative AI** at UCSD. Tinker has sponsored the class with API credits for RL training.

**Course Goals Addressed**:
- ✅ Detector-robust text generation (safety evaluation)
- ✅ Fairness in AI detection (ESL bias mitigation)
- ✅ Verifiable rewards (reject degenerate outputs)
- ✅ Production-ready RL system (not toy example)

**Research Questions**:
1. Can ensemble training transfer to unseen detectors? (Binoculars held-out)
2. Can RL reduce ESL false-positive bias? (Fairness reward term)
3. What's the Pareto frontier? (Reward weight sweeps)
4. How does StealthRL compare to DIPPER/SICO? (Baseline comparisons)

---

**For questions or issues**, please open a GitHub issue or contact the maintainers.
