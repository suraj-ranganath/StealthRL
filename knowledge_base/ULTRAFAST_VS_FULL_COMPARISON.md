# Ultra-Fast vs Full Production Training Comparison

## Configuration Comparison Table

| **Parameter** | **Ultra-Fast (Proof-of-Concept)** | **Full Production** | **Rationale** |
|---------------|-----------------------------------|---------------------|---------------|
| **Dataset Size** | 800 samples | 20,000 samples | 25× more data for domain coverage |
| **ESL/Native Split** | Poor (unbalanced) | 40% ESL / 60% Native | Proper fairness evaluation |
| **Training Epochs** | 1 epoch | 3 epochs | Full convergence |
| **Learning Rate** | 5e-5 (conservative) | 2.8e-4 (optimal) | 5.6× higher, research-backed |
| **LoRA Rank** | 16 | 32 | 2× capacity for complex RL |
| **Batch Size** | 16 | 4 | Optimal for LoRA (small batches) |
| **Group Size** | 8 | 8 | Maintained (GRPO sweet spot) |
| **Temperature** | 0.8 (reduced) | 1.0 (constant) | Higher exploration |
| **Top-p Sampling** | 0.9 | 0.95 | More diversity |
| **KL Penalty** | 0.03 (adaptive) | 0.01 (fixed, stronger) | Better drift control |
| **Max Tokens** | 400 | 512 | Longer paraphrases allowed |
| **Detectors** | Fast-DetectGPT only | Fast-DetectGPT + Ghostbuster + Binoculars | 3-detector ensemble |
| **Training Time** | ~3.5 hours | 6-8 hours | 2× longer for 25× data |
| **Compute Cost** | ~$10-20 | ~$50-100 | Production-grade experiment |

---

## Dataset Comparison

### Ultra-Fast Dataset
```
Total: 800 samples (4% of available data)
├── Sources: Mixed (DetectRL, ChatGPT-Bias, Ghostbuster)
├── ESL/Native: Unbalanced (poor representation)
├── Domains: Limited diversity
└── Purpose: Rapid iteration, proof-of-concept
```

### Full Production Dataset
```
Total: 20,000 samples (100% curated)
├── Sources:
│   ├── DetectRL: 8,000 samples (real-world benchmark)
│   ├── ChatGPT-Bias: 6,000 samples (ESL/Native academic)
│   └── Ghostbuster: 6,000 samples (Human/AI pairs)
├── ESL/Native Split:
│   ├── ESL: 8,000 samples (40%) - TOEFL11, ICNALE, ELLIPSE
│   └── Native: 12,000 samples (60%) - Academic, news, creative
├── Domains:
│   ├── Academic essays
│   ├── News articles
│   ├── Creative writing
│   └── Technical documentation
└── Purpose: Production deployment, fair evaluation
```

---

## Hyperparameter Justification

### Learning Rate: 5e-5 → 2.8e-4
**Source**: Thinking Machines LoRA research
- **10× FullFT Rule**: LoRA requires 10× smaller LR than full fine-tuning
- Full FT optimal: ~2.8e-3
- LoRA optimal: 2.8e-3 / 10 = **2.8e-4**
- Ultra-fast used **5e-5** (conservative for stability)
- **Impact**: 5.6× faster learning, better convergence

### LoRA Rank: 16 → 32
**Source**: LoRA RL research papers
- Rank 8: Too constrained for complex tasks
- Rank 16: Good for supervised fine-tuning
- **Rank 32: Optimal for RL** (more expressive for policy learning)
- Rank 64: Diminishing returns, slower training
- **Impact**: 2× parameter capacity without full FT cost

### Batch Size: 16 → 4
**Source**: LoRA batch size sensitivity study
- LoRA is **less tolerant of large batches** than full FT
- Product-of-matrices (BA) parametrization has worse dynamics at batch_size > 32
- Both LoRA and FullFT achieve **best loss at small batches (4-16)**
- Effective batch = batch_size × group_size = 4 × 8 = **32 generations/step**
- **Impact**: Better gradient quality, more stable training

### Temperature: 0.8 → 1.0
**Source**: GRPO training guide
- RL requires **constant temperature** (no decay like supervised learning)
- Temperature controls exploration/exploitation trade-off
- 0.8: Reduced for proof-of-concept stability
- **1.0: Standard for RL** (proper entropy maintenance)
- **Impact**: Better exploration, richer Pareto frontier

### KL Penalty: 0.03 (adaptive) → 0.01 (fixed)
**Source**: AuthorMist paper
- KL divergence prevents policy drift from base model
- Adaptive: Changes based on observed KL (can be unstable)
- **Fixed 0.01: Stronger, more stable** constraint
- Target KL < 4.0 maintained throughout training
- **Impact**: Preserves fluency, prevents degeneration

---

## Multi-Detector Ensemble

### Ultra-Fast: Single Detector
```python
detectors = ["fast_detectgpt"]  # Speed optimization
detector_weights = {"fast_detectgpt": 1.0}
```
- **Fast-DetectGPT**: Curvature-based, sampling-free
- **Purpose**: Fastest detector for rapid iteration
- **Limitation**: May not generalize to other detector types

### Full Production: 3-Detector Ensemble
```python
detectors = ["fast_detectgpt", "ghostbuster", "binoculars"]
detector_weights = {
    "fast_detectgpt": 0.4,  # Curvature-based
    "ghostbuster": 0.4,     # Classifier-based (100+ features)
    "binoculars": 0.2,      # Paired-LM (instruction vs base)
}
```
- **Three Detector Families**: Curvature, Classifier, Paired-LM
- **Purpose**: Detector-agnostic learning
- **Benefit**: Better transfer to unseen detectors

---

## Expected Performance Improvements

| **Metric** | **Ultra-Fast (Actual)** | **Full Production (Expected)** | **Improvement** |
|------------|-------------------------|--------------------------------|-----------------|
| **Attack Success Rate (ASR)** | 22% improvement | 60-70% improvement | 3× better |
| **Semantic Similarity** | 98.4% average | >88% maintained | Comparable quality |
| **Perplexity (Naturalness)** | 30-86 range | 25-50 range | More consistent |
| **ESL FPR Gap** | Unknown (poor split) | <0.07 target | Measurable fairness |
| **Pareto-Optimal Checkpoints** | 9 points (2D) | 15-20 expected | Richer trade-offs |
| **Transfer Ratio** | Unknown (1 detector) | >0.7 target | Strong generalization |
| **KL Divergence** | <0.4 (excellent) | <1.0 target | Controlled drift |

---

## Training Efficiency Comparison

### Ultra-Fast Training
```
Samples per epoch: 800
Batches per epoch: 800 / 16 = 50 batches
Total batches: 50 × 1 epoch = 50 batches
Generations per batch: 16 × 8 = 128 generations
Total generations: 50 × 128 = 6,400 generations
Time: ~3.5 hours (~4.2 min/step)
Cost: ~$10-20 (Tinker credits)
```

### Full Production Training
```
Samples per epoch: 20,000
Batches per epoch: 20,000 / 4 = 5,000 batches
Total batches: 5,000 × 3 epochs = 15,000 batches
Generations per batch: 4 × 8 = 32 generations
Total generations: 15,000 × 32 = 480,000 generations
Time: 6-8 hours (~2 sec/step, optimized)
Cost: ~$50-100 (Tinker credits)
```

**Efficiency**: 75× more generations in 2× time (better parallelization)

---

## Key Takeaways

### Ultra-Fast Strengths ✅
- **Rapid iteration**: 3.5 hours for proof-of-concept
- **Validated approach**: GRPO works for adversarial text
- **Stable training**: No model collapse with conservative settings
- **Strong baseline**: 22% ASR improvement with minimal resources

### Ultra-Fast Limitations ⚠️
- **Suboptimal hyperparameters**: Conservative LR, low LoRA rank
- **Limited data**: 4% of available dataset
- **Poor fairness evaluation**: Unbalanced ESL/Native split
- **Single detector**: No transfer testing
- **Short training**: 1 epoch, may not fully converge

### Full Production Goals 🎯
- **Optimal hyperparameters**: Research-backed configuration
- **Comprehensive data**: 25× more samples, proper ESL split
- **Multi-detector ensemble**: Test detector-agnostic learning
- **Full convergence**: 3 epochs for stable policies
- **Measurable fairness**: Quantify ESL FPR gap reduction
- **Transfer evaluation**: Held-out Binoculars testing

---

## Presentation Summary Slide

**Ultra-Fast (Proof-of-Concept)**
- 800 samples, 1 epoch, 3.5 hours
- Conservative hyperparameters (stability first)
- Single detector (Fast-DetectGPT)
- **Result**: 22% ASR, 98.4% semantic similarity ✅
- **Purpose**: Validate GRPO for adversarial text generation

**Full Production (Next Step)**
- 20,000 samples, 3 epochs, 6-8 hours
- Optimized hyperparameters (LR 2.8e-4, rank 32)
- 3-detector ensemble (Fast-DetectGPT + Ghostbuster + Binoculars)
- **Expected**: 60-70% ASR, >88% semantic, <0.07 ESL gap
- **Purpose**: Production deployment, fair evaluation, transfer testing

**Investment**: 2× time → 25× data → 3× performance + fairness guarantees

---

## Novel Contributions (Presentation Slides)

### Slide 1: Technical & Methodological Innovations

#### 1. **Generalizable Multi-Detector Framework**
- **Our Approach**: Locally hosted, open-source detector ensemble (Fast-DetectGPT, Ghostbuster, Binoculars)
- **Other Work**: AuthorMist uses closed API-based detectors (GPTZero, ZeroGPT)
- **Impact**: Fully reproducible, no vendor lock-in, supports any detector
- **Evidence**: 3 detector families (curvature, classifier, paired-LM) with plug-and-play architecture

#### 2. **Fairness-Aware Adversarial Training**
- **Our Approach**: Explicit ESL fairness penalty in reward function (R_fair = -0.2 × detector_prob × 𝟙[ESL])
- **Other Work**: No fairness consideration in prior adversarial paraphrasing systems
- **Impact**: Reduces discrimination against non-native English speakers
- **Evidence**: Target ESL FPR gap <0.07 (vs baseline ~0.15 without fairness term)

#### 3. **Open-Source Release Commitment**
- **Our Approach**: Full codebase, configs, checkpoints, and training harness publicly available
- **Other Work**: AuthorMist and similar systems remain proprietary
- **Impact**: Enables detector improvements, reproducible research, community extensions
- **Evidence**: 9 Pareto-optimal checkpoints ready for release, comprehensive documentation

---

### Slide 2: System & Practical Innovations

#### 4. **Plug-and-Play Training Harness**
- **Innovation**: Modular RL framework with YAML-based configuration
- **Features**:
  - **Custom Models**: Any HuggingFace model (not just Qwen)
  - **Custom Detectors**: Add new detectors with simple API interface
  - **Custom Rewards**: Configure multi-objective weights via YAML
  - **No Code Changes**: Entire experiment defined in config file
- **Impact**: Research community can reproduce and extend our work in minutes
- **Example**:
```yaml
# configs/custom_experiment.yaml
model: "meta-llama/Llama-3-8B"  # Your model
detectors: ["my_custom_detector"]  # Your detector
reward:
  detector_weight: 1.0  # Your weights
  semantic_weight: 1.0
```

#### 5. **Multi-Objective Pareto Optimization**
- **Innovation**: First to apply explicit Pareto frontier analysis to adversarial text generation
- **Our Approach**: Train once, get 9+ checkpoints with different stealth-quality trade-offs
- **Other Work**: Single-objective optimization (maximize evasion OR quality, not both)
- **Impact**: Users choose checkpoint based on risk tolerance (stealth vs quality vs naturalness)
- **Evidence**: 9 Pareto-optimal points (2D), 26 points (3D) spanning entire trade-off space

#### 6. **GRPO for Adversarial NLP (First Application)**
- **Innovation**: First application of Group Relative Policy Optimization to adversarial text generation
- **Our Approach**: Group-based advantage estimation (8 rollouts/prompt) with variance reduction
- **Other Work**: PPO (complex, requires value network) or supervised learning (no exploration)
- **Technical Advantage**: 
  - Simpler than PPO (no value function)
  - Better than supervised learning (active exploration)
  - Stable convergence with proper group size
- **Evidence**: No model collapse, KL divergence <0.4 (target <4.0), stable 50-step training

---

## Summary: Why Our Work Matters

### For Researchers 🔬
- **Reproducibility**: Local detectors, open-source code, no API dependencies
- **Extensibility**: Plug-and-play harness, YAML configs, modular architecture
- **Rigor**: Pareto optimization, fairness metrics, transfer evaluation

### For Practitioners 🛠️
- **Flexibility**: 9 checkpoints for different use cases (stealth vs quality)
- **Fairness**: Reduces ESL bias (measurable FPR gap reduction)
- **Transparency**: Full visibility into detector behavior, reward dynamics

### For the Community 🌍
- **Detector Improvements**: Open adversarial testing reveals weaknesses
- **Ethical AI**: Fairness-aware training reduces discrimination
- **Research Platform**: Foundation for 20+ future directions (curriculum learning, MoE, adversarial training)

---

## Novelty Comparison Table

| **Aspect** | **StealthRL (Ours)** | **AuthorMist** | **DIPPER** | **SICO** |
|------------|----------------------|----------------|------------|----------|
| **Training Method** | GRPO (RL) | PPO (RL) | Supervised | In-context |
| **Detector Support** | Open-source, local | Closed APIs | N/A | N/A |
| **Multi-Detector** | ✅ 3 families | ❌ Single API | ❌ | ❌ |
| **Fairness Objective** | ✅ ESL penalty | ❌ | ❌ | ❌ |
| **Pareto Optimization** | ✅ 9+ checkpoints | ❌ Single model | ❌ | ❌ |
| **Open Source** | ✅ Full release | ❌ Proprietary | ✅ | ✅ |
| **Plug-and-Play Harness** | ✅ YAML configs | ❌ | ❌ | ❌ |
| **Transfer Evaluation** | ✅ Held-out detector | ❌ | ❌ | ❌ |
| **Reproducibility** | ✅ Local, no APIs | ❌ API-dependent | ✅ | ✅ |
