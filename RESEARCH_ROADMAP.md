# StealthRL Research Roadmap - DSC 291

**Status as of November 25, 2025**

This document tracks the research implementation roadmap for the DSC 291 project proposal.

---

## ✅ Phase 1: Core Infrastructure (COMPLETE)

### Implementation Status
- ✅ Tinker integration with GRPO training
- ✅ LoRA-based paraphraser (Qwen3-4B)
- ✅ Multi-detector ensemble reward (Fast-DetectGPT, Ghostbuster, Binoculars)
- ✅ Semantic fidelity (E5 cosine similarity)
- ✅ Quality controls (perplexity banding with GPT-2)
- ✅ ESL fairness penalty (per-sample fairness proxy)
- ✅ Chunking inference for long texts (>512 tokens)
- ✅ Comprehensive evaluation suite (ASR, AUROC, F1, ESL gaps)
- ✅ Ablation configurations (detector-only, no-fairness, no-quality, no-semantic)

**Lines of Code**: ~3,555 lines across 14 Tinker modules

---

## 🔨 Phase 2: Research Experiments (IN PROGRESS)

### Priority 1: Core Research Questions (CRITICAL)

#### 1. Transfer Evaluation (Held-Out Detectors) 🎯
**Goal**: Answer central question: "Can ensemble training transfer to unseen detector families?"

**Implementation Needed**:
```python
# configs/tinker_transfer_eval.yaml
training:
  in_ensemble_detectors:
    - "fast_detectgpt"
    - "ghostbuster"
  
evaluation:
  held_out_detectors:
    - "binoculars"  # Paired-LM (different mechanism)
    # Add more if available: GPTZero, OpenAI detector, etc.
```

**Steps**:
1. Create training config with subset of detectors (e.g., Fast-DetectGPT + Ghostbuster)
2. Train StealthRL on in-ensemble detectors only
3. Evaluate on **both** in-ensemble and held-out detectors
4. Compute transfer metrics:
   - ASR (in-ensemble) vs ASR (held-out)
   - Detector prob drop (in-ensemble) vs (held-out)
   - Semantic similarity maintained for both
5. Compare against single-detector baseline

**Expected Script**: `scripts/evaluate_transfer.py`

**Deliverable**: Transfer analysis showing generalization to unseen detector families

---

#### 2. ESL-Stratified Evaluation Datasets 🎯
**Goal**: Enable fairness analysis with ESL vs native stratification

**Implementation Needed**:
1. **Dataset curation**:
   - Academic writing corpus with ESL labels
   - Potential sources:
     * TOEFL11 corpus (ESL student essays)
     * ICNALE (International Corpus Network of Asian Learners)
     * Native: ACL Anthology papers, academic blogs
   - Balance: ~40% ESL, 60% native (reflect real-world academic settings)

2. **Data preparation enhancement**:
   ```python
   # scripts/prepare_tinker_data.py enhancement
   def load_esl_corpus(source: str) -> List[dict]:
       """Load ESL-labeled corpus."""
       # TOEFL11, ICNALE, etc.
       pass
   
   def balance_esl_splits(examples: List[dict]) -> List[dict]:
       """Ensure train/test ESL representation."""
       pass
   ```

3. **Metadata requirements**:
   ```json
   {
     "ai_text": "...",
     "human_reference": "...",
     "domain": "academic",
     "is_esl": true,
     "esl_metadata": {
       "native_language": "chinese",
       "proficiency": "intermediate",
       "source": "toefl11"
     }
   }
   ```

**Deliverable**: ESL-stratified evaluation set with balanced representation

---

#### 3. Low-FPR Operating Point Evaluation 🎯
**Goal**: Evaluate at FPR@0.5% and FPR@1% (academic integrity thresholds)

**Implementation Needed**:
```python
# stealthrl/tinker/evaluation.py enhancement

def _compute_fpr_at_threshold(
    self,
    scores: np.ndarray,
    labels: np.ndarray,
    fpr_targets: List[float] = [0.005, 0.01],  # 0.5%, 1%
) -> Dict[float, float]:
    """Compute TPR at specific FPR thresholds."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    results = {}
    for fpr_target in fpr_targets:
        # Find threshold where FPR ≈ target
        idx = np.argmin(np.abs(fpr - fpr_target))
        results[fpr_target] = {
            'tpr': tpr[idx],
            'threshold': thresholds[idx],
            'actual_fpr': fpr[idx],
        }
    return results
```

**Add to ModelMetrics**:
```python
@dataclass
class ModelMetrics:
    # ... existing fields
    tpr_at_fpr_0_5: Dict[str, float]  # TPR at FPR=0.5% per detector
    tpr_at_fpr_1_0: Dict[str, float]  # TPR at FPR=1.0% per detector
```

**Deliverable**: Low-FPR metrics in evaluation reports

---

### Priority 2: Baselines & Comparisons (HIGH)

#### 4. SICO Baseline Implementation 🔧
**Goal**: Implement "strong, low-cost baseline" for comparison

**SICO Overview**:
- Prompt-space evasion technique
- Uses carefully crafted prompts to elicit paraphrases from LLMs
- No training required (zero-shot or few-shot prompting)

**Implementation**:
```python
# stealthrl/baselines/sico.py

class SICOParaphraser:
    """SICO prompt-based paraphrasing."""
    
    def __init__(
        self,
        model,  # Any LLM (GPT-3.5, GPT-4, Claude, etc.)
        prompt_template: str = "paraphrase_preserve_meaning",
    ):
        self.model = model
        self.prompt_template = self._load_template(prompt_template)
    
    def _load_template(self, template_name: str) -> str:
        """Load SICO prompt template."""
        templates = {
            "paraphrase_preserve_meaning": (
                "Rewrite the following text in a natural, human-like way "
                "while preserving its exact meaning. Vary sentence structure "
                "and word choice, but maintain clarity:\n\n{text}"
            ),
            "academic_style": (
                "Rewrite this academic text with natural phrasing, as if "
                "written by a knowledgeable human scholar:\n\n{text}"
            ),
            # Add more SICO variants
        }
        return templates[template_name]
    
    async def paraphrase(self, text: str) -> str:
        """Generate SICO paraphrase."""
        prompt = self.prompt_template.format(text=text)
        return await self.model.generate(prompt, max_tokens=512)
```

**Evaluation**:
- Run SICO through same evaluation suite
- Compare: StealthRL vs SICO vs Base AI text
- Metrics: ASR, semantic sim, detector scores, ESL fairness

**Deliverable**: SICO baseline results for comparison table

---

#### 5. Ablation Experiments 🔧
**Goal**: "Map the Pareto frontier between detectability, meaning preservation, and instruction-following"

**Implementation**:
```bash
# Train all ablation models
python scripts/train_stealthrl.py --config configs/ablations/detector_only.yaml
python scripts/train_stealthrl.py --config configs/ablations/no_fairness.yaml
python scripts/train_stealthrl.py --config configs/ablations/no_quality.yaml
python scripts/train_stealthrl.py --config configs/ablations/no_semantic.yaml
python scripts/train_stealthrl.py --config configs/ablations/single_detector_fast_detectgpt.yaml

# Evaluate all ablations
python scripts/evaluate_ablations.py \
    --checkpoints outputs/ablations/*/ \
    --output-dir outputs/ablation_analysis
```

**Visualization**:
```python
# scripts/plot_pareto_frontier.py

def plot_pareto_frontier(ablation_results: List[dict]):
    """
    Plot 3D Pareto frontier:
    - X-axis: Detectability (1 - avg_detector_prob)
    - Y-axis: Semantic similarity
    - Z-axis: Quality (perplexity score)
    - Color: Fairness (ESL gap)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for result in ablation_results:
        ax.scatter(
            result['detectability'],
            result['semantic_sim'],
            result['quality'],
            c=result['esl_gap'],
            s=100,
            label=result['name'],
        )
    
    ax.set_xlabel('Detectability (1 - P(AI))')
    ax.set_ylabel('Semantic Similarity')
    ax.set_zlabel('Quality (Perplexity Score)')
    plt.colorbar(label='ESL Fairness Gap')
    plt.savefig('outputs/pareto_frontier.png')
```

**Deliverable**: Ablation analysis with Pareto frontier visualization

---

### Priority 3: Evaluation Enhancements (MEDIUM)

#### 6. BERTScore Integration 📊
**Goal**: Add BERTScore alongside E5 cosine similarity

**Implementation**:
```python
# stealthrl/tinker/semantic.py enhancement

from bert_score import score as bert_score

class SemanticSimilarity:
    def __init__(self, use_bertscore: bool = True, use_e5: bool = True):
        self.use_bertscore = use_bertscore
        self.use_e5 = use_e5
        # ... existing E5 init
    
    async def compute(self, text1: str, text2: str) -> dict:
        results = {}
        
        if self.use_e5:
            # Existing E5 cosine similarity
            results['e5_similarity'] = await self._compute_e5(text1, text2)
        
        if self.use_bertscore:
            # BERTScore (P, R, F1)
            P, R, F1 = bert_score([text1], [text2], lang='en')
            results['bertscore_precision'] = P.item()
            results['bertscore_recall'] = R.item()
            results['bertscore_f1'] = F1.item()
        
        return results
```

**Dependencies**: Add `bert-score>=0.3.13` to `requirements.txt`

**Deliverable**: Dual semantic metrics (E5 + BERTScore) in evaluation

---

#### 7. Instruction-Following Accuracy 📊
**Goal**: Measure preservation of original intent/instructions

**Approaches**:

**Option A: LLM-as-Judge**
```python
# stealthrl/evaluation/instruction_following.py

class InstructionFollowingScorer:
    """Evaluate if paraphrase preserves instruction intent."""
    
    def __init__(self, judge_model: str = "gpt-4"):
        self.judge_model = judge_model
    
    async def score(
        self,
        original: str,
        paraphrase: str,
        instruction_type: str = "general",
    ) -> float:
        """
        Score instruction-following accuracy [0, 1].
        
        Uses LLM judge to rate whether paraphrase preserves
        original's instructional intent.
        """
        prompt = f"""
        Original text: {original}
        Paraphrased text: {paraphrase}
        
        Does the paraphrase preserve the same instructional meaning,
        factual content, and intent as the original?
        
        Rate from 0-10 (10 = perfect preservation).
        Provide only the numeric score.
        """
        
        response = await self.judge_model.generate(prompt)
        score = float(response.strip()) / 10.0
        return score
```

**Option B: Task-Specific Metrics**
```python
def compute_entity_overlap(original: str, paraphrase: str) -> float:
    """Measure preservation of named entities."""
    # Extract entities with spaCy
    original_entities = extract_entities(original)
    paraphrase_entities = extract_entities(paraphrase)
    return jaccard_similarity(original_entities, paraphrase_entities)

def compute_factual_consistency(original: str, paraphrase: str) -> float:
    """Measure factual consistency using QA."""
    # Generate QA pairs from original
    # Answer using paraphrase
    # Measure answer overlap
    pass
```

**Deliverable**: Instruction-following metric in evaluation suite

---

#### 8. StealthBench Visualization 📊
**Goal**: "Standardized tables and plots" for detector comparison

**Implementation**:
```python
# scripts/generate_stealthbench_report.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_roc_curves(evaluation_results: dict):
    """Plot ROC curves for all detectors on same axes."""
    plt.figure(figsize=(10, 8))
    
    for detector_name, scores in evaluation_results.items():
        fpr, tpr, _ = roc_curve(scores['labels'], scores['predictions'])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{detector_name} (AUC={auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: StealthRL vs Base AI Text')
    plt.legend()
    plt.savefig('outputs/stealthbench_roc.png')

def plot_fpr_comparison(evaluation_results: dict):
    """Bar chart comparing FPR at TPR=95% across detectors."""
    detectors = list(evaluation_results.keys())
    base_fprs = [evaluation_results[d]['base_fpr_at_tpr95'] for d in detectors]
    stealthrl_fprs = [evaluation_results[d]['stealthrl_fpr_at_tpr95'] for d in detectors]
    
    x = np.arange(len(detectors))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, base_fprs, width, label='Base AI Text')
    plt.bar(x + width/2, stealthrl_fprs, width, label='StealthRL')
    plt.xlabel('Detector')
    plt.ylabel('FPR @ TPR=95%')
    plt.title('False Positive Rate Comparison')
    plt.xticks(x, detectors, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/stealthbench_fpr_comparison.png')

def plot_esl_fairness_heatmap(evaluation_results: dict):
    """Heatmap of ESL fairness gaps across detectors and models."""
    # Rows: detectors, Columns: models (base, sft, stealthrl)
    # Values: ESL FPR gap (FPR_esl - FPR_native)
    
    data = []
    for detector in detectors:
        row = [
            evaluation_results[detector]['base_esl_gap'],
            evaluation_results[detector]['sft_esl_gap'],
            evaluation_results[detector]['stealthrl_esl_gap'],
        ]
        data.append(row)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        data,
        xticklabels=['Base', 'SFT', 'StealthRL'],
        yticklabels=detectors,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn_r',  # Red = high gap, green = low gap
        center=0,
    )
    plt.title('ESL Fairness Gap (FPR_ESL - FPR_Native)')
    plt.tight_layout()
    plt.savefig('outputs/stealthbench_esl_fairness.png')
```

**Deliverable**: Automated report generation with publication-ready figures

---

## 🚀 Phase 3: Stretch Goals (OPTIONAL)

### 9. Selective Fine-Tuning Ablation
**Goal**: "Probe whether added capacity improves cross-family transfer or merely overfits"

**Implementation**:
```yaml
# configs/tinker_selective_finetune.yaml

lora:
  rank: 0  # Disable LoRA

training:
  freeze_layers: true
  unfreeze_top_n: 6  # Unfreeze top 6 transformer blocks
  learning_rate: 1.0e-6  # Lower LR for full weights
```

**Experiment**:
1. Train with LoRA only (baseline)
2. Train with top 6 layers unfrozen
3. Train with top 12 layers unfrozen
4. Train with full model unfrozen (if compute allows)
5. Compare transfer performance on held-out detectors

**Hypothesis**: Limited unfreezing may improve generalization; full unfreezing may overfit in-ensemble.

---

### 10. Defender-Side Analysis
**Goal**: "Which detector mixtures harden best?"

**Implementation**:
```python
# scripts/defender_analysis.py

def test_detector_mixtures(
    stealthrl_policy,
    detector_pool: List[str],
    mixture_sizes: List[int] = [2, 3, 4],
) -> dict:
    """
    Test robustness of different detector ensemble combinations.
    
    For each mixture size:
    1. Generate all combinations of detectors
    2. Train adversarial policy against each mixture
    3. Measure ASR on held-out detectors
    4. Find mixture that minimizes ASR (best defense)
    """
    results = {}
    
    for size in mixture_sizes:
        for mixture in itertools.combinations(detector_pool, size):
            # Simulate training against this mixture
            asr = evaluate_asr(stealthrl_policy, held_out_detectors)
            results[mixture] = asr
    
    # Find best defensive mixture
    best_mixture = min(results, key=results.get)
    return {
        'best_mixture': best_mixture,
        'min_asr': results[best_mixture],
        'all_results': results,
    }
```

**Deliverable**: Recommendations for detector ensemble composition

---

## 📊 Expected Deliverables Summary

### Core Research (Must-Have)
1. ✅ Trained StealthRL model on Tinker with multi-detector ensemble
2. 🔨 Transfer evaluation showing out-of-ensemble generalization
3. 🔨 ESL-stratified fairness analysis
4. 🔨 Low-FPR operating point metrics (FPR@0.5%, FPR@1%)
5. 🔨 SICO baseline comparison
6. 🔨 Ablation experiments with Pareto frontier

### Evaluation Infrastructure (Must-Have)
7. 🔨 StealthBench visualizations (ROC curves, FPR comparison, ESL heatmaps)
8. 🔨 BERTScore + E5 semantic metrics
9. 🔨 Instruction-following accuracy metric

### Stretch Goals (Nice-to-Have)
10. Selective fine-tuning ablation
11. Defender-side analysis

---

## 🎯 Recommended Next Steps (Priority Order)

### Week 1: Data & Baselines
1. **Prepare ESL-stratified datasets** (TOEFL11, ICNALE, native academic)
2. **Implement SICO baseline** (prompt templates, evaluation)
3. **Enhance evaluation metrics** (low-FPR, BERTScore)

### Week 2: Training & Transfer
4. **Train in-ensemble model** (Fast-DetectGPT + Ghostbuster only)
5. **Train full ensemble model** (all three detectors)
6. **Implement transfer evaluation** (held-out Binoculars)

### Week 3: Ablations & Analysis
7. **Train ablation models** (detector-only, no-fairness, etc.)
8. **Run comprehensive evaluation** (all models, all metrics)
9. **Generate Pareto frontier plots**

### Week 4: Visualization & Reporting
10. **Create StealthBench report** (ROC curves, FPR comparison, ESL fairness)
11. **Compile results tables** (AUROC, ASR, semantic sim, ESL gaps)
12. **Draft research findings** (transfer analysis, fairness impact)

---

## 📝 Notes on Responsible Research

Per proposal: "We will not release evasion-tuned weights; instead we will share the evaluation harness and aggregate findings."

**Release Plan**:
- ✅ Release: StealthBench evaluation harness
- ✅ Release: Aggregate metrics and analysis
- ✅ Release: Ablation configurations
- ❌ Do NOT release: Trained model weights
- ❌ Do NOT release: Training data with evasion examples

**Documentation**:
- Include ethical considerations section in final report
- Discuss dual-use concerns and mitigation strategies
- Emphasize contribution to detection robustness research

---

## 🔗 Quick Reference

**Key Files**:
- Training: `stealthrl/tinker/train.py`
- Evaluation: `stealthrl/tinker/evaluation.py`
- Configs: `configs/tinker_stealthrl.yaml`
- Data prep: `scripts/prepare_tinker_data.py`
- Ablations: `scripts/evaluate_ablations.py`

**Documentation**:
- Setup: `TINKER_README.md`
- Project overview: `README.md`
- Implementation log: `interaction_records.md`
- This roadmap: `RESEARCH_ROADMAP.md`

---

*Last updated: November 25, 2025*
