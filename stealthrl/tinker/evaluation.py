"""
Comprehensive evaluation suite for StealthRL.

Compares base AI text, SFT baseline, and StealthRL policy across:
- Attack Success Rate (ASR): Fraction of texts evading all detectors
- AUROC: Area under ROC curve per detector
- F1: F1 score per detector
- Semantic similarity distributions
- ESL fairness gap analysis
"""

import asyncio
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvaluationExample:
    """Single example for evaluation."""
    
    ai_text: str
    human_reference: str
    domain: str
    is_esl: bool
    metadata: Dict
    
    # Generated paraphrases
    sft_paraphrase: Optional[str] = None
    stealthrl_paraphrase: Optional[str] = None
    
    # Detector scores
    base_detector_scores: Optional[Dict[str, float]] = None
    sft_detector_scores: Optional[Dict[str, float]] = None
    stealthrl_detector_scores: Optional[Dict[str, float]] = None
    
    # Semantic similarity (E5 cosine)
    sft_semantic_sim: Optional[float] = None
    stealthrl_semantic_sim: Optional[float] = None
    
    # BERTScore semantic similarity
    sft_bertscore_f1: Optional[float] = None
    stealthrl_bertscore_f1: Optional[float] = None


@dataclass
class ModelMetrics:
    """Metrics for a single model."""
    
    # Attack success
    asr_all: float  # Fraction evading all detectors
    asr_any: float  # Fraction evading at least one detector
    
    # Per-detector metrics
    auroc_per_detector: Dict[str, float]
    f1_per_detector: Dict[str, float]
    fpr_at_tpr95_per_detector: Dict[str, float]
    
    # Low-FPR operating points (academic integrity thresholds)
    tpr_at_fpr_0_5_per_detector: Dict[str, float]  # TPR when FPR=0.5%
    tpr_at_fpr_1_0_per_detector: Dict[str, float]  # TPR when FPR=1.0%
    threshold_at_fpr_0_5_per_detector: Dict[str, float]  # Threshold for FPR=0.5%
    threshold_at_fpr_1_0_per_detector: Dict[str, float]  # Threshold for FPR=1.0%
    
    # Semantic similarity (E5 cosine)
    semantic_sim_mean: float
    semantic_sim_std: float
    semantic_sim_min: float
    
    # ESL fairness (moved before fields with defaults)
    esl_fpr_gap: Dict[str, float]  # FPR(ESL) - FPR(native) per detector
    esl_auroc_gap: Dict[str, float]  # AUROC(ESL) - AUROC(native)
    
    # Quality
    avg_detector_prob: float
    avg_detector_prob_esl: float
    avg_detector_prob_native: float
    
    # BERTScore semantic similarity (with defaults)
    bertscore_f1_mean: float = 0.0
    bertscore_f1_std: float = 0.0
    bertscore_f1_min: float = 0.0


@dataclass
class ComparisonReport:
    """Comparison report across models."""
    
    base_metrics: ModelMetrics
    sft_metrics: Optional[ModelMetrics]
    stealthrl_metrics: ModelMetrics
    
    # Improvements
    asr_improvement_sft: Optional[float]  # StealthRL ASR - SFT ASR
    asr_improvement_base: float  # StealthRL ASR - Base ASR
    
    fairness_improvement_sft: Optional[float]  # Reduction in ESL gap
    fairness_improvement_base: float


class EvaluationSuite:
    """
    Comprehensive evaluation suite for StealthRL.
    
    Evaluates base AI text, SFT baseline, and StealthRL policy across
    multiple detectors and fairness metrics.
    """
    
    def __init__(
        self,
        detector_ensemble,
        semantic_similarity,
        detector_names: List[str],
        output_dir: str = "outputs/evaluation",
    ):
        """
        Initialize evaluation suite.
        
        Args:
            detector_ensemble: DetectorEnsemble instance
            semantic_similarity: SemanticSimilarity instance
            detector_names: List of detector names
            output_dir: Output directory for results
        """
        self.detector_ensemble = detector_ensemble
        self.semantic_similarity = semantic_similarity
        self.detector_names = detector_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def _score_with_detectors(self, text: str) -> Dict[str, float]:
        """
        Score text with all detectors.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping detector name to P(AI)
        """
        # Get ensemble prediction (includes individual detector scores)
        result = await self.detector_ensemble.compute(text)
        
        # Extract individual detector scores
        scores = {}
        for name in self.detector_names:
            detector_key = f"{name}_prob"
            if detector_key in result:
                scores[name] = result[detector_key]
            else:
                # Fallback: use ensemble score
                scores[name] = result["detector_prob"]
        
        return scores
    
    async def _compute_semantic_similarity(
        self,
        paraphrase: str,
        reference: str,
    ) -> float:
        """
        Compute semantic similarity between paraphrase and reference.
        
        Args:
            paraphrase: Paraphrased text
            reference: Reference text
            
        Returns:
            Similarity score [0, 1]
        """
        result = await self.semantic_similarity.compute(paraphrase, reference)
        return result["similarity"]
    
    async def evaluate_examples(
        self,
        examples: List[EvaluationExample],
        base_model,
        sft_model = None,
        stealthrl_model = None,
    ) -> List[EvaluationExample]:
        """
        Evaluate examples by generating paraphrases and scoring.
        
        Args:
            examples: List of EvaluationExample objects
            base_model: Base AI text generator (for comparison)
            sft_model: Optional SFT baseline model
            stealthrl_model: StealthRL policy model
            
        Returns:
            List of EvaluationExample objects with scores filled in
        """
        for i, example in enumerate(examples):
            print(f"Evaluating example {i+1}/{len(examples)}...")
            
            # Score base AI text
            example.base_detector_scores = await self._score_with_detectors(example.ai_text)
            
            # Generate and score SFT paraphrase
            if sft_model is not None:
                example.sft_paraphrase = await sft_model.generate(
                    f"Please paraphrase: {example.ai_text}",
                    max_tokens=512,
                )
                example.sft_detector_scores = await self._score_with_detectors(example.sft_paraphrase)
                example.sft_semantic_sim = await self._compute_semantic_similarity(
                    example.sft_paraphrase,
                    example.human_reference,
                )
            
            # Generate and score StealthRL paraphrase
            if stealthrl_model is not None:
                example.stealthrl_paraphrase = await stealthrl_model.generate(
                    f"Please paraphrase: {example.ai_text}",
                    max_tokens=512,
                )
                example.stealthrl_detector_scores = await self._score_with_detectors(example.stealthrl_paraphrase)
                example.stealthrl_semantic_sim = await self._compute_semantic_similarity(
                    example.stealthrl_paraphrase,
                    example.human_reference,
                )
        
        return examples
    
    def _compute_model_metrics(
        self,
        examples: List[EvaluationExample],
        score_key: str,  # "base_detector_scores", "sft_detector_scores", etc.
    ) -> ModelMetrics:
        """
        Compute metrics for a single model.
        
        Args:
            examples: List of EvaluationExample objects
            score_key: Attribute name for detector scores
            
        Returns:
            ModelMetrics object
        """
        # Extract detector scores
        detector_scores = defaultdict(list)  # detector_name -> [scores]
        labels = []  # Ground truth: 1 = AI, 0 = human
        is_esl_flags = []
        
        for ex in examples:
            scores = getattr(ex, score_key)
            if scores is None:
                continue
            
            for detector_name, score in scores.items():
                detector_scores[detector_name].append(score)
            
            labels.append(1)  # All examples are AI-generated
            is_esl_flags.append(ex.is_esl)
        
        # Compute ASR
        asr_all = self._compute_asr_all(detector_scores, self.detector_names)
        asr_any = self._compute_asr_any(detector_scores, self.detector_names)
        
        # Compute per-detector metrics
        auroc_per_detector = {}
        f1_per_detector = {}
        fpr_at_tpr95_per_detector = {}
        tpr_at_fpr_0_5_per_detector = {}
        tpr_at_fpr_1_0_per_detector = {}
        threshold_at_fpr_0_5_per_detector = {}
        threshold_at_fpr_1_0_per_detector = {}
        
        for detector_name in self.detector_names:
            scores = np.array(detector_scores[detector_name])
            
            # AUROC (using dummy labels: all 1s for AI)
            # NOTE: For proper AUROC, need human-written texts as negatives
            # This is a simplified version assuming threshold-based evaluation
            auroc_per_detector[detector_name] = self._compute_auroc_proxy(scores)
            
            # F1 at threshold 0.5
            preds = (scores >= 0.5).astype(int)
            f1_per_detector[detector_name] = f1_score(labels, preds)
            
            # FPR at TPR=0.95
            fpr_at_tpr95_per_detector[detector_name] = self._compute_fpr_at_tpr(scores, labels, tpr_target=0.95)
            
            # Low-FPR operating points (academic integrity thresholds)
            low_fpr_metrics = self._compute_low_fpr_metrics(scores, labels)
            tpr_at_fpr_0_5_per_detector[detector_name] = low_fpr_metrics['tpr_at_fpr_0.5']
            tpr_at_fpr_1_0_per_detector[detector_name] = low_fpr_metrics['tpr_at_fpr_1.0']
            threshold_at_fpr_0_5_per_detector[detector_name] = low_fpr_metrics['threshold_at_fpr_0.5']
            threshold_at_fpr_1_0_per_detector[detector_name] = low_fpr_metrics['threshold_at_fpr_1.0']
        
        # Semantic similarity
        semantic_sims = []
        for ex in examples:
            sim_key = score_key.replace("detector_scores", "semantic_sim")
            sim = getattr(ex, sim_key, None)
            if sim is not None:
                semantic_sims.append(sim)
        
        if semantic_sims:
            semantic_sim_mean = np.mean(semantic_sims)
            semantic_sim_std = np.std(semantic_sims)
            semantic_sim_min = np.min(semantic_sims)
        else:
            semantic_sim_mean = semantic_sim_std = semantic_sim_min = 0.0
        
        # ESL fairness gap
        esl_fpr_gap, esl_auroc_gap = self._compute_esl_gaps(
            detector_scores,
            is_esl_flags,
            labels,
        )
        
        # Average detector probabilities
        avg_detector_prob = np.mean([np.mean(scores) for scores in detector_scores.values()])
        
        # ESL vs native
        esl_scores = [scores[i] for i, is_esl in enumerate(is_esl_flags) if is_esl]
        native_scores = [scores[i] for i, is_esl in enumerate(is_esl_flags) if not is_esl]
        
        avg_detector_prob_esl = np.mean([np.mean(s) for s in esl_scores]) if esl_scores else 0.0
        avg_detector_prob_native = np.mean([np.mean(s) for s in native_scores]) if native_scores else 0.0
        
        return ModelMetrics(
            asr_all=asr_all,
            asr_any=asr_any,
            auroc_per_detector=auroc_per_detector,
            f1_per_detector=f1_per_detector,
            fpr_at_tpr95_per_detector=fpr_at_tpr95_per_detector,
            tpr_at_fpr_0_5_per_detector=tpr_at_fpr_0_5_per_detector,
            tpr_at_fpr_1_0_per_detector=tpr_at_fpr_1_0_per_detector,
            threshold_at_fpr_0_5_per_detector=threshold_at_fpr_0_5_per_detector,
            threshold_at_fpr_1_0_per_detector=threshold_at_fpr_1_0_per_detector,
            semantic_sim_mean=semantic_sim_mean,
            semantic_sim_std=semantic_sim_std,
            semantic_sim_min=semantic_sim_min,
            esl_fpr_gap=esl_fpr_gap,
            esl_auroc_gap=esl_auroc_gap,
            avg_detector_prob=avg_detector_prob,
            avg_detector_prob_esl=avg_detector_prob_esl,
            avg_detector_prob_native=avg_detector_prob_native,
        )
    
    def _compute_asr_all(
        self,
        detector_scores: Dict[str, List[float]],
        detector_names: List[str],
        threshold: float = 0.5,
    ) -> float:
        """Compute ASR: fraction of texts evading all detectors."""
        num_examples = len(detector_scores[detector_names[0]])
        evades_all = 0
        
        for i in range(num_examples):
            if all(detector_scores[name][i] < threshold for name in detector_names):
                evades_all += 1
        
        return evades_all / num_examples
    
    def _compute_asr_any(
        self,
        detector_scores: Dict[str, List[float]],
        detector_names: List[str],
        threshold: float = 0.5,
    ) -> float:
        """Compute ASR: fraction of texts evading at least one detector."""
        num_examples = len(detector_scores[detector_names[0]])
        evades_any = 0
        
        for i in range(num_examples):
            if any(detector_scores[name][i] < threshold for name in detector_names):
                evades_any += 1
        
        return evades_any / num_examples
    
    def _compute_auroc_proxy(self, scores: np.ndarray) -> float:
        """
        Compute AUROC proxy based on score distribution.
        
        NOTE: Proper AUROC requires human-written texts as negatives.
        This is a simplified version for quick evaluation.
        """
        # Proxy: lower mean score = better evasion
        return 1.0 - np.mean(scores)
    
    def _compute_fpr_at_tpr(
        self,
        scores: np.ndarray,
        labels: List[int],
        tpr_target: float = 0.95,
    ) -> float:
        """Compute FPR at given TPR target."""
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # Find threshold where TPR >= target
        idx = np.where(tpr >= tpr_target)[0]
        if len(idx) == 0:
            return 1.0
        
        return fpr[idx[0]]
    
    def _compute_low_fpr_metrics(
        self,
        scores: np.ndarray,
        labels: List[int],
        fpr_targets: List[float] = [0.005, 0.01],  # 0.5%, 1.0%
    ) -> Dict[str, float]:
        """
        Compute TPR and thresholds at low FPR operating points.
        
        These are critical for academic integrity applications where
        false accusations must be minimized.
        
        Args:
            scores: Detector scores (higher = more likely AI)
            labels: Ground truth labels (1 = AI, 0 = human)
            fpr_targets: Target FPR values
            
        Returns:
            Dictionary with TPR and threshold at each FPR target
        """
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        results = {}
        for fpr_target in fpr_targets:
            # Find threshold where FPR is closest to target
            idx = np.argmin(np.abs(fpr - fpr_target))
            
            key_suffix = str(fpr_target).replace('0.0', '')
            results[f'tpr_at_fpr_{key_suffix}'] = tpr[idx]
            results[f'threshold_at_fpr_{key_suffix}'] = thresholds[idx] if idx < len(thresholds) else 1.0
            results[f'actual_fpr_{key_suffix}'] = fpr[idx]
        
        return results
    
    def _compute_esl_gaps(
        self,
        detector_scores: Dict[str, List[float]],
        is_esl_flags: List[bool],
        labels: List[int],
    ) -> tuple:
        """Compute ESL fairness gaps."""
        esl_fpr_gap = {}
        esl_auroc_gap = {}
        
        for detector_name, scores in detector_scores.items():
            scores = np.array(scores)
            
            # Split by ESL status
            esl_scores = scores[np.array(is_esl_flags)]
            native_scores = scores[~np.array(is_esl_flags)]
            
            if len(esl_scores) == 0 or len(native_scores) == 0:
                esl_fpr_gap[detector_name] = 0.0
                esl_auroc_gap[detector_name] = 0.0
                continue
            
            # FPR gap (at threshold 0.5)
            esl_fpr = np.mean(esl_scores >= 0.5)
            native_fpr = np.mean(native_scores >= 0.5)
            esl_fpr_gap[detector_name] = esl_fpr - native_fpr
            
            # AUROC gap (proxy)
            esl_auroc_proxy = 1.0 - np.mean(esl_scores)
            native_auroc_proxy = 1.0 - np.mean(native_scores)
            esl_auroc_gap[detector_name] = esl_auroc_proxy - native_auroc_proxy
        
        return esl_fpr_gap, esl_auroc_gap
    
    def generate_comparison_report(
        self,
        examples: List[EvaluationExample],
        has_sft: bool = False,
    ) -> ComparisonReport:
        """
        Generate comparison report across models.
        
        Args:
            examples: List of EvaluationExample objects with scores
            has_sft: Whether SFT baseline is available
            
        Returns:
            ComparisonReport object
        """
        # Compute metrics for each model
        base_metrics = self._compute_model_metrics(examples, "base_detector_scores")
        stealthrl_metrics = self._compute_model_metrics(examples, "stealthrl_detector_scores")
        
        sft_metrics = None
        asr_improvement_sft = None
        fairness_improvement_sft = None
        
        if has_sft:
            sft_metrics = self._compute_model_metrics(examples, "sft_detector_scores")
            asr_improvement_sft = stealthrl_metrics.asr_all - sft_metrics.asr_all
            
            # Fairness improvement: reduction in ESL gap
            sft_gap = np.mean(list(sft_metrics.esl_fpr_gap.values()))
            stealthrl_gap = np.mean(list(stealthrl_metrics.esl_fpr_gap.values()))
            fairness_improvement_sft = sft_gap - stealthrl_gap
        
        # Improvements over base
        asr_improvement_base = stealthrl_metrics.asr_all - base_metrics.asr_all
        base_gap = np.mean(list(base_metrics.esl_fpr_gap.values()))
        stealthrl_gap = np.mean(list(stealthrl_metrics.esl_fpr_gap.values()))
        fairness_improvement_base = base_gap - stealthrl_gap
        
        return ComparisonReport(
            base_metrics=base_metrics,
            sft_metrics=sft_metrics,
            stealthrl_metrics=stealthrl_metrics,
            asr_improvement_sft=asr_improvement_sft,
            asr_improvement_base=asr_improvement_base,
            fairness_improvement_sft=fairness_improvement_sft,
            fairness_improvement_base=fairness_improvement_base,
        )
    
    def save_report(
        self,
        report: ComparisonReport,
        examples: List[EvaluationExample],
        filename: str = "evaluation_report.json",
    ):
        """Save evaluation report to JSON."""
        output_path = self.output_dir / filename
        
        report_dict = {
            "base_metrics": asdict(report.base_metrics),
            "stealthrl_metrics": asdict(report.stealthrl_metrics),
            "asr_improvement_base": report.asr_improvement_base,
            "fairness_improvement_base": report.fairness_improvement_base,
            "num_examples": len(examples),
            "num_esl": sum(1 for ex in examples if ex.is_esl),
        }
        
        if report.sft_metrics is not None:
            report_dict["sft_metrics"] = asdict(report.sft_metrics)
            report_dict["asr_improvement_sft"] = report.asr_improvement_sft
            report_dict["fairness_improvement_sft"] = report.fairness_improvement_sft
        
        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"Saved evaluation report to {output_path}")
    
    def print_summary(self, report: ComparisonReport):
        """Print summary of evaluation results."""
        print("\n" + "="*60)
        print("STEALTHRL EVALUATION SUMMARY")
        print("="*60)
        
        print("\n📊 Attack Success Rate (ASR)")
        print(f"  Base:      {report.base_metrics.asr_all:.1%}")
        if report.sft_metrics:
            print(f"  SFT:       {report.sft_metrics.asr_all:.1%}")
        print(f"  StealthRL: {report.stealthrl_metrics.asr_all:.1%}")
        print(f"  → Improvement over base: +{report.asr_improvement_base:.1%}")
        
        print("\n🎯 Semantic Similarity")
        if report.sft_metrics:
            print(f"  SFT:       {report.sft_metrics.semantic_sim_mean:.3f} ± {report.sft_metrics.semantic_sim_std:.3f}")
        print(f"  StealthRL: {report.stealthrl_metrics.semantic_sim_mean:.3f} ± {report.stealthrl_metrics.semantic_sim_std:.3f}")
        
        print("\n⚖️ ESL Fairness Gap (FPR difference)")
        base_gap = np.mean(list(report.base_metrics.esl_fpr_gap.values()))
        stealthrl_gap = np.mean(list(report.stealthrl_metrics.esl_fpr_gap.values()))
        print(f"  Base:      {base_gap:.3f}")
        if report.sft_metrics:
            sft_gap = np.mean(list(report.sft_metrics.esl_fpr_gap.values()))
            print(f"  SFT:       {sft_gap:.3f}")
        print(f"  StealthRL: {stealthrl_gap:.3f}")
        print(f"  → Reduction: {report.fairness_improvement_base:.3f}")
        
        print("\n🔍 Per-Detector AUROC Proxy")
        for detector_name in report.stealthrl_metrics.auroc_per_detector.keys():
            base_auroc = report.base_metrics.auroc_per_detector[detector_name]
            stealthrl_auroc = report.stealthrl_metrics.auroc_per_detector[detector_name]
            print(f"  {detector_name:20s}: {base_auroc:.3f} → {stealthrl_auroc:.3f}")
        
        print("\n🎓 Low-FPR Operating Points (Academic Integrity Thresholds)")
        print("\n  TPR at FPR=0.5% (very conservative):")
        for detector_name in report.stealthrl_metrics.tpr_at_fpr_0_5_per_detector.keys():
            base_tpr = report.base_metrics.tpr_at_fpr_0_5_per_detector[detector_name]
            stealthrl_tpr = report.stealthrl_metrics.tpr_at_fpr_0_5_per_detector[detector_name]
            print(f"    {detector_name:20s}: {base_tpr:.1%} → {stealthrl_tpr:.1%}")
        
        print("\n  TPR at FPR=1.0% (conservative):")
        for detector_name in report.stealthrl_metrics.tpr_at_fpr_1_0_per_detector.keys():
            base_tpr = report.base_metrics.tpr_at_fpr_1_0_per_detector[detector_name]
            stealthrl_tpr = report.stealthrl_metrics.tpr_at_fpr_1_0_per_detector[detector_name]
            print(f"    {detector_name:20s}: {base_tpr:.1%} → {stealthrl_tpr:.1%}")
        
        print("\n" + "="*60)


# Example usage
async def main():
    """Example: Run comprehensive evaluation."""
    from stealthrl.tinker import DetectorEnsemble, SemanticSimilarity
    
    # Initialize components
    detector_ensemble = DetectorEnsemble(
        detector_names=["fast_detectgpt", "ghostbuster", "binoculars"],
        weights={"fast_detectgpt": 0.4, "ghostbuster": 0.3, "binoculars": 0.3},
        use_mock=True,
    )
    
    semantic_similarity = SemanticSimilarity(use_mock=True)
    
    # Initialize evaluation suite
    eval_suite = EvaluationSuite(
        detector_ensemble=detector_ensemble,
        semantic_similarity=semantic_similarity,
        detector_names=["fast_detectgpt", "ghostbuster", "binoculars"],
        output_dir="outputs/evaluation",
    )
    
    # Create test examples
    examples = [
        EvaluationExample(
            ai_text="The implementation of neural networks requires careful consideration.",
            human_reference="Building neural networks demands thoughtful selection.",
            domain="academic",
            is_esl=False,
            metadata={"model_family": "gpt"},
        ),
        EvaluationExample(
            ai_text="Machine learning algorithms can process large datasets efficiently.",
            human_reference="ML algorithms handle big data well.",
            domain="technical",
            is_esl=True,
            metadata={"model_family": "claude"},
        ),
    ]
    
    # Mock models
    class MockModel:
        async def generate(self, prompt, **kwargs):
            return prompt.replace("Please paraphrase: ", "Paraphrased: ")
    
    # Evaluate
    examples = await eval_suite.evaluate_examples(
        examples=examples,
        base_model=MockModel(),
        sft_model=MockModel(),
        stealthrl_model=MockModel(),
    )
    
    # Generate report
    report = eval_suite.generate_comparison_report(examples, has_sft=True)
    
    # Print and save
    eval_suite.print_summary(report)
    eval_suite.save_report(report, examples)


if __name__ == "__main__":
    asyncio.run(main())
