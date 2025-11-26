"""
Run ESL/native fairness evaluation with BERTScore support.

Usage:
    python scripts/run_esl_eval.py --eval_data data/processed/esl_native_test.jsonl \
        --base_model base_ai_model \
        --stealthrl_model outputs/stealthrl_policy \
        --output_dir results/esl_native_eval

Features:
    - Loads unified ESL/native JSONL data
    - Computes detector metrics grouped by ESL status
    - Computes E5 cosine + BERTScore semantic similarity
    - Saves grouped metrics to JSON
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from stealthrl.data.esl_native_corpus import ESLNativeRecord, load_esl_native_jsonl
from stealthrl.metrics.bertscore_metrics import BERTScoreConfig, compute_bertscore_grouped
from stealthrl.tinker.evaluation import EvaluationExample, EvaluationSuite

logger = logging.getLogger(__name__)


def load_esl_native_eval_data(
    file_path: Path,
) -> tuple[List[EvaluationExample], List[str]]:
    """
    Load ESL/native data and convert to EvaluationExample format.
    
    Args:
        file_path: Path to ESL/native JSONL file
        
    Returns:
        (examples, esl_labels) tuple
        - examples: List of EvaluationExample objects
        - esl_labels: List of "esl" or "native" strings for grouping
    """
    records = load_esl_native_jsonl(file_path)
    
    examples = []
    esl_labels = []
    
    for rec in records:
        # Convert to EvaluationExample
        example = EvaluationExample(
            ai_text=rec.text,  # Assume text is AI-generated for this eval
            human_reference=rec.text,  # Use same text as reference
            domain=rec.source,
            is_esl=rec.is_esl,
            metadata={
                "id": rec.id,
                "source": rec.source,
                "proficiency_level": rec.proficiency_level,
                "prompt_id": rec.prompt_id,
            },
        )
        examples.append(example)
        esl_labels.append("esl" if rec.is_esl else "native")
    
    logger.info(f"Loaded {len(examples)} evaluation examples")
    logger.info(f"  ESL: {sum(1 for x in esl_labels if x == 'esl')}")
    logger.info(f"  Native: {sum(1 for x in esl_labels if x == 'native')}")
    
    return examples, esl_labels


async def run_esl_native_evaluation(
    eval_data_path: Path,
    detector_ensemble,
    semantic_similarity,
    detector_names: List[str],
    base_model,
    stealthrl_model,
    sft_model=None,
    bertscore_config: Optional[BERTScoreConfig] = None,
    output_dir: Path = Path("results/esl_native_eval"),
) -> dict:
    """
    Run comprehensive ESL/native evaluation with BERTScore.
    
    Args:
        eval_data_path: Path to ESL/native test JSONL
        detector_ensemble: DetectorEnsemble instance
        semantic_similarity: SemanticSimilarity instance (E5)
        detector_names: List of detector names
        base_model: Base AI model
        stealthrl_model: StealthRL policy
        sft_model: Optional SFT baseline
        bertscore_config: BERTScore configuration
        output_dir: Output directory for results
        
    Returns:
        Dictionary with grouped metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load evaluation data
    logger.info(f"Loading evaluation data from {eval_data_path}")
    examples, esl_labels = load_esl_native_eval_data(eval_data_path)
    
    # Initialize evaluation suite
    eval_suite = EvaluationSuite(
        detector_ensemble=detector_ensemble,
        semantic_similarity=semantic_similarity,
        detector_names=detector_names,
        output_dir=str(output_dir),
    )
    
    # Evaluate examples (generate paraphrases and score)
    logger.info("Running evaluation...")
    examples = await eval_suite.evaluate_examples(
        examples=examples,
        base_model=base_model,
        sft_model=sft_model,
        stealthrl_model=stealthrl_model,
    )
    
    # Generate comparison report (overall metrics)
    logger.info("Computing comparison report...")
    report = eval_suite.generate_comparison_report(
        examples=examples,
        has_sft=(sft_model is not None),
    )
    
    # Save overall report
    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "base_metrics": report.base_metrics.__dict__,
                "sft_metrics": report.sft_metrics.__dict__ if report.sft_metrics else None,
                "stealthrl_metrics": report.stealthrl_metrics.__dict__,
                "improvements": {
                    "asr_improvement_sft": report.asr_improvement_sft,
                    "asr_improvement_base": report.asr_improvement_base,
                    "fairness_improvement_sft": report.fairness_improvement_sft,
                    "fairness_improvement_base": report.fairness_improvement_base,
                },
            },
            f,
            indent=2,
        )
    logger.info(f"Saved overall report to {report_path}")
    
    # Compute ESL-grouped metrics
    logger.info("Computing ESL-grouped metrics...")
    grouped_metrics = compute_grouped_metrics(examples, esl_labels, detector_names)
    
    # Save grouped metrics
    grouped_metrics_path = output_dir / "esl_native_grouped_metrics.json"
    with open(grouped_metrics_path, "w") as f:
        json.dump(grouped_metrics, f, indent=2)
    logger.info(f"Saved grouped metrics to {grouped_metrics_path}")
    
    # Compute BERTScore if enabled
    if bertscore_config and bertscore_config.enabled:
        logger.info("Computing BERTScore...")
        bertscore_results = compute_bertscore_esl_native(
            examples=examples,
            esl_labels=esl_labels,
            config=bertscore_config,
        )
        
        # Save BERTScore results
        bertscore_path = output_dir / "bertscore_results.json"
        with open(bertscore_path, "w") as f:
            json.dump(bertscore_results, f, indent=2)
        logger.info(f"Saved BERTScore results to {bertscore_path}")
        
        # Save detailed per-sample BERTScore
        save_detailed_bertscore(
            examples=examples,
            esl_labels=esl_labels,
            bertscore_results=bertscore_results,
            output_path=output_dir / "bertscore_esl_native.jsonl",
        )
    
    return {
        "overall_report": report,
        "grouped_metrics": grouped_metrics,
        "bertscore_results": bertscore_results if bertscore_config and bertscore_config.enabled else None,
    }


def compute_grouped_metrics(
    examples: List[EvaluationExample],
    esl_labels: List[str],
    detector_names: List[str],
) -> dict:
    """
    Compute metrics grouped by ESL status.
    
    Returns:
        Dictionary with "overall", "esl", and "native" keys
    """
    from collections import defaultdict
    from sklearn.metrics import roc_auc_score, f1_score
    
    # Group examples
    groups = {"overall": [], "esl": [], "native": []}
    for ex, label in zip(examples, esl_labels):
        groups["overall"].append(ex)
        groups[label].append(ex)
    
    results = {}
    for group_name, group_examples in groups.items():
        if not group_examples:
            continue
        
        # Compute metrics for each model
        for model_name in ["base", "sft", "stealthrl"]:
            score_key = f"{model_name}_detector_scores"
            
            # Skip if not available
            if not hasattr(group_examples[0], score_key):
                continue
            
            if getattr(group_examples[0], score_key) is None:
                continue
            
            # Extract scores per detector
            detector_scores = defaultdict(list)
            semantic_sims = []
            
            for ex in group_examples:
                scores = getattr(ex, score_key, None)
                if scores:
                    for det_name, score in scores.items():
                        detector_scores[det_name].append(score)
                
                # E5 semantic similarity
                sim_key = f"{model_name}_semantic_sim"
                sim = getattr(ex, sim_key, None)
                if sim is not None:
                    semantic_sims.append(sim)
            
            # Compute ASR (all detectors)
            asr_all = compute_asr_all(detector_scores, detector_names)
            
            # Compute per-detector metrics
            per_detector_metrics = {}
            for det_name in detector_names:
                if det_name in detector_scores:
                    scores = np.array(detector_scores[det_name])
                    per_detector_metrics[det_name] = {
                        "mean_prob": float(np.mean(scores)),
                        "std_prob": float(np.std(scores)),
                        "median_prob": float(np.median(scores)),
                    }
            
            # Store results
            model_key = f"{group_name}_{model_name}"
            results[model_key] = {
                "asr_all": asr_all,
                "per_detector": per_detector_metrics,
                "semantic_similarity": {
                    "mean": float(np.mean(semantic_sims)) if semantic_sims else 0.0,
                    "std": float(np.std(semantic_sims)) if semantic_sims else 0.0,
                    "median": float(np.median(semantic_sims)) if semantic_sims else 0.0,
                },
                "n_samples": len(group_examples),
            }
    
    return results


def compute_asr_all(
    detector_scores: dict,
    detector_names: List[str],
    threshold: float = 0.5,
) -> float:
    """Compute ASR: fraction evading all detectors."""
    if not detector_scores or not detector_names:
        return 0.0
    
    n_samples = len(detector_scores[detector_names[0]])
    evades_all = 0
    
    for i in range(n_samples):
        if all(detector_scores[name][i] < threshold for name in detector_names if name in detector_scores):
            evades_all += 1
    
    return evades_all / n_samples if n_samples > 0 else 0.0


def compute_bertscore_esl_native(
    examples: List[EvaluationExample],
    esl_labels: List[str],
    config: BERTScoreConfig,
) -> dict:
    """
    Compute BERTScore grouped by ESL status.
    
    Returns:
        Dictionary with BERTScore results for each model and group
    """
    results = {}
    
    for model_name in ["base", "sft", "stealthrl"]:
        paraphrase_key = f"{model_name}_paraphrase"
        
        # Collect outputs and references
        outputs = []
        references = []
        labels = []
        
        for ex, label in zip(examples, esl_labels):
            paraphrase = getattr(ex, paraphrase_key, None)
            if paraphrase:
                outputs.append(paraphrase)
                references.append(ex.human_reference)
                labels.append(label)
        
        if not outputs:
            continue
        
        # Compute grouped BERTScore
        logger.info(f"Computing BERTScore for {model_name} model...")
        grouped_scores = compute_bertscore_grouped(
            outputs=outputs,
            references=references,
            groups=labels,
            config=config,
        )
        
        results[model_name] = grouped_scores
    
    return results


def save_detailed_bertscore(
    examples: List[EvaluationExample],
    esl_labels: List[str],
    bertscore_results: dict,
    output_path: Path,
):
    """Save detailed per-sample BERTScore results to JSONL."""
    with open(output_path, "w") as f:
        for model_name in ["base", "sft", "stealthrl"]:
            if model_name not in bertscore_results:
                continue
            
            overall_scores = bertscore_results[model_name].get("overall", {})
            per_sample_f1 = overall_scores.get("per_sample_f1", [])
            
            paraphrase_key = f"{model_name}_paraphrase"
            
            idx = 0
            for ex, label in zip(examples, esl_labels):
                paraphrase = getattr(ex, paraphrase_key, None)
                if not paraphrase:
                    continue
                
                record = {
                    "id": ex.metadata.get("id"),
                    "is_esl": label == "esl",
                    "source": ex.domain,
                    "system": model_name,
                    "bertscore_f1": per_sample_f1[idx] if idx < len(per_sample_f1) else None,
                    "e5_cosine": getattr(ex, f"{model_name}_semantic_sim", None),
                }
                
                f.write(json.dumps(record) + "\n")
                idx += 1
    
    logger.info(f"Saved detailed BERTScore to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ESL/native fairness evaluation")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to ESL/native JSONL")
    parser.add_argument("--base_model", type=str, default="base", help="Base model identifier")
    parser.add_argument("--sft_model", type=str, default=None, help="SFT model (optional)")
    parser.add_argument("--stealthrl_model", type=str, required=True, help="StealthRL model path")
    parser.add_argument("--output_dir", type=str, default="results/esl_native_eval", help="Output directory")
    
    # BERTScore options
    parser.add_argument("--enable_bertscore", action="store_true", help="Enable BERTScore computation")
    parser.add_argument("--bertscore_model", type=str, default="roberta-large", help="BERTScore model type")
    parser.add_argument("--bertscore_batch_size", type=int, default=16, help="BERTScore batch size")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Mock models for demonstration
    logger.warning("Using mock models - replace with actual model loading")
    
    class MockModel:
        async def generate(self, prompt, max_tokens=512):
            return "Mock generated text."
    
    class MockDetectorEnsemble:
        async def compute(self, text):
            return {"detector_prob": 0.5}
    
    class MockSemanticSimilarity:
        async def compute(self, text1, text2):
            return {"similarity": 0.85}
    
    # Configure BERTScore
    bertscore_config = None
    if args.enable_bertscore:
        bertscore_config = BERTScoreConfig(
            enabled=True,
            model_type=args.bertscore_model,
            batch_size=args.bertscore_batch_size,
        )
    
    # Run evaluation
    results = asyncio.run(
        run_esl_native_evaluation(
            eval_data_path=Path(args.eval_data),
            detector_ensemble=MockDetectorEnsemble(),
            semantic_similarity=MockSemanticSimilarity(),
            detector_names=["fast_detectgpt", "ghostbuster", "binoculars"],
            base_model=MockModel(),
            stealthrl_model=MockModel(),
            sft_model=MockModel() if args.sft_model else None,
            bertscore_config=bertscore_config,
            output_dir=Path(args.output_dir),
        )
    )
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
