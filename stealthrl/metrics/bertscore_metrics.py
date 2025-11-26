"""
BERTScore semantic similarity metrics for StealthRL.

Uses the official bert-score library to compute semantic similarity between
generated paraphrases and reference texts.

Installation:
    pip install bert-score

Reference:
    https://github.com/Tiiiger/bert_score
    Zhang et al. (2020) "BERTScore: Evaluating Text Generation with BERT"
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BERTScoreConfig:
    """Configuration for BERTScore computation."""
    
    enabled: bool = True
    model_type: str = "roberta-large"  # Options: bert-base, roberta-large, etc.
    batch_size: int = 16
    num_layers: Optional[int] = None  # Use best layer if None
    verbose: bool = False
    device: Optional[str] = None  # Auto-detect if None
    
    # Rescaling parameters (optional)
    rescale_with_baseline: bool = False
    baseline_path: Optional[str] = None


def compute_bertscore(
    outputs: List[str],
    references: List[str],
    config: Optional[BERTScoreConfig] = None,
) -> dict:
    """
    Compute BERTScore F1 between outputs and references.
    
    Args:
        outputs: List of generated texts (e.g., StealthRL paraphrases)
        references: List of reference texts to compare against
        config: BERTScore configuration (uses defaults if None)
        
    Returns:
        Dictionary containing:
            - per_sample_f1: List[float] - F1 score for each sample
            - per_sample_precision: List[float] - Precision for each sample
            - per_sample_recall: List[float] - Recall for each sample
            - mean_f1: float - Mean F1 across all samples
            - median_f1: float - Median F1
            - std_f1: float - Standard deviation of F1
            - mean_precision: float - Mean precision
            - mean_recall: float - Mean recall
            
    Raises:
        ImportError: If bert-score is not installed
        ValueError: If inputs are invalid
    """
    if config is None:
        config = BERTScoreConfig()
    
    if not config.enabled:
        logger.warning("BERTScore is disabled in config")
        return _empty_bertscore_result(len(outputs))
    
    # Validate inputs
    if not outputs or not references:
        raise ValueError("outputs and references must be non-empty lists")
    
    if len(outputs) != len(references):
        raise ValueError(
            f"Length mismatch: {len(outputs)} outputs vs {len(references)} references"
        )
    
    # Check for empty strings
    if any(not s.strip() for s in outputs):
        logger.warning("Some outputs are empty strings")
    
    if any(not s.strip() for s in references):
        logger.warning("Some references are empty strings")
    
    try:
        from bert_score import score
    except ImportError:
        logger.error(
            "bert-score not installed. Install with: pip install bert-score"
        )
        raise ImportError(
            "bert-score is required for BERTScore metrics. "
            "Install with: pip install bert-score"
        )
    
    # Compute BERTScore
    logger.info(
        f"Computing BERTScore for {len(outputs)} samples "
        f"(model={config.model_type}, batch_size={config.batch_size})"
    )
    
    try:
        P, R, F1 = score(
            cands=outputs,
            refs=references,
            model_type=config.model_type,
            num_layers=config.num_layers,
            batch_size=config.batch_size,
            verbose=config.verbose,
            device=config.device,
            rescale_with_baseline=config.rescale_with_baseline,
        )
        
        # Convert tensors to numpy
        precision = P.cpu().numpy()
        recall = R.cpu().numpy()
        f1 = F1.cpu().numpy()
        
        # Compute statistics
        result = {
            "per_sample_f1": f1.tolist(),
            "per_sample_precision": precision.tolist(),
            "per_sample_recall": recall.tolist(),
            "mean_f1": float(np.mean(f1)),
            "median_f1": float(np.median(f1)),
            "std_f1": float(np.std(f1)),
            "mean_precision": float(np.mean(precision)),
            "mean_recall": float(np.mean(recall)),
            "min_f1": float(np.min(f1)),
            "max_f1": float(np.max(f1)),
        }
        
        logger.info(
            f"BERTScore computed: mean F1={result['mean_f1']:.4f} "
            f"(±{result['std_f1']:.4f})"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error computing BERTScore: {e}")
        raise


def compute_bertscore_grouped(
    outputs: List[str],
    references: List[str],
    groups: List[str],
    config: Optional[BERTScoreConfig] = None,
) -> dict:
    """
    Compute BERTScore separately for different groups (e.g., ESL vs native).
    
    Args:
        outputs: List of generated texts
        references: List of reference texts
        groups: List of group labels (e.g., "esl", "native")
        config: BERTScore configuration
        
    Returns:
        Dictionary mapping group names to BERTScore results
    """
    if len(outputs) != len(references) != len(groups):
        raise ValueError("outputs, references, and groups must have same length")
    
    # Group samples by label
    grouped_samples = {}
    for output, reference, group in zip(outputs, references, groups):
        if group not in grouped_samples:
            grouped_samples[group] = {"outputs": [], "references": []}
        grouped_samples[group]["outputs"].append(output)
        grouped_samples[group]["references"].append(reference)
    
    # Compute BERTScore for each group
    results = {}
    for group_name, samples in grouped_samples.items():
        logger.info(f"Computing BERTScore for group '{group_name}' ({len(samples['outputs'])} samples)")
        results[group_name] = compute_bertscore(
            outputs=samples["outputs"],
            references=samples["references"],
            config=config,
        )
    
    # Compute overall metrics
    results["overall"] = compute_bertscore(
        outputs=outputs,
        references=references,
        config=config,
    )
    
    return results


def _empty_bertscore_result(n_samples: int) -> dict:
    """Return empty BERTScore result when disabled."""
    return {
        "per_sample_f1": [0.0] * n_samples,
        "per_sample_precision": [0.0] * n_samples,
        "per_sample_recall": [0.0] * n_samples,
        "mean_f1": 0.0,
        "median_f1": 0.0,
        "std_f1": 0.0,
        "mean_precision": 0.0,
        "mean_recall": 0.0,
        "min_f1": 0.0,
        "max_f1": 0.0,
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Mock data
    outputs = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Climate change poses significant risks to global ecosystems.",
    ]
    
    references = [
        "A fast brown fox leaps over a sleepy canine.",
        "ML represents a branch of AI technology.",
        "Global warming threatens ecological systems worldwide.",
    ]
    
    # Test with default config
    config = BERTScoreConfig(
        model_type="microsoft/deberta-base",  # Smaller model for testing
        batch_size=4,
        verbose=True,
    )
    
    try:
        result = compute_bertscore(outputs, references, config)
        
        print("\n=== BERTScore Results ===")
        print(f"Mean F1: {result['mean_f1']:.4f}")
        print(f"Median F1: {result['median_f1']:.4f}")
        print(f"Std F1: {result['std_f1']:.4f}")
        print(f"\nPer-sample F1 scores:")
        for i, f1 in enumerate(result['per_sample_f1']):
            print(f"  Sample {i+1}: {f1:.4f}")
        
        # Test grouped computation
        groups = ["similar", "similar", "different"]
        grouped_results = compute_bertscore_grouped(
            outputs, references, groups, config
        )
        
        print("\n=== Grouped Results ===")
        for group_name, group_result in grouped_results.items():
            print(f"{group_name}: mean F1 = {group_result['mean_f1']:.4f}")
        
    except ImportError as e:
        print(f"\nError: {e}")
        print("Install bert-score with: pip install bert-score")
