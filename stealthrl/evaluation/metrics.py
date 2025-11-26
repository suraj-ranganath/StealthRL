"""
Evaluation metrics for StealthRL.

This module provides all evaluation metrics including AUROC, FPR@TPR, 
BERTScore, perplexity, and group fairness metrics.

Note: For RL training, use stealthrl.rewards.fairness_metrics.compute_fairness_proxy
for per-sample gradients. This module provides global metrics for evaluation only.
"""

from typing import List, Dict, Tuple
import torch
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from bert_score import score as bertscore_fn


def compute_auroc(y_true: List[int], y_scores: List[float]) -> float:
    """
    Compute Area Under ROC Curve.
    
    Args:
        y_true: True labels (0 = human, 1 = AI)
        y_scores: Predicted scores
        
    Returns:
        AUROC score
    """
    try:
        return roc_auc_score(y_true, y_scores)
    except ValueError as e:
        print(f"Error computing AUROC: {e}")
        return 0.0


def compute_fpr_at_tpr(
    y_true: List[int],
    y_scores: List[float],
    target_tpr: float = 0.95
) -> float:
    """
    Compute False Positive Rate at a target True Positive Rate.
    
    Args:
        y_true: True labels
        y_scores: Predicted scores
        target_tpr: Target TPR (e.g., 0.95 for 95% TPR)
        
    Returns:
        FPR at target TPR
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        idx = np.argmin(np.abs(tpr - target_tpr))
        return float(fpr[idx])
    except ValueError as e:
        print(f"Error computing FPR@TPR: {e}")
        return 0.0


def compute_bertscore(
    original_texts: List[str],
    paraphrased_texts: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    verbose: bool = False
) -> Dict[str, float]:
    """
    Compute BERTScore for semantic similarity.
    
    Args:
        original_texts: Original text samples
        paraphrased_texts: Paraphrased versions
        model_type: Model to use for BERTScore
        verbose: Whether to show progress
        
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    try:
        P, R, F1 = bertscore_fn(
            paraphrased_texts,
            original_texts,
            model_type=model_type,
            verbose=verbose,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        return {
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F1.mean())
        }
    except Exception as e:
        print(f"Error computing BERTScore: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def compute_perplexity(
    texts: List[str],
    model_name: str = "gpt2",
    device: str = "cuda"
) -> List[float]:
    """
    Compute perplexity for texts using a language model.
    
    Args:
        texts: Text samples to evaluate
        model_name: Model to use for perplexity
        device: Device to run on
        
    Returns:
        List of perplexity values
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    
    perplexities = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs.input_ids.to(device)
            
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            ppl = torch.exp(loss).item()
            perplexities.append(ppl)
    
    return perplexities


def compute_fpr_gap(
    y_true_esl: List[int],
    y_scores_esl: List[float],
    y_true_native: List[int],
    y_scores_native: List[float],
    threshold: float = 0.5
) -> float:
    """
    Compute FPR gap between ESL and native writers.
    
    Args:
        y_true_esl: True labels for ESL texts
        y_scores_esl: Predicted scores for ESL texts
        y_true_native: True labels for native texts
        y_scores_native: Predicted scores for native texts
        threshold: Detection threshold
        
    Returns:
        FPR gap (ESL FPR - native FPR)
    """
    # Get human texts only (label=0)
    human_esl_scores = [s for l, s in zip(y_true_esl, y_scores_esl) if l == 0]
    human_native_scores = [s for l, s in zip(y_true_native, y_scores_native) if l == 0]
    
    if not human_esl_scores or not human_native_scores:
        return 0.0
    
    # Compute FPR for each group
    fpr_esl = sum(1 for s in human_esl_scores if s > threshold) / len(human_esl_scores)
    fpr_native = sum(1 for s in human_native_scores if s > threshold) / len(human_native_scores)
    
    return fpr_esl - fpr_native


def compute_group_fpr_gap_eval(
    predictions: np.ndarray,
    labels: np.ndarray,
    group_labels: np.ndarray,
    threshold: float = 0.5
) -> Tuple[float, float, float]:
    """
    Compute FPR gap between ESL and native groups for evaluation and reporting.
    
    This is the global evaluation version. For per-sample fairness penalties
    in RL training, use stealthrl.rewards.fairness_metrics.compute_fairness_proxy.
    
    Args:
        predictions: Detector predictions [n_samples]
        labels: True labels [n_samples] (0 = human, 1 = AI)
        group_labels: Group membership [n_samples] (0 = native, 1 = ESL)
        threshold: Classification threshold
        
    Returns:
        Tuple of (fpr_esl, fpr_native, fpr_gap)
        
    Example:
        >>> preds = np.array([0.8, 0.3, 0.9, 0.2, 0.7, 0.4])
        >>> labels = np.array([0, 0, 0, 0, 0, 0])  # All human-written
        >>> groups = np.array([1, 1, 1, 0, 0, 0])  # ESL, ESL, ESL, native, native, native
        >>> fpr_esl, fpr_native, gap = compute_group_fpr_gap_eval(preds, labels, groups)
        >>> print(f"ESL FPR: {fpr_esl:.2f}, Native FPR: {fpr_native:.2f}, Gap: {gap:.2f}")
    """
    # Get human-written samples only (label = 0)
    human_mask = (labels == 0)
    
    # ESL group
    esl_mask = (group_labels == 1) & human_mask
    if esl_mask.sum() > 0:
        esl_preds = predictions[esl_mask]
        fpr_esl = (esl_preds > threshold).mean()
    else:
        fpr_esl = 0.0
    
    # Native group
    native_mask = (group_labels == 0) & human_mask
    if native_mask.sum() > 0:
        native_preds = predictions[native_mask]
        fpr_native = (native_preds > threshold).mean()
    else:
        fpr_native = 0.0
    
    fpr_gap = fpr_esl - fpr_native
    
    return fpr_esl, fpr_native, fpr_gap
