"""
Per-sample fairness metrics for ESL-aware RL training.

This module provides learnable, per-sample fairness penalties rather than
global FPR gaps, making fairness optimization directly tractable in RL loops.
"""

import torch
from typing import Optional


def compute_fairness_proxy(
    detector_scores: torch.Tensor,
    group_labels: torch.Tensor,
    mode: str = "esl_penalty"
) -> torch.Tensor:
    """
    Return a per-sample fairness penalty F', higher when ESL examples are more detectable.
    
    This allows the RL policy to directly optimize for fairness by providing per-sample
    gradients. Global FPR gaps are computed separately in evaluation for reporting.
    
    For mode='esl_penalty':
        F'_i = detector_scores[i] if group_labels[i] == 1 (ESL)
             = 0 otherwise
    
    This encourages the RL policy to suppress detector scores more aggressively on ESL text,
    reducing the ESL vs native false positive rate gap.
    
    Args:
        detector_scores: Detector scores for batch [batch_size]
        group_labels: Group membership labels [batch_size]
                     1 = ESL (non-native), 0 = native
        mode: Fairness proxy mode, currently supports 'esl_penalty'
        
    Returns:
        Per-sample fairness penalties [batch_size]
        
    Example:
        >>> detector_scores = torch.tensor([0.8, 0.3, 0.9, 0.2])
        >>> group_labels = torch.tensor([1, 0, 1, 0])  # ESL, native, ESL, native
        >>> fairness = compute_fairness_proxy(detector_scores, group_labels)
        >>> print(fairness)  # [0.8, 0.0, 0.9, 0.0] - only penalizes ESL detections
    """
    if mode == "esl_penalty":
        esl_mask = (group_labels == 1).float()
        return detector_scores * esl_mask
    else:
        raise NotImplementedError(f"Unknown fairness mode: {mode}")


def compute_group_fpr_gap(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    group_labels: torch.Tensor,
    threshold: float = 0.5
) -> tuple:
    """
    Compute true FPR gap between ESL and native groups for evaluation.
    
    This is used for reporting and model selection, NOT in the RL training loop.
    The per-sample proxy above is used during training for gradient-based optimization.
    
    Args:
        predictions: Detector predictions [n_samples]
        labels: True labels [n_samples] (0 = human, 1 = AI)
        group_labels: Group membership [n_samples] (0 = native, 1 = ESL)
        threshold: Classification threshold
        
    Returns:
        tuple of (fpr_esl, fpr_native, fpr_gap)
    """
    # Get human-written samples (label = 0)
    human_mask = (labels == 0)
    
    # ESL group
    esl_mask = (group_labels == 1) & human_mask
    if esl_mask.sum() > 0:
        esl_preds = predictions[esl_mask]
        fpr_esl = (esl_preds > threshold).float().mean().item()
    else:
        fpr_esl = 0.0
    
    # Native group
    native_mask = (group_labels == 0) & human_mask
    if native_mask.sum() > 0:
        native_preds = predictions[native_mask]
        fpr_native = (native_preds > threshold).float().mean().item()
    else:
        fpr_native = 0.0
    
    fpr_gap = fpr_esl - fpr_native
    
    return fpr_esl, fpr_native, fpr_gap
