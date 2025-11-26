"""
Fairness penalty computation for ESL vs native FPR gap.
"""

from typing import List, Dict, Optional
import torch
import numpy as np


class FairnessReward:
    """
    Computes fairness penalty based on ESL vs native false positive rate gap.
    """
    
    def __init__(
        self,
        esl_validation_set: Optional[List[str]] = None,
        native_validation_set: Optional[List[str]] = None,
        threshold: float = 0.5
    ):
        """
        Initialize fairness reward.
        
        Args:
            esl_validation_set: Validation texts from ESL writers
            native_validation_set: Validation texts from native writers
            threshold: Detection threshold for computing FPR
        """
        self.esl_validation_set = esl_validation_set or []
        self.native_validation_set = native_validation_set or []
        self.threshold = threshold
        
    def compute(
        self,
        detector_scores_esl: torch.Tensor,
        detector_scores_native: torch.Tensor,
        batch_penalties: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute fairness penalty based on FPR gap.
        
        Args:
            detector_scores_esl: Detector scores on ESL validation set
            detector_scores_native: Detector scores on native validation set
            batch_penalties: Optional per-sample penalties for the current batch
            
        Returns:
            Tensor of fairness penalties (lower = more fair)
        """
        if batch_penalties is not None:
            # Use provided batch penalties
            return batch_penalties
        
        # Compute false positive rates
        # Assuming all validation texts are human-written (label=0)
        fpr_esl = self._compute_fpr(detector_scores_esl, self.threshold)
        fpr_native = self._compute_fpr(detector_scores_native, self.threshold)
        
        # Gap between ESL and native FPR (positive = ESL flagged more)
        fpr_gap = fpr_esl - fpr_native
        
        # Return absolute gap as penalty (higher gap = higher penalty)
        # Broadcast to match batch size
        batch_size = len(detector_scores_esl) if len(detector_scores_esl.shape) > 0 else 1
        penalty = torch.abs(torch.tensor([fpr_gap], dtype=torch.float32))
        
        return penalty.expand(batch_size)
    
    def _compute_fpr(self, scores: torch.Tensor, threshold: float) -> float:
        """
        Compute false positive rate given detection scores.
        
        Args:
            scores: Detection scores (0-1, higher = more likely AI)
            threshold: Classification threshold
            
        Returns:
            False positive rate
        """
        if len(scores) == 0:
            return 0.0
        
        # Count how many scores exceed threshold
        false_positives = (scores > threshold).sum().item()
        total = len(scores)
        
        return false_positives / total if total > 0 else 0.0
    
    def compute_fpr_gap(
        self,
        detector_scores_esl: torch.Tensor,
        detector_scores_native: torch.Tensor,
        threshold: float = None
    ) -> float:
        """
        Compute the FPR gap between ESL and native writers.
        
        Args:
            detector_scores_esl: Scores on ESL texts
            detector_scores_native: Scores on native texts
            threshold: Detection threshold (uses instance threshold if None)
            
        Returns:
            FPR gap (ESL FPR - native FPR)
        """
        threshold = threshold or self.threshold
        fpr_esl = self._compute_fpr(detector_scores_esl, threshold)
        fpr_native = self._compute_fpr(detector_scores_native, threshold)
        return fpr_esl - fpr_native
