"""
Composite reward computation combining multiple reward terms.

Implements a stable, normalized reward function for GRPO/PPO training,
inspired by AuthorMist (https://arxiv.org/abs/2503.08716) and RLHF best practices.
"""

from typing import Dict, List
import torch


class CompositeReward:
    """
    Combines multiple reward components into a single scalar reward with normalization.
    
    The composite reward formula is:
        R = -w₁·D' + w₂·S' + w₃·Q' - w₄·F'
    
    where:
        - D' = normalized detector scores (z-score or raw)
        - S' = thresholded semantic fidelity (maps [semantic_min, 1] → [0, 1])
        - Q' = thresholded quality score (maps [quality_min, 1] → [0, 1])
        - F' = per-sample fairness penalty (ESL-aware proxy)
    
    Our detector term D is analogous to AuthorMist's 1 - mean(detector_prob), but we
    augment it with semantic, quality, and fairness terms; see https://arxiv.org/abs/2503.08716
    for details on single-detector RL rewards.
    
    The normalization and thresholding make rewards more stable for GRPO/PPO optimization
    and prevent degenerate solutions (e.g., semantic collapse below acceptable threshold).
    """
    
    def __init__(
        self,
        detector_weight: float = 1.0,
        semantic_weight: float = 1.0,
        quality_weight: float = 0.5,
        fairness_weight: float = 0.2,
        normalize_terms: bool = True,
        detector_zscore: bool = True,
        semantic_min: float = 0.90,
        quality_min: float = 0.80,
    ):
        """
        Initialize composite reward with component weights and normalization settings.
        
        Args:
            detector_weight: Weight w₁ for detector ensemble term
            semantic_weight: Weight w₂ for semantic fidelity term
            quality_weight: Weight w₃ for quality metrics term
            fairness_weight: Weight w₄ for fairness penalty term
            normalize_terms: If True, apply thresholding to semantic and quality
            detector_zscore: If True, apply z-score normalization to detector scores
            semantic_min: Minimum acceptable semantic similarity (below this → 0 reward)
            quality_min: Minimum acceptable quality score (below this → 0 reward)
        """
        self.detector_weight = detector_weight
        self.semantic_weight = semantic_weight
        self.quality_weight = quality_weight
        self.fairness_weight = fairness_weight
        
        self.normalize_terms = normalize_terms
        self.detector_zscore = detector_zscore
        self.semantic_min = semantic_min
        self.quality_min = quality_min
        
    def _normalize_detectors(self, detector_scores: torch.Tensor) -> torch.Tensor:
        """
        Normalize detector scores using z-score for stability.
        
        Args:
            detector_scores: Raw detector scores [batch]
            
        Returns:
            Normalized detector scores (clamped to [-3, 3] for stability)
        """
        if self.detector_zscore and detector_scores.numel() > 1:
            mu = detector_scores.mean()
            sigma = detector_scores.std().clamp_min(1e-6)
            detector_scores_norm = ((detector_scores - mu) / sigma).clamp(-3.0, 3.0)
        else:
            detector_scores_norm = detector_scores
        
        return detector_scores_norm
    
    def _normalize_semantics(self, semantic_scores: torch.Tensor) -> torch.Tensor:
        """
        Threshold semantic scores to enforce minimum acceptable similarity.
        
        Maps [semantic_min, 1.0] → [0, 1], below threshold → 0.
        This prevents the model from drifting too far from the original meaning.
        
        Args:
            semantic_scores: Raw semantic similarity scores [batch]
            
        Returns:
            Thresholded semantic scores
        """
        if self.normalize_terms:
            S = semantic_scores
            S_thresh = torch.clamp(
                (S - self.semantic_min) / (1.0 - self.semantic_min + 1e-6), 
                min=0.0, 
                max=1.0
            )
        else:
            S_thresh = semantic_scores
        
        return S_thresh
    
    def _normalize_quality(self, quality_scores: torch.Tensor) -> torch.Tensor:
        """
        Threshold quality scores to enforce minimum acceptable quality.
        
        Maps [quality_min, 1.0] → [0, 1], below threshold → 0.
        This prevents degenerate outputs with poor fluency.
        
        Args:
            quality_scores: Raw quality scores [batch]
            
        Returns:
            Thresholded quality scores
        """
        if self.normalize_terms:
            Q = quality_scores
            Q_thresh = torch.clamp(
                (Q - self.quality_min) / (1.0 - self.quality_min + 1e-6), 
                min=0.0, 
                max=1.0
            )
        else:
            Q_thresh = quality_scores
        
        return Q_thresh
        
    def compute(
        self,
        detector_scores: torch.Tensor,
        semantic_scores: torch.Tensor,
        quality_scores: torch.Tensor,
        fairness_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute composite reward from component scores with normalization.
        
        Formula:
            R = -w₁·D' + w₂·S' + w₃·Q' - w₄·F'
        
        where D', S', Q', F' are normalized/thresholded versions of raw metrics.
        
        Args:
            detector_scores: Raw detector ensemble scores [batch] (lower = less detectable)
            semantic_scores: Raw semantic fidelity scores [batch] (higher = better preservation)
            quality_scores: Raw quality metric scores [batch] (higher = better quality)
            fairness_scores: Per-sample fairness penalties [batch] (ESL-aware proxy)
            
        Returns:
            Composite reward tensor [batch]
        """
        # Apply normalization/thresholding
        D_prime = self._normalize_detectors(detector_scores)
        S_prime = self._normalize_semantics(semantic_scores)
        Q_prime = self._normalize_quality(quality_scores)
        F_prime = fairness_scores  # Already per-sample proxy from fairness_metrics
        
        # Compute weighted sum
        reward = (
            -self.detector_weight * D_prime +      # Minimize detectability
            self.semantic_weight * S_prime +       # Maximize semantic preservation
            self.quality_weight * Q_prime +        # Maximize quality
            -self.fairness_weight * F_prime        # Minimize ESL bias
        )
        
        return reward
