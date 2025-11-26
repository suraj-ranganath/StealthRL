"""
Reward computation modules for StealthRL.

This module provides components for computing composite rewards including:
- Detector ensemble scores
- Semantic fidelity (BERTScore, cosine similarity)
- Quality metrics (perplexity, readability)
- Fairness penalties (ESL vs native FPR gap)
- Per-sample fairness proxies for learnable RL optimization
"""

from .composite_reward import CompositeReward
from .detector_reward import DetectorEnsembleReward
from .semantic_reward import SemanticFidelityReward
from .quality_reward import QualityReward
from .fairness_reward import FairnessReward
from .fairness_metrics import compute_fairness_proxy, compute_group_fpr_gap

__all__ = [
    "CompositeReward",
    "DetectorEnsembleReward",
    "SemanticFidelityReward",
    "QualityReward",
    "FairnessReward",
    "compute_fairness_proxy",
    "compute_group_fpr_gap",
]
