"""
Tinker-adapted Composite Reward for StealthRL.

This module adapts the existing reward components to work with Tinker's
asynchronous compute model and provides a unified interface for computing
multi-objective rewards during RL training.
"""

import logging
from typing import Dict, Any
import torch

logger = logging.getLogger(__name__)


class TinkerCompositeReward:
    """
    Composite reward function for StealthRL adapted for Tinker.
    
    Combines multiple reward components:
    - R_det: Detector evasion (weighted ensemble)
    - R_sem: Semantic similarity (E5 encoder)
    - R_ppl: Fluency/perplexity (frozen LM)
    - R_fair: ESL fairness penalty
    
    Total: R = α*R_det + β*R_sem + γ*R_ppl + δ*R_fair
    """
    
    def __init__(
        self,
        # Reward weights
        detector_weight: float = 1.0,
        semantic_weight: float = 1.0,
        perplexity_weight: float = 0.5,
        fairness_weight: float = 0.2,
        
        # Detector config
        detector_names: list[str] = ["fast_detectgpt", "ghostbuster"],
        detector_weights: Dict[str, float] | None = None,
        detector_cache_path: str | None = None,
        
        # Semantic config
        semantic_model: str = "intfloat/e5-large-v2",
        semantic_threshold: float = 0.90,
        
        # Perplexity config
        ppl_model: str = "gpt2",
        ppl_min: float = 5.0,
        ppl_max: float = 80.0,
        ppl_target: float = 30.0,
        
        # Fairness config
        fairness_mode: str = "esl_penalty",
        
        # Normalization config (from Session 4 refinements)
        normalize_terms: bool = True,
        detector_zscore: bool = True,
        semantic_min: float = 0.90,
        quality_min: float = 0.80,
    ):
        """
        Initialize composite reward function.
        
        Args:
            detector_weight: Weight for detector evasion term (α)
            semantic_weight: Weight for semantic similarity term (β)
            perplexity_weight: Weight for perplexity term (γ)
            fairness_weight: Weight for fairness term (δ)
            detector_names: List of detector names to use
            detector_weights: Optional custom weights for each detector
            detector_cache_path: Path to SQLite cache for detector scores
            semantic_model: E5 model for semantic similarity
            semantic_threshold: Minimum acceptable semantic similarity
            ppl_model: Model for perplexity computation
            ppl_min: Minimum perplexity for normalization
            ppl_max: Maximum perplexity for normalization
            ppl_target: Target perplexity (human-like)
            fairness_mode: Fairness computation mode
            normalize_terms: Whether to normalize reward terms
            detector_zscore: Whether to z-score normalize detector scores
            semantic_min: Minimum semantic similarity threshold
            quality_min: Minimum quality threshold
        """
        self.detector_weight = detector_weight
        self.semantic_weight = semantic_weight
        self.perplexity_weight = perplexity_weight
        self.fairness_weight = fairness_weight
        
        # Initialize detector ensemble
        from stealthrl.tinker.detectors import DetectorEnsemble
        self.detector_ensemble = DetectorEnsemble(
            detector_names=detector_names,
            detector_weights=detector_weights,
            cache_path=detector_cache_path,
        )
        
        # Initialize semantic similarity
        from stealthrl.tinker.semantic import SemanticSimilarity
        self.semantic_sim = SemanticSimilarity(
            model_name=semantic_model,
            threshold=semantic_threshold,
        )
        
        # Initialize perplexity
        from stealthrl.tinker.perplexity import PerplexityReward
        self.ppl_reward = PerplexityReward(
            model_name=ppl_model,
            ppl_min=ppl_min,
            ppl_max=ppl_max,
            ppl_target=ppl_target,
        )
        
        # Fairness config
        self.fairness_mode = fairness_mode
        
        # Normalization config
        self.normalize_terms = normalize_terms
        self.detector_zscore = detector_zscore
        self.semantic_min = semantic_min
        self.quality_min = quality_min
        
        # Running statistics for z-score normalization
        self.detector_mean = 0.5
        self.detector_std = 0.15
        self.detector_count = 0
        
        logger.info(f"Initialized TinkerCompositeReward with weights: "
                   f"det={detector_weight}, sem={semantic_weight}, "
                   f"ppl={perplexity_weight}, fair={fairness_weight}")
    
    async def compute(
        self,
        original_text: str,
        paraphrase_text: str,
        human_reference: str,
        domain: str,
        is_esl: bool,
    ) -> Dict[str, Any]:
        """
        Compute composite reward for a paraphrase.
        
        Args:
            original_text: Original AI-generated text
            paraphrase_text: Generated paraphrase
            human_reference: Human reference text
            domain: Text domain
            is_esl: Whether text is ESL-style
        
        Returns:
            Dictionary with total_reward and component metrics
        """
        # Verifiable reward checks (reject degenerate outputs)
        if not paraphrase_text.strip():
            return {
                "total_reward": -1.0,
                "detector_reward": 0.0,
                "semantic_reward": 0.0,
                "perplexity_reward": 0.0,
                "fairness_reward": 0.0,
            }
        
        # Length check (reject too short/long)
        para_len = len(paraphrase_text.split())
        orig_len = len(original_text.split())
        if para_len < 10 or para_len > 3 * orig_len:
            return {
                "total_reward": -0.5,
                "detector_reward": 0.0,
                "semantic_reward": 0.0,
                "perplexity_reward": 0.0,
                "fairness_reward": 0.0,
                "length_penalty": 1.0,
            }
        
        # Compute detector ensemble score
        detector_result = await self.detector_ensemble.compute(paraphrase_text)
        detector_prob = detector_result["ensemble_prob"]  # P(AI | text)
        
        # R_det = 1 - P(AI) (higher is better = more human-like)
        detector_reward_raw = 1.0 - detector_prob
        
        # Compute semantic similarity
        semantic_result = await self.semantic_sim.compute(
            text1=original_text,
            text2=paraphrase_text,
        )
        semantic_sim = semantic_result["similarity"]
        
        # R_sem = max(0, sim - threshold) after normalization
        semantic_reward_raw = max(0.0, semantic_sim - self.semantic_min) if semantic_sim >= self.semantic_min else 0.0
        
        # Compute perplexity reward
        ppl_result = await self.ppl_reward.compute(paraphrase_text)
        perplexity = ppl_result["perplexity"]
        ppl_reward_raw = ppl_result["reward"]
        
        # Compute fairness penalty (per-sample for ESL)
        if is_esl:
            # F' = detector_prob * 1[ESL] (penalize high detection on ESL)
            fairness_penalty = detector_prob
        else:
            fairness_penalty = 0.0
        
        # Apply normalization if enabled
        if self.normalize_terms:
            detector_reward = self._normalize_detector(detector_reward_raw)
            semantic_reward = self._normalize_semantic(semantic_reward_raw)
            ppl_reward = self._normalize_quality(ppl_reward_raw)
        else:
            detector_reward = detector_reward_raw
            semantic_reward = semantic_reward_raw
            ppl_reward = ppl_reward_raw
        
        # Compute total reward
        total_reward = (
            self.detector_weight * detector_reward +
            self.semantic_weight * semantic_reward +
            self.perplexity_weight * ppl_reward -
            self.fairness_weight * fairness_penalty
        )
        
        return {
            "total_reward": total_reward,
            "detector_reward": detector_reward,
            "semantic_reward": semantic_reward,
            "perplexity_reward": ppl_reward,
            "fairness_reward": -fairness_penalty,
            "detector_prob": detector_prob,
            "semantic_sim": semantic_sim,
            "perplexity": perplexity,
            "is_esl": float(is_esl),
        }
    
    def _normalize_detector(self, score: float) -> float:
        """
        Z-score normalization for detector scores.
        
        Normalizes using running mean/std and clips to [-3, 3].
        """
        if not self.detector_zscore:
            return score
        
        # Update running statistics
        self.detector_count += 1
        delta = score - self.detector_mean
        self.detector_mean += delta / self.detector_count
        delta2 = score - self.detector_mean
        self.detector_std = ((self.detector_std ** 2 * (self.detector_count - 1) + delta * delta2) / self.detector_count) ** 0.5
        
        # Z-score with clipping
        if self.detector_std > 1e-6:
            normalized = (score - self.detector_mean) / (self.detector_std + 1e-6)
            return max(-3.0, min(3.0, normalized))
        return score
    
    def _normalize_semantic(self, score: float) -> float:
        """
        Threshold-based normalization for semantic similarity.
        
        Maps [semantic_min, 1.0] -> [0, 1], below threshold -> 0.
        """
        if not self.normalize_terms:
            return score
        
        if score < 0.0:
            return 0.0
        
        # Already thresholded in compute(), just scale to [0, 1]
        # score is (sim - semantic_min) if sim >= semantic_min else 0
        scale = 1.0 - self.semantic_min
        if scale > 1e-6:
            return min(1.0, score / scale)
        return score
    
    def _normalize_quality(self, score: float) -> float:
        """
        Threshold-based normalization for quality (perplexity).
        
        Maps [quality_min, 1.0] -> [0, 1], below threshold -> 0.
        """
        if not self.normalize_terms:
            return score
        
        # Assuming ppl_reward is already normalized to [0, 1]
        if score < self.quality_min:
            return 0.0
        
        # Map [quality_min, 1.0] to [0, 1]
        scale = 1.0 - self.quality_min
        if scale > 1e-6:
            return (score - self.quality_min) / scale
        return score
