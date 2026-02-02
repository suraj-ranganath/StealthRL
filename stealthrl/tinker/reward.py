"""
Tinker-adapted Composite Reward for StealthRL.

This module adapts the existing reward components to work with Tinker's
asynchronous compute model and provides a unified interface for computing
multi-objective rewards during RL training.
"""

import logging
import time
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
    
    Total: R = α*R_det + β*R_sem + γ*R_ppl
    """
    
    def __init__(
        self,
        # Reward weights
        detector_weight: float = 1.0,
        semantic_weight: float = 1.0,
        perplexity_weight: float = 0.5,
        
        # Component enable/disable flags (NEW)
        enable_semantic: bool = True,
        enable_perplexity: bool = True,
        compute_perplexity_eval_only: bool = False,  # Compute ppl for logging only (not in reward)
        
        # Detector config
        detector_names: list[str] = ["fast_detectgpt", "ghostbuster"],
        detector_weights: Dict[str, float] | None = None,
        detector_cache_path: str | None = None,
        detector_batch_size: int = 32,  # Batch size for detector inference
        
        # Detector model selection (VALIDATED MODELS - Jan 2026)
        fast_detectgpt_model: str = "gpt-neo-2.7B",  # AUROC: 0.691
        roberta_openai_model: str = "roberta-large-openai-detector",  # AUROC: 0.891
        ghostbuster_model: str = "roberta-base",
        binoculars_performer: str = "gpt2",
        binoculars_observer: str = "gpt2-medium",
        
        # Batch sizes for detector inference
        roberta_batch_size: int = 128,
        fast_detectgpt_batch_size: int = 32,
        
        # Semantic config
        semantic_model: str = "intfloat/e5-large-v2",
        semantic_threshold: float = 0.90,
        
        # Perplexity config
        ppl_model: str = "gpt2",
        ppl_min: float = 5.0,
        ppl_max: float = 80.0,
        ppl_target: float = 30.0,
        
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
            detector_names: List of detector names to use
            detector_weights: Optional custom weights for each detector
            detector_cache_path: Path to SQLite cache for detector scores
            fast_detectgpt_model: Model for Fast-DetectGPT ("gpt2", "gpt-neo-2.7B", "falcon-7b")
            roberta_openai_model: Model for RoBERTa OpenAI detector (validated: "roberta-large-openai-detector")
            ghostbuster_model: Model for Ghostbuster
            binoculars_performer: Performer model for Binoculars
            binoculars_observer: Observer model for Binoculars
            semantic_model: E5 model for semantic similarity
            semantic_threshold: Minimum acceptable semantic similarity
            ppl_model: Model for perplexity computation
            ppl_min: Minimum perplexity for normalization
            ppl_max: Maximum perplexity for normalization
            ppl_target: Target perplexity (human-like)
            normalize_terms: Whether to normalize reward terms
            detector_zscore: Whether to z-score normalize detector scores
            semantic_min: Minimum semantic similarity threshold
            quality_min: Minimum quality threshold
        """
        self.detector_weight = detector_weight
        self.semantic_weight = semantic_weight
        self.perplexity_weight = perplexity_weight
        
        # Component enable/disable flags
        self.enable_semantic = enable_semantic
        self.enable_perplexity = enable_perplexity
        self.compute_perplexity_eval_only = compute_perplexity_eval_only
        self.roberta_batch_size = roberta_batch_size
        self.fast_detectgpt_batch_size = fast_detectgpt_batch_size
        
        # Initialize detector ensemble
        from stealthrl.tinker.detectors import DetectorEnsemble
        self.detector_ensemble = DetectorEnsemble(
            detector_names=detector_names,
            detector_weights=detector_weights,
            cache_path=detector_cache_path,
            fast_detectgpt_model=fast_detectgpt_model,
            roberta_openai_model=roberta_openai_model,
            ghostbuster_model=ghostbuster_model,
            binoculars_performer=binoculars_performer,
            binoculars_observer=binoculars_observer,
            roberta_batch_size=roberta_batch_size,
            fast_detectgpt_batch_size=fast_detectgpt_batch_size,
        )
        
        # Pre-warm detector models to avoid lazy loading during training
        logger.info("Pre-warming detector models...")
        self.detector_ensemble.prewarm_models()
        
        # Initialize semantic similarity (only if enabled)
        self.semantic_sim = None
        if enable_semantic:
            from stealthrl.tinker.semantic import SemanticSimilarity
            self.semantic_sim = SemanticSimilarity(
                model_name=semantic_model,
                threshold=semantic_threshold,
            )
        
        # Initialize perplexity (if enabled OR compute_eval_only)
        self.ppl_reward = None
        if enable_perplexity or compute_perplexity_eval_only:
            logger.info(f"Initializing PerplexityReward (enable_perplexity={enable_perplexity}, compute_eval_only={compute_perplexity_eval_only})")
            from stealthrl.tinker.perplexity import PerplexityReward
            self.ppl_reward = PerplexityReward(
                model_name=ppl_model,
                ppl_min=ppl_min,
                ppl_max=ppl_max,
                ppl_target=ppl_target,
            )
            logger.info("✓ PerplexityReward initialized successfully")
        else:
            logger.info(f"Perplexity disabled (enable_perplexity={enable_perplexity}, compute_eval_only={compute_perplexity_eval_only})")

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
                   f"ppl={perplexity_weight}")
    
    async def compute(
        self,
        original_text: str,
        paraphrase_text: str,
        human_reference: str,
        domain: str,
    ) -> Dict[str, Any]:
        """
        Compute composite reward for a paraphrase.
        
        Args:
            original_text: Original AI-generated text
            paraphrase_text: Generated paraphrase
            human_reference: Human reference text
            domain: Text domain

        
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
        
        start_time = time.perf_counter()

        # Compute detector ensemble score
        detector_start = time.perf_counter()
        detector_result = await self.detector_ensemble.compute(paraphrase_text)
        detector_time = time.perf_counter() - detector_start
        detector_prob = detector_result["ensemble_prob"]  # P(AI | text)
        
        # R_det = 1 - P(AI) (higher is better = more human-like)
        detector_reward_raw = 1.0 - detector_prob
        
        # Compute semantic similarity (only if enabled)
        semantic_sim = 1.0  # Default to perfect if disabled
        semantic_reward_raw = 1.0
        semantic_time = 0.0
        if self.enable_semantic and self.semantic_sim is not None:
            semantic_start = time.perf_counter()
            semantic_result = await self.semantic_sim.compute(
                text1=original_text,
                text2=paraphrase_text,
            )
            semantic_time = time.perf_counter() - semantic_start
            semantic_sim = semantic_result["similarity"]
            # Use full [0, 1] range for proper multi-objective RL
            semantic_reward_raw = semantic_sim
        
        # Compute perplexity reward (only if enabled for training)
        perplexity = 30.0  # Default to target if disabled
        ppl_reward_raw = 1.0
        perplexity_time = 0.0
        compute_ppl = self.enable_perplexity or self.compute_perplexity_eval_only
        if compute_ppl and self.ppl_reward is not None:
            perplexity_start = time.perf_counter()
            ppl_result = await self.ppl_reward.compute(paraphrase_text)
            perplexity_time = time.perf_counter() - perplexity_start
            perplexity = ppl_result["perplexity"]
            ppl_reward_raw = ppl_result["reward"]
        
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
            self.perplexity_weight * ppl_reward
        )
        
        total_time = time.perf_counter() - start_time

        # Build return dict - only include computed metrics, not defaults
        result = {
            "total_reward": total_reward,
            "detector_reward": detector_reward,
            "detector_prob": detector_prob,
            "time/reward/total": total_time,
            "time/reward/detector": detector_time,
        }
        
        # Add individual detector scores for analysis
        for detector_name, score in detector_result.get("detector_scores", {}).items():
            result[f"detector/{detector_name}"] = score
        
        # Only include semantic/perplexity metrics if enabled (don't waste GPU during training)
        if self.enable_semantic:
            result["semantic_reward"] = semantic_reward
            result["semantic_sim"] = semantic_sim
            result["time/reward/semantic"] = semantic_time
        
        # Include perplexity if enabled for reward OR eval-only monitoring
        if self.enable_perplexity or self.compute_perplexity_eval_only:
            if self.enable_perplexity:
                result["perplexity_reward"] = ppl_reward
            result["perplexity"] = perplexity
            result["time/reward/perplexity"] = perplexity_time
        
        return result

    async def compute_batch(
        self,
        original_texts: list[str],
        paraphrase_texts: list[str],
        human_references: list[str],
        domains: list[str],
    ) -> list[Dict[str, Any]]:
        """Compute composite rewards for a batch of paraphrases."""
        start_time = time.perf_counter()

        results: list[Dict[str, Any]] = []
        valid_indices: list[int] = []
        valid_originals: list[str] = []
        valid_paraphrases: list[str] = []

        for idx, (original, paraphrase) in enumerate(zip(original_texts, paraphrase_texts, strict=True)):
            para_len = len(paraphrase.split())
            orig_len = len(original.split())
            if para_len < 10 or (orig_len > 0 and para_len > 3 * orig_len):
                results.append(
                    {
                        "total_reward": -0.5,
                        "detector_reward": 0.0,
                        "semantic_reward": 0.0,
                        "perplexity_reward": 0.0,
                        "detector_prob": 0.5,
                        "semantic_sim": 0.0,
                        "perplexity": 0.0,
                        "length_penalty": 1.0,
                        "text_length": len(paraphrase),
                    }
                )
            else:
                results.append({})
                valid_indices.append(idx)
                valid_originals.append(original)
                valid_paraphrases.append(paraphrase)

        if not valid_indices:
            return results

        detector_start = time.perf_counter()
        detector_results = await self.detector_ensemble.compute_batch(valid_paraphrases)
        detector_time = time.perf_counter() - detector_start

        # Compute semantic similarity only if enabled
        semantic_time = 0.0
        semantic_sims = [1.0] * len(valid_paraphrases)  # Default: perfect similarity
        if self.enable_semantic and self.semantic_sim is not None:
            semantic_start = time.perf_counter()
            semantic_results = await self.semantic_sim.compute_batch(valid_originals, valid_paraphrases)
            semantic_time = time.perf_counter() - semantic_start
            semantic_sims = semantic_results["similarities"]

        # Compute perplexity only if enabled (or eval-only monitoring)
        perplexity_time = 0.0
        perplexities = [30.0] * len(valid_paraphrases)  # Default: target perplexity
        ppl_rewards_raw = [1.0] * len(valid_paraphrases)  # Default: perfect score
        compute_ppl = self.enable_perplexity or self.compute_perplexity_eval_only
        if compute_ppl and self.ppl_reward is not None:
            perplexity_start = time.perf_counter()
            perplexity_results = await self.ppl_reward.compute_batch(valid_paraphrases)
            perplexity_time = time.perf_counter() - perplexity_start
            perplexities = perplexity_results["perplexities"]
            ppl_rewards_raw = perplexity_results["rewards"]

        total_time = time.perf_counter() - start_time

        for idx, detector_result in enumerate(detector_results):
            detector_prob = detector_result["ensemble_prob"]
            detector_reward_raw = 1.0 - detector_prob

            semantic_sim = semantic_sims[idx]
            # Use full [0, 1] range for proper multi-objective RL
            semantic_reward_raw = semantic_sim

            ppl_reward_raw = ppl_rewards_raw[idx]
            perplexity = perplexities[idx]

            if self.normalize_terms:
                detector_reward = self._normalize_detector(detector_reward_raw)
                semantic_reward = self._normalize_semantic(semantic_reward_raw)
                ppl_reward = self._normalize_quality(ppl_reward_raw)
            else:
                detector_reward = detector_reward_raw
                semantic_reward = semantic_reward_raw
                ppl_reward = ppl_reward_raw

            total_reward = (
                self.detector_weight * detector_reward +
                self.semantic_weight * semantic_reward +
                self.perplexity_weight * ppl_reward
            )

            result = {
                "total_reward": total_reward,
                "detector_reward": detector_reward,
                "detector_prob": detector_prob,
                "time/reward/total": total_time,
                "time/reward/detector": detector_time,
                "text_length": len(valid_paraphrases[idx]),
            }
            
            # Add individual detector scores for analysis
            for detector_name, score in detector_result.get("detector_scores", {}).items():
                result[f"detector/{detector_name}"] = score
            
            # Only include semantic/perplexity metrics if enabled
            if self.enable_semantic:
                result["semantic_reward"] = semantic_reward
                result["semantic_sim"] = semantic_sim
                result["time/reward/semantic"] = semantic_time
            
            # Include perplexity if enabled for reward OR eval-only monitoring
            if self.enable_perplexity or self.compute_perplexity_eval_only:
                if self.enable_perplexity:
                    result["perplexity_reward"] = ppl_reward
                result["perplexity"] = perplexity
                result["time/reward/perplexity"] = perplexity_time
            results[valid_indices[idx]] = result

        return results
    
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
