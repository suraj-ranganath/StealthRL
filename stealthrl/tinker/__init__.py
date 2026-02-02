"""
StealthRL Tinker Integration.

This module provides Tinker-based RL training infrastructure for StealthRL.
"""

import logging

logger = logging.getLogger(__name__)

# Prefer lazy/guarded imports so analysis scripts that only need detectors
# don't fail if heavy deps (e.g., tinker) aren't installed in the current env.
try:
    from .env import StealthEnv, StealthEnvGroupBuilder
    from .dataset import StealthRLDataset, StealthRLDatasetBuilder, StealthRLExample
    from .reward import TinkerCompositeReward
    from .detectors import DetectorEnsemble
    from .semantic import SemanticSimilarity
    from .perplexity import PerplexityReward
    from .inference import ChunkingInference, ChunkCandidate, ChunkResult
    from .evaluation import EvaluationSuite, EvaluationExample, ModelMetrics, ComparisonReport
except Exception as e:  # pragma: no cover - best effort for analysis scripts
    logger.warning("stealthrl.tinker: partial import (some deps missing): %s", e)

__all__ = [
    # May be missing if guarded imports fail; callers should import modules directly if needed.
    "StealthEnv",
    "StealthEnvGroupBuilder",
    "StealthRLDataset",
    "StealthRLDatasetBuilder",
    "StealthRLExample",
    "TinkerCompositeReward",
    "DetectorEnsemble",
    "SemanticSimilarity",
    "PerplexityReward",
    "ChunkingInference",
    "ChunkCandidate",
    "ChunkResult",
    "EvaluationSuite",
    "EvaluationExample",
    "ModelMetrics",
    "ComparisonReport",
]
