"""
StealthRL Tinker Integration.

This module provides Tinker-based RL training infrastructure for StealthRL.
"""

from .env import StealthEnv, StealthEnvGroupBuilder
from .dataset import StealthRLDataset, StealthRLDatasetBuilder, StealthRLExample
from .reward import TinkerCompositeReward
from .detectors import DetectorEnsemble
from .semantic import SemanticSimilarity
from .perplexity import PerplexityReward
# Note: train module not imported here to avoid RuntimeWarning when running as __main__
from .inference import ChunkingInference, ChunkCandidate, ChunkResult
from .evaluation import EvaluationSuite, EvaluationExample, ModelMetrics, ComparisonReport

__all__ = [
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
