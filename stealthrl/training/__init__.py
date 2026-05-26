"""
RL training loops for StealthRL.

This module provides training utilities built on HuggingFace TRL
for GRPO and PPO-based RL training.
"""

from .trainer import StealthRLTrainer

__all__ = ["StealthRLTrainer"]
