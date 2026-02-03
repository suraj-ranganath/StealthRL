"""
StealthRL: Ensemble-Guided Text Transformation for Multi-Detector Transfer and Fair Detection Robustness

A research framework for training RL-based paraphrasers that reduce AI-detector scores
while preserving semantic meaning and addressing ESL fairness.

Key modules:
- data: ESL/native corpus loading and preprocessing
- metrics: BERTScore and other evaluation metrics
- tinker: Tinker platform integration for training
- rewards: Composite reward computation
- detectors: Detector wrappers
- baselines: Baseline methods (SICO)
"""

__version__ = "0.1.0"
__author__ = "Anonymous"
__affiliation__ = "Anonymous"
