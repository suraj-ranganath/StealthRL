"""
Evaluation utilities and StealthBench harness.

This module provides standardized evaluation metrics and the StealthBench
harness for comparing multiple detectors.
"""

from .stealthbench import StealthBench
from .metrics import compute_auroc, compute_fpr_at_tpr, compute_bertscore

__all__ = [
    "StealthBench",
    "compute_auroc",
    "compute_fpr_at_tpr",
    "compute_bertscore",
]
