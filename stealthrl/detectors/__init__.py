"""
AI text detector wrappers.

This module provides unified interfaces for multiple AI text detection methods.
"""

from .base_detector import BaseDetector
from .fast_detectgpt import FastDetectGPTDetector
from .ghostbuster import GhostbusterDetector
from .binoculars import BinocularsDetector

__all__ = [
    "BaseDetector",
    "FastDetectGPTDetector",
    "GhostbusterDetector",
    "BinocularsDetector",
]
