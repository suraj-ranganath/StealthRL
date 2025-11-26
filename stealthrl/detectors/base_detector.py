"""
Base class for AI text detectors.
"""

from abc import ABC, abstractmethod
from typing import List, Union
import torch


class BaseDetector(ABC):
    """
    Abstract base class for AI text detectors.
    All detector implementations should inherit from this class.
    """
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initialize detector.
        
        Args:
            model_name: Model identifier or path
            device: Device to run detector on
        """
        self.model_name = model_name
        self.device = device
        
    @abstractmethod
    def detect(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Detect AI-generated text.
        
        Args:
            texts: Single text or list of texts to detect
            
        Returns:
            Tensor of detection scores (0-1, higher = more likely AI-generated)
        """
        pass
    
    @abstractmethod
    def load_model(self):
        """Load the detector model."""
        pass
