"""
Detector ensemble reward computation.
"""

from typing import List, Dict, Any
import torch
import numpy as np
from ..detectors import FastDetectGPTDetector, GhostbusterDetector, BinocularsDetector


class DetectorEnsembleReward:
    """
    Computes normalized ensemble score from multiple AI text detectors.
    """
    
    def __init__(self, detectors: List[str], device: str = "cuda"):
        """
        Initialize detector ensemble.
        
        Args:
            detectors: List of detector names to use in ensemble
            device: Device to run detectors on
        """
        self.detector_names = detectors
        self.device = device
        self.detector_instances = {}
        self._initialize_detectors()
        
    def _initialize_detectors(self):
        """Initialize detector instances based on names."""
        detector_map = {
            "fast-detectgpt": FastDetectGPTDetector,
            "ghostbuster": GhostbusterDetector,
            "binoculars": BinocularsDetector,
        }
        
        for detector_name in self.detector_names:
            if detector_name in detector_map:
                self.detector_instances[detector_name] = detector_map[detector_name](device=self.device)
            else:
                print(f"Warning: Unknown detector {detector_name}, skipping")
        
    def compute(self, texts: List[str]) -> torch.Tensor:
        """
        Compute ensemble detector scores for texts.
        
        Args:
            texts: List of text samples to score
            
        Returns:
            Tensor of normalized detector scores (0-1, lower = less detectable)
        """
        if not self.detector_instances:
            # No detectors available, return zeros
            return torch.zeros(len(texts), dtype=torch.float32)
        
        all_scores = []
        
        # Run each detector
        for detector_name, detector in self.detector_instances.items():
            try:
                scores = detector.detect(texts)
                # Ensure scores are on CPU and normalized to 0-1
                scores = scores.cpu()
                scores = torch.clamp(scores, 0.0, 1.0)
                all_scores.append(scores)
            except Exception as e:
                print(f"Error running detector {detector_name}: {e}")
                # Skip this detector
                continue
        
        if not all_scores:
            # All detectors failed
            return torch.zeros(len(texts), dtype=torch.float32)
        
        # Stack and compute mean across detectors
        scores_tensor = torch.stack(all_scores, dim=0)  # [num_detectors, batch_size]
        ensemble_scores = torch.mean(scores_tensor, dim=0)  # [batch_size]
        
        return ensemble_scores
    
    def compute_individual_scores(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Compute individual detector scores (useful for analysis).
        
        Args:
            texts: List of text samples to score
            
        Returns:
            Dictionary mapping detector names to their scores
        """
        scores_dict = {}
        
        for detector_name, detector in self.detector_instances.items():
            try:
                scores = detector.detect(texts)
                scores_dict[detector_name] = scores.cpu()
            except Exception as e:
                print(f"Error running detector {detector_name}: {e}")
                scores_dict[detector_name] = torch.zeros(len(texts), dtype=torch.float32)
        
        return scores_dict
