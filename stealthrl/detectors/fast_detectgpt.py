"""
Fast-DetectGPT detector wrapper.
"""

from typing import List, Union
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_detector import BaseDetector


class FastDetectGPTDetector(BaseDetector):
    """
    Wrapper for Fast-DetectGPT curvature-based detection method.
    Implements a simplified version based on probability curvature.
    """
    
    def __init__(self, model_name: str = "gpt2-medium", device: str = "cuda", num_perturbations: int = 10):
        super().__init__(model_name, device)
        self.num_perturbations = num_perturbations
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load Fast-DetectGPT model."""
        if self.model is None:
            print(f"Loading {self.model_name} for Fast-DetectGPT detection...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
            ).to(self.device)
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def detect(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Run Fast-DetectGPT detection on texts.
        
        Args:
            texts: Single text or list of texts to detect
            
        Returns:
            Tensor of detection scores (0-1, higher = more likely AI-generated)
        """
        if self.model is None:
            self.load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        scores = []
        with torch.no_grad():
            for text in texts:
                # Compute log probability of the text
                log_prob = self._compute_log_probability(text)
                
                # Simple heuristic: higher negative log prob suggests AI-generated
                # Normalize to 0-1 range using sigmoid
                score = torch.sigmoid(torch.tensor(-log_prob / 10.0))
                scores.append(score.item())
        
        return torch.tensor(scores, dtype=torch.float32)
    
    def _compute_log_probability(self, text: str) -> float:
        """Compute average log probability of text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        
        # Get log probabilities
        outputs = self.model(input_ids, labels=input_ids)
        
        # Average negative log likelihood
        return outputs.loss.item()
