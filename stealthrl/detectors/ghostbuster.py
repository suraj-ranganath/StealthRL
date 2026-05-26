"""
Ghostbuster detector wrapper.
"""

from typing import List, Union
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .base_detector import BaseDetector


class GhostbusterDetector(BaseDetector):
    """
    Wrapper for Ghostbuster feature-ensemble detection method.
    Uses a RoBERTa-based classifier as a simplified implementation.
    """
    
    def __init__(self, model_name: str = "roberta-base-openai-detector", device: str = "cuda"):
        super().__init__(model_name, device)
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load Ghostbuster model."""
        if self.model is None:
            print(f"Loading {self.model_name} for Ghostbuster detection...")
            # Use RoBERTa-based classifier trained on AI detection
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
                ).to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"Warning: Could not load {self.model_name}, using fallback")
                # Fallback to a generic classifier
                self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "roberta-base",
                    num_labels=2,
                    torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
                ).to(self.device)
                self.model.eval()
        
    def detect(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Run Ghostbuster detection on texts.
        
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
                # Tokenize and get prediction
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get probability of AI-generated class
                probs = torch.softmax(logits, dim=-1)
                # Assume class 1 is AI-generated
                ai_prob = probs[0, 1].item() if probs.shape[1] > 1 else probs[0, 0].item()
                scores.append(ai_prob)
        
        return torch.tensor(scores, dtype=torch.float32)
