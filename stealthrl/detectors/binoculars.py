"""
Binoculars detector wrapper.
"""

from typing import List, Union
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_detector import BaseDetector


class BinocularsDetector(BaseDetector):
    """
    Wrapper for Binoculars paired-LM detection method.
    Uses cross-entropy between two LMs to detect AI-generated text.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        observer_model: str = "gpt2-large",
        device: str = "cuda"
    ):
        super().__init__(model_name, device)
        self.observer_model_name = observer_model
        self.model = None
        self.observer = None
        self.tokenizer = None
        
    def load_model(self):
        """Load Binoculars models (performer and observer)."""
        if self.model is None:
            print(f"Loading Binoculars models: {self.model_name} and {self.observer_model_name}...")
            
            # Load performer model (smaller)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
            ).to(self.device)
            self.model.eval()
            
            # Load observer model (larger)
            self.observer = AutoModelForCausalLM.from_pretrained(
                self.observer_model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
            ).to(self.device)
            self.observer.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def detect(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Run Binoculars detection on texts.
        
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
                # Get perplexities from both models
                ppl_performer = self._compute_perplexity(text, self.model)
                ppl_observer = self._compute_perplexity(text, self.observer)
                
                # Binoculars score: cross-entropy difference
                # Higher difference suggests human-written (more surprising to observer)
                # Lower difference suggests AI-generated (similar to both models)
                ce_diff = abs(np.log(ppl_observer) - np.log(ppl_performer))
                
                # Normalize: lower CE difference = higher AI probability
                score = torch.sigmoid(torch.tensor(-ce_diff))
                scores.append(score.item())
        
        return torch.tensor(scores, dtype=torch.float32)
    
    def _compute_perplexity(self, text: str, model) -> float:
        """Compute perplexity of text using given model."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
        return torch.exp(loss).item()
