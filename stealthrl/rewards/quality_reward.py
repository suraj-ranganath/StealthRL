"""
Quality metrics reward computation with explicit normalization.

Note: Perplexity is only a partial quality signal and should be combined with
other metrics like readability, coherence, or human evaluation. See literature
on language model evaluation (e.g., GPT-2 paper, COMET metrics) for context.
"""

from typing import List, Dict
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import textstat


class QualityReward:
    """
    Computes text quality metrics including perplexity and readability.
    
    Quality score formula:
        Q = α·(1 - normalize(perplexity)) + (1-α)·normalize(readability)
    
    where normalization uses configurable min/max bounds for stability.
    """
    
    def __init__(
        self, 
        perplexity_model: str = "gpt2", 
        device: str = "cuda",
        perplexity_min: float = 5.0,
        perplexity_max: float = 80.0,
        readability_min: float = 0.0,
        readability_max: float = 100.0,
        quality_balance: float = 0.5,
    ):
        """
        Initialize quality reward with normalization bounds.
        
        Args:
            perplexity_model: Model to use for perplexity computation
            device: Device to run model on
            perplexity_min: Minimum perplexity for normalization
            perplexity_max: Maximum perplexity for normalization
            readability_min: Minimum readability score (Flesch)
            readability_max: Maximum readability score (Flesch)
            quality_balance: α in formula (weight for perplexity vs readability)
        """
        self.perplexity_model_name = perplexity_model
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # Normalization bounds
        self.perplexity_min = perplexity_min
        self.perplexity_max = perplexity_max
        self.readability_min = readability_min
        self.readability_max = readability_max
        self.quality_balance = quality_balance
        
    def _load_model(self):
        """Load perplexity model lazily."""
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.perplexity_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.perplexity_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self.device)
            self.model.eval()
        
    def _minmax_norm(self, x: torch.Tensor, x_min: float, x_max: float) -> torch.Tensor:
        """
        Min-max normalization with clamping for stability.
        
        Args:
            x: Input tensor
            x_min: Minimum value for normalization
            x_max: Maximum value for normalization
            
        Returns:
            Normalized tensor in [0, 1]
        """
        x_clamped = x.clamp(x_min, x_max)
        return (x_clamped - x_min) / (x_max - x_min + 1e-6)
    
    def compute(self, texts: List[str]) -> torch.Tensor:
        """
        Compute quality scores for texts using explicit min-max normalization.
        
        Formula:
            Q = α·(1 - normalize(perplexity)) + (1-α)·normalize(readability)
        
        Args:
            texts: Text samples to evaluate
            
        Returns:
            Tensor of quality scores (0-1, higher = better quality)
        """
        self._load_model()
        
        # Compute perplexity
        perplexities = self._compute_perplexity(texts)
        
        # Compute readability (Flesch Reading Ease)
        readability_scores = torch.tensor(
            self._compute_readability(texts), 
            dtype=torch.float32
        )
        
        # Normalize perplexity: lower is better, so flip with (1 - norm)
        norm_ppl = self._minmax_norm(perplexities, self.perplexity_min, self.perplexity_max)
        ppl_scores = 1.0 - norm_ppl  # Invert: lower perplexity → higher score
        
        # Normalize readability
        read_scores = self._minmax_norm(
            readability_scores, 
            self.readability_min, 
            self.readability_max
        )
        
        # Combine with configurable balance (α)
        quality_scores = (
            self.quality_balance * ppl_scores + 
            (1.0 - self.quality_balance) * read_scores
        )
        
        return quality_scores
    
    def _compute_perplexity(self, texts: List[str]) -> torch.Tensor:
        """Compute perplexity for each text."""
        perplexities = []
        
        with torch.no_grad():
            for text in texts:
                encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                input_ids = encodings.input_ids.to(self.device)
                
                # Get log probabilities
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss
                
                # Perplexity is exp(loss)
                ppl = torch.exp(loss)
                perplexities.append(ppl.item())
        
        return torch.tensor(perplexities, dtype=torch.float32)
    
    def _compute_readability(self, texts: List[str]) -> List[float]:
        """Compute Flesch Reading Ease for each text."""
        scores = []
        for text in texts:
            try:
                # Flesch Reading Ease: higher is better (easier to read)
                score = textstat.flesch_reading_ease(text)
                # Handle edge cases
                score = max(0.0, min(100.0, score))
            except:
                # Default to neutral score if computation fails
                score = 50.0
            scores.append(score)
        
        return scores
