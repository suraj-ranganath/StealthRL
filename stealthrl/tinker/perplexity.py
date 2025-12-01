"""
Perplexity Reward for StealthRL using Frozen Language Model.

This module computes perplexity-based quality rewards, penalizing
both too-low (LLM-like) and too-high (incoherent) perplexity.
"""

import logging
import asyncio
from typing import Dict, Any

import torch

logger = logging.getLogger(__name__)


class PerplexityReward:
    """
    Perplexity-based fluency reward.
    
    Uses a frozen language model to compute perplexity, then rewards
    perplexity values in a "human-like" range.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        ppl_min: float = 5.0,
        ppl_max: float = 80.0,
        ppl_target: float = 30.0,
        device: str | None = None,
    ):
        """
        Initialize perplexity reward.
        
        Args:
            model_name: HuggingFace model for perplexity computation
            ppl_min: Minimum perplexity (too predictable/LLM-like)
            ppl_max: Maximum perplexity (too unpredictable/incoherent)
            ppl_target: Target perplexity (human-like)
            device: Device to run model on
        """
        self.model_name = model_name
        self.ppl_min = ppl_min
        self.ppl_max = ppl_max
        self.ppl_target = ppl_target
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model (lazy load)
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initialized PerplexityReward with {model_name} on {self.device}")
        logger.info(f"Target range: [{ppl_min}, {ppl_max}], optimal: {ppl_target}")
    
    async def compute(self, text: str) -> Dict[str, Any]:
        """
        Compute perplexity reward for text.
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with perplexity and reward
        """
        # Run in thread pool
        perplexity = await asyncio.to_thread(self._compute_perplexity, text)
        
        # Compute reward based on distance from target
        reward = self._perplexity_to_reward(perplexity)
        
        return {
            "perplexity": perplexity,
            "reward": reward,
            "in_range": float(self.ppl_min <= perplexity <= self.ppl_max),
        }
    
    def _load_model(self):
        """Lazy load the model on first use."""
        if self.model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading {self.model_name} for perplexity computation...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"✓ Perplexity model loaded on {self.device}")
    
    def _compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity (synchronous).
        
        Args:
            text: Input text
        
        Returns:
            Perplexity value
        """
        # Load model if not already loaded
        self._load_model()
        
        try:
            with torch.no_grad():
                encodings = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = self.model(**encodings, labels=encodings["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            return perplexity
        
        except Exception as e:
            logger.error(f"Perplexity computation error: {e}")
            return self.ppl_target  # Return target on error
    
    def _perplexity_to_reward(self, perplexity: float) -> float:
        """
        Convert perplexity to reward using target-based scoring.
        
        Args:
            perplexity: Perplexity value
        
        Returns:
            Reward in [0, 1]
        """
        # Clip to valid range
        ppl = max(self.ppl_min, min(self.ppl_max, perplexity))
        
        # Compute distance from target
        distance = abs(ppl - self.ppl_target)
        max_distance = max(self.ppl_target - self.ppl_min, self.ppl_max - self.ppl_target)
        
        # Reward inversely proportional to distance
        reward = 1.0 - (distance / max_distance)
        
        return max(0.0, reward)
