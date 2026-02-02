"""
Simple paraphrase baseline (M1).

Uses base LM (Qwen3-4B via Ollama) without RL training.
"""

import logging
from typing import List, Optional

import requests

from .base import BaseAttackMethod, AttackOutput

logger = logging.getLogger(__name__)


PARAPHRASE_PROMPT = """Please paraphrase the following text while maintaining its meaning and style. Output only the paraphrased text without any additional explanation.

Original text:
{text}

Paraphrased text:"""


class SimpleParaphrase(BaseAttackMethod):
    """
    M1: Simple paraphrase using base LM.
    
    Uses Qwen3-4B via Ollama without RL training.
    Can generate multiple candidates and optionally rerank by a scoring function.
    """
    
    DEFAULT_MODEL = "qwen3:4b-instruct"  # Ollama model name
    DEFAULT_OLLAMA_URL = "http://localhost:11434"
    
    def __init__(
        self,
        model_name: str = None,
        ollama_url: str = None,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        scorer_fn: callable = None,
        rerank_detector: str = "roberta",
        device: str = None,
        **kwargs,
    ):
        """
        Initialize simple paraphrase method.
        
        Args:
            model_name: Ollama model name (default: qwen3:4b)
            ollama_url: Ollama server URL (default: http://localhost:11434)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            max_new_tokens: Maximum tokens to generate
            scorer_fn: Optional function to score candidates (for reranking)
            rerank_detector: Detector for best-of-N selection (default: roberta)
            device: Device for detector (cpu/cuda)
        """
        super().__init__(name="simple_paraphrase")
        
        self.model_name = model_name or self.DEFAULT_MODEL
        self.ollama_url = ollama_url or self.DEFAULT_OLLAMA_URL
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.scorer_fn = scorer_fn
        self.rerank_detector_name = rerank_detector
        self.device = device
        self.rerank_detector = None
    
    def load(self):
        """Verify Ollama connection and load reranking detector."""
        import torch
        from ..detectors import get_detector
        
        logger.info(f"Connecting to Ollama at {self.ollama_url} for {self.model_name}...")
        
        # Check if Ollama is running and model is available
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            
            # Check if our model is available (handle tag variations)
            model_base = self.model_name.split(":")[0]
            available = any(model_base in m for m in models)
            
            if not available:
                logger.warning(f"Model {self.model_name} not found in Ollama. Available: {models}")
                logger.info(f"Attempting to pull {self.model_name}...")
                # Try to use it anyway - Ollama might pull it automatically
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to Ollama at {self.ollama_url}. Is Ollama running?")
        except Exception as e:
            logger.warning(f"Could not verify Ollama model: {e}")
        
        self._loaded = True
        logger.info(f"✓ {self.name} ready (Ollama: {self.model_name})")
        logger.info(f"Reranking detector ({self.rerank_detector_name}) will load lazily if n_candidates > 1")
    
    def _generate_single(self, text: str) -> str:
        """Generate a single paraphrase using Ollama."""
        prompt = PARAPHRASE_PROMPT.format(text=text)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_new_tokens,
            },
        }
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        
        result = response.json()
        generated = result.get("response", "").strip()
        
        return generated
    
    def _attack_impl(
        self,
        text: str,
        n_candidates: int = 1,
        **kwargs,
    ) -> AttackOutput:
        """
        Generate paraphrase(s) of the input text.
        
        Args:
            text: Original AI-generated text
            n_candidates: Number of candidates to generate
        
        Returns:
            AttackOutput with best candidate
        """
        # Generate candidates
        candidates = []
        for _ in range(n_candidates):
            try:
                para = self._generate_single(text)
                if para:  # Non-empty
                    candidates.append(para)
            except Exception as e:
                logger.warning(f"Generation failed: {e}")
        
        if not candidates:
            # Fallback to original
            return AttackOutput(
                text=text,
                metadata={"method": self.name, "error": "no_valid_candidates"},
            )
        
        # Score and select best candidate
        if n_candidates > 1:
            # Lazy load detector if not already loaded
            if self.rerank_detector is None:
                import torch
                from ..detectors import get_detector
                self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Loading reranking detector: {self.rerank_detector_name}")
                self.rerank_detector = get_detector(self.rerank_detector_name, device=self.device)
                self.rerank_detector.load()
            # Use detector for reranking
            scores = [self.rerank_detector.get_scores(c) for c in candidates]
            best_idx = scores.index(min(scores))  # Lower score = better evasion
        else:
            scores = [0.0]
            best_idx = 0
        
        return AttackOutput(
            text=candidates[best_idx],
            metadata={
                "method": self.name,
                "n_candidates": n_candidates,
                "best_idx": best_idx,
                "best_detector_score": scores[best_idx],
                "rerank_detector": self.rerank_detector_name,
            },
            all_candidates=candidates,
            candidate_scores=scores,
        )


class SimpleParaphraseWithReranking(SimpleParaphrase):
    """
    M1 variant: Simple paraphrase with detector-guided reranking.
    
    Generates N candidates and selects the one with lowest detector score.
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        rerank_detector: str = "roberta",
        **kwargs,
    ):
        super().__init__(model_name=model_name, device=device, **kwargs)
        self.name = f"simple_paraphrase_rerank_{rerank_detector}"
        self.rerank_detector_name = rerank_detector
        self.rerank_detector = None
    
    def load(self):
        """Load model and reranking detector."""
        super().load()
        
        # Import here to avoid circular imports
        from ..detectors import get_detector
        
        logger.info(f"Loading reranking detector: {self.rerank_detector_name}")
        self.rerank_detector = get_detector(self.rerank_detector_name, device=self.device)
        self.rerank_detector.load()
        
        # Set scorer function
        self.scorer_fn = lambda text: self.rerank_detector.get_scores(text)
