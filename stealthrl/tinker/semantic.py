"""
Semantic Similarity for StealthRL using E5 Encoders.

This module computes semantic similarity between original and paraphrased
text using sentence transformers (E5 model).
"""

import logging
import asyncio
import threading
from typing import Dict, Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Global model cache with thread-safe access (same pattern as detectors)
_SEMANTIC_MODEL_CACHE: Dict[str, Any] = {}
_SEMANTIC_CACHE_LOCK = threading.Lock()


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_semantic_model_cached(model_name: str, device: torch.device):
    """
    Thread-safe singleton model loading for semantic similarity.
    
    Uses double-checked locking pattern to ensure only one model is loaded
    even with concurrent access from multiple threads.
    
    Args:
        model_name: HuggingFace model name
        device: Target device
    
    Returns:
        Loaded SentenceTransformer model
    """
    cache_key = f"{model_name}_{device}"
    
    # First check without lock (fast path)
    if cache_key in _SEMANTIC_MODEL_CACHE:
        logger.debug(f"✓ Using cached {model_name} on {device}")
        return _SEMANTIC_MODEL_CACHE[cache_key]
    
    # Acquire lock for loading
    with _SEMANTIC_CACHE_LOCK:
        # Double-check: another thread might have loaded it while we waited
        if cache_key in _SEMANTIC_MODEL_CACHE:
            logger.debug(f"✓ Using cached {model_name} on {device} (loaded by another thread)")
            return _SEMANTIC_MODEL_CACHE[cache_key]
        
        # Load model (only one thread gets here)
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"🔄 Loading {model_name} for semantic similarity (first time, will be cached)...")
        
        # Load without specifying device first (avoids meta tensor issues)
        model = SentenceTransformer(model_name)
        
        # Then move to target device explicitly
        model = model.to(device)
        
        # Cache it
        _SEMANTIC_MODEL_CACHE[cache_key] = model
        
        logger.info(f"✓ {model_name} loaded and cached on {device}")
        return model


class SemanticSimilarity:
    """
    Semantic similarity using E5 sentence encoder.
    
    Computes cosine similarity between sentence embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "intfloat/e5-large-v2",
        threshold: float = 0.90,
        device: str | None = None,
    ):
        """
        Initialize semantic similarity model.
        
        Args:
            model_name: HuggingFace model name for sentence encoder
            threshold: Minimum acceptable similarity
            device: Device to run model on (cuda/cpu)
        """
        self.model_name = model_name
        self.threshold = threshold
        
        # Set device
        self.device = torch.device(device) if device else _default_device()
        
        # Initialize model
        self.model = None  # Lazy load on first use
        
        logger.info(f"Initialized SemanticSimilarity with {model_name} on {self.device}")
    
    async def compute(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compute semantic similarity between two texts.
        
        Args:
            text1: First text (original)
            text2: Second text (paraphrase)
        
        Returns:
            Dictionary with similarity score and metrics
        """
        # Run in thread pool to avoid blocking
        similarity = await asyncio.to_thread(self._compute_similarity, text1, text2)
        
        return {
            "similarity": similarity,
            "above_threshold": float(similarity >= self.threshold),
        }

    async def compute_batch(self, texts1: list[str], texts2: list[str]) -> Dict[str, Any]:
        """Compute semantic similarity for a batch of text pairs."""
        similarities = await asyncio.to_thread(self._compute_similarity_batch, texts1, texts2)
        return {"similarities": similarities}
    
    def _load_model(self):
        """Lazy load the model on first use using thread-safe cache."""
        if self.model is None:
            logger.info(f"Loading {self.model_name} for semantic similarity...")
            # Use cached singleton loader (thread-safe)
            self.model = load_semantic_model_cached(self.model_name, self.device)
            logger.info(f"✓ Semantic similarity model loaded on {self.device}")
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity (synchronous).
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Cosine similarity [0, 1]
        """
        # Load model if not already loaded
        self._load_model()
        
        try:
            # Encode texts
            embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
            
            # Map [-1, 1] to [0, 1]
            return (similarity + 1.0) / 2.0
        
        except Exception as e:
            logger.error(f"Semantic similarity error: {e}")
            return 0.5  # Return neutral score on error

    def _compute_similarity_batch(self, texts1: list[str], texts2: list[str]) -> list[float]:
        """Compute cosine similarity for a batch of text pairs."""
        self._load_model()
        if len(texts1) != len(texts2):
            raise ValueError("texts1 and texts2 must have the same length")

        try:
            embeddings1 = self.model.encode(texts1, convert_to_tensor=True)
            embeddings2 = self.model.encode(texts2, convert_to_tensor=True)
            similarities = F.cosine_similarity(embeddings1, embeddings2).tolist()
            return [float((sim + 1.0) / 2.0) for sim in similarities]
        except Exception as e:
            logger.error(f"Semantic similarity batch error: {e}")
            return [0.5 for _ in texts1]
