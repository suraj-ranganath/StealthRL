"""
Semantic Similarity for StealthRL using E5 Encoders.

This module computes semantic similarity between original and paraphrased
text using sentence transformers (E5 model).
"""

import logging
import asyncio
from typing import Dict, Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
    
    def _load_model(self):
        """Lazy load the model on first use."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading {self.model_name} for semantic similarity...")
            self.model = SentenceTransformer(self.model_name).to(self.device)
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
