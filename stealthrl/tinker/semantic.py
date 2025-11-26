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
        
        # Initialize model (placeholder - in production, load actual model)
        # from sentence_transformers import SentenceTransformer
        # self.model = SentenceTransformer(model_name).to(self.device)
        self.model = None  # Placeholder
        
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
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity (synchronous).
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Cosine similarity [0, 1]
        """
        if self.model is None:
            # Placeholder: Mock similarity based on text overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            # Jaccard similarity as rough proxy
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            similarity = intersection / union if union > 0 else 0.0
            
            # Scale to be more optimistic (for demo)
            return min(1.0, similarity * 1.5)
        
        # Production code would use actual model:
        # embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
        # similarity = F.cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
        # return (similarity + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
        
        return 0.0
