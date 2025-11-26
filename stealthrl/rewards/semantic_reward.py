"""
Semantic fidelity reward computation.
"""

from typing import List, Optional
import torch
import numpy as np
from bert_score import score as bertscore_compute
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity


class SemanticFidelityReward:
    """
    Computes semantic similarity between original and paraphrased text.
    Uses BERTScore or cosine similarity of embeddings.
    """
    
    def __init__(self, metric: str = "bertscore", model_type: Optional[str] = None):
        """
        Initialize semantic fidelity reward.
        
        Args:
            metric: Similarity metric to use ("bertscore" or "cosine")
            model_type: Model for BERTScore or embedding model for cosine
        """
        self.metric = metric
        self.model_type = model_type
        self.embedding_model = None
        
        if metric == "cosine":
            # Initialize sentence transformer for embeddings
            model_name = model_type or "all-MiniLM-L6-v2"
            self.embedding_model = SentenceTransformer(model_name)
        
    def compute(self, original_texts: List[str], paraphrased_texts: List[str]) -> torch.Tensor:
        """
        Compute semantic similarity scores.
        
        Args:
            original_texts: Original text samples
            paraphrased_texts: Paraphrased versions
            
        Returns:
            Tensor of similarity scores (0-1, higher = better preservation)
        """
        if self.metric == "bertscore":
            return self._compute_bertscore(original_texts, paraphrased_texts)
        elif self.metric == "cosine":
            return self._compute_cosine(original_texts, paraphrased_texts)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _compute_bertscore(self, original_texts: List[str], paraphrased_texts: List[str]) -> torch.Tensor:
        """Compute BERTScore between texts."""
        model_type = self.model_type or "microsoft/deberta-xlarge-mnli"
        
        # BERTScore returns (P, R, F1)
        P, R, F1 = bertscore_compute(
            paraphrased_texts,
            original_texts,
            model_type=model_type,
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Return F1 scores as the primary metric
        return F1
    
    def _compute_cosine(self, original_texts: List[str], paraphrased_texts: List[str]) -> torch.Tensor:
        """Compute cosine similarity between text embeddings."""
        # Encode texts
        original_embeddings = self.embedding_model.encode(
            original_texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        paraphrased_embeddings = self.embedding_model.encode(
            paraphrased_texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Compute cosine similarity
        similarities = cosine_similarity(
            original_embeddings,
            paraphrased_embeddings,
            dim=1
        )
        
        # Normalize to 0-1 range (cosine is -1 to 1)
        similarities = (similarities + 1.0) / 2.0
        
        return similarities
