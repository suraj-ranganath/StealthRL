"""
Adversarial Paraphrasing baseline (M3).

Based on the NeurIPS 2025 paper: https://arxiv.org/abs/2506.07001
Implements detector-guided selection as a fallback approach.
"""

import logging
from typing import List, Optional

import requests

from .base import BaseAttackMethod, AttackOutput
from .simple_paraphrase import PARAPHRASE_PROMPT

logger = logging.getLogger(__name__)


class AdversarialParaphrasing(BaseAttackMethod):
    """
    M3: Adversarial Paraphrasing baseline.
    
    Fallback implementation using detector-guided selection:
    1. Sample K paraphrases from base LM (via Ollama)
    2. Select candidate minimizing AI score from guidance detector
    3. Enforce similarity threshold to avoid semantic drift
    
    Paper: https://arxiv.org/abs/2506.07001
    """
    
    DEFAULT_MODEL = "qwen3:4b-instruct"  # Ollama model (same as M1)
    DEFAULT_OLLAMA_URL = "http://localhost:11434"
    
    def __init__(
        self,
        model_name: str = None,
        ollama_url: str = None,
        device: str = None,  # kept for detector/similarity scorer
        guidance_detector: str = "roberta",
        similarity_threshold: float = 0.90,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
    ):
        """
        Initialize Adversarial Paraphrasing.
        
        Args:
            model_name: Ollama model name (default: qwen3:4b-instruct)
            ollama_url: Ollama server URL
            device: Device for detector/similarity scorer
            guidance_detector: Detector to guide selection (default: roberta)
            similarity_threshold: Minimum similarity to accept candidate
            temperature: Sampling temperature (higher for diversity)
            top_p: Nucleus sampling threshold
            max_new_tokens: Maximum tokens to generate
        """
        super().__init__(name="adversarial_paraphrasing")
        
        self.model_name = model_name or self.DEFAULT_MODEL
        self.ollama_url = ollama_url or self.DEFAULT_OLLAMA_URL
        self.device = device
        self.guidance_detector_name = guidance_detector
        self.similarity_threshold = similarity_threshold
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        
        self.guidance_detector = None
        self.similarity_scorer = None
    
    def load(self):
        """Load detector and similarity scorer (LLM runs via Ollama)."""
        import torch
        from ..detectors import get_detector
        from ..metrics import E5SimilarityScorer
        
        # Verify Ollama connection
        logger.info(f"Connecting to Ollama at {self.ollama_url} for {self.model_name}...")
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            model_base = self.model_name.split(":")[0]
            if not any(model_base in m for m in models):
                logger.warning(f"Model {self.model_name} not found. Available: {models}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to Ollama at {self.ollama_url}. Is Ollama running?")
        except Exception as e:
            logger.warning(f"Could not verify Ollama model: {e}")
        
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load guidance detector
        logger.info(f"Loading guidance detector: {self.guidance_detector_name}")
        self.guidance_detector = get_detector(self.guidance_detector_name, device=self.device)
        self.guidance_detector.load()
        
        # Load similarity scorer (uses BGE-M3 via Ollama by default)
        logger.info("Loading similarity scorer (BGE-M3 via Ollama)...")
        self.similarity_scorer = E5SimilarityScorer(device=self.device)
        self.similarity_scorer.load()
        
        self._loaded = True
        logger.info(f"✓ Adversarial Paraphrasing ready (Ollama: {self.model_name}, detector: {self.guidance_detector_name})")
    
    def _generate_single(self, text: str) -> str:
        """Generate a single paraphrase candidate via Ollama."""
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
        n_candidates: int = 4,
        **kwargs,
    ) -> AttackOutput:
        """
        Generate adversarial paraphrase using detector-guided selection.
        
        Args:
            text: Original AI-generated text
            n_candidates: Number of candidates to generate (K)
        
        Returns:
            AttackOutput with best candidate that passes similarity check
        """
        
        # Generate K candidates
        candidates = []
        for _ in range(n_candidates):
            try:
                para = self._generate_single(text)
                if para:
                    candidates.append(para)
            except Exception as e:
                logger.warning(f"Generation failed: {e}")
        
        if not candidates:
            return AttackOutput(
                text=text,
                metadata={"method": self.name, "error": "no_valid_candidates"},
            )
        
        # Score candidates with guidance detector
        detector_scores = []
        for c in candidates:
            score = self.guidance_detector.get_scores(c)
            detector_scores.append(score)
        
        # Compute similarity scores
        similarities = self.similarity_scorer.compute_similarity(
            [text] * len(candidates),
            candidates,
        )
        
        # Select best candidate that passes similarity threshold
        best_candidate = None
        best_score = float('inf')
        best_idx = 0
        
        for i, (c, det_score, sim) in enumerate(zip(candidates, detector_scores, similarities)):
            # Must pass similarity threshold
            if sim >= self.similarity_threshold:
                if det_score < best_score:
                    best_candidate = c
                    best_score = det_score
                    best_idx = i
        
        # Fallback: if no candidate passes threshold, take lowest detector score
        if best_candidate is None:
            best_idx = detector_scores.index(min(detector_scores))
            best_candidate = candidates[best_idx]
            logger.warning(f"No candidate passed similarity threshold {self.similarity_threshold}")
        
        return AttackOutput(
            text=best_candidate,
            metadata={
                "method": self.name,
                "guidance_detector": self.guidance_detector_name,
                "similarity_threshold": self.similarity_threshold,
                "n_candidates": n_candidates,
                "best_idx": best_idx,
                "best_detector_score": detector_scores[best_idx],
                "best_similarity": similarities[best_idx],
            },
            all_candidates=candidates,
            candidate_scores=detector_scores,
        )


class AdversarialParaphrasingEnsemble(AdversarialParaphrasing):
    """
    Adversarial Paraphrasing with ensemble guidance.
    
    Uses mean score from multiple detectors for selection.
    """
    
    def __init__(
        self,
        model_name: str = None,
        ollama_url: str = None,
        device: str = None,
        guidance_detectors: List[str] = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, ollama_url=ollama_url, device=device, **kwargs)
        self.name = "adversarial_paraphrasing_ensemble"
        self.guidance_detector_names = guidance_detectors or ["roberta", "fast_detectgpt"]
        self.guidance_detectors = []
    
    def load(self):
        """Load multiple guidance detectors (LLM runs via Ollama)."""
        import torch
        from ..detectors import get_detector
        from ..metrics import E5SimilarityScorer
        
        # Verify Ollama connection
        logger.info(f"Connecting to Ollama at {self.ollama_url} for {self.model_name}...")
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to Ollama at {self.ollama_url}. Is Ollama running?")
        
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load multiple guidance detectors
        for det_name in self.guidance_detector_names:
            logger.info(f"Loading guidance detector: {det_name}")
            det = get_detector(det_name, device=self.device)
            det.load()
            self.guidance_detectors.append(det)
        
        # Load similarity scorer (uses BGE-M3 via Ollama by default)
        self.similarity_scorer = E5SimilarityScorer(device=self.device)
        self.similarity_scorer.load()
        
        self._loaded = True
        logger.info(f"✓ Adversarial Paraphrasing Ensemble ready (Ollama: {self.model_name})")
    
    def _attack_impl(
        self,
        text: str,
        n_candidates: int = 4,
        **kwargs,
    ) -> AttackOutput:
        """Generate adversarial paraphrase using ensemble guidance."""
        
        # Generate candidates
        candidates = []
        for _ in range(n_candidates):
            try:
                para = self._generate_single(text)
                if para:
                    candidates.append(para)
            except Exception as e:
                logger.warning(f"Generation failed: {e}")
        
        if not candidates:
            return AttackOutput(
                text=text,
                metadata={"method": self.name, "error": "no_valid_candidates"},
            )
        
        # Score candidates with ensemble (mean of all detectors)
        ensemble_scores = []
        for c in candidates:
            det_scores = [det.get_scores(c) for det in self.guidance_detectors]
            ensemble_scores.append(sum(det_scores) / len(det_scores))
        
        # Compute similarity
        similarities = self.similarity_scorer.compute_similarity(
            [text] * len(candidates),
            candidates,
        )
        
        # Select best
        best_candidate = None
        best_score = float('inf')
        best_idx = 0
        
        for i, (c, ens_score, sim) in enumerate(zip(candidates, ensemble_scores, similarities)):
            if sim >= self.similarity_threshold and ens_score < best_score:
                best_candidate = c
                best_score = ens_score
                best_idx = i
        
        if best_candidate is None:
            best_idx = ensemble_scores.index(min(ensemble_scores))
            best_candidate = candidates[best_idx]
        
        return AttackOutput(
            text=best_candidate,
            metadata={
                "method": self.name,
                "guidance_detectors": self.guidance_detector_names,
                "n_candidates": n_candidates,
                "best_idx": best_idx,
                "best_ensemble_score": ensemble_scores[best_idx],
                "best_similarity": similarities[best_idx],
            },
            all_candidates=candidates,
            candidate_scores=ensemble_scores,
        )
