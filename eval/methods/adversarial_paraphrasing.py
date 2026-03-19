"""
Adversarial Paraphrasing baseline (M3).

Based on the NeurIPS 2025 paper: https://arxiv.org/abs/2506.07001
Implements detector-guided selection using batched vLLM generation.
"""

import logging
from typing import List, Optional

from .base import (
    BaseAttackMethod,
    AttackOutput,
    estimate_generation_max_tokens,
    iter_length_bucket_indices,
)
from .simple_paraphrase import PARAPHRASE_PROMPT
from .vllm_backend import get_vllm_generator

logger = logging.getLogger(__name__)


class AdversarialParaphrasing(BaseAttackMethod):
    """
    M3: Adversarial Paraphrasing baseline.
    
    Detector-guided selection:
    1. Sample K paraphrases from base LM (via vLLM)
    2. Select candidate minimizing AI score from guidance detector
    3. Enforce similarity threshold to avoid semantic drift
    
    Paper: https://arxiv.org/abs/2506.07001
    """
    
    DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
    GENERATION_BUCKET_SIZE = 96
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,  # kept for detector/similarity scorer
        guidance_detector: str = "roberta",
        similarity_threshold: float = 0.90,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
    ):
        """
        Initialize Adversarial Paraphrasing.
        
        Args:
            model_name: Base model name (default: Qwen/Qwen3-4B-Instruct-2507)
            device: Device for detector/similarity scorer
            guidance_detector: Detector to guide selection (default: roberta)
            similarity_threshold: Minimum similarity to accept candidate
            temperature: Sampling temperature (higher for diversity)
            top_p: Nucleus sampling threshold
            max_new_tokens: Maximum tokens to generate
        """
        super().__init__(name="adversarial_paraphrasing")
        
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.guidance_detector_name = guidance_detector
        self.similarity_threshold = similarity_threshold
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        
        self.guidance_detector = None
        self.similarity_scorer = None
        self.generator = None

    def _estimate_max_tokens(self, text: str) -> int:
        return estimate_generation_max_tokens(text, self.max_new_tokens)
    
    def load(self):
        """Load detector, similarity scorer, and shared vLLM generator."""
        import torch
        from ..detectors import get_detector
        from ..metrics import E5SimilarityScorer
        
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = get_vllm_generator(
            model_name=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
        )
        self.generator.load()
        
        # Load guidance detector
        logger.info(f"Loading guidance detector: {self.guidance_detector_name}")
        self.guidance_detector = get_detector(self.guidance_detector_name, device=self.device)
        self.guidance_detector.load()
        
        # Load similarity scorer (uses HF fallback if Ollama is unavailable)
        logger.info("Loading similarity scorer...")
        self.similarity_scorer = E5SimilarityScorer(device=self.device)
        self.similarity_scorer.load()
        
        self._loaded = True
        logger.info(f"✓ Adversarial Paraphrasing ready (vLLM: {self.model_name}, detector: {self.guidance_detector_name})")
    
    def _generate_candidates(self, text: str, n_candidates: int) -> List[str]:
        """Generate candidate paraphrases with vLLM."""
        prompt = PARAPHRASE_PROMPT.format(text=text)
        return self.generator.generate_batch(
            [prompt],
            n=n_candidates,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self._estimate_max_tokens(text),
        )[0]
    
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
        
        try:
            candidates = self._generate_candidates(text, n_candidates)
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            candidates = []
        
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
                "backend": "vllm",
                "model": self.model_name,
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

    def attack_batch(
        self,
        texts: List[str],
        n_candidates: int = 4,
        **kwargs,
    ) -> List[AttackOutput]:
        if not self._loaded:
            self.load()

        results: List[Optional[AttackOutput]] = [None] * len(texts)

        completed = 0
        total = len(texts)
        for batch in iter_length_bucket_indices(texts, self.GENERATION_BUCKET_SIZE):
            batch_indices = [index for index, _ in batch]
            batch_texts = [text for _, text in batch]
            prompts = [PARAPHRASE_PROMPT.format(text=text) for text in batch_texts]
            batch_max_tokens = max(self._estimate_max_tokens(text) for text in batch_texts)

            grouped_candidates = self.generator.generate_batch(
                prompts,
                n=n_candidates,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=batch_max_tokens,
            )

            flat_candidates: List[str] = []
            flat_originals: List[str] = []
            group_sizes: List[int] = []
            for original_text, candidates in zip(batch_texts, grouped_candidates):
                filtered = [candidate for candidate in candidates if candidate]
                flat_candidates.extend(filtered)
                flat_originals.extend([original_text] * len(filtered))
                group_sizes.append(len(filtered))

            detector_scores = self.guidance_detector.get_scores(flat_candidates) if flat_candidates else []
            similarities = (
                self.similarity_scorer.compute_similarity(flat_originals, flat_candidates)
                if flat_candidates else []
            )

            cursor = 0
            for index, text, size in zip(batch_indices, batch_texts, group_sizes):
                candidates = flat_candidates[cursor:cursor + size]
                det_scores = detector_scores[cursor:cursor + size] if detector_scores else []
                sims = similarities[cursor:cursor + size] if similarities else []
                cursor += size

                if not candidates:
                    results[index] = AttackOutput(
                        text=text,
                        original_text=text,
                        valid=False,
                        fail_reason="no_valid_candidates",
                        metadata={"method": self.name, "backend": "vllm"},
                    )
                    continue

                best_candidate = None
                best_score = float("inf")
                best_idx = 0
                for idx, (candidate, det_score, sim) in enumerate(zip(candidates, det_scores, sims)):
                    if sim >= self.similarity_threshold and det_score < best_score:
                        best_candidate = candidate
                        best_score = det_score
                        best_idx = idx

                if best_candidate is None:
                    best_idx = det_scores.index(min(det_scores)) if det_scores else 0
                    best_candidate = candidates[best_idx]

                results[index] = AttackOutput(
                    text=best_candidate,
                    original_text=text,
                    metadata={
                        "method": self.name,
                        "backend": "vllm",
                        "model": self.model_name,
                        "guidance_detector": self.guidance_detector_name,
                        "similarity_threshold": self.similarity_threshold,
                        "n_candidates": n_candidates,
                        "best_idx": best_idx,
                        "best_detector_score": det_scores[best_idx] if det_scores else 0.0,
                        "best_similarity": sims[best_idx] if sims else 0.0,
                    },
                    all_candidates=candidates,
                    candidate_scores=det_scores,
                )

            completed += len(batch_texts)
            logger.info("[%s] Progress: %d/%d", self.name, completed, total)

        return [result for result in results if result is not None]


class AdversarialParaphrasingEnsemble(AdversarialParaphrasing):
    """
    Adversarial Paraphrasing with ensemble guidance.
    
    Uses mean score from multiple detectors for selection.
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        guidance_detectors: List[str] = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, device=device, **kwargs)
        self.name = "adversarial_paraphrasing_ensemble"
        self.guidance_detector_names = guidance_detectors or ["roberta", "fast_detectgpt"]
        self.guidance_detectors = []
    
    def load(self):
        """Load multiple guidance detectors and shared vLLM generator."""
        import torch
        from ..detectors import get_detector
        from ..metrics import E5SimilarityScorer
        
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = get_vllm_generator(
            model_name=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
        )
        self.generator.load()
        
        # Load multiple guidance detectors
        for det_name in self.guidance_detector_names:
            logger.info(f"Loading guidance detector: {det_name}")
            det = get_detector(det_name, device=self.device)
            det.load()
            self.guidance_detectors.append(det)
        
        # Load similarity scorer (uses HF fallback if Ollama is unavailable)
        self.similarity_scorer = E5SimilarityScorer(device=self.device)
        self.similarity_scorer.load()
        
        self._loaded = True
        logger.info(f"✓ Adversarial Paraphrasing Ensemble ready (vLLM: {self.model_name})")
    
    def _attack_impl(
        self,
        text: str,
        n_candidates: int = 4,
        **kwargs,
    ) -> AttackOutput:
        """Generate adversarial paraphrase using ensemble guidance."""
        
        try:
            candidates = self._generate_candidates(text, n_candidates)
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            candidates = []
        
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
                "backend": "vllm",
                "model": self.model_name,
                "guidance_detectors": self.guidance_detector_names,
                "n_candidates": n_candidates,
                "best_idx": best_idx,
                "best_ensemble_score": ensemble_scores[best_idx],
                "best_similarity": similarities[best_idx],
            },
            all_candidates=candidates,
            candidate_scores=ensemble_scores,
        )
