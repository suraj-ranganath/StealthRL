"""
Simple paraphrase baseline (M1).

Uses base LM (Qwen3-4B via Ollama) without RL training.
"""

import logging
from typing import List, Optional

import requests

from .base import BaseAttackMethod, AttackOutput, validate_attack_output

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


class SimpleParaphraseTinker(BaseAttackMethod):
    """
    M1: Simple paraphrase using Tinker cloud inference.

    Uses the base model via Tinker (no RL training) to avoid local compute.
    Supports concurrent sampling for high throughput.
    """

    DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

    def __init__(
        self,
        base_model: str = None,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        rerank_detector: str = "roberta",
        device: str = None,
        tinker_concurrency: int = 64,
        tinker_chunk_size: int = 256,
        tinker_max_retries: int = 2,
        tinker_backoff_s: float = 0.5,
        tinker_resume_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name="simple_paraphrase_tinker")

        self.base_model = base_model or self.DEFAULT_BASE_MODEL
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.rerank_detector_name = rerank_detector
        self.device = device
        self.tinker_concurrency = tinker_concurrency
        self.tinker_chunk_size = tinker_chunk_size
        self.tinker_max_retries = tinker_max_retries
        self.tinker_backoff_s = tinker_backoff_s
        self.tinker_resume_path = tinker_resume_path

        self.sampling_client = None
        self.tokenizer = None
        self.rerank_detector = None

    def load(self):
        """Initialize Tinker sampling client and tokenizer."""
        import os

        api_key = os.environ.get("TINKER_API_KEY")
        if not api_key:
            raise RuntimeError("TINKER_API_KEY environment variable not set")

        try:
            import tinker
            from tinker import ServiceClient
        except ImportError:
            raise RuntimeError("Tinker not installed. Install with: pip install tinker")

        logger.info(f"Connecting to Tinker for base model {self.base_model}...")
        service_client = ServiceClient()
        self.sampling_client = service_client.create_sampling_client(base_model=self.base_model)
        self.tokenizer = self.sampling_client.get_tokenizer()

        logger.info("✓ SimpleParaphrase (Tinker) ready - supports batched num_samples")
        logger.info(f"Reranking detector ({self.rerank_detector_name}) will load lazily if n_candidates > 1")
        self._loaded = True

    def _generate_candidates(self, text: str, n_candidates: int) -> List[str]:
        """Generate candidates with a single Tinker API call."""
        import time
        from tinker import types

        prompt_text = PARAPHRASE_PROMPT.format(text=text)
        input_words = len(text.split())

        messages = [{"role": "user", "content": prompt_text}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted = prompt_text

        input_ids = self.tokenizer.encode(formatted)
        model_input = types.ModelInput.from_ints(input_ids)

        params = types.SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        api_start = time.time()
        logger.info(f"[TINKER-M1] Requesting {n_candidates} candidates for {input_words}-word text...")

        try:
            future = self.sampling_client.sample(
                prompt=model_input,
                sampling_params=params,
                num_samples=n_candidates,
            )
            result = future.result()
            api_elapsed = time.time() - api_start

            candidates = []
            for i, sample in enumerate(result.sequences):
                output_text = self.tokenizer.decode(sample.tokens, skip_special_tokens=True).strip()
                if output_text:
                    candidates.append(output_text)
                    logger.debug(f"[TINKER-M1] Candidate {i+1}: {len(output_text.split())} words")

            logger.info(f"[TINKER-M1] Got {len(candidates)}/{n_candidates} candidates in {api_elapsed:.1f}s")
            return candidates
        except Exception as e:
            api_elapsed = time.time() - api_start
            logger.warning(f"[TINKER-M1] Request failed after {api_elapsed:.1f}s: {e}")
            return []

    def _attack_impl(
        self,
        text: str,
        n_candidates: int = 1,
        **kwargs,
    ) -> AttackOutput:
        if not self._loaded:
            self.load()

        candidates = self._generate_candidates(text, n_candidates)

        if not candidates:
            return AttackOutput(
                text=text,
                metadata={"method": self.name, "backend": "tinker", "error": "no_valid_candidates"},
            )

        if n_candidates > 1:
            if self.rerank_detector is None:
                import torch
                from ..detectors import get_detector
                self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Loading reranking detector: {self.rerank_detector_name}")
                self.rerank_detector = get_detector(self.rerank_detector_name, device=self.device)
                self.rerank_detector.load()
            scores = self.rerank_detector.get_scores(candidates)
            if isinstance(scores, float):
                scores = [scores]
            best_idx = scores.index(min(scores))
        else:
            scores = [0.0]
            best_idx = 0

        return AttackOutput(
            text=candidates[best_idx],
            metadata={
                "method": self.name,
                "backend": "tinker",
                "base_model": self.base_model,
                "n_candidates": n_candidates,
                "best_idx": best_idx,
                "best_detector_score": scores[best_idx],
                "rerank_detector": self.rerank_detector_name,
            },
            all_candidates=candidates,
            candidate_scores=scores,
            original_text=text,
        )

    def attack_batch(
        self,
        texts: List[str],
        n_candidates: int = 1,
        **kwargs,
    ) -> List[AttackOutput]:
        """Concurrent batch attack using Tinker native async sampling."""
        if not self._loaded:
            self.load()

        if self.tinker_concurrency <= 1:
            return super().attack_batch(texts, n_candidates=n_candidates, **kwargs)

        from tinker import types
        from ..detectors import get_detector
        from ..tinker_concurrency import run_sampling_concurrent

        prompt_template = PARAPHRASE_PROMPT

        def build_model_input(text: str):
            prompt_text = prompt_template.format(text=text)
            messages = [{"role": "user", "content": prompt_text}]
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                formatted = prompt_text
            input_ids = self.tokenizer.encode(formatted)
            return types.ModelInput.from_ints(input_ids)

        params = types.SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        params_dict = {
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_samples": n_candidates,
        }

        results = run_sampling_concurrent(
            texts=texts,
            build_model_input=build_model_input,
            sampling_client=self.sampling_client,
            sampling_params=params,
            num_samples=n_candidates,
            tokenizer=self.tokenizer,
            concurrency=self.tinker_concurrency,
            chunk_size=self.tinker_chunk_size,
            max_retries=self.tinker_max_retries,
            backoff_base_s=self.tinker_backoff_s,
            resume_cache_path=self.tinker_resume_path,
            sampling_params_dict=params_dict,
        )

        if n_candidates > 1 and self.rerank_detector is None:
            import torch
            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading reranking detector: {self.rerank_detector_name}")
            self.rerank_detector = get_detector(self.rerank_detector_name, device=self.device)
            self.rerank_detector.load()

        outputs: List[AttackOutput] = []
        for text, result in zip(texts, results):
            if result.error or not result.candidates:
                outputs.append(AttackOutput(
                    text=text,
                    original_text=text,
                    valid=False,
                    fail_reason=f"tinker_error: {result.error or 'no_candidates'}",
                    metadata={
                        "method": self.name,
                        "backend": "tinker",
                        "base_model": self.base_model,
                        "n_candidates": n_candidates,
                        "error": result.error,
                        "attempts": result.attempts,
                        "latency_s": result.latency_s,
                        "sampling_params_hash": result.sampling_params_hash,
                        "generation_version": "tinker-concurrent-v1",
                    },
                ))
                continue

            candidates = result.candidates
            if n_candidates > 1:
                scores = self.rerank_detector.get_scores(candidates)
                if isinstance(scores, float):
                    scores = [scores]
                best_idx = scores.index(min(scores))
            else:
                scores = [0.0]
                best_idx = 0

            best_text = candidates[best_idx]
            valid, fail_reason = validate_attack_output(
                text,
                best_text,
                min_words=self.min_words,
                max_length_ratio=self.max_length_ratio,
            )

            outputs.append(AttackOutput(
                text=best_text,
                metadata={
                    "method": self.name,
                    "backend": "tinker",
                    "base_model": self.base_model,
                    "n_candidates": n_candidates,
                    "best_idx": best_idx,
                    "best_detector_score": scores[best_idx],
                    "rerank_detector": self.rerank_detector_name,
                    "attempts": result.attempts,
                    "latency_s": result.latency_s,
                    "sampling_params_hash": result.sampling_params_hash,
                    "generation_version": "tinker-concurrent-v1",
                },
                all_candidates=candidates,
                candidate_scores=scores,
                original_text=text,
                valid=valid,
                fail_reason=fail_reason,
            ))

        return outputs
