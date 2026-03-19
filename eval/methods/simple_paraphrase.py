"""
Simple paraphrase baseline (M1).

Uses base LM (Qwen3-4B via vLLM) without RL training.
"""

import logging
from typing import List, Optional

from .base import (
    BaseAttackMethod,
    AttackOutput,
    estimate_generation_max_tokens,
    iter_length_bucket_indices,
    validate_attack_output,
)
from .vllm_backend import get_vllm_generator

logger = logging.getLogger(__name__)


PARAPHRASE_PROMPT = """Please paraphrase the following text while maintaining its meaning and style. Keep the paraphrase close to the original length, do not add new details, and output only the paraphrased text without any additional explanation.

Original text:
{text}

Paraphrased text:"""

STRICT_LENGTH_PARAPHRASE_PROMPT = """Please paraphrase the following text while preserving its meaning and style. Keep the paraphrase very close to the original length, do not add new details, and return only the paraphrased text. Do not exceed the original length by more than about 20%.

Original text:
{text}

Paraphrased text:"""


class SimpleParaphrase(BaseAttackMethod):
    """
    M1: Simple paraphrase using base LM.
    
    Uses Qwen3-4B via vLLM without RL training.
    Can generate multiple candidates and optionally rerank by a scoring function.
    """
    
    DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
    GENERATION_BUCKET_SIZE = 128
    
    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        scorer_fn: callable = None,
        rerank_detector: str = "roberta",
        device: str = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        **kwargs,
    ):
        """
        Initialize simple paraphrase method.
        
        Args:
            model_name: Base model name (default: Qwen/Qwen3-4B-Instruct-2507)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            max_new_tokens: Maximum tokens to generate
            scorer_fn: Optional function to score candidates (for reranking)
            rerank_detector: Detector for best-of-N selection (default: roberta)
            device: Device for detector (cpu/cuda)
        """
        super().__init__(name="simple_paraphrase")
        
        self.model_name = model_name or self.DEFAULT_MODEL
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.scorer_fn = scorer_fn
        self.rerank_detector_name = rerank_detector
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.rerank_detector = None
        self.generator = None

    def _estimate_max_tokens(self, text: str) -> int:
        return estimate_generation_max_tokens(text, self.max_new_tokens)
    
    def load(self):
        """Load shared vLLM generator."""
        self.generator = get_vllm_generator(
            model_name=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
        )
        self.generator.load()
        self._loaded = True
        logger.info(f"✓ {self.name} ready (vLLM: {self.model_name})")
        logger.info(f"Reranking detector ({self.rerank_detector_name}) will load lazily if n_candidates > 1")
    
    def _generate_candidates(self, text: str, n_candidates: int) -> List[str]:
        """Generate paraphrase candidates with vLLM."""
        prompt = PARAPHRASE_PROMPT.format(text=text)
        return self.generator.generate_batch(
            [prompt],
            n=n_candidates,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self._estimate_max_tokens(text),
        )[0]

    def _repair_invalid_output(
        self,
        text: str,
        fail_reason: Optional[str],
    ) -> Optional[AttackOutput]:
        if not fail_reason:
            return None

        prompt = STRICT_LENGTH_PARAPHRASE_PROMPT.format(text=text)
        retry_max_tokens = max(
            48,
            min(
                self._estimate_max_tokens(text),
                int(len(text.split()) * 1.2),
            ),
        )

        try:
            candidates = self.generator.generate_batch(
                [prompt],
                n=1,
                temperature=min(self.temperature, 0.7),
                top_p=min(self.top_p, 0.9),
                max_tokens=retry_max_tokens,
            )[0]
        except Exception as e:
            logger.warning("Length-repair generation failed: %s", e)
            return None

        if not candidates:
            return None

        candidate = candidates[0]
        valid, repaired_fail_reason = validate_attack_output(
            text,
            candidate,
            min_words=self.min_words,
            max_length_ratio=self.max_length_ratio,
        )
        return AttackOutput(
            text=candidate,
            metadata={
                "method": self.name,
                "backend": "vllm",
                "model": self.model_name,
                "repair_attempt": True,
                "repair_source_fail_reason": fail_reason,
            },
            all_candidates=candidates,
            candidate_scores=[0.0],
            original_text=text,
            valid=valid,
            fail_reason=repaired_fail_reason,
        )
    
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
        try:
            candidates = self._generate_candidates(text, n_candidates)
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            candidates = []
        
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
                "backend": "vllm",
                "model": self.model_name,
                "n_candidates": n_candidates,
                "best_idx": best_idx,
                "best_detector_score": scores[best_idx],
                "rerank_detector": self.rerank_detector_name,
            },
            all_candidates=candidates,
            candidate_scores=scores,
        )

    def attack(
        self,
        text: str,
        n_candidates: int = 1,
        **kwargs,
    ) -> AttackOutput:
        result = super().attack(text, n_candidates=n_candidates, **kwargs)
        if result.valid or not self.validate_outputs:
            return result

        repaired = self._repair_invalid_output(text, result.fail_reason)
        if repaired is not None and repaired.valid:
            logger.info("Recovered invalid %s output via strict length repair", self.name)
            return repaired
        return result

    def attack_batch(
        self,
        texts: List[str],
        n_candidates: int = 1,
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

            try:
                grouped_candidates = self.generator.generate_batch(
                    prompts,
                    n=n_candidates,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=batch_max_tokens,
                )
            except Exception as e:
                logger.error(f"Batch generation failed: {e}")
                grouped_candidates = [[] for _ in batch_texts]

            flat_candidates: List[str] = []
            group_sizes: List[int] = []
            for candidates in grouped_candidates:
                filtered = [candidate for candidate in candidates if candidate]
                flat_candidates.extend(filtered)
                group_sizes.append(len(filtered))

            flat_scores: List[float] = []
            if n_candidates > 1 and flat_candidates:
                if self.rerank_detector is None:
                    import torch
                    from ..detectors import get_detector

                    self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
                    logger.info(f"Loading reranking detector: {self.rerank_detector_name}")
                    self.rerank_detector = get_detector(self.rerank_detector_name, device=self.device)
                    self.rerank_detector.load()
                flat_scores = self.rerank_detector.get_scores(flat_candidates)

            cursor = 0
            for index, text, size in zip(batch_indices, batch_texts, group_sizes):
                candidates = flat_candidates[cursor:cursor + size]
                scores = flat_scores[cursor:cursor + size] if flat_scores else [0.0] * len(candidates)
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

                best_idx = scores.index(min(scores)) if scores else 0
                output = AttackOutput(
                    text=candidates[best_idx],
                    metadata={
                        "method": self.name,
                        "backend": "vllm",
                        "model": self.model_name,
                        "n_candidates": n_candidates,
                        "best_idx": best_idx,
                        "best_detector_score": scores[best_idx] if scores else 0.0,
                        "rerank_detector": self.rerank_detector_name,
                    },
                    all_candidates=candidates,
                    candidate_scores=scores,
                    original_text=text,
                )
                if self.validate_outputs:
                    valid, fail_reason = validate_attack_output(
                        text,
                        output.text,
                        min_words=self.min_words,
                        max_length_ratio=self.max_length_ratio,
                    )
                    output.valid = valid
                    output.fail_reason = fail_reason
                results[index] = output

            completed += len(batch_texts)
            logger.info("[%s] Progress: %d/%d", self.name, completed, total)

        return [result for result in results if result is not None]


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
