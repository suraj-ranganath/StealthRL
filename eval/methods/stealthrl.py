"""
StealthRL attack method (M2).

Uses RL-trained policy checkpoint for paraphrasing.

Supports two backends:
1. Local PEFT: Load LoRA adapter from local directory
2. Tinker API: Use Tinker cloud inference with checkpoint JSON
"""

import json
import logging
from pathlib import Path
from typing import Optional, List

import torch

from .base import BaseAttackMethod, AttackOutput

logger = logging.getLogger(__name__)


PARAPHRASE_PROMPT = """Please paraphrase the following text while maintaining its meaning and style. Output only the paraphrased text without any additional explanation.

Original text:
{text}

Paraphrased text:"""


class StealthRLTinker(BaseAttackMethod):
    """
    M2: StealthRL attack using Tinker cloud inference.
    
    Uses Tinker's native SamplingClient for efficient batched inference.
    The native API supports num_samples for getting multiple completions
    from a single prompt in one API call.
    """
    
    def __init__(
        self,
        checkpoint_json: str,
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
        """
        Initialize StealthRL with Tinker backend.
        
        Args:
            checkpoint_json: Path to Tinker checkpoint JSON file
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            max_new_tokens: Maximum tokens to generate
            rerank_detector: Detector for best-of-N selection (default: roberta)
            device: Device for detector
        """
        super().__init__(name="stealthrl")
        
        self.checkpoint_json = Path(checkpoint_json)
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
        self.checkpoint_info = None
        self.sampler_path = None
        self.tokenizer = None
        self.rerank_detector = None
    
    def load(self):
        """Load Tinker native sampling client and reranking detector."""
        import os
        import torch
        from ..detectors import get_detector
        
        # Load checkpoint info
        logger.info(f"Loading checkpoint info from {self.checkpoint_json}...")
        with open(self.checkpoint_json) as f:
            self.checkpoint_info = json.load(f)
        
        self.sampler_path = self.checkpoint_info["checkpoints"]["sampler_weights"]
        base_model = self.checkpoint_info.get("base_model", "Qwen/Qwen3-4B-Instruct-2507")
        
        logger.info(f"Connecting to Tinker for {base_model}...")
        logger.info(f"Sampler: {self.sampler_path}")
        
        # Check for API key
        api_key = os.environ.get("TINKER_API_KEY")
        if not api_key:
            raise RuntimeError("TINKER_API_KEY environment variable not set")
        
        # Use native Tinker sampling client (supports num_samples for batching)
        try:
            import tinker
            from tinker import ServiceClient
        except ImportError:
            raise RuntimeError("Tinker not installed. Install with: pip install tinker")
        
        service_client = ServiceClient()
        # Create sampling client from the checkpoint path
        self.sampling_client = service_client.create_sampling_client(model_path=self.sampler_path)
        self.tokenizer = self.sampling_client.get_tokenizer()
        
        logger.info(f"✓ StealthRL (Tinker Native API) ready - supports batched num_samples")
        logger.info(f"Reranking detector ({self.rerank_detector_name}) will load lazily if n_candidates > 1")
        self._loaded = True
    
    def _generate_candidates(self, text: str, n_candidates: int) -> List[str]:
        """
        Generate multiple paraphrase candidates in a SINGLE API call.
        
        Uses Tinker's native num_samples parameter for efficient batching.
        This gets N completions from the same prompt in one request.
        """
        import time
        from tinker import types
        
        prompt_text = PARAPHRASE_PROMPT.format(text=text)
        input_words = len(text.split())
        
        # Apply chat template and tokenize
        messages = [{"role": "user", "content": prompt_text}]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = self.tokenizer.encode(formatted)
        model_input = types.ModelInput.from_ints(input_ids)
        
        params = types.SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        
        api_start = time.time()
        logger.info(f"[TINKER] Requesting {n_candidates} candidates for {input_words}-word text (single API call)...")
        
        try:
            # num_samples=N gives N independent completions in ONE call
            future = self.sampling_client.sample(
                prompt=model_input,
                sampling_params=params,
                num_samples=n_candidates,
            )
            result = future.result()
            
            api_elapsed = time.time() - api_start
            
            # Extract all candidates from the response
            candidates = []
            for i, sample in enumerate(result.sequences):
                output_text = self.tokenizer.decode(sample.tokens, skip_special_tokens=True)
                output_text = output_text.strip()
                if output_text:
                    candidates.append(output_text)
                    logger.debug(f"[TINKER] Candidate {i+1}: {len(output_text.split())} words")
            
            logger.info(f"[TINKER] Got {len(candidates)}/{n_candidates} candidates in {api_elapsed:.1f}s (single call)")
            return candidates
            
        except Exception as e:
            api_elapsed = time.time() - api_start
            logger.warning(f"[TINKER] Batch request failed after {api_elapsed:.1f}s: {e}")
            return []
    
    def _attack_impl(
        self,
        text: str,
        n_candidates: int = 1,
        **kwargs,
    ) -> AttackOutput:
        """Generate StealthRL paraphrase(s) via Tinker."""
        import time
        if not self._loaded:
            self.load()
        
        attack_start = time.time()
        
        # Generate all candidates in a single API call using num_samples
        candidates = self._generate_candidates(text, n_candidates)
        
        attack_elapsed = time.time() - attack_start
        logger.info(f"[TINKER] Total: {len(candidates)}/{n_candidates} candidates in {attack_elapsed:.1f}s")
        
        if not candidates:
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
            scores = [self.rerank_detector.get_scores(c) for c in candidates]
            best_idx = scores.index(min(scores))  # Lower score = better evasion
        else:
            scores = [0.0]
            best_idx = 0
        
        return AttackOutput(
            text=candidates[best_idx],
            metadata={
                "method": self.name,
                "backend": "tinker",
                "checkpoint": str(self.checkpoint_json),
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
        """
        Concurrent batch attack for Tinker-backed StealthRL (M2).

        Uses sample_async with concurrency + chunking for high throughput.
        """
        if not self._loaded:
            self.load()

        if self.tinker_concurrency <= 1:
            return super().attack_batch(texts, n_candidates=n_candidates, **kwargs)

        from tinker import types
        from ..detectors import get_detector
        from .base import validate_attack_output
        from ..tinker_concurrency import run_sampling_concurrent

        prompt_template = PARAPHRASE_PROMPT

        def build_model_input(text: str):
            prompt_text = prompt_template.format(text=text)
            messages = [{"role": "user", "content": prompt_text}]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
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
                        "checkpoint": str(self.checkpoint_json),
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
                    "checkpoint": str(self.checkpoint_json),
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


class StealthRLAttack(BaseAttackMethod):
    """
    M2: StealthRL attack using RL-trained policy.
    
    Loads a LoRA checkpoint trained with GRPO and uses it for paraphrasing.
    Supports candidate generation and reranking.
    
    NOTE: Base model must match what the LoRA adapter was trained on.
    Default is Qwen3-4B-Instruct-2507 (same as M1 for fair comparison).
    """
    
    # Base model used for training - must match Tinker config
    BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
    
    def __init__(
        self,
        checkpoint_path: str,
        base_model: str = None,
        device: str = None,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        scorer_fn: callable = None,
    ):
        """
        Initialize StealthRL attack.
        
        Args:
            checkpoint_path: Path to LoRA checkpoint directory
            base_model: Base model name (default: Qwen2.5-3B-Instruct)
            device: Device to use
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            max_new_tokens: Maximum tokens to generate
            scorer_fn: Optional function to score candidates
        """
        super().__init__(name="stealthrl")
        
        self.checkpoint_path = Path(checkpoint_path)
        self.base_model_name = base_model or self.BASE_MODEL
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.scorer_fn = scorer_fn
        
        self.model = None
        self.tokenizer = None
    
    def load(self):
        """Load base model with LoRA adapter."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        
        logger.info(f"Loading StealthRL checkpoint from {self.checkpoint_path}...")
        
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load base model
        logger.info(f"Loading base model: {self.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        # Load LoRA adapter
        logger.info(f"Loading LoRA adapter from {self.checkpoint_path}")
        self.model = PeftModel.from_pretrained(
            base_model,
            str(self.checkpoint_path),
            is_trainable=False,
        )
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self._loaded = True
        logger.info(f"✓ StealthRL loaded on {self.device}")
    
    def _generate_single(self, text: str) -> str:
        """Generate a single paraphrase."""
        prompt = PARAPHRASE_PROMPT.format(text=text)
        
        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted = prompt
        
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        return generated.strip()
    
    def _attack_impl(
        self,
        text: str,
        n_candidates: int = 1,
        **kwargs,
    ) -> AttackOutput:
        """
        Generate StealthRL paraphrase(s).
        
        Args:
            text: Original AI-generated text
            n_candidates: Number of candidates to generate
        
        Returns:
            AttackOutput with best candidate
        """
        if not self._loaded:
            self.load()
        
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
        
        # Score and select best
        if self.scorer_fn and n_candidates > 1:
            scores = [self.scorer_fn(c) for c in candidates]
            best_idx = scores.index(min(scores))
        else:
            scores = [0.0] * len(candidates)
            best_idx = 0
        
        return AttackOutput(
            text=candidates[best_idx],
            metadata={
                "method": self.name,
                "backend": "peft",
                "checkpoint": str(self.checkpoint_path),
                "n_candidates": n_candidates,
                "best_idx": best_idx,
            },
            all_candidates=candidates,
            candidate_scores=scores,
            original_text=text,
        )


class StealthRLAttackWithReranking(StealthRLAttack):
    """
    M2 variant: StealthRL with detector-guided reranking.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        rerank_detector: str = "roberta",
        **kwargs,
    ):
        super().__init__(checkpoint_path=checkpoint_path, **kwargs)
        self.name = f"stealthrl_rerank_{rerank_detector}"
        self.rerank_detector_name = rerank_detector
        self.rerank_detector = None
    
    def load(self):
        """Load model and reranking detector."""
        super().load()
        
        from ..detectors import get_detector
        
        logger.info(f"Loading reranking detector: {self.rerank_detector_name}")
        self.rerank_detector = get_detector(self.rerank_detector_name, device=self.device)
        self.rerank_detector.load()
        
        self.scorer_fn = lambda text: self.rerank_detector.get_scores(text)


def load_stealthrl_checkpoint(
    checkpoint_path: str,
    rerank_detector: Optional[str] = None,
    **kwargs,
) -> BaseAttackMethod:
    """
    Convenience function to load StealthRL with optional reranking.
    
    Args:
        checkpoint_path: Path to checkpoint
        rerank_detector: Optional detector name for reranking
        **kwargs: Additional arguments
    
    Returns:
        StealthRL attack method
    """
    if rerank_detector:
        return StealthRLAttackWithReranking(
            checkpoint_path=checkpoint_path,
            rerank_detector=rerank_detector,
            **kwargs,
        )
    return StealthRLAttack(checkpoint_path=checkpoint_path, **kwargs)
