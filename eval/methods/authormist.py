"""
AuthorMist baseline (M4).

Uses the published AuthorMist model for text humanization.
Paper: https://arxiv.org/abs/2503.08716
Model: https://huggingface.co/authormist/authormist-originality

Supports two backends:
1. HuggingFace Transformers (default, requires GPU)
2. Ollama (for local GGUF inference on M4 Mac)
"""

import concurrent.futures
import logging
import random
import threading
import time
from typing import Optional, List, Tuple, Any, Dict

import requests
import torch

from .base import BaseAttackMethod, AttackOutput

logger = logging.getLogger(__name__)


class AuthorMistOllama(BaseAttackMethod):
    """
    M4: AuthorMist attack using Ollama backend (for GGUF on Mac).
    
    This is the preferred method for M4 MacBook as it uses Ollama
    for efficient GGUF inference.
    
    Setup:
        1. Place GGUF file in models/authormist/
        2. Create Modelfile: ollama create authormist -f Modelfile
        3. Or: ollama run authormist (if already created)
    """
    
    DEFAULT_MODEL = "authormist"  # Ollama model name
    DEFAULT_OLLAMA_URL = "http://localhost:11434"
    DEFAULT_CONCURRENCY = 4
    DEFAULT_CHUNK_SIZE = 256
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_BACKOFF_BASE_S = 0.5
    DEFAULT_KEEP_ALIVE = -1
    
    def __init__(
        self,
        model_name: str = None,
        ollama_url: str = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        rerank_detector: str = "roberta",
        device: str = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
        keep_alive: Optional[int] = DEFAULT_KEEP_ALIVE,
        request_timeout_s: int = 120,
        use_chat: bool = False,
        warmup: bool = True,
        **kwargs,
    ):
        """
        Initialize AuthorMist with Ollama backend.
        
        Args:
            model_name: Ollama model name (default: authormist)
            ollama_url: Ollama server URL
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            max_new_tokens: Maximum tokens to generate
            rerank_detector: Detector for best-of-N selection (default: roberta)
            device: Device for detector
            concurrency: Max concurrent Ollama requests
            chunk_size: Max prompts to submit per chunk
            max_retries: Retries for overload/timeout errors
            backoff_base_s: Base seconds for exponential backoff
            keep_alive: Ollama keep_alive value (negative keeps model loaded)
            request_timeout_s: Request timeout for Ollama calls
            use_chat: Use /api/chat instead of /api/generate
            warmup: Pre-warm model once at load
        """
        super().__init__(name="authormist")
        
        self.model_name = model_name or self.DEFAULT_MODEL
        self.ollama_url = ollama_url or self.DEFAULT_OLLAMA_URL
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.rerank_detector_name = rerank_detector
        self.device = device
        self.rerank_detector = None
        self.ollama_concurrency = max(1, int(concurrency))
        self.ollama_chunk_size = max(1, int(chunk_size))
        self.ollama_max_retries = max(0, int(max_retries))
        self.ollama_backoff_base_s = float(backoff_base_s)
        self.ollama_keep_alive = keep_alive
        self.ollama_request_timeout_s = int(request_timeout_s)
        self.ollama_use_chat = bool(use_chat)
        self.ollama_warmup = bool(warmup)
        self._rerank_lock = threading.Lock()
        self._warm_lock = threading.Lock()
        self._warmed = False
    
    def load(self):
        """Verify Ollama connection and load reranking detector."""
        import torch
        from ..detectors import get_detector
        
        logger.info(f"Connecting to Ollama for AuthorMist ({self.model_name})...")
        
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            
            model_base = self.model_name.split(":")[0]
            available = any(model_base in m for m in models)
            
            if not available:
                logger.warning(f"Model {self.model_name} not found in Ollama. Available: {models}")
                logger.info("Please run: ollama create authormist -f models/authormist/Modelfile")
                raise RuntimeError(f"AuthorMist model not found in Ollama. Create it first.")
            
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to Ollama at {self.ollama_url}. Is Ollama running?")
        
        self._loaded = True
        logger.info(f"✓ AuthorMist (Ollama) ready: {self.model_name}")
        logger.info(f"Reranking detector ({self.rerank_detector_name}) will load lazily if n_candidates > 1")
        if self.ollama_warmup:
            self._warm_model()
    
    def _warm_model(self) -> None:
        """Pre-warm Ollama model to avoid cold-start latency."""
        with self._warm_lock:
            if self._warmed:
                return
            try:
                logger.info(f"Warming Ollama model ({self.model_name})...")
                _ = self._ollama_request(
                    prompt="Hello",
                    num_predict=1,
                    allow_retry=True,
                )
                self._warmed = True
            except Exception as e:
                logger.warning(f"Ollama warmup failed: {e}")

    def _build_ollama_payload(
        self,
        prompt: str,
        num_predict: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        options = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": num_predict if num_predict is not None else self.max_new_tokens,
        }
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "stream": False,
            "options": options,
        }
        if self.ollama_keep_alive is not None:
            payload["keep_alive"] = self.ollama_keep_alive

        if self.ollama_use_chat:
            payload["messages"] = [{"role": "user", "content": prompt}]
            url = f"{self.ollama_url}/api/chat"
        else:
            payload["prompt"] = prompt
            url = f"{self.ollama_url}/api/generate"

        return url, payload

    def _ollama_request(
        self,
        prompt: str,
        num_predict: Optional[int] = None,
        allow_retry: bool = True,
    ) -> Dict[str, Any]:
        """Send a non-streaming request to Ollama with retries for overload."""
        url, payload = self._build_ollama_payload(prompt=prompt, num_predict=num_predict)
        last_error: Optional[Exception] = None
        attempts = self.ollama_max_retries + 1 if allow_retry else 1

        for attempt in range(attempts):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.ollama_request_timeout_s,
                )
                if response.status_code in (429, 503):
                    raise requests.HTTPError(
                        f"Ollama overloaded (HTTP {response.status_code})",
                        response=response,
                    )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                last_error = e
                if attempt >= attempts - 1:
                    break
                sleep_s = self.ollama_backoff_base_s * (2 ** attempt) + random.uniform(0, self.ollama_backoff_base_s)
                time.sleep(sleep_s)

        raise last_error if last_error is not None else RuntimeError("Ollama request failed")

    def _extract_response_text(self, result: Dict[str, Any]) -> str:
        if self.ollama_use_chat:
            message = result.get("message", {})
            if isinstance(message, dict):
                return (message.get("content") or "").strip()
            return ""
        return (result.get("response") or "").strip()

    def _humanize(self, text: str) -> str:
        """Apply AuthorMist humanization via Ollama."""
        prompt = f"""Please paraphrase the following text to make it more human-like while preserving the original meaning:

{text}

Paraphrased text:"""
        
        try:
            result = self._ollama_request(prompt=prompt)
            return self._extract_response_text(result)
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return ""
    
    def _attack_impl(
        self,
        text: str,
        n_candidates: int = 1,
        **kwargs,
    ) -> AttackOutput:
        """Apply AuthorMist humanization."""
        if not self._loaded:
            self.load()
        
        candidates = []
        for _ in range(n_candidates):
            try:
                humanized = self._humanize(text)
                if humanized:
                    candidates.append(humanized)
            except Exception as e:
                logger.warning(f"AuthorMist generation failed: {e}")
        
        if not candidates:
            return AttackOutput(
                text=text,
                metadata={"method": self.name, "error": "generation_failed"},
            )
        
        # Score and select best candidate
        if n_candidates > 1:
            # Lazy load detector if not already loaded
            with self._rerank_lock:
                if self.rerank_detector is None:
                    from ..detectors import get_detector
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
                "model": self.model_name,
                "backend": "ollama",
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
        """Concurrent batch attack using bounded Ollama requests."""
        if not self._loaded:
            self.load()

        if self.ollama_concurrency <= 1 or len(texts) <= 1:
            return super().attack_batch(texts, n_candidates=n_candidates, **kwargs)

        results: List[Optional[AttackOutput]] = [None] * len(texts)
        batch_start = time.time()
        log_interval = max(1, len(texts) // 10)
        completed = 0

        def _run_one(index: int, text: str) -> Tuple[int, AttackOutput, float]:
            sample_start = time.time()
            try:
                result = self.attack(text, n_candidates=n_candidates, **kwargs)
            except Exception as e:
                result = AttackOutput(
                    text=text,
                    original_text=text,
                    valid=False,
                    fail_reason=f"attack_exception: {str(e)}",
                    metadata={"error": str(e)},
                )
            sample_elapsed = time.time() - sample_start
            return index, result, sample_elapsed

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.ollama_concurrency) as executor:
            for start in range(0, len(texts), self.ollama_chunk_size):
                end = min(start + self.ollama_chunk_size, len(texts))
                futures = [
                    executor.submit(_run_one, idx, texts[idx])
                    for idx in range(start, end)
                ]
                for future in concurrent.futures.as_completed(futures):
                    idx, result, sample_elapsed = future.result()
                    results[idx] = result
                    completed += 1
                    if completed % log_interval == 0 or completed == len(texts):
                        batch_elapsed = time.time() - batch_start
                        rate = completed / batch_elapsed if batch_elapsed > 0 else 0
                        eta = (len(texts) - completed) / rate if rate > 0 else 0
                        logger.info(
                            f"[{self.name}] Progress: {completed}/{len(texts)} "
                            f"({sample_elapsed:.2f}s/sample, {rate:.2f} samples/s, ETA: {eta:.0f}s)"
                        )

        finalized: List[AttackOutput] = []
        for i, result in enumerate(results):
            if result is None:
                result = AttackOutput(
                    text=texts[i],
                    original_text=texts[i],
                    valid=False,
                    fail_reason="missing_result",
                    metadata={"error": "missing_result"},
                )
            finalized.append(result)
        return finalized


class AuthorMist(BaseAttackMethod):
    """
    M4: AuthorMist attack using published HF model.
    
    AuthorMist is trained for text humanization to evade AI detectors.
    Paper: https://arxiv.org/abs/2503.08716
    """
    
    MODEL_NAME = "authormist/authormist-originality"
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        rerank_detector: str = "roberta",
    ):
        """
        Initialize AuthorMist.
        
        Args:
            model_name: HuggingFace model name (default: official AuthorMist)
            device: Device to use
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            rerank_detector: Detector for best-of-N selection (default: roberta)
        """
        super().__init__(name="authormist")
        
        self.model_name = model_name or self.MODEL_NAME
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.rerank_detector_name = rerank_detector
        
        self.model = None
        self.tokenizer = None
        self.rerank_detector = None
    
    def load(self):
        """Load AuthorMist model from HuggingFace and reranking detector."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from ..detectors import get_detector
        
        logger.info(f"Loading AuthorMist from {self.model_name}...")
        
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self._loaded = True
            logger.info(f"✓ AuthorMist loaded on {self.device}")
            logger.info(f"Reranking detector ({self.rerank_detector_name}) will load lazily if n_candidates > 1")
            
        except Exception as e:
            logger.error(f"Failed to load AuthorMist: {e}")
            raise
    
    def _humanize(self, text: str) -> str:
        """Apply AuthorMist humanization."""
        # Official AuthorMist prompt format from HF model card:
        # https://huggingface.co/authormist/authormist-originality
        prompt = f"""Please paraphrase the following text to make it more human-like while preserving the original meaning:

{text}

Paraphrased text:"""
        
        # Check for chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                formatted = prompt
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
        Apply AuthorMist humanization.
        
        Args:
            text: Original AI-generated text
            n_candidates: Number of candidates (AuthorMist typically uses 1)
        
        Returns:
            AttackOutput with humanized text
        """
        if not self._loaded:
            self.load()
        
        candidates = []
        for _ in range(n_candidates):
            try:
                humanized = self._humanize(text)
                if humanized:
                    candidates.append(humanized)
            except Exception as e:
                logger.warning(f"AuthorMist generation failed: {e}")
        
        if not candidates:
            return AttackOutput(
                text=text,
                metadata={"method": self.name, "error": "generation_failed"},
            )
        
        # Score and select best candidate
        if n_candidates > 1:
            # Lazy load detector if not already loaded
            if self.rerank_detector is None:
                from ..detectors import get_detector
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
                "model": self.model_name,
                "n_candidates": n_candidates,
                "best_idx": best_idx,
                "best_detector_score": scores[best_idx],
                "rerank_detector": self.rerank_detector_name,
            },
            all_candidates=candidates,
            candidate_scores=scores,
            original_text=text,
        )


class AuthorMistFallback(BaseAttackMethod):
    """
    Fallback AuthorMist using a standard paraphrase model.
    
    Use this if the official AuthorMist model is not available.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = None,
        **kwargs,
    ):
        super().__init__(name="authormist_fallback")
        
        from .simple_paraphrase import SimpleParaphrase
        self._paraphraser = SimpleParaphrase(
            model_name=model_name,
            device=device,
            **kwargs,
        )
    
    def load(self):
        """Load fallback model."""
        self._paraphraser.load()
        self._loaded = True
    
    def attack(
        self,
        text: str,
        n_candidates: int = 1,
        **kwargs,
    ) -> AttackOutput:
        """Generate paraphrase using fallback."""
        if not self._loaded:
            self.load()
        
        result = self._paraphraser.attack(text, n_candidates, **kwargs)
        result.metadata["method"] = self.name
        result.metadata["fallback"] = True
        
        return result
