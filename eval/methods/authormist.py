"""
AuthorMist baseline (M4).

Uses the published AuthorMist model for text humanization.
Paper: https://arxiv.org/abs/2503.08716
Model: https://huggingface.co/authormist/authormist-originality

Supports two backends:
1. HuggingFace Transformers (default, requires GPU)
2. Ollama (for local GGUF inference on M4 Mac)
"""

import logging
from typing import Optional

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
    
    def __init__(
        self,
        model_name: str = None,
        ollama_url: str = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        rerank_detector: str = "roberta",
        device: str = None,
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
    
    def _humanize(self, text: str) -> str:
        """Apply AuthorMist humanization via Ollama."""
        prompt = f"""Please paraphrase the following text to make it more human-like while preserving the original meaning:

{text}

Paraphrased text:"""
        
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
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
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
