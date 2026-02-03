"""
Detector implementations for StealthRL evaluation.

Provides a unified interface for multiple detector families:
- Classifier-based: RoBERTa OpenAI Detector
- Curvature-based: Fast-DetectGPT, DetectGPT
- Zero-shot: Binoculars
- Feature-based: Ghostbuster (optional)
- Longformer-based: MAGE

All detectors output scores in [0, 1] where higher = more likely AI-generated.
"""

import asyncio
import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

import torch
import numpy as np

logger = logging.getLogger(__name__)


def get_device() -> str:
    """Get best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class DetectorResult:
    """Result from a detector."""
    score: float  # AI probability [0, 1], higher = more AI
    raw_score: float  # Original score before normalization
    metadata: Dict[str, Any]


class BaseEvalDetector(ABC):
    """Base class for evaluation detectors."""
    
    # Convention: higher score = more likely AI-generated
    HIGHER_IS_AI = True
    
    def __init__(
        self,
        name: str,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.name = name
        self.device = device or get_device()
        self.batch_size = batch_size
        self._loaded = False
    
    @abstractmethod
    def load(self):
        """Load model(s) required for detection."""
        pass
    
    @abstractmethod
    def _detect_single(self, text: str) -> DetectorResult:
        """Detect single text. Must be implemented by subclasses."""
        pass
    
    def detect(self, texts: Union[str, List[str]]) -> Union[DetectorResult, List[DetectorResult]]:
        """
        Detect AI-generated text.
        
        Args:
            texts: Single text or list of texts
        
        Returns:
            Single DetectorResult or list of DetectorResults
        """
        if not self._loaded:
            self.load()
        
        if isinstance(texts, str):
            return self._detect_single(texts)
        
        return self._detect_batch(texts)
    
    def _detect_batch(self, texts: List[str]) -> List[DetectorResult]:
        """Batch detection (default: sequential). Override for true batching."""
        results = []
        for text in texts:
            results.append(self._detect_single(text))
        return results
    
    def get_scores(self, texts: Union[str, List[str]]) -> Union[float, List[float]]:
        """Convenience method to get just scores."""
        results = self.detect(texts)
        if isinstance(results, DetectorResult):
            return results.score
        return [r.score for r in results]


class RoBERTaOpenAIDetector(BaseEvalDetector):
    """
    RoBERTa-based OpenAI detector (classifier family).
    
    HF Model: openai-community/roberta-large-openai-detector
    Outputs probability of AI-generated text.
    """
    
    MODEL_NAME = "openai-community/roberta-large-openai-detector"
    
    def __init__(
        self,
        device: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name="roberta_openai", device=device, **kwargs)
        self.model_name = model_name or self.MODEL_NAME
        self.model = None
        self.tokenizer = None
    
    def load(self):
        """Load RoBERTa model from HuggingFace."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        logger.info(f"Loading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        
        self._loaded = True
        logger.info(f"✓ {self.name} loaded on {self.device}")
    
    def _detect_single(self, text: str) -> DetectorResult:
        """Detect single text using RoBERTa classifier."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            # Class 0 = Fake/AI for OpenAI detectors
            ai_prob = probs[0, 0].item()
        
        return DetectorResult(
            score=ai_prob,
            raw_score=ai_prob,
            metadata={"model": self.model_name},
        )
    
    def _detect_batch(self, texts: List[str]) -> List[DetectorResult]:
        """Efficient batch detection."""
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                ai_probs = probs[:, 0].cpu().numpy()
            
            for prob in ai_probs:
                results.append(DetectorResult(
                    score=float(prob),
                    raw_score=float(prob),
                    metadata={"model": self.model_name},
                ))
        
        return results


class FastDetectGPTDetector(BaseEvalDetector):
    """
    Fast-DetectGPT detector (curvature family).
    
    Based on sampling discrepancy analytic method.
    Paper: https://arxiv.org/abs/2310.05130
    
    Uses GPT-Neo 2.7B as the standard scoring model (same as paper).
    """
    
    DEFAULT_MODEL = "EleutherAI/gpt-neo-2.7B"
    
    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = None,
        **kwargs,
    ):
        super().__init__(name="fast_detectgpt", device=device, **kwargs)
        self.model_name = model_name or self.DEFAULT_MODEL
        self.model = None
        self.tokenizer = None
    
    def load(self):
        """Load scoring model."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info(f"Loading {self.model_name} for Fast-DetectGPT...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self._loaded = True
        logger.info(f"✓ {self.name} loaded on {self.device}")
    
    @staticmethod
    def _sampling_discrepancy_analytic(logits, labels) -> float:
        """
        Official Fast-DetectGPT sampling discrepancy criterion.
        Uses same model for reference and scoring (white-box setting).
        """
        lprobs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        
        # Gather log probabilities at label positions
        labels_expanded = labels.unsqueeze(-1)
        log_likelihood = lprobs.gather(dim=-1, index=labels_expanded).squeeze(-1)
        
        # Compute mean and variance under model distribution
        mean_ref = (probs * lprobs).sum(dim=-1)
        var_ref = (probs * lprobs.pow(2)).sum(dim=-1) - mean_ref.pow(2)
        
        # Sampling discrepancy: (LL - mean) / sqrt(var)
        discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
        
        return discrepancy.mean().item()
    
    def _detect_single(self, text: str) -> DetectorResult:
        """Compute Fast-DetectGPT score."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, :-1]
            labels = inputs.input_ids[:, 1:]
            
            crit = self._sampling_discrepancy_analytic(logits, labels)
            
            # Convert to probability: positive criterion -> AI
            score = torch.sigmoid(torch.tensor(crit * 0.5)).item()
        
        return DetectorResult(
            score=score,
            raw_score=crit,
            metadata={"model": self.model_name, "criterion": crit},
        )


class DetectGPTDetector(BaseEvalDetector):
    """
    Original DetectGPT detector (curvature family).
    
    Uses perturbation-based detection with T5 mask filling.
    Paper: https://arxiv.org/abs/2301.11305
    
    Note: Much slower than Fast-DetectGPT due to perturbations.
    """
    
    DEFAULT_MODEL = "gpt2-medium"
    MASK_MODEL = "t5-large"
    
    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = None,
        mask_model: str = None,
        n_perturbations: int = 10,
        **kwargs,
    ):
        super().__init__(name="detectgpt", device=device, **kwargs)
        self.model_name = model_name or self.DEFAULT_MODEL
        self.mask_model_name = mask_model or self.MASK_MODEL
        self.n_perturbations = n_perturbations
        self.model = None
        self.tokenizer = None
        self.mask_model = None
        self.mask_tokenizer = None
    
    def load(self):
        """Load scoring model and mask-filling model."""
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            T5ForConditionalGeneration, T5Tokenizer
        )
        
        logger.info(f"Loading {self.model_name} for DetectGPT...")
        
        # Scoring model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Mask-filling model for perturbations
        logger.info(f"Loading {self.mask_model_name} for perturbations...")
        self.mask_tokenizer = T5Tokenizer.from_pretrained(self.mask_model_name)
        self.mask_model = T5ForConditionalGeneration.from_pretrained(
            self.mask_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.mask_model.eval()
        
        self._loaded = True
        logger.info(f"✓ {self.name} loaded on {self.device}")
    
    def _compute_log_likelihood(self, text: str) -> float:
        """Compute average log-likelihood of text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs.input_ids)
            return -outputs.loss.item()  # Negative NLL
    
    def _perturb_text(self, text: str, mask_pct: float = 0.15) -> str:
        """Generate perturbation using T5 mask filling."""
        import re
        
        words = text.split()
        n_mask = max(1, int(len(words) * mask_pct))
        
        # Randomly select positions to mask
        mask_indices = np.random.choice(len(words), size=min(n_mask, len(words)), replace=False)
        
        # Replace with T5 mask tokens
        masked_words = words.copy()
        for i, idx in enumerate(sorted(mask_indices)):
            masked_words[idx] = f"<extra_id_{i}>"
        
        masked_text = " ".join(masked_words)
        
        # Generate fill-ins
        inputs = self.mask_tokenizer(masked_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.mask_model.generate(
                **inputs,
                max_length=256,
                do_sample=True,
                temperature=1.0,
            )
        
        filled = self.mask_tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Parse T5 output and reconstruct text
        result_words = words.copy()
        for i, idx in enumerate(sorted(mask_indices)):
            pattern = f"<extra_id_{i}>(.*?)(?:<extra_id_|$)"
            match = re.search(pattern, filled)
            if match:
                fill = match.group(1).strip()
                if fill:
                    result_words[idx] = fill
        
        return " ".join(result_words)
    
    def _detect_single(self, text: str) -> DetectorResult:
        """
        Compute DetectGPT score using perturbation comparison.
        
        Criterion: (LL_original - mean(LL_perturbed)) / std(LL_perturbed)
        """
        # Original log-likelihood
        ll_original = self._compute_log_likelihood(text)
        
        # Perturbed log-likelihoods
        ll_perturbed = []
        for _ in range(self.n_perturbations):
            try:
                perturbed = self._perturb_text(text)
                ll_perturbed.append(self._compute_log_likelihood(perturbed))
            except Exception as e:
                logger.warning(f"Perturbation failed: {e}")
                continue
        
        if len(ll_perturbed) < 2:
            # Not enough perturbations succeeded
            return DetectorResult(
                score=0.5,
                raw_score=0.0,
                metadata={"error": "insufficient_perturbations"},
            )
        
        # DetectGPT criterion
        mean_perturbed = np.mean(ll_perturbed)
        std_perturbed = np.std(ll_perturbed) + 1e-10
        criterion = (ll_original - mean_perturbed) / std_perturbed
        
        # Convert to probability
        score = torch.sigmoid(torch.tensor(criterion * 0.5)).item()
        
        return DetectorResult(
            score=score,
            raw_score=criterion,
            metadata={
                "model": self.model_name,
                "criterion": criterion,
                "ll_original": ll_original,
                "ll_perturbed_mean": mean_perturbed,
                "n_perturbations": len(ll_perturbed),
            },
        )


class BinocularsDetector(BaseEvalDetector):
    """
    Binoculars detector (zero-shot family).
    
    Correct implementation based on official repo:
    https://github.com/ahans30/Binoculars
    
    Formula: binoculars_score = ppl / cross_ppl
    - ppl: perplexity from performer model
    - cross_ppl: cross-entropy comparing observer probs with performer logits
    
    Lower score = more likely AI-generated
    Paper: https://arxiv.org/abs/2401.12070
    """
    
    # Default model pair from paper (requires ~14GB GPU memory)
    PERFORMER = "tiiuae/falcon-7b-instruct"
    OBSERVER = "tiiuae/falcon-7b"
    
    # Lightweight alternatives for faster evaluation (~2GB GPU)
    PERFORMER_LIGHT = "gpt2-medium"
    OBSERVER_LIGHT = "gpt2-large"
    
    # Thresholds from official implementation
    THRESHOLD_LOW_FPR = 0.8536432310785527  # 0.01% FPR
    THRESHOLD_ACCURACY = 0.9015310749276843  # optimized F1
    
    # Convention: LOWER score = more AI
    HIGHER_IS_AI = False
    
    def __init__(
        self,
        device: Optional[str] = None,
        performer: str = None,
        observer: str = None,
        use_lightweight: bool = True,
        max_length: int = 512,
        **kwargs,
    ):
        super().__init__(name="binoculars", device=device, **kwargs)
        
        if use_lightweight:
            self.performer_name = performer or self.PERFORMER_LIGHT
            self.observer_name = observer or self.OBSERVER_LIGHT
        else:
            self.performer_name = performer or self.PERFORMER
            self.observer_name = observer or self.OBSERVER
        
        self.max_length = max_length
        self.performer = None
        self.observer = None
        self.tokenizer = None
        self.ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        self.softmax_fn = torch.nn.Softmax(dim=-1)
    
    def load(self):
        """Load performer and observer models."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info(f"Loading Binoculars: performer={self.performer_name}, observer={self.observer_name}")
        
        # Load tokenizer (from observer, same as official)
        self.tokenizer = AutoTokenizer.from_pretrained(self.observer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Observer model
        self.observer = AutoModelForCausalLM.from_pretrained(
            self.observer_name,
            torch_dtype=dtype,
        ).to(self.device)
        self.observer.eval()
        
        # Performer model  
        self.performer = AutoModelForCausalLM.from_pretrained(
            self.performer_name,
            torch_dtype=dtype,
        ).to(self.device)
        self.performer.eval()
        
        self._loaded = True
        logger.info(f"✓ {self.name} loaded on {self.device}")
    
    def _compute_perplexity(self, logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
        """
        Compute perplexity from logits (official Binoculars formula).
        
        PPL = exp(mean(cross_entropy_per_token))
        """
        # Shift for causal LM
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        shifted_mask = attention_mask[..., 1:].contiguous()
        
        # Cross-entropy per token
        ce = self.ce_loss_fn(
            shifted_logits.transpose(1, 2),
            shifted_labels
        )
        
        # Mean over valid tokens
        ppl = (ce * shifted_mask).sum(1) / shifted_mask.sum(1)
        return ppl.cpu().float().numpy()
    
    def _compute_cross_entropy(
        self, 
        observer_logits: torch.Tensor, 
        performer_logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute cross-entropy between observer and performer (official formula).
        
        Uses softmax(observer) as target distribution, performer logits as input.
        """
        vocab_size = observer_logits.shape[-1]
        
        # Softmax of observer to get probability distribution
        observer_probs = self.softmax_fn(observer_logits).view(-1, vocab_size)
        
        # Performer logits
        performer_scores = performer_logits.view(-1, vocab_size)
        
        # Cross-entropy: -sum(p * log(q))
        ce = self.ce_loss_fn(
            input=performer_scores,
            target=observer_probs
        ).view(input_ids.shape[0], -1)
        
        # Create padding mask
        padding_mask = (input_ids != self.tokenizer.pad_token_id).float()
        
        # Mean over valid tokens
        agg_ce = (ce * padding_mask).sum(1) / padding_mask.sum(1)
        return agg_ce.cpu().float().numpy()
    
    def _detect_single(self, text: str) -> DetectorResult:
        """
        Compute Binoculars score using official formula.
        
        Score = PPL(performer) / CrossEntropy(observer, performer)
        Lower score = more likely AI-generated
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        ).to(self.device)
        
        with torch.no_grad():
            observer_logits = self.observer(**inputs).logits
            performer_logits = self.performer(**inputs).logits
        
        # PPL from performer
        ppl = self._compute_perplexity(
            performer_logits, 
            inputs.input_ids, 
            inputs.attention_mask
        )[0]
        
        # Cross-entropy between observer and performer
        cross_ppl = self._compute_cross_entropy(
            observer_logits,
            performer_logits,
            inputs.input_ids,
            inputs.attention_mask,
        )[0]
        
        # Binoculars score
        raw_score = ppl / (cross_ppl + 1e-10)
        
        # Convert to unified convention (higher = more AI)
        # Since lower Binoculars score = more AI, we invert
        # Normalize to [0, 1] using sigmoid on log ratio
        normalized_score = 1.0 - torch.sigmoid(torch.tensor(np.log(raw_score + 1e-10) - np.log(self.THRESHOLD_LOW_FPR))).item()
        
        return DetectorResult(
            score=normalized_score,  # Higher = more AI (unified)
            raw_score=float(raw_score),  # Original Binoculars score (lower = more AI)
            metadata={
                "performer": self.performer_name,
                "observer": self.observer_name,
                "ppl": float(ppl),
                "cross_ppl": float(cross_ppl),
                "threshold": self.THRESHOLD_LOW_FPR,
            },
        )


class GhostbusterDetector(BaseEvalDetector):
    """
    Ghostbuster detector (optional, feature-based).
    
    Paper: https://arxiv.org/abs/2305.15047
    Official repo: https://github.com/vivek3141/ghostbuster
    
    NOTE: The original Ghostbuster requires OpenAI API access for computing
    log probabilities from ada/davinci models. This is expensive and slow.
    
    This implementation provides two modes:
    1. 'proxy': Uses a local RoBERTa model trained on similar data (fast, free)
    2. 'api': Uses the original method with OpenAI API (requires API key)
    
    Default is 'proxy' mode for evaluation without API costs.
    """
    
    # Proxy model trained for AI detection (used in proxy mode)
    PROXY_MODEL = "openai-community/roberta-large-openai-detector"
    
    def __init__(
        self,
        device: Optional[str] = None,
        mode: str = "proxy",  # 'proxy' or 'api'
        openai_api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name="ghostbuster", device=device, **kwargs)
        self.mode = mode
        self.openai_api_key = openai_api_key
        self.model = None
        self.tokenizer = None
        
        if mode == "api" and not openai_api_key:
            logger.warning(
                "Ghostbuster 'api' mode requires OpenAI API key. "
                "Falling back to 'proxy' mode."
            )
            self.mode = "proxy"
    
    def load(self):
        """Load Ghostbuster model."""
        if self.mode == "proxy":
            self._load_proxy()
        else:
            self._load_api()
        
        self._loaded = True
    
    def _load_proxy(self):
        """Load proxy RoBERTa model for fast evaluation."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        logger.info(f"Loading Ghostbuster (proxy mode) with {self.PROXY_MODEL}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.PROXY_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.PROXY_MODEL,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        
        logger.info(f"✓ {self.name} loaded on {self.device} (proxy mode)")
    
    def _load_api(self):
        """Initialize API-based Ghostbuster (full implementation)."""
        # Would require downloading feature extractors and OpenAI setup
        # For now, fall back to proxy
        logger.warning("Full API mode not implemented. Using proxy mode.")
        self._load_proxy()
    
    def _detect_single(self, text: str) -> DetectorResult:
        """Compute Ghostbuster score."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            # For OpenAI detector: Class 0 = Fake/AI
            ai_prob = probs[0, 0].item()
        
        return DetectorResult(
            score=ai_prob,
            raw_score=ai_prob,
            metadata={
                "mode": self.mode,
                "model": self.PROXY_MODEL if self.mode == "proxy" else "ghostbuster_api",
            },
        )
    
    def _detect_batch(self, texts: List[str]) -> List[DetectorResult]:
        """Efficient batch detection for proxy mode."""
        if self.mode != "proxy":
            return super()._detect_batch(texts)
        
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                ai_probs = probs[:, 0].cpu().numpy()
            
            for prob in ai_probs:
                results.append(DetectorResult(
                    score=float(prob),
                    raw_score=float(prob),
                    metadata={"mode": self.mode, "model": self.PROXY_MODEL},
                ))
        
        return results


class MageDetector(BaseEvalDetector):
    """
    MAGE detector (Longformer-based).
    
    Model: nealcly/detection-longformer
    Labels: 0 = machine-generated, 1 = human-written
    
    The returned score is always AI probability (p_machine) to preserve
    the higher-is-AI convention used throughout evaluation.
    """
    
    MODEL_NAME = "nealcly/detection-longformer"
    
    def __init__(
        self,
        device: Optional[str] = None,
        model_name: Optional[str] = None,
        max_length: int = 4096,
        score_mode: str = "ai_prob",  # ai_prob | human_prob | log_odds
        batch_size: int = 4,
        **kwargs,
    ):
        super().__init__(name="mage", device=device, batch_size=batch_size, **kwargs)
        self.model_name = model_name or self.MODEL_NAME
        self.max_length = max_length
        self.score_mode = score_mode
        self.model = None
        self.tokenizer = None
    
    @staticmethod
    def _preprocess(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\u00a0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    
    def load(self):
        """Load Longformer classifier."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        logger.info(f"Loading {self.model_name} for MAGE detector...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.score_mode not in {"ai_prob", "human_prob", "log_odds"}:
            logger.warning(f"Unknown mage score_mode '{self.score_mode}', falling back to ai_prob")
            self.score_mode = "ai_prob"
        
        self._loaded = True
        logger.info(f"✓ {self.name} loaded on {self.device}")
    
    def _detect_batch(self, texts: List[str]) -> List[DetectorResult]:
        """Batch detection with Longformer global attention mask."""
        results = []
        eps = 1e-8
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = [self._preprocess(t) for t in texts[i:i + self.batch_size]]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            global_attention_mask = torch.zeros_like(attention_mask)
            global_attention_mask[:, 0] = 1
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                )
                probs = torch.softmax(outputs.logits, dim=-1)
                p_machine = probs[:, 0]
                p_human = probs[:, 1]
            
            # Raw score for requested mode, but always return AI-prob for eval
            if self.score_mode == "human_prob":
                raw_scores = p_human
            elif self.score_mode == "log_odds":
                raw_scores = torch.log(p_human + eps) - torch.log(p_machine + eps)
            else:
                raw_scores = p_machine
            
            for raw, ai_prob, human_prob in zip(raw_scores, p_machine, p_human):
                results.append(
                    DetectorResult(
                        score=float(ai_prob.detach().cpu().item()),
                        raw_score=float(raw.detach().cpu().item()),
                        metadata={
                            "model": self.model_name,
                            "score_mode": self.score_mode,
                            "p_human": float(human_prob.detach().cpu().item()),
                        },
                    )
                )
        
        return results

    def _detect_single(self, text: str) -> DetectorResult:
        """Detect a single text (uses batch path for consistency)."""
        return self._detect_batch([text])[0]


class EnsembleDetector(BaseEvalDetector):
    """
    Ensemble detector combining multiple detector scores.
    
    Supports different aggregation strategies:
    - mean: Simple average of normalized scores
    - max: Maximum score (most conservative)
    - vote: Majority voting at threshold
    - weighted: Weighted average with custom weights
    """
    
    def __init__(
        self,
        detectors: List[BaseEvalDetector],
        strategy: str = "mean",
        weights: Optional[List[float]] = None,
        threshold: float = 0.5,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name="ensemble", device=device, **kwargs)
        self.detectors = detectors
        self.strategy = strategy
        self.weights = weights or [1.0] * len(detectors)
        self.threshold = threshold
        
        if len(self.weights) != len(detectors):
            raise ValueError("Number of weights must match number of detectors")
    
    def load(self):
        """Load all child detectors."""
        for detector in self.detectors:
            if not detector._loaded:
                detector.load()
        self._loaded = True
        logger.info(f"✓ Ensemble loaded with {len(self.detectors)} detectors")
    
    def _detect_single(self, text: str) -> DetectorResult:
        """Ensemble detection on single text."""
        scores = []
        metadata = {"individual_scores": {}}
        
        for detector in self.detectors:
            result = detector.detect(text)
            scores.append(result.score)
            metadata["individual_scores"][detector.name] = result.score
        
        # Aggregate scores
        if self.strategy == "mean":
            final_score = np.average(scores, weights=self.weights)
        elif self.strategy == "max":
            final_score = max(scores)
        elif self.strategy == "vote":
            votes = sum(1 for s in scores if s > self.threshold)
            final_score = votes / len(scores)
        elif self.strategy == "weighted":
            final_score = np.average(scores, weights=self.weights)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        metadata["strategy"] = self.strategy
        
        return DetectorResult(
            score=float(final_score),
            raw_score=float(np.mean(scores)),
            metadata=metadata,
        )


class WatermarkDetector(BaseEvalDetector):
    """
    Watermark-based detector (optional, watermark family).
    
    Detects watermarks embedded during text generation.
    Based on: Kirchenbauer et al., "A Watermark for Large Language Models" (2023)
    
    NOTE: This detector requires:
    1. Access to the watermark key/algorithm used during generation
    2. The watermarked model's vocabulary
    
    This implementation provides:
    - 'simulate': Simulated detection (for testing pipeline)
    - 'kirchenbauer': Full Kirchenbauer watermark detection (requires setup)
    
    ⚠️ LARGE DOWNLOAD WARNING: Full implementation requires:
    - Watermarked model weights (several GB)
    - Knowledge of watermark parameters (gamma, delta, seed)
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        mode: str = "simulate",
        gamma: float = 0.25,
        delta: float = 2.0,
        watermark_key: int = 42,
        vocab_size: int = 50257,  # GPT-2 default
        **kwargs,
    ):
        """
        Initialize watermark detector.
        
        Args:
            device: Compute device
            mode: Detection mode ('simulate' or 'kirchenbauer')
            gamma: Fraction of vocabulary in green list
            delta: Logit bias added to green list tokens
            watermark_key: Seed for watermark key generation
            vocab_size: Model vocabulary size
        """
        super().__init__(name="watermark", device=device, **kwargs)
        self.mode = mode
        self.gamma = gamma
        self.delta = delta
        self.watermark_key = watermark_key
        self.vocab_size = vocab_size
        
        self.tokenizer = None
        
        if mode == "kirchenbauer":
            logger.warning(
                "Kirchenbauer watermark detection requires watermark parameters "
                "that match the generation process. Results may be meaningless "
                "if parameters don't match."
            )
    
    def load(self):
        """Load tokenizer for watermark detection."""
        from transformers import AutoTokenizer
        
        if self.mode == "simulate":
            logger.info("Loading watermark detector in simulate mode (no real detection)")
            self._loaded = True
            return
        
        logger.info("Loading tokenizer for watermark detection...")
        
        # Use GPT-2 tokenizer as default
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        self._loaded = True
        logger.info(f"✓ Watermark detector loaded (mode={self.mode})")
    
    def _get_green_list(self, prev_token: int) -> set:
        """
        Get green list tokens based on previous token.
        
        Uses hash-based selection as in Kirchenbauer et al.
        """
        import hashlib
        
        # Hash previous token with watermark key
        hash_input = f"{self.watermark_key}_{prev_token}".encode()
        hash_value = int(hashlib.sha256(hash_input).hexdigest(), 16)
        
        # Deterministic random selection based on hash
        np.random.seed(hash_value % (2**32))
        
        n_green = int(self.vocab_size * self.gamma)
        green_list = set(np.random.choice(self.vocab_size, size=n_green, replace=False))
        
        return green_list
    
    def _compute_watermark_score(self, text: str) -> float:
        """
        Compute watermark detection score using z-statistic.
        
        Higher score = more likely watermarked.
        """
        if self.tokenizer is None:
            return 0.5  # Simulate mode
        
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) < 2:
            return 0.5
        
        # Count green tokens
        n_green = 0
        n_total = 0
        
        for i in range(1, len(tokens)):
            prev_token = tokens[i - 1]
            curr_token = tokens[i]
            green_list = self._get_green_list(prev_token)
            
            if curr_token in green_list:
                n_green += 1
            n_total += 1
        
        if n_total == 0:
            return 0.5
        
        # Compute z-statistic
        # Under null hypothesis (no watermark): E[n_green/n_total] = gamma
        observed_green_rate = n_green / n_total
        expected_green_rate = self.gamma
        
        # Z-score
        std = np.sqrt(self.gamma * (1 - self.gamma) / n_total)
        z_score = (observed_green_rate - expected_green_rate) / (std + 1e-10)
        
        # Convert to probability (sigmoid)
        score = 1 / (1 + np.exp(-z_score))
        
        return score
    
    def _detect_single(self, text: str) -> DetectorResult:
        """Detect watermark in text."""
        if self.mode == "simulate":
            # Simulate: return random score for testing
            # In real use, this should be replaced with actual detection
            score = 0.5 + 0.1 * (hash(text) % 10 - 5) / 5
            score = max(0, min(1, score))
            
            return DetectorResult(
                score=score,
                raw_score=score,
                metadata={
                    "mode": "simulate",
                    "warning": "Simulated scores for pipeline testing only",
                },
            )
        
        # Full Kirchenbauer detection
        score = self._compute_watermark_score(text)
        
        return DetectorResult(
            score=score,
            raw_score=score,
            metadata={
                "mode": self.mode,
                "gamma": self.gamma,
                "delta": self.delta,
            },
        )


# Registry of available detectors
DETECTOR_REGISTRY = {
    "roberta": RoBERTaOpenAIDetector,
    "roberta_openai": RoBERTaOpenAIDetector,
    "fast_detectgpt": FastDetectGPTDetector,
    "detectgpt": DetectGPTDetector,
    "binoculars": BinocularsDetector,
    "ghostbuster": GhostbusterDetector,
    "mage": MageDetector,
    "ensemble": EnsembleDetector,
    "watermark": WatermarkDetector,
}

# Score direction convention documentation
DETECTOR_CONVENTIONS = {
    "roberta": {"higher_is_ai": True, "score_range": (0, 1)},
    "fast_detectgpt": {"higher_is_ai": True, "score_range": (0, 1)},
    "detectgpt": {"higher_is_ai": True, "score_range": (0, 1)},
    "binoculars": {"higher_is_ai": True, "score_range": (0, 1), "note": "Internally lower=AI, normalized"},
    "ghostbuster": {"higher_is_ai": True, "score_range": (0, 1)},
    "mage": {"higher_is_ai": True, "score_range": (0, 1), "note": "Longformer-based MAGE detector"},
    "ensemble": {"higher_is_ai": True, "score_range": (0, 1)},
    "watermark": {"higher_is_ai": True, "score_range": (0, 1), "note": "Requires matching watermark params"},
}


def get_detector(name: str, **kwargs) -> BaseEvalDetector:
    """
    Get detector by name.
    
    Args:
        name: Detector name (roberta, fast_detectgpt, detectgpt, binoculars, ghostbuster)
        **kwargs: Detector-specific arguments
    
    Returns:
        Detector instance
    """
    if name not in DETECTOR_REGISTRY:
        raise ValueError(f"Unknown detector: {name}. Available: {list(DETECTOR_REGISTRY.keys())}")
    
    return DETECTOR_REGISTRY[name](**kwargs)


def load_detectors(
    names: List[str],
    device: Optional[str] = None,
    binoculars_full: bool = False,
    **kwargs,
) -> Dict[str, BaseEvalDetector]:
    """
    Load multiple detectors.
    
    Args:
        names: List of detector names
        device: Device to use
        binoculars_full: Use Falcon-7B for Binoculars (requires ~14GB VRAM)
        **kwargs: Additional arguments
    
    Returns:
        Dictionary mapping names to detector instances
    """
    detectors = {}
    device = device or get_device()
    
    for name in names:
        logger.info(f"Initializing detector: {name}")
        
        # Handle detector-specific kwargs
        det_kwargs = dict(kwargs)
        if name == "binoculars":
            det_kwargs["use_lightweight"] = not binoculars_full
            if binoculars_full:
                logger.info("  → Binoculars: Using Falcon-7B pair (paper-grade, ~14GB VRAM)")
            else:
                logger.info("  → Binoculars: Using GPT-2 pair (lightweight)")
        
        detectors[name] = get_detector(name, device=device, **det_kwargs)
    
    return detectors
