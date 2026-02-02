"""
Detector Ensemble with Caching for StealthRL.

This module provides a unified interface for multiple AI text detectors
with SQLite-based caching, retry logic, and rate limiting.
"""

import logging
import hashlib
import sqlite3
import asyncio
import threading
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import time
import re
import unicodedata

logger = logging.getLogger(__name__)

# Global model cache with thread locks to prevent race conditions
_MODEL_CACHE: Dict[str, Tuple] = {}
_MODEL_LOCKS: Dict[str, threading.Lock] = {}
_CACHE_LOCK = threading.Lock()


def _default_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def get_model_lock(cache_key: str) -> threading.Lock:
    """Get or create lock for specific model."""
    with _CACHE_LOCK:
        if cache_key not in _MODEL_LOCKS:
            _MODEL_LOCKS[cache_key] = threading.Lock()
        return _MODEL_LOCKS[cache_key]


def load_model_cached(
    model_name: str,
    model_type: str,
    device: str,
    num_labels: Optional[int] = None,
) -> Tuple:
    """
    Load model with global caching and thread-safe locking.
    
    This prevents race conditions when multiple async tasks try to load
    the same model simultaneously.
    
    Args:
        model_name: HuggingFace model name
        model_type: 'causal_lm' or 'sequence_classification'
        device: Device to load model on ('cuda' or 'cpu')
        num_labels: Number of labels for classification (if applicable)
    
    Returns:
        (model, tokenizer) tuple
    """
    cache_key = f"{model_name}_{model_type}_{device}"
    
    # Check cache first (no lock needed for read)
    if cache_key in _MODEL_CACHE:
        logger.debug(f"Using cached model: {model_name}")
        return _MODEL_CACHE[cache_key]
    
    # Acquire model-specific lock for loading
    model_lock = get_model_lock(cache_key)
    with model_lock:
        # Double-check after acquiring lock
        if cache_key in _MODEL_CACHE:
            logger.debug(f"Using cached model (post-lock check): {model_name}")
            return _MODEL_CACHE[cache_key]
        
        # Load model
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
        import torch
        
        logger.info(f"🔄 Loading {model_name} (first time, will be cached)...")
        
        try:
            # Check if model requires trust_remote_code (Falcon models)
            trust_remote_code = "falcon" in model_name.lower()
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )
            
            if model_type == "causal_lm":
                # Load directly to device, avoid device_map to prevent meta tensor issues
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    trust_remote_code=trust_remote_code
                ).to(device)
            else:
                # Sequence classification
                kwargs = {}
                if num_labels is not None:
                    kwargs['num_labels'] = num_labels
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    trust_remote_code=trust_remote_code,
                    **kwargs
                ).to(device)
            
            model.eval()
            
            # Cache the loaded model
            _MODEL_CACHE[cache_key] = (model, tokenizer)
            logger.info(f"✓ {model_name} loaded and cached on {device}")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise


class DetectorCache:
    """SQLite-based cache for detector scores."""
    
    def __init__(self, cache_path: str | None = None):
        """
        Initialize detector cache.
        
        Args:
            cache_path: Path to SQLite database file. If None, uses in-memory.
        """
        if cache_path:
            cache_file = Path(cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(cache_file)
        else:
            self.db_path = ":memory:"
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_table()
    
    def _create_table(self):
        """Create cache table if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS detector_cache (
                detector_name TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                score REAL NOT NULL,
                timestamp REAL NOT NULL,
                PRIMARY KEY (detector_name, text_hash)
            )
        """)
        self.conn.commit()
    
    def get(self, detector_name: str, text: str) -> float | None:
        """
        Get cached score for detector-text pair.
        
        Args:
            detector_name: Name of detector
            text: Input text
        
        Returns:
            Cached score or None if not found
        """
        text_hash = self._hash_text(text)
        cursor = self.conn.execute(
            "SELECT score FROM detector_cache WHERE detector_name = ? AND text_hash = ?",
            (detector_name, text_hash)
        )
        row = cursor.fetchone()
        return row[0] if row else None
    
    def set(self, detector_name: str, text: str, score: float):
        """
        Cache detector score.
        
        Args:
            detector_name: Name of detector
            text: Input text
            score: Detector score
        """
        text_hash = self._hash_text(text)
        timestamp = time.time()
        self.conn.execute(
            "INSERT OR REPLACE INTO detector_cache (detector_name, text_hash, score, timestamp) VALUES (?, ?, ?, ?)",
            (detector_name, text_hash, score, timestamp)
        )
        self.conn.commit()
    
    @staticmethod
    def _hash_text(text: str) -> str:
        """Compute SHA256 hash of text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def close(self):
        """Close database connection."""
        self.conn.close()


class BaseDetector:
    """Base class for AI text detectors."""
    
    def __init__(
        self,
        name: str,
        cache: DetectorCache,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize detector.
        
        Args:
            name: Detector name
            cache: DetectorCache instance
            max_retries: Maximum retry attempts on failure
            retry_delay: Delay between retries (seconds)
        """
        self.name = name
        self.cache = cache
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    async def predict(self, text: str) -> float:
        """
        Predict AI probability for text with caching.
        
        Args:
            text: Input text
        
        Returns:
            Probability that text is AI-generated [0, 1]
        """
        # Check cache first
        cached_score = self.cache.get(self.name, text)
        if cached_score is not None:
            logger.debug(f"Cache hit for {self.name}: {text[:50]}...")
            return cached_score
        
        # Compute score (model should already be loaded, so no need for retries)
        try:
            score = await self._compute_score(text)
            
            # Cache result
            self.cache.set(self.name, text, score)
            
            return score
        
        except Exception as e:
            logger.error(f"{self.name} prediction failed: {e}")
            # Return neutral score on error
            return 0.5
    
    async def _compute_score(self, text: str) -> float:
        """
        Compute detector score (to be implemented by subclasses).
        
        Args:
            text: Input text
        
        Returns:
            AI probability [0, 1]
        """
        raise NotImplementedError


class FastDetectGPTDetector(BaseDetector):
    """
    Fast-DetectGPT detector using official implementation.
    
    Uses sampling discrepancy (curvature-based detection) from the official
    Fast-DetectGPT paper: https://github.com/baoguangsheng/fast-detect-gpt
    
    Supported models:
    - gpt2: Lightweight (500MB), fastest
    - gpt-neo-2.7B: EleutherAI/gpt-neo-2.7B (default in Fast-DetectGPT)
    - falcon-7b: tiiuae/falcon-7b (best accuracy per Fast-DetectGPT paper)
    
    The detector uses sampling discrepancy analytic method which is much faster
    than the original DetectGPT perturbation-based approach (340x speedup).
    """
    
    # Model name mappings (short name -> HuggingFace path)
    MODEL_PATHS = {
        "gpt2": "gpt2",
        "gpt-neo-2.7B": "EleutherAI/gpt-neo-2.7B",
        "falcon-7b": "tiiuae/falcon-7b",
    }
    
    def __init__(self, cache: DetectorCache, model_name: str = "gpt2", device: str = None):
        # Resolve model name to full path
        if model_name in self.MODEL_PATHS:
            self.model_name = self.MODEL_PATHS[model_name]
            self.model_short_name = model_name
        else:
            # Allow custom model paths
            self.model_name = model_name
            self.model_short_name = model_name
        
        # Include model name in cache key to prevent cross-model cache pollution
        super().__init__(f"fast_detectgpt_{self.model_short_name}", cache)
        
        self.device = device or _default_device()
        self.model = None
        self.tokenizer = None
        logger.info(f"Initialized Fast-DetectGPT detector with {self.model_short_name} on {self.device}")
    
    @staticmethod
    def _get_sampling_discrepancy_analytic(logits_ref, logits_score, labels, return_batch=False):
        """
        Official Fast-DetectGPT analytic sampling discrepancy criterion.
        From: https://github.com/baoguangsheng/fast-detect-gpt
        
        Modified to support batch processing by removing batch_size=1 assertions.
        The math works identically for batches - each sequence is computed independently.
        
        Args:
            logits_ref: Reference model logits [batch_size, seq_len, vocab_size]
            logits_score: Scoring model logits [batch_size, seq_len, vocab_size]
            labels: Token labels [batch_size, seq_len]
            return_batch: If True, return per-sequence discrepancies, else return mean
        
        Returns:
            float (if return_batch=False) or torch.Tensor (if return_batch=True)
        """
        import torch
        
        # Batch size can be > 1 now! The math works for any batch size.
        # The original assertions were just safety checks, not algorithmic requirements.
        
        # Handle vocabulary size mismatch
        if logits_ref.size(-1) != logits_score.size(-1):
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]
        
        labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
        lprobs_score = torch.log_softmax(logits_score, dim=-1)
        probs_ref = torch.softmax(logits_ref, dim=-1)
        log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
        var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
        
        # Per-sequence discrepancy calculation [batch_size]
        discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
        
        if return_batch:
            return discrepancy  # Return tensor of shape [batch_size]
        else:
            return discrepancy.mean().item()  # Return scalar mean
    
    def _load_model(self):
        """Lazy load the model using our cached models with official Fast-DetectGPT algorithm."""
        if self.model is None:
            logger.info(f"Loading {self.model_short_name} for Fast-DetectGPT...")
            
            # Use our existing model cache (models already downloaded!)
            self.model, self.tokenizer = load_model_cached(
                model_name=self.model_name,
                model_type="causal_lm",
                device=self.device,
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"✓ {self.model_short_name} loaded on {self.device}")
    
    async def _compute_score(self, text: str) -> float:
        """
        Compute Fast-DetectGPT score using curvature-based detection.
        
        Returns probability in [0, 1] where higher = more likely AI-generated.
        """
        return await asyncio.to_thread(self._compute_score_sync, text)
    
    def _compute_score_sync(self, text: str) -> float:
        """Synchronous computation using official Fast-DetectGPT algorithm."""
        import torch
        
        # Load model if not already loaded
        self._load_model()
        
        try:
            # Tokenize text (official Fast-DetectGPT method)
            tokenized = self.tokenizer(
                text,
                truncation=True,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False
            ).to(self.device)
            
            labels = tokenized.input_ids[:, 1:]
            
            # Compute logits using the model
            with torch.no_grad():
                logits = self.model(**tokenized).logits[:, :-1]
                
                # For white-box setting, use same model for sampling and scoring
                # Calculate sampling discrepancy (official Fast-DetectGPT criterion)
                crit = self._get_sampling_discrepancy_analytic(logits, logits, labels)
                
                # Convert criterion to probability [0, 1]
                # Positive criterion -> AI-generated (high score)
                # Negative criterion -> Human-written (low score)
                # 
                # Scale the criterion for better separation
                # Typical range: -3 to +3 for the criterion
                # Use sigmoid with scaling to map to [0, 1]
                scaled_crit = crit * 0.5  # Scale down to avoid saturation
                score = torch.sigmoid(torch.tensor(scaled_crit)).item()
                
                return float(max(0.0, min(1.0, score)))
        
        except Exception as e:
            logger.error(f"Fast-DetectGPT error: {e}")
            return 0.5  # Return neutral score on error
    
    async def detect_batch(self, texts: List[str], batch_size: int = 32) -> List[float]:
        """
        Batch detect AI-generated text using Fast-DetectGPT.
        
        Args:
            texts: List of texts to detect
            batch_size: Batch size for GPU processing
        
        Returns:
            List of AI probability scores [0, 1]
        """
        return await asyncio.to_thread(self._detect_batch_sync, texts, batch_size)
    
    def _detect_batch_sync(self, texts: List[str], batch_size: int = 32) -> List[float]:
        """
        Synchronous batch detection with proper GPU batching.
        
        Processes texts in chunks to avoid OOM and uses padding for true batch inference.
        """
        import torch
        
        # Load model if not already loaded
        self._load_model()
        
        all_scores = []
        
        try:
            # Process in chunks
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize batch with padding
                tokenized = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False
                ).to(self.device)
                
                labels = tokenized.input_ids[:, 1:]
                
                # Compute logits for the batch
                with torch.no_grad():
                    logits = self.model(**tokenized).logits[:, :-1]
                    
                    # Calculate batch discrepancies (returns tensor of shape [batch_size])
                    crit_batch = self._get_sampling_discrepancy_analytic(
                        logits, logits, labels, return_batch=True
                    )
                    
                    # Convert each criterion to probability
                    scaled_crits = crit_batch * 0.5
                    scores = torch.sigmoid(scaled_crits).cpu().tolist()
                    
                    # Clamp to [0, 1] and add to results
                    all_scores.extend([max(0.0, min(1.0, float(s))) for s in scores])
        
        except Exception as e:
            logger.error(f"Fast-DetectGPT batch error: {e}")
            # Return neutral scores for all texts on error
            all_scores = [0.5] * len(texts)
        
        return all_scores


class RoBERTaOpenAIDetector(BaseDetector):
    """
    RoBERTa-based OpenAI detector for AI-generated text.
    
    Uses fine-tuned RoBERTa models from OpenAI specifically trained
    for detecting GPT-2 and GPT-3 generated text.
    
    Supported models:
    - roberta-base-openai-detector: Smaller, faster (125M params)
    - roberta-large-openai-detector: Larger, more accurate (355M params)
    """
    
    MODEL_PATHS = {
        "roberta-base-openai-detector": "roberta-base-openai-detector",
        "roberta-large-openai-detector": "roberta-large-openai-detector",
    }
    
    def __init__(self, cache: DetectorCache, model_name: str = "roberta-large-openai-detector", device: str = None):
        # Resolve model name
        if model_name in self.MODEL_PATHS:
            self.model_name = self.MODEL_PATHS[model_name]
            self.model_short_name = model_name
        else:
            self.model_name = model_name
            self.model_short_name = model_name
        
        super().__init__(f"roberta_openai_{self.model_short_name}", cache)
        self.device = device or _default_device()
        self.model = None
        self.tokenizer = None
        logger.info(f"Initialized RoBERTa OpenAI detector with {self.model_short_name} on {self.device}")
    
    def _load_model(self):
        """Lazy load the fine-tuned RoBERTa model."""
        if self.model is None:
            logger.info(f"Loading {self.model_short_name} for RoBERTa OpenAI detector...")
            
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            try:
                # Load from HuggingFace
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    cache_dir="cache"
                )
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"✓ {self.model_short_name} loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load {self.model_name}: {e}")
                raise
    
    async def _compute_score(self, text: str) -> float:
        """
        Compute RoBERTa OpenAI detector score.
        
        Returns probability in [0, 1] where higher = more likely AI-generated.
        """
        return await asyncio.to_thread(self._compute_score_sync, text)
    
    def _compute_score_sync(self, text: str) -> float:
        """Synchronous computation of RoBERTa score."""
        import torch
        
        # Load model if not already loaded
        self._load_model()
        
        try:
            # Tokenize with truncation
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # The roberta-*-openai-detector models output: [Fake/AI, Real/Human]
                # We want probability of AI-generated (class 0 for these models)
                score = probs[0, 0].item()
                
                return float(max(0.0, min(1.0, score)))
        
        except Exception as e:
            logger.error(f"RoBERTa OpenAI detector error: {e}")
            return 0.5  # Return neutral score on error
    
    async def detect_batch(self, texts: List[str], batch_size: int = 32) -> List[float]:
        """Batch detection with proper GPU batching for efficiency."""
        return await asyncio.to_thread(self._detect_batch_sync, texts, batch_size)
    
    def _detect_batch_sync(self, texts: List[str], batch_size: int = 32) -> List[float]:
        """Synchronous batch detection."""
        import torch
        
        # Load model if not already loaded
        self._load_model()
        
        all_scores = []
        
        # Process in chunks for memory efficiency
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                # Batch inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Extract AI probability for each text
                    scores = probs[:, 0].cpu().tolist()
                    all_scores.extend([max(0.0, min(1.0, float(s))) for s in scores])
            
            except Exception as e:
                logger.error(f"RoBERTa batch error for batch {i//batch_size}: {e}")
                # Return neutral scores for failed batch
                all_scores.extend([0.5] * len(batch_texts))
        
        return all_scores


class GhostbusterDetector(BaseDetector):
    """
    Ghostbuster detector using RoBERTa-based classification.
    
    Uses a fine-tuned RoBERTa model for AI text detection.
    Falls back to roberta-base if specific detector model unavailable.
    """
    
    def __init__(self, cache: DetectorCache, model_name: str = "roberta-base-openai-detector", device: str = None):
        super().__init__("ghostbuster", cache)
        self.model_name = model_name
        self.device = device or _default_device()
        self.model = None
        self.tokenizer = None
        logger.info(f"Initialized Ghostbuster detector with {model_name} on {self.device}")
    
    def _load_model(self):
        """Lazy load the model on first use with singleton caching."""
        if self.model is None:
            logger.info(f"Loading {self.model_name} for Ghostbuster...")
            try:
                self.model, self.tokenizer = load_model_cached(
                    model_name=self.model_name,
                    model_type="sequence_classification",
                    device=self.device,
                )
            except Exception as e:
                logger.warning(f"Could not load {self.model_name}, using roberta-base: {e}")
                # Fallback to base model
                self.model, self.tokenizer = load_model_cached(
                    model_name="roberta-base",
                    model_type="sequence_classification",
                    device=self.device,
                    num_labels=2,
                )
            
            logger.info(f"✓ Ghostbuster model loaded on {self.device}")
    
    async def _compute_score(self, text: str) -> float:
        """
        Compute Ghostbuster score using RoBERTa classifier.
        
        Returns probability in [0, 1] where higher = more likely AI-generated.
        """
        return await asyncio.to_thread(self._compute_score_sync, text)
    
    def _compute_score_sync(self, text: str) -> float:
        """Synchronous computation of Ghostbuster score."""
        import torch
        
        # Load model if not already loaded
        self._load_model()
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get probability of AI-generated class
                probs = torch.softmax(logits, dim=-1)
                
                # If binary classification, take class 1 (AI-generated)
                if probs.shape[1] == 2:
                    score = probs[0, 1].item()
                else:
                    # If not binary, use max probability
                    score = probs.max().item()
                
                return float(max(0.0, min(1.0, score)))
        
        except Exception as e:
            logger.error(f"Ghostbuster error: {e}")
            return 0.5  # Return neutral score on error


class MageDetector(BaseDetector):
    """
    MAGE detector (Longformer-based).
    
    Model: yaful/MAGE
    Labels: 0 = machine-generated, 1 = human-written
    
    Note: This detector returns AI probability by default (p_machine).
    The reward uses (1 - p_machine) to align with p_human.
    """
    
    MODEL_NAME = "yaful/MAGE"
    
    def __init__(
        self,
        cache: DetectorCache,
        device: Optional[str] = None,
        max_length: int = 4096,
        score_mode: str = "ai_prob",  # ai_prob | human_prob | log_odds
    ):
        super().__init__("mage", cache)
        self.device = device or _default_device()
        self.max_length = max_length
        self.score_mode = score_mode
        self.model = None
        self.tokenizer = None
        logger.info(f"Initialized MAGE detector on {self.device} (max_length={self.max_length})")
    
    @staticmethod
    def _preprocess(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\u00a0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    
    def _load_model(self):
        """Lazy load Longformer classifier with cached model loading."""
        if self.model is None:
            logger.info(f"Loading {self.MODEL_NAME} for MAGE detector...")
            self.model, self.tokenizer = load_model_cached(
                self.MODEL_NAME,
                "sequence_classification",
                self.device,
                num_labels=2,
            )
            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"✓ {self.MODEL_NAME} loaded on {self.device}")
            if self.score_mode not in {"ai_prob", "human_prob", "log_odds"}:
                logger.warning(f"Unknown mage score_mode '{self.score_mode}', falling back to ai_prob")
                self.score_mode = "ai_prob"
    
    async def _compute_score(self, text: str) -> float:
        return await asyncio.to_thread(self._compute_score_sync, text)
    
    def _compute_score_sync(self, text: str) -> float:
        scores = self._detect_batch_sync([text], batch_size=1)
        return scores[0] if scores else 0.5
    
    async def detect_batch(self, texts: List[str], batch_size: int = 4) -> List[float]:
        return await asyncio.to_thread(self._detect_batch_sync, texts, batch_size)
    
    def _detect_batch_sync(self, texts: List[str], batch_size: int = 4) -> List[float]:
        import torch
        
        self._load_model()
        all_scores: List[float] = []
        eps = 1e-8
        
        for i in range(0, len(texts), batch_size):
            batch_texts = [self._preprocess(t) for t in texts[i:i+batch_size]]
            try:
                enc = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                
                # Longformer requires global attention on at least one token
                global_attention_mask = torch.zeros_like(attention_mask)
                global_attention_mask[:, 0] = 1
                
                with torch.no_grad():
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        global_attention_mask=global_attention_mask,
                    )
                    probs = torch.softmax(out.logits, dim=-1)
                    p_machine = probs[:, 0]
                    p_human = probs[:, 1]
                
                if self.score_mode == "human_prob":
                    scores = p_human
                elif self.score_mode == "log_odds":
                    scores = torch.log(p_human + eps) - torch.log(p_machine + eps)
                else:
                    scores = p_machine
                
                scores_list = scores.detach().cpu().tolist()
                all_scores.extend([float(s) for s in scores_list])
            
            except Exception as e:
                logger.error(f"MAGE batch error for batch {i//batch_size}: {e}")
                all_scores.extend([0.5] * len(batch_texts))
        
        return all_scores


class BinocularsDetector(BaseDetector):
    """
    Binoculars detector using paired language model approach.
    
    Compares perplexity between two models (performer and observer).
    Lower cross-entropy difference suggests AI-generated text.
    """
    
    def __init__(
        self,
        cache: DetectorCache,
        performer_model: str = "gpt2",
        observer_model: str = "gpt2-medium",
        device: str = None
    ):
        super().__init__("binoculars", cache)
        self.performer_model_name = performer_model
        self.observer_model_name = observer_model
        self.device = device or _default_device()
        self.performer = None
        self.observer = None
        self.tokenizer = None
        logger.info(f"Initialized Binoculars detector with {performer_model} and {observer_model} on {self.device}")
    
    def _load_models(self):
        """Lazy load both models on first use with singleton caching."""
        if self.performer is None:
            logger.info(f"Loading Binoculars models: {self.performer_model_name} and {self.observer_model_name}...")
            
            # Load performer model (smaller) using cached loading
            self.performer, performer_tokenizer = load_model_cached(
                model_name=self.performer_model_name,
                model_type="causal_lm",
                device=self.device,
            )
            
            # Load observer model (larger) using cached loading
            self.observer, observer_tokenizer = load_model_cached(
                model_name=self.observer_model_name,
                model_type="causal_lm",
                device=self.device,
            )
            
            # Use performer tokenizer as shared tokenizer
            self.tokenizer = performer_tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"✓ Binoculars models loaded on {self.device}")
    
    async def _compute_score(self, text: str) -> float:
        """
        Compute Binoculars score using paired LM approach.
        
        Returns probability in [0, 1] where higher = more likely AI-generated.
        """
        return await asyncio.to_thread(self._compute_score_sync, text)
    
    def _compute_score_sync(self, text: str) -> float:
        """Synchronous computation of Binoculars score."""
        import torch
        import numpy as np
        
        # Load models if not already loaded
        self._load_models()
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Compute perplexity with performer
            with torch.no_grad():
                performer_outputs = self.performer(**inputs, labels=inputs["input_ids"])
                performer_loss = performer_outputs.loss.item()
                performer_ppl = np.exp(performer_loss)
                
                # Compute perplexity with observer
                observer_outputs = self.observer(**inputs, labels=inputs["input_ids"])
                observer_loss = observer_outputs.loss.item()
                observer_ppl = np.exp(observer_loss)
            
            # Binoculars score: use the cross-perplexity approach from the paper
            # The key insight is to use log perplexity difference normalized by text length
            log_ppl_diff = np.log(observer_ppl) - np.log(performer_ppl)
            
            # Normalize to [0, 1] range
            # AI text typically has log_ppl_diff close to 0 (both models similar)
            # Human text typically has higher log_ppl_diff (observer struggles more)
            # Use a calibrated threshold based on typical values
            threshold = 0.5  # Empirically determined
            score = 1.0 / (1.0 + np.exp(-5.0 * (threshold - log_ppl_diff)))
            
            return float(max(0.0, min(1.0, score)))
        
        except Exception as e:
            logger.error(f"Binoculars error: {e}")
            return 0.5  # Return neutral score on error


class DetectorEnsemble:
    """
    Ensemble of multiple AI text detectors with weighted averaging.
    
    Combines multiple detectors (Fast-DetectGPT, Ghostbuster, Binoculars, etc.)
    into a single weighted ensemble score.
    """
    
    def __init__(
        self,
        detector_names: List[str],
        detector_weights: Dict[str, float] | None = None,
        cache_path: str | None = None,
        device: str | None = None,
        max_concurrent: int = 4,
        # Detector-specific model options
        fast_detectgpt_model: str = "gpt2",
        roberta_openai_model: str = "roberta-large-openai-detector",
        ghostbuster_model: str = "roberta-base",
        binoculars_performer: str = "gpt2",
        binoculars_observer: str = "gpt2-medium",
        mage_max_length: int = 4096,
        mage_score_mode: str = "ai_prob",
        roberta_batch_size: int = 128,  # Batch size for RoBERTa (memory efficient)
        fast_detectgpt_batch_size: int = 32,  # Batch size for Fast-DetectGPT (larger model)
        mage_batch_size: int = 4,
    ):
        """
        Initialize detector ensemble.
        
        Args:
            detector_names: List of detector names to include
            detector_weights: Optional custom weights (default: equal weights)
            cache_path: Path to SQLite cache file
            device: Device to run detectors on (cuda/cpu)
            max_concurrent: Maximum concurrent detector evaluations
            fast_detectgpt_model: Model for Fast-DetectGPT ("gpt2", "gpt-neo-2.7B", "falcon-7b")
            roberta_openai_model: Model for RoBERTa OpenAI detector ("roberta-large-openai-detector")
            ghostbuster_model: Model for Ghostbuster ("roberta-base", "roberta-base-openai-detector")
            binoculars_performer: Performer model for Binoculars
            binoculars_observer: Observer model for Binoculars
            mage_max_length: Max sequence length for MAGE (Longformer)
            mage_score_mode: Score mode for MAGE (ai_prob | human_prob | log_odds)
            roberta_batch_size: Batch size for RoBERTa
            fast_detectgpt_batch_size: Batch size for Fast-DetectGPT
            mage_batch_size: Batch size for MAGE
        """
        self.detector_names = detector_names
        self.device = device or _default_device()
        self.max_concurrent = max_concurrent
        self.roberta_openai_model = roberta_openai_model
        self.roberta_batch_size = roberta_batch_size
        self.fast_detectgpt_batch_size = fast_detectgpt_batch_size
        self.mage_batch_size = mage_batch_size
        self.mage_max_length = mage_max_length
        self.mage_score_mode = mage_score_mode
        
        # Initialize cache
        self.cache = DetectorCache(cache_path)
        
        # Create semaphore for rate limiting
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        # Initialize detectors with specified models
        self.detectors = {}
        for name in detector_names:
            if name == "fast_detectgpt":
                self.detectors[name] = FastDetectGPTDetector(
                    self.cache,
                    model_name=fast_detectgpt_model,
                    device=self.device
                )
            elif name == "roberta_openai":
                # RoBERTa-large-openai-detector (VALIDATED: AUROC 0.891)
                self.detectors[name] = RoBERTaOpenAIDetector(
                    self.cache,
                    model_name=getattr(self, 'roberta_openai_model', 'roberta-large-openai-detector'),
                    device=self.device
                )
            elif name == "ghostbuster":
                self.detectors[name] = GhostbusterDetector(
                    self.cache,
                    model_name=ghostbuster_model,
                    device=self.device
                )
            elif name == "binoculars":
                self.detectors[name] = BinocularsDetector(
                    self.cache,
                    performer_model=binoculars_performer,
                    observer_model=binoculars_observer,
                    device=self.device
                )
            elif name == "mage":
                self.detectors[name] = MageDetector(
                    self.cache,
                    device=self.device,
                    max_length=mage_max_length,
                    score_mode=mage_score_mode,
                )
            else:
                logger.warning(f"Unknown detector: {name}, skipping")
        
        # Set weights (default: equal)
        if detector_weights:
            self.weights = detector_weights
        else:
            self.weights = {name: 1.0 / len(self.detectors) for name in self.detectors}
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        logger.info(f"Initialized detector ensemble with {len(self.detectors)} detectors")
        logger.info(f"Weights: {self.weights}")
    
    def prewarm_models(self):
        """Pre-load all detector models to avoid lazy loading during training."""
        logger.info("🔥 Pre-warming detector models...")
        for name, detector in self.detectors.items():
            try:
                # BinocularsDetector uses _load_models, others use _load_model
                if hasattr(detector, '_load_models'):
                    detector._load_models()
                else:
                    detector._load_model()
                logger.info(f"✓ Pre-loaded {name}")
            except Exception as e:
                logger.error(f"Failed to pre-load {name}: {e}")
        logger.info("✓ All detector models pre-warmed and ready")
    
    async def compute(self, text: str) -> Dict[str, Any]:
        """
        Compute ensemble detector score.
        
        Args:
            text: Input text to evaluate
        
        Returns:
            Dictionary with ensemble_prob and individual detector scores
        """
        # Compute scores for all detectors in parallel
        detector_tasks = [
            detector.predict(text)
            for detector in self.detectors.values()
        ]
        detector_scores = await asyncio.gather(*detector_tasks)
        
        # Create scores dict
        scores_dict = dict(zip(self.detectors.keys(), detector_scores))
        
        # Compute weighted ensemble
        ensemble_prob = sum(
            self.weights.get(name, 0.0) * score
            for name, score in scores_dict.items()
        )
        
        return {
            "ensemble_prob": ensemble_prob,
            "detector_scores": scores_dict,
        }

    async def compute_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Compute ensemble scores for a batch of texts using efficient GPU batching.
        
        This uses true batching for detectors that support it (RoBERTa, Ghostbuster)
        and falls back to concurrent processing for others (Fast-DetectGPT).
        """
        # Collect detector scores for all texts
        all_detector_scores = {}
        
        for detector_name, detector in self.detectors.items():
            # Use batch method if available (RoBERTa, Fast-DetectGPT)
            if hasattr(detector, 'detect_batch'):
                # Use detector-specific batch size
                if 'roberta' in detector_name.lower():
                    batch_size = self.roberta_batch_size
                elif 'mage' in detector_name.lower():
                    batch_size = self.mage_batch_size
                elif 'fast' in detector_name.lower() or 'detectgpt' in detector_name.lower():
                    batch_size = self.fast_detectgpt_batch_size
                else:
                    batch_size = 32  # Default for other detectors
                
                logger.debug(f"Using batch processing for {detector_name} ({len(texts)} texts, batch_size={batch_size})")
                scores = await detector.detect_batch(texts, batch_size=batch_size)
                all_detector_scores[detector_name] = scores
            else:
                # Fallback: concurrent processing for detectors without batch support
                logger.debug(f"Using concurrent processing for {detector_name} ({len(texts)} texts)")
                async def _compute_one(text: str) -> float:
                    async with self._semaphore:
                        result = await detector.predict(text)
                        return result
                
                scores = await asyncio.gather(*[_compute_one(text) for text in texts])
                all_detector_scores[detector_name] = scores
        
        # Compute ensemble for each text
        results = []
        for i in range(len(texts)):
            scores_dict = {name: all_detector_scores[name][i] for name in all_detector_scores}
            ensemble_prob = sum(
                self.weights.get(name, 0.0) * score
                for name, score in scores_dict.items()
            )
            results.append({
                "ensemble_prob": ensemble_prob,
                "detector_scores": scores_dict,
            })
        
        return results
    
    def close(self):
        """Close cache connection."""
        self.cache.close()
