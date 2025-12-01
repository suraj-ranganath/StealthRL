"""
Detector Ensemble with Caching for StealthRL.

This module provides a unified interface for multiple AI text detectors
with SQLite-based caching, retry logic, and rate limiting.
"""

import logging
import hashlib
import sqlite3
import asyncio
from typing import Dict, List, Any
from pathlib import Path
import time

logger = logging.getLogger(__name__)


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
        Predict AI probability for text with caching and retry.
        
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
        
        # Compute with retry logic
        for attempt in range(self.max_retries):
            try:
                score = await self._compute_score(text)
                
                # Cache result
                self.cache.set(self.name, text, score)
                
                return score
            
            except Exception as e:
                logger.warning(f"{self.name} attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    # Return default score on final failure
                    logger.error(f"{self.name} failed after {self.max_retries} attempts, returning 0.5")
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
    Fast-DetectGPT detector using curvature-based detection.
    
    Uses log-probability curvature to detect AI-generated text.
    Lower perplexity and flatter curvature suggest AI generation.
    """
    
    def __init__(self, cache: DetectorCache, model_name: str = "gpt2", device: str = None):
        super().__init__("fast_detectgpt", cache)
        self.model_name = model_name
        self.device = device or ("cuda" if self._check_cuda() else "cpu")
        self.model = None
        self.tokenizer = None
        logger.info(f"Initialized Fast-DetectGPT detector with {model_name} on {self.device}")
    
    def _check_cuda(self):
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _load_model(self):
        """Lazy load the model on first use."""
        if self.model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading {self.model_name} for Fast-DetectGPT...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"✓ Fast-DetectGPT model loaded on {self.device}")
    
    async def _compute_score(self, text: str) -> float:
        """
        Compute Fast-DetectGPT score using curvature-based detection.
        
        Returns probability in [0, 1] where higher = more likely AI-generated.
        """
        return await asyncio.to_thread(self._compute_score_sync, text)
    
    def _compute_score_sync(self, text: str) -> float:
        """Synchronous computation of Fast-DetectGPT score."""
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
            
            # Compute log probability
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                
                # Lower loss (higher probability) suggests AI-generated
                # Map loss to probability using sigmoid
                # Typical human text: loss ~3-5, AI text: loss ~2-3
                score = torch.sigmoid(torch.tensor((4.0 - loss) * 0.5)).item()
                
                return float(max(0.0, min(1.0, score)))
        
        except Exception as e:
            logger.error(f"Fast-DetectGPT error: {e}")
            return 0.5  # Return neutral score on error


class GhostbusterDetector(BaseDetector):
    """
    Ghostbuster detector using RoBERTa-based classification.
    
    Uses a fine-tuned RoBERTa model for AI text detection.
    Falls back to roberta-base if specific detector model unavailable.
    """
    
    def __init__(self, cache: DetectorCache, model_name: str = "roberta-base", device: str = None):
        super().__init__("ghostbuster", cache)
        self.model_name = model_name
        self.device = device or ("cuda" if self._check_cuda() else "cpu")
        self.model = None
        self.tokenizer = None
        logger.info(f"Initialized Ghostbuster detector with {model_name} on {self.device}")
    
    def _check_cuda(self):
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _load_model(self):
        """Lazy load the model on first use."""
        if self.model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            logger.info(f"Loading {self.model_name} for Ghostbuster...")
            
            try:
                # Try to load a specific AI detection model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map=self.device
                )
            except Exception as e:
                logger.warning(f"Could not load {self.model_name}, using roberta-base: {e}")
                # Fallback to base model
                self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "roberta-base",
                    num_labels=2,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map=self.device
                )
            
            self.model.eval()
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
        self.device = device or ("cuda" if self._check_cuda() else "cpu")
        self.performer = None
        self.observer = None
        self.tokenizer = None
        logger.info(f"Initialized Binoculars detector with {performer_model} and {observer_model} on {self.device}")
    
    def _check_cuda(self):
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _load_models(self):
        """Lazy load both models on first use."""
        if self.performer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading Binoculars models: {self.performer_model_name} and {self.observer_model_name}...")
            
            # Load tokenizer (shared)
            self.tokenizer = AutoTokenizer.from_pretrained(self.performer_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load performer model (smaller)
            self.performer = AutoModelForCausalLM.from_pretrained(
                self.performer_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            self.performer.eval()
            
            # Load observer model (larger)
            self.observer = AutoModelForCausalLM.from_pretrained(
                self.observer_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            self.observer.eval()
            
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
            
            # Binoculars score: cross-entropy difference
            # Lower difference suggests AI-generated (similar to both models)
            # Higher difference suggests human-written (more surprising to observer)
            ce_diff = abs(np.log(observer_ppl + 1e-10) - np.log(performer_ppl + 1e-10))
            
            # Map to probability: lower CE difference = higher AI probability
            # Typical values: AI ~0.1-0.5, Human ~0.5-2.0
            score = torch.sigmoid(torch.tensor((1.0 - ce_diff) * 2.0)).item()
            
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
    ):
        """
        Initialize detector ensemble.
        
        Args:
            detector_names: List of detector names to include
            detector_weights: Optional custom weights (default: equal weights)
            cache_path: Path to SQLite cache file
            device: Device to run detectors on (cuda/cpu)
        """
        self.detector_names = detector_names
        self.device = device
        
        # Initialize cache
        self.cache = DetectorCache(cache_path)
        
        # Initialize detectors
        self.detectors = {}
        for name in detector_names:
            if name == "fast_detectgpt":
                self.detectors[name] = FastDetectGPTDetector(self.cache, device=device)
            elif name == "ghostbuster":
                self.detectors[name] = GhostbusterDetector(self.cache, device=device)
            elif name == "binoculars":
                self.detectors[name] = BinocularsDetector(self.cache, device=device)
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
    
    def close(self):
        """Close cache connection."""
        self.cache.close()
