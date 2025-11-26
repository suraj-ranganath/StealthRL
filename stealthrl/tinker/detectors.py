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
    """Fast-DetectGPT detector wrapper."""
    
    def __init__(self, cache: DetectorCache):
        super().__init__("fast_detectgpt", cache)
        # Initialize Fast-DetectGPT model
        # This would load the actual model in production
        logger.info("Initialized Fast-DetectGPT detector")
    
    async def _compute_score(self, text: str) -> float:
        """Compute Fast-DetectGPT score using curvature-based detection."""
        # Placeholder: In production, this would call the actual Fast-DetectGPT model
        # For now, return a mock score
        await asyncio.sleep(0.1)  # Simulate computation
        
        # Mock score based on text characteristics
        score = min(1.0, max(0.0, 0.5 + (len(text) % 100) / 200.0))
        return score


class GhostbusterDetector(BaseDetector):
    """Ghostbuster (RoBERTa-based) detector wrapper."""
    
    def __init__(self, cache: DetectorCache):
        super().__init__("ghostbuster", cache)
        # Initialize Ghostbuster model
        logger.info("Initialized Ghostbuster detector")
    
    async def _compute_score(self, text: str) -> float:
        """Compute Ghostbuster score using RoBERTa classifier."""
        # Placeholder: In production, this would call the actual Ghostbuster model
        await asyncio.sleep(0.1)  # Simulate computation
        
        # Mock score
        score = min(1.0, max(0.0, 0.6 - (len(text) % 100) / 200.0))
        return score


class BinocularsDetector(BaseDetector):
    """Binoculars (paired-LM) detector wrapper."""
    
    def __init__(self, cache: DetectorCache):
        super().__init__("binoculars", cache)
        # Initialize Binoculars model
        logger.info("Initialized Binoculars detector")
    
    async def _compute_score(self, text: str) -> float:
        """Compute Binoculars score using paired language model approach."""
        # Placeholder: In production, this would call the actual Binoculars model
        await asyncio.sleep(0.1)  # Simulate computation
        
        # Mock score
        score = min(1.0, max(0.0, 0.55 + (hash(text) % 100) / 500.0))
        return score


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
    ):
        """
        Initialize detector ensemble.
        
        Args:
            detector_names: List of detector names to include
            detector_weights: Optional custom weights (default: equal weights)
            cache_path: Path to SQLite cache file
        """
        self.detector_names = detector_names
        
        # Initialize cache
        self.cache = DetectorCache(cache_path)
        
        # Initialize detectors
        self.detectors = {}
        for name in detector_names:
            if name == "fast_detectgpt":
                self.detectors[name] = FastDetectGPTDetector(self.cache)
            elif name == "ghostbuster":
                self.detectors[name] = GhostbusterDetector(self.cache)
            elif name == "binoculars":
                self.detectors[name] = BinocularsDetector(self.cache)
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
