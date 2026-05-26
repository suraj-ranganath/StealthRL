#!/usr/bin/env python3
"""
Optional: Remote Fast-DetectGPT via Hugging Face Inference API

This is a drop-in replacement for DetectorEnsemble if you want to offload
to Hugging Face cloud infrastructure. Useful for:
- Scaling beyond local GPU
- Reducing local memory usage
- Production inference (not training)

Usage:
    # Replace DetectorEnsemble in reward.py:
    self.detector_ensemble = RemoteDetectorEnsemble(
        hf_token=os.getenv("HF_TOKEN"),
        use_remote=True  # Toggle between local and remote
    )
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)


class RemoteDetectorEnsemble:
    """
    Detector ensemble that offloads Fast-DetectGPT to HF Inference API.
    
    Falls back to local detection if API is unavailable.
    """
    
    def __init__(
        self,
        hf_token: Optional[str] = None,
        use_remote: bool = False,
        cache_dir: str = "/tmp/detector_cache",
    ):
        """
        Initialize remote detector ensemble.
        
        Args:
            hf_token: Hugging Face API token (required for remote)
            use_remote: Whether to use HF Inference API
            cache_dir: Directory for caching results locally
        """
        self.use_remote = use_remote
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
        if use_remote and not self.hf_token:
            logger.warning("⚠️  HF_TOKEN not set. Falling back to local detection.")
            self.use_remote = False
        
        # Local detector as fallback
        from stealthrl.tinker.detectors import DetectorEnsemble
        self.local_ensemble = DetectorEnsemble(
            detector_names=["roberta_openai", "fast_detectgpt"],
            detector_weights={"roberta_openai": 0.6, "fast_detectgpt": 0.4},
        )
        
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        if use_remote:
            self._init_remote()
    
    def _init_remote(self):
        """Initialize HF Inference API client."""
        try:
            from huggingface_hub import AsyncInferenceClient
            self.hf_client = AsyncInferenceClient(token=self.hf_token)
            logger.info("✅ HF Inference API initialized (async)")
        except ImportError:
            logger.error("❌ huggingface_hub not installed. Install with: pip install huggingface_hub")
            self.use_remote = False
    
    async def compute(
        self,
        text: str,
    ) -> Dict[str, Any]:
        """
        Compute detector probability for single text.
        
        Uses cache if available, then tries remote API, falls back to local.
        
        Args:
            text: Text to detect
        
        Returns:
            {"ensemble_prob": float} (probability text is AI-generated)
        """
        # Check cache
        text_hash = hash(text)
        if text_hash in self.cache:
            self.cache_hits += 1
            return self.cache[text_hash]
        
        self.cache_misses += 1
        
        if self.use_remote:
            try:
                result = await self._remote_detect(text)
                self.cache[text_hash] = result
                return result
            except Exception as e:
                logger.warning(f"Remote detection failed: {e}. Using local fallback.")
                # Fall back to local
        
        # Local detection
        result = await self.local_ensemble.compute(text)
        self.cache[text_hash] = result
        return result
    
    async def _remote_detect(self, text: str) -> Dict[str, Any]:
        """
        Call HF Inference API for Fast-DetectGPT detection.
        
        Note: This requires setting up a custom endpoint or using
        a deployed model. Standard model cards may not expose detection.
        """
        # Option 1: If you have a custom HF endpoint for detection
        try:
            # This would require deploying a detection model to HF
            response = await self.hf_client.text_classification(
                text,
                model="your-hf-username/gpt-neo-detector"  # Custom model
            )
            # Assuming model outputs [{"label": "ai", "score": 0.7}, ...]
            ai_score = next(
                (r["score"] for r in response if r["label"] == "ai"),
                0.5
            )
            return {"ensemble_prob": ai_score}
        except Exception as e:
            logger.error(f"HF API call failed: {e}")
            raise
    
    async def compute_batch(
        self,
        texts: list[str],
    ) -> Dict[str, list[float]]:
        """
        Compute detector probabilities for batch of texts.
        
        Uses batch API for efficiency.
        
        Args:
            texts: List of texts to detect
        
        Returns:
            {"ensemble_probs": [float, ...]} (list of AI probabilities)
        """
        results = []
        
        if self.use_remote:
            try:
                probs = await self._remote_detect_batch(texts)
                results = [{"ensemble_prob": p} for p in probs]
            except Exception as e:
                logger.warning(f"Remote batch detection failed: {e}. Using local fallback.")
                results = []
        
        if not results:  # Fall back to local
            batch_results = await self.local_ensemble.compute_batch(texts)
            results = batch_results
        
        ensemble_probs = [r["ensemble_prob"] for r in results]
        return {"ensemble_probs": ensemble_probs}
    
    async def _remote_detect_batch(self, texts: list[str]) -> list[float]:
        """Call HF Inference API with batch of texts."""
        # This would need a batch endpoint
        # For now, fall back to sequential calls
        probs = []
        for text in texts:
            result = await self._remote_detect(text)
            probs.append(result["ensemble_prob"])
        return probs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.1%}",
            "backend": "remote (HF API)" if self.use_remote else "local (GPU)",
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test():
        # Test local (default)
        detector = RemoteDetectorEnsemble(use_remote=False)
        
        test_text = "The quick brown fox jumps over the lazy dog."
        result = await detector.compute(test_text)
        print(f"Local detection: {result}")
        
        # Test batch
        batch_results = await detector.compute_batch(
            [test_text, test_text, "This is different text."]
        )
        print(f"Batch detection: {batch_results}")
        print(f"Stats: {detector.get_stats()}")
    
    asyncio.run(test())
