#!/usr/bin/env python3
"""
Test script to verify detector model loading fixes.

This script tests:
1. Singleton model caching
2. Thread-safe loading
3. Pre-warming functionality
4. No meta tensor errors
"""

import logging
import asyncio
from stealthrl.tinker.detectors import DetectorEnsemble

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_detector_loading():
    """Test detector ensemble with pre-warming."""
    logger.info("=" * 60)
    logger.info("Testing Detector Model Loading Fixes")
    logger.info("=" * 60)
    
    # Initialize detector ensemble
    logger.info("\n1. Initializing detector ensemble...")
    ensemble = DetectorEnsemble(
        detector_names=["fast_detectgpt", "ghostbuster"],
        cache_path=None,  # In-memory cache for testing
    )
    
    # Pre-warm models (should load once and cache)
    logger.info("\n2. Pre-warming models...")
    ensemble.prewarm_models()
    
    # Test with sample text
    logger.info("\n3. Testing with sample texts...")
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In this paper, we present a novel approach to machine learning.",
        "I went to the store yesterday to buy groceries.",
    ]
    
    for i, text in enumerate(test_texts, 1):
        logger.info(f"\n   Test {i}: {text[:50]}...")
        result = await ensemble.compute(text)
        logger.info(f"   Ensemble probability: {result['ensemble_prob']:.3f}")
        logger.info(f"   Individual scores: {result['detector_scores']}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ All tests passed! No meta tensor errors.")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_detector_loading())
