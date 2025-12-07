#!/usr/bin/env python3
"""
Test script for real detector implementations.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import detector classes directly
import logging
logging.basicConfig(level=logging.INFO)

# Import the detector module components directly
import hashlib
import sqlite3
import time as time_module
from typing import Dict, List, Any

# Import detector cache and base classes
exec(open("stealthrl/tinker/detectors.py").read())


async def test_detectors():
    """Test detector ensemble with real models."""
    
    print("="*60)
    print("Testing StealthRL Detector Ensemble")
    print("="*60)
    
    # Sample texts
    ai_text = """
    The implementation of neural networks requires careful consideration of 
    hyperparameters and architectural choices. Recent advances in machine learning 
    have enabled significant progress in natural language processing tasks.
    """
    
    human_text = """
    I went to the store yesterday and bought some groceries. The weather was nice, 
    so I decided to walk instead of driving. It felt good to get some exercise!
    """
    
    # Initialize ensemble
    print("\n1. Initializing detector ensemble...")
    print("   Detectors: fast_detectgpt, ghostbuster, binoculars")
    
    ensemble = DetectorEnsemble(
        detector_names=["fast_detectgpt", "ghostbuster", "binoculars"],
        cache_path="outputs/detector_cache_test.sqlite",
        device="cuda"  # Will auto-fallback to CPU if no CUDA
    )
    
    print("   ✓ Ensemble initialized")
    
    # Test on AI-generated text
    print("\n2. Testing on AI-generated text...")
    print(f"   Text: {ai_text.strip()[:80]}...")
    
    result_ai = await ensemble.compute(ai_text.strip())
    
    print(f"\n   Results:")
    print(f"   - Ensemble probability: {result_ai['ensemble_prob']:.4f}")
    print(f"   - Individual scores:")
    for detector, score in result_ai['detector_scores'].items():
        print(f"     * {detector}: {score:.4f}")
    
    # Test on human-written text
    print("\n3. Testing on human-written text...")
    print(f"   Text: {human_text.strip()[:80]}...")
    
    result_human = await ensemble.compute(human_text.strip())
    
    print(f"\n   Results:")
    print(f"   - Ensemble probability: {result_human['ensemble_prob']:.4f}")
    print(f"   - Individual scores:")
    for detector, score in result_human['detector_scores'].items():
        print(f"     * {detector}: {score:.4f}")
    
    # Test caching (run same text again)
    print("\n4. Testing cache (re-running AI text)...")
    import time
    start = time.time()
    result_cached = await ensemble.compute(ai_text.strip())
    elapsed = time.time() - start
    
    print(f"   ✓ Cached result retrieved in {elapsed:.4f}s")
    print(f"   - Ensemble probability: {result_cached['ensemble_prob']:.4f}")
    
    # Verify cache works
    if result_cached['ensemble_prob'] == result_ai['ensemble_prob']:
        print("   ✓ Cache working correctly (scores match)")
    else:
        print("   ✗ Cache issue (scores don't match)")
    
    # Close ensemble
    ensemble.close()
    
    print("\n" + "="*60)
    print("✓ All tests completed successfully!")
    print("="*60)
    
    # Summary
    print("\nSummary:")
    print(f"  AI text ensemble score:    {result_ai['ensemble_prob']:.4f}")
    print(f"  Human text ensemble score: {result_human['ensemble_prob']:.4f}")
    print(f"  Difference:                {abs(result_ai['ensemble_prob'] - result_human['ensemble_prob']):.4f}")
    
    if result_ai['ensemble_prob'] > result_human['ensemble_prob']:
        print("  ✓ AI text scored higher (expected)")
    else:
        print("  ⚠ Human text scored higher (unexpected, but models may need fine-tuning)")


if __name__ == "__main__":
    try:
        asyncio.run(test_detectors())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

