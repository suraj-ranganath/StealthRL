#!/usr/bin/env python3
"""
Test that local detectors are working correctly.
This will verify that all 3 detectors load and produce reasonable scores.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from stealthrl.tinker.detectors import (
    DetectorCache,
    FastDetectGPTDetector,
    GhostbusterDetector,
    BinocularsDetector
)


async def test_detectors():
    """Test all local detectors."""
    
    # Create cache
    cache = DetectorCache(cache_path="cache/test_detectors.db")
    
    # Test texts
    human_text = (
        "I really enjoyed reading this book last weekend. "
        "The characters felt genuine and the plot kept me engaged throughout. "
        "Would definitely recommend it to friends who enjoy this genre."
    )
    
    ai_text = (
        "The implementation demonstrates significant improvements in performance metrics "
        "across multiple evaluation benchmarks. The proposed methodology achieves "
        "state-of-the-art results through optimized parameter configurations."
    )
    
    # Initialize detectors
    detectors = {
        "Fast-DetectGPT (GPT-2)": FastDetectGPTDetector(
            cache,
            model_name="gpt2"
        ),
        "Fast-DetectGPT (GPT-Neo-2.7B)": FastDetectGPTDetector(
            cache,
            model_name="gpt-neo-2.7B"
        ),
        "Fast-DetectGPT (Falcon-7B)": FastDetectGPTDetector(
            cache,
            model_name="falcon-7b"
        ),
        "Ghostbuster": GhostbusterDetector(
            cache,
            model_name="roberta-base"  # Use fallback since specific detector may not exist
        ),
        "Binoculars": BinocularsDetector(
            cache,
            performer_model="gpt2",
            observer_model="gpt2-medium"
        )
    }
    
    print("=" * 80)
    print("LOCAL DETECTOR TEST")
    print("=" * 80)
    print("\nThis will test that all local detectors load and work correctly.")
    print("First run will download models (~10-15GB total) - may take 10-20 minutes.")
    print("\nModels being tested:")
    print("  - Fast-DetectGPT (GPT-2): 500MB")
    print("  - Fast-DetectGPT (GPT-Neo-2.7B): 5.5GB")
    print("  - Fast-DetectGPT (Falcon-7B): 14GB")
    print("  - Ghostbuster: 1.4GB")
    print("  - Binoculars: 1.9GB")
    print("=" * 80)
    
    results = {}
    
    for name, detector in detectors.items():
        print(f"\n{'=' * 80}")
        print(f"Testing: {name}")
        print("=" * 80)
        
        try:
            # Test on human text
            print("Testing on human-written text...")
            human_score = await detector.predict(human_text)
            print(f"  ✓ Human text score: {human_score:.3f}")
            
            # Test on AI text
            print("Testing on AI-generated text...")
            ai_score = await detector.predict(ai_text)
            print(f"  ✓ AI text score:    {ai_score:.3f}")
            
            # Check discrimination ability
            discrimination = abs(ai_score - human_score)
            print(f"\n  Discrimination: {discrimination:.3f}")
            
            # Evaluate performance
            if discrimination > 0.2:
                status = "✅ EXCELLENT (strong discrimination)"
            elif discrimination > 0.1:
                status = "✓ GOOD (moderate discrimination)"
            elif discrimination > 0.05:
                status = "⚠ WEAK (low discrimination)"
            else:
                status = "❌ POOR (no discrimination)"
            
            print(f"  Status: {status}")
            
            results[name] = {
                "human_score": human_score,
                "ai_score": ai_score,
                "discrimination": discrimination,
                "success": True
            }
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results[name] = {
                "success": False,
                "error": str(e)
            }
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results.values() if r.get("success"))
    total = len(results)
    
    print(f"\nDetectors tested: {successful}/{total}")
    
    for name, result in results.items():
        if result.get("success"):
            disc = result["discrimination"]
            if disc > 0.1:
                print(f"  ✅ {name}: Working (discrimination: {disc:.3f})")
            else:
                print(f"  ⚠ {name}: Working but weak (discrimination: {disc:.3f})")
        else:
            print(f"  ❌ {name}: Failed - {result['error']}")
    
    # Device info
    print("\nDevice Information:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  ✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  ✅ Apple Silicon (MPS) available")
        else:
            print("  ⚠ Running on CPU (slower, but works)")
    except:
        print("  ⚠ Could not determine device")
    
    # Cache info
    print("\nCache Information:")
    print(f"  Location: cache/test_detectors.db")
    print(f"  Note: Second run will be instant (cached results)")
    
    # Final recommendation
    print("\n" + "=" * 80)
    if successful == total:
        print("✅ ALL DETECTORS WORKING!")
        print("\nYou're ready to train! Your configuration uses:")
        print("  - Fast-DetectGPT (curvature-based)")
        print("  - Ghostbuster (classifier-based)")
        print("\nOptionally, you can enable Binoculars in your config for 3-detector ensemble.")
    elif successful > 0:
        print(f"⚠ PARTIAL SUCCESS ({successful}/{total} working)")
        print("\nYou can proceed with working detectors.")
        print("Check errors above to debug failing detectors.")
    else:
        print("❌ NO DETECTORS WORKING")
        print("\nCheck:")
        print("  1. Internet connection (to download models)")
        print("  2. Disk space (~3GB needed)")
        print("  3. PyTorch installation: pip install torch transformers")
    print("=" * 80)
    
    cache.close()
    
    return successful == total


if __name__ == "__main__":
    success = asyncio.run(test_detectors())
    sys.exit(0 if success else 1)
