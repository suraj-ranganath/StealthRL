#!/usr/bin/env python3
"""
Test Fast-DetectGPT with different model sizes.
This allows you to compare gpt2, gpt-neo-2.7B, and falcon-7b.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from stealthrl.tinker.detectors import DetectorCache, FastDetectGPTDetector


async def test_fast_detectgpt_models():
    """Test Fast-DetectGPT with different model sizes."""
    
    # Create cache
    cache = DetectorCache(cache_path="cache/test_fast_detectgpt.db")
    
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
    
    # Models to test
    models = {
        "GPT-2 (500MB, fastest)": "gpt2",
        "GPT-Neo-2.7B (5.5GB, default)": "gpt-neo-2.7B",
        "Falcon-7B (14GB, best accuracy)": "falcon-7b",
    }
    
    print("=" * 80)
    print("FAST-DETECTGPT MODEL COMPARISON")
    print("=" * 80)
    print("\nTesting Fast-DetectGPT with 3 different model sizes.")
    print("\nNOTE: First run will download models:")
    print("  - GPT-2: 500MB")
    print("  - GPT-Neo-2.7B: 5.5GB (may take 5-10 minutes)")
    print("  - Falcon-7B: 14GB (may take 10-20 minutes)")
    print("\nYou can skip larger models with Ctrl+C if needed.")
    print("=" * 80)
    
    results = {}
    
    for display_name, model_name in models.items():
        print(f"\n{'=' * 80}")
        print(f"Testing: {display_name}")
        print("=" * 80)
        
        try:
            # Initialize detector
            detector = FastDetectGPTDetector(cache, model_name=model_name)
            
            # Test on human text
            print("Loading model and testing on human text...")
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
            
            results[display_name] = {
                "human_score": human_score,
                "ai_score": ai_score,
                "discrimination": discrimination,
                "success": True
            }
            
        except KeyboardInterrupt:
            print(f"\n  ⚠ Skipped by user")
            results[display_name] = {"success": False, "skipped": True}
            break
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results[display_name] = {"success": False, "error": str(e)}
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    successful = [name for name, r in results.items() if r.get("success")]
    
    if successful:
        print("\nDiscrimination scores (higher = better):")
        for name in successful:
            disc = results[name]["discrimination"]
            print(f"  {name:.<50} {disc:.3f}")
        
        # Recommendation
        best = max(successful, key=lambda n: results[n]["discrimination"])
        print(f"\n  ✅ Best performer: {best}")
        print(f"     (Discrimination: {results[best]['discrimination']:.3f})")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("\n1. For development/quick iterations:")
    print("   → Use gpt2 (500MB, fast)")
    print("\n2. For balanced accuracy and speed:")
    print("   → Use gpt-neo-2.7B (5.5GB, official default)")
    print("\n3. For best accuracy (workshop submission):")
    print("   → Use falcon-7b (14GB, best per paper)")
    print("\nUpdate configs/tinker_stealthrl.yaml:")
    print("  fast_detectgpt_model: \"gpt2\"  # or \"gpt-neo-2.7B\", \"falcon-7b\"")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(test_fast_detectgpt_models())
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        sys.exit(0)
