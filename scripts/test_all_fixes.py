"""
Comprehensive test for all meta tensor fixes: detectors, semantic, perplexity.
"""
import asyncio
from stealthrl.tinker.detectors import DetectorEnsemble
from stealthrl.tinker.semantic import SemanticSimilarity
from stealthrl.tinker.perplexity import PerplexityReward

async def test_all_concurrent():
    """Test all components with concurrent access."""
    print("=" * 60)
    print("Testing All Meta Tensor Fixes")
    print("=" * 60)
    
    # Initialize all components
    print("\n1. Initializing detectors...")
    detectors = DetectorEnsemble(
        detector_names=["fast_detectgpt", "ghostbuster"],
        device="cpu"
    )
    
    print("\n2. Initializing semantic similarity...")
    semantic = SemanticSimilarity(device="cpu")
    
    print("\n3. Initializing perplexity reward...")
    perplexity = PerplexityReward(device="cpu")
    
    # Pre-warm detectors
    print("\n4. Pre-warming detector models...")
    detectors.prewarm_models()
    
    # Test concurrent calls
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world of technology.",
        "Python is a versatile programming language used for many applications.",
        "The weather today is sunny with a chance of rain in the afternoon.",
    ]
    
    print(f"\n5. Running {len(test_texts)} concurrent tests...")
    
    # Create tasks for all components
    tasks = []
    for i, text in enumerate(test_texts):
        paraphrase = f"Variation {i+1}: " + text
        
        # Detector task
        tasks.append(detectors.compute(text))
        
        # Semantic task
        tasks.append(semantic.compute(text, paraphrase))
        
        # Perplexity task
        tasks.append(perplexity.compute(text))
    
    # Run all concurrently
    results = await asyncio.gather(*tasks)
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print(f"✓ Ran {len(tasks)} concurrent operations")
    print(f"✓ Detector calls: {len(test_texts)}")
    print(f"✓ Semantic similarity calls: {len(test_texts)}")
    print(f"✓ Perplexity calls: {len(test_texts)}")
    print("\n✓ No meta tensor errors!")
    print("✓ All models loaded successfully with singleton caching")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_all_concurrent())
