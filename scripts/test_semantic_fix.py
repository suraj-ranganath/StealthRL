"""
Quick test to verify semantic similarity threading fix.
"""
import asyncio
from stealthrl.tinker.semantic import SemanticSimilarity

async def test_concurrent_semantic():
    """Test that multiple concurrent semantic similarity calls work."""
    print("Testing concurrent semantic similarity calls...")
    
    # Create semantic similarity instance
    semantic = SemanticSimilarity(model_name="intfloat/e5-large-v2", device="cpu")
    
    # Test with concurrent calls (simulating parallel rollouts)
    texts = [
        ("The cat sat on the mat.", "A feline rested on the rug."),
        ("Machine learning is fascinating.", "AI and deep learning are interesting."),
        ("The weather is nice today.", "It's a beautiful day outside."),
        ("Python is a programming language.", "Python is used for coding."),
    ]
    
    print(f"Running {len(texts)} concurrent similarity computations...")
    
    # Run all similarity computations concurrently
    tasks = [semantic.compute(text1, text2) for text1, text2 in texts]
    results = await asyncio.gather(*tasks)
    
    # Print results
    print("\n✓ All computations completed successfully!")
    for i, (texts_pair, result) in enumerate(zip(texts, results)):
        text1, text2 = texts_pair
        print(f"\n{i+1}. Similarity: {result['similarity']:.4f}")
        print(f"   Text 1: {text1}")
        print(f"   Text 2: {text2}")
    
    print("\n✅ Test passed! No meta tensor errors.")

if __name__ == "__main__":
    asyncio.run(test_concurrent_semantic())
