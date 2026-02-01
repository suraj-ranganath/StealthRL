"""
Test that detector lazy loading works correctly.
"""
import logging
from eval.methods import get_method

logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_lazy_loading():
    """Test that detector only loads when n_candidates > 1"""
    
    # Test M1: Simple Paraphrase
    print("\n=== Testing M1 (Simple Paraphrase) ===")
    m1 = get_method("simple_paraphrase", device="cpu")
    
    print("\n1. Loading method (detector should NOT load):")
    m1.load()
    
    print("\n2. Running with n_candidates=1 (detector should NOT load):")
    result1 = m1.attack("This is a test sentence.", n_candidates=1)
    print(f"Result: {result1.text[:50]}...")
    
    print("\n3. Running with n_candidates=2 (detector SHOULD load now):")
    result2 = m1.attack("This is another test sentence.", n_candidates=2)
    print(f"Result: {result2.text[:50]}...")
    
    # Test M2: StealthRL
    print("\n\n=== Testing M2 (StealthRL) ===")
    m2 = get_method("stealthrl", device="cpu")
    
    print("\n1. Loading method (detector should NOT load):")
    m2.load()
    
    print("\n2. Running with n_candidates=1 (detector should NOT load):")
    result1 = m2.attack("This is a test sentence.", n_candidates=1)
    print(f"Result: {result1.text[:50]}...")
    
    print("\n3. Running with n_candidates=2 (detector SHOULD load now):")
    result2 = m2.attack("This is another test sentence.", n_candidates=2)
    print(f"Result: {result2.text[:50]}...")

if __name__ == "__main__":
    test_lazy_loading()
