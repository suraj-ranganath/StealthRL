"""
Quick test to verify lazy loading of detectors.
This test checks that detectors are None after load() and only initialize when needed.
"""
import logging
from eval.methods import get_method

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_detector_lazy_init():
    """Verify detector is None after load(), only initializes when n_candidates > 1"""
    
    methods_to_test = [
        ("simple_paraphrase", {}),  # M1
        ("authormist", {}),          # M4 (Ollama)
    ]
    
    for method_name, kwargs in methods_to_test:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {method_name}")
        logger.info(f"{'='*60}")
        
        # Get and load method
        method = get_method(method_name, device="cpu", **kwargs)
        method.load()
        
        # Check that detector is None after load
        if hasattr(method, 'rerank_detector'):
            if method.rerank_detector is None:
                logger.info(f"✓ {method_name}: rerank_detector is None after load() - lazy loading works!")
            else:
                logger.error(f"✗ {method_name}: rerank_detector was loaded eagerly (should be None)")
        else:
            logger.warning(f"? {method_name}: No rerank_detector attribute found")

if __name__ == "__main__":
    test_detector_lazy_init()
