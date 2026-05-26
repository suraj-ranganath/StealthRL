#!/usr/bin/env python3
"""
Quick test script to verify the eval module works correctly.

Run: python -m eval.test_eval_module
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all major components import correctly."""
    logger.info("Testing imports...")
    
    # Core imports
    from eval import (
        # Data
        MAGEDataset,
        RAIDDataset,
        PadBenDataset,
        EvalSample,
        # Detectors
        RoBERTaOpenAIDetector,
        FastDetectGPTDetector,
        DetectGPTDetector,
        BinocularsDetector,
        GhostbusterDetector,
        EnsembleDetector,
        DETECTOR_REGISTRY,
        DETECTOR_CONVENTIONS,
        get_detector,
        # Methods
        METHOD_REGISTRY,
        GUIDANCE_VARIANTS,
        get_method,
        # Metrics
        compute_auroc,
        compute_tpr_at_fpr,
        compute_asr,
        # Plots
        create_heatmap,
        create_tradeoff_plot,
        # Runner
        EvalRunner,
        # Sanitization
        sanitize,
        compute_sanitization_diff,
        run_sanitize_evaluation,
        ZERO_WIDTH_CHARS,
        HOMOGLYPH_MAP,
        # Enhanced runner
        RunManager,
        EnhancedEvalRunner,
    )
    
    logger.info("✓ All imports successful")
    return True


def test_detector_registry():
    """Test detector registry."""
    logger.info("Testing detector registry...")
    
    from eval.detectors import DETECTOR_REGISTRY, DETECTOR_CONVENTIONS
    
    expected = ["roberta", "fast_detectgpt", "detectgpt", "binoculars", "ghostbuster", "ensemble"]
    for det in expected:
        if det not in DETECTOR_REGISTRY:
            logger.error(f"✗ Missing detector: {det}")
            return False
        logger.info(f"  ✓ {det} registered")
    
    # Check conventions
    for det, conv in DETECTOR_CONVENTIONS.items():
        assert "higher_is_ai" in conv, f"Missing higher_is_ai for {det}"
        assert "score_range" in conv, f"Missing score_range for {det}"
    
    logger.info("✓ Detector registry OK")
    return True


def test_method_registry():
    """Test method registry."""
    logger.info("Testing method registry...")
    
    from eval.methods import METHOD_REGISTRY, GUIDANCE_VARIANTS
    
    # Check base methods (m0-m5)
    expected = ["m0", "m1", "m2", "m3", "m4", "m5"]
    for method in expected:
        if method not in METHOD_REGISTRY:
            logger.error(f"✗ Missing method: {method}")
            return False
        logger.info(f"  ✓ {method} registered")
    
    # Check guidance variants
    for variant, config in GUIDANCE_VARIANTS.items():
        logger.info(f"  ✓ guidance variant {variant}: {config}")
    
    logger.info("✓ Method registry OK")
    return True


def test_sanitization():
    """Test sanitization module."""
    logger.info("Testing sanitization...")
    
    from eval.sanitize import sanitize, compute_sanitization_diff
    
    # Test basic sanitization
    test_cases = [
        # (input, expected_output, description)
        ("Hello World", "Hello World", "normal text"),
        ("Ηello", "Hello", "Greek H homoglyph"),  # Greek Eta
        ("test\u200btext", "testtext", "zero-width space"),
        ("café", "café", "NFKC normalization preserves diacritics"),
        ("ⅰⅱⅲ", "iiiiii", "Roman numerals normalized"),
    ]
    
    for input_text, expected, desc in test_cases:
        result = sanitize(input_text)
        if result != expected:
            logger.error(f"✗ Sanitization failed for '{desc}': got '{result}', expected '{expected}'")
            return False
        logger.info(f"  ✓ {desc}: '{input_text}' -> '{result}'")
    
    # Test diff computation
    diff = compute_sanitization_diff("Hello", "Ηello")  # Greek H
    assert diff["chars_changed"] > 0 or diff["original_length"] != diff["sanitized_length"]
    logger.info(f"  ✓ Diff computation: found {diff['chars_changed']} character changes")
    
    logger.info("✓ Sanitization OK")
    return True


def test_eval_sample():
    """Test EvalSample creation."""
    logger.info("Testing EvalSample...")
    
    from eval.data import EvalSample
    
    sample = EvalSample(
        id="test_1",
        label="ai",
        domain="news",
        generator="chatgpt",
        text="This is a test text.",
        metadata={"source": "test"},
    )
    
    assert sample.id == "test_1"
    assert sample.label == "ai"
    assert sample.is_ai == True
    assert sample.domain == "news"
    
    logger.info("✓ EvalSample OK")
    return True


def test_binoculars_formula():
    """Verify Binoculars uses correct formula."""
    logger.info("Testing Binoculars formula...")
    
    from eval.detectors import BinocularsDetector
    
    # Check that HIGHER_IS_AI is False for raw scores
    # (but normalized score should be higher = more AI)
    assert BinocularsDetector.HIGHER_IS_AI == False
    logger.info("  ✓ HIGHER_IS_AI = False (raw score: lower = AI)")
    
    # Check threshold is set
    assert hasattr(BinocularsDetector, "THRESHOLD_LOW_FPR")
    logger.info(f"  ✓ Threshold: {BinocularsDetector.THRESHOLD_LOW_FPR}")
    
    logger.info("✓ Binoculars formula OK")
    return True


def main():
    """Run all tests."""
    logger.info("=" * 50)
    logger.info("StealthRL Eval Module Test Suite")
    logger.info("=" * 50)
    
    tests = [
        test_imports,
        test_detector_registry,
        test_method_registry,
        test_sanitization,
        test_eval_sample,
        test_binoculars_formula,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"✗ Test {test.__name__} raised exception: {e}")
            failed += 1
    
    logger.info("=" * 50)
    logger.info(f"Results: {passed} passed, {failed} failed")
    logger.info("=" * 50)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
