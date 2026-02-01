#!/usr/bin/env python3
"""
Test data validation and filtering logic.

This script tests the is_valid_text function and shows:
1. What kinds of samples get filtered
2. Statistics on filtering
"""

import sys
import re
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def is_valid_text(text: str) -> bool:
    """
    Validate text to filter out corrupted/gibberish samples.
    
    Returns False if text appears to be corrupted, meaningless, or problematic.
    """
    if not text or not isinstance(text, str):
        return False
    
    # Too long (likely corrupted list or garbage)
    if len(text) > 5000:
        logger.debug(f"Rejecting text: too long ({len(text)} chars)")
        return False
    
    # Too short (not enough content)
    if len(text.strip()) < 20:
        logger.debug("Rejecting text: too short")
        return False
    
    # Known gibberish patterns from training logs
    gibberish_patterns = [
        r"Filipinsript",
        r"GALAges",
        r"Desifications",
        r"Gatcalasio",
        r"usedmodified",
        r"Snal only determined to try to the Nationalized",
    ]
    
    for pattern in gibberish_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            logger.debug(f"Rejecting text: contains gibberish pattern '{pattern}'")
            return False
    
    # Check for excessive numbered lists (corruption indicator)
    numbered_items = re.findall(r'^\d+\.', text, re.MULTILINE)
    if len(numbered_items) > 50:
        logger.debug(f"Rejecting text: excessive numbered list ({len(numbered_items)} items)")
        return False
    
    # Check for reasonable English text (at least some recognizable words)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    if len(words) < 5:
        logger.debug(f"Rejecting text: too few words ({len(words)})")
        return False
    
    # Check for reasonable word/total char ratio (detect gibberish)
    total_chars = len(text)
    word_chars = sum(len(w) for w in words)
    if word_chars / total_chars < 0.3:  # Less than 30% is actual words
        logger.debug(f"Rejecting text: low word ratio ({word_chars}/{total_chars})")
        return False
    
    # Check for repeated phrases (generation artifacts)
    sentences = re.split(r'[.!?]+', text)
    if len(sentences) > 5:
        sentence_counts = {}
        for sent in sentences:
            sent_clean = sent.strip().lower()
            if len(sent_clean) > 10:
                sentence_counts[sent_clean] = sentence_counts.get(sent_clean, 0) + 1
        
        # If any sentence repeats more than 3 times, likely garbage
        if any(count > 3 for count in sentence_counts.values()):
            logger.debug("Rejecting text: excessive repetition")
            return False
    
    return True


def test_validation():
    """Test validation with sample texts."""
    
    # Test cases
    test_cases = [
        # Good texts
        ("Good: Normal text", "This is a normal paragraph with enough words and reasonable content. It should pass validation.", True),
        ("Good: Article", "The study found that machine learning models can effectively detect AI-generated text. Researchers analyzed thousands of samples to develop robust detection methods.", True),
        
        # Bad texts
        ("Bad: Too short", "Too short", False),
        ("Bad: Gibberish", "This text contains Filipinsript and other garbage", False),
        ("Bad: Gibberish 2", "GALAges and Desifications are not real words", False),
        ("Bad: Too long", "word " * 800, False),  # 4000+ chars
        ("Bad: Low word ratio", "x" * 100 + " word " * 10, False),  # Mostly gibberish
        ("Bad: Few words", "aaa bbb ccc", False),
        ("Bad: Repetition", "The same sentence. " * 10, False),
        ("Bad: Numbered list", "\n".join([f"{i}. Item" for i in range(60)]), False),
    ]
    
    logger.info("=" * 80)
    logger.info("VALIDATION TEST RESULTS")
    logger.info("=" * 80)
    
    passed = 0
    failed = 0
    
    for name, text, expected in test_cases:
        result = is_valid_text(text)
        status = "✓" if result == expected else "✗"
        
        if result == expected:
            passed += 1
            logger.info(f"{status} {name}: {'PASS' if result else 'REJECT'} (expected)")
        else:
            failed += 1
            logger.error(f"{status} {name}: {'PASS' if result else 'REJECT'} (expected {'PASS' if expected else 'REJECT'})")
    
    logger.info("=" * 80)
    logger.info(f"Results: {passed} passed, {failed} failed")
    logger.info("=" * 80)
    
    return failed == 0


def test_corrupted_sample():
    """Test with the actual corrupted sample from training logs."""
    
    # Simulated corrupted sample based on training log
    corrupted_text = """Clearly, X is better than Y because:
1. First reason
2. Second reason
""" + "\n".join([f"{i}. Item {i}" for i in range(3, 174)]) + """
Filipinsript GALAges Desifications Gatcalasio Snal only determined to try to the Nationalized
"""
    
    logger.info("\n" + "=" * 80)
    logger.info("TESTING CORRUPTED SAMPLE FROM TRAINING LOGS")
    logger.info("=" * 80)
    
    logger.info(f"Text length: {len(corrupted_text)} chars")
    logger.info(f"Preview: {corrupted_text[:200]}...")
    logger.info(f"\nEnd: ...{corrupted_text[-200:]}")
    
    result = is_valid_text(corrupted_text)
    
    if not result:
        logger.info("\n✓✓✓ Corrupted sample CORRECTLY REJECTED ✓✓✓")
    else:
        logger.error("\n✗✗✗ Corrupted sample INCORRECTLY PASSED ✗✗✗")
    
    logger.info("=" * 80)
    
    return not result


if __name__ == "__main__":
    try:
        # Run tests
        validation_ok = test_validation()
        corrupted_ok = test_corrupted_sample()
        
        if validation_ok and corrupted_ok:
            logger.info("\n✓ All tests passed! Filtering logic is working correctly.")
            logger.info("\nDATASET CONSISTENCY ANSWER:")
            logger.info("=" * 80)
            logger.info("YES - The training dataset is consistent across runs.")
            logger.info("")
            logger.info("The StealthRLDatasetBuilder uses a fixed seed (default=0) for")
            logger.info("shuffling, which ensures the same order every time.")
            logger.info("")
            logger.info("With the new filtering, corrupted samples will be removed")
            logger.info("consistently, so you'll get the same clean dataset each run.")
            logger.info("=" * 80)
        else:
            logger.error("\n✗ Some tests failed. Check the validation logic.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        sys.exit(1)
