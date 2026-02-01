#!/usr/bin/env python3
"""
Quick test to verify data filtering is working.
Loads a small dataset and checks for corrupted samples.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def quick_test():
    """Load dataset with filtering and verify it works."""
    
    logger.info("=" * 80)
    logger.info("QUICK FILTERING TEST")
    logger.info("=" * 80)
    
    try:
        from stealthrl.tinker.dataset import StealthRLDatasetBuilder
        
        logger.info("\n1. Loading small dataset with filtering enabled...")
        
        # StealthRLDatasetBuilder needs: data_path, batch_size, group_size, 
        # model_name_for_tokenizer, renderer_name, reward_config
        # For quick test, we just need to check data loading works
        builder = StealthRLDatasetBuilder(
            data_path="data/mage",  # Path to dataset
            batch_size=8,
            group_size=4,
            model_name_for_tokenizer="gpt2",
            renderer_name="default",
            reward_config={
                "detector_config": {"name": "roberta_openai"},
                "sem_config": {"model_name": "all-MiniLM-L6-v2"},
                "ppl_config": {"model_name": "gpt2"},
                "det_weight": 1.0,
                "sem_weight": 1.0,
                "ppl_weight": 0.1,
            },
            seed=42,
            max_train_examples=100,  # Small for speed
            max_test_examples=20,
        )
        
        # Builder returns (train_dataset, test_dataset) tuple
        import asyncio
        train_dataset, test_dataset = asyncio.run(builder())
        
        logger.info(f"   ✓ Loaded {len(train_dataset.examples)} train examples")
        if test_dataset:
            logger.info(f"   ✓ Loaded {len(test_dataset.examples)} test examples")
        
        logger.info("\n2. Sampling examples to check quality...")
        # Get first 5 examples directly
        sample_size = min(5, len(train_dataset.examples))
        
        all_valid = True
        for i in range(sample_size):
            ex = train_dataset.examples[i]
            text = ex.ai_text
            
            # Check for gibberish patterns
            has_gibberish = any(
                pattern in text 
                for pattern in ["Filipinsript", "GALAges", "Desifications"]
            )
            
            # Check length
            too_long = len(text) > 3000
            too_short = len(text) < 20
            
            # Count numbered items
            numbered = text.count("\n") > 50
            
            if has_gibberish or too_long or too_short or numbered:
                logger.error(f"   ✗ Example {i+1} has issues:")
                if has_gibberish:
                    logger.error(f"     - Contains gibberish")
                if too_long:
                    logger.error(f"     - Too long ({len(text)} chars)")
                if too_short:
                    logger.error(f"     - Too short ({len(text)} chars)")
                if numbered:
                    logger.error(f"     - Excessive line breaks")
                all_valid = False
            else:
                logger.info(f"   ✓ Example {i+1}: {len(text)} chars, domain={ex.domain}")
        
        logger.info("\n" + "=" * 80)
        if all_valid:
            logger.info("✓✓✓ ALL CHECKS PASSED ✓✓✓")
            logger.info("\nFiltering is working correctly!")
            logger.info("You can now run a full training job safely.")
            logger.info("\n" + "=" * 80)
            return True
        else:
            logger.error("✗✗✗ SOME CHECKS FAILED ✗✗✗")
            logger.error("\nFiltering may have issues. Check the logs above.")
            logger.info("\n" + "=" * 80)
            return False
            
    except Exception as e:
        logger.error(f"\n✗ Error during test: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
