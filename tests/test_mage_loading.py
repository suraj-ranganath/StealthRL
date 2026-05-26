#!/usr/bin/env python3
"""Test MAGE dataset loading through StealthRLDatasetBuilder."""
import asyncio
from pathlib import Path
from stealthrl.tinker.dataset import StealthRLDatasetBuilder

async def test_mage_loading():
    """Test loading MAGE dataset."""
    builder = StealthRLDatasetBuilder(
        data_path="data/mage",
        batch_size=2,
        group_size=4,
        model_name_for_tokenizer="gpt2",
        renderer_name="qwen3",
        reward_config={
            "detector_names": ["roberta_openai", "fast_detectgpt"],
            "detector_weights": {"roberta_openai": 0.6, "fast_detectgpt": 0.4},
            "fast_detectgpt_model": "gpt-neo-2.7B",
            "roberta_openai_model": "roberta-large-openai-detector",
        },
        max_examples=10,  # Small test
    )
    
    print("Loading MAGE dataset...")
    train_dataset, test_dataset = await builder()
    
    print(f"\n✅ Successfully loaded MAGE data!")
    print(f"   Train batches: {len(train_dataset)}")
    if test_dataset:
        print(f"   Test batches: {len(test_dataset)}")
    
    # Check one batch
    print("\n📋 Sample batch (first 2 examples):")
    batch = train_dataset.get_batch(0)
    for i, env_builder in enumerate(batch[:2]):
        print(f"\n   Example {i+1}:")
        print(f"   - Domain: {env_builder.domain}")
        print(f"   - Human text: {env_builder.human_reference[:100]}...")
        print(f"   - Metadata: {env_builder.renderer}")

if __name__ == "__main__":
    asyncio.run(test_mage_loading())
