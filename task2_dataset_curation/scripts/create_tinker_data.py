#!/usr/bin/env python3
"""
Create Tinker training data for TASK 2.
Converts ESL/native data to StealthRL training format.
"""

import json
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_tinker_data():
    # Input paths
    esl_file = Path("data/esl/toefl11.jsonl")
    native_file = Path("data/native/native_academic.jsonl")

    # Output paths
    output_dir = Path("data/tinker")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading ESL data from {esl_file}")
    esl_records = []
    with open(esl_file, 'r') as f:
        for line in f:
            if line.strip():
                esl_records.append(json.loads(line))

    logger.info(f"Loading native data from {native_file}")
    native_records = []
    with open(native_file, 'r') as f:
        for line in f:
            if line.strip():
                native_records.append(json.loads(line))

    logger.info(f"Loaded {len(esl_records)} ESL and {len(native_records)} native records")

    # Shuffle with seed
    random.seed(42)
    random.shuffle(esl_records)
    random.shuffle(native_records)

    # Skip first 42 ESL (used for eval) and first 63 native (used for eval)
    # Use remaining for training
    esl_train = esl_records[42:]
    native_train = native_records[63:]

    logger.info(f"Using {len(esl_train)} ESL and {len(native_train)} native for training")

    # Convert to Tinker format
    def convert_to_tinker_format(record):
        """Convert ESL/native record to Tinker training format."""
        return {
            "ai_text": record["text"],  # Will be paraphrased during training
            "human_reference": record["text"],  # Original text as reference
            "domain": "academic",  # All our data is academic
            "is_esl": record["is_esl"],
            "metadata": {
                "id": record["id"],
                "source": record["source"],
            }
        }

    # Convert all records
    train_data = []
    for record in esl_train + native_train:
        train_data.append(convert_to_tinker_format(record))

    # Shuffle combined training data
    random.shuffle(train_data)

    # Split into train (80%) and test (20%)
    split_idx = int(len(train_data) * 0.8)
    train_split = train_data[:split_idx]
    test_split = train_data[split_idx:]

    # Save train split
    train_path = output_dir / "train.jsonl"
    with open(train_path, 'w') as f:
        for record in train_split:
            f.write(json.dumps(record) + '\n')
    logger.info(f"Saved train split to {train_path}")

    # Save test split
    test_path = output_dir / "test.jsonl"
    with open(test_path, 'w') as f:
        for record in test_split:
            f.write(json.dumps(record) + '\n')
    logger.info(f"Saved test split to {test_path}")

    # Calculate statistics
    train_esl = sum(1 for r in train_split if r["is_esl"])
    test_esl = sum(1 for r in test_split if r["is_esl"])

    # Print statistics
    print("\n" + "="*60)
    print("TINKER TRAINING DATA CREATED")
    print("="*60)
    print(f"Train split: {len(train_split)} samples")
    print(f"  ESL: {train_esl} ({train_esl/len(train_split)*100:.1f}%)")
    print(f"  Native: {len(train_split) - train_esl} ({(len(train_split)-train_esl)/len(train_split)*100:.1f}%)")
    print(f"  Saved to: {train_path}")
    print()
    print(f"Test split:  {len(test_split)} samples")
    print(f"  ESL: {test_esl} ({test_esl/len(test_split)*100:.1f}%)")
    print(f"  Native: {len(test_split) - test_esl} ({(len(test_split)-test_esl)/len(test_split)*100:.1f}%)")
    print(f"  Saved to: {test_path}")
    print()
    print(f"Total training samples: {len(train_data)}")
    print("="*60)

if __name__ == "__main__":
    create_tinker_data()
