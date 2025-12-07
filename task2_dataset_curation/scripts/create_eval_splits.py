#!/usr/bin/env python3
"""
Create ESL/native evaluation splits for TASK 2.
"""

import json
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_splits():
    # Input paths
    esl_file = Path("data/esl/toefl11.jsonl")
    native_file = Path("data/native/native_academic.jsonl")

    # Output paths
    output_dir = Path("data/processed")
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

    # We have 91 ESL and 215 native records (306 total)
    # For evaluation splits, use:
    # - dev: 30 samples (12 ESL, 18 native) = 40/60 ratio
    # - test: 75 samples (30 ESL, 45 native) = 40/60 ratio
    # - remaining: 201 samples for training

    # Split ESL: 12 dev, 30 test, rest for training
    esl_dev = esl_records[:12]
    esl_test = esl_records[12:42]
    esl_train = esl_records[42:]

    # Split native: 18 dev, 45 test, rest for training
    native_dev = native_records[:18]
    native_test = native_records[18:63]
    native_train = native_records[63:]

    # Create dev split
    dev_split = esl_dev + native_dev
    random.shuffle(dev_split)

    # Create test split
    test_split = esl_test + native_test
    random.shuffle(test_split)

    # Save dev split
    dev_path = output_dir / "esl_native_dev.jsonl"
    with open(dev_path, 'w') as f:
        for record in dev_split:
            f.write(json.dumps(record) + '\n')
    logger.info(f"Saved dev split to {dev_path}")

    # Save test split
    test_path = output_dir / "esl_native_test.jsonl"
    with open(test_path, 'w') as f:
        for record in test_split:
            f.write(json.dumps(record) + '\n')
    logger.info(f"Saved test split to {test_path}")

    # Print statistics
    print("\n" + "="*60)
    print("EVALUATION SPLITS CREATED")
    print("="*60)
    print(f"Dev split:  {len(dev_split)} samples ({len(esl_dev)} ESL, {len(native_dev)} native)")
    print(f"  ESL ratio: {len(esl_dev)/len(dev_split)*100:.1f}%")
    print(f"  Saved to: {dev_path}")
    print()
    print(f"Test split: {len(test_split)} samples ({len(esl_test)} ESL, {len(native_test)} native)")
    print(f"  ESL ratio: {len(esl_test)/len(test_split)*100:.1f}%")
    print(f"  Saved to: {test_path}")
    print()
    print(f"Remaining for training: {len(esl_train) + len(native_train)} samples")
    print(f"  ({len(esl_train)} ESL, {len(native_train)} native)")
    print("="*60)

if __name__ == "__main__":
    create_splits()
