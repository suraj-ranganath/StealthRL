#!/usr/bin/env python3
"""
Prepare training and evaluation datasets for StealthRL.

This script processes human and LLM-generated text, including
ESL vs native subsets for fairness analysis.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import random


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_json(file_path: str) -> List[Dict]:
    """Load data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def load_human_texts(data_path: str) -> List[Dict]:
    """Load human-written texts."""
    path = Path(data_path)
    texts = []
    
    # Try different file formats
    for pattern in ["*.jsonl", "*.json", "*.txt"]:
        for file_path in path.glob(pattern):
            if file_path.stem.endswith("human") or "human" in file_path.stem.lower():
                if file_path.suffix == ".jsonl":
                    texts.extend(load_jsonl(file_path))
                elif file_path.suffix == ".json":
                    texts.extend(load_json(file_path))
                elif file_path.suffix == ".txt":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                texts.append({"text": line.strip(), "label": 0})
    
    return texts


def load_ai_texts(data_path: str) -> List[Dict]:
    """Load AI-generated texts."""
    path = Path(data_path)
    texts = []
    
    for pattern in ["*.jsonl", "*.json", "*.txt"]:
        for file_path in path.glob(pattern):
            if file_path.stem.endswith("ai") or "generated" in file_path.stem.lower():
                if file_path.suffix == ".jsonl":
                    texts.extend(load_jsonl(file_path))
                elif file_path.suffix == ".json":
                    texts.extend(load_json(file_path))
                elif file_path.suffix == ".txt":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                texts.append({"text": line.strip(), "label": 1})
    
    return texts


def split_esl_native(texts: List[Dict], metadata_key: str = "language_background") -> Tuple[List[Dict], List[Dict]]:
    """Split texts into ESL and native subsets."""
    esl_texts = []
    native_texts = []
    
    for item in texts:
        if metadata_key in item:
            if item[metadata_key].lower() in ["esl", "non-native", "nonnative"]:
                esl_texts.append(item)
            elif item[metadata_key].lower() in ["native", "english"]:
                native_texts.append(item)
        else:
            # If no metadata, randomly assign (not ideal, but fallback)
            if random.random() > 0.5:
                native_texts.append(item)
    
    return esl_texts, native_texts


def create_train_test_split(
    data: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Split data into train and test sets."""
    random.seed(seed)
    random.shuffle(data)
    
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    return train_data, test_data


def save_jsonl(data: List[Dict], file_path: str):
    """Save data to JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Prepare StealthRL datasets")
    parser.add_argument("--input_dir", type=str, required=True, help="Input data directory")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--train_split", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("Loading human texts...")
    human_texts = load_human_texts(args.input_dir)
    print(f"Loaded {len(human_texts)} human texts")
    
    print("Loading AI texts...")
    ai_texts = load_ai_texts(args.input_dir)
    print(f"Loaded {len(ai_texts)} AI texts")
    
    # Combine and shuffle
    all_texts = human_texts + ai_texts
    
    # Split into train and test
    print(f"Splitting data with {args.train_split} train ratio...")
    train_data, test_data = create_train_test_split(all_texts, args.train_split, args.seed)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Create eval set from part of train
    train_data, eval_data = create_train_test_split(train_data, 0.9, args.seed)
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}, Test: {len(test_data)}")
    
    # Split into ESL and native for fairness analysis
    print("Splitting into ESL and native subsets...")
    esl_texts, native_texts = split_esl_native(human_texts)
    print(f"ESL texts: {len(esl_texts)}, Native texts: {len(native_texts)}")
    
    # Save datasets
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Saving processed datasets...")
    save_jsonl(train_data, output_dir / "train.jsonl")
    save_jsonl(eval_data, output_dir / "eval.jsonl")
    save_jsonl(test_data, output_dir / "test.jsonl")
    
    if esl_texts:
        esl_train, esl_val = create_train_test_split(esl_texts, 0.7, args.seed)
        save_jsonl(esl_val, output_dir / "esl_validation.jsonl")
        save_jsonl(esl_texts, output_dir / "esl_test.jsonl")
    
    if native_texts:
        native_train, native_val = create_train_test_split(native_texts, 0.7, args.seed)
        save_jsonl(native_val, output_dir / "native_validation.jsonl")
        save_jsonl(native_texts, output_dir / "native_test.jsonl")
    
    # Save human and AI separately for evaluation
    human_test = [item for item in test_data if item.get("label", 0) == 0]
    ai_test = [item for item in test_data if item.get("label", 0) == 1]
    save_jsonl(human_test, output_dir / "human_test.jsonl")
    save_jsonl(ai_test, output_dir / "ai_test.jsonl")
    
    print(f"\nDataset preparation complete!")
    print(f"Output saved to {output_dir}")
    print(f"\nDataset statistics:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Eval: {len(eval_data)} samples")
    print(f"  Test: {len(test_data)} samples ({len(human_test)} human, {len(ai_test)} AI)")
    print(f"  ESL: {len(esl_texts)} samples")
    print(f"  Native: {len(native_texts)} samples")


if __name__ == "__main__":
    main()
