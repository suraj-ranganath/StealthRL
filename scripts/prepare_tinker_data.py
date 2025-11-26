"""
Data Preparation for StealthRL Tinker Training.

This script prepares training data in the JSONL format expected by StealthRLDataset.

Expected format:
{
    "ai_text": "...",
    "human_reference": "...",
    "domain": "academic"|"informal"|"news",
    "is_esl": true|false,
    "metadata": {...}
}
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


def prepare_from_existing_datasets(
    input_paths: List[str],
    output_dir: str,
    train_split: float = 0.8,
    max_examples: int | None = None,
    seed: int = 42,
):
    """
    Prepare StealthRL data from existing datasets.
    
    Args:
        input_paths: Paths to existing dataset files (JSONL or JSON)
        output_dir: Output directory for train/test splits
        train_split: Fraction of data for training
        max_examples: Optional limit on total examples
        seed: Random seed for shuffling
    """
    random.seed(seed)
    
    # Load all examples
    all_examples = []
    for input_path in input_paths:
        examples = load_dataset(input_path)
        all_examples.extend(examples)
        logger.info(f"Loaded {len(examples)} examples from {input_path}")
    
    # Limit if specified
    if max_examples:
        all_examples = all_examples[:max_examples]
    
    # Shuffle
    random.shuffle(all_examples)
    
    # Split
    split_idx = int(len(all_examples) * train_split)
    train_examples = all_examples[:split_idx]
    test_examples = all_examples[split_idx:]
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_examples(train_examples, output_path / "train.jsonl")
    save_examples(test_examples, output_path / "test.jsonl")
    
    logger.info(f"Saved {len(train_examples)} train examples")
    logger.info(f"Saved {len(test_examples)} test examples")
    
    # Print statistics
    print_statistics(train_examples, "Train")
    print_statistics(test_examples, "Test")


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from JSONL or JSON file.
    
    Args:
        path: Path to dataset file
    
    Returns:
        List of example dictionaries
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    examples = []
    
    if path_obj.suffix == ".jsonl":
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
    elif path_obj.suffix == ".json":
        with open(path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                examples = data
            else:
                examples = [data]
    else:
        raise ValueError(f"Unsupported file format: {path_obj.suffix}")
    
    # Convert to StealthRL format if needed
    examples = [convert_to_stealthrl_format(ex) for ex in examples]
    
    return examples


def convert_to_stealthrl_format(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert example to StealthRL format.
    
    Handles various input formats and standardizes to:
    {
        "ai_text": "...",
        "human_reference": "...",
        "domain": "...",
        "is_esl": bool,
        "metadata": {...}
    }
    
    Args:
        example: Input example dict
    
    Returns:
        StealthRL-formatted example
    """
    # If already in correct format, return as-is
    if all(k in example for k in ["ai_text", "human_reference", "domain", "is_esl"]):
        return example
    
    # Try to map common field names
    ai_text = example.get("ai_text") or example.get("generated_text") or example.get("text") or ""
    human_reference = example.get("human_reference") or example.get("human_text") or example.get("reference") or ""
    domain = example.get("domain") or example.get("category") or "unknown"
    is_esl = example.get("is_esl") or example.get("esl") or example.get("is_non_native") or False
    
    # Extract metadata
    metadata = example.get("metadata", {})
    metadata.update({
        k: v for k, v in example.items()
        if k not in ["ai_text", "human_reference", "domain", "is_esl", "metadata"]
    })
    
    return {
        "ai_text": ai_text,
        "human_reference": human_reference,
        "domain": domain,
        "is_esl": bool(is_esl),
        "metadata": metadata,
    }


def save_examples(examples: List[Dict[str, Any]], path: Path):
    """
    Save examples to JSONL file.
    
    Args:
        examples: List of example dicts
        path: Output path
    """
    with open(path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')


def print_statistics(examples: List[Dict[str, Any]], split_name: str):
    """
    Print dataset statistics.
    
    Args:
        examples: List of examples
        split_name: Name of split (Train/Test)
    """
    print(f"\\n{split_name} Statistics:")
    print(f"  Total examples: {len(examples)}")
    
    # Domain distribution
    domains = defaultdict(int)
    for ex in examples:
        domains[ex["domain"]] += 1
    
    print("  Domain distribution:")
    for domain, count in sorted(domains.items()):
        print(f"    {domain}: {count} ({count/len(examples)*100:.1f}%)")
    
    # ESL distribution
    esl_count = sum(1 for ex in examples if ex["is_esl"])
    print(f"  ESL examples: {esl_count} ({esl_count/len(examples)*100:.1f}%)")
    print(f"  Non-ESL examples: {len(examples) - esl_count} ({(len(examples) - esl_count)/len(examples)*100:.1f}%)")
    
    # Text length statistics
    ai_lengths = [len(ex["ai_text"].split()) for ex in examples]
    print(f"  AI text length: mean={sum(ai_lengths)/len(ai_lengths):.1f} words, "
          f"min={min(ai_lengths)}, max={max(ai_lengths)}")


def create_synthetic_example() -> Dict[str, Any]:
    """
    Create a synthetic example for testing.
    
    Returns:
        Example dictionary
    """
    return {
        "ai_text": (
            "The implementation of neural networks requires careful consideration "
            "of hyperparameters and architectural choices. Recent advances in "
            "optimization algorithms have enabled training of deeper models."
        ),
        "human_reference": (
            "Building neural networks demands thoughtful selection of hyperparameters "
            "and design decisions. New optimization techniques allow us to train "
            "networks with more layers."
        ),
        "domain": "academic",
        "is_esl": False,
        "metadata": {
            "model_family": "gpt",
            "original_source": "synthetic",
        },
    }


def create_synthetic_dataset(
    output_dir: str,
    num_train: int = 100,
    num_test: int = 20,
):
    """
    Create synthetic dataset for testing.
    
    Args:
        output_dir: Output directory
        num_train: Number of train examples
        num_test: Number of test examples
    """
    # Create examples
    train_examples = [create_synthetic_example() for _ in range(num_train)]
    test_examples = [create_synthetic_example() for _ in range(num_test)]
    
    # Vary domains and ESL flags
    domains = ["academic", "informal", "news"]
    for i, ex in enumerate(train_examples + test_examples):
        ex["domain"] = domains[i % len(domains)]
        ex["is_esl"] = (i % 4 == 0)  # 25% ESL
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_examples(train_examples, output_path / "train.jsonl")
    save_examples(test_examples, output_path / "test.jsonl")
    
    logger.info(f"Created synthetic dataset: {num_train} train, {num_test} test")


def main():
    parser = argparse.ArgumentParser(description="Prepare StealthRL training data")
    
    parser.add_argument(
        "--input-paths",
        type=str,
        nargs="+",
        help="Input dataset paths (JSONL or JSON)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/tinker",
        help="Output directory for train/test splits"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to use"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Create synthetic dataset for testing"
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=100,
        help="Number of synthetic train examples"
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=20,
        help="Number of synthetic test examples"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.synthetic:
        # Create synthetic dataset
        create_synthetic_dataset(
            output_dir=args.output_dir,
            num_train=args.num_train,
            num_test=args.num_test,
        )
    else:
        # Prepare from existing datasets
        if not args.input_paths:
            raise ValueError("--input-paths required when not using --synthetic")
        
        prepare_from_existing_datasets(
            input_paths=args.input_paths,
            output_dir=args.output_dir,
            train_split=args.train_split,
            max_examples=args.max_examples,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
