#!/usr/bin/env python3
"""
Extract human-AI text pairs from DetectRL dataset.

DetectRL has 349,165 samples total with 57,008 human texts.
This script extracts academic domain samples that are suitable for training.
"""

import json
import random
from pathlib import Path
from collections import defaultdict
import argparse


def load_detectrl_file(filepath):
    """Load a DetectRL JSON file."""
    with open(filepath) as f:
        return json.load(f)


def is_suitable_for_training(sample):
    """
    Check if sample is suitable for training.
    
    Criteria:
    - Must have 'human' label text
    - Text length > 50 words
    - Academic or professional domain preferred
    """
    if sample.get('label') != 'human':
        return False
    
    text = sample.get('text', '')
    word_count = len(text.split())
    
    if word_count < 50:
        return False
    
    # Accept academic and creative writing domains
    data_type = sample.get('data_type', '')
    accepted_domains = ['abstract', 'arxiv', 'essay', 'writing', 'story', 'article', 'review']
    
    return any(domain in data_type.lower() for domain in accepted_domains)


def extract_pairs_from_detectrl(detectrl_dir, max_samples=5000, seed=42):
    """
    Extract human-AI pairs from DetectRL.
    
    Args:
        detectrl_dir: Path to DetectRL/Benchmark/Tasks/Task1
        max_samples: Maximum number of pairs to extract
        seed: Random seed for reproducibility
    
    Returns:
        List of dicts with 'human_text', 'ai_text', 'metadata'
    """
    random.seed(seed)
    detectrl_path = Path(detectrl_dir)
    
    # Focus on academic domains and avoid attack datasets
    preferred_files = [
        'multi_domains_arxiv_train.json',
        'multi_domains_arxiv_test.json',
        'multi_domains_writing_prompt_train.json',
        'multi_domains_writing_prompt_test.json',
    ]
    
    all_pairs = []
    stats = defaultdict(int)
    
    print(f"Extracting from DetectRL...")
    print("=" * 60)
    
    for filename in preferred_files:
        filepath = detectrl_path / filename
        if not filepath.exists():
            print(f"⚠️  Skipping {filename} (not found)")
            continue
        
        data = load_detectrl_file(filepath)
        print(f"\nProcessing {filename}...")
        print(f"  Total samples: {len(data):,}")
        
        # Separate human and AI texts (don't group by data_type - they don't match!)
        human_samples = []
        ai_samples = []
        
        for sample in data:
            data_type = sample.get('data_type', 'unknown')
            
            if sample.get('label') == 'human':
                if is_suitable_for_training(sample):
                    human_samples.append(sample)
            elif sample.get('label') == 'llm':  # DetectRL uses 'llm' label for AI text
                text = sample.get('text', '')
                if len(text.split()) >= 50:  # Same length requirement
                    ai_samples.append(sample)
        
        print(f"  Suitable human samples: {len(human_samples):,}")
        print(f"  Suitable AI samples: {len(ai_samples):,}")
        
        # Create pairs - pair each human with a random AI from same file
        if human_samples and ai_samples:
            for human_sample in human_samples:
                if len(all_pairs) >= max_samples:
                    break
                
                ai_sample = random.choice(ai_samples)
                human_data_type = human_sample.get('data_type', 'unknown')
                
                pair = {
                    'human_reference': human_sample['text'],
                    'ai_text': ai_sample['text'],
                    'domain': 'academic',
                    'is_esl': False,  # DetectRL doesn't have ESL labels
                    'metadata': {
                        'source': f'DetectRL_{human_data_type}',
                        'dataset': 'DetectRL',
                        'human_data_type': human_data_type,
                        'ai_data_type': ai_sample.get('data_type', 'unknown'),
                        'ai_llm_type': ai_sample.get('llm_type', 'unknown'),
                        'original_file': filename
                    }
                }
                
                all_pairs.append(pair)
                stats[human_data_type] += 1
        
        print(f"  Extracted so far: {len(all_pairs):,} pairs")
        
        if len(all_pairs) >= max_samples:
            print(f"\n✅ Reached max_samples limit ({max_samples:,})")
            break
    
    print("\n" + "=" * 60)
    print(f"Total pairs extracted: {len(all_pairs):,}")
    print("\nBreakdown by data type:")
    for data_type, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {data_type}: {count:,}")
    
    return all_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Extract training pairs from DetectRL dataset"
    )
    parser.add_argument(
        '--detectrl-dir',
        default='data/raw/DetectRL/Benchmark/Tasks/Task1',
        help='Path to DetectRL Task1 directory'
    )
    parser.add_argument(
        '--output',
        default='data/native/detectrl_native.jsonl',
        help='Output JSONL file'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=5000,
        help='Maximum number of pairs to extract (default: 5000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Extract pairs
    pairs = extract_pairs_from_detectrl(
        args.detectrl_dir,
        max_samples=args.max_samples,
        seed=args.seed
    )
    
    if not pairs:
        print("\n❌ No pairs extracted!")
        return
    
    # Save to JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')
    
    print(f"\n✅ Saved {len(pairs):,} pairs to {output_path}")
    
    # Show statistics
    word_counts = [len(pair['human_reference'].split()) for pair in pairs]
    avg_words = sum(word_counts) / len(word_counts)
    min_words = min(word_counts)
    max_words = max(word_counts)
    
    print(f"\nText statistics:")
    print(f"  Average words: {avg_words:.1f}")
    print(f"  Min words: {min_words}")
    print(f"  Max words: {max_words}")


if __name__ == '__main__':
    main()
