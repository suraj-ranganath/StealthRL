#!/usr/bin/env python3
"""
Analyze dataset sizes and estimate API costs.
"""

import json
import os

def analyze_file(filepath):
    """Count samples and words in a JSONL file."""
    if not os.path.exists(filepath):
        return 0, 0, 0, 0
        
    samples = 0
    total_words = 0
    human_words = 0
    ai_words = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                samples += 1
                
                if 'human_reference' in data:
                    hw = len(data['human_reference'].split())
                    human_words += hw
                    total_words += hw
                
                if 'ai_text' in data:
                    aw = len(data['ai_text'].split())
                    ai_words += aw
                    total_words += aw
            except:
                pass
    
    return samples, total_words, human_words, ai_words

def main():
    datasets = {
        "tinker_full_esl40_nodup (MAIN DATASET)": {
            "train": "data/tinker_full_esl40_nodup/train.jsonl",
            "test": "data/tinker_full_esl40_nodup/test.jsonl"
        },
        "tinker": {
            "train": "data/tinker/train.jsonl",
            "test": "data/tinker/test.jsonl"
        },
        "tinker_esl40": {
            "train": "data/tinker_esl40/train.jsonl",
            "test": "data/tinker_esl40/test.jsonl"
        }
    }
    
    print("=" * 80)
    print("STEALTHRL DATASET SIZE ANALYSIS")
    print("=" * 80)
    
    for dataset_name, splits in datasets.items():
        print(f"\n📁 {dataset_name}")
        print("-" * 80)
        
        total_samples = 0
        total_words = 0
        total_human = 0
        total_ai = 0
        
        for split_name, filepath in splits.items():
            samples, words, human, ai = analyze_file(filepath)
            
            if samples > 0:
                total_samples += samples
                total_words += words
                total_human += human
                total_ai += ai
                
                print(f"  {split_name:6s}: {samples:6,} samples | {words:10,} words | "
                      f"avg {words/samples:.0f} words/sample")
        
        if total_samples > 0:
            print(f"  {'TOTAL':6s}: {total_samples:6,} samples | {total_words:10,} words | "
                  f"avg {total_words/total_samples:.0f} words/sample")
            print(f"           Human text: {total_human:,} words")
            print(f"           AI text:    {total_ai:,} words")
    
    # API Cost Estimation
    print("\n" + "=" * 80)
    print("API COST ESTIMATION (Main Dataset: tinker_full_esl40_nodup)")
    print("=" * 80)
    
    # Analyze main dataset
    train_samples, train_words, _, _ = analyze_file("data/tinker_full_esl40_nodup/train.jsonl")
    test_samples, test_words, _, _ = analyze_file("data/tinker_full_esl40_nodup/test.jsonl")
    
    total_samples = train_samples + test_samples
    total_words = train_words + test_words
    
    # During training, we generate paraphrases (need to detect GENERATED text)
    # During evaluation, we detect both human and AI text
    # Typical experiment: 1 training run + 1 evaluation
    
    # Training: detect generated paraphrases (multiple per sample with GRPO)
    group_size = 16  # GRPO rollouts per prompt
    training_detections = train_samples * group_size
    
    # Evaluation: detect test set (both human and paraphrased)
    eval_detections = test_samples * 2  # human + paraphrased
    
    total_detections = training_detections + eval_detections
    
    # Words per detection (average from one text field)
    avg_words_per_detection = (total_words / (total_samples * 2))  # divided by 2 fields
    total_words_detected = total_detections * avg_words_per_detection
    
    print(f"\nDataset Statistics:")
    print(f"  Train samples:    {train_samples:,}")
    print(f"  Test samples:     {test_samples:,}")
    print(f"  Total samples:    {total_samples:,}")
    print(f"  Total words:      {total_words:,}")
    print(f"  Avg words/text:   {avg_words_per_detection:.0f}")
    
    print(f"\nExperiment Scope (1 training run + 1 evaluation):")
    print(f"  Training detections:   {training_detections:,} (train × {group_size} GRPO rollouts)")
    print(f"  Evaluation detections: {eval_detections:,} (test × 2 for human+paraphrased)")
    print(f"  Total detections:      {total_detections:,}")
    print(f"  Total words detected:  {total_words_detected:,.0f}")
    
    print(f"\n" + "-" * 80)
    print("API Cost Estimates per Experiment:")
    print("-" * 80)
    
    # GPTZero: $0.01 per detection (~500 words)
    gptzero_cost = (total_words_detected / 500) * 0.01
    print(f"  GPTZero:        ${gptzero_cost:,.2f} (~$0.01 per 500 words)")
    
    # Originality.AI: $0.01 per 100 words
    originality_cost = (total_words_detected / 100) * 0.01
    print(f"  Originality.AI: ${originality_cost:,.2f} (~$0.01 per 100 words)")
    
    # Sapling: 100 free/day, then $25/month unlimited
    sapling_days = total_detections / 100
    sapling_cost = 25 if sapling_days > 30 else 0
    print(f"  Sapling.ai:     ${sapling_cost:.2f} (100 free/day = {sapling_days:.1f} days, then $25/month)")
    
    print(f"\n" + "-" * 80)
    print("Recommended Combinations:")
    print("-" * 80)
    
    print(f"\n  Option A: All Open Source")
    print(f"    Cost: $0")
    print(f"    Detectors: RoBERTa + Fast-DetectGPT + Binoculars")
    print(f"    Requirements: GPU (16GB VRAM)")
    
    print(f"\n  Option B: Mixed (1 API + 2 Local)")
    print(f"    Cost: ${gptzero_cost:,.2f} (GPTZero only)")
    print(f"    Detectors: GPTZero + Fast-DetectGPT + Binoculars")
    print(f"    Requirements: GPU (16GB VRAM) + API key")
    
    print(f"\n  Option C: Quick Test (Free Tier)")
    print(f"    Cost: $0")
    print(f"    Detectors: Sapling (free 100/day) + RoBERTa (local)")
    print(f"    Time: {sapling_days:.0f} days for full experiment")
    print(f"    Note: Use for small-scale testing only")
    
    print(f"\n" + "-" * 80)
    print("Full Research Project (3-5 experiments):")
    print("-" * 80)
    
    num_experiments = 5  # Full ensemble + transfer + 3 ablations
    print(f"\n  Experiments: {num_experiments}")
    print(f"    - 1× Full ensemble training")
    print(f"    - 1× Transfer evaluation")
    print(f"    - 3× Ablation studies")
    
    print(f"\n  Total API Costs:")
    print(f"    GPTZero:        ${gptzero_cost * num_experiments:,.2f}")
    print(f"    Originality.AI: ${originality_cost * num_experiments:,.2f}")
    print(f"    Mixed (1 API):  ${gptzero_cost * num_experiments:,.2f}")
    
    print(f"\n  Budget Recommendation: ${gptzero_cost * num_experiments * 1.2:,.2f}")
    print(f"    (includes 20% buffer for retries/errors)")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
