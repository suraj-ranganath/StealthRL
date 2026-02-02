#!/usr/bin/env python3
"""
Quick evaluation of detector on test set.
Get baseline metrics before training.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm.asyncio import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from stealthrl.tinker.detectors import DetectorCache, FastDetectGPTDetector, GhostbusterDetector


async def evaluate_detector(detector, test_data: List[Dict], max_samples: int = 200):
    """Evaluate detector on HUMAN texts only."""
    
    # Limit samples for quick evaluation
    test_data = test_data[:max_samples]
    
    print(f"\nEvaluating HUMAN texts only on {len(test_data)} samples...")
    print("This may take 2-5 minutes on first run (downloading model)...")
    
    human_scores = []
    
    for item in tqdm(test_data, desc="Evaluating human texts"):
        human_text = item["human_reference"]
        
        # Score only human text
        human_score = await detector.predict(human_text)
        human_scores.append(human_score)
    
    # Calculate statistics for human texts only
    human_score_mean = np.mean(human_scores)
    human_score_std = np.std(human_scores)
    human_score_min = np.min(human_scores)
    human_score_max = np.max(human_scores)
    human_score_median = np.median(human_scores)
    
    # Count how many are classified as AI (score > 0.5)
    false_positives = sum(1 for score in human_scores if score > 0.5)
    false_positive_rate = false_positives / len(human_scores)
    
    # Count distribution
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(human_scores, bins=bins)
    
    return {
        "human_score_mean": human_score_mean,
        "human_score_std": human_score_std,
        "human_score_min": human_score_min,
        "human_score_max": human_score_max,
        "human_score_median": human_score_median,
        "false_positive_rate": false_positive_rate,  # % of humans classified as AI
        "false_positives": false_positives,
        "total_samples": len(human_scores),
        "distribution": hist.tolist(),
    }


async def main():
    """Main evaluation pipeline."""
    
    # Check if test file exists
    test_file = Path("data/tinker_full_esl40_nodup/test.jsonl")
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        print("Make sure you're in the project root directory.")
        return
    
    print(f"\n📂 Loading test data from: {test_file}")
    test_data = []
    with open(test_file) as f:
        for line in f:
            test_data.append(json.loads(line))
    
    print(f"✓ Loaded {len(test_data)} test samples")
    
    # Initialize detectors
    cache = DetectorCache(cache_path="cache/eval_detectors.db")
    
    detectors = {
        "Fast-DetectGPT (gpt-neo-2.7B)": FastDetectGPTDetector(
            cache,
            model_name="gpt-neo-2.7B"
        ),
        "Ghostbuster (roberta-base)": GhostbusterDetector(
            cache,
            model_name="roberta-base"
        ),
    }
    
    results = {}
    
    for name, detector in detectors.items():
        print("\n" + "=" * 80)
        print(f"Testing: {name}")
        print("=" * 80)
        
        try:
            metrics = await evaluate_detector(detector, test_data, max_samples=200)
            results[name] = metrics
            
            print("\n📊 Results (Human Texts Only):")
            print(f"  Samples tested:        {metrics['total_samples']}")
            print(f"  Mean score:            {metrics['human_score_mean']:.3f} ± {metrics['human_score_std']:.3f}")
            print(f"  Median score:          {metrics['human_score_median']:.3f}")
            print(f"  Min/Max:               {metrics['human_score_min']:.3f} / {metrics['human_score_max']:.3f}")
            print(f"\n  False Positive Rate:   {metrics['false_positive_rate']:.1%}")
            print(f"  (Humans labeled as AI: {metrics['false_positives']}/{metrics['total_samples']})")
            print(f"\n  Score Distribution:")
            bins = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', 
                    '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
            for bin_name, count in zip(bins, metrics['distribution']):
                print(f"    {bin_name}: {count:3d} samples")
            
        except Exception as e:
            print(f"\n❌ Error testing {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary comparison
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("COMPARISON (Human Texts Only)")
        print("=" * 80)
        print("\n{:<35} {:>12} {:>12} {:>15}".format("Detector", "Mean Score", "Median", "False Pos Rate"))
        print("-" * 80)
        for name, metrics in results.items():
            print("{:<35} {:>12.3f} {:>12.3f} {:>14.1%}".format(
                name, 
                metrics['human_score_mean'], 
                metrics['human_score_median'],
                metrics['false_positive_rate']
            ))
    
    # Save results
    output_file = Path("outputs/baseline_detector_metrics.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("💡 Key Metrics Explained:")
    print("  - Mean Score: Average detector confidence (0=human, 1=AI)")
    print("  - False Positive Rate: % of HUMAN texts wrongly labeled as AI")
    print("  - Good detector should give LOW scores to human text")
    print("\n🎯 What This Tells Us:")
    print("  - High FP rate = detector labels everything as AI (broken)")
    print("  - Scores near 0.5 = detector is just guessing")
    print("  - Scores near 0.0 = detector correctly identifies human text")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user")
        sys.exit(0)
