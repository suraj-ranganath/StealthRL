#!/usr/bin/env python3
"""
Evaluate detectors on MAGE dataset.
Extract 200 samples (100 human + 100 AI) and test both detectors.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm.asyncio import tqdm
from datasets import load_from_disk

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from stealthrl.tinker.detectors import (
    DetectorCache,
    FastDetectGPTDetector,
    RoBERTaOpenAIDetector,
)


async def evaluate_detector(detector, test_data: List[Dict], max_samples: int = 100):
    """Evaluate detector on both human and AI texts."""
    
    print(f"\nEvaluating on {len(test_data)} samples...")
    print("This may take 2-5 minutes on first run (downloading model)...")
    
    human_scores = []
    ai_scores = []
    
    for item in tqdm(test_data, desc="Evaluating"):
        text = item["text"]
        label = item["label"]  # 1 = human, 0 = AI
        
        # Score text
        score = await detector.predict(text)
        
        if label == 1:  # Human
            human_scores.append(score)
        else:  # AI
            ai_scores.append(score)
    
    # Calculate statistics
    human_score_mean = np.mean(human_scores)
    human_score_std = np.std(human_scores)
    ai_score_mean = np.mean(ai_scores)
    ai_score_std = np.std(ai_scores)
    
    # False positives: humans classified as AI (score > 0.5)
    false_positives = sum(1 for score in human_scores if score > 0.5)
    false_positive_rate = false_positives / len(human_scores)
    
    # True positives: AI correctly classified as AI (score > 0.5)
    true_positives = sum(1 for score in ai_scores if score > 0.5)
    true_positive_rate = true_positives / len(ai_scores)
    
    # Discrimination: how well separated are the distributions
    discrimination = abs(ai_score_mean - human_score_mean)
    
    # AUROC approximation (assuming scores are probabilities)
    from sklearn.metrics import roc_auc_score
    # For AUROC: we want to predict AI (positive class)
    # Label: 0 = human, 1 = AI
    # Higher score should mean more likely AI (positive class)
    all_labels = [0] * len(human_scores) + [1] * len(ai_scores)  # 0=human, 1=AI
    all_scores = human_scores + ai_scores
    try:
        auroc = roc_auc_score(all_labels, all_scores)
    except:
        auroc = 0.5
    
    # Count distribution for human texts
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    human_hist, _ = np.histogram(human_scores, bins=bins)
    ai_hist, _ = np.histogram(ai_scores, bins=bins)
    
    return {
        "human_score_mean": human_score_mean,
        "human_score_std": human_score_std,
        "ai_score_mean": ai_score_mean,
        "ai_score_std": ai_score_std,
        "discrimination": discrimination,
        "false_positive_rate": false_positive_rate,
        "true_positive_rate": true_positive_rate,
        "auroc": auroc,
        "human_samples": len(human_scores),
        "ai_samples": len(ai_scores),
        "human_distribution": human_hist.tolist(),
        "ai_distribution": ai_hist.tolist(),
    }


async def main():
    """Main evaluation pipeline."""
    
    # Load MAGE test set
    mage_path = Path("data/mage/test")
    if not mage_path.exists():
        print(f"❌ MAGE dataset not found: {mage_path}")
        return
    
    print(f"\n📂 Loading MAGE test data from: {mage_path}")
    dataset = load_from_disk(str(mage_path))
    
    print(f"✓ Loaded {len(dataset)} total samples")
    print(f"  Columns: {dataset.column_names}")
    
    # Extract 100 human + 100 AI samples
    human_samples = [item for item in dataset if item['label'] == 1][:100]
    ai_samples = [item for item in dataset if item['label'] == 0][:100]
    
    test_data = human_samples + ai_samples
    
    print(f"\n✓ Extracted {len(human_samples)} human + {len(ai_samples)} AI samples")
    print(f"  Total: {len(test_data)} samples")
    
    # Initialize detectors
    cache = DetectorCache(cache_path="cache/eval_detectors.db")
    
    detectors = {
        # "Fast-DetectGPT (gpt-neo-2.7B)": FastDetectGPTDetector(
        #     cache,
        #     model_name="gpt-neo-2.7B"
        # ),
        "RoBERTa-large-openai-detector": RoBERTaOpenAIDetector(
            cache,
            model_name="roberta-large-openai-detector"
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
            
            print("\n📊 Results:")
            print(f"  Samples: {metrics['human_samples']} human + {metrics['ai_samples']} AI")
            print(f"\n  Human scores:  {metrics['human_score_mean']:.3f} ± {metrics['human_score_std']:.3f}")
            print(f"  AI scores:     {metrics['ai_score_mean']:.3f} ± {metrics['ai_score_std']:.3f}")
            print(f"  Discrimination: {metrics['discrimination']:.3f}")
            print(f"\n  AUROC:                {metrics['auroc']:.3f}")
            print(f"  False Positive Rate:  {metrics['false_positive_rate']:.1%} (human → AI)")
            print(f"  True Positive Rate:   {metrics['true_positive_rate']:.1%} (AI → AI)")
            
            print(f"\n  Human Score Distribution:")
            bins = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', 
                    '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
            for bin_name, count in zip(bins, metrics['human_distribution']):
                print(f"    {bin_name}: {count:3d} samples")
            
            print(f"\n  AI Score Distribution:")
            for bin_name, count in zip(bins, metrics['ai_distribution']):
                print(f"    {bin_name}: {count:3d} samples")
            
        except Exception as e:
            print(f"\n❌ Error testing {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary comparison
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print("\n{:<35} {:>8} {:>10} {:>10} {:>12}".format(
            "Detector", "AUROC", "Human", "AI", "Discrim"
        ))
        print("-" * 80)
        for name, metrics in results.items():
            print("{:<35} {:>8.3f} {:>10.3f} {:>10.3f} {:>12.3f}".format(
                name, 
                metrics['auroc'],
                metrics['human_score_mean'],
                metrics['ai_score_mean'],
                metrics['discrimination']
            ))
    
    # Save results
    output_file = Path("outputs/mage_detector_metrics.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("💡 Key Metrics Explained:")
    print("  - AUROC: 0.5=random, 1.0=perfect (higher is better)")
    print("  - Discrimination: |AI_score - Human_score| (higher is better)")
    print("  - False Positive: % humans wrongly labeled as AI (lower is better)")
    print("  - True Positive: % AI correctly labeled as AI (higher is better)")
    print("\n🎯 Interpretation:")
    print("  - Good detector: Low human scores, high AI scores, high AUROC")
    print("  - AUROC ~0.5 + low discrimination = detector is broken/guessing")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user")
        sys.exit(0)
