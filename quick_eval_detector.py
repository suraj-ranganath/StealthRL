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
    """Evaluate detector on test data."""
    
    # Limit samples for quick evaluation
    test_data = test_data[:max_samples]
    
    print(f"\nEvaluating on {len(test_data)} samples...")
    print("This may take 5-10 minutes on first run (downloading model)...")
    
    human_scores = []
    ai_scores = []
    predictions = []
    labels = []  # 0 = human, 1 = AI
    
    for item in tqdm(test_data, desc="Evaluating"):
        human_text = item["human_reference"]
        ai_text = item["ai_text"]
        
        # Score both texts
        human_score = await detector.predict(human_text)
        ai_score = await detector.predict(ai_text)
        
        human_scores.append(human_score)
        ai_scores.append(ai_score)
        
        # Binary predictions (threshold = 0.5)
        predictions.extend([
            1 if human_score > 0.5 else 0,  # Human text prediction
            1 if ai_score > 0.5 else 0       # AI text prediction
        ])
        labels.extend([0, 1])  # Human, AI
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    # AUROC
    all_scores = human_scores + ai_scores
    auroc = roc_auc_score(labels, all_scores)
    
    # Attack Success Rate (ASR) - how often AI text is misclassified as human
    ai_predictions = [predictions[i] for i in range(1, len(predictions), 2)]
    asr = 1.0 - sum(ai_predictions) / len(ai_predictions)  # % classified as human
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "asr": asr,
        "human_score_mean": np.mean(human_scores),
        "human_score_std": np.std(human_scores),
        "ai_score_mean": np.mean(ai_scores),
        "ai_score_std": np.std(ai_scores),
        "discrimination": abs(np.mean(ai_scores) - np.mean(human_scores)),
    }


async def main():
    """Run quick evaluation."""
    
    print("=" * 80)
    print("QUICK DETECTOR EVALUATION")
    print("=" * 80)
    
    # Load test data
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
            
            print("\n📊 Results:")
            print(f"  Accuracy:      {metrics['accuracy']:.3f}")
            print(f"  Precision:     {metrics['precision']:.3f}")
            print(f"  Recall:        {metrics['recall']:.3f}")
            print(f"  F1 Score:      {metrics['f1']:.3f}")
            print(f"  AUROC:         {metrics['auroc']:.3f}")
            print(f"  ASR (1-acc):   {metrics['asr']:.3f}")
            print(f"\n  Human scores:  {metrics['human_score_mean']:.3f} ± {metrics['human_score_std']:.3f}")
            print(f"  AI scores:     {metrics['ai_score_mean']:.3f} ± {metrics['ai_score_std']:.3f}")
            print(f"  Discrimination: {metrics['discrimination']:.3f}")
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary comparison
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print("\n{:<35} {:>10} {:>10} {:>10}".format("Detector", "AUROC", "F1", "ASR"))
        print("-" * 80)
        for name, metrics in results.items():
            print("{:<35} {:>10.3f} {:>10.3f} {:>10.3f}".format(
                name, metrics['auroc'], metrics['f1'], metrics['asr']
            ))
    
    # Save results
    output_file = Path("outputs/baseline_detector_metrics.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✓ Results saved to: {output_file}")
    print("=" * 80)
    
    print("\n💡 Key Metrics Explained:")
    print("  - AUROC: Area under ROC curve (higher = better, 0.5 = random)")
    print("  - F1: Harmonic mean of precision/recall (higher = better)")
    print("  - ASR: Attack Success Rate = % of AI texts detected as human")
    print("       (Lower ASR = detector is harder to fool)")
    print("\n🎯 Goal of StealthRL:")
    print("  Train model to increase ASR (make AI text look more human)")
    print("  While maintaining semantic similarity to original")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user")
        sys.exit(0)
