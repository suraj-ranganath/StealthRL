#!/usr/bin/env python3
"""
Evaluate detectors on Tinker custom dataset.
Extract 200 samples (100 human + 100 AI) and test both detectors.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm.asyncio import tqdm

try:
    from datasets import load_from_disk
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from stealthrl.tinker.detectors import (
    DetectorCache,
    FastDetectGPTDetector,
    RoBERTaOpenAIDetector,
    GhostbusterDetector,
    BinocularsDetector,
)


async def evaluate_detector(detector, test_data: List[Dict], max_samples: int = 200, batch_size: int = 16, data_format: str = "tinker"):
    """Evaluate detector on both human and AI texts with batch processing."""
    
    print(f"\nEvaluating on {len(test_data)} samples (batch_size={batch_size})...")
    print("This may take 2-5 minutes on first run (downloading model)...")
    
    human_scores = []
    ai_scores = []
    
    if data_format == "mage":
        # MAGE format: each item has 'text' and 'label' (1=human, 0=AI)
        texts = [item["text"] for item in test_data]
        labels = [item["label"] for item in test_data]
        
        # Process in batches for faster evaluation
        all_scores = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
            batch_texts = texts[i:i+batch_size]
            # Process batch concurrently
            batch_scores = await asyncio.gather(*[detector.predict(text) for text in batch_texts])
            all_scores.extend(batch_scores)
        
        # Separate by label
        for score, label in zip(all_scores, labels):
            if label == 1:  # Human
                human_scores.append(score)
            else:  # AI
                ai_scores.append(score)
    else:
        # Tinker format: each item has 'human_reference' and 'ai_text'
        human_texts = [item["human_reference"] for item in test_data]
        ai_texts = [item["ai_text"] for item in test_data]
        
        # Process human texts in batches
        for i in tqdm(range(0, len(human_texts), batch_size), desc="Evaluating human"):
            batch = human_texts[i:i+batch_size]
            batch_scores = await asyncio.gather(*[detector.predict(text) for text in batch])
            human_scores.extend(batch_scores)
        
        # Process AI texts in batches
        for i in tqdm(range(0, len(ai_texts), batch_size), desc="Evaluating AI"):
            batch = ai_texts[i:i+batch_size]
            batch_scores = await asyncio.gather(*[detector.predict(text) for text in batch])
            ai_scores.extend(batch_scores)
    
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


async def main(detector_names: List[str] = None, prewarm: bool = False, max_samples: int = 100, dataset_name: str = "tinker", batch_size: int = 16):
    """Main evaluation pipeline."""
    
    # Load dataset based on name
    if dataset_name == "mage":
        if not HAS_DATASETS:
            print("❌ 'datasets' package not installed. Please install: pip install datasets")
            print("   Or run: pip install -r requirements.txt")
            return
        
        mage_path = Path("data/mage/test")
        if not mage_path.exists():
            print(f"❌ MAGE dataset not found: {mage_path}")
            return
        
        print(f"\n📂 Loading MAGE test data from: {mage_path}")
        dataset_hf = load_from_disk(str(mage_path))
        
        print(f"✓ Loaded {len(dataset_hf)} total samples")
        
        # Extract balanced samples (label 1=human, 0=AI)
        human_samples = [item for item in dataset_hf if item['label'] == 1][:max_samples]
        ai_samples = [item for item in dataset_hf if item['label'] == 0][:max_samples]
        test_samples = human_samples + ai_samples
        
        print(f"\n✓ Extracted {len(human_samples)} human + {len(ai_samples)} AI samples")
        print(f"  Total evaluations: {len(test_samples)} texts")
        data_format = "mage"
    else:
        # Load Tinker test set
        tinker_path = Path("data/tinker/test.jsonl")
        if not tinker_path.exists():
            print(f"❌ Tinker dataset not found: {tinker_path}")
            return
        
        print(f"\n📂 Loading Tinker test data from: {tinker_path}")
        
        # Load JSONL data
        dataset = []
        with open(tinker_path, 'r') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        
        print(f"✓ Loaded {len(dataset)} total samples")
        
        # Extract samples (each has both human and AI text)
        # Total = max_samples samples = max_samples human texts + max_samples AI texts
        sample_size = max_samples
        test_samples = dataset[:sample_size]
        
        print(f"\n✓ Extracted {len(test_samples)} samples")
        print(f"  Each sample has: human_reference + ai_text")
        print(f"  Total evaluations: {len(test_samples) * 2} ({len(test_samples)} human + {len(test_samples)} AI)")
        data_format = "tinker"
    
    # Show sample metadata (Tinker only)
    if data_format == "tinker":
        domains = {}
        sources = {}
        for sample in test_samples:
            domain = sample.get('domain', 'unknown')
            source = sample.get('metadata', {}).get('source', 'unknown')
            domains[domain] = domains.get(domain, 0) + 1
            sources[source] = sources.get(source, 0) + 1
        
        print(f"\n  Domain distribution:")
        for domain, count in sorted(domains.items()):
            print(f"    {domain}: {count} samples")
        
        print(f"\n  Source distribution:")
        for source, count in sorted(sources.items()):
            print(f"    {source}: {count} samples")
    
    # Initialize detectors
    cache = DetectorCache(cache_path="cache/eval_detectors.db")
    
    # Define available detectors
    available_detectors = {
        "roberta_openai": lambda: RoBERTaOpenAIDetector(
            cache,
            model_name="roberta-large-openai-detector"
        ),
        "fast_detectgpt": lambda: FastDetectGPTDetector(
            cache,
            model_name="gpt-neo-2.7B"
        ),
        "ghostbuster": lambda: GhostbusterDetector(
            cache,
            model_name="roberta-base"
        ),
        "binoculars": lambda: BinocularsDetector(
            cache,
            performer_model="gpt2",
            observer_model="gpt2-medium"
        ),  # NOTE: GPT-2 lightweight mode has poor performance. Use Falcon-7B for production.
    }
    
    # Select detectors based on command-line args or use all
    if detector_names:
        detector_keys = [name.lower() for name in detector_names]
        # Validate detector names
        invalid = [name for name in detector_keys if name not in available_detectors]
        if invalid:
            print(f"\n❌ Invalid detector names: {', '.join(invalid)}")
            print(f"Available detectors: {', '.join(available_detectors.keys())}")
            return
    else:
        detector_keys = list(available_detectors.keys())
    
    # Initialize selected detectors
    detectors = {}
    detector_display_names = {
        "roberta_openai": "RoBERTa-large-openai-detector",
        "fast_detectgpt": "Fast-DetectGPT (gpt-neo-2.7B)",
        "ghostbuster": "Ghostbuster (roberta-base)",
        "binoculars": "Binoculars (gpt2/gpt2-medium)",
    }
    
    for key in detector_keys:
        display_name = detector_display_names.get(key, key)
        detectors[display_name] = available_detectors[key]()
    
    # Prewarm detectors if requested
    if prewarm:
        print("\n🔥 Prewarming detectors (loading models)...")
        for name, detector in detectors.items():
            try:
                print(f"  Loading {name}...")
                # Call _load_models if it exists
                if hasattr(detector, '_load_models'):
                    detector._load_models()
                print(f"  ✓ {name} loaded")
            except Exception as e:
                print(f"  ❌ Failed to prewarm {name}: {e}")
    
    results = {}
    
    for name, detector in detectors.items():
        print("\n" + "=" * 80)
        print(f"Testing: {name}")
        print("=" * 80)
        
        try:
            metrics = await evaluate_detector(detector, test_samples, max_samples=max_samples, batch_size=batch_size, data_format=data_format)
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
    output_file = Path(f"outputs/{dataset_name}_detector_metrics.json")
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
    print("\n📊 Dataset Info:")
    print(f"  - Dataset: {dataset_name}")
    if data_format == "tinker":
        print(f"  - Total available: {len(dataset)} samples")
        print(f"  - Evaluated: {len(test_samples)} samples ({len(test_samples) * 2} texts total)")
    else:
        print(f"  - Total available: {len(dataset_hf)} samples")
        print(f"  - Evaluated: {len(test_samples)} texts total")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate detectors on Tinker custom dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Available detectors:
  roberta_openai  - RoBERTa-large-openai-detector
  fast_detectgpt  - Fast-DetectGPT (gpt-neo-2.7B)
  ghostbuster     - Ghostbuster (roberta-base)
  binoculars      - Binoculars (gpt2/gpt2-medium)

Examples:
  # Evaluate all detectors on 100 samples
  python eval_tinker_detector.py
  
  # Evaluate specific detectors on 200 samples
  python eval_tinker_detector.py --detectors roberta_openai binoculars --max-samples 200
  
  # Prewarm models before evaluation
  python eval_tinker_detector.py --detectors fast_detectgpt --prewarm
        """
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=["roberta_openai", "fast_detectgpt", "ghostbuster", "binoculars"],
        help="Detectors to evaluate (default: all)"
    )
    parser.add_argument(
        "--prewarm",
        action="store_true",
        help="Prewarm (load) all models before starting evaluation"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples to evaluate (default: 100)"
    )
    parser.add_argument(
        "--dataset",
        choices=["tinker", "mage"],
        default="tinker",
        help="Dataset to evaluate on (default: tinker)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for parallel evaluation (default: 16)"
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main(
            detector_names=args.detectors,
            prewarm=args.prewarm,
            max_samples=args.max_samples,
            dataset_name=args.dataset,
            batch_size=args.batch_size
        ))
    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user")
        sys.exit(0)
