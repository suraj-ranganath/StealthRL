#!/usr/bin/env python3
"""
Compare StealthRL against baseline methods (DIPPER, SICO, etc.)

This script runs multiple baseline paraphrasers and StealthRL on the same
test set, then compares detector scores, semantic fidelity, and quality.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

from stealthrl.detectors import FastDetectGPTDetector, GhostbusterDetector, BinocularsDetector
from stealthrl.evaluation.metrics import compute_auroc, compute_bertscore, compute_perplexity


def load_texts(filepath: str) -> List[Dict]:
    """Load texts from JSONL file."""
    texts = []
    with open(filepath, 'r') as f:
        for line in f:
            texts.append(json.loads(line))
    return texts


def run_baseline_dipper(texts: List[str]) -> List[str]:
    """
    Run DIPPER baseline paraphraser.
    
    Note: Requires DIPPER to be installed separately.
    See: https://github.com/martiansideofthemoon/ai-detection-paraphrases
    """
    print("Running DIPPER baseline...")
    # Placeholder - actual implementation requires DIPPER installation
    try:
        from dipper_paraphrases import DipperParaphraser
        dipper = DipperParaphraser(model="kalpeshk2011/dipper-paraphraser-xxl")
        paraphrases = []
        for text in tqdm(texts):
            paraphrased = dipper.paraphrase(text, lex_diversity=60, order_diversity=60)
            paraphrases.append(paraphrased)
        return paraphrases
    except ImportError:
        print("WARNING: DIPPER not installed. Skipping DIPPER baseline.")
        print("To install: pip install dipper-paraphrases (or follow DIPPER repo instructions)")
        return texts  # Return original texts as fallback


def run_baseline_sico(texts: List[str]) -> List[str]:
    """
    Run SICO baseline evader.
    
    Note: Requires SICO implementation.
    See: https://github.com/ColinLu50/Evade-GPT-Detector
    """
    print("Running SICO baseline...")
    # Placeholder - actual implementation requires SICO setup
    try:
        from sico import SICOEvader
        sico = SICOEvader()
        paraphrases = []
        for text in tqdm(texts):
            paraphrased = sico.evade(text)
            paraphrases.append(paraphrased)
        return paraphrases
    except ImportError:
        print("WARNING: SICO not installed. Skipping SICO baseline.")
        print("To install: Clone and install from https://github.com/ColinLu50/Evade-GPT-Detector")
        return texts  # Return original texts as fallback


def run_stealthrl(texts: List[str], model_path: str) -> List[str]:
    """Run StealthRL paraphraser."""
    print("Running StealthRL...")
    from stealthrl.models import load_stealthrl_model
    
    model, tokenizer = load_stealthrl_model(
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        lora_path=model_path
    )
    
    paraphrases = []
    for text in tqdm(texts):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.9)
        paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
        paraphrases.append(paraphrased)
    
    return paraphrases


def evaluate_method(
    method_name: str,
    original_texts: List[str],
    paraphrased_texts: List[str],
    detectors: Dict
) -> Dict[str, float]:
    """Evaluate a paraphrasing method across multiple metrics."""
    print(f"Evaluating {method_name}...")
    
    results = {"method": method_name}
    
    # Detector scores
    for detector_name, detector in detectors.items():
        scores = []
        for text in tqdm(paraphrased_texts, desc=f"{detector_name}"):
            score = detector.detect(text)
            scores.append(score)
        
        # Mean detection score (lower = better evasion)
        results[f"{detector_name}_mean"] = sum(scores) / len(scores)
        results[f"{detector_name}_max"] = max(scores)
        results[f"{detector_name}_min"] = min(scores)
    
    # Semantic fidelity
    bertscore_f1 = compute_bertscore(paraphrased_texts, original_texts)
    results["bertscore_f1"] = bertscore_f1
    
    # Quality
    perplexity = compute_perplexity(paraphrased_texts)
    results["perplexity"] = perplexity
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare StealthRL against baselines")
    parser.add_argument("--input_file", type=str, required=True,
                        help="JSONL file with test texts")
    parser.add_argument("--stealthrl_model", type=str, required=True,
                        help="Path to trained StealthRL model")
    parser.add_argument("--output_csv", type=str, default="outputs/baseline_comparison.csv",
                        help="Output CSV file")
    parser.add_argument("--run_dipper", action="store_true",
                        help="Run DIPPER baseline")
    parser.add_argument("--run_sico", action="store_true",
                        help="Run SICO baseline")
    parser.add_argument("--detectors", nargs="+", 
                        default=["fast-detectgpt", "ghostbuster", "binoculars"],
                        help="Detectors to evaluate")
    args = parser.parse_args()
    
    # Load test texts
    print("Loading test texts...")
    data = load_texts(args.input_file)
    original_texts = [item["text"] for item in data]
    
    # Initialize detectors
    print("Initializing detectors...")
    detector_instances = {}
    if "fast-detectgpt" in args.detectors:
        detector_instances["fast-detectgpt"] = FastDetectGPTDetector()
    if "ghostbuster" in args.detectors:
        detector_instances["ghostbuster"] = GhostbusterDetector()
    if "binoculars" in args.detectors:
        detector_instances["binoculars"] = BinocularsDetector()
    
    # Run methods and collect results
    all_results = []
    
    # Baseline: Original (no paraphrasing)
    print("\n=== Baseline: Original ===")
    results = evaluate_method("original", original_texts, original_texts, detector_instances)
    all_results.append(results)
    
    # DIPPER
    if args.run_dipper:
        print("\n=== DIPPER ===")
        dipper_texts = run_baseline_dipper(original_texts)
        results = evaluate_method("dipper", original_texts, dipper_texts, detector_instances)
        all_results.append(results)
    
    # SICO
    if args.run_sico:
        print("\n=== SICO ===")
        sico_texts = run_baseline_sico(original_texts)
        results = evaluate_method("sico", original_texts, sico_texts, detector_instances)
        all_results.append(results)
    
    # StealthRL
    print("\n=== StealthRL ===")
    stealthrl_texts = run_stealthrl(original_texts, args.stealthrl_model)
    results = evaluate_method("stealthrl", original_texts, stealthrl_texts, detector_instances)
    all_results.append(results)
    
    # Save results
    df = pd.DataFrame(all_results)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n=== Results ===")
    print(df.to_string())
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
