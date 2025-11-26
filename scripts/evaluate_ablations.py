#!/usr/bin/env python3
"""
Evaluate all ablation experiments and generate comparison report.

This script loads all trained ablation models, evaluates them on the
same test set, and produces a comprehensive comparison table.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from stealthrl.models import load_stealthrl_model
from stealthrl.detectors import FastDetectGPTDetector, GhostbusterDetector, BinocularsDetector
from stealthrl.evaluation.metrics import compute_auroc, compute_bertscore, compute_perplexity, compute_fpr_gap


def load_test_data(filepath: str) -> List[Dict]:
    """Load test data from JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def evaluate_model(
    model_path: str,
    model_name: str,
    test_texts: List[str],
    esl_texts: List[str],
    native_texts: List[str],
    detectors: Dict
) -> Dict[str, float]:
    """Evaluate a single model on all metrics."""
    print(f"\n=== Evaluating: {model_name} ===")
    
    # Load model
    try:
        model, tokenizer = load_stealthrl_model(
            base_model="Qwen/Qwen2.5-1.5B-Instruct",
            lora_path=model_path
        )
    except Exception as e:
        print(f"ERROR loading model {model_name}: {e}")
        return {"model": model_name, "error": str(e)}
    
    results = {"model": model_name}
    
    # Generate paraphrases
    print("Generating paraphrases...")
    paraphrases = []
    for text in tqdm(test_texts):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.9)
        paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
        paraphrases.append(paraphrased)
    
    # Evaluate detectors (transfer to held-out detectors)
    print("Running detectors...")
    for detector_name, detector in detectors.items():
        scores = []
        for text in tqdm(paraphrases, desc=detector_name):
            score = detector.detect(text)
            scores.append(score)
        results[f"{detector_name}_mean"] = sum(scores) / len(scores)
    
    # Semantic fidelity
    print("Computing BERTScore...")
    bertscore = compute_bertscore(paraphrases, test_texts)
    results["bertscore_f1"] = bertscore
    
    # Quality
    print("Computing perplexity...")
    perplexity = compute_perplexity(paraphrases)
    results["perplexity"] = perplexity
    
    # Fairness (ESL vs native gap)
    print("Computing fairness metrics...")
    # Generate paraphrases for ESL and native subsets
    esl_paraphrases = []
    for text in tqdm(esl_texts[:100], desc="ESL"):  # Limit for speed
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True)
        paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
        esl_paraphrases.append(paraphrased)
    
    native_paraphrases = []
    for text in tqdm(native_texts[:100], desc="Native"):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True)
        paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
        native_paraphrases.append(paraphrased)
    
    # Compute FPR gap on primary detector
    primary_detector = list(detectors.values())[0]
    esl_scores = [primary_detector.detect(text) for text in esl_paraphrases]
    native_scores = [primary_detector.detect(text) for text in native_paraphrases]
    fpr_gap = compute_fpr_gap(esl_scores, native_scores, threshold=0.5)
    results["fpr_gap"] = fpr_gap
    
    return results


def plot_ablation_comparison(df: pd.DataFrame, output_dir: Path):
    """Generate comparison plots for ablations."""
    print("\nGenerating comparison plots...")
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Detector scores comparison
    detector_cols = [col for col in df.columns if col.endswith("_mean")]
    if detector_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        df_plot = df[["model"] + detector_cols].set_index("model")
        df_plot.plot(kind="bar", ax=ax)
        ax.set_title("Detector Scores by Ablation (Lower = Better Evasion)")
        ax.set_ylabel("Mean Detection Score")
        ax.set_xlabel("Ablation")
        ax.legend(title="Detector", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / "ablation_detector_scores.png", dpi=300)
        print(f"Saved: ablation_detector_scores.png")
    
    # 2. Semantic fidelity comparison
    if "bertscore_f1" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        df.plot(x="model", y="bertscore_f1", kind="bar", ax=ax, legend=False)
        ax.set_title("Semantic Fidelity by Ablation (Higher = Better)")
        ax.set_ylabel("BERTScore F1")
        ax.set_xlabel("Ablation")
        ax.axhline(y=0.8, color='r', linestyle='--', label='Target Threshold')
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / "ablation_bertscore.png", dpi=300)
        print(f"Saved: ablation_bertscore.png")
    
    # 3. Fairness gap comparison
    if "fpr_gap" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        df.plot(x="model", y="fpr_gap", kind="bar", ax=ax, legend=False, color='coral')
        ax.set_title("ESL Fairness Gap by Ablation (Lower = More Fair)")
        ax.set_ylabel("FPR Gap (ESL - Native)")
        ax.set_xlabel("Ablation")
        ax.axhline(y=0, color='g', linestyle='--', label='No Gap (Ideal)')
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / "ablation_fairness_gap.png", dpi=300)
        print(f"Saved: ablation_fairness_gap.png")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ablation experiments")
    parser.add_argument("--ablation_dir", type=str, default="checkpoints",
                        help="Directory containing ablation checkpoints")
    parser.add_argument("--test_data", type=str, default="data/processed/test.jsonl",
                        help="Test data JSONL file")
    parser.add_argument("--esl_data", type=str, default="data/processed/esl_test.jsonl",
                        help="ESL test data")
    parser.add_argument("--native_data", type=str, default="data/processed/native_test.jsonl",
                        help="Native test data")
    parser.add_argument("--output_dir", type=str, default="outputs/ablations",
                        help="Output directory for results")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    print("Loading test data...")
    test_data = load_test_data(args.test_data)
    test_texts = [item["text"] for item in test_data]
    
    esl_data = load_test_data(args.esl_data)
    esl_texts = [item["text"] for item in esl_data]
    
    native_data = load_test_data(args.native_data)
    native_texts = [item["text"] for item in native_data]
    
    # Initialize detectors (held-out for transfer evaluation)
    print("Initializing detectors...")
    detectors = {
        "fast-detectgpt": FastDetectGPTDetector(),
        "ghostbuster": GhostbusterDetector(),
        "binoculars": BinocularsDetector(),
    }
    
    # Find all ablation checkpoints
    ablation_base = Path(args.ablation_dir)
    ablation_models = {
        "baseline": ablation_base / "stealthrl-small",
        "single-detector": ablation_base / "ablation-single-fast-detectgpt",
        "no-fairness": ablation_base / "ablation-no-fairness",
        "no-semantic": ablation_base / "ablation-no-semantic",
        "no-quality": ablation_base / "ablation-no-quality",
        "detector-only": ablation_base / "ablation-detector-only",
    }
    
    # Evaluate each model
    all_results = []
    for model_name, model_path in ablation_models.items():
        if not model_path.exists():
            print(f"WARNING: Model not found: {model_path}")
            continue
        
        results = evaluate_model(
            str(model_path),
            model_name,
            test_texts[:100],  # Limit for speed
            esl_texts,
            native_texts,
            detectors
        )
        all_results.append(results)
    
    # Save results
    df = pd.DataFrame(all_results)
    csv_path = output_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n=== Results saved to {csv_path} ===")
    print(df.to_string())
    
    # Generate plots
    plot_ablation_comparison(df, output_dir)
    
    print(f"\n✓ Ablation evaluation complete!")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
