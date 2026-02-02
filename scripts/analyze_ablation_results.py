#!/usr/bin/env python3
"""
Analyze ablation study results and plot Pareto frontier.

This script:
1. Loads results from all ablation runs
2. Computes final metrics (detector evasion, semantic similarity)
3. Plots Pareto frontier
4. Identifies best configurations
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


ABLATION_OUTPUT_DIR = Path("outputs/ablations")


def load_run_metrics(run_dir: Path) -> Dict:
    """Load final metrics from a training run."""
    metrics_file = run_dir / "metrics.jsonl"
    
    if not metrics_file.exists():
        print(f"Warning: No metrics found for {run_dir.name}")
        return None
    
    # Read last line (final metrics)
    with open(metrics_file, 'r') as f:
        lines = f.readlines()
        if not lines:
            return None
        
        # Try last few lines in case of incomplete writes
        for line in reversed(lines[-5:]):
            try:
                metrics = json.loads(line)
                return metrics
            except json.JSONDecodeError:
                continue
    
    return None


def extract_key_metrics(metrics: Dict) -> Dict:
    """Extract key metrics for analysis."""
    if not metrics:
        return None
    
    return {
        # Detector evasion (lower is better for attacker)
        "detector_prob": metrics.get("test/detector_prob/mean", metrics.get("detector_prob", 0.5)),
        "detector_evasion": 1.0 - metrics.get("test/detector_prob/mean", metrics.get("detector_prob", 0.5)),
        
        # Individual detectors
        "roberta_score": metrics.get("test/detector/roberta_openai/mean", 0.5),
        "fast_detectgpt_score": metrics.get("test/detector/fast_detectgpt/mean", 0.5),
        
        # Semantic similarity
        "semantic_sim": metrics.get("test/semantic_sim/mean", metrics.get("semantic_sim", 0.0)),
        
        # Total reward
        "total_reward": metrics.get("test/reward/mean", metrics.get("total_reward", 0.0)),
        
        # Training steps
        "step": metrics.get("step", 0),
    }


def load_all_results() -> List[Dict]:
    """Load results from all ablation runs."""
    results = []
    
    if not ABLATION_OUTPUT_DIR.exists():
        print(f"Error: Ablation output dir not found: {ABLATION_OUTPUT_DIR}")
        return results
    
    for run_dir in sorted(ABLATION_OUTPUT_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        
        # Parse weights from directory name (e.g., det1.0_sem1.0)
        name = run_dir.name
        try:
            parts = name.split('_')
            det_weight = float(parts[0].replace('det', ''))
            sem_weight = float(parts[1].replace('sem', ''))
        except (IndexError, ValueError):
            print(f"Warning: Could not parse weights from {name}")
            continue
        
        # Load metrics
        metrics = load_run_metrics(run_dir)
        if not metrics:
            continue
        
        key_metrics = extract_key_metrics(metrics)
        if not key_metrics:
            continue
        
        results.append({
            "name": name,
            "detector_weight": det_weight,
            "semantic_weight": sem_weight,
            **key_metrics,
        })
    
    return results


def identify_pareto_frontier(results: List[Dict]) -> List[Dict]:
    """Identify Pareto-optimal configurations."""
    if not results:
        return []
    
    pareto = []
    
    for candidate in results:
        dominated = False
        
        for other in results:
            if other == candidate:
                continue
            
            # Check if 'other' dominates 'candidate'
            # Maximize: detector_evasion, semantic_sim
            better_evasion = other["detector_evasion"] >= candidate["detector_evasion"]
            better_semantic = other["semantic_sim"] >= candidate["semantic_sim"]
            strictly_better = (
                (other["detector_evasion"] > candidate["detector_evasion"]) or
                (other["semantic_sim"] > candidate["semantic_sim"])
            )
            
            if better_evasion and better_semantic and strictly_better:
                dominated = True
                break
        
        if not dominated:
            pareto.append(candidate)
    
    return pareto


def plot_results(results: List[Dict], pareto: List[Dict], output_file: str = "ablation_pareto.png"):
    """Plot Pareto frontier and all results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Extract data
    all_evasion = [r["detector_evasion"] for r in results]
    all_semantic = [r["semantic_sim"] for r in results]
    all_reward = [r["total_reward"] for r in results]
    
    pareto_evasion = [r["detector_evasion"] for r in pareto]
    pareto_semantic = [r["semantic_sim"] for r in pareto]
    
    # Plot 1: Pareto frontier
    ax = axes[0, 0]
    ax.scatter(all_semantic, all_evasion, alpha=0.6, s=100, label="All configs")
    ax.scatter(pareto_semantic, pareto_evasion, c='red', s=150, marker='*', 
               label="Pareto frontier", edgecolors='black', linewidths=2)
    
    # Annotate Pareto points
    for r in pareto:
        ax.annotate(
            f"α={r['detector_weight']:.1f}\nβ={r['semantic_weight']:.1f}",
            xy=(r["semantic_sim"], r["detector_evasion"]),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
        )
    
    ax.set_xlabel("Semantic Similarity", fontsize=12)
    ax.set_ylabel("Detector Evasion (1 - P(AI))", fontsize=12)
    ax.set_title("Pareto Frontier: Evasion vs Fidelity", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Weight heatmap
    ax = axes[0, 1]
    det_weights = sorted(set(r["detector_weight"] for r in results))
    sem_weights = sorted(set(r["semantic_weight"] for r in results))
    
    reward_grid = np.full((len(sem_weights), len(det_weights)), np.nan)
    for r in results:
        i = sem_weights.index(r["semantic_weight"])
        j = det_weights.index(r["detector_weight"])
        reward_grid[i, j] = r["total_reward"]
    
    im = ax.imshow(reward_grid, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xticks(range(len(det_weights)))
    ax.set_yticks(range(len(sem_weights)))
    ax.set_xticklabels([f"{w:.1f}" for w in det_weights])
    ax.set_yticklabels([f"{w:.1f}" for w in sem_weights])
    ax.set_xlabel("Detector Weight (α)", fontsize=12)
    ax.set_ylabel("Semantic Weight (β)", fontsize=12)
    ax.set_title("Total Reward Heatmap", fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    # Plot 3: Individual detector breakdown
    ax = axes[1, 0]
    roberta_scores = [r["roberta_score"] for r in results]
    fast_detectgpt_scores = [r["fast_detectgpt_score"] for r in results]
    
    indices = np.arange(len(results))
    width = 0.35
    
    ax.bar(indices - width/2, roberta_scores, width, label='RoBERTa', alpha=0.8)
    ax.bar(indices + width/2, fast_detectgpt_scores, width, label='Fast-DetectGPT', alpha=0.8)
    
    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("AI Probability", fontsize=12)
    ax.set_title("Detector Scores by Configuration", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Semantic vs Reward
    ax = axes[1, 1]
    scatter = ax.scatter(all_semantic, all_reward, c=all_evasion, s=100, cmap='coolwarm', alpha=0.7)
    ax.set_xlabel("Semantic Similarity", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title("Reward vs Semantic Quality", fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label="Detector Evasion")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    
    return fig


def print_summary(results: List[Dict], pareto: List[Dict]):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    
    print(f"\nTotal configurations: {len(results)}")
    print(f"Pareto-optimal configurations: {len(pareto)}")
    
    print("\n" + "-"*80)
    print("PARETO FRONTIER (Non-dominated configurations):")
    print("-"*80)
    
    # Sort by detector evasion
    pareto_sorted = sorted(pareto, key=lambda x: x["detector_evasion"], reverse=True)
    
    print(f"{'Config':<20} {'α':>6} {'β':>6} {'Evasion':>9} {'Semantic':>9} {'Reward':>9}")
    print("-"*80)
    
    for r in pareto_sorted:
        print(
            f"{r['name']:<20} "
            f"{r['detector_weight']:>6.1f} "
            f"{r['semantic_weight']:>6.1f} "
            f"{r['detector_evasion']:>9.3f} "
            f"{r['semantic_sim']:>9.3f} "
            f"{r['total_reward']:>9.3f}"
        )
    
    print("\n" + "-"*80)
    print("BEST CONFIGURATIONS:")
    print("-"*80)
    
    # Best evasion
    best_evasion = max(results, key=lambda x: x["detector_evasion"])
    print(f"\n1. Best Detector Evasion:")
    print(f"   Config: {best_evasion['name']}")
    print(f"   Weights: α={best_evasion['detector_weight']:.1f}, β={best_evasion['semantic_weight']:.1f}")
    print(f"   Evasion: {best_evasion['detector_evasion']:.3f}")
    print(f"   Semantic: {best_evasion['semantic_sim']:.3f}")
    
    # Best semantic
    best_semantic = max(results, key=lambda x: x["semantic_sim"])
    print(f"\n2. Best Semantic Similarity:")
    print(f"   Config: {best_semantic['name']}")
    print(f"   Weights: α={best_semantic['detector_weight']:.1f}, β={best_semantic['semantic_weight']:.1f}")
    print(f"   Evasion: {best_semantic['detector_evasion']:.3f}")
    print(f"   Semantic: {best_semantic['semantic_sim']:.3f}")
    
    # Best balanced (high on both)
    balanced_scores = [(r["detector_evasion"] + r["semantic_sim"]) / 2 for r in results]
    best_balanced_idx = np.argmax(balanced_scores)
    best_balanced = results[best_balanced_idx]
    print(f"\n3. Best Balanced (Highest Average):")
    print(f"   Config: {best_balanced['name']}")
    print(f"   Weights: α={best_balanced['detector_weight']:.1f}, β={best_balanced['semantic_weight']:.1f}")
    print(f"   Evasion: {best_balanced['detector_evasion']:.3f}")
    print(f"   Semantic: {best_balanced['semantic_sim']:.3f}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument(
        "--output",
        default="ablation_pareto.png",
        help="Output plot file",
    )
    
    args = parser.parse_args()
    
    print("Loading ablation results...")
    results = load_all_results()
    
    if not results:
        print("Error: No results found!")
        print(f"Expected directory: {ABLATION_OUTPUT_DIR}")
        return
    
    print(f"Loaded {len(results)} completed runs")
    
    # Identify Pareto frontier
    pareto = identify_pareto_frontier(results)
    
    # Plot results
    plot_results(results, pareto, output_file=args.output)
    
    # Print summary
    print_summary(results, pareto)


if __name__ == "__main__":
    main()
