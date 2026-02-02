#!/usr/bin/env python3
"""
Comprehensive visualization script for StealthRL training results.
Creates publication-quality plots including training curves, Pareto frontiers,
and trade-off analysis for presentation.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'serif'

def load_metrics(metrics_path: str) -> pd.DataFrame:
    """Load metrics from JSONL file."""
    data = []
    with open(metrics_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """
    Find Pareto efficient points (minimize all objectives).
    
    Args:
        costs: An (n_points, n_costs) array
        
    Returns:
        A boolean array of Pareto efficient points
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep points if no other point dominates
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient

def plot_training_curves(df: pd.DataFrame, output_dir: Path):
    """Plot training progression curves."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('StealthRL Training Progression (GRPO)', fontsize=20, fontweight='bold')
    
    # 1. Total Reward
    ax = axes[0, 0]
    ax.plot(df['step'], df['env/all/reward/total'], 'b-', linewidth=2, label='Train')
    
    # Support both old and new metric formats
    test_reward_col = 'test/reward/mean' if 'test/reward/mean' in df.columns else 'test/env/all/reward/total'
    eval_reward_col = 'eval/reward/mean' if 'eval/reward/mean' in df.columns else 'eval/env/all/reward/total'
    
    if test_reward_col in df.columns:
        test_steps = df[df[test_reward_col].notna()]['step']
        test_rewards = df[df[test_reward_col].notna()][test_reward_col]
        ax.plot(test_steps, test_rewards, 'r--', linewidth=2, marker='o', markersize=6, label='Test')
    if eval_reward_col in df.columns:
        eval_steps = df[df[eval_reward_col].notna()]['step']
        eval_rewards = df[df[eval_reward_col].notna()][eval_reward_col]
        ax.plot(eval_steps, eval_rewards, 'g--', linewidth=2, marker='s', markersize=6, label='Eval')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Total Reward')
    ax.set_title('Total Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Detector Evasion (1 - P(AI))
    ax = axes[0, 1]
    train_evasion = 1 - df['env/all/detector_prob']
    ax.plot(df['step'], train_evasion, 'b-', linewidth=2, label='Train')
    
    # Support both old and new formats
    test_det_col = 'test/detector_prob/mean' if 'test/detector_prob/mean' in df.columns else 'test/env/all/detector_prob'
    eval_det_col = 'eval/detector_prob/mean' if 'eval/detector_prob/mean' in df.columns else 'eval/env/all/detector_prob'
    
    if test_det_col in df.columns:
        test_steps = df[df[test_det_col].notna()]['step']
        test_evasion = 1 - df[df[test_det_col].notna()][test_det_col]
        ax.plot(test_steps, test_evasion, 'r--', linewidth=2, marker='o', markersize=6, label='Test')
    
    if eval_det_col in df.columns:
        eval_steps = df[df[eval_det_col].notna()]['step']
        eval_evasion = 1 - df[df[eval_det_col].notna()][eval_det_col]
        ax.plot(eval_steps, eval_evasion, 'g--', linewidth=2, marker='s', markersize=6, label='Eval')
    
    ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, label='Random')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Detector Evasion (1 - P(AI))')
    ax.set_title('Detector Evasion Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Semantic Similarity
    ax = axes[0, 2]
    ax.plot(df['step'], df['env/all/semantic_sim'], 'b-', linewidth=2, label='Train')
    
    # Support both old and new formats
    test_sem_col = 'test/semantic_sim/mean' if 'test/semantic_sim/mean' in df.columns else 'test/env/all/semantic_sim'
    eval_sem_col = 'eval/semantic_sim/mean' if 'eval/semantic_sim/mean' in df.columns else 'eval/env/all/semantic_sim'
    
    if test_sem_col in df.columns:
        test_steps = df[df[test_sem_col].notna()]['step']
        test_sem = df[df[test_sem_col].notna()][test_sem_col]
        ax.plot(test_steps, test_sem, 'r--', linewidth=2, marker='o', markersize=6, label='Test')
    
    if eval_sem_col in df.columns:
        eval_steps = df[df[eval_sem_col].notna()]['step']
        eval_sem = df[df[eval_sem_col].notna()][eval_sem_col]
        ax.plot(eval_steps, eval_sem, 'g--', linewidth=2, marker='s', markersize=6, label='Eval')
    
    ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='Target (0.95)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Semantic Similarity')
    ax.set_title('Semantic Quality Preservation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Perplexity (if available)
    ax = axes[1, 0]
    if 'env/all/perplexity' in df.columns:
        ax.plot(df['step'], df['env/all/perplexity'], 'b-', linewidth=2, label='Train')
        
        test_ppl_col = 'test/perplexity/mean' if 'test/perplexity/mean' in df.columns else 'test/env/all/perplexity'
        eval_ppl_col = 'eval/perplexity/mean' if 'eval/perplexity/mean' in df.columns else 'eval/env/all/perplexity'
        
        if test_ppl_col in df.columns:
            test_steps = df[df[test_ppl_col].notna()]['step']
            test_ppl = df[df[test_ppl_col].notna()][test_ppl_col]
            ax.plot(test_steps, test_ppl, 'r--', linewidth=2, marker='o', markersize=6, label='Test')
        
        if eval_ppl_col in df.columns:
            eval_steps = df[df[eval_ppl_col].notna()]['step']
            eval_ppl = df[df[eval_ppl_col].notna()][eval_ppl_col]
            ax.plot(eval_steps, eval_ppl, 'g--', linewidth=2, marker='s', markersize=6, label='Eval')
        
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Target (30)')
        ax.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='Warning (80)')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Perplexity')
        ax.set_title('Text Naturalness (Perplexity)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Perplexity not computed', ha='center', va='center', fontsize=14, color='gray')
        ax.axis('off')
    
    # 5. Entropy (exploration)
    ax = axes[1, 1]
    if 'optim/entropy' in df.columns:
        ax.plot(df['step'], df['optim/entropy'], 'purple', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Policy Entropy')
        ax.set_title('Exploration Level')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Entropy not available', ha='center', va='center', fontsize=14, color='gray')
        ax.axis('off')
    
    # 6. Parse Success Rate
    ax = axes[1, 2]
    ax.plot(df['step'], df['env/all/parse_success'] * 100, 'b-', linewidth=2, label='Train')
    
    # Support both old and new formats
    test_parse_col = 'test/valid_output_rate/mean' if 'test/valid_output_rate/mean' in df.columns else 'test/env/all/parse_success'
    eval_parse_col = 'eval/valid_output_rate/mean' if 'eval/valid_output_rate/mean' in df.columns else 'eval/env/all/parse_success'
    
    if test_parse_col in df.columns:
        test_steps = df[df[test_parse_col].notna()]['step']
        test_parse = df[df[test_parse_col].notna()][test_parse_col] * 100
        ax.plot(test_steps, test_parse, 'r--', linewidth=2, marker='o', markersize=6, label='Test')
    
    if eval_parse_col in df.columns:
        eval_steps = df[df[eval_parse_col].notna()]['step']
        eval_parse = df[df[eval_parse_col].notna()][eval_parse_col] * 100
        ax.plot(eval_steps, eval_parse, 'g--', linewidth=2, marker='s', markersize=6, label='Eval')
    
    ax.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='Target (90%)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Valid Output Rate (%)')
    ax.set_title('Valid Output Generation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. KL Divergence
    ax = axes[2, 0]
    ax.plot(df['step'], df['kl_policy_base'], 'purple', linewidth=2)
    ax.axhline(y=4.0, color='orange', linestyle='--', alpha=0.5, label='Target (4.0)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('KL Divergence')
    ax.set_title('Policy Drift from Base Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 8. Detector Comparison (Test vs Eval)
    ax = axes[2, 1]
    
    # Support both old and new formats
    test_det_col = 'test/detector_prob/mean' if 'test/detector_prob/mean' in df.columns else 'test/env/all/detector_prob'
    eval_det_col = 'eval/detector_prob/mean' if 'eval/detector_prob/mean' in df.columns else 'eval/env/all/detector_prob'
    
    if test_det_col in df.columns and eval_det_col in df.columns:
        test_steps = df[df[test_det_col].notna()]['step']
        test_prob = df[df[test_det_col].notna()][test_det_col]
        eval_steps = df[df[eval_det_col].notna()]['step']
        eval_prob = df[df[eval_det_col].notna()][eval_det_col]
        
        ax.plot(test_steps, test_prob, 'r-', linewidth=2, marker='o', markersize=6, label='Test P(AI)')
        ax.plot(eval_steps, eval_prob, 'g-', linewidth=2, marker='s', markersize=6, label='Eval P(AI)')
        ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, label='Random')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Detector Probability')
        ax.set_title('Generalization: Test vs Eval')
        ax.legend()
        ax.grid(True, alpha=0.3)
    elif test_det_col in df.columns:
        # Show only test if eval not available
        test_steps = df[df[test_det_col].notna()]['step']
        test_prob = df[df[test_det_col].notna()][test_det_col]
        ax.plot(test_steps, test_prob, 'r-', linewidth=2, marker='o', markersize=6, label='Test P(AI)')
        ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, label='Random')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Detector Probability')
        ax.set_title('Test Set Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No test/eval data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
    
    # 9. Learning Rate
    ax = axes[2, 2]
    ax.plot(df['step'], df['optim/lr'], 'orange', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'training_curves.pdf', bbox_inches='tight')
    print(f"✓ Saved training curves to {output_dir / 'training_curves.png'}")
    plt.close()

def plot_pareto_frontiers(df: pd.DataFrame, output_dir: Path):
    """Plot Pareto frontier analysis for multi-objective optimization."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Pareto Frontier Analysis: Multi-Objective Trade-offs', 
                 fontsize=18, fontweight='bold')
    
    # Filter out invalid points
    valid_df = df[df['env/all/parse_success'] > 0.8].copy()
    
    # 1. Detector Evasion vs Semantic Similarity
    ax = axes[0]
    
    # Convert detector prob to evasion score (lower prob = better evasion)
    valid_df['detector_evasion'] = 1 - valid_df['env/all/detector_prob']
    
    # For Pareto, we minimize: -evasion, -semantic_sim
    costs = np.column_stack([
        -valid_df['detector_evasion'].values,
        -valid_df['env/all/semantic_sim'].values
    ])
    
    pareto_mask = is_pareto_efficient(costs)
    
    # Plot all points
    scatter = ax.scatter(
        valid_df['detector_evasion'],
        valid_df['env/all/semantic_sim'],
        c=valid_df['step'],
        cmap='viridis',
        alpha=0.6,
        s=100,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Highlight Pareto frontier
    pareto_points = valid_df[pareto_mask]
    ax.scatter(
        pareto_points['detector_evasion'],
        pareto_points['env/all/semantic_sim'],
        c='red',
        s=200,
        marker='*',
        edgecolors='darkred',
        linewidth=2,
        label='Pareto Optimal',
        zorder=5
    )
    
    # Connect Pareto points
    pareto_sorted = pareto_points.sort_values('detector_evasion')
    ax.plot(
        pareto_sorted['detector_evasion'],
        pareto_sorted['env/all/semantic_sim'],
        'r--',
        linewidth=2,
        alpha=0.7,
        zorder=4
    )
    
    ax.set_xlabel('Detector Evasion Score\n(higher = better stealth)', fontsize=14)
    ax.set_ylabel('Semantic Similarity\n(higher = better quality)', fontsize=14)
    ax.set_title('Stealth vs Quality Trade-off', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Training Step', fontsize=12)
    
    # 2. 3-way trade-off: Detector Evasion vs Semantic vs Perplexity
    ax = axes[1]
    
    # Normalize perplexity to 0-1 range (target is 30, max is 80)
    valid_df['ppl_score'] = 1 - np.abs(valid_df['env/all/perplexity'] - 30) / 50
    valid_df['ppl_score'] = valid_df['ppl_score'].clip(0, 1)
    
    # For Pareto with 3 objectives: minimize -evasion, -semantic, -ppl_score
    costs_3d = np.column_stack([
        -valid_df['detector_evasion'].values,
        -valid_df['env/all/semantic_sim'].values,
        -valid_df['ppl_score'].values
    ])
    
    pareto_mask_3d = is_pareto_efficient(costs_3d)
    
    # Color by perplexity score
    scatter2 = ax.scatter(
        valid_df['detector_evasion'],
        valid_df['env/all/semantic_sim'],
        c=valid_df['ppl_score'],
        cmap='RdYlGn',
        alpha=0.6,
        s=100,
        edgecolors='black',
        linewidth=0.5,
        vmin=0,
        vmax=1
    )
    
    # Highlight 3D Pareto frontier
    pareto_3d_points = valid_df[pareto_mask_3d]
    ax.scatter(
        pareto_3d_points['detector_evasion'],
        pareto_3d_points['env/all/semantic_sim'],
        c='blue',
        s=200,
        marker='D',
        edgecolors='darkblue',
        linewidth=2,
        label='3D Pareto Optimal',
        zorder=5
    )
    
    ax.set_xlabel('Detector Evasion Score', fontsize=14)
    ax.set_ylabel('Semantic Similarity', fontsize=14)
    ax.set_title('3-Objective Trade-off\n(Stealth × Quality × Naturalness)', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    cbar2 = plt.colorbar(scatter2, ax=ax)
    cbar2.set_label('Naturalness Score\n(perplexity ~30)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_frontiers.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'pareto_frontiers.pdf', bbox_inches='tight')
    print(f"✓ Saved Pareto frontiers to {output_dir / 'pareto_frontiers.png'}")
    
    # Print Pareto optimal points
    print("\n" + "="*70)
    print("PARETO OPTIMAL CHECKPOINTS")
    print("="*70)
    print("\n2D Pareto (Stealth × Quality):")
    print(pareto_points[['step', 'detector_evasion', 'env/all/semantic_sim', 
                         'env/all/perplexity', 'env/all/reward/total']].to_string())
    
    print("\n3D Pareto (Stealth × Quality × Naturalness):")
    print(pareto_3d_points[['step', 'detector_evasion', 'env/all/semantic_sim', 
                            'ppl_score', 'env/all/perplexity']].to_string())
    
    plt.close()

def plot_reward_decomposition(df: pd.DataFrame, output_dir: Path):
    """Plot reward component contributions over time."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Reward Component Analysis', fontsize=20, fontweight='bold')
    
    # Find available reward components
    all_components = ['env/all/reward/detector', 'env/all/reward/semantic', 
                      'env/all/reward/perplexity', 'env/all/reward/fairness']
    components = [c for c in all_components if c in df.columns]
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(components)]
    
    # 1. Stacked area chart of reward components
    ax = axes[0, 0]
    if components:
        # Stack positive and negative separately
        positive_data = df[components].clip(lower=0)
        ax.stackplot(df['step'], positive_data.T, labels=[c.split('/')[-1] for c in components], 
                     colors=colors, alpha=0.7)
        ax.set_xlabel('Training Step', fontsize=14)
        ax.set_ylabel('Reward Contribution', fontsize=14)
        ax.set_title('Positive Reward Components', fontsize=16)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No reward components found', ha='center', va='center', fontsize=14)
        ax.axis('off')
    
    # 2. Individual component trajectories
    ax = axes[0, 1]
    if components:
        for comp, color in zip(components, colors):
            ax.plot(df['step'], df[comp], linewidth=2, label=comp.split('/')[-1], 
                    color=color)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Training Step', fontsize=14)
        ax.set_ylabel('Reward Value', fontsize=14)
        ax.set_title('Individual Reward Components', fontsize=16)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No reward components found', ha='center', va='center', fontsize=14)
        ax.axis('off')
    
    # 3. Detector probability histogram by training phase
    ax = axes[1, 0]
    early = df[df['step'] < 10]
    mid = df[(df['step'] >= 10) & (df['step'] < 30)]
    late = df[df['step'] >= 30]
    
    ax.hist([early['env/all/detector_prob'], mid['env/all/detector_prob'], 
             late['env/all/detector_prob']], 
            bins=20, label=['Early (0-10)', 'Mid (10-30)', 'Late (30+)'],
            alpha=0.6, color=['red', 'orange', 'green'])
    ax.axvline(x=0.5, color='k', linestyle='--', linewidth=2, label='Random Guess')
    ax.set_xlabel('Detector Probability', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('Detector Probability Distribution\n(lower = better evasion)', 
                 fontsize=16)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Correlation heatmap
    ax = axes[1, 1]
    # Find available metrics for correlation
    corr_candidates = ['env/all/reward/detector', 'env/all/reward/semantic',
                      'env/all/reward/perplexity', 'env/all/semantic_sim',
                      'env/all/detector_prob', 'env/all/perplexity']
    corr_cols = [c for c in corr_candidates if c in df.columns]
    
    if len(corr_cols) >= 2:
        corr_data = df[corr_cols].corr()
        
        im = ax.imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_data.columns)))
        ax.set_yticks(range(len(corr_data.columns)))
        ax.set_xticklabels([c.split('/')[-1] for c in corr_data.columns], 
                           rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels([c.split('/')[-1] for c in corr_data.columns], fontsize=10)
        ax.set_title('Metric Correlations', fontsize=16)
        
        # Add correlation values
        for i in range(len(corr_data.columns)):
            for j in range(len(corr_data.columns)):
                text = ax.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, 'Insufficient metrics for correlation', ha='center', va='center', fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_decomposition.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'reward_decomposition.pdf', bbox_inches='tight')
    print(f"✓ Saved reward decomposition to {output_dir / 'reward_decomposition.png'}")
    plt.close()

def plot_stability_metrics(df: pd.DataFrame, output_dir: Path):
    """Plot training stability and convergence metrics."""
    time_col = None
    for candidate in ("time/total", "time/training_loop/total", "time/train"):
        if candidate in df.columns:
            time_col = candidate
            break

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Stability & Convergence Analysis', 
                 fontsize=20, fontweight='bold')
    
    # 1. Entropy over time (exploration)
    ax = axes[0, 0]
    ax.plot(df['step'], df['optim/entropy'], 'b-', linewidth=2)
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('Policy Entropy', fontsize=14)
    ax.set_title('Exploration Level\n(higher = more diverse outputs)', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # 2. Learning rate schedule
    ax = axes[0, 1]
    ax.plot(df['step'], df['optim/lr'], 'g-', linewidth=2)
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('Learning Rate', fontsize=14)
    ax.set_title('Learning Rate Schedule', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 3. Token generation statistics
    ax = axes[1, 0]
    ax.plot(df['step'], df['env/all/ac_tokens_per_turn'], 'purple', 
            linewidth=2, label='Generated Tokens')
    ax.plot(df['step'], df['env/all/ob_tokens_per_turn'], 'orange', 
            linewidth=2, label='Input Tokens', alpha=0.7)
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('Token Count', fontsize=14)
    ax.set_title('Generation Length Statistics', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 4. Training time per step
    ax = axes[1, 1]
    if time_col:
        ax.plot(df['step'], df[time_col], 'red', linewidth=2)
        ax.set_xlabel('Training Step', fontsize=14)
        ax.set_ylabel('Time (seconds)', fontsize=14)
        ax.set_title('Iteration Time\n(includes generation + training)', fontsize=16)
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')
        ax.text(
            0.5,
            0.5,
            "No time metrics found",
            ha="center",
            va="center",
            fontsize=14,
        )
    
    plt.tight_layout()
    plt.savefig(output_dir / 'stability_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'stability_metrics.pdf', bbox_inches='tight')
    print(f"✓ Saved stability metrics to {output_dir / 'stability_metrics.png'}")
    plt.close()

def generate_summary_stats(df: pd.DataFrame, output_dir: Path):
    """Generate summary statistics table."""
    time_col = None
    for candidate in ("time/total", "time/training_loop/total", "time/train"):
        if candidate in df.columns:
            time_col = candidate
            break

    summary = {
        'Metric': [],
        'Train Initial': [],
        'Train Final': [],
        'Train Best': [],
        'Test Final': [],
        'Eval Final': []
    }
    
    metrics = {
        'Total Reward': ('env/all/reward/total', ['test/reward/mean', 'test/env/all/reward/total'], ['eval/reward/mean', 'eval/env/all/reward/total'], True),
        'Detector Evasion': ('env/all/detector_prob', ['test/detector_prob/mean', 'test/env/all/detector_prob'], ['eval/detector_prob/mean', 'eval/env/all/detector_prob'], False),
        'Semantic Similarity': ('env/all/semantic_sim', ['test/semantic_sim/mean', 'test/env/all/semantic_sim'], ['eval/semantic_sim/mean', 'eval/env/all/semantic_sim'], True),
        'Perplexity': ('env/all/perplexity', ['test/perplexity/mean', 'test/env/all/perplexity'], ['eval/perplexity/mean', 'eval/env/all/perplexity'], False),
        'KL Divergence': ('kl_policy_base', None, None, False),
        'Parse Success (%)': ('env/all/parse_success', ['test/valid_output_rate/mean', 'test/env/all/parse_success'], ['eval/valid_output_rate/mean', 'eval/env/all/parse_success'], True),
    }
    
    for name, (train_col, test_cols, eval_cols, maximize) in metrics.items():
        summary['Metric'].append(name)
        
        # Train metrics
        if train_col and train_col in df.columns:
            # Convert detector prob to evasion for display
            if 'detector_prob' in train_col:
                train_data = 1 - df[train_col]
            else:
                train_data = df[train_col]
            
            summary['Train Initial'].append(f"{train_data.iloc[0]:.4f}")
            summary['Train Final'].append(f"{train_data.iloc[-1]:.4f}")
            
            if maximize:
                summary['Train Best'].append(f"{train_data.max():.4f}")
            else:
                summary['Train Best'].append(f"{train_data.min():.4f}")
        else:
            summary['Train Initial'].append('N/A')
            summary['Train Final'].append('N/A')
            summary['Train Best'].append('N/A')
        
        # Test metrics - try multiple column names
        test_col = None
        if test_cols:
            for col in test_cols:
                if col in df.columns:
                    test_col = col
                    break
        
        if test_col:
            test_data = df[df[test_col].notna()][test_col]
            if len(test_data) > 0:
                # Convert detector prob to evasion
                if 'detector' in test_col and 'reward' not in test_col:
                    test_final = 1 - test_data.iloc[-1]
                else:
                    test_final = test_data.iloc[-1]
                summary['Test Final'].append(f"{test_final:.4f}")
            else:
                summary['Test Final'].append('N/A')
        else:
            summary['Test Final'].append('N/A')
        
        # Eval metrics - try multiple column names
        eval_col = None
        if eval_cols:
            for col in eval_cols:
                if col in df.columns:
                    eval_col = col
                    break
        
        if eval_col:
            eval_data = df[df[eval_col].notna()][eval_col]
            if len(eval_data) > 0:
                # Convert detector prob to evasion
                if 'detector' in eval_col and 'reward' not in eval_col:
                    eval_final = 1 - eval_data.iloc[-1]
                else:
                    eval_final = eval_data.iloc[-1]
                summary['Eval Final'].append(f"{eval_final:.4f}")
            else:
                summary['Eval Final'].append('N/A')
        else:
            summary['Eval Final'].append('N/A')
    
    summary_df = pd.DataFrame(summary)
    
    # Save as CSV
    summary_df.to_csv(output_dir / 'training_summary.csv', index=False)
    
    # Save as formatted text
    with open(output_dir / 'training_summary.txt', 'w') as f:
        f.write("="*90 + "\n")
        f.write("STEALTHRL TRAINING SUMMARY\n")
        f.write("="*90 + "\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")
        f.write("="*90 + "\n")
        f.write("GENERALIZATION ANALYSIS (Test vs Eval)\n")
        f.write("="*90 + "\n\n")
        
        # Add generalization metrics - support both old and new formats
        test_det_col = 'test/detector_prob/mean' if 'test/detector_prob/mean' in df.columns else 'test/env/all/detector_prob'
        eval_det_col = 'eval/detector_prob/mean' if 'eval/detector_prob/mean' in df.columns else 'eval/env/all/detector_prob'
        
        if test_det_col in df.columns and eval_det_col in df.columns:
            test_prob = df[df[test_det_col].notna()][test_det_col].iloc[-1]
            eval_prob = df[df[eval_det_col].notna()][eval_det_col].iloc[-1]
            f.write(f"Final Test Detector P(AI): {test_prob:.4f} (Evasion: {1-test_prob:.4f})\n")
            f.write(f"Final Eval Detector P(AI): {eval_prob:.4f} (Evasion: {1-eval_prob:.4f})\n")
            f.write(f"Generalization Gap: {abs(test_prob - eval_prob):.4f}\n\n")
        elif test_det_col in df.columns:
            test_prob = df[df[test_det_col].notna()][test_det_col].iloc[-1]
            f.write(f"Final Test Detector P(AI): {test_prob:.4f} (Evasion: {1-test_prob:.4f})\n")
            f.write("No eval set data available\n\n")
        else:
            f.write("No test/eval data available\n\n")
        
        f.write(f"Total Training Steps: {len(df)}\n")
        if time_col:
            f.write(f"Total Training Time: {df[time_col].sum() / 3600:.2f} hours\n")
        else:
            f.write("Total Training Time: n/a (time metric missing)\n")
        f.write(f"Final Learning Rate: {df['optim/lr'].iloc[-1]}\n")
    
    print(f"✓ Saved summary statistics to {output_dir / 'training_summary.csv'}")
    print("\nTraining Summary:")
    print(summary_df.to_string(index=False))

def main():
    """Main visualization pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate training visualizations")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to run directory containing metrics.jsonl",
    )
    args = parser.parse_args()

    metrics_path = Path(args.run_dir) / "metrics.jsonl"
    output_dir = Path(args.run_dir) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("STEALTHRL TRAINING VISUALIZATION")
    print("="*70)
    print(f"\nLoading metrics from: {metrics_path}")
    
    # Load data
    df = load_metrics(str(metrics_path))
    print(f"✓ Loaded {len(df)} training steps")
    
    # Generate all plots
    print("\nGenerating visualizations...")
    plot_training_curves(df, output_dir)
    plot_pareto_frontiers(df, output_dir)
    plot_reward_decomposition(df, output_dir)
    plot_stability_metrics(df, output_dir)
    generate_summary_stats(df, output_dir)
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - training_curves.png/pdf")
    print("  - pareto_frontiers.png/pdf")
    print("  - reward_decomposition.png/pdf")
    print("  - stability_metrics.png/pdf")
    print("  - training_summary.csv/txt")
    print("\nYou can now use these for your presentation!")

if __name__ == "__main__":
    main()
