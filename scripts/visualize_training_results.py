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
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('StealthRL Training Progression (GRPO)', fontsize=20, fontweight='bold')
    
    # 1. Total Reward
    ax = axes[0, 0]
    ax.plot(df['step'], df['env/all/reward/total'], 'b-', linewidth=2, label='Train')
    if 'test/env/all/reward/total' in df.columns:
        test_steps = df[df['test/env/all/reward/total'].notna()]['step']
        test_rewards = df[df['test/env/all/reward/total'].notna()]['test/env/all/reward/total']
        ax.plot(test_steps, test_rewards, 'r--', linewidth=2, marker='o', label='Test')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Total Reward')
    ax.set_title('Total Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Detector Evasion
    ax = axes[0, 1]
    ax.plot(df['step'], df['env/all/reward/detector'], 'g-', linewidth=2, label='Detector Reward')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Detector Reward (higher = better evasion)')
    ax.set_title('Detector Evasion Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Semantic Similarity
    ax = axes[0, 2]
    ax.plot(df['step'], df['env/all/semantic_sim'], 'purple', linewidth=2)
    ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Quality Threshold')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Semantic Similarity')
    ax.set_title('Semantic Quality Preservation')
    ax.set_ylim([0.93, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Perplexity
    ax = axes[1, 0]
    ax.plot(df['step'], df['env/all/perplexity'], 'orange', linewidth=2)
    ax.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Target (30)')
    ax.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='Max (80)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Perplexity')
    ax.set_title('Text Naturalness (Perplexity)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. KL Divergence
    ax = axes[1, 1]
    ax.plot(df['step'], df['kl_policy_base'], 'red', linewidth=2)
    ax.axhline(y=4.0, color='orange', linestyle='--', alpha=0.5, label='Target (4.0)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('KL Divergence')
    ax.set_title('Policy Drift from Base Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Parse Success Rate
    ax = axes[1, 2]
    ax.plot(df['step'], df['env/all/parse_success'] * 100, 'teal', linewidth=2)
    ax.axhline(y=90, color='g', linestyle='--', alpha=0.5, label='Target (90%)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Parse Success Rate (%)')
    ax.set_title('Valid Output Generation')
    ax.set_ylim([60, 102])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
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
    
    # 1. Stacked area chart of reward components
    ax = axes[0, 0]
    components = ['env/all/reward/detector', 'env/all/reward/semantic', 
                  'env/all/reward/perplexity', 'env/all/reward/fairness']
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Stack positive and negative separately
    positive_data = df[components].clip(lower=0)
    ax.stackplot(df['step'], positive_data.T, labels=components, 
                 colors=colors, alpha=0.7)
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('Reward Contribution', fontsize=14)
    ax.set_title('Positive Reward Components', fontsize=16)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Individual component trajectories
    ax = axes[0, 1]
    for comp, color in zip(components, colors):
        ax.plot(df['step'], df[comp], linewidth=2, label=comp.split('/')[-1], 
                color=color)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('Reward Value', fontsize=14)
    ax.set_title('Individual Reward Components', fontsize=16)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
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
    corr_data = df[['env/all/reward/detector', 'env/all/reward/semantic',
                    'env/all/reward/perplexity', 'env/all/semantic_sim',
                    'env/all/detector_prob', 'env/all/perplexity']].corr()
    
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
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_decomposition.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'reward_decomposition.pdf', bbox_inches='tight')
    print(f"✓ Saved reward decomposition to {output_dir / 'reward_decomposition.png'}")
    plt.close()

def plot_stability_metrics(df: pd.DataFrame, output_dir: Path):
    """Plot training stability and convergence metrics."""
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
    ax.plot(df['step'], df['time/total'], 'red', linewidth=2)
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('Time (seconds)', fontsize=14)
    ax.set_title('Iteration Time\n(includes generation + training)', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'stability_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'stability_metrics.pdf', bbox_inches='tight')
    print(f"✓ Saved stability metrics to {output_dir / 'stability_metrics.png'}")
    plt.close()

def generate_summary_stats(df: pd.DataFrame, output_dir: Path):
    """Generate summary statistics table."""
    summary = {
        'Metric': [],
        'Initial': [],
        'Final': [],
        'Best': [],
        'Mean': []
    }
    
    metrics = {
        'Total Reward': 'env/all/reward/total',
        'Detector Evasion': 'env/all/reward/detector',
        'Semantic Similarity': 'env/all/semantic_sim',
        'Perplexity': 'env/all/perplexity',
        'KL Divergence': 'kl_policy_base',
        'Parse Success (%)': 'env/all/parse_success',
        'Detector Prob': 'env/all/detector_prob',
    }
    
    for name, col in metrics.items():
        summary['Metric'].append(name)
        summary['Initial'].append(f"{df[col].iloc[0]:.4f}")
        summary['Final'].append(f"{df[col].iloc[-1]:.4f}")
        
        if 'reward' in col or 'similarity' in col or 'success' in col:
            summary['Best'].append(f"{df[col].max():.4f}")
        elif 'prob' in col or 'perplexity' in col or 'kl' in col:
            summary['Best'].append(f"{df[col].min():.4f}")
        else:
            summary['Best'].append(f"{df[col].max():.4f}")
            
        summary['Mean'].append(f"{df[col].mean():.4f}")
    
    summary_df = pd.DataFrame(summary)
    
    # Save as CSV
    summary_df.to_csv(output_dir / 'training_summary.csv', index=False)
    
    # Save as formatted text
    with open(output_dir / 'training_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("STEALTHRL TRAINING SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")
        f.write(f"Total Training Steps: {len(df)}\n")
        f.write(f"Total Training Time: {df['time/total'].sum() / 3600:.2f} hours\n")
        f.write(f"Final Learning Rate: {df['optim/lr'].iloc[-1]}\n")
    
    print(f"✓ Saved summary statistics to {output_dir / 'training_summary.csv'}")
    print("\nTraining Summary:")
    print(summary_df.to_string(index=False))

def main():
    """Main visualization pipeline."""
    # Path to metrics file
    metrics_path = Path(__file__).parent.parent / "outputs" / "tinker_ultrafast" / "run_20251207_212110" / "metrics.jsonl"
    output_dir = metrics_path.parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
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
