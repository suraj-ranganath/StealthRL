"""
Plotting and table generation for StealthRL evaluation.

Generates paper-ready figures and tables as specified in SPEC.md:
- Figure 1: Transfer heatmap (detector × method)
- Figure 2: Tradeoff curve (similarity vs TPR@1%FPR)
- Figure 3: Budget sweep (N candidates vs metrics)
- Table A: Main results
- Table B: Transfer matrix
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# 15 colorblind-friendly colors (from Adversarial Paraphrasing paper)
COLORBLIND_COLORS = [
    "#0072B2",  # Blue
    "#009E73",  # Green
    "#D55E00",  # Orange
    "#CC79A7",  # Pink
    "#F0E442",  # Yellow
    "#56B4E9",  # Light blue
    "#E69F00",  # Dark orange
    "#000000",  # Black
]

# Method-specific color mapping for consistent paper figures
COLORS = {
    "no_attack": "#0072B2",          # Blue - baseline
    "simple_paraphrase": "#E69F00",  # Orange - M1
    "stealthrl": "#009E73",          # Green - M2 (ours)
    "tinker": "#009E73",             # Green - M2 alias
    "adversarial_paraphrasing": "#D55E00",  # Red-orange - M3
    "authormist": "#CC79A7",         # Pink - M4
    "homoglyph": "#56B4E9",          # Light blue - M5
    "silverspeak": "#56B4E9",        # Light blue - M5 alias
    "human": "#7f7f7f",              # Gray - human reference
    # Short aliases
    "m0": "#0072B2",
    "m1": "#E69F00",
    "m2": "#009E73",
    "m3": "#D55E00",
    "m4": "#CC79A7",
    "m5": "#56B4E9",
}

# Method display names for paper figures
METHOD_NAMES = {
    "no_attack": "No Attack (M0)",
    "simple_paraphrase": "Simple Para. (M1)",
    "stealthrl": "StealthRL (M2)",
    "tinker": "Tinker (M2)",
    "adversarial_paraphrasing": "Adv. Para. (M3)",
    "authormist": "AuthorMist (M4)",
    "homoglyph": "Homoglyph (M5)",
    "silverspeak": "SilverSpeak (M5)",
    # Short aliases
    "m0": "No Attack (M0)",
    "m1": "Simple Para. (M1)",
    "m2": "StealthRL (M2)",
    "m3": "Adv. Para. (M3)",
    "m4": "AuthorMist (M4)",
    "m5": "Homoglyph (M5)",
}


def create_heatmap(
    data: pd.DataFrame,
    value_col: str,
    row_col: str = "detector",
    col_col: str = "method",
    title: str = "TPR@1%FPR Heatmap",
    output_path: str = None,
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = "RdYlGn_r",
    vmin: float = 0.0,
    vmax: float = 1.0,
    annot_fmt: str = ".2f",
) -> None:
    """
    Create detector × method heatmap (Figure 1 in SPEC.md).
    
    Args:
        data: DataFrame with metrics
        value_col: Column to use for heatmap values
        row_col: Column for rows (default: detector)
        col_col: Column for columns (default: method)
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
        cmap: Colormap (RdYlGn_r = red=high, green=low, good for TPR)
        vmin, vmax: Value range
        annot_fmt: Annotation format
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Pivot data for heatmap
    pivot = data.pivot(index=row_col, columns=col_col, values=value_col)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        pivot,
        annot=True,
        fmt=annot_fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_kws={"label": value_col},
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Detector", fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved heatmap to {output_path}")
    
    plt.close()


def create_tradeoff_plot(
    data: pd.DataFrame,
    x_col: str = "sim_e5",
    y_col: str = "mean_tpr",
    label_col: str = "method",
    title: str = "Evasion-Quality Tradeoff",
    output_path: str = None,
    figsize: Tuple[int, int] = (8, 6),
    annotate: bool = True,
) -> None:
    """
    Create tradeoff/Pareto curve (Figure 2 in SPEC.md).
    
    X-axis: Quality metric (similarity)
    Y-axis: Evasion metric (TPR@1%FPR, lower = better attack)
    
    Args:
        data: DataFrame with method-level aggregated metrics
        x_col: Column for x-axis (quality)
        y_col: Column for y-axis (detection rate)
        label_col: Column for labels
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
        annotate: Whether to add method labels
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for _, row in data.iterrows():
        method = row[label_col]
        color = COLORS.get(method, "#333333")
        
        ax.scatter(
            row[x_col],
            row[y_col],
            s=150,
            c=color,
            label=method,
            edgecolors='white',
            linewidth=2,
        )
        
        if annotate:
            ax.annotate(
                method,
                (row[x_col], row[y_col]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
            )
    
    ax.set_xlabel("Semantic Similarity (E5)", fontsize=12)
    ax.set_ylabel("Mean TPR@1%FPR (↓ better attack)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.axvline(x=0.9, color='gray', linestyle=':', alpha=0.5, label='Min similarity')
    
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim(0.7, 1.0)
    ax.set_ylim(0.0, 1.0)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved tradeoff plot to {output_path}")
    
    plt.close()


def create_budget_sweep_plot(
    data: pd.DataFrame,
    x_col: str = "n_candidates",
    y_col: str = "mean_tpr",
    y2_col: str = "sim_e5",
    group_col: str = "method",
    title: str = "Candidate Budget Sweep",
    output_path: str = None,
    figsize: Tuple[int, int] = (10, 5),
) -> None:
    """
    Create budget sweep plot (Figure 3 in SPEC.md).
    
    Shows how metrics change with number of candidates.
    
    Args:
        data: DataFrame with (method, n_candidates, metrics)
        x_col: Column for x-axis (candidate count)
        y_col: Column for primary y-axis (TPR)
        y2_col: Column for secondary y-axis (similarity)
        group_col: Column for grouping
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    methods = data[group_col].unique()
    
    # Plot 1: TPR vs N
    for method in methods:
        method_data = data[data[group_col] == method]
        color = COLORS.get(method, "#333333")
        
        ax1.plot(
            method_data[x_col],
            method_data[y_col],
            marker='o',
            label=method,
            color=color,
            linewidth=2,
            markersize=8,
        )
    
    ax1.set_xlabel("Number of Candidates (N)", fontsize=12)
    ax1.set_ylabel("Mean TPR@1%FPR", fontsize=12)
    ax1.set_title("Detection Rate vs Budget", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Similarity vs N
    for method in methods:
        method_data = data[data[group_col] == method]
        color = COLORS.get(method, "#333333")
        
        ax2.plot(
            method_data[x_col],
            method_data[y2_col],
            marker='s',
            label=method,
            color=color,
            linewidth=2,
            markersize=8,
        )
    
    ax2.set_xlabel("Number of Candidates (N)", fontsize=12)
    ax2.set_ylabel("Median Similarity (E5)", fontsize=12)
    ax2.set_title("Quality vs Budget", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.7, 1.0)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved budget sweep plot to {output_path}")
    
    plt.close()


def create_sanitize_plot(
    data: pd.DataFrame,
    output_path: str = None,
    figsize: Tuple[int, int] = (10, 5),
) -> None:
    """
    Create sanitization effect plot (optional figure in SPEC.md).
    
    Shows homoglyph attack effectiveness before/after sanitization.
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Expected columns: method, detector, tpr_before, tpr_after
    methods = data['method'].unique()
    x = np.arange(len(methods))
    width = 0.35
    
    ax.bar(x - width/2, data.groupby('method')['tpr_before'].mean(), width, 
           label='Before Sanitize', color='#d62728', alpha=0.8)
    ax.bar(x + width/2, data.groupby('method')['tpr_after'].mean(), width,
           label='After Sanitize', color='#2ca02c', alpha=0.8)
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Mean TPR@1%FPR', fontsize=12)
    ax.set_title('Effect of Sanitization Defense', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved sanitize plot to {output_path}")
    
    plt.close()


# ============================================================================
# AUROC Visualization (AuthorMist-style)
# ============================================================================

def create_auroc_bar_chart(
    data: pd.DataFrame,
    method_col: str = "method",
    detector_col: str = "detector",
    auroc_col: str = "auroc",
    title: str = "AUROC Across Detectors",
    output_path: str = None,
    figsize: Tuple[int, int] = (12, 6),
    show_baseline: bool = True,
) -> None:
    """
    Create AuthorMist-style AUROC bar chart comparison.
    
    Grouped bars showing AUROC for each method across detectors.
    Key insight: AUROC ≈ 0.5 means random classification (successful attack).
    
    Args:
        data: DataFrame with method, detector, auroc columns
        method_col: Column name for methods
        detector_col: Column name for detectors
        auroc_col: Column name for AUROC values
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
        show_baseline: Whether to show 0.5 baseline (random)
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    methods = data[method_col].unique()
    detectors = data[detector_col].unique()
    n_detectors = len(detectors)
    n_methods = len(methods)
    
    # Bar positioning
    x = np.arange(n_detectors)
    width = 0.8 / n_methods
    
    for i, method in enumerate(methods):
        method_data = data[data[method_col] == method]
        
        # Get AUROC values in detector order
        aurocs = []
        for det in detectors:
            det_row = method_data[method_data[detector_col] == det]
            if len(det_row) > 0:
                aurocs.append(det_row[auroc_col].values[0])
            else:
                aurocs.append(np.nan)
        
        color = COLORS.get(method, f"C{i}")
        offset = (i - n_methods/2 + 0.5) * width
        
        bars = ax.bar(
            x + offset, 
            aurocs, 
            width, 
            label=method.replace('_', ' ').title(),
            color=color,
            edgecolor='white',
            linewidth=0.5,
        )
        
        # Add value labels on bars
        for bar, val in zip(bars, aurocs):
            if not np.isnan(val):
                height = bar.get_height()
                ax.annotate(
                    f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8,
                    rotation=90 if n_methods > 4 else 0,
                )
    
    # Random baseline
    if show_baseline:
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, 
                   label='Random (AUROC=0.5)', alpha=0.7)
    
    ax.set_xlabel('Detector', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('_', '\n') for d in detectors], fontsize=10)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotation explaining AUROC interpretation
    ax.text(
        0.02, 0.02,
        "↓ Lower AUROC = Better attack (0.5 = random)",
        transform=ax.transAxes,
        fontsize=9,
        style='italic',
        color='gray',
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved AUROC bar chart to {output_path}")
    
    plt.close()


def create_auroc_radar_chart(
    data: pd.DataFrame,
    method_col: str = "method",
    detector_col: str = "detector",
    auroc_col: str = "auroc",
    title: str = "AUROC Radar: Attack Effectiveness",
    output_path: str = None,
    figsize: Tuple[int, int] = (8, 8),
) -> None:
    """
    Create radar chart showing AUROC across detectors for each method.
    
    Useful for visualizing attack coverage across multiple detectors.
    Smaller area = better attack (closer to 0.5 center).
    """
    import matplotlib.pyplot as plt
    from math import pi
    
    methods = data[method_col].unique()
    detectors = list(data[detector_col].unique())
    n_detectors = len(detectors)
    
    # Compute angles for radar chart
    angles = [n / float(n_detectors) * 2 * pi for n in range(n_detectors)]
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    for method in methods:
        method_data = data[data[method_col] == method]
        
        values = []
        for det in detectors:
            det_row = method_data[method_data[detector_col] == det]
            if len(det_row) > 0:
                values.append(det_row[auroc_col].values[0])
            else:
                values.append(0.5)
        values += values[:1]  # Complete the loop
        
        color = COLORS.get(method, None)
        ax.plot(angles, values, 'o-', linewidth=2, label=method.replace('_', ' ').title(),
                color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Add detector labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.replace('_', '\n') for d in detectors], size=9)
    
    # Add baseline circle at 0.5
    baseline_values = [0.5] * (n_detectors + 1)
    ax.plot(angles, baseline_values, 'r--', linewidth=1.5, alpha=0.5, label='Random (0.5)')
    
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved AUROC radar chart to {output_path}")
    
    plt.close()


def create_method_comparison_summary(
    data: pd.DataFrame,
    method_col: str = "method",
    detector_col: str = "detector",
    output_path: str = None,
    figsize: Tuple[int, int] = (14, 8),
) -> None:
    """
    Create comprehensive method comparison figure (AuthorMist-style).
    
    Multi-panel figure showing:
    1. AUROC per detector (grouped bars)
    2. Mean AUROC with confidence intervals
    3. TPR@1%FPR summary
    4. Attack Success Rate
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    methods = data[method_col].unique()
    detectors = data[detector_col].unique()
    
    # Panel 1: AUROC grouped bars
    ax1 = axes[0, 0]
    x = np.arange(len(detectors))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        method_data = data[data[method_col] == method]
        aurocs = [method_data[method_data[detector_col] == d]['auroc'].values[0] 
                  if len(method_data[method_data[detector_col] == d]) > 0 else np.nan 
                  for d in detectors]
        color = COLORS.get(method, f"C{i}")
        offset = (i - len(methods)/2 + 0.5) * width
        ax1.bar(x + offset, aurocs, width, label=method.replace('_', ' ').title(), color=color)
    
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    ax1.set_ylabel('AUROC')
    ax1.set_title('(a) AUROC by Detector', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([d[:10] for d in detectors], rotation=45, ha='right', fontsize=8)
    ax1.legend(fontsize=8, ncol=2)
    ax1.set_ylim(0, 1.05)
    
    # Panel 2: Mean AUROC with error bars
    ax2 = axes[0, 1]
    mean_aurocs = data.groupby(method_col)['auroc'].agg(['mean', 'std']).reset_index()
    x2 = np.arange(len(mean_aurocs))
    colors = [COLORS.get(m, f"C{i}") for i, m in enumerate(mean_aurocs[method_col])]
    
    bars = ax2.bar(x2, mean_aurocs['mean'], yerr=mean_aurocs['std'], capsize=5,
                   color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random')
    ax2.set_ylabel('Mean AUROC ± std')
    ax2.set_title('(b) Mean AUROC Across Detectors', fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels([m.replace('_', '\n')[:12] for m in mean_aurocs[method_col]], 
                        fontsize=9)
    ax2.set_ylim(0, 1.05)
    
    # Add value labels
    for bar, val in zip(bars, mean_aurocs['mean']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
    
    # Panel 3: TPR@1%FPR grouped bars
    ax3 = axes[1, 0]
    
    for i, method in enumerate(methods):
        method_data = data[data[method_col] == method]
        tprs = [method_data[method_data[detector_col] == d]['tpr_at_1fpr'].values[0] 
                if len(method_data[method_data[detector_col] == d]) > 0 else np.nan 
                for d in detectors]
        color = COLORS.get(method, f"C{i}")
        offset = (i - len(methods)/2 + 0.5) * width
        ax3.bar(x + offset, tprs, width, label=method.replace('_', ' ').title(), color=color)
    
    ax3.set_ylabel('TPR@1%FPR')
    ax3.set_title('(c) True Positive Rate at 1% FPR', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([d[:10] for d in detectors], rotation=45, ha='right', fontsize=8)
    ax3.set_ylim(0, 1.05)
    
    # Panel 4: Attack Success Rate summary
    ax4 = axes[1, 1]
    
    if 'asr' in data.columns:
        mean_asr = data.groupby(method_col)['asr'].mean().reset_index()
        x4 = np.arange(len(mean_asr))
        colors = [COLORS.get(m, f"C{i}") for i, m in enumerate(mean_asr[method_col])]
        
        bars = ax4.bar(x4, mean_asr['asr'], color=colors, edgecolor='black', linewidth=0.5)
        ax4.set_ylabel('Attack Success Rate (ASR)')
        ax4.set_title('(d) Mean Attack Success Rate', fontweight='bold')
        ax4.set_xticks(x4)
        ax4.set_xticklabels([m.replace('_', '\n')[:12] for m in mean_asr[method_col]], 
                            fontsize=9)
        ax4.set_ylim(0, 1.05)
        
        # Add value labels
        for bar, val in zip(bars, mean_asr['asr']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.1%}', ha='center', fontsize=9, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'ASR data not available', ha='center', va='center',
                 transform=ax4.transAxes, fontsize=12)
        ax4.set_title('(d) Attack Success Rate', fontweight='bold')
    
    fig.suptitle('Detection Evasion Results Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved method comparison summary to {output_path}")
    
    plt.close()


def create_score_distribution_plot(
    scores_data: pd.DataFrame,
    detector_col: str = "detector_name",
    method_col: str = "method",
    score_col: str = "detector_score",
    title: str = "Detection Score Distributions",
    output_path: str = None,
    figsize: Tuple[int, int] = (14, 5),
    include_human: bool = True,
) -> None:
    """
    Create violin/box plots showing score distributions per method (Adversarial Paraphrasing-inspired).
    
    This visualization shows the full distribution of detector scores, helping to understand
    how well each method shifts scores toward human-like values.
    
    Lower scores = more human-like (for most detectors).
    
    Args:
        scores_data: DataFrame with detector, method, score columns
        detector_col: Column name for detectors
        method_col: Column name for methods
        score_col: Column name for detection scores
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
        include_human: Whether human baseline is included in data
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    detectors = scores_data[detector_col].unique()
    n_detectors = len(detectors)
    
    fig, axes = plt.subplots(1, n_detectors, figsize=figsize)
    if n_detectors == 1:
        axes = [axes]
    
    for ax, detector in zip(axes, detectors):
        det_data = scores_data[scores_data[detector_col] == detector]
        
        # Create violin plot with method-specific colors
        methods = det_data[method_col].unique()
        palette = {m: COLORS.get(m, COLORBLIND_COLORS[i % len(COLORBLIND_COLORS)]) 
                   for i, m in enumerate(methods)}
        
        sns.violinplot(
            data=det_data,
            x=method_col,
            y=score_col,
            palette=palette,
            ax=ax,
            inner='box',
            cut=0,
        )
        
        # Add median markers
        medians = det_data.groupby(method_col)[score_col].median()
        for i, method in enumerate(methods):
            ax.scatter(i, medians[method], color='white', s=30, zorder=10, edgecolor='black')
        
        ax.set_xlabel('')
        ax.set_ylabel('Detection Score' if ax == axes[0] else '')
        ax.set_title(detector.replace('_', ' ').title(), fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add threshold line at 0.5 (typical decision boundary)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Decision threshold')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.text(0.5, 0.02, '↓ Lower score = More human-like', ha='center', fontsize=10, style='italic')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved score distribution plot to {output_path}")
    
    plt.close()


def create_score_shift_plot(
    before_scores: Dict[str, np.ndarray],
    after_scores: Dict[str, Dict[str, np.ndarray]],
    detector_name: str = "detector",
    title: str = "Score Shift: Before vs After Paraphrasing",
    output_path: str = None,
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """
    Create before/after score comparison (Adversarial Paraphrasing-inspired).
    
    Shows how detection scores change after applying different attack methods.
    
    Args:
        before_scores: Dict {detector: np.array of original text scores}
        after_scores: Dict {detector: {method: np.array of paraphrased scores}}
        detector_name: Name of detector for title
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    methods = list(after_scores.keys()) if isinstance(after_scores, dict) else []
    if not methods:
        logger.warning("No methods found in after_scores")
        return
    
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    if n_methods == 1:
        axes = [axes]
    
    for ax, method in zip(axes, methods):
        original = before_scores
        paraphrased = after_scores[method]
        
        # Paired comparison: plot original vs paraphrased
        ax.scatter(original, paraphrased, alpha=0.3, s=10, 
                   color=COLORS.get(method, COLORBLIND_COLORS[0]))
        
        # Diagonal line (no change)
        lims = [
            min(original.min(), paraphrased.min()),
            max(original.max(), paraphrased.max()),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='No change')
        
        # Decision boundaries
        ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5)
        ax.axvline(x=0.5, color='red', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Original Score')
        ax.set_ylabel('After Paraphrasing' if ax == axes[0] else '')
        ax.set_title(METHOD_NAMES.get(method, method), fontweight='bold')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add summary stats
        reduction = (original - paraphrased).mean()
        ax.text(0.05, 0.95, f'Mean reduction: {reduction:.3f}', 
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle(f"{title} ({detector_name})", fontsize=14, fontweight='bold')
    fig.text(0.5, 0.02, '↓ Points below diagonal = Successful evasion', ha='center', fontsize=10, style='italic')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved score shift plot to {output_path}")
    
    plt.close()


def create_human_ai_separation_plot(
    human_scores: np.ndarray,
    ai_scores: Dict[str, np.ndarray],
    detector_name: str = "Detector",
    title: str = "Human vs AI Score Separation",
    output_path: str = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Create overlapping histograms showing human vs AI score distributions.
    
    This helps visualize how well different attack methods overlap with human distribution.
    
    Args:
        human_scores: Array of scores for human-written text
        ai_scores: Dict {method: scores} for AI-generated (possibly paraphrased) text
        detector_name: Name of detector
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Human distribution
    ax.hist(human_scores, bins=50, alpha=0.5, label='Human', color='#7f7f7f', 
            density=True, edgecolor='black', linewidth=0.5)
    
    # AI distributions for each method
    for i, (method, scores) in enumerate(ai_scores.items()):
        color = COLORS.get(method, COLORBLIND_COLORS[i % len(COLORBLIND_COLORS)])
        ax.hist(scores, bins=50, alpha=0.4, label=METHOD_NAMES.get(method, method),
                color=color, density=True, histtype='step', linewidth=2)
    
    ax.set_xlabel('Detection Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f"{title}\n({detector_name})", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add decision threshold
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved human-AI separation plot to {output_path}")
    
    plt.close()


def create_roc_curves(
    predictions: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    title: str = "ROC Curves by Detector",
    output_path: str = None,
    figsize: Tuple[int, int] = (12, 4),
) -> None:
    """
    Create ROC curves for each detector, comparing methods.
    
    Args:
        predictions: Nested dict {detector: {method: (y_true, y_scores)}}
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    detectors = list(predictions.keys())
    n_detectors = len(detectors)
    
    fig, axes = plt.subplots(1, n_detectors, figsize=figsize)
    if n_detectors == 1:
        axes = [axes]
    
    for ax, detector in zip(axes, detectors):
        detector_preds = predictions[detector]
        
        for method, (y_true, y_scores) in detector_preds.items():
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            color = COLORS.get(method, None)
            label = METHOD_NAMES.get(method, method.replace("_", " ").title())
            ax.plot(fpr, tpr, lw=2, color=color,
                    label=f'{label} (AUC={roc_auc:.2f})')
        
        # Random baseline
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(detector.replace('_', ' ').title())
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curves to {output_path}")
    
    plt.close()


# ============================================================================
# Adversarial Paraphrasing Paper-Style Plots (NeurIPS '25)
# ============================================================================

def create_roc_curves_logscale(
    predictions: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    title: str = "ROC Curves (Log-Scale FPR)",
    output_path: str = None,
    figsize: Tuple[int, int] = (14, 4),
    fpr_min: float = 1e-3,
) -> None:
    """
    Create ROC curves with log-scale FPR axis (Adversarial Paraphrasing Figure 3 style).
    
    Log-scale FPR highlights performance in the critical low-FPR regime,
    which is most relevant for practical deployment where false positives are costly.
    
    Args:
        predictions: Nested dict {detector: {method: (y_true, y_scores)}}
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
        fpr_min: Minimum FPR for x-axis (log scale)
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    detectors = list(predictions.keys())
    n_detectors = len(detectors)
    
    fig, axes = plt.subplots(1, n_detectors, figsize=figsize)
    if n_detectors == 1:
        axes = [axes]
    
    for ax, detector in zip(axes, detectors):
        detector_preds = predictions[detector]
        
        for method, (y_true, y_scores) in detector_preds.items():
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            # Filter to valid FPR range for log scale
            valid_idx = fpr >= fpr_min
            fpr_valid = fpr[valid_idx]
            tpr_valid = tpr[valid_idx]
            
            color = COLORS.get(method, None)
            ax.plot(fpr_valid, tpr_valid, lw=2, color=color,
                    label=f'{method.replace("_", " ").title()} (AUC={roc_auc:.2f})')
        
        # Random baseline (diagonal in log scale)
        fpr_line = np.logspace(np.log10(fpr_min), 0, 100)
        ax.plot(fpr_line, fpr_line, 'k--', lw=1, alpha=0.5, label='Random')
        
        ax.set_xscale('log')
        ax.set_xlim([fpr_min, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (log scale)')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(detector.replace('_', ' ').title(), fontweight='bold')
        ax.legend(loc='lower right', fontsize=7)
        ax.grid(True, alpha=0.3, which='both')
        
        # Add vertical line at 1% FPR
        ax.axvline(x=0.01, color='gray', linestyle=':', alpha=0.7)
        ax.text(0.012, 0.05, '1% FPR', fontsize=7, color='gray')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved log-scale ROC curves to {output_path}")
    
    plt.close()


def create_transferability_heatmap(
    data: pd.DataFrame,
    baseline_col: str = "no_attack",
    value_col: str = "tpr_at_1fpr",
    method_col: str = "method",
    detector_col: str = "detector",
    title: str = "Relative TPR@1%FPR Reduction (%)",
    output_path: str = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Create transferability heatmap showing relative metric drop (Adversarial Paraphrasing Figure 4 style).
    
    Shows percentage reduction in TPR@1%FPR relative to no-attack baseline.
    Higher reduction = more effective attack.
    
    Args:
        data: DataFrame with method, detector, and metric columns
        baseline_col: Method name to use as baseline (default: "no_attack")
        value_col: Metric column to compute reduction for
        method_col: Column name for methods
        detector_col: Column name for detectors
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Pivot to get methods as rows, detectors as columns
    pivot = data.pivot(index=method_col, columns=detector_col, values=value_col)
    
    # Get baseline row
    if baseline_col not in pivot.index:
        logger.warning(f"Baseline '{baseline_col}' not found. Using first method as baseline.")
        baseline_col = pivot.index[0]
    
    baseline = pivot.loc[baseline_col]
    
    # Compute relative reduction: (baseline - value) / baseline * 100
    reduction = pivot.apply(lambda row: (baseline - row) / baseline * 100, axis=1)
    
    # Remove baseline row from display (it would be all zeros)
    reduction = reduction.drop(baseline_col, errors='ignore')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use diverging colormap: green=high reduction (good attack), red=negative (worse than baseline)
    sns.heatmap(
        reduction,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        ax=ax,
        cbar_kws={"label": "% Reduction in TPR@1%FPR"},
        linewidths=0.5,
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Deployed Detector", fontsize=12)
    ax.set_ylabel("Attack Method", fontsize=12)
    
    # Add annotation
    ax.text(
        0.5, -0.12,
        "↑ Higher % = More effective attack (greater reduction from baseline)",
        transform=ax.transAxes,
        ha='center',
        fontsize=9,
        style='italic',
        color='gray',
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved transferability heatmap to {output_path}")
    
    plt.close()


def create_quality_likert_chart(
    quality_data: pd.DataFrame,
    method_col: str = "method",
    rating_col: str = "quality_rating",
    similarity_col: str = "similarity_rating",
    title: str = "Text Quality Evaluation (GPT-4o Ratings)",
    output_path: str = None,
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """
    Create Likert-scale bar charts for quality evaluation (Adversarial Paraphrasing Figure 5 style).
    
    Shows mean ratings (1-5 scale) with error bars for quality and semantic similarity.
    
    Args:
        quality_data: DataFrame with method and rating columns
        method_col: Column name for methods
        rating_col: Column name for quality ratings
        similarity_col: Column name for similarity ratings
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    methods = quality_data[method_col].unique()
    x = np.arange(len(methods))
    
    # Quality ratings
    quality_means = quality_data.groupby(method_col)[rating_col].mean()
    quality_stds = quality_data.groupby(method_col)[rating_col].std()
    
    colors = [COLORS.get(m, f"C{i}") for i, m in enumerate(methods)]
    
    bars1 = ax1.bar(x, [quality_means.get(m, 0) for m in methods],
                    yerr=[quality_stds.get(m, 0) for m in methods],
                    capsize=5, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Quality Rating (1-5)', fontsize=11)
    ax1.set_title('(a) Overall Quality', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    ax1.set_ylim(0, 5.5)
    ax1.axhline(y=4, color='gray', linestyle='--', alpha=0.5, label='Good threshold')
    
    # Add value labels
    for bar, mean in zip(bars1, [quality_means.get(m, 0) for m in methods]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 f'{mean:.2f}', ha='center', fontsize=9)
    
    # Similarity ratings (if available)
    if similarity_col in quality_data.columns:
        sim_means = quality_data.groupby(method_col)[similarity_col].mean()
        sim_stds = quality_data.groupby(method_col)[similarity_col].std()
        
        bars2 = ax2.bar(x, [sim_means.get(m, 0) for m in methods],
                        yerr=[sim_stds.get(m, 0) for m in methods],
                        capsize=5, color=colors, edgecolor='black', linewidth=0.5)
        ax2.set_ylabel('Similarity Rating (1-5)', fontsize=11)
        ax2.set_title('(b) Semantic Similarity', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
        ax2.set_ylim(0, 5.5)
        ax2.axhline(y=4, color='gray', linestyle='--', alpha=0.5)
        
        for bar, mean in zip(bars2, [sim_means.get(m, 0) for m in methods]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                     f'{mean:.2f}', ha='center', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'Similarity data\nnot available', 
                 ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('(b) Semantic Similarity', fontweight='bold')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved quality Likert chart to {output_path}")
    
    plt.close()


def create_winrate_chart(
    winrate_data: pd.DataFrame,
    title: str = "Head-to-Head Win Rates",
    output_path: str = None,
    figsize: Tuple[int, int] = (10, 5),
) -> None:
    """
    Create win-rate comparison chart (Adversarial Paraphrasing Figure 5 bottom style).
    
    Shows head-to-head comparison win/tie/lose rates between methods.
    
    Args:
        winrate_data: DataFrame with columns [comparison, win, tie, lose]
                      e.g., "M2 vs M1": win=0.4, tie=0.35, lose=0.25
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    comparisons = winrate_data['comparison'].values
    wins = winrate_data['win'].values
    ties = winrate_data['tie'].values
    loses = winrate_data['lose'].values
    
    x = np.arange(len(comparisons))
    width = 0.25
    
    ax.bar(x - width, wins, width, label='Win', color='#2ca02c', edgecolor='black', linewidth=0.5)
    ax.bar(x, ties, width, label='Tie', color='#7f7f7f', edgecolor='black', linewidth=0.5)
    ax.bar(x + width, loses, width, label='Lose', color='#d62728', edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Proportion', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparisons, fontsize=10)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved win-rate chart to {output_path}")
    
    plt.close()


def create_perplexity_comparison_table(
    data: pd.DataFrame,
    method_col: str = "method",
    ppl_col: str = "perplexity",
    output_path: str = None,
    format: str = "markdown",
) -> str:
    """
    Create perplexity comparison table (Adversarial Paraphrasing Table 3 style).
    
    Args:
        data: DataFrame with method and perplexity columns
        method_col: Column for method names
        ppl_col: Column for perplexity values
        output_path: Path to save table
        format: Output format (markdown, latex)
    
    Returns:
        Table string
    """
    # Aggregate by method
    summary = data.groupby(method_col)[ppl_col].agg(['mean', 'std']).round(2)
    summary.columns = ['PPL (mean)', 'PPL (std)']
    summary['PPL (mean±std)'] = summary.apply(
        lambda r: f"{r['PPL (mean)']:.2f} ± {r['PPL (std)']:.2f}", axis=1
    )
    
    result = summary[['PPL (mean±std)']].reset_index()
    result.columns = ['Text Type', 'PPL (mean±std)']
    
    if format == "markdown":
        table = result.to_markdown(index=False)
    elif format == "latex":
        table = result.to_latex(index=False)
    else:
        table = result.to_string(index=False)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(table)
        logger.info(f"Saved perplexity table to {output_path}")
    
    return table


def create_combined_results_table(
    detector_metrics: List[Dict],
    quality_metrics: Optional[List[Dict]] = None,
    output_path: str = None,
    format: str = "markdown",
) -> str:
    """
    Create combined detection + quality results table (Adversarial Paraphrasing Table 2 style).
    
    Columns: Method | Det1 AUC | Det1 T@1%F | Det2 AUC | Det2 T@1%F | ... | Rating
    
    Args:
        detector_metrics: List of dicts with method, detector, auroc, tpr_at_1fpr
        quality_metrics: Optional list of dicts with method, quality_rating
        output_path: Path to save table
        format: Output format
    
    Returns:
        Table string
    """
    df = pd.DataFrame(detector_metrics)
    methods = df['method'].unique()
    detectors = df['detector'].unique()
    
    rows = []
    for method in methods:
        row = {"Method": method}
        method_data = df[df['method'] == method]
        
        for detector in detectors:
            det_data = method_data[method_data['detector'] == detector]
            if len(det_data) > 0:
                d = det_data.iloc[0]
                # Use shorter detector name
                det_short = detector[:8] if len(detector) > 8 else detector
                row[f"{det_short}_AUC"] = f"{d['auroc']:.3f}"
                row[f"{det_short}_T@1%F"] = f"{d['tpr_at_1fpr']:.3f}"
        
        # Add quality rating if available
        if quality_metrics:
            qdf = pd.DataFrame(quality_metrics)
            method_quality = qdf[qdf['method'] == method]
            if len(method_quality) > 0 and 'quality_rating' in method_quality.columns:
                mean_rating = method_quality['quality_rating'].mean()
                std_rating = method_quality['quality_rating'].std()
                row["Rating"] = f"{mean_rating:.2f}±{std_rating:.2f}"
        
        rows.append(row)
    
    result_df = pd.DataFrame(rows)
    
    if format == "markdown":
        table = result_df.to_markdown(index=False)
    elif format == "latex":
        table = result_df.to_latex(index=False, escape=False)
    else:
        table = result_df.to_string(index=False)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(table)
        logger.info(f"Saved combined results table to {output_path}")
    
    return table


# ============================================================================
# Table Generation
# ============================================================================

def format_metric_with_ci(
    mean: float,
    ci_low: float,
    ci_high: float,
    fmt: str = ".3f",
) -> str:
    """Format metric with confidence interval."""
    return f"{mean:{fmt}} [{ci_low:{fmt}}, {ci_high:{fmt}}]"


def create_main_results_table(
    metrics: List[Dict],
    output_path: str = None,
    format: str = "markdown",
) -> str:
    """
    Create main results table (Table A in SPEC.md).
    
    Columns: Method | Detector1 AUC/TPR/ASR | Detector2 ... | Mean
    """
    df = pd.DataFrame(metrics)
    
    # Pivot to get methods as rows, detectors as columns
    methods = df['method'].unique()
    detectors = df['detector'].unique()
    
    # Build table
    rows = []
    for method in methods:
        row = {"Method": method}
        
        method_data = df[df['method'] == method]
        
        for detector in detectors:
            det_data = method_data[method_data['detector'] == detector]
            if len(det_data) > 0:
                d = det_data.iloc[0]
                row[f"{detector}_AUC"] = f"{d['auroc']:.3f}"
                row[f"{detector}_TPR"] = f"{d['tpr_at_1fpr']:.3f}"
                row[f"{detector}_ASR"] = f"{d['asr']:.3f}"
        
        # Mean across detectors
        row["Mean_TPR"] = f"{method_data['tpr_at_1fpr'].mean():.3f}"
        row["Mean_ASR"] = f"{method_data['asr'].mean():.3f}"
        
        rows.append(row)
    
    result_df = pd.DataFrame(rows)
    
    if format == "markdown":
        table = result_df.to_markdown(index=False)
    elif format == "latex":
        table = result_df.to_latex(index=False)
    else:
        table = result_df.to_string(index=False)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(table)
        logger.info(f"Saved main results table to {output_path}")
    
    return table


def create_transfer_table(
    metrics: List[Dict],
    output_path: str = None,
    format: str = "markdown",
) -> str:
    """
    Create transfer matrix table (Table B in SPEC.md).
    
    Shows TPR@1%FPR for each (method × detector) combination.
    """
    df = pd.DataFrame(metrics)
    
    # Pivot: rows=detectors, columns=methods, values=TPR
    pivot = df.pivot_table(
        index='detector',
        columns='method',
        values='tpr_at_1fpr',
        aggfunc='mean',
    )
    
    if format == "markdown":
        table = pivot.round(3).to_markdown()
    elif format == "latex":
        table = pivot.round(3).to_latex()
    else:
        table = pivot.round(3).to_string()
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(table)
        logger.info(f"Saved transfer table to {output_path}")
    
    return table


def create_quality_table(
    quality_metrics: List[Dict],
    output_path: str = None,
    format: str = "markdown",
) -> str:
    """
    Create quality metrics summary table.
    """
    df = pd.DataFrame(quality_metrics)
    
    # Aggregate by method
    summary = df.groupby('method').agg({
        'sim_e5': ['mean', 'std'],
        'ppl_score': ['mean', 'std'],
        'edit_rate': ['mean', 'std'],
        'valid': 'mean',
    }).round(3)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    if format == "markdown":
        table = summary.to_markdown(index=False)
    elif format == "latex":
        table = summary.to_latex(index=False)
    else:
        table = summary.to_string(index=False)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(table)
        logger.info(f"Saved quality table to {output_path}")
    
    return table


def generate_qualitative_examples(
    examples: List[Dict],
    n_examples: int = 10,
    output_path: str = None,
) -> str:
    """
    Generate qualitative examples document.
    
    Args:
        examples: List of dicts with original, paraphrased, method, scores
        n_examples: Number of examples to include
        output_path: Path to save markdown file
    
    Returns:
        Markdown string
    """
    markdown = "# Qualitative Examples\n\n"
    
    for i, ex in enumerate(examples[:n_examples]):
        markdown += f"## Example {i+1}\n\n"
        markdown += f"**Method**: {ex.get('method', 'N/A')}\n\n"
        
        markdown += "### Original\n"
        markdown += f"```\n{ex.get('original', 'N/A')}\n```\n\n"
        
        markdown += "### Paraphrased\n"
        markdown += f"```\n{ex.get('paraphrased', 'N/A')}\n```\n\n"
        
        if 'scores' in ex:
            markdown += "### Detector Scores\n"
            for det, score in ex['scores'].items():
                markdown += f"- {det}: {score:.3f}\n"
        
        markdown += "\n---\n\n"
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(markdown)
        logger.info(f"Saved qualitative examples to {output_path}")
    
    return markdown


def _create_quality_evasion_scatter(
    detector_metrics: pd.DataFrame,
    quality_metrics: pd.DataFrame,
    output_path: str = None,
) -> None:
    """
    Create scatter plot showing quality vs evasion tradeoff per method.
    
    X-axis: Semantic similarity (higher = better quality)
    Y-axis: Mean ASR or (1 - TPR@1%FPR) (higher = better evasion)
    """
    import matplotlib.pyplot as plt
    
    # Aggregate metrics by method
    evasion_by_method = detector_metrics.groupby('method').agg({
        'asr': 'mean',
        'tpr_at_1fpr': 'mean',
    }).reset_index()
    
    quality_by_method = quality_metrics.groupby('method').agg({
        'sim_e5': 'mean',
        'ppl_score': 'mean',
    }).reset_index()
    
    merged = evasion_by_method.merge(quality_by_method, on='method', how='inner')
    
    if merged.empty:
        logger.warning("No data for quality vs evasion scatter")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for _, row in merged.iterrows():
        method = row['method']
        color = COLORS.get(method, '#333333')
        label = METHOD_NAMES.get(method, method)
        
        ax.scatter(
            row['sim_e5'],
            row['asr'],
            s=200,
            c=color,
            label=label,
            edgecolors='black',
            linewidths=1.5,
            zorder=5,
        )
        
        # Add method label next to point
        ax.annotate(
            label.split('(')[0].strip(),
            (row['sim_e5'], row['asr']),
            xytext=(8, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
        )
    
    ax.set_xlabel('Semantic Similarity (E5) ↑', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attack Success Rate (ASR) ↑', fontsize=12, fontweight='bold')
    ax.set_title('Quality vs Evasion Tradeoff', fontsize=14, fontweight='bold')
    
    # Add ideal region annotation
    ax.annotate(
        '★ Ideal\n(high quality,\nhigh evasion)',
        xy=(0.95, 0.95),
        xycoords='axes fraction',
        fontsize=10,
        ha='right',
        va='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
    )
    
    ax.set_xlim(0.5, 1.05)
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved quality vs evasion scatter to {output_path}")
    
    plt.close()


def _create_asr_comparison_chart(
    detector_metrics: pd.DataFrame,
    output_path: str = None,
) -> None:
    """
    Create grouped bar chart comparing ASR across methods and detectors.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Pivot to get methods as rows, detectors as columns
    pivot = detector_metrics.pivot_table(
        index='method',
        columns='detector',
        values='asr',
        aggfunc='mean',
    )
    
    if pivot.empty:
        logger.warning("No data for ASR comparison chart")
        return
    
    methods = pivot.index.tolist()
    detectors = pivot.columns.tolist()
    
    x = np.arange(len(methods))
    width = 0.8 / len(detectors)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    detector_colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442']
    
    for i, detector in enumerate(detectors):
        offset = (i - len(detectors)/2 + 0.5) * width
        values = pivot[detector].values
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=detector.replace('_', ' ').title(),
            color=detector_colors[i % len(detector_colors)],
            edgecolor='black',
            linewidth=0.5,
        )
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{val:.0%}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=45,
                )
    
    ax.set_xlabel('Attack Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attack Success Rate (ASR) ↑', fontsize=12, fontweight='bold')
    ax.set_title('ASR Comparison: Methods × Detectors', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_NAMES.get(m, m) for m in methods], rotation=30, ha='right')
    ax.legend(title='Detector', loc='upper left', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ASR comparison chart to {output_path}")
    
    plt.close()


def _create_perplexity_similarity_scatter(
    quality_metrics: pd.DataFrame,
    output_path: str = None,
) -> None:
    """
    Create scatter plot showing perplexity vs semantic similarity per method.
    
    Ideal position: high similarity (right), low perplexity (bottom).
    """
    import matplotlib.pyplot as plt
    
    # Aggregate by method
    agg = quality_metrics.groupby('method').agg({
        'sim_e5': ['mean', 'std'],
        'ppl_score': ['mean', 'std'],
    }).reset_index()
    
    # Flatten column names
    agg.columns = ['method', 'sim_mean', 'sim_std', 'ppl_mean', 'ppl_std']
    
    if agg.empty:
        logger.warning("No data for perplexity vs similarity scatter")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for _, row in agg.iterrows():
        method = row['method']
        color = COLORS.get(method, '#333333')
        label = METHOD_NAMES.get(method, method)
        
        # Plot point with error bars
        ax.errorbar(
            row['sim_mean'],
            row['ppl_mean'],
            xerr=row['sim_std'],
            yerr=row['ppl_std'],
            fmt='o',
            markersize=12,
            color=color,
            label=label,
            capsize=5,
            capthick=2,
            elinewidth=1.5,
            markeredgecolor='black',
            markeredgewidth=1,
        )
        
        # Add method label
        ax.annotate(
            label.split('(')[0].strip(),
            (row['sim_mean'], row['ppl_mean']),
            xytext=(10, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
        )
    
    ax.set_xlabel('Semantic Similarity (E5) ↑', fontsize=12, fontweight='bold')
    ax.set_ylabel('Perplexity ↓', fontsize=12, fontweight='bold')
    ax.set_title('Quality Tradeoff: Similarity vs Perplexity', fontsize=14, fontweight='bold')
    
    # Add ideal region annotation
    ax.annotate(
        '★ Ideal\n(high similarity,\nlow perplexity)',
        xy=(0.95, 0.05),
        xycoords='axes fraction',
        fontsize=10,
        ha='right',
        va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
    )
    
    ax.set_xlim(0.5, 1.05)
    ax.set_yscale('log')  # Log scale for perplexity (can vary widely)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved perplexity vs similarity scatter to {output_path}")
    
    plt.close()


def _create_human_ai_separation_from_df(
    scores_data: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Create human vs AI separation plots from a scores DataFrame.
    Creates one plot per detector showing human vs AI (per method) distributions.
    """
    import matplotlib.pyplot as plt
    
    detectors = scores_data['detector_name'].unique()
    
    for detector in detectors:
        det_data = scores_data[scores_data['detector_name'] == detector]
        
        # Get human scores (label == 'human' and method == 'm0' or 'no_attack')
        human_data = det_data[det_data['label'] == 'human']
        if human_data.empty:
            continue
        human_scores = human_data['detector_score'].values
        
        # Get AI scores per method (only AI samples)
        ai_data = det_data[det_data['label'] == 'ai']
        if ai_data.empty:
            continue
        
        ai_scores_by_method = {}
        for method in ai_data['method'].unique():
            method_scores = ai_data[ai_data['method'] == method]['detector_score'].values
            if len(method_scores) > 0:
                ai_scores_by_method[method] = method_scores
        
        if not ai_scores_by_method:
            continue
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Human distribution
        ax.hist(human_scores, bins=30, alpha=0.6, label='Human', color='#7f7f7f', 
                density=True, edgecolor='black', linewidth=0.5)
        
        # AI distributions for each method
        for i, (method, scores) in enumerate(ai_scores_by_method.items()):
            color = COLORS.get(method, COLORBLIND_COLORS[i % len(COLORBLIND_COLORS)])
            label = METHOD_NAMES.get(method, method)
            ax.hist(scores, bins=30, alpha=0.5, label=f'AI: {label}',
                    color=color, density=True, histtype='step', linewidth=2.5)
        
        ax.set_xlabel('Detection Score (higher = more AI-like)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f"Human vs AI Score Separation\n({detector.replace('_', ' ').title()})", 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = output_dir / f"fig_human_ai_separation_{detector}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved human-AI separation plot to {output_path}")
        
        plt.close()


def generate_all_plots(
    detector_metrics: pd.DataFrame,
    quality_metrics: pd.DataFrame,
    budget_sweep_data: Optional[pd.DataFrame] = None,
    predictions: Optional[Dict] = None,
    scores_data: Optional[pd.DataFrame] = None,
    output_dir: str = "artifacts/figures",
):
    """
    Generate all paper figures including AuthorMist-style AUROC diagrams
    and Adversarial Paraphrasing-style visualizations.
    
    Args:
        detector_metrics: DataFrame with detector evaluation metrics
        quality_metrics: DataFrame with quality metrics
        budget_sweep_data: Optional DataFrame with budget sweep results
        predictions: Optional dict {detector: {method: (y_true, y_scores)}} for ROC curves
        output_dir: Directory to save figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Heatmaps (SPEC.md Figure 1)
    # ========================================================================
    
    # Figure 1a: TPR@1%FPR heatmap
    create_heatmap(
        detector_metrics,
        value_col="tpr_at_1fpr",
        title="TPR@1%FPR: Detector × Method",
        output_path=str(output_dir / "fig_heatmap_tpr.png"),
    )
    
    # Figure 1b: AUROC heatmap
    if 'auroc' in detector_metrics.columns:
        create_heatmap(
            detector_metrics,
            value_col="auroc",
            title="AUROC: Detector × Method (↓ better attack)",
            output_path=str(output_dir / "fig_heatmap_auroc.png"),
            cmap="RdYlGn",  # Green=low AUROC=good attack
        )
    
    # Figure 1c: Transferability heatmap (relative reduction from baseline)
    if 'no_attack' in detector_metrics['method'].values:
        create_transferability_heatmap(
            detector_metrics,
            baseline_col="no_attack",
            title="Relative TPR@1%FPR Reduction from Baseline (%)",
            output_path=str(output_dir / "fig_transferability_heatmap.png"),
        )
    
    # ========================================================================
    # AUROC Visualizations (AuthorMist-style)
    # ========================================================================
    
    if 'auroc' in detector_metrics.columns:
        # Figure 2a: AUROC bar chart
        create_auroc_bar_chart(
            detector_metrics,
            title="AUROC Comparison Across Detectors",
            output_path=str(output_dir / "fig_auroc_bars.png"),
        )
        
        # Figure 2b: AUROC radar chart
        create_auroc_radar_chart(
            detector_metrics,
            title="Attack Effectiveness Across Detectors",
            output_path=str(output_dir / "fig_auroc_radar.png"),
        )
    
    # Figure 3: Comprehensive 4-panel method comparison
    create_method_comparison_summary(
        detector_metrics,
        output_path=str(output_dir / "fig_method_comparison.png"),
    )
    
    # ========================================================================
    # ROC Curves (Adversarial Paraphrasing-style)
    # ========================================================================
    
    if predictions is not None:
        # Figure 4a: Standard ROC curves
        create_roc_curves(
            predictions,
            title="ROC Curves by Detector",
            output_path=str(output_dir / "fig_roc_curves.png"),
        )
        
        # Figure 4b: Log-scale FPR ROC curves (highlights low-FPR regime)
        create_roc_curves_logscale(
            predictions,
            title="ROC Curves (Log-Scale FPR)",
            output_path=str(output_dir / "fig_roc_curves_logscale.png"),
        )
    
    # ========================================================================
    # Tradeoff Plot (SPEC.md Figure 2)
    # ========================================================================
    
    # Prepare tradeoff data
    tradeoff_data = detector_metrics.groupby('method').agg({
        'tpr_at_1fpr': 'mean',
    }).reset_index()
    tradeoff_data.columns = ['method', 'mean_tpr']
    
    # Add quality metrics
    if 'sim_e5' in quality_metrics.columns:
        quality_agg = quality_metrics.groupby('method')['sim_e5'].median().reset_index()
        tradeoff_data = tradeoff_data.merge(quality_agg, on='method', how='left')
        
        # Figure 5: Tradeoff plot
        create_tradeoff_plot(
            tradeoff_data,
            title="Evasion-Quality Tradeoff",
            output_path=str(output_dir / "fig_tradeoff.png"),
        )
    
    # ========================================================================
    # Budget Sweep (SPEC.md Figure 3)
    # ========================================================================
    
    if budget_sweep_data is not None:
        create_budget_sweep_plot(
            budget_sweep_data,
            title="Candidate Budget Sweep",
            output_path=str(output_dir / "fig_budget_sweep.png"),
        )
    
    # ========================================================================
    # Quality Visualizations (Adversarial Paraphrasing-style)
    # ========================================================================
    
    if 'quality_rating' in quality_metrics.columns:
        create_quality_likert_chart(
            quality_metrics,
            title="Text Quality Evaluation",
            output_path=str(output_dir / "fig_quality_likert.png"),
        )
    
    # ========================================================================
    # Score Distribution Plots (NEW - Adversarial Paraphrasing-style)
    # ========================================================================
    
    if scores_data is not None and not scores_data.empty:
        # Check if required columns exist
        required_cols = ['detector_name', 'method', 'detector_score']
        if all(col in scores_data.columns for col in required_cols):
            # Figure: Score distributions per detector and method
            create_score_distribution_plot(
                scores_data,
                title="Detection Score Distributions by Method",
                output_path=str(output_dir / "fig_score_distributions.png"),
            )
            
            # Figure: Human vs AI separation per detector (only if we have label column)
            if 'label' in scores_data.columns:
                _create_human_ai_separation_from_df(
                    scores_data,
                    output_dir=output_dir,
                )
        else:
            logger.warning(f"Scores data missing required columns: {required_cols}. Found: {scores_data.columns.tolist()}")
    
    # ========================================================================
    # Quality vs Evasion Scatter Plot (NEW)
    # ========================================================================
    
    if 'sim_e5' in quality_metrics.columns and not quality_metrics.empty:
        _create_quality_evasion_scatter(
            detector_metrics,
            quality_metrics,
            output_path=str(output_dir / "fig_quality_vs_evasion.png"),
        )
    
    # ========================================================================
    # ASR Comparison Bar Chart (NEW)
    # ========================================================================
    
    if 'asr' in detector_metrics.columns:
        _create_asr_comparison_chart(
            detector_metrics,
            output_path=str(output_dir / "fig_asr_comparison.png"),
        )
    
    # ========================================================================
    # Perplexity vs Similarity Scatter (NEW - Quality Tradeoff)
    # ========================================================================
    
    if 'sim_e5' in quality_metrics.columns and 'ppl_score' in quality_metrics.columns:
        _create_perplexity_similarity_scatter(
            quality_metrics,
            output_path=str(output_dir / "fig_perplexity_vs_similarity.png"),
        )
    
    logger.info(f"All figures saved to {output_dir}")


# ============================================================================
# GPT-4o Quality Evaluation Templates (Adversarial Paraphrasing-style)
# ============================================================================

QUALITY_RATING_PROMPT = """
You are an expert linguist and paraphrase evaluator. Your task is to assess the quality of a paraphrased text compared to the original source text. Use the following scoring criteria:

5 - Approximately equivalent: Meaning is preserved; differences are only in wording or structure.
4 - Nearly equivalent: Meaning is mostly preserved; minor factual details differ.
3 - Somewhat equivalent: Some meaning is preserved; important details or meanings differ.
2 - Topically related: The texts are on the same topic but most meaning is lost.
1 - Not topically related: The texts are not related in topic or meaning.

Provide your final output as a JSON object in this format:
{
"score": <score from 1 to 5>,
"justification": "<brief explanation>"
}
"""

WIN_RATE_PROMPT = """
You are a neutral and highly capable evaluator. Your task is to compare two versions of a text and select the one that communicates the ideas more clearly, accurately, and fluently. Consider:

- Fidelity of meaning (how clearly the core message is conveyed)
- Clarity and conciseness
- Grammatical correctness and fluency
- Naturalness and appropriateness of phrasing

Give your vote solely based on quality. If one of them is a CLEAR winner, ONLY then vote for that one. Otherwise, vote for `tie`.

Respond with **only one of the following**, and nothing else:
- text1
- text2
- tie
"""


def get_quality_rating_messages(original_text: str, paraphrased_text: str) -> List[Dict]:
    """
    Generate GPT-4o messages for quality rating evaluation.
    
    Args:
        original_text: The original AI-generated text
        paraphrased_text: The paraphrased version
    
    Returns:
        List of message dicts for OpenAI API
    """
    user_prompt = f"""
Evaluate the following paraphrase using the criteria above:

Original Text:
\"\"\"{original_text}\"\"\"

Paraphrased Text:
\"\"\"{paraphrased_text}\"\"\"

What score (1 to 5) would you assign to this paraphrase, and why?
"""
    
    return [
        {"role": "system", "content": QUALITY_RATING_PROMPT},
        {"role": "user", "content": user_prompt}
    ]


def get_win_rate_messages(text1: str, text2: str) -> List[Dict]:
    """
    Generate GPT-4o messages for head-to-head quality comparison.
    
    Args:
        text1: First text (e.g., original or simple paraphrase)
        text2: Second text (e.g., adversarial paraphrase)
    
    Returns:
        List of message dicts for OpenAI API
    """
    user_prompt = f"""
Compare the following two texts and give your vote depending on meaning clarity, fluency, and overall quality. If one of them is a CLEAR winner, ONLY then vote for that one. Otherwise, vote for `tie`. Respond with one of these 3 options: `text1`, `text2`, `tie`.

Text 1:
\"\"\"{text1}\"\"\"

Text 2:
\"\"\"{text2}\"\"\"
"""
    
    return [
        {"role": "system", "content": WIN_RATE_PROMPT},
        {"role": "user", "content": user_prompt}
    ]


def parse_quality_rating_response(response: str) -> Tuple[int, str]:
    """
    Parse GPT-4o quality rating response.
    
    Args:
        response: Raw response from GPT-4o
    
    Returns:
        Tuple of (score, justification)
    """
    import json
    import re
    
    try:
        # Try to parse as JSON
        data = json.loads(response)
        return int(data.get('score', 3)), data.get('justification', '')
    except json.JSONDecodeError:
        # Fallback: extract score from text
        match = re.search(r'(?:score[:\s]*)?([1-5])', response.lower())
        if match:
            return int(match.group(1)), response
        return 3, response  # Default to neutral score


def parse_win_rate_response(response: str) -> str:
    """
    Parse GPT-4o win-rate response.
    
    Args:
        response: Raw response from GPT-4o
    
    Returns:
        One of: 'text1', 'text2', 'tie'
    """
    response = response.strip().lower()
    if 'text1' not in response and 'text2' not in response and 'tie' not in response:
        return 'tie'
    elif 'text1' in response and 'text2' in response:
        return 'tie'
    elif 'text1' in response:
        return 'text1'
    elif 'text2' in response:
        return 'text2'
    else:
        return 'tie'


def generate_all_tables(
    detector_metrics: List[Dict],
    quality_metrics: List[Dict],
    output_dir: str = "artifacts/tables",
    format: str = "markdown",
):
    """
    Generate all paper tables.
    
    Args:
        detector_metrics: List of detector metric dicts
        quality_metrics: List of quality metric dicts
        output_dir: Directory to save tables
        format: Output format (markdown, latex)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ext = ".md" if format == "markdown" else ".tex"
    
    create_main_results_table(
        detector_metrics,
        output_path=str(output_dir / f"table_main_mage{ext}"),
        format=format,
    )
    
    create_transfer_table(
        detector_metrics,
        output_path=str(output_dir / f"table_transfer{ext}"),
        format=format,
    )
    
    create_quality_table(
        quality_metrics,
        output_path=str(output_dir / f"table_quality{ext}"),
        format=format,
    )
    
    logger.info(f"All tables saved to {output_dir}")
