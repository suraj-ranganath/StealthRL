"""
StealthBench Visualization Scripts.

Generates publication-ready plots for detector comparison:
- ROC curves (all detectors on same axes)
- FPR comparison bar charts
- Pareto frontier (detectability vs semantic vs quality)
- ESL fairness heatmaps
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve

# Set publication-quality defaults
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
})


class StealthBenchVisualizer:
    """
    Visualization suite for StealthBench evaluation results.
    
    Generates standardized plots for comparing StealthRL against
    base AI text and optional SFT baseline across multiple detectors.
    """
    
    def __init__(self, output_dir: str = "outputs/visualizations"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_roc_curves(
        self,
        evaluation_results: Dict[str, Dict],
        save_name: str = "roc_curves.pdf",
    ):
        """
        Plot ROC curves for all detectors on same axes.
        
        Args:
            evaluation_results: Dictionary mapping model names to results
                Format: {
                    "base": {
                        "detector_name": {"scores": [...], "labels": [...]},
                        ...
                    },
                    "stealthrl": {...}
                }
            save_name: Output filename
        """
        fig, axes = plt.subplots(1, len(evaluation_results["base"]), figsize=(15, 5))
        if len(evaluation_results["base"]) == 1:
            axes = [axes]
        
        colors = {
            "base": "#e74c3c",  # Red
            "sft": "#3498db",   # Blue
            "stealthrl": "#2ecc71",  # Green
        }
        
        detector_names = list(evaluation_results["base"].keys())
        
        for idx, detector_name in enumerate(detector_names):
            ax = axes[idx]
            
            for model_name in ["base", "sft", "stealthrl"]:
                if model_name not in evaluation_results:
                    continue
                
                data = evaluation_results[model_name][detector_name]
                scores = np.array(data["scores"])
                labels = np.array(data["labels"])
                
                fpr, tpr, _ = roc_curve(labels, scores)
                auc_score = auc(fpr, tpr)
                
                label = f"{model_name.capitalize()} (AUC={auc_score:.3f})"
                ax.plot(fpr, tpr, label=label, color=colors[model_name], linewidth=2)
            
            # Plot random baseline
            ax.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.3)
            
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{detector_name.replace("_", " ").title()}')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved ROC curves to {self.output_dir / save_name}")
    
    def plot_fpr_comparison(
        self,
        metrics: Dict[str, Dict],
        save_name: str = "fpr_comparison.pdf",
    ):
        """
        Bar chart comparing FPR at TPR=95% across detectors.
        
        Args:
            metrics: Dictionary mapping model names to ModelMetrics
                Format: {
                    "base": ModelMetrics(...),
                    "stealthrl": ModelMetrics(...)
                }
            save_name: Output filename
        """
        detector_names = list(metrics["base"].fpr_at_tpr95_per_detector.keys())
        
        x = np.arange(len(detector_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = {
            "base": "#e74c3c",
            "sft": "#3498db",
            "stealthrl": "#2ecc71",
        }
        
        offset = 0
        for model_name in ["base", "sft", "stealthrl"]:
            if model_name not in metrics:
                continue
            
            fprs = [
                metrics[model_name].fpr_at_tpr95_per_detector[d]
                for d in detector_names
            ]
            
            ax.bar(
                x + offset,
                fprs,
                width,
                label=model_name.capitalize(),
                color=colors[model_name],
            )
            offset += width
        
        ax.set_xlabel('Detector')
        ax.set_ylabel('FPR @ TPR=95%')
        ax.set_title('False Positive Rate Comparison at TPR=95%')
        ax.set_xticks(x + width)
        ax.set_xticklabels([d.replace('_', ' ').title() for d in detector_names], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved FPR comparison to {self.output_dir / save_name}")
    
    def plot_low_fpr_comparison(
        self,
        metrics: Dict[str, Dict],
        save_name: str = "low_fpr_comparison.pdf",
    ):
        """
        Bar chart comparing TPR at low FPR thresholds (0.5%, 1.0%).
        
        Critical for academic integrity applications.
        """
        detector_names = list(metrics["base"].tpr_at_fpr_0_5_per_detector.keys())
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        colors = {
            "base": "#e74c3c",
            "sft": "#3498db",
            "stealthrl": "#2ecc71",
        }
        
        # FPR = 0.5%
        x = np.arange(len(detector_names))
        width = 0.25
        
        offset = 0
        for model_name in ["base", "sft", "stealthrl"]:
            if model_name not in metrics:
                continue
            
            tprs = [
                metrics[model_name].tpr_at_fpr_0_5_per_detector[d]
                for d in detector_names
            ]
            
            axes[0].bar(
                x + offset,
                tprs,
                width,
                label=model_name.capitalize(),
                color=colors[model_name],
            )
            offset += width
        
        axes[0].set_xlabel('Detector')
        axes[0].set_ylabel('TPR @ FPR=0.5%')
        axes[0].set_title('True Positive Rate at FPR=0.5% (Very Conservative)')
        axes[0].set_xticks(x + width)
        axes[0].set_xticklabels([d.replace('_', ' ').title() for d in detector_names], rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # FPR = 1.0%
        offset = 0
        for model_name in ["base", "sft", "stealthrl"]:
            if model_name not in metrics:
                continue
            
            tprs = [
                metrics[model_name].tpr_at_fpr_1_0_per_detector[d]
                for d in detector_names
            ]
            
            axes[1].bar(
                x + offset,
                tprs,
                width,
                label=model_name.capitalize(),
                color=colors[model_name],
            )
            offset += width
        
        axes[1].set_xlabel('Detector')
        axes[1].set_ylabel('TPR @ FPR=1.0%')
        axes[1].set_title('True Positive Rate at FPR=1.0% (Conservative)')
        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels([d.replace('_', ' ').title() for d in detector_names], rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved low-FPR comparison to {self.output_dir / save_name}")
    
    def plot_esl_fairness_heatmap(
        self,
        metrics: Dict[str, Dict],
        save_name: str = "esl_fairness_heatmap.pdf",
    ):
        """
        Heatmap of ESL fairness gaps across detectors and models.
        
        Shows FPR(ESL) - FPR(native) for each detector/model combination.
        Negative values (green) indicate reduced bias.
        """
        detector_names = list(metrics["base"].esl_fpr_gap.keys())
        model_names = list(metrics.keys())
        
        # Build data matrix
        data = []
        for detector in detector_names:
            row = [metrics[model].esl_fpr_gap[detector] for model in model_names]
            data.append(row)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=-0.2, vmax=0.2)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(model_names)))
        ax.set_yticks(np.arange(len(detector_names)))
        ax.set_xticklabels([m.capitalize() for m in model_names])
        ax.set_yticklabels([d.replace('_', ' ').title() for d in detector_names])
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('ESL FPR Gap (FPR_ESL - FPR_Native)', rotation=-90, va="bottom")
        
        # Add text annotations
        for i in range(len(detector_names)):
            for j in range(len(model_names)):
                text = ax.text(j, i, f"{data[i][j]:.3f}",
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('ESL Fairness Gap Across Detectors\n(Lower = More Fair)')
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved ESL fairness heatmap to {self.output_dir / save_name}")
    
    def plot_pareto_frontier(
        self,
        ablation_results: List[Dict],
        save_name: str = "pareto_frontier.pdf",
    ):
        """
        3D scatter plot showing Pareto frontier of ablation experiments.
        
        Axes:
        - X: Detectability (1 - avg_detector_prob)
        - Y: Semantic similarity
        - Z: Quality (perplexity score)
        Color: ESL fairness gap
        
        Args:
            ablation_results: List of ablation result dictionaries
                Each dict should have: {
                    "name": "detector_only",
                    "detectability": 0.7,
                    "semantic_sim": 0.85,
                    "quality": 0.9,
                    "esl_gap": 0.05,
                }
            save_name: Output filename
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract coordinates
        x = [r['detectability'] for r in ablation_results]
        y = [r['semantic_sim'] for r in ablation_results]
        z = [r['quality'] for r in ablation_results]
        colors = [r['esl_gap'] for r in ablation_results]
        labels = [r['name'] for r in ablation_results]
        
        # Scatter plot
        scatter = ax.scatter(
            x, y, z,
            c=colors,
            s=200,
            cmap='RdYlGn_r',
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5,
        )
        
        # Add labels
        for i, label in enumerate(labels):
            ax.text(x[i], y[i], z[i], f'  {label}', fontsize=8)
        
        # Labels and title
        ax.set_xlabel('Detectability (1 - P(AI))', labelpad=10)
        ax.set_ylabel('Semantic Similarity', labelpad=10)
        ax.set_zlabel('Quality (Perplexity Score)', labelpad=10)
        ax.set_title('Pareto Frontier: Ablation Experiments\n(Higher is Better for All Axes)', pad=20)
        
        # Colorbar
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.ax.set_ylabel('ESL Fairness Gap\n(Lower is Better)', rotation=-90, va="bottom")
        
        # Adjust viewing angle
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved Pareto frontier to {self.output_dir / save_name}")
    
    def plot_semantic_distributions(
        self,
        metrics: Dict[str, Dict],
        save_name: str = "semantic_distributions.pdf",
    ):
        """
        Violin plots showing semantic similarity distributions.
        
        Args:
            metrics: Dictionary with semantic similarity scores
                Format: {
                    "base": [0.85, 0.90, ...],
                    "stealthrl": [0.87, 0.92, ...]
                }
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = []
        labels = []
        colors = []
        
        color_map = {
            "base": "#e74c3c",
            "sft": "#3498db",
            "stealthrl": "#2ecc71",
        }
        
        for model_name in ["base", "sft", "stealthrl"]:
            if model_name in metrics and len(metrics[model_name]) > 0:
                data.append(metrics[model_name])
                labels.append(model_name.capitalize())
                colors.append(color_map[model_name])
        
        parts = ax.violinplot(data, positions=range(len(data)), showmeans=True, showmedians=True)
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Semantic Similarity')
        ax.set_title('Semantic Similarity Distributions')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal line at threshold
        ax.axhline(y=0.90, color='red', linestyle='--', alpha=0.5, label='Threshold (0.90)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved semantic distributions to {self.output_dir / save_name}")
    
    def generate_full_report(
        self,
        evaluation_results: Dict,
        metrics: Dict,
        ablation_results: Optional[List[Dict]] = None,
    ):
        """
        Generate complete visualization report.
        
        Args:
            evaluation_results: ROC curve data
            metrics: ModelMetrics for all models
            ablation_results: Optional ablation experiment results
        """
        print("\n" + "="*60)
        print("GENERATING STEALTHBENCH VISUALIZATION REPORT")
        print("="*60 + "\n")
        
        # Generate all plots
        self.plot_roc_curves(evaluation_results)
        self.plot_fpr_comparison(metrics)
        self.plot_low_fpr_comparison(metrics)
        self.plot_esl_fairness_heatmap(metrics)
        
        if ablation_results:
            self.plot_pareto_frontier(ablation_results)
        
        print(f"\n✓ All visualizations saved to {self.output_dir}/")
        print("="*60)


# Example usage
def main():
    """Example: Generate StealthBench visualizations."""
    
    # Mock data for demonstration
    evaluation_results = {
        "base": {
            "fast_detectgpt": {
                "scores": np.random.beta(6, 2, 100),
                "labels": np.ones(100),
            },
            "ghostbuster": {
                "scores": np.random.beta(5, 2, 100),
                "labels": np.ones(100),
            },
        },
        "stealthrl": {
            "fast_detectgpt": {
                "scores": np.random.beta(3, 5, 100),
                "labels": np.ones(100),
            },
            "ghostbuster": {
                "scores": np.random.beta(2, 5, 100),
                "labels": np.ones(100),
            },
        },
    }
    
    # Mock metrics (would come from EvaluationSuite)
    from stealthrl.tinker.evaluation import ModelMetrics
    
    metrics = {
        "base": ModelMetrics(
            asr_all=0.1, asr_any=0.3,
            auroc_per_detector={"fast_detectgpt": 0.7, "ghostbuster": 0.75},
            f1_per_detector={"fast_detectgpt": 0.8, "ghostbuster": 0.82},
            fpr_at_tpr95_per_detector={"fast_detectgpt": 0.3, "ghostbuster": 0.25},
            tpr_at_fpr_0_5_per_detector={"fast_detectgpt": 0.85, "ghostbuster": 0.88},
            tpr_at_fpr_1_0_per_detector={"fast_detectgpt": 0.90, "ghostbuster": 0.92},
            threshold_at_fpr_0_5_per_detector={"fast_detectgpt": 0.7, "ghostbuster": 0.75},
            threshold_at_fpr_1_0_per_detector={"fast_detectgpt": 0.6, "ghostbuster": 0.65},
            semantic_sim_mean=0.87, semantic_sim_std=0.05, semantic_sim_min=0.75,
            esl_fpr_gap={"fast_detectgpt": 0.15, "ghostbuster": 0.12},
            esl_auroc_gap={"fast_detectgpt": -0.05, "ghostbuster": -0.03},
            avg_detector_prob=0.75, avg_detector_prob_esl=0.82, avg_detector_prob_native=0.70,
        ),
        "stealthrl": ModelMetrics(
            asr_all=0.6, asr_any=0.8,
            auroc_per_detector={"fast_detectgpt": 0.4, "ghostbuster": 0.45},
            f1_per_detector={"fast_detectgpt": 0.5, "ghostbuster": 0.52},
            fpr_at_tpr95_per_detector={"fast_detectgpt": 0.6, "ghostbuster": 0.55},
            tpr_at_fpr_0_5_per_detector={"fast_detectgpt": 0.45, "ghostbuster": 0.50},
            tpr_at_fpr_1_0_per_detector={"fast_detectgpt": 0.55, "ghostbuster": 0.60},
            threshold_at_fpr_0_5_per_detector={"fast_detectgpt": 0.35, "ghostbuster": 0.40},
            threshold_at_fpr_1_0_per_detector={"fast_detectgpt": 0.30, "ghostbuster": 0.35},
            semantic_sim_mean=0.91, semantic_sim_std=0.03, semantic_sim_min=0.85,
            esl_fpr_gap={"fast_detectgpt": 0.05, "ghostbuster": 0.04},
            esl_auroc_gap={"fast_detectgpt": -0.02, "ghostbuster": -0.01},
            avg_detector_prob=0.35, avg_detector_prob_esl=0.38, avg_detector_prob_native=0.33,
        ),
    }
    
    # Mock ablation results
    ablation_results = [
        {"name": "full", "detectability": 0.65, "semantic_sim": 0.91, "quality": 0.88, "esl_gap": 0.05},
        {"name": "detector_only", "detectability": 0.70, "semantic_sim": 0.80, "quality": 0.70, "esl_gap": 0.15},
        {"name": "no_fairness", "detectability": 0.68, "semantic_sim": 0.90, "quality": 0.87, "esl_gap": 0.12},
        {"name": "no_semantic", "detectability": 0.72, "semantic_sim": 0.75, "quality": 0.85, "esl_gap": 0.08},
    ]
    
    # Generate visualizations
    visualizer = StealthBenchVisualizer(output_dir="outputs/visualizations")
    visualizer.generate_full_report(evaluation_results, metrics, ablation_results)


if __name__ == "__main__":
    main()
