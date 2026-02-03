#!/usr/bin/env python3
"""
Run all ablation experiments for StealthRL evaluation.

This script runs a comprehensive set of ablation studies:
1. Guidance Transfer (§7.2) - Test guidance detector vs deploy detector transfer
2. Budget Sweep (§8.1) - Test N=1,2,4,8 candidates for M1, M2, M3
3. Homoglyph Sweep (§8.3) - Test p=0.1%, 0.5%, 1%, 2% substitution rates
4. Sanitize Defense (§9) - Test sanitization defense on homoglyph outputs
5. Cross-Dataset Transfer - Test on PadBen dataset

Usage:
    python scripts/run_ablations.py \
        --stealthrl-checkpoint checkpoints/ckpt_example.json \
        --n-human 100 --n-ai 100 \
        --out-dir outputs/ablations
        
    # Run specific ablations only:
    python scripts/run_ablations.py \
        --ablations guidance budget sanitize padben \
        --stealthrl-checkpoint checkpoints/ckpt_example.json
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.data import load_eval_dataset
from eval.detectors import load_detectors
from eval.methods import get_method, METHOD_REGISTRY, GUIDANCE_VARIANTS, HomoglyphSweep
from eval.metrics import compute_detector_metrics, compute_quality_metrics, E5SimilarityScorer, PerplexityScorer
from eval.sanitize import sanitize, run_sanitize_evaluation, create_sanitize_report
from eval.plots import (
    create_heatmap,
    create_budget_sweep_plot,
    create_sanitize_plot,
    create_tradeoff_plot,
)

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str, level: str = "INFO"):
    """Setup logging configuration."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(log_dir) / f"ablations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ]
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )


class AblationRunner:
    """
    Runs all ablation experiments.
    """
    
    def __init__(
        self,
        output_dir: str,
        stealthrl_checkpoint: str = None,
        device: str = None,
        seed: int = 42,
        n_bootstrap: int = 500,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stealthrl_checkpoint = stealthrl_checkpoint
        self.device = device
        self.seed = seed
        self.n_bootstrap = n_bootstrap
        
        # Will be populated during run
        self.detectors = {}
        self.datasets = {}
        self.results = {}
        
    def load_detectors(self, detector_names: List[str] = None):
        """Load detector panel."""
        if detector_names is None:
            detector_names = ["roberta", "fast_detectgpt", "binoculars"]
        
        logger.info(f"Loading detectors: {detector_names}")
        self.detectors = load_detectors(detector_names, device=self.device)
        
        for name, detector in self.detectors.items():
            detector.load()
            
    def load_dataset(self, name: str, n_human: int, n_ai: int, cache_dir: str = None):
        """Load a dataset."""
        logger.info(f"Loading dataset: {name} ({n_human} human, {n_ai} AI)")
        dataset = load_eval_dataset(
            name=name,
            n_human=n_human,
            n_ai=n_ai,
            cache_dir=cache_dir,
            seed=self.seed,
        )
        self.datasets[name] = dataset
        return dataset
    
    # =========================================================================
    # Ablation 7.2: Guidance Transfer
    # =========================================================================
    
    def run_guidance_transfer_ablation(
        self,
        dataset_name: str = "mage",
        n_human: int = 100,
        n_ai: int = 100,
        n_candidates: int = 4,
    ) -> pd.DataFrame:
        """
        Run guidance transfer ablation (§7.2).
        
        Tests whether the detector used for guidance during candidate selection
        transfers to other detectors at evaluation time.
        
        Guidance variants:
        - m3_roberta: Select candidates minimizing RoBERTa score
        - m3_fastdetect: Select candidates minimizing Fast-DetectGPT score
        - m3_ensemble: Select candidates minimizing ensemble mean score
        
        Returns:
            DataFrame with TPR@1%FPR for each (guidance_detector, eval_detector) pair
        """
        logger.info("=" * 60)
        logger.info("ABLATION 7.2: Guidance Transfer")
        logger.info("=" * 60)
        
        ablation_dir = self.output_dir / "guidance_transfer"
        ablation_dir.mkdir(exist_ok=True)
        
        # Load dataset
        if dataset_name not in self.datasets:
            self.load_dataset(dataset_name, n_human, n_ai)
        dataset = self.datasets[dataset_name]
        
        # Guidance methods to test
        guidance_methods = ["m3_roberta", "m3_fastdetect", "m3_ensemble"]
        
        all_results = []
        
        for guidance_method in guidance_methods:
            logger.info(f"\nRunning {guidance_method}...")
            
            # Get method
            method = get_method(guidance_method)
            
            # Run attacks
            ai_texts = [s.text for s in dataset.ai_samples]
            results = method.attack_batch(ai_texts, n_candidates=n_candidates)
            attacked_texts = [r.text for r in results]
            
            # Score with all detectors
            for det_name, detector in self.detectors.items():
                # Get human scores for threshold calibration
                human_texts = [s.text for s in dataset.human_samples]
                human_scores = detector.get_scores(human_texts)
                
                # Get AI scores
                ai_scores = detector.get_scores(attacked_texts)
                
                # Compute metrics
                metrics = compute_detector_metrics(
                    human_scores=human_scores,
                    ai_scores=ai_scores,
                    detector=det_name,
                    method=guidance_method,
                    dataset=dataset_name,
                    n_bootstrap=self.n_bootstrap,
                    seed=self.seed,
                )
                
                all_results.append({
                    "guidance_detector": guidance_method.replace("m3_", ""),
                    "eval_detector": det_name,
                    "tpr_at_1fpr": metrics.tpr_at_1fpr,
                    "asr": metrics.asr,
                    "auroc": metrics.auroc,
                })
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        df.to_csv(ablation_dir / "guidance_transfer_results.csv", index=False)
        
        # Create heatmap
        pivot_df = df.pivot(index="eval_detector", columns="guidance_detector", values="tpr_at_1fpr")
        
        # Save pivot table
        pivot_df.to_csv(ablation_dir / "guidance_transfer_matrix.csv")
        
        logger.info(f"\nGuidance Transfer Results:")
        logger.info(f"\n{pivot_df.to_string()}")
        
        self.results["guidance_transfer"] = df
        return df
    
    # =========================================================================
    # Ablation 8.1: Budget Sweep
    # =========================================================================
    
    def run_budget_sweep_ablation(
        self,
        dataset_name: str = "mage",
        n_human: int = 100,
        n_ai: int = 100,
        n_values: List[int] = None,
        methods: List[str] = None,
    ) -> pd.DataFrame:
        """
        Run budget sweep ablation (§8.1).
        
        Tests how performance changes with number of candidates N.
        
        Args:
            n_values: List of N values to test (default: [1, 2, 4, 8])
            methods: Methods to test (default: ["m1", "m2"])
            
        Returns:
            DataFrame with metrics for each (method, N) combination
        """
        logger.info("=" * 60)
        logger.info("ABLATION 8.1: Budget Sweep")
        logger.info("=" * 60)
        
        ablation_dir = self.output_dir / "budget_sweep"
        ablation_dir.mkdir(exist_ok=True)
        
        if n_values is None:
            n_values = [1, 2, 4, 8]
        if methods is None:
            methods = ["m1", "m2"]
        
        # Load dataset
        if dataset_name not in self.datasets:
            self.load_dataset(dataset_name, n_human, n_ai)
        dataset = self.datasets[dataset_name]
        
        # Initialize quality scorers
        sim_scorer = E5SimilarityScorer(device=self.device)
        ppl_scorer = PerplexityScorer(device=self.device)
        
        all_results = []
        
        for method_name in methods:
            logger.info(f"\nMethod: {method_name}")
            
            # Get method
            if method_name == "m2" and self.stealthrl_checkpoint:
                method = get_method(method_name, checkpoint_json=self.stealthrl_checkpoint)
            else:
                method = get_method(method_name)
            
            for n_cand in n_values:
                logger.info(f"  N={n_cand}...")
                
                # Run attacks
                ai_texts = [s.text for s in dataset.ai_samples]
                results = method.attack_batch(ai_texts, n_candidates=n_cand)
                attacked_texts = [r.text for r in results]
                
                # Compute quality metrics
                similarities = sim_scorer.compute_similarity(ai_texts, attacked_texts)
                mean_sim = np.mean(similarities)
                
                # Score with all detectors
                for det_name, detector in self.detectors.items():
                    human_texts = [s.text for s in dataset.human_samples]
                    human_scores = detector.get_scores(human_texts)
                    ai_scores = detector.get_scores(attacked_texts)
                    
                    metrics = compute_detector_metrics(
                        human_scores=human_scores,
                        ai_scores=ai_scores,
                        detector=det_name,
                        method=method_name,
                        dataset=dataset_name,
                        n_bootstrap=self.n_bootstrap,
                        seed=self.seed,
                    )
                    
                    all_results.append({
                        "method": method_name,
                        "n_candidates": n_cand,
                        "detector": det_name,
                        "tpr_at_1fpr": metrics.tpr_at_1fpr,
                        "asr": metrics.asr,
                        "auroc": metrics.auroc,
                        "mean_similarity": mean_sim,
                    })
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        df.to_csv(ablation_dir / "budget_sweep_results.csv", index=False)
        
        # Aggregate across detectors
        agg_df = df.groupby(["method", "n_candidates"]).agg({
            "tpr_at_1fpr": "mean",
            "asr": "mean",
            "auroc": "mean",
            "mean_similarity": "first",
        }).reset_index()
        agg_df.columns = ["method", "n_candidates", "mean_tpr", "mean_asr", "mean_auroc", "mean_similarity"]
        agg_df.to_csv(ablation_dir / "budget_sweep_aggregated.csv", index=False)
        
        # Create budget sweep plot
        create_budget_sweep_plot(
            agg_df,
            title="Candidate Budget Sweep",
            output_path=str(ablation_dir / "fig_budget_sweep.png"),
        )
        
        logger.info(f"\nBudget Sweep Results (aggregated):")
        logger.info(f"\n{agg_df.to_string()}")
        
        self.results["budget_sweep"] = df
        return df
    
    # =========================================================================
    # Ablation 8.3: Homoglyph Sweep
    # =========================================================================
    
    def run_homoglyph_sweep_ablation(
        self,
        dataset_name: str = "mage",
        n_human: int = 100,
        n_ai: int = 100,
        rates: List[float] = None,
    ) -> pd.DataFrame:
        """
        Run homoglyph sweep ablation (§8.3).
        
        Tests how detection changes with homoglyph substitution rate.
        
        Args:
            rates: Substitution rates to test (default: [0.001, 0.005, 0.01, 0.02])
            
        Returns:
            DataFrame with metrics for each rate
        """
        logger.info("=" * 60)
        logger.info("ABLATION 8.3: Homoglyph Sweep")
        logger.info("=" * 60)
        
        ablation_dir = self.output_dir / "homoglyph_sweep"
        ablation_dir.mkdir(exist_ok=True)
        
        if rates is None:
            rates = [0.001, 0.005, 0.01, 0.02]  # 0.1%, 0.5%, 1%, 2%
        
        # Load dataset
        if dataset_name not in self.datasets:
            self.load_dataset(dataset_name, n_human, n_ai)
        dataset = self.datasets[dataset_name]
        
        all_results = []
        
        for rate in rates:
            logger.info(f"\nRate: {rate*100:.1f}%...")
            
            # Create homoglyph attack with this rate
            from eval.methods import HomoglyphAttack
            attack = HomoglyphAttack(substitution_rate=rate, seed=self.seed)
            
            # Run attacks
            ai_texts = [s.text for s in dataset.ai_samples]
            results = attack.attack_batch(ai_texts)
            attacked_texts = [r.text for r in results]
            
            # Score with all detectors
            for det_name, detector in self.detectors.items():
                human_texts = [s.text for s in dataset.human_samples]
                human_scores = detector.get_scores(human_texts)
                ai_scores = detector.get_scores(attacked_texts)
                
                metrics = compute_detector_metrics(
                    human_scores=human_scores,
                    ai_scores=ai_scores,
                    detector=det_name,
                    method=f"homoglyph_p{rate}",
                    dataset=dataset_name,
                    n_bootstrap=self.n_bootstrap,
                    seed=self.seed,
                )
                
                all_results.append({
                    "rate": rate,
                    "rate_pct": f"{rate*100:.1f}%",
                    "detector": det_name,
                    "tpr_at_1fpr": metrics.tpr_at_1fpr,
                    "asr": metrics.asr,
                    "auroc": metrics.auroc,
                })
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        df.to_csv(ablation_dir / "homoglyph_sweep_results.csv", index=False)
        
        # Create plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for det_name in df["detector"].unique():
            det_df = df[df["detector"] == det_name]
            ax.plot(det_df["rate"] * 100, det_df["tpr_at_1fpr"], marker='o', label=det_name)
        
        ax.set_xlabel("Substitution Rate (%)")
        ax.set_ylabel("TPR@1%FPR")
        ax.set_title("Homoglyph Substitution Rate vs Detection")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(ablation_dir / "fig_homoglyph_sweep.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"\nHomoglyph Sweep Results:")
        logger.info(f"\n{df.to_string()}")
        
        self.results["homoglyph_sweep"] = df
        return df
    
    # =========================================================================
    # Ablation 9: Sanitize Defense
    # =========================================================================
    
    def run_sanitize_ablation(
        self,
        dataset_name: str = "mage",
        n_human: int = 100,
        n_ai: int = 100,
        methods_to_test: List[str] = None,
    ) -> Dict:
        """
        Run sanitization defense ablation (§9).
        
        Tests whether Unicode normalization (NFKC) and homoglyph removal
        can recover detection ability.
        
        Args:
            methods_to_test: Methods to test sanitization on (default: ["m5", "m2"])
            
        Returns:
            Dict with before/after scores for each method
        """
        logger.info("=" * 60)
        logger.info("ABLATION 9: Sanitize Defense")
        logger.info("=" * 60)
        
        ablation_dir = self.output_dir / "sanitize"
        ablation_dir.mkdir(exist_ok=True)
        
        if methods_to_test is None:
            methods_to_test = ["m5", "m2"]  # Homoglyph and StealthRL
        
        # Load dataset
        if dataset_name not in self.datasets:
            self.load_dataset(dataset_name, n_human, n_ai)
        dataset = self.datasets[dataset_name]
        
        all_results = []
        
        for method_name in methods_to_test:
            logger.info(f"\nMethod: {method_name}")
            
            # Get method
            if method_name == "m2" and self.stealthrl_checkpoint:
                method = get_method(method_name, checkpoint_json=self.stealthrl_checkpoint)
            else:
                method = get_method(method_name)
            
            # Run attacks
            ai_texts = [s.text for s in dataset.ai_samples]
            results = method.attack_batch(ai_texts)
            attacked_texts = [r.text for r in results]
            
            # Sanitize texts
            sanitized_texts = [sanitize(t) for t in attacked_texts]
            
            # Count how many texts changed
            n_changed = sum(1 for a, s in zip(attacked_texts, sanitized_texts) if a != s)
            logger.info(f"  Texts changed by sanitization: {n_changed}/{len(attacked_texts)}")
            
            # Score with all detectors before and after
            for det_name, detector in self.detectors.items():
                human_texts = [s.text for s in dataset.human_samples]
                human_scores = detector.get_scores(human_texts)
                
                # Before sanitization
                before_scores = detector.get_scores(attacked_texts)
                before_metrics = compute_detector_metrics(
                    human_scores=human_scores,
                    ai_scores=before_scores,
                    detector=det_name,
                    method=method_name,
                    dataset=dataset_name,
                    n_bootstrap=self.n_bootstrap,
                    seed=self.seed,
                )
                
                # After sanitization
                after_scores = detector.get_scores(sanitized_texts)
                after_metrics = compute_detector_metrics(
                    human_scores=human_scores,
                    ai_scores=after_scores,
                    detector=det_name,
                    method=f"{method_name}_sanitized",
                    dataset=dataset_name,
                    n_bootstrap=self.n_bootstrap,
                    seed=self.seed,
                )
                
                all_results.append({
                    "method": method_name,
                    "detector": det_name,
                    "tpr_before": before_metrics.tpr_at_1fpr,
                    "tpr_after": after_metrics.tpr_at_1fpr,
                    "tpr_delta": after_metrics.tpr_at_1fpr - before_metrics.tpr_at_1fpr,
                    "asr_before": before_metrics.asr,
                    "asr_after": after_metrics.asr,
                    "auroc_before": before_metrics.auroc,
                    "auroc_after": after_metrics.auroc,
                })
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        df.to_csv(ablation_dir / "sanitize_results.csv", index=False)
        
        # Create sanitize plot
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, len(methods_to_test), figsize=(6*len(methods_to_test), 5))
        if len(methods_to_test) == 1:
            axes = [axes]
        
        for idx, method_name in enumerate(methods_to_test):
            method_df = df[df["method"] == method_name]
            
            x = np.arange(len(method_df))
            width = 0.35
            
            axes[idx].bar(x - width/2, method_df["tpr_before"], width, label='Before Sanitize', color='#d62728', alpha=0.8)
            axes[idx].bar(x + width/2, method_df["tpr_after"], width, label='After Sanitize', color='#2ca02c', alpha=0.8)
            
            axes[idx].set_xlabel('Detector')
            axes[idx].set_ylabel('TPR@1%FPR')
            axes[idx].set_title(f'Sanitization Effect on {method_name}')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(method_df["detector"])
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(ablation_dir / "fig_sanitize.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create markdown report
        report_lines = [
            "# Sanitization Defense Evaluation\n",
            "\n## Summary\n",
            "| Method | Detector | TPR Before | TPR After | Δ TPR |",
            "|--------|----------|------------|-----------|-------|",
        ]
        for _, row in df.iterrows():
            report_lines.append(
                f"| {row['method']} | {row['detector']} | {row['tpr_before']:.3f} | "
                f"{row['tpr_after']:.3f} | {row['tpr_delta']:+.3f} |"
            )
        
        with open(ablation_dir / "table_sanitize.md", "w") as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"\nSanitize Results:")
        logger.info(f"\n{df.to_string()}")
        
        self.results["sanitize"] = df
        return df
    
    # =========================================================================
    # Ablation: Cross-Dataset Transfer (PadBen)
    # =========================================================================
    
    def run_padben_ablation(
        self,
        n_human: int = 100,
        n_ai: int = 100,
        methods: List[str] = None,
    ) -> pd.DataFrame:
        """
        Run cross-dataset transfer ablation on PadBen.
        
        Tests whether methods trained/tuned on MAGE generalize to PadBen.
        
        Args:
            methods: Methods to test (default: ["m0", "m1", "m2"])
            
        Returns:
            DataFrame with metrics on PadBen
        """
        logger.info("=" * 60)
        logger.info("ABLATION: Cross-Dataset Transfer (PadBen)")
        logger.info("=" * 60)
        
        ablation_dir = self.output_dir / "padben_transfer"
        ablation_dir.mkdir(exist_ok=True)
        
        if methods is None:
            methods = ["m0", "m1", "m2"]
        
        # Load PadBen dataset
        dataset = self.load_dataset("padben", n_human, n_ai)
        
        all_results = []
        
        for method_name in methods:
            logger.info(f"\nMethod: {method_name}")
            
            # Skip M2 if no checkpoint
            if method_name == "m2" and not self.stealthrl_checkpoint:
                logger.warning("  Skipping M2 - no checkpoint provided")
                continue
            
            # Get method
            if method_name == "m2":
                method = get_method(method_name, checkpoint_json=self.stealthrl_checkpoint)
            else:
                method = get_method(method_name)
            
            # Run attacks
            ai_texts = [s.text for s in dataset.ai_samples]
            results = method.attack_batch(ai_texts)
            attacked_texts = [r.text for r in results]
            
            # Score with all detectors
            for det_name, detector in self.detectors.items():
                human_texts = [s.text for s in dataset.human_samples]
                human_scores = detector.get_scores(human_texts)
                ai_scores = detector.get_scores(attacked_texts)
                
                metrics = compute_detector_metrics(
                    human_scores=human_scores,
                    ai_scores=ai_scores,
                    detector=det_name,
                    method=method_name,
                    dataset="padben",
                    n_bootstrap=self.n_bootstrap,
                    seed=self.seed,
                )
                
                all_results.append({
                    "dataset": "padben",
                    "method": method_name,
                    "detector": det_name,
                    "tpr_at_1fpr": metrics.tpr_at_1fpr,
                    "asr": metrics.asr,
                    "auroc": metrics.auroc,
                })
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        df.to_csv(ablation_dir / "padben_results.csv", index=False)
        
        # Create heatmap
        pivot_df = df.pivot(index="detector", columns="method", values="tpr_at_1fpr")
        create_heatmap(
            df,
            value_col="tpr_at_1fpr",
            title="PadBen Transfer: TPR@1%FPR",
            output_path=str(ablation_dir / "fig_padben_heatmap.png"),
        )
        
        logger.info(f"\nPadBen Transfer Results:")
        logger.info(f"\n{pivot_df.to_string()}")
        
        self.results["padben"] = df
        return df
    
    # =========================================================================
    # Run All Ablations
    # =========================================================================
    
    def run_all(
        self,
        ablations: List[str] = None,
        n_human: int = 100,
        n_ai: int = 100,
        **kwargs,
    ):
        """
        Run all specified ablations.
        
        Args:
            ablations: List of ablations to run (default: all)
                Options: "guidance", "budget", "homoglyph", "sanitize", "padben"
            n_human: Number of human samples
            n_ai: Number of AI samples
        """
        if ablations is None:
            ablations = ["guidance", "budget", "homoglyph", "sanitize", "padben"]
        
        logger.info("=" * 70)
        logger.info("StealthRL Ablation Studies")
        logger.info("=" * 70)
        logger.info(f"Ablations to run: {ablations}")
        logger.info(f"Samples: {n_human} human, {n_ai} AI")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 70)
        
        # Load detectors once
        self.load_detectors()
        
        start_time = time.time()
        
        if "guidance" in ablations:
            self.run_guidance_transfer_ablation(n_human=n_human, n_ai=n_ai)
        
        if "budget" in ablations:
            self.run_budget_sweep_ablation(n_human=n_human, n_ai=n_ai)
        
        if "homoglyph" in ablations:
            self.run_homoglyph_sweep_ablation(n_human=n_human, n_ai=n_ai)
        
        if "sanitize" in ablations:
            self.run_sanitize_ablation(n_human=n_human, n_ai=n_ai)
        
        if "padben" in ablations:
            self.run_padben_ablation(n_human=n_human, n_ai=n_ai)
        
        elapsed = time.time() - start_time
        
        # Save summary
        summary = {
            "ablations_run": ablations,
            "n_human": n_human,
            "n_ai": n_ai,
            "elapsed_seconds": elapsed,
            "output_dir": str(self.output_dir),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(self.output_dir / "ablation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info("=" * 70)
        logger.info("Ablation Studies Complete!")
        logger.info(f"Total time: {elapsed:.1f}s")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("=" * 70)
        
        return self.results


def main():
    parser = argparse.ArgumentParser(
        description="Run StealthRL ablation studies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Ablation selection
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=["guidance", "budget", "homoglyph", "sanitize", "padben"],
        choices=["guidance", "budget", "homoglyph", "sanitize", "padben", "all"],
        help="Ablations to run",
    )
    
    # Data arguments
    parser.add_argument("--n-human", type=int, default=100, help="Number of human samples")
    parser.add_argument("--n-ai", type=int, default=100, help="Number of AI samples")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory")
    
    # Model arguments
    parser.add_argument(
        "--stealthrl-checkpoint",
        type=str,
        default=None,
        help="Path to StealthRL checkpoint JSON (required for M2)",
    )
    
    # Detector arguments
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["roberta", "fast_detectgpt", "binoculars"],
        help="Detectors to evaluate",
    )
    
    # Output arguments
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/ablations",
        help="Output directory",
    )
    
    # Runtime arguments
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-bootstrap", type=int, default=500, help="Bootstrap samples")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Handle "all" ablation option
    if "all" in args.ablations:
        args.ablations = ["guidance", "budget", "homoglyph", "sanitize", "padben"]
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir) / f"ablations_{timestamp}"
    
    # Setup logging
    setup_logging(str(output_dir), args.log_level)
    
    # Create runner
    runner = AblationRunner(
        output_dir=str(output_dir),
        stealthrl_checkpoint=args.stealthrl_checkpoint,
        device=args.device,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
    )
    
    # Run ablations
    runner.run_all(
        ablations=args.ablations,
        n_human=args.n_human,
        n_ai=args.n_ai,
    )


if __name__ == "__main__":
    main()
