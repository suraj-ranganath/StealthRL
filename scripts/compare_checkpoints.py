#!/usr/bin/env python3
"""
Compare multiple StealthRL checkpoints against each other.

This script runs evaluation for multiple StealthRL checkpoints side-by-side,
allowing easy comparison of training runs, hyperparameters, or training progress.

Usage:
    # Compare two checkpoints
    python scripts/compare_checkpoints.py \
        --checkpoints checkpoints/run1.json checkpoints/run2.json \
        --n-human 100 --n-ai 100

    # Compare with custom names
    python scripts/compare_checkpoints.py \
        --checkpoints checkpoints/early.json checkpoints/late.json \
        --names "Early (5k steps)" "Late (20k steps)" \
        --n-human 100 --n-ai 100

    # Quick comparison (fewer samples)
    python scripts/compare_checkpoints.py \
        --checkpoints checkpoints/*.json \
        --quick
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.data import load_eval_dataset
from eval.detectors import load_detectors
from eval.methods import get_method
from eval.metrics import compute_detector_metrics, compute_quality_metrics, E5SimilarityScorer

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str, level: str = "INFO"):
    """Setup logging configuration."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(log_dir) / f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ]
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )


class CheckpointComparator:
    """Compare multiple StealthRL checkpoints with optimized parallel processing."""
    
    def __init__(
        self,
        output_dir: str,
        device: str = None,
        seed: int = 42,
        n_bootstrap: int = 500,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.seed = seed
        self.n_bootstrap = n_bootstrap
        
        self.detectors = {}
        self.results = {}
        self.human_scores_cache = {}  # Cache human scores per detector
        self.sim_scorer = None  # Reusable similarity scorer
    
    def load_detectors(self, detector_names: List[str] = None):
        """Load detector panel."""
        if detector_names is None:
            detector_names = ["roberta", "fast_detectgpt", "binoculars"]
        
        logger.info(f"Loading detectors: {detector_names}")
        self.detectors = load_detectors(detector_names, device=self.device)
    
    def compare_checkpoints(
        self,
        checkpoints: List[str],
        names: List[str] = None,
        n_human: int = 100,
        n_ai: int = 100,
        n_candidates: int = 1,
        dataset: str = "mage",
    ) -> pd.DataFrame:
        """
        Compare multiple StealthRL checkpoints.
        
        Args:
            checkpoints: List of checkpoint paths
            names: Optional display names for each checkpoint
            n_human: Number of human samples
            n_ai: Number of AI samples
            n_candidates: Candidates per sample
            dataset: Dataset to use
            
        Returns:
            DataFrame with comparison results
        """
        if names is None:
            names = [Path(cp).stem for cp in checkpoints]
        
        if len(names) != len(checkpoints):
            raise ValueError("Number of names must match number of checkpoints")
        
        logger.info("=" * 70)
        logger.info("StealthRL Checkpoint Comparison")
        logger.info("=" * 70)
        logger.info(f"Checkpoints: {len(checkpoints)}")
        for name, cp in zip(names, checkpoints):
            logger.info(f"  - {name}: {cp}")
        logger.info(f"Dataset: {dataset}")
        logger.info(f"Samples: {n_human} human, {n_ai} AI")
        logger.info(f"N candidates: {n_candidates}")
        logger.info("=" * 70)
        
        # Load dataset
        logger.info(f"\nLoading dataset: {dataset}")
        ds = load_eval_dataset(
            dataset,
            split="test",
            n_human=n_human,
            n_ai=n_ai,
            seed=self.seed,
        )
        
        human_texts = [s.text for s in ds.human_samples]
        ai_texts = [s.text for s in ds.ai_samples]
        
        # Load detectors if not already loaded
        if not self.detectors:
            self.load_detectors()
        if self.sim_scorer is None:
            logger.info("Initializing similarity scorer...")
            self.sim_scorer = E5SimilarityScorer()
        
        # Cache human scores for all detectors (compute once, reuse for all checkpoints)
        logger.info("\nCaching human detector scores...")
        for det_name, detector in tqdm(self.detectors.items(), desc="Scoring human samples"):
            if det_name not in self.human_scores_cache:
                self.human_scores_cache[det_name] = detector.get_scores(human_texts)
        
        # Run attacks in parallel for all checkpoints
        logger.info("\n" + "="*70)
        logger.info("Running parallel attacks for all checkpoints...")
        logger.info("="*70)
        
        attack_results = asyncio.run(self._parallel_attacks(
            checkpoints, names, ai_texts, n_candidates
        ))
        
        # Batch process all attacked texts with detectors
        logger.info("\nBatch scoring all attacked texts...")
        all_attacked_texts = []
        checkpoint_indices = []
        
        for idx, result in enumerate(attack_results):
            if result is not None:
                all_attacked_texts.extend(result["attacked_texts"])
                checkpoint_indices.extend([idx] * len(result["attacked_texts"]))
        
        # Batch score all texts at once per detector
        detector_scores_batch = {}
        for det_name, detector in tqdm(self.detectors.items(), desc="Batch scoring detectors"):
            if len(all_attacked_texts) > 0:
                detector_scores_batch[det_name] = detector.get_scores(all_attacked_texts)
        
        # Compute metrics for each checkpoint
        logger.info("\nComputing metrics for each checkpoint...")
        all_results = []
        
        for cp_idx, result in enumerate(tqdm(attack_results, desc="Processing results")):
            if result is None:
                continue
            
            cp_name = result["name"]
            cp_path = result["path"]
            attacked_texts = result["attacked_texts"]
            attack_time = result["attack_time"]
            
            # Compute quality metrics
            similarities = self.sim_scorer.compute_similarity(ai_texts, attacked_texts)
            mean_sim = np.mean(similarities)
            
            # Get scores for this checkpoint from batched results
            start_idx = sum(len(attack_results[i]["attacked_texts"]) for i in range(cp_idx) if attack_results[i] is not None)
            end_idx = start_idx + len(attacked_texts)
            
            # Evaluate with each detector using cached human scores and batched AI scores
            for det_name in self.detectors.keys():
                human_scores = self.human_scores_cache[det_name]
                ai_scores = detector_scores_batch[det_name][start_idx:end_idx]
                
                metrics = compute_detector_metrics(
                    human_scores=human_scores,
                    ai_scores=ai_scores,
                    detector=det_name,
                    method="m2",
                    dataset=dataset,
                    n_bootstrap=self.n_bootstrap,
                )
                
                all_results.append({
                    "checkpoint": cp_name,
                    "checkpoint_path": cp_path,
                    "detector": det_name,
                    "auroc": metrics.auroc,
                    "auroc_ci_low": metrics.auroc_ci_low,
                    "auroc_ci_high": metrics.auroc_ci_high,
                    "tpr_at_1fpr": metrics.tpr_at_1fpr,
                    "tpr_ci_low": metrics.tpr_at_1fpr_ci_low,
                    "tpr_ci_high": metrics.tpr_at_1fpr_ci_high,
                    "asr": metrics.asr,
                    "asr_ci_low": metrics.asr_ci_low,
                    "asr_ci_high": metrics.asr_ci_high,
                    "mean_similarity": mean_sim,
                    "n_samples": len(ai_texts),
                    "n_candidates": n_candidates,
                    "attack_time_s": attack_time,
                })
        
        # Create results DataFrame
        df = pd.DataFrame(all_results)
        
        # Save results
        self._save_results(df, checkpoints, names)
        
        return df
    
    async def _parallel_attacks(
        self,
        checkpoints: List[str],
        names: List[str],
        ai_texts: List[str],
        n_candidates: int,
    ) -> List[Dict]:
        """
        Run attacks in parallel for all checkpoints.
        
        Note: This parallelizes checkpoint loading, but within each checkpoint,
        the Tinker API already efficiently batches:
        - Multiple candidates per prompt (via num_samples parameter)
        - attack_batch() processes texts sequentially but each prompt gets N candidates in 1 API call
        
        For further optimization, consider modifying the StealthRL method to use
        sample_async() for concurrent processing of multiple prompts.
        """
        async def attack_checkpoint(cp_path: str, cp_name: str, idx: int):
            """Attack with a single checkpoint."""
            try:
                logger.info(f"[{idx+1}/{len(checkpoints)}] Loading {cp_name}...")
                method = get_method("m2", checkpoint_json=cp_path)
                
                logger.info(f"[{idx+1}/{len(checkpoints)}] Attacking with {cp_name}...")
                logger.info(f"[{idx+1}/{len(checkpoints)}] Tinker API will use num_samples={n_candidates} for efficient candidate generation")
                start_time = time.time()
                results = method.attack_batch(ai_texts, n_candidates=n_candidates)
                attack_time = time.time() - start_time
                
                attacked_texts = [r.text for r in results]
                logger.info(f"[{idx+1}/{len(checkpoints)}] {cp_name} completed in {attack_time:.1f}s ({len(ai_texts)/attack_time:.2f} samples/s)")
                
                return {
                    "name": cp_name,
                    "path": cp_path,
                    "attacked_texts": attacked_texts,
                    "attack_time": attack_time,
                }
            except Exception as e:
                logger.error(f"Failed to process checkpoint {cp_path}: {e}")
                return None
        
        # Run all checkpoint attacks concurrently
        tasks = [
            attack_checkpoint(cp_path, cp_name, idx)
            for idx, (cp_path, cp_name) in enumerate(zip(checkpoints, names))
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
            elif result is not None:
                valid_results.append(result)
        
        return valid_results
    
    def _save_results(
        self,
        df: pd.DataFrame,
        checkpoints: List[str],
        names: List[str],
    ):
        """Save comparison results and generate figures."""
        # Save CSV
        csv_path = self.output_dir / "checkpoint_comparison.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved results to {csv_path}")
        
        # Save summary JSON
        summary = {
            "checkpoints": [
                {"name": name, "path": path}
                for name, path in zip(names, checkpoints)
            ],
            "timestamp": datetime.now().isoformat(),
            "n_samples": int(df["n_samples"].iloc[0]) if len(df) > 0 else 0,
            "detectors": df["detector"].unique().tolist() if len(df) > 0 else [],
        }
        
        with open(self.output_dir / "comparison_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Generate comparison table
        self._generate_comparison_table(df)
        
        # Generate comparison plot
        self._generate_comparison_plot(df)
        
        # Print summary to console
        self._print_summary(df)
    
    def _generate_comparison_table(self, df: pd.DataFrame):
        """Generate Markdown comparison table."""
        if len(df) == 0:
            return
        
        # Pivot for easy comparison
        pivot = df.pivot_table(
            index="checkpoint",
            columns="detector",
            values=["auroc", "tpr_at_1fpr", "asr", "mean_similarity"],
            aggfunc="first"
        )
        
        # Create markdown table
        lines = ["# StealthRL Checkpoint Comparison\n"]
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # ASR comparison (higher is better)
        lines.append("## Attack Success Rate (ASR) - Higher is Better\n")
        lines.append("| Checkpoint | " + " | ".join(df["detector"].unique()) + " |")
        lines.append("|" + "---|" * (len(df["detector"].unique()) + 1))
        
        for cp in df["checkpoint"].unique():
            row = f"| {cp} |"
            for det in df["detector"].unique():
                val = df[(df["checkpoint"] == cp) & (df["detector"] == det)]["asr"].values
                if len(val) > 0:
                    row += f" {val[0]:.1%} |"
                else:
                    row += " - |"
            lines.append(row)
        
        lines.append("")
        
        # AUROC comparison (lower is better for attack)
        lines.append("## Detector AUROC - Lower is Better (for attacker)\n")
        lines.append("| Checkpoint | " + " | ".join(df["detector"].unique()) + " |")
        lines.append("|" + "---|" * (len(df["detector"].unique()) + 1))
        
        for cp in df["checkpoint"].unique():
            row = f"| {cp} |"
            for det in df["detector"].unique():
                val = df[(df["checkpoint"] == cp) & (df["detector"] == det)]["auroc"].values
                if len(val) > 0:
                    row += f" {val[0]:.3f} |"
                else:
                    row += " - |"
            lines.append(row)
        
        lines.append("")
        
        # Quality metrics
        lines.append("## Quality Metrics\n")
        lines.append("| Checkpoint | Mean Similarity |")
        lines.append("|---|---|")
        
        for cp in df["checkpoint"].unique():
            sim = df[df["checkpoint"] == cp]["mean_similarity"].iloc[0]
            lines.append(f"| {cp} | {sim:.3f} |")
        
        table_path = self.output_dir / "comparison_table.md"
        with open(table_path, "w") as f:
            f.write("\n".join(lines))
        
        logger.info(f"Saved comparison table to {table_path}")
    
    def _generate_comparison_plot(self, df: pd.DataFrame):
        """Generate comparison bar plot."""
        if len(df) == 0:
            return
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            checkpoints = df["checkpoint"].unique()
            detectors = df["detector"].unique()
            n_checkpoints = len(checkpoints)
            n_detectors = len(detectors)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # ASR comparison
            ax1 = axes[0]
            x = np.arange(n_checkpoints)
            width = 0.8 / n_detectors
            
            for i, det in enumerate(detectors):
                values = []
                for cp in checkpoints:
                    val = df[(df["checkpoint"] == cp) & (df["detector"] == det)]["asr"].values
                    values.append(val[0] * 100 if len(val) > 0 else 0)
                
                ax1.bar(x + i * width - width * (n_detectors - 1) / 2, values, width, label=det)
            
            ax1.set_xlabel("Checkpoint")
            ax1.set_ylabel("ASR (%)")
            ax1.set_title("Attack Success Rate by Checkpoint")
            ax1.set_xticks(x)
            ax1.set_xticklabels(checkpoints, rotation=45, ha="right")
            ax1.legend()
            ax1.set_ylim(0, 105)
            
            # AUROC comparison
            ax2 = axes[1]
            
            for i, det in enumerate(detectors):
                values = []
                for cp in checkpoints:
                    val = df[(df["checkpoint"] == cp) & (df["detector"] == det)]["auroc"].values
                    values.append(val[0] if len(val) > 0 else 0)
                
                ax2.bar(x + i * width - width * (n_detectors - 1) / 2, values, width, label=det)
            
            ax2.set_xlabel("Checkpoint")
            ax2.set_ylabel("AUROC")
            ax2.set_title("Detector AUROC by Checkpoint (Lower = Better Attack)")
            ax2.set_xticks(x)
            ax2.set_xticklabels(checkpoints, rotation=45, ha="right")
            ax2.legend()
            ax2.set_ylim(0, 1.05)
            ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
            
            plt.tight_layout()
            
            fig_path = self.output_dir / "fig_checkpoint_comparison.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved comparison plot to {fig_path}")
            
        except ImportError:
            logger.warning("matplotlib not available, skipping plot generation")
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary to console."""
        if len(df) == 0:
            logger.warning("No results to summarize")
            return
        
        logger.info("\n" + "=" * 70)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 70)
        
        # Find best checkpoint per detector
        for det in df["detector"].unique():
            det_df = df[df["detector"] == det]
            best_idx = det_df["asr"].idxmax()
            best_cp = det_df.loc[best_idx, "checkpoint"]
            best_asr = det_df.loc[best_idx, "asr"]
            best_auroc = det_df.loc[best_idx, "auroc"]
            
            logger.info(f"\n{det}:")
            logger.info(f"  Best checkpoint: {best_cp}")
            logger.info(f"  ASR: {best_asr:.1%}")
            logger.info(f"  AUROC: {best_auroc:.3f}")
        
        # Overall best (mean ASR across detectors)
        mean_asr = df.groupby("checkpoint")["asr"].mean()
        overall_best = mean_asr.idxmax()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"OVERALL BEST: {overall_best}")
        logger.info(f"Mean ASR across detectors: {mean_asr[overall_best]:.1%}")
        logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare StealthRL checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Paths to checkpoint JSON files to compare",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Display names for each checkpoint (defaults to filename)",
    )
    
    # Data arguments
    parser.add_argument("--dataset", type=str, default="mage", help="Dataset to use")
    parser.add_argument("--n-human", type=int, default=100, help="Number of human samples")
    parser.add_argument("--n-ai", type=int, default=100, help="Number of AI samples")
    parser.add_argument("--n-candidates", type=int, default=1, help="Candidates per sample")
    
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
        default="outputs/checkpoint_comparison",
        help="Output directory",
    )
    
    # Runtime arguments
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-bootstrap", type=int, default=500, help="Bootstrap samples")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    # Quick mode
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (20 samples, 1 detector)",
    )
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.n_human = 20
        args.n_ai = 20
        args.detectors = ["roberta"]
        args.n_bootstrap = 100
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir) / f"comparison_{timestamp}"
    
    # Setup logging
    setup_logging(str(output_dir), args.log_level)
    
    # Create comparator
    comparator = CheckpointComparator(
        output_dir=str(output_dir),
        device=args.device,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
    )
    
    # Load detectors
    comparator.load_detectors(args.detectors)
    
    # Run comparison
    df = comparator.compare_checkpoints(
        checkpoints=args.checkpoints,
        names=args.names,
        n_human=args.n_human,
        n_ai=args.n_ai,
        n_candidates=args.n_candidates,
        dataset=args.dataset,
    )
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
