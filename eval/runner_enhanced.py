"""
Enhanced evaluation runner with full SPEC.md compliance.

Features:
- Budget sweep (N=1,2,4,8)
- Guidance-transfer ablation
- Sanitization defense evaluation
- Structured run management with timestamps
- Qualitative examples generation
- Complete logging and metrics
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

from .data import load_eval_dataset, BaseEvalDataset, EvalSample
from .detectors import get_detector, load_detectors, BaseEvalDetector
from .methods import get_method, METHOD_REGISTRY, GUIDANCE_VARIANTS
from .metrics import (
    compute_detector_metrics,
    compute_quality_metrics,
    calibrate_thresholds,
    save_thresholds,
    E5SimilarityScorer,
    PerplexityScorer,
    DetectorMetrics,
)
from .plots import (
    generate_all_plots,
    generate_all_tables,
    create_budget_sweep_plot,
    create_heatmap,
    create_tradeoff_plot,
)
from .sanitize import (
    sanitize_batch,
    run_sanitize_evaluation,
    create_sanitize_report,
)

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path, level: str = "INFO") -> logging.Logger:
    """Setup comprehensive logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    detailed_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    simple_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    
    # File handler (detailed)
    fh = logging.FileHandler(log_dir / "eval.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(detailed_fmt)
    
    # Console handler (simple)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level.upper()))
    ch.setFormatter(simple_fmt)
    
    # Configure root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)
    
    return root


class RunManager:
    """
    Manages structured output for multiple evaluation runs.
    
    Directory structure:
    outputs/
      runs/
        {run_name}_{timestamp}/
          config.json
          logs/
            eval.log
          scores/
            scores.parquet
            scores_by_n/
              n1.parquet
              n2.parquet
              n4.parquet
              n8.parquet
          quality/
            quality.parquet
          metrics/
            metrics.json
            thresholds.json
            per_detector/
              roberta.json
              fast_detectgpt.json
          figures/
            heatmap_tpr.png
            heatmap_asr.png
            tradeoff.png
            budget_sweep.png
            sanitize.png
          tables/
            main_results.md
            transfer_matrix.md
            quality.md
            sanitize.md
          examples/
            qualitative_examples.md
            raw_outputs.jsonl
    """
    
    def __init__(
        self,
        base_dir: str = "outputs/runs",
        run_name: str = None,
    ):
        self.base_dir = Path(base_dir)
        # Format: YYYYMMDD_HHMMSS (24hr clock, e.g., 20260131_143052)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or "eval"
        
        # Create run directory: {run_name}_{date}_{time}
        self.run_dir = self.base_dir / f"{self.run_name}_{self.timestamp}"
        self._create_structure()
    
    def _create_structure(self):
        """Create directory structure."""
        dirs = [
            self.run_dir / "logs",
            self.run_dir / "scores" / "scores_by_n",
            self.run_dir / "quality",
            self.run_dir / "metrics" / "per_detector",
            self.run_dir / "figures",
            self.run_dir / "tables",
            self.run_dir / "examples",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"
    
    @property
    def scores_dir(self) -> Path:
        return self.run_dir / "scores"
    
    @property
    def quality_dir(self) -> Path:
        return self.run_dir / "quality"
    
    @property
    def metrics_dir(self) -> Path:
        return self.run_dir / "metrics"
    
    @property
    def figures_dir(self) -> Path:
        return self.run_dir / "figures"
    
    @property
    def tables_dir(self) -> Path:
        return self.run_dir / "tables"
    
    @property
    def examples_dir(self) -> Path:
        return self.run_dir / "examples"
    
    def save_config(self, config: Dict):
        """Save run configuration."""
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)


class EnhancedEvalRunner:
    """
    Enhanced evaluation runner with full SPEC.md compliance.
    """
    
    def __init__(
        self,
        run_manager: RunManager,
        device: str = None,
        seed: int = 42,
        n_bootstrap: int = 1000,
    ):
        self.rm = run_manager
        self.device = device
        self.seed = seed
        self.n_bootstrap = n_bootstrap
        
        # Storage
        self.datasets: Dict[str, BaseEvalDataset] = {}
        self.detectors: Dict[str, BaseEvalDetector] = {}
        
        # Results by budget
        self.scores_by_n: Dict[int, List[Dict]] = {}
        self.quality_by_n: Dict[int, List[Dict]] = {}
        self.metrics_by_n: Dict[int, List[Dict]] = {}
        
        self.thresholds: Dict[str, float] = {}
        
        # Quality scorers (lazy loaded)
        self._sim_scorer = None
        self._ppl_scorer = None
    
    @property
    def sim_scorer(self):
        if self._sim_scorer is None:
            self._sim_scorer = E5SimilarityScorer(device=self.device)
            self._sim_scorer.load()
        return self._sim_scorer
    
    @property
    def ppl_scorer(self):
        if self._ppl_scorer is None:
            self._ppl_scorer = PerplexityScorer(device=self.device)
            self._ppl_scorer.load()
        return self._ppl_scorer
    
    def load_datasets(
        self,
        dataset_names: List[str],
        n_human: int = 1000,
        n_ai: int = 1000,
        cache_dir: str = None,
    ):
        """Load evaluation datasets."""
        logger.info(f"Loading datasets: {dataset_names}")
        
        for name in dataset_names:
            try:
                dataset = load_eval_dataset(
                    name=name,
                    n_human=n_human,
                    n_ai=n_ai,
                    cache_dir=cache_dir,
                    seed=self.seed,
                )
                self.datasets[name] = dataset
                logger.info(f"  ✓ {name}: {len(dataset)} samples "
                           f"({len(dataset.human_samples)} human, {len(dataset.ai_samples)} AI)")
            except Exception as e:
                logger.error(f"  ✗ Failed to load {name}: {e}")
    
    def load_detectors(self, detector_names: List[str]):
        """Load detector panel."""
        logger.info(f"Loading detectors: {detector_names}")
        
        for name in detector_names:
            try:
                det = get_detector(name, device=self.device)
                det.load()
                self.detectors[name] = det
                logger.info(f"  ✓ {name} loaded")
            except Exception as e:
                logger.error(f"  ✗ Failed to load {name}: {e}")
    
    def calibrate_thresholds(self, target_fpr: float = 0.01):
        """Calibrate thresholds on human samples only."""
        logger.info(f"Calibrating thresholds at FPR={target_fpr}")
        
        human_scores: Dict[str, List[float]] = {d: [] for d in self.detectors}
        
        for dataset_name, dataset in self.datasets.items():
            human_texts = [s.text for s in dataset.human_samples]
            
            for det_name, detector in self.detectors.items():
                scores = detector.get_scores(human_texts)
                if isinstance(scores, float):
                    scores = [scores]
                human_scores[det_name].extend(scores)
        
        self.thresholds = calibrate_thresholds(human_scores, target_fpr=target_fpr)
        save_thresholds(self.thresholds, str(self.rm.metrics_dir / "thresholds.json"))
        
        for det_name, threshold in self.thresholds.items():
            logger.info(f"  {det_name}: threshold={threshold:.4f}")
    
    def run_budget_sweep(
        self,
        method_names: List[str],
        n_values: List[int] = [1, 2, 4, 8],
        stealthrl_checkpoint: str = None,
    ):
        """
        Run methods with different candidate budgets (§8.1).
        
        Args:
            method_names: Methods to evaluate
            n_values: Candidate budget values
            stealthrl_checkpoint: StealthRL checkpoint path
        """
        logger.info(f"Running budget sweep: N={n_values}")
        
        for n in n_values:
            logger.info(f"\n{'='*60}")
            logger.info(f"Budget N={n}")
            logger.info(f"{'='*60}")
            
            self.scores_by_n[n] = []
            self.quality_by_n[n] = []
            
            for method_name in method_names:
                # Skip methods that don't benefit from N>1
                if method_name in ("m0", "no_attack") and n > 1:
                    continue
                
                try:
                    # Get method
                    if method_name in ("m2", "stealthrl") and stealthrl_checkpoint:
                        # Use Tinker backend with checkpoint JSON
                        method = get_method(method_name, checkpoint_json=stealthrl_checkpoint)
                    elif method_name in ("m2_local", "stealthrl_local") and stealthrl_checkpoint:
                        # Use local PEFT backend
                        method = get_method(method_name, checkpoint_path=stealthrl_checkpoint)
                    else:
                        method = get_method(method_name)
                    
                    self._run_method_on_datasets(method, n_candidates=n)
                    
                except Exception as e:
                    logger.error(f"  ✗ Method {method_name} failed: {e}")
            
            # Compute metrics for this N
            self._compute_metrics_for_n(n)
            
            # Save intermediate results
            self._save_scores_for_n(n)
    
    def _run_method_on_datasets(
        self,
        method,
        n_candidates: int,
    ):
        """Run a method on all datasets."""
        logger.info(f"Running {method.name} (N={n_candidates})...")
        
        for dataset_name, dataset in self.datasets.items():
            ai_samples = dataset.ai_samples
            ai_texts = [s.text for s in ai_samples]
            ai_ids = [s.id for s in ai_samples]
            
            # Run attack
            results = method.attack_batch(ai_texts, n_candidates=n_candidates)
            attacked_texts = [r.text for r in results]
            
            # Score with all detectors
            for det_name, detector in self.detectors.items():
                scores = detector.get_scores(attacked_texts)
                if isinstance(scores, float):
                    scores = [scores]
                
                for i, (sample_id, orig, attacked, score) in enumerate(
                    zip(ai_ids, ai_texts, attacked_texts, scores)
                ):
                    self.scores_by_n[n_candidates].append({
                        "sample_id": sample_id,
                        "dataset": dataset_name,
                        "method": method.name,
                        "setting": f"N={n_candidates}",
                        "n_candidates": n_candidates,
                        "label": "ai",
                        "detector": det_name,
                        "score": score,
                        "text_in": orig[:500],  # Truncate for storage
                        "text_out": attacked[:500],
                    })
            
            # Compute quality metrics
            similarities = self.sim_scorer.compute_similarity(ai_texts, attacked_texts)
            perplexities = self.ppl_scorer.compute_perplexity(attacked_texts)
            
            for i, (sample_id, sim, ppl) in enumerate(
                zip(ai_ids, similarities, perplexities)
            ):
                edit_rate = self._compute_edit_rate(ai_texts[i], attacked_texts[i])
                
                self.quality_by_n[n_candidates].append({
                    "sample_id": sample_id,
                    "dataset": dataset_name,
                    "method": method.name,
                    "setting": f"N={n_candidates}",
                    "n_candidates": n_candidates,
                    "sim_e5": sim,
                    "ppl": ppl,
                    "edit_rate": edit_rate,
                })
            
            logger.info(f"  ✓ {method.name} on {dataset_name}: {len(attacked_texts)} samples")
    
    def _compute_edit_rate(self, original: str, modified: str) -> float:
        """Compute character-level edit rate."""
        if not original or not modified:
            return 1.0
        
        # Simple character-level Levenshtein approximation
        m, n = len(original), len(modified)
        if m == 0:
            return 1.0 if n > 0 else 0.0
        
        # Count changed characters (approximate)
        changes = sum(1 for a, b in zip(original, modified) if a != b)
        changes += abs(m - n)
        
        return min(changes / max(m, n), 1.0)
    
    def _compute_metrics_for_n(self, n: int):
        """Compute aggregated metrics for a specific N."""
        scores_df = pd.DataFrame(self.scores_by_n[n])
        
        if scores_df.empty:
            return
        
        self.metrics_by_n[n] = []
        
        for dataset_name in self.datasets.keys():
            ds_scores = scores_df[scores_df["dataset"] == dataset_name]
            
            for det_name in self.detectors.keys():
                det_scores = ds_scores[ds_scores["detector"] == det_name]
                
                for method_name in det_scores["method"].unique():
                    method_scores = det_scores[det_scores["method"] == method_name]
                    ai_scores = method_scores["score"].tolist()
                    
                    # Get human scores for AUROC
                    human_texts = [s.text for s in self.datasets[dataset_name].human_samples]
                    human_scores = self.detectors[det_name].get_scores(human_texts)
                    if isinstance(human_scores, float):
                        human_scores = [human_scores]
                    
                    # Compute metrics
                    metrics = compute_detector_metrics(
                        human_scores=human_scores,
                        ai_scores=ai_scores,
                        detector=det_name,
                        method=method_name,
                        dataset=dataset_name,
                        n_bootstrap=min(100, self.n_bootstrap),  # Faster for sweeps
                        seed=self.seed,
                    )
                    
                    metrics_dict = metrics.to_dict()
                    metrics_dict["n_candidates"] = n
                    self.metrics_by_n[n].append(metrics_dict)
    
    def _save_scores_for_n(self, n: int):
        """Save scores for a specific N value."""
        if self.scores_by_n.get(n):
            df = pd.DataFrame(self.scores_by_n[n])
            df.to_parquet(self.rm.scores_dir / "scores_by_n" / f"n{n}.parquet")
            df.to_csv(self.rm.scores_dir / "scores_by_n" / f"n{n}.csv", index=False)
            logger.info(f"  Saved scores for N={n}")
    
    def run_sanitize_evaluation(self, method_name: str = "m5"):
        """
        Run sanitization defense evaluation (§9).
        
        Tests detector robustness against sanitization.
        """
        logger.info("\n" + "="*60)
        logger.info("Running Sanitization Defense Evaluation")
        logger.info("="*60)
        
        # Get homoglyph attacked texts
        if not self.scores_by_n:
            logger.warning("No attack outputs to evaluate. Run budget sweep first.")
            return
        
        # Find homoglyph outputs
        all_scores = []
        for n, scores in self.scores_by_n.items():
            all_scores.extend(scores)
        
        scores_df = pd.DataFrame(all_scores)
        homoglyph_outputs = scores_df[
            scores_df["method"].str.contains("homoglyph", case=False)
        ]
        
        if homoglyph_outputs.empty:
            logger.warning("No homoglyph outputs found")
            return
        
        texts = homoglyph_outputs["text_out"].unique().tolist()
        
        # Run sanitization evaluation
        results = run_sanitize_evaluation(
            texts=texts,
            detectors=self.detectors,
        )
        
        # Save results
        with open(self.rm.metrics_dir / "sanitize_results.json", "w") as f:
            json.dump({
                "before_mean": {k: float(np.mean(v)) for k, v in results["before"].items()},
                "after_mean": {k: float(np.mean(v)) for k, v in results["after"].items()},
                "n_samples": len(texts),
            }, f, indent=2)
        
        # Create report
        report = create_sanitize_report(
            results,
            list(self.detectors.keys()),
            method_name=method_name,
        )
        with open(self.rm.tables_dir / "sanitize.md", "w") as f:
            f.write(report)
        
        # Also save CSV version
        csv_rows = []
        for det in self.detectors.keys():
            csv_rows.append({
                "detector": det,
                "before_sanitize": float(np.mean(results["before"].get(det, []))),
                "after_sanitize": float(np.mean(results["after"].get(det, []))),
            })
        pd.DataFrame(csv_rows).to_csv(self.rm.tables_dir / "sanitize.csv", index=False)
        
        logger.info(f"  ✓ Sanitization evaluation complete")
    
    def generate_qualitative_examples(self, n_examples: int = 10):
        """Generate qualitative examples (§2.1)."""
        logger.info("\nGenerating qualitative examples...")
        
        if not self.scores_by_n:
            return
        
        # Collect all outputs
        all_scores = []
        for n, scores in self.scores_by_n.items():
            all_scores.extend(scores)
        
        if not all_scores:
            return
        
        scores_df = pd.DataFrame(all_scores)
        
        # Select diverse examples
        examples = []
        
        for method in scores_df["method"].unique():
            method_df = scores_df[scores_df["method"] == method]
            
            # Get examples with different detector scores
            if len(method_df) > 0:
                # High score (detected as AI)
                high_score = method_df.nlargest(1, "score").iloc[0]
                # Low score (evaded detection)
                low_score = method_df.nsmallest(1, "score").iloc[0]
                
                examples.append({
                    "method": method,
                    "type": "detected",
                    "detector": high_score["detector"],
                    "score": high_score["score"],
                    "original": high_score["text_in"],
                    "paraphrased": high_score["text_out"],
                })
                
                examples.append({
                    "method": method,
                    "type": "evaded",
                    "detector": low_score["detector"],
                    "score": low_score["score"],
                    "original": low_score["text_in"],
                    "paraphrased": low_score["text_out"],
                })
        
        # Limit to n_examples
        examples = examples[:n_examples]
        
        # Write markdown
        lines = [
            "# Qualitative Examples",
            "",
            "Selected examples showing original AI text and paraphrased output.",
            "",
        ]
        
        for i, ex in enumerate(examples, 1):
            lines.extend([
                f"## Example {i}: {ex['method']} ({ex['type']})",
                f"",
                f"**Detector:** {ex['detector']} | **Score:** {ex['score']:.4f}",
                f"",
                f"### Original",
                f"```",
                ex['original'],
                f"```",
                f"",
                f"### Paraphrased",
                f"```",
                ex['paraphrased'],
                f"```",
                f"",
                "---",
                "",
            ])
        
        with open(self.rm.examples_dir / "qualitative_examples.md", "w") as f:
            f.write("\n".join(lines))
        
        # Save raw examples as JSONL
        with open(self.rm.examples_dir / "raw_outputs.jsonl", "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        
        logger.info(f"  ✓ Generated {len(examples)} qualitative examples")
    
    def generate_all_outputs(self):
        """Generate all figures, tables, and reports."""
        logger.info("\nGenerating outputs...")
        
        # Combine all metrics
        all_metrics = []
        for n, metrics in self.metrics_by_n.items():
            all_metrics.extend(metrics)
        
        if not all_metrics:
            logger.warning("No metrics to generate outputs from")
            return
        
        metrics_df = pd.DataFrame(all_metrics)
        
        # Save combined metrics
        with open(self.rm.metrics_dir / "metrics.json", "w") as f:
            json.dump({
                "metrics": all_metrics,
                "thresholds": self.thresholds,
                "n_bootstrap": self.n_bootstrap,
                "seed": self.seed,
            }, f, indent=2)
        
        # Per-detector metrics
        for det_name in self.detectors.keys():
            det_metrics = [m for m in all_metrics if m.get("detector") == det_name]
            with open(self.rm.metrics_dir / "per_detector" / f"{det_name}.json", "w") as f:
                json.dump(det_metrics, f, indent=2)
        
        # Generate figures
        try:
            self._generate_figures(metrics_df)
        except Exception as e:
            logger.error(f"Figure generation failed: {e}")
        
        # Generate tables
        try:
            self._generate_tables(metrics_df)
        except Exception as e:
            logger.error(f"Table generation failed: {e}")
        
        # Save combined scores
        all_scores = []
        for n, scores in self.scores_by_n.items():
            all_scores.extend(scores)
        if all_scores:
            scores_df = pd.DataFrame(all_scores)
            scores_df.to_parquet(self.rm.scores_dir / "scores.parquet")
            scores_df.to_csv(self.rm.scores_dir / "scores.csv", index=False)

        # Save combined quality
        all_quality = []
        for n, quality in self.quality_by_n.items():
            all_quality.extend(quality)
        if all_quality:
            quality_df = pd.DataFrame(all_quality)
            quality_df.to_parquet(self.rm.quality_dir / "quality.parquet")
            quality_df.to_csv(self.rm.quality_dir / "quality.csv", index=False)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Figure 1: TPR Heatmap
        if "tpr_at_1fpr" in metrics_df.columns:
            # Use N=4 by default for heatmap
            n4_metrics = metrics_df[metrics_df["n_candidates"] == 4]
            if not n4_metrics.empty:
                create_heatmap(
                    n4_metrics,
                    value_col="tpr_at_1fpr",
                    title="TPR@1%FPR by Detector and Method",
                    output_path=str(self.rm.figures_dir / "heatmap_tpr.png"),
                )
        
        # Figure 2: ASR Heatmap
        if "asr" in metrics_df.columns:
            n4_metrics = metrics_df[metrics_df["n_candidates"] == 4]
            if not n4_metrics.empty:
                create_heatmap(
                    n4_metrics,
                    value_col="asr",
                    title="ASR@1%FPR by Detector and Method",
                    output_path=str(self.rm.figures_dir / "heatmap_asr.png"),
                    cmap="RdYlGn",  # Higher ASR = better attack = green
                )
        
        # Figure 3: Budget sweep
        if len(self.metrics_by_n) > 1:
            self._create_budget_sweep_figure(metrics_df)
        
        # Figure 4: Tradeoff
        self._create_tradeoff_figure()
    
    def _create_budget_sweep_figure(self, metrics_df: pd.DataFrame):
        """Create budget sweep figure."""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Group by method and N
        for method in metrics_df["method"].unique():
            method_df = metrics_df[metrics_df["method"] == method]
            
            # TPR vs N
            tpr_by_n = method_df.groupby("n_candidates")["tpr_at_1fpr"].mean()
            ax1.plot(tpr_by_n.index, tpr_by_n.values, marker='o', label=method)
        
        ax1.set_xlabel("Number of Candidates (N)")
        ax1.set_ylabel("Mean TPR@1%FPR")
        ax1.set_title("Detection Rate vs Budget")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Similarity vs N (from quality metrics)
        all_quality = []
        for n, quality in self.quality_by_n.items():
            all_quality.extend(quality)
        
        if all_quality:
            quality_df = pd.DataFrame(all_quality)
            for method in quality_df["method"].unique():
                method_df = quality_df[quality_df["method"] == method]
                sim_by_n = method_df.groupby("n_candidates")["sim_e5"].mean()
                ax2.plot(sim_by_n.index, sim_by_n.values, marker='s', label=method)
        
        ax2.set_xlabel("Number of Candidates (N)")
        ax2.set_ylabel("Mean E5 Similarity")
        ax2.set_title("Quality vs Budget")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.rm.figures_dir / "budget_sweep.png", dpi=300)
        plt.close()
    
    def _create_tradeoff_figure(self):
        """Create tradeoff figure."""
        import matplotlib.pyplot as plt
        
        # Combine quality and metrics
        all_quality = []
        for n, quality in self.quality_by_n.items():
            all_quality.extend(quality)
        
        all_metrics = []
        for n, metrics in self.metrics_by_n.items():
            all_metrics.extend(metrics)
        
        if not all_quality or not all_metrics:
            return
        
        quality_df = pd.DataFrame(all_quality)
        metrics_df = pd.DataFrame(all_metrics)
        
        # Aggregate by method
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for method in metrics_df["method"].unique():
            # Mean TPR across detectors
            method_metrics = metrics_df[metrics_df["method"] == method]
            mean_tpr = method_metrics["tpr_at_1fpr"].mean()
            
            # Mean similarity
            method_quality = quality_df[quality_df["method"] == method]
            if len(method_quality) > 0:
                mean_sim = method_quality["sim_e5"].mean()
            else:
                mean_sim = 1.0
            
            ax.scatter(mean_sim, mean_tpr, s=150, label=method, edgecolors='white', linewidth=2)
            ax.annotate(method, (mean_sim, mean_tpr), xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel("Semantic Similarity (E5)")
        ax.set_ylabel("Mean TPR@1%FPR (↓ better)")
        ax.set_title("Evasion-Quality Tradeoff")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.rm.figures_dir / "tradeoff.png", dpi=300)
        plt.close()
    
    def _generate_tables(self, metrics_df: pd.DataFrame):
        """Generate all tables."""
        # Main results table
        self._create_main_table(metrics_df)
        
        # Transfer matrix
        self._create_transfer_table(metrics_df)
        
        # Quality table
        self._create_quality_table()
    
    def _create_main_table(self, metrics_df: pd.DataFrame):
        """Create main results table."""
        # Use N=4 results
        n4 = metrics_df[metrics_df["n_candidates"] == 4]
        
        if n4.empty:
            n4 = metrics_df
        
        # Pivot by method and detector
        pivot = n4.pivot_table(
            index="method",
            columns="detector",
            values=["auroc", "tpr_at_1fpr", "asr"],
            aggfunc="mean",
        )
        
        lines = [
            "# Main Results (MAGE)",
            "",
            "| Method |",
        ]
        
        # Build header
        detectors = list(self.detectors.keys())
        header = "| Method |"
        for det in detectors:
            header += f" {det} AUC | {det} TPR | {det} ASR |"
        header += " Mean TPR | Mean ASR |"
        
        lines = [
            "# Main Results",
            "",
            header,
            "|" + "---|" * (len(detectors) * 3 + 3),
        ]
        
        for method in n4["method"].unique():
            row = f"| {method} |"
            
            tprs = []
            asrs = []
            
            for det in detectors:
                det_method = n4[(n4["method"] == method) & (n4["detector"] == det)]
                if len(det_method) > 0:
                    auc = det_method["auroc"].values[0]
                    tpr = det_method["tpr_at_1fpr"].values[0]
                    asr = det_method["asr"].values[0]
                    row += f" {auc:.3f} | {tpr:.3f} | {asr:.3f} |"
                    tprs.append(tpr)
                    asrs.append(asr)
                else:
                    row += " - | - | - |"
            
            mean_tpr = np.mean(tprs) if tprs else 0
            mean_asr = np.mean(asrs) if asrs else 0
            row += f" {mean_tpr:.3f} | {mean_asr:.3f} |"
            
            lines.append(row)
        
        with open(self.rm.tables_dir / "main_results.md", "w") as f:
            f.write("\n".join(lines))
        
        # Also save CSV version
        csv_rows = []
        for method in n4["method"].unique():
            row_data = {"method": method}
            for det in detectors:
                det_method = n4[(n4["method"] == method) & (n4["detector"] == det)]
                if len(det_method) > 0:
                    row_data[f"{det}_auc"] = det_method["auroc"].values[0]
                    row_data[f"{det}_tpr"] = det_method["tpr_at_1fpr"].values[0]
                    row_data[f"{det}_asr"] = det_method["asr"].values[0]
            csv_rows.append(row_data)
        pd.DataFrame(csv_rows).to_csv(self.rm.tables_dir / "main_results.csv", index=False)
    
    def _create_transfer_table(self, metrics_df: pd.DataFrame):
        """Create transfer matrix table."""
        n4 = metrics_df[metrics_df["n_candidates"] == 4]
        if n4.empty:
            n4 = metrics_df
        
        # Pivot: rows=detectors, cols=methods, values=TPR
        lines = ["# Transfer Matrix (TPR@1%FPR)", ""]
        
        methods = list(n4["method"].unique())
        detectors = list(self.detectors.keys())
        
        header = "| Detector |" + " | ".join(methods) + " |"
        sep = "|" + "---|" * (len(methods) + 1)
        
        lines.extend([header, sep])
        
        for det in detectors:
            row = f"| {det} |"
            for method in methods:
                val = n4[(n4["method"] == method) & (n4["detector"] == det)]
                if len(val) > 0:
                    tpr = val["tpr_at_1fpr"].values[0]
                    row += f" {tpr:.3f} |"
                else:
                    row += " - |"
            lines.append(row)
        
        with open(self.rm.tables_dir / "transfer_matrix.md", "w") as f:
            f.write("\n".join(lines))
        
        # Also save CSV version
        csv_rows = []
        for det in detectors:
            row_data = {"detector": det}
            for method in methods:
                val = n4[(n4["method"] == method) & (n4["detector"] == det)]
                if len(val) > 0:
                    row_data[method] = val["tpr_at_1fpr"].values[0]
            csv_rows.append(row_data)
        pd.DataFrame(csv_rows).to_csv(self.rm.tables_dir / "transfer_matrix.csv", index=False)
    
    def _create_quality_table(self):
        """Create quality metrics table."""
        all_quality = []
        for n, quality in self.quality_by_n.items():
            all_quality.extend(quality)
        
        if not all_quality:
            return
        
        quality_df = pd.DataFrame(all_quality)
        
        lines = [
            "# Quality Metrics",
            "",
            "| Method | N | Similarity (E5) | Perplexity | Edit Rate |",
            "|--------|---|-----------------|------------|-----------|",
        ]
        
        for method in quality_df["method"].unique():
            for n in sorted(quality_df["n_candidates"].unique()):
                subset = quality_df[(quality_df["method"] == method) & (quality_df["n_candidates"] == n)]
                if len(subset) > 0:
                    sim = subset["sim_e5"].mean()
                    ppl = subset["ppl"].mean()
                    edit = subset["edit_rate"].mean()
                    lines.append(f"| {method} | {n} | {sim:.4f} | {ppl:.2f} | {edit:.4f} |")
        
        with open(self.rm.tables_dir / "quality.md", "w") as f:
            f.write("\n".join(lines))
        
        # Also save CSV version
        quality_df.groupby(["method", "n_candidates"]).agg({
            "sim_e5": "mean",
            "ppl": "mean",
            "edit_rate": "mean"
        }).reset_index().to_csv(self.rm.tables_dir / "quality.csv", index=False)
    
    def run_full_evaluation(
        self,
        datasets: List[str],
        methods: List[str],
        detectors: List[str],
        n_values: List[int] = [1, 2, 4, 8],
        n_human: int = 1000,
        n_ai: int = 1000,
        stealthrl_checkpoint: str = None,
        run_sanitize: bool = True,
        cache_dir: str = None,
    ):
        """Run complete evaluation pipeline."""
        logger.info("="*60)
        logger.info("StealthRL Full Evaluation Pipeline")
        logger.info(f"Run: {self.rm.run_dir}")
        logger.info("="*60)
        
        # Save config
        config = {
            "datasets": datasets,
            "methods": methods,
            "detectors": detectors,
            "n_values": n_values,
            "n_human": n_human,
            "n_ai": n_ai,
            "stealthrl_checkpoint": stealthrl_checkpoint,
            "seed": self.seed,
            "n_bootstrap": self.n_bootstrap,
            "timestamp": self.rm.timestamp,
        }
        self.rm.save_config(config)
        
        # Step 1: Load datasets
        self.load_datasets(datasets, n_human=n_human, n_ai=n_ai, cache_dir=cache_dir)
        
        if not self.datasets:
            logger.error("No datasets loaded. Exiting.")
            return
        
        # Step 2: Load detectors
        self.load_detectors(detectors)
        
        if not self.detectors:
            logger.error("No detectors loaded. Exiting.")
            return
        
        # Step 3: Calibrate thresholds
        self.calibrate_thresholds()
        
        # Step 4: Run budget sweep
        self.run_budget_sweep(methods, n_values=n_values, stealthrl_checkpoint=stealthrl_checkpoint)
        
        # Step 5: Sanitization evaluation
        if run_sanitize and "m5" in methods or "homoglyph" in methods:
            self.run_sanitize_evaluation()
        
        # Step 6: Generate outputs
        self.generate_all_outputs()
        
        # Step 7: Qualitative examples
        self.generate_qualitative_examples()
        
        logger.info("\n" + "="*60)
        logger.info("Evaluation Complete!")
        logger.info(f"Results: {self.rm.run_dir}")
        logger.info("="*60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="StealthRL Enhanced Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Run management
    parser.add_argument("--run-name", type=str, default="eval", help="Run name")
    parser.add_argument("--out-dir", type=str, default="outputs/runs", help="Output base directory")
    
    # Datasets
    parser.add_argument("--datasets", nargs="+", default=["mage"], help="Datasets")
    parser.add_argument("--n-human", type=int, default=1000, help="Human samples")
    parser.add_argument("--n-ai", type=int, default=1000, help="AI samples")
    
    # Methods
    parser.add_argument("--methods", nargs="+", default=["m0", "m1"], help="Methods")
    parser.add_argument("--stealthrl-checkpoint", type=str, help="StealthRL checkpoint")
    
    # Budget sweep
    parser.add_argument("--n-values", nargs="+", type=int, default=[1, 2, 4, 8], help="Candidate budgets")
    
    # Detectors
    parser.add_argument("--detectors", nargs="+", default=["roberta", "fast_detectgpt"], help="Detectors")
    
    # Options
    parser.add_argument("--no-sanitize", action="store_true", help="Skip sanitization evaluation")
    parser.add_argument("--cache-dir", type=str, help="Cache directory")
    parser.add_argument("--device", type=str, help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap samples")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    
    # Quick mode
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.n_human = 50
        args.n_ai = 50
        args.n_values = [1, 2]
        args.n_bootstrap = 100
        args.methods = ["m0", "m1"]
        args.detectors = ["roberta"]
    
    # Setup run
    rm = RunManager(base_dir=args.out_dir, run_name=args.run_name)
    setup_logging(rm.logs_dir, args.log_level)
    
    # Run evaluation
    runner = EnhancedEvalRunner(
        run_manager=rm,
        device=args.device,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
    )
    
    runner.run_full_evaluation(
        datasets=args.datasets,
        methods=args.methods,
        detectors=args.detectors,
        n_values=args.n_values,
        n_human=args.n_human,
        n_ai=args.n_ai,
        stealthrl_checkpoint=args.stealthrl_checkpoint,
        run_sanitize=not args.no_sanitize,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
