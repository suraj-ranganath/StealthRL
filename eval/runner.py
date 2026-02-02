"""
Main evaluation runner for StealthRL.

Orchestrates the full evaluation pipeline:
1. Load datasets (MAGE, RAID)
2. Run attack methods to generate outputs
3. Score outputs with detector panel
4. Compute metrics with bootstrap CIs
5. Generate figures and tables
6. Save all artifacts

Usage:
    python -m eval.run --datasets mage --methods m0 m1 m2 --detectors roberta fast_detectgpt
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

from .data import load_eval_dataset, BaseEvalDataset, EvalSample
from .detectors import get_detector, load_detectors, BaseEvalDetector
from .methods import get_method, NoAttack, METHOD_REGISTRY
from .metrics import (
    compute_detector_metrics,
    compute_quality_metrics,
    calibrate_thresholds,
    save_thresholds,
    E5SimilarityScorer,
    PerplexityScorer,
)
from .plots import (
    generate_all_plots,
    generate_all_tables,
    generate_qualitative_examples,
)

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str = None, level: str = "INFO"):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_dir:
        log_path = Path(log_dir) / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )


class EvalRunner:
    """
    Main evaluation runner.
    
    Coordinates datasets, methods, detectors, and output generation.
    """
    
    def __init__(
        self,
        output_dir: str = "artifacts",
        device: str = None,
        seed: int = 42,
        n_bootstrap: int = 1000,
    ):
        """
        Initialize evaluation runner.
        
        Args:
            output_dir: Directory for all outputs
            device: Device for model inference
            seed: Random seed for reproducibility
            n_bootstrap: Number of bootstrap samples for CIs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device
        self.seed = seed
        self.n_bootstrap = n_bootstrap
        
        # Will be populated during run
        self.datasets: Dict[str, BaseEvalDataset] = {}
        self.detectors: Dict[str, BaseEvalDetector] = {}
        self.methods: Dict[str, Any] = {}
        
        # Results storage
        self.all_scores: List[Dict] = []
        self.all_quality: List[Dict] = []
        self.all_metrics: List[Dict] = []
        self.thresholds: Dict[str, float] = {}
    
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
                logger.info(f"Loaded {name}: {len(dataset)} samples "
                           f"({len(dataset.human_samples)} human, {len(dataset.ai_samples)} AI)")
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
    
    def load_detectors(self, detector_names: List[str], binoculars_full: bool = False):
        """Load detector panel."""
        logger.info(f"Loading detectors: {detector_names}")
        
        self.detectors = load_detectors(
            detector_names, 
            device=self.device,
            binoculars_full=binoculars_full,
        )
        
        # Pre-load all detectors
        for name, detector in self.detectors.items():
            try:
                detector.load()
            except Exception as e:
                logger.error(f"Failed to load detector {name}: {e}")
    
    def load_methods(
        self,
        method_names: List[str],
        stealthrl_checkpoint: str = None,
        **method_kwargs,
    ):
        """Load attack methods."""
        methods_start = time.time()
        logger.info(f"[LOAD] Loading methods: {method_names}")
        
        for name in method_names:
            try:
                method_start = time.time()
                logger.info(f"[LOAD] Loading method {name}...")
                
                if name in ("m2", "stealthrl") and stealthrl_checkpoint:
                    method = get_method(name, checkpoint_json=stealthrl_checkpoint, **method_kwargs)
                else:
                    method = get_method(name, **method_kwargs)
                
                self.methods[name] = method
                method_elapsed = time.time() - method_start
                logger.info(f"[LOAD] ✓ Loaded method {name} in {method_elapsed:.2f}s")
            except Exception as e:
                logger.error(f"[LOAD] ✗ Failed to load method {name}: {e}")
        
        methods_elapsed = time.time() - methods_start
        logger.info(f"[LOAD] All methods loaded in {methods_elapsed:.2f}s")
    
    def calibrate_detector_thresholds(self, target_fpr: float = 0.01):
        """
        Calibrate detection thresholds on human samples.
        
        CRITICAL: Must use human samples only!
        """
        calibrate_start = time.time()
        logger.info(f"[CALIBRATE] Starting threshold calibration at FPR={target_fpr}")
        
        # Collect human scores from all datasets
        human_scores: Dict[str, List[float]] = {d: [] for d in self.detectors}
        
        for dataset_name, dataset in self.datasets.items():
            human_texts = [s.text for s in dataset.human_samples]
            logger.info(f"[CALIBRATE] Processing {len(human_texts)} human texts from {dataset_name}")
            
            for det_name, detector in self.detectors.items():
                det_start = time.time()
                scores = detector.get_scores(human_texts)
                det_elapsed = time.time() - det_start
                if isinstance(scores, float):
                    scores = [scores]
                human_scores[det_name].extend(scores)
                logger.info(f"[CALIBRATE] {det_name} scored {len(scores)} texts in {det_elapsed:.2f}s")
        
        # Compute thresholds
        self.thresholds = calibrate_thresholds(human_scores, target_fpr=target_fpr)
        
        # Save thresholds
        save_thresholds(self.thresholds, str(self.output_dir / "thresholds.json"))
        calibrate_elapsed = time.time() - calibrate_start
        logger.info(f"[CALIBRATE] Threshold calibration completed in {calibrate_elapsed:.2f}s")
    
    def run_attacks(
        self,
        n_candidates: int = 4,
        save_outputs: bool = True,
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Run all attack methods on AI samples.
        
        Returns:
            Dict mapping dataset -> method -> list of attacked texts
        """
        attacks_start = time.time()
        logger.info(f"[ATTACKS] Starting attacks with n_candidates={n_candidates}")
        
        outputs: Dict[str, Dict[str, List[str]]] = {}
        
        for dataset_name, dataset in self.datasets.items():
            outputs[dataset_name] = {}
            ai_texts = [s.text for s in dataset.ai_samples]
            ai_ids = [s.id for s in dataset.ai_samples]
            logger.info(f"[ATTACKS] Dataset {dataset_name}: {len(ai_texts)} AI texts to process")
            
            for method_name, method in self.methods.items():
                method_start = time.time()
                logger.info(f"[ATTACKS] Running {method_name} on {dataset_name} ({len(ai_texts)} texts)...")
                
                try:
                    # Run attack on all AI samples
                    results = method.attack_batch(ai_texts, n_candidates=n_candidates)
                    attacked_texts = [r.text for r in results]
                    
                    outputs[dataset_name][method_name] = attacked_texts
                    
                    # Store detailed outputs
                    for i, (sample_id, orig, attacked, result) in enumerate(
                        zip(ai_ids, ai_texts, attacked_texts, results)
                    ):
                        self.all_scores.append({
                            "sample_id": sample_id,
                            "dataset": dataset_name,
                            "method": method_name,
                            "label": "ai",
                            "setting": f"N={n_candidates}",
                            "text_in": orig,
                            "text_out": attacked,
                            "metadata": result.metadata,
                        })
                    
                    method_elapsed = time.time() - method_start
                    rate = len(attacked_texts) / method_elapsed if method_elapsed > 0 else 0
                    logger.info(f"[ATTACKS] Completed {method_name}: {len(attacked_texts)} outputs in {method_elapsed:.2f}s ({rate:.2f} texts/s)")
                    
                except Exception as e:
                    logger.error(f"[ATTACKS] Failed {method_name} on {dataset_name}: {e}")
                    outputs[dataset_name][method_name] = ai_texts  # Fallback to original
        
        # Save raw outputs
        if save_outputs:
            outputs_path = self.output_dir / "raw_outputs.json"
            with open(outputs_path, "w") as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved raw outputs to {outputs_path}")
        
        return outputs
    
    def score_outputs(
        self,
        outputs: Dict[str, Dict[str, List[str]]],
    ):
        """
        Score all outputs with detector panel.
        """
        score_start = time.time()
        logger.info("[SCORE] Scoring outputs with detectors...")
        
        for dataset_name, dataset in self.datasets.items():
            # Get human texts for this dataset
            human_texts = [s.text for s in dataset.human_samples]
            human_ids = [s.id for s in dataset.human_samples]
            
            for det_name, detector in self.detectors.items():
                # Score human samples
                det_start = time.time()
                human_scores = detector.get_scores(human_texts)
                det_elapsed = time.time() - det_start
                logger.info(f"[SCORE] {det_name} scored {len(human_texts)} human texts in {det_elapsed:.2f}s")
                
                if isinstance(human_scores, float):
                    human_scores = [human_scores]
                
                for sample_id, text, score in zip(human_ids, human_texts, human_scores):
                    self.all_scores.append({
                        "sample_id": sample_id,
                        "dataset": dataset_name,
                        "method": "human",
                        "label": "human",
                        "detector_name": det_name,
                        "detector_score": score,
                    })
                
                # Score attacked outputs for each method
                for method_name, attacked_texts in outputs.get(dataset_name, {}).items():
                    ai_ids = [s.id for s in dataset.ai_samples]
                    
                    det_start = time.time()
                    scores = detector.get_scores(attacked_texts)
                    det_elapsed = time.time() - det_start
                    logger.info(f"[SCORE] {det_name} scored {len(attacked_texts)} {method_name} outputs in {det_elapsed:.2f}s")
                    
                    if isinstance(scores, float):
                        scores = [scores]
                    
                    for sample_id, text, score in zip(ai_ids, attacked_texts, scores):
                        # Update existing record or add detector score
                        for record in self.all_scores:
                            if (record["sample_id"] == sample_id and 
                                record["method"] == method_name and
                                record.get("detector_name") is None):
                                record["detector_name"] = det_name
                                record["detector_score"] = score
                                break
                        else:
                            self.all_scores.append({
                                "sample_id": sample_id,
                                "dataset": dataset_name,
                                "method": method_name,
                                "label": "ai",
                                "detector_name": det_name,
                                "detector_score": score,
                            })
        
        score_elapsed = time.time() - score_start
        logger.info(f"[SCORE] Scoring complete: {len(self.all_scores)} records in {score_elapsed:.2f}s")
    
    def compute_all_metrics(self):
        """Compute all evaluation metrics."""
        logger.info("Computing metrics...")
        
        scores_df = pd.DataFrame(self.all_scores)
        
        # Group by dataset, method, detector
        for dataset_name in self.datasets.keys():
            ds_scores = scores_df[scores_df["dataset"] == dataset_name]
            
            # Get human scores for this dataset
            human_scores_all = ds_scores[ds_scores["label"] == "human"]
            
            for det_name in self.detectors.keys():
                human_det_scores = human_scores_all[
                    human_scores_all["detector_name"] == det_name
                ]["detector_score"].tolist()
                
                # Skip if no human scores
                if not human_det_scores:
                    continue
                
                for method_name in self.methods.keys():
                    ai_scores = ds_scores[
                        (ds_scores["method"] == method_name) &
                        (ds_scores["detector_name"] == det_name) &
                        (ds_scores["label"] == "ai")
                    ]["detector_score"].tolist()
                    
                    if not ai_scores:
                        continue
                    
                    # Compute metrics
                    metrics = compute_detector_metrics(
                        human_scores=human_det_scores,
                        ai_scores=ai_scores,
                        detector=det_name,
                        method=method_name,
                        dataset=dataset_name,
                        n_bootstrap=self.n_bootstrap,
                        seed=self.seed,
                    )
                    
                    self.all_metrics.append(metrics.to_dict())
        
        logger.info(f"Computed metrics for {len(self.all_metrics)} combinations")
    
    def compute_quality_metrics(
        self,
        outputs: Dict[str, Dict[str, List[str]]],
    ):
        """Compute text quality metrics for all outputs."""
        logger.info("Computing quality metrics...")
        
        # Initialize scorers
        sim_scorer = E5SimilarityScorer(device=self.device)
        ppl_scorer = PerplexityScorer(device=self.device)
        
        for dataset_name, dataset in self.datasets.items():
            original_texts = [s.text for s in dataset.ai_samples]
            sample_ids = [s.id for s in dataset.ai_samples]
            
            for method_name, attacked_texts in outputs.get(dataset_name, {}).items():
                quality = compute_quality_metrics(
                    original_texts=original_texts,
                    paraphrased_texts=attacked_texts,
                    sample_ids=sample_ids,
                    method=method_name,
                    setting="default",
                    similarity_scorer=sim_scorer,
                    perplexity_scorer=ppl_scorer,
                )
                
                for q in quality:
                    self.all_quality.append(q.to_dict())
        
        logger.info(f"Computed quality metrics for {len(self.all_quality)} samples")
    
    def save_all_artifacts(self):
        """Save all artifacts to disk."""
        logger.info("Saving artifacts...")
        
        # Create output directories
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        
        # Save scores
        scores_df = pd.DataFrame(self.all_scores)
        scores_df.to_parquet(self.output_dir / "scores.parquet")
        scores_df.to_csv(self.output_dir / "scores.csv", index=False)
        
        # Save quality metrics
        quality_df = pd.DataFrame(self.all_quality)
        quality_df.to_parquet(self.output_dir / "quality.parquet")
        quality_df.to_csv(self.output_dir / "quality.csv", index=False)

        # Save metrics
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump({
                "metrics": self.all_metrics,
                "thresholds": self.thresholds,
                "config": {
                    "seed": self.seed,
                    "n_bootstrap": self.n_bootstrap,
                    "datasets": list(self.datasets.keys()),
                    "methods": list(self.methods.keys()),
                    "detectors": list(self.detectors.keys()),
                },
            }, f, indent=2)
        
        # Generate figures
        if self.all_metrics:
            metrics_df = pd.DataFrame(self.all_metrics)
            generate_all_plots(
                detector_metrics=metrics_df,
                quality_metrics=quality_df,
                scores_data=scores_df,
                output_dir=str(self.output_dir / "figures"),
            )
        
        # Generate tables
        if self.all_metrics:
            generate_all_tables(
                detector_metrics=self.all_metrics,
                quality_metrics=self.all_quality,
                output_dir=str(self.output_dir / "tables"),
            )
        
        logger.info(f"All artifacts saved to {self.output_dir}")
    
    def run(
        self,
        datasets: List[str],
        methods: List[str],
        detectors: List[str],
        n_candidates: int = 4,
        n_human: int = 1000,
        n_ai: int = 1000,
        stealthrl_checkpoint: str = None,
        cache_dir: str = None,
        binoculars_full: bool = False,
        setting_suffix: str = None,
    ):
        """
        Run complete evaluation pipeline.
        
        Args:
            datasets: List of dataset names
            methods: List of method names
            detectors: List of detector names
            n_candidates: Number of candidates per method
            n_human: Number of human samples per dataset
            n_ai: Number of AI samples per dataset
            stealthrl_checkpoint: Path to StealthRL checkpoint
            cache_dir: Cache directory for downloads
            binoculars_full: Use Falcon-7B for Binoculars (paper-grade)
            setting_suffix: Suffix for this run setting (e.g., "N=4")
        """
        pipeline_start = time.time()
        step_times = {}
        
        logger.info("=" * 60)
        logger.info("Starting StealthRL Evaluation Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Load datasets
        step_start = time.time()
        self.load_datasets(datasets, n_human=n_human, n_ai=n_ai, cache_dir=cache_dir)
        step_times["1_load_datasets"] = time.time() - step_start
        logger.info(f"[TIMING] Step 1 (Load datasets): {step_times['1_load_datasets']:.2f}s")
        
        if not self.datasets:
            logger.error("No datasets loaded. Exiting.")
            return
        
        # Step 2: Load detectors
        step_start = time.time()
        self.load_detectors(detectors, binoculars_full=binoculars_full)
        step_times["2_load_detectors"] = time.time() - step_start
        logger.info(f"[TIMING] Step 2 (Load detectors): {step_times['2_load_detectors']:.2f}s")
        
        if not self.detectors:
            logger.error("No detectors loaded. Exiting.")
            return
        
        # Step 3: Load methods
        step_start = time.time()
        self.load_methods(methods, stealthrl_checkpoint=stealthrl_checkpoint)
        step_times["3_load_methods"] = time.time() - step_start
        logger.info(f"[TIMING] Step 3 (Load methods): {step_times['3_load_methods']:.2f}s")
        
        if not self.methods:
            logger.error("No methods loaded. Exiting.")
            return
        
        # Step 4: Calibrate thresholds
        step_start = time.time()
        self.calibrate_detector_thresholds()
        step_times["4_calibrate_thresholds"] = time.time() - step_start
        logger.info(f"[TIMING] Step 4 (Calibrate thresholds): {step_times['4_calibrate_thresholds']:.2f}s")
        
        # Step 5: Run attacks
        step_start = time.time()
        outputs = self.run_attacks(n_candidates=n_candidates)
        step_times["5_run_attacks"] = time.time() - step_start
        logger.info(f"[TIMING] Step 5 (Run attacks): {step_times['5_run_attacks']:.2f}s")
        
        # Step 6: Score outputs
        step_start = time.time()
        self.score_outputs(outputs)
        step_times["6_score_outputs"] = time.time() - step_start
        logger.info(f"[TIMING] Step 6 (Score outputs): {step_times['6_score_outputs']:.2f}s")
        
        # Step 7: Compute metrics
        step_start = time.time()
        self.compute_all_metrics()
        step_times["7_compute_metrics"] = time.time() - step_start
        logger.info(f"[TIMING] Step 7 (Compute metrics): {step_times['7_compute_metrics']:.2f}s")
        
        # Step 8: Compute quality metrics
        step_start = time.time()
        self.compute_quality_metrics(outputs)
        step_times["8_quality_metrics"] = time.time() - step_start
        logger.info(f"[TIMING] Step 8 (Quality metrics): {step_times['8_quality_metrics']:.2f}s")
        
        # Step 9: Save artifacts
        step_start = time.time()
        self.save_all_artifacts()
        step_times["9_save_artifacts"] = time.time() - step_start
        logger.info(f"[TIMING] Step 9 (Save artifacts): {step_times['9_save_artifacts']:.2f}s")
        
        pipeline_elapsed = time.time() - pipeline_start
        
        logger.info("=" * 60)
        logger.info("Evaluation Complete!")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("=" * 60)
        logger.info("[TIMING] Pipeline Summary:")
        for step_name, elapsed in step_times.items():
            pct = (elapsed / pipeline_elapsed) * 100 if pipeline_elapsed > 0 else 0
            logger.info(f"  {step_name}: {elapsed:.2f}s ({pct:.1f}%)")
        logger.info(f"  TOTAL: {pipeline_elapsed:.2f}s")
        logger.info("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="StealthRL Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Dataset arguments
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mage"],
        choices=["mage", "raid", "padben"],
        help="Datasets to evaluate on",
    )
    parser.add_argument("--n-human", type=int, default=1000, help="Number of human samples")
    parser.add_argument("--n-ai", type=int, default=1000, help="Number of AI samples")
    
    # Method arguments
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["m0", "m1"],
        help="Methods to evaluate (m0=no_attack, m1=simple, m2=stealthrl, etc.)",
    )
    parser.add_argument(
        "--stealthrl-checkpoint",
        type=str,
        default=None,
        help="Path to StealthRL LoRA checkpoint (required for m2/stealthrl)",
    )
    parser.add_argument(
        "--n-candidates",
        nargs="+",
        type=int,
        default=[4],
        help="Number of candidates per sample (supports sweep: --n-candidates 1 2 4 8)",
    )
    parser.add_argument(
        "--budget-sweep",
        action="store_true",
        help="Run budget sweep with N=1,2,4,8 candidates",
    )
    
    # Detector arguments
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["roberta", "fast_detectgpt"],
        choices=["roberta", "fast_detectgpt", "detectgpt", "binoculars", "ghostbuster"],
        help="Detectors to evaluate against (note: watermark removed - not applicable without watermarked text)",
    )
    parser.add_argument(
        "--binoculars-full",
        action="store_true",
        help="Use full Falcon-7B models for Binoculars (requires ~14GB VRAM). Default uses lightweight GPT-2 pair.",
    )
    
    # Output arguments
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts",
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run (creates timestamped subdirectory)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for downloads",
    )
    
    # Output format arguments
    parser.add_argument(
        "--save-parquet",
        action="store_true",
        default=True,
        help="Save results as parquet files",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Also save results as CSV files",
    )
    
    # Runtime arguments
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap samples")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Handle budget sweep flag
    if args.budget_sweep:
        args.n_candidates = [1, 2, 4, 8]
    
    # Create run directory with timestamp if run_name provided
    if args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = str(Path(args.out_dir) / f"{args.run_name}_{timestamp}")
    
    # Setup logging
    setup_logging(args.out_dir, args.log_level)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("StealthRL Evaluation Pipeline")
    logger.info("=" * 60)
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Detectors: {args.detectors}")
    logger.info(f"Binoculars mode: {'Falcon-7B (full)' if args.binoculars_full else 'GPT-2 (lightweight)'}")
    logger.info(f"N candidates: {args.n_candidates}")
    logger.info(f"Output dir: {args.out_dir}")
    logger.info("=" * 60)
    
    # Create and run evaluator
    runner = EvalRunner(
        output_dir=args.out_dir,
        device=args.device,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
    )
    
    # Run for each N value (budget sweep)
    for n_cand in args.n_candidates:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running with N={n_cand} candidates")
        logger.info(f"{'='*60}\n")
        
        runner.run(
            datasets=args.datasets,
            methods=args.methods,
            detectors=args.detectors,
            n_candidates=n_cand,
            n_human=args.n_human,
            n_ai=args.n_ai,
            stealthrl_checkpoint=args.stealthrl_checkpoint,
            cache_dir=args.cache_dir,
            binoculars_full=args.binoculars_full,
            setting_suffix=f"N={n_cand}",
        )


if __name__ == "__main__":
    main()
