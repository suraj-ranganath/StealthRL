#!/usr/bin/env python3
"""
Run full StealthRL evaluation pipeline.

This script runs the complete evaluation as specified in SPEC.md:
1. Load MAGE dataset (balanced human/AI)
2. Run all attack methods (M0-M5)
3. Score with detector panel (RoBERTa, Fast-DetectGPT, DetectGPT, Binoculars)
4. Compute metrics (AUROC, TPR@1%FPR, ASR)
5. Generate paper-ready figures and tables

Usage:
    # Quick test (small dataset, few methods)
    python scripts/run_eval.py --quick
    
    # Full evaluation
    python scripts/run_eval.py --datasets mage raid --methods m0 m1 m2 --stealthrl-checkpoint outputs/checkpoint
    
    # With specific detectors
    python scripts/run_eval.py --detectors roberta fast_detectgpt binoculars
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.runner import EvalRunner, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run StealthRL Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test
    python scripts/run_eval.py --quick
    
    # Full MAGE evaluation
    python scripts/run_eval.py --datasets mage --n-human 1000 --n-ai 1000
    
    # With StealthRL checkpoint
    python scripts/run_eval.py --stealthrl-checkpoint outputs/runs/best_checkpoint
    
    # All methods and detectors
    python scripts/run_eval.py --methods m0 m1 m2 m3 m4 m5 --detectors roberta fast_detectgpt detectgpt binoculars
        """,
    )
    
    # Quick mode
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (50 samples, 2 methods, 2 detectors)",
    )
    
    # Dataset options
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mage"],
        help="Datasets to evaluate (mage, raid, padben)",
    )
    parser.add_argument("--n-human", type=int, default=500, help="Human samples per dataset")
    parser.add_argument("--n-ai", type=int, default=500, help="AI samples per dataset")
    parser.add_argument(
        "--reuse-samples-from",
        type=str,
        default=None,
        help="Reuse exact sample ids from a prior run directory (ensures identical samples)",
    )
    
    # Method options
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["m0", "m1"],
        help="Attack methods (m0=no_attack, m1=simple, m2=stealthrl, m3=adv_para, m4=authormist, m5=homoglyph)",
    )
    parser.add_argument(
        "--stealthrl-checkpoint",
        type=str,
        default=None,
        help="Path to StealthRL checkpoint (required for m2)",
    )
    parser.add_argument(
        "--n-candidates",
        nargs="+",
        type=int,
        default=[2],
        help="Candidates per sample (supports budget sweep: --n-candidates 1 2 4 8)",
    )

    # Tinker concurrency options (StealthRL / M2 only)
    parser.add_argument(
        "--tinker-concurrency",
        type=int,
        default=64,
        help="Max concurrent Tinker sampling requests for M2 (default: 64)",
    )
    parser.add_argument(
        "--tinker-chunk-size",
        type=int,
        default=256,
        help="Chunk size for concurrent Tinker sampling (default: 256)",
    )
    parser.add_argument(
        "--tinker-max-retries",
        type=int,
        default=2,
        help="Max retries per Tinker request (default: 2)",
    )
    parser.add_argument(
        "--tinker-backoff-s",
        type=float,
        default=0.5,
        help="Base backoff seconds for retries (default: 0.5)",
    )
    parser.add_argument(
        "--tinker-resume",
        action="store_true",
        help="Enable resume cache for M2 sampling (writes to output dir)",
    )
    parser.add_argument(
        "--tinker-resume-path",
        type=str,
        default=None,
        help="Optional path for M2 resume cache (JSONL). Overrides --tinker-resume",
    )
    
    # Detector options
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["roberta", "fast_detectgpt"],
        help="Detectors to use (roberta, fast_detectgpt, detectgpt, binoculars, ghostbuster, mage)",
    )
    parser.add_argument("--roberta-batch-size", type=int, default=None, help="Batch size for RoBERTa detector")
    parser.add_argument("--fast-detectgpt-batch-size", type=int, default=None, help="Batch size for Fast-DetectGPT detector")
    parser.add_argument("--mage-batch-size", type=int, default=None, help="Batch size for MAGE detector")
    parser.add_argument("--binoculars-batch-size", type=int, default=None, help="Batch size for Binoculars detector")
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate outputs after each method/detector (safer for long runs)",
    )
    
    # Output options
    parser.add_argument("--out-dir", type=str, default="outputs/eval_runs", help="Output directory")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Cache directory")
    
    # Runtime options
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-bootstrap", type=int, default=500, help="Bootstrap samples")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")

    # Optional GPT quality evaluation
    parser.add_argument(
        "--gpt-quality",
        action="store_true",
        help="Enable GPT-based quality evaluation (requires OpenAI API key)",
    )
    parser.add_argument(
        "--gpt-quality-max-per-method",
        type=int,
        default=200,
        help="Maximum number of samples per method to judge",
    )
    parser.add_argument(
        "--gpt-quality-model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for GPT quality judging",
    )
    parser.add_argument(
        "--gpt-quality-methods",
        nargs="+",
        default=None,
        help="Methods to run GPT quality judging on (default: m2/stealthrl)",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (overrides OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--gpt-quality-no-cache",
        action="store_true",
        help="Disable GPT quality cache (always re-call API)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.datasets = ["mage"]
        args.methods = ["m0", "m1"]
        args.detectors = ["roberta", "fast_detectgpt"]
        args.n_human = 50
        args.n_ai = 50
        args.n_candidates = [2]
        args.n_bootstrap = 100
        args.out_dir = "artifacts_quick"
    
    # Setup logging
    setup_logging(args.out_dir, args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("StealthRL Evaluation Pipeline")
    logger.info("=" * 70)
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Detectors: {args.detectors}")
    logger.info(f"Samples: {args.n_human} human, {args.n_ai} AI per dataset")
    logger.info(f"Candidates: {args.n_candidates}")
    logger.info(f"Output: {args.out_dir}")
    if args.reuse_samples_from:
        logger.info(f"Reusing samples from: {args.reuse_samples_from}")
    logger.info("=" * 70)
    
    # Check for StealthRL checkpoint if needed
    if "m2" in args.methods or "stealthrl" in args.methods:
        if not args.stealthrl_checkpoint:
            logger.warning("StealthRL method requested but no checkpoint provided!")
            logger.warning("Use --stealthrl-checkpoint to specify checkpoint path")
            logger.warning("Removing StealthRL from methods...")
            args.methods = [m for m in args.methods if m not in ("m2", "stealthrl")]
    
    # Resolve OpenAI key if needed
    openai_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")

    # Resolve Tinker resume path (if enabled)
    tinker_resume_path = None
    if args.tinker_resume_path:
        tinker_resume_path = args.tinker_resume_path
    elif args.tinker_resume:
        tinker_resume_path = str(Path(args.out_dir) / "tinker_m2_cache.jsonl")

    # Load fixed samples if requested
    sample_ids = None
    if args.reuse_samples_from:
        reuse_dir = Path(args.reuse_samples_from)
        ids_path = reuse_dir / "dataset_samples.json"
        if ids_path.exists():
            sample_ids = json.loads(ids_path.read_text())
        else:
            # Fallback: infer from scores
            scores_path = reuse_dir / "scores.parquet"
            if not scores_path.exists():
                scores_path = reuse_dir / "scores.csv"
            if not scores_path.exists():
                raise FileNotFoundError("reuse-samples-from requires dataset_samples.json or scores.parquet/csv")
            scores_df = pd.read_parquet(scores_path) if scores_path.suffix == ".parquet" else pd.read_csv(scores_path)
            sample_ids = {}
            for ds in scores_df["dataset"].unique():
                ds_df = scores_df[scores_df["dataset"] == ds]
                human_ids = ds_df[ds_df["label"] == "human"]["sample_id"].drop_duplicates().tolist()
                ai_ids = ds_df[ds_df["label"] == "ai"]["sample_id"].drop_duplicates().tolist()
                sample_ids[ds] = {"human_ids": human_ids, "ai_ids": ai_ids}

    # Create runner
    runner = EvalRunner(
        output_dir=args.out_dir,
        device=args.device,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
    )
    
    # Run evaluation for each n_candidates value (budget sweep)
    try:
        for n_cand in args.n_candidates:
            if len(args.n_candidates) > 1:
                logger.info(f"\n{'='*70}")
                logger.info(f"Running with N={n_cand} candidates")
                logger.info(f"{'='*70}\n")
            
        runner.run(
            datasets=args.datasets,
            methods=args.methods,
            detectors=args.detectors,
            n_candidates=n_cand,
            n_human=args.n_human,
            n_ai=args.n_ai,
            stealthrl_checkpoint=args.stealthrl_checkpoint,
            cache_dir=args.cache_dir,
            roberta_batch_size=args.roberta_batch_size,
            fast_detectgpt_batch_size=args.fast_detectgpt_batch_size,
            mage_batch_size=args.mage_batch_size,
            binoculars_batch_size=args.binoculars_batch_size,
            save_intermediate=args.save_intermediate,
            tinker_concurrency=args.tinker_concurrency,
            tinker_chunk_size=args.tinker_chunk_size,
            tinker_max_retries=args.tinker_max_retries,
            tinker_backoff_s=args.tinker_backoff_s,
            tinker_resume_path=tinker_resume_path,
            setting_suffix=f"N={n_cand}" if len(args.n_candidates) > 1 else None,
            gpt_quality=args.gpt_quality,
            gpt_quality_methods=args.gpt_quality_methods,
            gpt_quality_max_per_method=args.gpt_quality_max_per_method,
            gpt_quality_model=args.gpt_quality_model,
            openai_api_key=openai_key,
            gpt_quality_cache=not args.gpt_quality_no_cache,
            sample_ids=sample_ids,
        )
        
        logger.info("Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
