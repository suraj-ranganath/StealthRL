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
import logging
import sys
from pathlib import Path

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
    parser.add_argument("--n-candidates", type=int, default=2, help="Candidates per sample (batched in single API call)")
    
    # Detector options
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["roberta", "fast_detectgpt"],
        help="Detectors to use (roberta, fast_detectgpt, detectgpt, binoculars, ghostbuster)",
    )
    
    # Output options
    parser.add_argument("--out-dir", type=str, default="outputs/eval_runs", help="Output directory")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Cache directory")
    
    # Runtime options
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-bootstrap", type=int, default=500, help="Bootstrap samples")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    
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
        args.n_candidates = 2
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
    logger.info("=" * 70)
    
    # Check for StealthRL checkpoint if needed
    if "m2" in args.methods or "stealthrl" in args.methods:
        if not args.stealthrl_checkpoint:
            logger.warning("StealthRL method requested but no checkpoint provided!")
            logger.warning("Use --stealthrl-checkpoint to specify checkpoint path")
            logger.warning("Removing StealthRL from methods...")
            args.methods = [m for m in args.methods if m not in ("m2", "stealthrl")]
    
    # Create runner
    runner = EvalRunner(
        output_dir=args.out_dir,
        device=args.device,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
    )
    
    # Run evaluation
    try:
        runner.run(
            datasets=args.datasets,
            methods=args.methods,
            detectors=args.detectors,
            n_candidates=args.n_candidates,
            n_human=args.n_human,
            n_ai=args.n_ai,
            stealthrl_checkpoint=args.stealthrl_checkpoint,
            cache_dir=args.cache_dir,
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
