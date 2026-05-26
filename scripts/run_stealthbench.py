#!/usr/bin/env python3
"""
Run StealthBench unified evaluation harness.

This script runs multiple detectors on common datasets and produces
standardized metrics and comparison plots.
"""

import argparse
import yaml
from pathlib import Path

from stealthrl.evaluation import StealthBench


def load_config(config_path: str) -> dict:
    """Load StealthBench configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run StealthBench evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="outputs/stealthbench_results",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # Initialize StealthBench
    bench = StealthBench(
        detectors=config['detectors'],
        output_dir=args.output_dir
    )
    
    # TODO: Load datasets and run evaluation
    print("StealthBench evaluation to be implemented...")


if __name__ == "__main__":
    main()
