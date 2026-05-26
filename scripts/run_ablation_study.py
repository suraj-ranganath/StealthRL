#!/usr/bin/env python3
"""
Run ablation study with all generated configs.

Can run sequentially or generate commands for parallel execution.
"""

import argparse
import subprocess
from pathlib import Path
from typing import List


ABLATION_DIR = Path("configs/ablations")
TRAIN_COMMAND = "python -m stealthrl.tinker.train --config {config}"


def find_ablation_configs() -> List[Path]:
    """Find all ablation config files."""
    if not ABLATION_DIR.exists():
        raise FileNotFoundError(
            f"Ablation dir not found: {ABLATION_DIR}\n"
            "Run: python scripts/generate_ablation_configs.py"
        )
    
    configs = sorted(ABLATION_DIR.glob("*.yaml"))
    return configs


def run_sequential(configs: List[Path], dry_run: bool = False):
    """Run training jobs sequentially."""
    print(f"Running {len(configs)} ablation configs sequentially...")
    print()
    
    for i, config in enumerate(configs, 1):
        cmd = TRAIN_COMMAND.format(config=config)
        print(f"[{i}/{len(configs)}] {config.name}")
        print(f"  Command: {cmd}")
        
        if dry_run:
            print("  (dry run - skipped)")
        else:
            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0:
                print(f"  ✗ Failed with exit code {result.returncode}")
                print("  Continuing to next config...")
            else:
                print(f"  ✓ Completed")
        print()


def generate_parallel_commands(configs: List[Path], output_file: str = "ablation_commands.sh"):
    """Generate shell script with all training commands."""
    with open(output_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Ablation study training commands\n")
        f.write("# Run in parallel using GNU parallel or manually\n\n")
        
        for config in configs:
            cmd = TRAIN_COMMAND.format(config=config)
            f.write(f"{cmd}\n")
    
    print(f"Generated commands: {output_file}")
    print()
    print("To run in parallel (if you have GNU parallel):")
    print(f"  parallel -j 4 < {output_file}")
    print()
    print("Or run manually on different GPUs:")
    print("  # Terminal 1 (GPU 0):")
    print("  CUDA_VISIBLE_DEVICES=0 python -m stealthrl.tinker.train --config configs/ablations/det1.0_sem0.0.yaml")
    print("  # Terminal 2 (GPU 1):")
    print("  CUDA_VISIBLE_DEVICES=1 python -m stealthrl.tinker.train --config configs/ablations/det1.0_sem0.5.yaml")


def main():
    parser = argparse.ArgumentParser(description="Run reward weight ablation study")
    parser.add_argument(
        "--mode",
        choices=["sequential", "parallel", "commands"],
        default="commands",
        help="Execution mode: sequential (run one by one), parallel (not implemented), commands (generate script)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    
    args = parser.parse_args()
    
    # Find configs
    configs = find_ablation_configs()
    print(f"Found {len(configs)} ablation configs")
    print()
    
    # Run based on mode
    if args.mode == "sequential":
        run_sequential(configs, dry_run=args.dry_run)
    elif args.mode == "commands":
        generate_parallel_commands(configs)
    else:
        print(f"Mode '{args.mode}' not implemented")


if __name__ == "__main__":
    main()
