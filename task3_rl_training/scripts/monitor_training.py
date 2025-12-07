#!/usr/bin/env python3
"""
Training monitoring tool for TASK 3.

Displays real-time training progress, key metrics, and alerts for issues.

Usage:
    python task3_rl_training/scripts/monitor_training.py --run-name stealthrl_full_ensemble
    python task3_rl_training/scripts/monitor_training.py --run-name stealthrl_full_ensemble --watch
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional
import sys

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'

def clear_screen():
    """Clear terminal screen."""
    print('\033[2J\033[H', end='')

def print_header(text: str):
    """Print section header."""
    print(f"{Colors.CYAN}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.CYAN}{'─' * len(text)}{Colors.END}")

def format_metric(value: float, target: Optional[float] = None, higher_better: bool = True) -> str:
    """Format metric with color based on target."""
    if target is None:
        return f"{value:.4f}"

    if higher_better:
        if value >= target:
            return f"{Colors.GREEN}{value:.4f}{Colors.END} ✓"
        else:
            return f"{Colors.YELLOW}{value:.4f}{Colors.END}"
    else:
        if value <= target:
            return f"{Colors.GREEN}{value:.4f}{Colors.END} ✓"
        else:
            return f"{Colors.YELLOW}{value:.4f}{Colors.END}"

def load_status(run_dir: Path) -> Dict:
    """Load training status."""
    status_file = run_dir / "status.json"

    if not status_file.exists():
        return {"status": "NOT_STARTED", "message": "Training not started"}

    with open(status_file, 'r') as f:
        return json.load(f)

def load_metrics(run_dir: Path) -> Optional[Dict]:
    """Load latest metrics."""
    metrics_file = run_dir / "metrics.json"

    if not metrics_file.exists():
        return None

    with open(metrics_file, 'r') as f:
        return json.load(f)

def parse_training_log(run_dir: Path, tail_lines: int = 50) -> list:
    """Parse recent training log lines."""
    log_file = run_dir / "training.log"

    if not log_file.exists():
        return []

    with open(log_file, 'r') as f:
        lines = f.readlines()

    return lines[-tail_lines:]

def check_for_issues(metrics: Dict, log_lines: list) -> list:
    """Check for common training issues."""
    issues = []

    if not metrics:
        return issues

    # Check all-negative groups
    if metrics.get('all_negative_frac_total', 0) > 0.7:
        issues.append({
            "level": "WARNING",
            "message": f"High all-negative fraction: {metrics['all_negative_frac_total']:.2%}",
            "suggestion": "Consider increasing all_negative.min_reward in config"
        })

    # Check KL divergence
    if metrics.get('kl', 0) > 0.05:
        issues.append({
            "level": "WARNING",
            "message": f"High KL divergence: {metrics['kl']:.4f}",
            "suggestion": "Consider increasing kl.penalty_coef in config"
        })

    # Check semantic similarity
    if metrics.get('reward/semantic', 0) < 0.75:
        issues.append({
            "level": "WARNING",
            "message": f"Low semantic similarity: {metrics['reward/semantic']:.4f}",
            "suggestion": "Consider increasing semantic_weight in config"
        })

    # Check for errors in logs
    error_lines = [line for line in log_lines if 'ERROR' in line or 'Exception' in line]
    if error_lines:
        issues.append({
            "level": "ERROR",
            "message": f"Found {len(error_lines)} error(s) in recent logs",
            "suggestion": "Check training.log for details"
        })

    return issues

def display_training_status(run_name: str, run_dir: Path, watch_mode: bool = False):
    """Display training status dashboard."""

    if watch_mode:
        clear_screen()

    print(f"\n{Colors.BOLD}{'═' * 70}{Colors.END}")
    print(f"{Colors.BOLD}StealthRL Training Monitor - {run_name}{Colors.END}")
    print(f"{Colors.BOLD}{'═' * 70}{Colors.END}\n")

    # Load data
    status = load_status(run_dir)
    metrics = load_metrics(run_dir)
    log_lines = parse_training_log(run_dir)

    # Display status
    print_header("Training Status")
    status_text = status.get('status', 'UNKNOWN')
    if status_text == 'RUNNING':
        print(f"Status: {Colors.GREEN}{status_text}{Colors.END}")
    elif status_text == 'COMPLETED':
        print(f"Status: {Colors.BLUE}{status_text}{Colors.END}")
    elif status_text == 'FAILED':
        print(f"Status: {Colors.RED}{status_text}{Colors.END}")
    else:
        print(f"Status: {Colors.YELLOW}{status_text}{Colors.END}")

    if 'epoch' in status:
        print(f"Epoch: {status['epoch']}/{status.get('total_epochs', '?')}")
    if 'batch' in status:
        print(f"Batch: {status['batch']}")
    if 'message' in status:
        print(f"Message: {status['message']}")

    print()

    # Display metrics
    if metrics:
        print_header("Key Metrics")

        # Reward components
        print(f"\nReward Components:")
        print(f"  Total:    {format_metric(metrics.get('reward/total', 0))}")
        print(f"  Detector: {format_metric(metrics.get('reward/detector', 0), 0.5, True)}")
        print(f"  Semantic: {format_metric(metrics.get('reward/semantic', 0), 0.8, True)}")
        print(f"  Quality:  {format_metric(metrics.get('reward/quality', 0))}")
        print(f"  Fairness: {format_metric(metrics.get('reward/fairness', 0))}")

        # Training stats
        print(f"\nTraining Stats:")
        print(f"  KL Divergence:     {format_metric(metrics.get('kl', 0), 0.01, False)}")
        print(f"  All-Negative Frac: {metrics.get('all_negative_frac_total', 0):.2%}")
        print(f"  Loss:              {metrics.get('loss', 0):.4f}")

        # Performance
        if 'asr_all' in metrics:
            print(f"\nPerformance:")
            print(f"  ASR (all detectors): {metrics['asr_all']:.2%}")
            print(f"  Semantic Similarity: {metrics.get('semantic_sim_mean', 0):.4f}")

        print()
    else:
        print(f"{Colors.YELLOW}No metrics available yet{Colors.END}\n")

    # Check for issues
    issues = check_for_issues(metrics or {}, log_lines)
    if issues:
        print_header("Issues Detected")
        for issue in issues:
            level_color = Colors.RED if issue['level'] == 'ERROR' else Colors.YELLOW
            print(f"{level_color}[{issue['level']}]{Colors.END} {issue['message']}")
            print(f"  → {issue['suggestion']}\n")

    # Recent logs
    if log_lines:
        print_header("Recent Logs (last 10 lines)")
        for line in log_lines[-10:]:
            # Color-code by log level
            if 'ERROR' in line:
                print(f"{Colors.RED}{line.strip()}{Colors.END}")
            elif 'WARNING' in line:
                print(f"{Colors.YELLOW}{line.strip()}{Colors.END}")
            else:
                print(line.strip())

    print(f"\n{Colors.BOLD}{'═' * 70}{Colors.END}\n")

    # Footer
    if watch_mode:
        print(f"{Colors.CYAN}Refreshing every 30 seconds... (Ctrl+C to exit){Colors.END}")
    else:
        print(f"Run with --watch flag for continuous monitoring")
        print(f"Logs: {run_dir / 'training.log'}")

def main():
    parser = argparse.ArgumentParser(description="Monitor StealthRL training progress")
    parser.add_argument(
        '--run-name',
        type=str,
        required=True,
        help='Training run name'
    )
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Continuous monitoring mode (refresh every 30s)'
    )
    parser.add_argument(
        '--refresh-interval',
        type=int,
        default=30,
        help='Refresh interval in seconds (default: 30)'
    )

    args = parser.parse_args()

    run_dir = Path("outputs/runs") / args.run_name

    if not run_dir.exists():
        print(f"{Colors.RED}Error: Run directory not found: {run_dir}{Colors.END}")
        print(f"\nAvailable runs:")
        runs_dir = Path("outputs/runs")
        if runs_dir.exists():
            for run in runs_dir.iterdir():
                if run.is_dir():
                    print(f"  - {run.name}")
        sys.exit(1)

    if args.watch:
        try:
            while True:
                display_training_status(args.run_name, run_dir, watch_mode=True)
                time.sleep(args.refresh_interval)
        except KeyboardInterrupt:
            print(f"\n{Colors.CYAN}Monitoring stopped{Colors.END}")
    else:
        display_training_status(args.run_name, run_dir, watch_mode=False)

if __name__ == "__main__":
    main()
