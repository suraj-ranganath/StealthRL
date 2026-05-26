#!/usr/bin/env python3
"""
Analyze and compare training run configurations from run directories.

This script extracts configuration information from training logs and metadata
to understand what settings were used for each run, especially useful for 
comparing successful runs or preparing for inference.

Usage:
    # Analyze a single run
    python scripts/analyze_run_configs.py --run-dir outputs/mage/run_20260130_013157
    
    # Compare multiple runs
    python scripts/analyze_run_configs.py --run-dirs outputs/mage/run_* --compare
    
    # Find runs with specific config values
    python scripts/analyze_run_configs.py --search-dir outputs/mage --filter "batch_size==16"
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import sys


def extract_config_from_log(log_file: Path) -> Dict[str, Any]:
    """
    Extract configuration parameters from training.log file.
    
    Args:
        log_file: Path to training.log
        
    Returns:
        Dictionary of config parameters
    """
    config = {}
    
    if not log_file.exists():
        return config
    
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Extract key parameters from log lines
    patterns = {
        # Training hyperparameters
        'learning_rate': r'learning_rate[:\s=]+([0-9.e-]+)',
        'batch_size': r'batch_size[:\s=]+(\d+)',
        'group_size': r'group_size[:\s=]+(\d+)',
        'num_epochs': r'num_epochs[:\s=]+(\d+)',
        'lora_rank': r'(?:lora_)?rank[:\s=]+(\d+)',
        'lora_alpha': r'(?:lora_)?alpha[:\s=]+(\d+)',
        'lora_dropout': r'(?:lora_)?dropout[:\s=]+([0-9.]+)',
        
        # Reward weights - match pattern like "det=1.0, sem=0.2, ppl=0.05"
        'detector_weight': r'det=([0-9.]+)',
        'semantic_weight': r'sem=([0-9.]+)',
        'perplexity_weight': r'ppl=([0-9.]+)',
        'fairness_weight': r'fairness[_\s]weight[:\s=]+([0-9.]+)',
        
        # RL parameters
        'kl_penalty_coef': r'kl_penalty(?:_coef)?[:\s=]+([0-9.e-]+)',
        'temperature': r'temperature[:\s=]+([0-9.]+)',
        'max_tokens': r'max_tokens[:\s=]+(\d+)',
        
        # Training mode
        'training_mode': r'training_mode[:\s=]+[\'"]*(\w+)[\'"]*',
        
        # Dataset info
        'train_dataset_size': r'Train dataset: (\d+)',
        'test_dataset_size': r'Test dataset: (\d+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, log_content, re.IGNORECASE)
        if match:
            value = match.group(1)
            # Try to convert to appropriate type
            try:
                if '.' in value or 'e' in value.lower():
                    config[key] = float(value)
                elif value.isdigit():
                    config[key] = int(value)
                else:
                    config[key] = value
            except ValueError:
                config[key] = value
    
    return config


def extract_config_from_metadata(metadata_file: Path) -> Dict[str, Any]:
    """
    Extract configuration from run_metadata.json.
    
    Args:
        metadata_file: Path to run_metadata.json
        
    Returns:
        Dictionary of config parameters
    """
    if not metadata_file.exists():
        return {}
    
    with open(metadata_file, 'r') as f:
        return json.load(f)


def extract_metrics_summary(metrics_file: Path) -> Dict[str, Any]:
    """
    Extract final metrics from metrics.jsonl.
    
    Args:
        metrics_file: Path to metrics.jsonl
        
    Returns:
        Dictionary with final metrics
    """
    if not metrics_file.exists():
        return {}
    
    # Read all lines and find last test metrics and last training metrics
    last_train_metrics = None
    last_test_metrics = None
    
    with open(metrics_file, 'r') as f:
        for line in f:
            metrics = json.loads(line)
            
            # Check if this is a test evaluation line
            if any(k.startswith('test/') for k in metrics.keys()):
                last_test_metrics = metrics
            else:
                last_train_metrics = metrics
    
    # Extract key metrics from test set (preferred) or training set
    summary = {}
    
    # Try test metrics first
    if last_test_metrics:
        summary['final_detector_evasion'] = last_test_metrics.get('test/env/all/detector_prob')
        summary['final_semantic_sim'] = last_test_metrics.get('test/env/all/semantic_sim')
        summary['final_total_reward'] = last_test_metrics.get('test/env/all/reward/total')
        summary['final_perplexity'] = last_test_metrics.get('test/env/all/perplexity')
        summary['final_kl'] = last_test_metrics.get('test/env/all/kl')
        summary['metrics_source'] = 'test'
    
    # Fallback to training metrics
    if last_train_metrics:
        if 'final_detector_evasion' not in summary or summary['final_detector_evasion'] is None:
            summary['final_detector_evasion'] = last_train_metrics.get('env/all/detector_prob')
        if 'final_semantic_sim' not in summary or summary['final_semantic_sim'] is None:
            summary['final_semantic_sim'] = last_train_metrics.get('env/all/semantic_sim')
        if 'final_total_reward' not in summary or summary['final_total_reward'] is None:
            summary['final_total_reward'] = last_train_metrics.get('env/all/reward/total')
        if 'final_perplexity' not in summary or summary['final_perplexity'] is None:
            summary['final_perplexity'] = last_train_metrics.get('env/all/perplexity')
        if 'final_kl' not in summary or summary['final_kl'] is None:
            summary['final_kl'] = last_train_metrics.get('env/all/kl')
        
        summary['total_iterations'] = last_train_metrics.get('iteration')
        
        if 'metrics_source' not in summary:
            summary['metrics_source'] = 'train'
    
    return {k: v for k, v in summary.items() if v is not None}


def analyze_run(run_dir: Path) -> Dict[str, Any]:
    """
    Analyze a single training run directory.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Complete config and metrics dictionary
    """
    run_info = {
        'run_dir': str(run_dir),
        'run_name': run_dir.name,
    }
    
    # Extract from metadata
    metadata_file = run_dir / "run_metadata.json"
    metadata = extract_config_from_metadata(metadata_file)
    run_info.update(metadata)
    
    # Extract from training log
    log_file = run_dir / "training.log"
    log_config = extract_config_from_log(log_file)
    run_info.update(log_config)
    
    # Extract final metrics
    metrics_file = run_dir / "metrics.jsonl"
    metrics = extract_metrics_summary(metrics_file)
    run_info.update(metrics)
    
    # Look for config.yaml if referenced
    if 'config_file' in run_info and run_info['config_file']:
        config_path = Path(run_info['config_file'])
        if config_path.exists():
            run_info['config_file_exists'] = True
        else:
            run_info['config_file_exists'] = False
    
    return run_info


def print_run_summary(run_info: Dict[str, Any], detailed: bool = False):
    """Pretty print run configuration summary."""
    print("\n" + "=" * 80)
    print(f"Run: {run_info['run_name']}")
    print("=" * 80)
    
    # Basic info
    print("\n📁 Run Information:")
    print(f"  Directory:      {run_info['run_dir']}")
    print(f"  Start Time:     {run_info.get('start_time', 'N/A')}")
    print(f"  Status:         {run_info.get('status', 'N/A')}")
    if 'end_time' in run_info:
        print(f"  End Time:       {run_info['end_time']}")
    
    # Config file
    if 'config_file' in run_info:
        exists = run_info.get('config_file_exists', False)
        status = "✓" if exists else "✗"
        print(f"  Config File:    {run_info['config_file']} [{status}]")
    
    # Training settings
    print("\n🎯 Training Configuration:")
    print(f"  Data Path:      {run_info.get('data_path', 'N/A')}")
    print(f"  Epochs:         {run_info.get('num_epochs', 'N/A')}")
    print(f"  Batch Size:     {run_info.get('batch_size', 'N/A')}")
    print(f"  Group Size:     {run_info.get('group_size', 'N/A')}")
    print(f"  Training Mode:  {run_info.get('training_mode', 'N/A')}")
    if 'train_dataset_size' in run_info:
        print(f"  Train Batches:  {run_info['train_dataset_size']}")
    if 'test_dataset_size' in run_info:
        print(f"  Test Batches:   {run_info['test_dataset_size']}")
    
    # Model settings
    print("\n🤖 Model Configuration:")
    print(f"  Learning Rate:  {run_info.get('learning_rate', 'N/A')}")
    print(f"  LoRA Rank:      {run_info.get('lora_rank', 'N/A')}")
    if 'lora_alpha' in run_info:
        print(f"  LoRA Alpha:     {run_info['lora_alpha']}")
    if 'lora_dropout' in run_info:
        print(f"  LoRA Dropout:   {run_info['lora_dropout']}")
    print(f"  Temperature:    {run_info.get('temperature', 'N/A')}")
    print(f"  Max Tokens:     {run_info.get('max_tokens', 'N/A')}")
    print(f"  KL Penalty:     {run_info.get('kl_penalty_coef', 'N/A')}")
    
    # Reward weights
    print("\n⚖️ Reward Weights:")
    print(f"  Detector:       {run_info.get('detector_weight', 'N/A')}")
    print(f"  Semantic:       {run_info.get('semantic_weight', 'N/A')}")
    print(f"  Perplexity:     {run_info.get('perplexity_weight', 'N/A')}")
    print(f"  Fairness:       {run_info.get('fairness_weight', 'N/A')}")
    
    # Final metrics
    print("\n📊 Final Metrics:")
    print(f"  Detector Evasion:  {run_info.get('final_detector_evasion', 'N/A')}")
    print(f"  Semantic Sim:      {run_info.get('final_semantic_sim', 'N/A')}")
    print(f"  Total Reward:      {run_info.get('final_total_reward', 'N/A')}")
    print(f"  Perplexity:        {run_info.get('final_perplexity', 'N/A')}")
    print(f"  Total Iterations:  {run_info.get('total_iterations', 'N/A')}")
    
    if detailed:
        # Print all extracted fields
        print("\n🔍 All Extracted Fields:")
        for key, value in sorted(run_info.items()):
            if key not in ['run_dir', 'run_name']:
                print(f"  {key}: {value}")


def compare_runs(run_infos: List[Dict[str, Any]]):
    """Compare multiple runs and highlight differences."""
    if len(run_infos) < 2:
        print("Need at least 2 runs to compare")
        return
    
    print("\n" + "=" * 80)
    print(f"Comparing {len(run_infos)} Runs")
    print("=" * 80)
    
    # Collect all keys
    all_keys = set()
    for info in run_infos:
        all_keys.update(info.keys())
    
    # Skip these keys
    skip_keys = {'run_dir', 'start_time', 'end_time', 'status', 'total_iterations'}
    
    # Group keys by category
    config_keys = sorted([k for k in all_keys if k not in skip_keys])
    
    # Print comparison table
    print("\n📊 Configuration Comparison:")
    print(f"{'Parameter':<25} " + " ".join([f"Run {i+1:>8}" for i in range(len(run_infos))]))
    print("-" * (25 + 10 * len(run_infos)))
    
    for key in config_keys:
        values = [str(info.get(key, 'N/A')) for info in run_infos]
        
        # Check if values differ
        unique_values = set(values)
        if len(unique_values) > 1:
            marker = " *"  # Mark different values
        else:
            marker = "  "
        
        print(f"{key:<25}{marker}" + " ".join([f"{v:>10}" for v in values]))
    
    print("\n* = Different values across runs")
    
    # Highlight key differences
    print("\n🔍 Key Differences:")
    important_keys = [
        'batch_size', 'learning_rate', 'detector_weight', 'semantic_weight',
        'final_detector_evasion', 'final_semantic_sim', 'final_total_reward'
    ]
    
    for key in important_keys:
        values = [info.get(key) for info in run_infos]
        unique_values = set(v for v in values if v is not None)
        if len(unique_values) > 1:
            print(f"  {key}: {', '.join([str(v) for v in values])}")


def find_best_run(run_infos: List[Dict[str, Any]], metric: str = 'final_detector_evasion'):
    """Find the run with best metric value."""
    valid_runs = [r for r in run_infos if metric in r and r[metric] is not None]
    
    if not valid_runs:
        print(f"No runs with {metric} found")
        return None
    
    # Detector evasion is 1 - detection_prob, so higher is better
    best_run = max(valid_runs, key=lambda r: r[metric])
    
    print(f"\n🏆 Best Run by {metric}:")
    print(f"  Run:    {best_run['run_name']}")
    print(f"  Value:  {best_run[metric]:.4f}")
    print(f"  Path:   {best_run['run_dir']}")
    
    return best_run


def save_config_json(run_info: Dict[str, Any], output_file: Path):
    """Save extracted config to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(run_info, f, indent=2, default=str)
    print(f"\n💾 Config saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare training run configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single run
  python scripts/analyze_run_configs.py --run-dir outputs/mage/run_20260130_013157
  
  # Compare multiple runs
  python scripts/analyze_run_configs.py --run-dirs outputs/mage/run_* --compare
  
  # Find best run by detector evasion
  python scripts/analyze_run_configs.py --search-dir outputs/mage --best
  
  # Export config to JSON
  python scripts/analyze_run_configs.py --run-dir outputs/mage/run_20260130_013157 --export config.json
        """
    )
    
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Single run directory to analyze"
    )
    parser.add_argument(
        "--run-dirs",
        type=str,
        nargs="+",
        help="Multiple run directories to analyze"
    )
    parser.add_argument(
        "--search-dir",
        type=str,
        help="Search for all runs in directory"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple runs"
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Find best run by detector evasion"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="final_detector_evasion",
        help="Metric to use for --best (default: final_detector_evasion)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed output with all fields"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export config to JSON file"
    )
    
    args = parser.parse_args()
    
    # Collect run directories
    run_dirs = []
    
    if args.run_dir:
        run_dirs.append(Path(args.run_dir))
    
    if args.run_dirs:
        run_dirs.extend([Path(d) for d in args.run_dirs])
    
    if args.search_dir:
        search_path = Path(args.search_dir)
        # Find all directories that look like runs
        for item in search_path.iterdir():
            if item.is_dir() and (item / "run_metadata.json").exists():
                run_dirs.append(item)
    
    if not run_dirs:
        print("Error: No run directories specified")
        print("Use --run-dir, --run-dirs, or --search-dir")
        sys.exit(1)
    
    # Analyze runs
    run_infos = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"Warning: {run_dir} does not exist, skipping")
            continue
        
        run_info = analyze_run(run_dir)
        run_infos.append(run_info)
    
    if not run_infos:
        print("Error: No valid runs found")
        sys.exit(1)
    
    # Single run analysis
    if len(run_infos) == 1:
        print_run_summary(run_infos[0], detailed=args.detailed)
        
        if args.export:
            save_config_json(run_infos[0], Path(args.export))
    
    # Multiple runs
    else:
        if args.compare:
            compare_runs(run_infos)
        else:
            # Print each run
            for run_info in run_infos:
                print_run_summary(run_info, detailed=args.detailed)
        
        if args.best:
            best_run = find_best_run(run_infos, metric=args.metric)
            
            if best_run and args.export:
                save_config_json(best_run, Path(args.export))


if __name__ == "__main__":
    main()
