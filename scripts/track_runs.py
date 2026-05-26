#!/usr/bin/env python3
"""
Track and analyze StealthRL training runs.

Features:
- List all runs with metadata
- Show cost and credit information
- Display training metrics
- Compare multiple runs
- Export reports
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse


class RunTracker:
    """Track and analyze training runs."""
    
    def __init__(self, runs_dir: str = "outputs/runs"):
        self.runs_dir = Path(runs_dir)
        
    def list_runs(self) -> List[Dict[str, Any]]:
        """List all training runs with metadata."""
        runs = []
        
        if not self.runs_dir.exists():
            return runs
        
        for run_dir in sorted(self.runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
                
            metadata_file = run_dir / "run_metadata.json"
            if not metadata_file.exists():
                continue
            
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                
                # Calculate duration if run completed
                duration = None
                if "start_time" in metadata and "end_time" in metadata:
                    start = datetime.fromisoformat(metadata["start_time"])
                    end = datetime.fromisoformat(metadata["end_time"])
                    duration = (end - start).total_seconds()
                
                # Load metrics summary
                metrics_file = run_dir / "metrics.jsonl"
                metrics_summary = self._summarize_metrics(metrics_file)
                
                runs.append({
                    "run_name": metadata.get("run_name", run_dir.name),
                    "status": metadata.get("status", "unknown"),
                    "start_time": metadata.get("start_time"),
                    "end_time": metadata.get("end_time"),
                    "duration_seconds": duration,
                    "num_epochs": metadata.get("num_epochs"),
                    "batch_size": metadata.get("batch_size"),
                    "metrics": metrics_summary,
                    "path": str(run_dir),
                })
            except Exception as e:
                print(f"Warning: Failed to load run {run_dir.name}: {e}", file=sys.stderr)
                
        return runs
    
    def _summarize_metrics(self, metrics_file: Path) -> Dict[str, Any]:
        """Summarize metrics from a metrics.jsonl file."""
        if not metrics_file.exists():
            return {}
        
        try:
            total_iterations = 0
            total_reward = 0.0
            total_kl = 0.0
            
            with open(metrics_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    total_iterations += 1
                    total_reward += entry.get("reward", 0.0)
                    total_kl += entry.get("kl_divergence", 0.0)
            
            if total_iterations == 0:
                return {}
            
            return {
                "iterations": total_iterations,
                "avg_reward": total_reward / total_iterations,
                "avg_kl": total_kl / total_iterations,
            }
        except Exception as e:
            print(f"Warning: Failed to parse metrics: {e}", file=sys.stderr)
            return {}
    
    def get_run_details(self, run_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific run."""
        runs = self.list_runs()
        for run in runs:
            if run["run_name"] == run_name:
                return run
        return None
    
    def print_runs_table(self, runs: List[Dict[str, Any]]):
        """Print runs in a formatted table."""
        if not runs:
            print("No runs found.")
            return
        
        # Header
        print("\n" + "=" * 120)
        print(f"{'Run Name':<30} {'Status':<12} {'Start Time':<20} {'Duration':<12} {'Epochs':<8} {'Batch':<8} {'Avg Reward':<12}")
        print("=" * 120)
        
        # Rows
        for run in runs:
            run_name = run["run_name"][:28]
            status = run["status"]
            start = run.get("start_time", "N/A")
            if start != "N/A":
                start = datetime.fromisoformat(start).strftime("%Y-%m-%d %H:%M:%S")
            
            duration = "N/A"
            if run.get("duration_seconds"):
                mins = int(run["duration_seconds"] / 60)
                secs = int(run["duration_seconds"] % 60)
                duration = f"{mins}m {secs}s"
            
            epochs = str(run.get("num_epochs", "N/A"))
            batch = str(run.get("batch_size", "N/A"))
            
            avg_reward = "N/A"
            if run.get("metrics") and "avg_reward" in run["metrics"]:
                avg_reward = f"{run['metrics']['avg_reward']:.4f}"
            
            print(f"{run_name:<30} {status:<12} {start:<20} {duration:<12} {epochs:<8} {batch:<8} {avg_reward:<12}")
        
        print("=" * 120)
        print(f"Total runs: {len(runs)}\n")
    
    def get_cost_info(self) -> Dict[str, Any]:
        """
        Get cost and credit information from Tinker API.
        
        Note: This requires Tinker API integration. Currently returns placeholder data.
        TODO: Implement actual Tinker API queries for billing info.
        """
        # Placeholder - would query Tinker API for actual cost data
        return {
            "credits_remaining": "N/A (requires Tinker API integration)",
            "total_cost": "N/A",
            "cost_per_run": {},
            "note": "To implement: Use tinker_cookbook.rest.RestClient to query billing/usage endpoints"
        }
    
    def compare_runs(self, run_names: List[str]):
        """Compare metrics across multiple runs."""
        runs = self.list_runs()
        selected_runs = [r for r in runs if r["run_name"] in run_names]
        
        if not selected_runs:
            print("No matching runs found.")
            return
        
        print("\n" + "=" * 80)
        print("Run Comparison")
        print("=" * 80)
        
        for run in selected_runs:
            print(f"\nRun: {run['run_name']}")
            print(f"  Status: {run['status']}")
            print(f"  Duration: {run.get('duration_seconds', 'N/A')} seconds")
            print(f"  Epochs: {run.get('num_epochs', 'N/A')}")
            print(f"  Batch Size: {run.get('batch_size', 'N/A')}")
            
            if run.get("metrics"):
                print(f"  Iterations: {run['metrics'].get('iterations', 'N/A')}")
                print(f"  Avg Reward: {run['metrics'].get('avg_reward', 'N/A'):.4f}")
                print(f"  Avg KL: {run['metrics'].get('avg_kl', 'N/A'):.4f}")
    
    def export_report(self, output_file: str = "run_report.json"):
        """Export comprehensive run report to JSON."""
        runs = self.list_runs()
        cost_info = self.get_cost_info()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_runs": len(runs),
            "runs": runs,
            "cost_info": cost_info,
        }
        
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"Report exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Track and analyze StealthRL training runs")
    parser.add_argument("--runs-dir", default="outputs/runs", help="Directory containing training runs")
    parser.add_argument("--list", action="store_true", help="List all runs")
    parser.add_argument("--details", type=str, metavar="RUN_NAME", help="Show detailed info for a specific run")
    parser.add_argument("--compare", nargs="+", metavar="RUN_NAME", help="Compare multiple runs")
    parser.add_argument("--export", type=str, metavar="OUTPUT_FILE", help="Export report to JSON file")
    parser.add_argument("--cost", action="store_true", help="Show cost and credit information")
    
    args = parser.parse_args()
    
    tracker = RunTracker(runs_dir=args.runs_dir)
    
    if args.list or not any([args.details, args.compare, args.export, args.cost]):
        runs = tracker.list_runs()
        tracker.print_runs_table(runs)
    
    if args.details:
        run = tracker.get_run_details(args.details)
        if run:
            print(json.dumps(run, indent=2))
        else:
            print(f"Run not found: {args.details}")
    
    if args.compare:
        tracker.compare_runs(args.compare)
    
    if args.export:
        tracker.export_report(args.export)
    
    if args.cost:
        cost_info = tracker.get_cost_info()
        print("\nCost Information:")
        print(json.dumps(cost_info, indent=2))


if __name__ == "__main__":
    main()
