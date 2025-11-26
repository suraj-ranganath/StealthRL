#!/usr/bin/env python3
"""
Real-time monitoring script for StealthRL training runs.

Displays live updates of training progress, metrics, and system status.
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime
import argparse


def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def monitor_run(run_dir: Path, refresh_interval: int = 5):
    """Monitor a training run in real-time."""
    metadata_file = run_dir / "run_metadata.json"
    metrics_file = run_dir / "metrics.jsonl"
    
    if not metadata_file.exists():
        print(f"Error: Run metadata not found in {run_dir}")
        return
    
    print(f"Monitoring run: {run_dir.name}")
    print(f"Press Ctrl+C to stop\n")
    
    last_line_count = 0
    
    try:
        while True:
            clear_screen()
            
            # Load metadata
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            # Display header
            print("=" * 80)
            print(f"Run: {metadata.get('run_name', 'Unknown')}")
            print(f"Status: {metadata.get('status', 'unknown')}")
            print("=" * 80)
            
            # Calculate runtime
            start_time = metadata.get("start_time")
            if start_time:
                start_dt = datetime.fromisoformat(start_time)
                runtime = (datetime.now() - start_dt).total_seconds()
                print(f"Runtime: {format_duration(runtime)}")
            
            # Display config
            print(f"\nConfiguration:")
            print(f"  Epochs: {metadata.get('num_epochs', 'N/A')}")
            print(f"  Batch Size: {metadata.get('batch_size', 'N/A')}")
            print(f"  Data Path: {metadata.get('data_path', 'N/A')}")
            
            # Display recent metrics
            if metrics_file.exists():
                print(f"\nTraining Metrics:")
                with open(metrics_file) as f:
                    lines = f.readlines()
                
                if lines:
                    # Show last 5 iterations
                    recent_lines = lines[-5:]
                    
                    print(f"  Total Iterations: {len(lines)}")
                    print(f"\n  Recent Iterations:")
                    print(f"  {'Iter':<8} {'Reward':<12} {'KL Div':<12} {'Timestamp'}")
                    print(f"  {'-'*60}")
                    
                    for i, line in enumerate(recent_lines):
                        try:
                            entry = json.loads(line)
                            iter_num = len(lines) - len(recent_lines) + i + 1
                            reward = entry.get("reward", 0.0)
                            kl = entry.get("kl_divergence", 0.0)
                            timestamp = entry.get("timestamp", "N/A")
                            if timestamp != "N/A":
                                timestamp = datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
                            
                            print(f"  {iter_num:<8} {reward:<12.4f} {kl:<12.6f} {timestamp}")
                        except:
                            pass
                    
                    # Calculate averages
                    total_reward = 0.0
                    total_kl = 0.0
                    for line in lines:
                        try:
                            entry = json.loads(line)
                            total_reward += entry.get("reward", 0.0)
                            total_kl += entry.get("kl_divergence", 0.0)
                        except:
                            pass
                    
                    if len(lines) > 0:
                        print(f"\n  Averages:")
                        print(f"  Reward: {total_reward / len(lines):.4f}")
                        print(f"  KL Divergence: {total_kl / len(lines):.6f}")
                    
                    # Show if new data arrived
                    if len(lines) > last_line_count:
                        print(f"\n  [New data: {len(lines) - last_line_count} iterations]")
                    last_line_count = len(lines)
                else:
                    print("  No metrics data yet...")
            else:
                print(f"\nMetrics file not found. Waiting for training to start...")
            
            # Check if completed
            if metadata.get("status") == "completed":
                print("\n" + "=" * 80)
                print("Training completed!")
                print("=" * 80)
                break
            
            print(f"\nLast updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Refreshing in {refresh_interval}s... (Ctrl+C to stop)")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def main():
    parser = argparse.ArgumentParser(description="Monitor StealthRL training runs in real-time")
    parser.add_argument("run_name", help="Name of the run to monitor")
    parser.add_argument("--runs-dir", default="outputs/runs", help="Directory containing training runs")
    parser.add_argument("--interval", type=int, default=5, help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    run_dir = Path(args.runs_dir) / args.run_name
    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return
    
    monitor_run(run_dir, args.interval)


if __name__ == "__main__":
    main()
