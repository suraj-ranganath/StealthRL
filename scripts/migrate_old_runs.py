#!/usr/bin/env python3
"""
Migrate old run files to proper structure.

IMPORTANT: This issue is now FIXED in train.py (line 512: log_path=str(output_dir))
           Future runs will save all files correctly in their run directory.
           This script is ONLY for migrating old misplaced files.

Old structure (incorrect - from runs before the fix):
  outputs/runs/
    metrics.jsonl/metrics.jsonl  <- wrong: folder with file inside
    eval_test_iteration_*.html   <- wrong: should be in run directory  
    train_iteration_*.html       <- wrong: should be in run directory
    run_20251125_221823/

New structure (correct - after the fix):
  outputs/runs/
    run_20251125_221823/
      metrics.jsonl              <- file directly in run dir
      run_metadata.json
      training.log
      eval_test_iteration_*.html
      train_iteration_*.html
"""

import shutil
from pathlib import Path
import sys

def migrate_runs():
    """Move misplaced metrics.jsonl and HTML files into run directories."""
    
    runs_dir = Path("outputs/runs")
    if not runs_dir.exists():
        print(f"❌ Directory not found: {runs_dir}")
        return
    
    print("🔍 Scanning for misplaced files...")
    
    # Find misplaced metrics.jsonl folder
    metrics_folder = runs_dir / "metrics.jsonl"
    if metrics_folder.is_dir():
        metrics_file = metrics_folder / "metrics.jsonl"
        if metrics_file.exists():
            print(f"📁 Found misplaced metrics.jsonl folder")
            
            # Find the most recent run directory
            run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
            if run_dirs:
                latest_run = sorted(run_dirs)[-1]
                target = latest_run / "metrics.jsonl"
                print(f"   Moving to: {target}")
                shutil.copy2(metrics_file, target)
                print(f"   ✅ Copied metrics.jsonl to {latest_run.name}")
                
                # Don't delete original yet - let user verify
                print(f"   ⚠️  Original folder kept at: {metrics_folder}")
                print(f"   ⚠️  Verify the copy, then manually delete: rm -rf {metrics_folder}")
            else:
                print(f"   ⚠️  No run directories found to move metrics.jsonl into")
    
    # Find misplaced HTML files
    html_files = list(runs_dir.glob("*.html"))
    if html_files:
        print(f"📄 Found {len(html_files)} misplaced HTML files")
        
        # Find the most recent run directory
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if run_dirs:
            latest_run = sorted(run_dirs)[-1]
            print(f"   Moving to: {latest_run}")
            
            for html_file in html_files:
                target = latest_run / html_file.name
                print(f"   - {html_file.name}")
                shutil.copy2(html_file, target)
            
            print(f"   ✅ Copied {len(html_files)} HTML files to {latest_run.name}")
            print(f"   ⚠️  Original files kept in: {runs_dir}")
            print(f"   ⚠️  Verify the copies, then manually delete: rm {runs_dir}/*.html")
        else:
            print(f"   ⚠️  No run directories found to move HTML files into")
    
    if not metrics_folder.exists() and not html_files:
        print("✅ No misplaced files found - structure is correct!")
    else:
        print("\n📝 Summary:")
        print("   1. Check the migrated files in the run directory")
        print("   2. If correct, manually delete the old files:")
        print(f"      rm -rf {metrics_folder}")
        print(f"      rm {runs_dir}/*.html")

if __name__ == "__main__":
    migrate_runs()
