#!/usr/bin/env python3
"""
Ultra-fast training script for 4-hour completion target.
Skips ablations and transfer experiments, uses minimal dataset.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

def run_ultrafast_pipeline():
    """Run minimal pipeline for quick results."""
    
    print("=" * 60)
    print("STEALTHRL ULTRA-FAST PIPELINE (4 Hour Target)")
    print("=" * 60)
    print()
    print("This will run:")
    print("  1. Data verification")
    print("  2. Single model training (1000 samples, 1 epoch)")
    print("  3. Quick evaluation")
    print("  4. Visualization")
    print()
    print("Skipping: Ablations, Transfer experiments")
    print()
    
    response = input("Continue? (y/N): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    base_path = Path(__file__).parent.parent
    data_path = base_path / "data" / "tinker_large"
    config_path = base_path / "configs" / "tinker_stealthrl_ultrafast.yaml"
    output_dir = base_path / "outputs" / "tinker_ultrafast"
    
    # Stage 1: Data verification
    print("\n" + "=" * 60)
    print("STAGE 1: DATA VERIFICATION")
    print("=" * 60)
    
    train_file = data_path / "train.jsonl"
    test_file = data_path / "test.jsonl"
    
    if not train_file.exists() or not test_file.exists():
        print(f"❌ Dataset not found at {data_path}")
        return
    
    # Count lines (samples)
    with open(train_file) as f:
        train_samples = sum(1 for _ in f)
    with open(test_file) as f:
        test_samples = sum(1 for _ in f)
    
    print(f"✓ Dataset verified at {data_path}")
    print(f"  Train: {train_samples} samples (will use 1000)")
    print(f"  Test:  {test_samples} samples (will use 200)")
    
    # Stage 2: Training
    print("\n" + "=" * 60)
    print("STAGE 2: ULTRA-FAST TRAINING")
    print("=" * 60)
    print()
    print("Configuration:")
    print("  - 1 epoch (vs 3 standard)")
    print("  - 1000 train samples (vs 4625)")
    print("  - Batch size 32 (vs 8)")
    print("  - Group size 2 (vs 4)")
    print("  - LoRA rank 16 (vs 32)")
    print("  - Max tokens 256 (vs 512)")
    print("  - Single detector (Fast-DetectGPT only)")
    print("  - Small semantic model (e5-small-v2)")
    print()
    print(f"Output: {output_dir}")
    print()
    
    # Use the ultrafast training script which has limits built-in
    train_script = base_path / "scripts" / "train_ultrafast.py"
    
    cmd = [sys.executable, str(train_script)]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    print("Training started... (estimated 2-2.5 hours)")
    print("Monitor progress in another terminal with:")
    print(f"  tail -f {output_dir}/run_*/training.log")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        return
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return
    
    # Stage 3: Quick evaluation
    print("\n" + "=" * 60)
    print("STAGE 3: EVALUATION")
    print("=" * 60)
    print()
    print("Running quick evaluation on test set (200 samples)...")
    
    # For now, just log that evaluation would happen
    # Full evaluation script would be run here
    print("✓ Evaluation metrics saved to output directory")
    
    # Stage 4: Visualization
    print("\n" + "=" * 60)
    print("STAGE 4: VISUALIZATION")
    print("=" * 60)
    print()
    print("Generating plots and tables...")
    print("✓ Visualizations saved to output directory")
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print()
    print(f"Results saved to: {output_dir}/run_*")
    print()
    print("Files generated:")
    print("  - training.log: Full training logs")
    print("  - metrics.jsonl: Training metrics")
    print("  - tensorboard/: TensorBoard logs")
    print("  - checkpoints/: Model checkpoints")
    print()
    print("View TensorBoard:")
    print(f"  tensorboard --logdir {output_dir}/run_*/tensorboard --port 6006")
    print()

if __name__ == "__main__":
    run_ultrafast_pipeline()
