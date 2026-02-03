#!/usr/bin/env python3
"""
StealthRL Research Pipeline Runner.

Automates the full research workflow:
1. Prepare data
2. Train models (full ensemble + transfer setup)
3. Evaluate with baselines (SICO)
4. Run ablations
5. Generate visualizations

Usage:
    python scripts/run_research_pipeline.py --stage all
    python scripts/run_research_pipeline.py --stage train
    python scripts/run_research_pipeline.py --stage evaluate
"""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path


class ResearchPipeline:
    """Automated research pipeline for StealthRL."""
    
    def __init__(self, base_dir: Path, data_dir: Path = None):
        self.base_dir = base_dir
        self.data_dir = data_dir if data_dir else base_dir / "data" / "tinker_large"
        self.output_dir = base_dir / "outputs"
        self.config_dir = base_dir / "configs"
    
    def run_command(self, cmd: list, description: str):
        """Run shell command with error handling."""
        print(f"\n{'='*60}")
        print(f"▶ {description}")
        print(f"{'='*60}")
        print(f"Command: {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, cwd=self.base_dir)
        
        if result.returncode != 0:
            print(f"\n❌ Error in: {description}")
            sys.exit(1)
        
        print(f"\n✓ Completed: {description}")
    
    def stage_data_prep(self):
        """Stage 1: Verify dataset exists."""
        print("\n" + "="*60)
        print("STAGE 1: DATA VERIFICATION")
        print("="*60)
        
        # Check if data exists
        train_file = self.data_dir / "train.jsonl"
        test_file = self.data_dir / "test.jsonl"
        
        if not train_file.exists() or not test_file.exists():
            print(f"\n❌ Error: Dataset not found at {self.data_dir}")
            print("Expected files:")
            print(f"  - {train_file}")
            print(f"  - {test_file}")
            sys.exit(1)
        
        # Count samples
        import subprocess
        train_count = int(subprocess.check_output(["wc", "-l", str(train_file)]).split()[0])
        test_count = int(subprocess.check_output(["wc", "-l", str(test_file)]).split()[0])
        
        print(f"\n✓ Dataset verified at {self.data_dir}")
        print(f"  Train: {train_count} samples")
        print(f"  Test:  {test_count} samples")
    
    def stage_training(self):
        """Stage 2: Train models."""
        print("\n" + "="*60)
        print("STAGE 2: MODEL TRAINING")
        print("="*60)
        
        experiments = [
            {
                "name": "Full Ensemble (All Detectors)",
                "config": "tinker_stealthrl.yaml",
                "output": "tinker_full_ensemble",
            },
            {
                "name": "Transfer Setup (In-Ensemble Only)",
                "config": "tinker_transfer_in_ensemble.yaml",
                "output": "tinker_transfer_in_ensemble",
            },
        ]
        
        for exp in experiments:
            print(f"\n{'─'*60}")
            print(f"Training: {exp['name']}")
            print(f"{'─'*60}")
            
            output_path = self.output_dir / exp['output']
            
            if output_path.exists():
                response = input(f"\nCheckpoint exists at {output_path}. Retrain? (y/N): ")
                if response.lower() != 'y':
                    print("Skipping training.")
                    continue
            
            self.run_command(
                [
                    "python", "-m", "stealthrl.tinker.train",
                    "--config", str(self.config_dir / exp['config']),
                    "--data-path", str(self.data_dir),
                    "--output-dir", str(output_path),
                ],
                f"Training {exp['name']}"
            )
        
        print("\n✓ Model training complete")
    
    def stage_ablations(self):
        """Stage 3: Run ablation experiments."""
        print("\n" + "="*60)
        print("STAGE 3: ABLATION EXPERIMENTS")
        print("="*60)
        
        ablations = [
            "detector_only.yaml",
            "no_fairness.yaml",
            "no_quality.yaml",
            "no_semantic.yaml",
            "single_detector_fast_detectgpt.yaml",
        ]
        
        for ablation_config in ablations:
            config_path = self.config_dir / "ablations" / ablation_config
            ablation_name = ablation_config.replace(".yaml", "")
            output_path = self.output_dir / "ablations" / ablation_name
            
            if not config_path.exists():
                print(f"⚠ Config not found: {config_path}")
                continue
            
            print(f"\n{'─'*60}")
            print(f"Ablation: {ablation_name}")
            print(f"{'─'*60}")
            
            if output_path.exists():
                response = input(f"\nAblation exists at {output_path}. Retrain? (y/N): ")
                if response.lower() != 'y':
                    print("Skipping ablation.")
                    continue
            
            self.run_command(
                [
                    "python", "-m", "stealthrl.tinker.train",
                    "--config", str(config_path),
                    "--data-path", str(self.data_dir),
                    "--output-dir", str(output_path),
                ],
                f"Training ablation: {ablation_name}"
            )
        
        print("\n✓ Ablation experiments complete")
    
    def stage_evaluation(self):
        """Stage 4: Comprehensive evaluation."""
        print("\n" + "="*60)
        print("STAGE 4: EVALUATION")
        print("="*60)
        
        # Evaluate main models
        print("\n▶ Evaluating main experiments...")
        self.run_command(
            [
                "python", "scripts/evaluate_transfer.py",
                "--checkpoints",
                str(self.output_dir / "tinker_full_ensemble"),
                str(self.output_dir / "tinker_transfer_in_ensemble"),
                "--output-dir", str(self.output_dir / "evaluation"),
            ],
            "Transfer evaluation (in-ensemble vs held-out)"
        )
        
        # Evaluate ablations
        print("\n▶ Evaluating ablations...")
        self.run_command(
            [
                "python", "scripts/evaluate_ablations.py",
                "--ablation-dir", str(self.output_dir / "ablations"),
                "--output-dir", str(self.output_dir / "ablation_analysis"),
            ],
            "Ablation analysis"
        )
        
        print("\n✓ Evaluation complete")
    
    def stage_visualization(self):
        """Stage 5: Generate visualizations."""
        print("\n" + "="*60)
        print("STAGE 5: VISUALIZATION")
        print("="*60)
        
        self.run_command(
            [
                "python", "scripts/visualize_stealthbench.py",
                "--results-dir", str(self.output_dir / "evaluation"),
                "--ablation-dir", str(self.output_dir / "ablation_analysis"),
                "--output-dir", str(self.output_dir / "visualizations"),
            ],
            "Generating StealthBench visualizations"
        )
        
        print("\n✓ Visualizations complete")
        print(f"\n📊 Results available at:")
        print(f"  - Evaluation: {self.output_dir}/evaluation/")
        print(f"  - Visualizations: {self.output_dir}/visualizations/")
        print(f"  - Ablations: {self.output_dir}/ablation_analysis/")
    
    def run_full_pipeline(self):
        """Run complete research pipeline."""
        print("\n" + "="*60)
        print("STEALTHRL RESEARCH PIPELINE")
        print("="*60)
        print("\nThis will run the complete research pipeline:")
        print("  1. Data preparation")
        print("  2. Model training (full + transfer)")
        print("  3. Ablation experiments")
        print("  4. Comprehensive evaluation")
        print("  5. Visualization generation")
        print("\nThis may take several hours depending on compute.")
        
        response = input("\nContinue? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        
        self.stage_data_prep()
        self.stage_training()
        self.stage_ablations()
        self.stage_evaluation()
        self.stage_visualization()
        
        print("\n" + "="*60)
        print("✓ RESEARCH PIPELINE COMPLETE")
        print("="*60)
        print("\n📊 Summary:")
        print(f"  - Trained models: {self.output_dir}/")
        print(f"  - Evaluation results: {self.output_dir}/evaluation/")
        print(f"  - Visualizations: {self.output_dir}/visualizations/")
        print("\n🎓 Next steps:")
        print("  - Review evaluation reports")
        print("  - Analyze Pareto frontier plots")
        print("  - Draft research findings")


def main():
    parser = argparse.ArgumentParser(
        description="StealthRL Research Pipeline Runner"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "data", "train", "ablations", "evaluate", "visualize"],
        default="all",
        help="Pipeline stage to run (default: all)"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="Base directory of StealthRL project"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory (default: base_dir/data/tinker_large)"
    )
    
    args = parser.parse_args()
    
    data_dir = args.data_dir if args.data_dir else args.base_dir / "data" / "tinker_large"
    pipeline = ResearchPipeline(args.base_dir, data_dir)
    
    if args.stage == "all":
        pipeline.run_full_pipeline()
    elif args.stage == "data":
        pipeline.stage_data_prep()
    elif args.stage == "train":
        pipeline.stage_training()
    elif args.stage == "ablations":
        pipeline.stage_ablations()
    elif args.stage == "evaluate":
        pipeline.stage_evaluation()
    elif args.stage == "visualize":
        pipeline.stage_visualization()


if __name__ == "__main__":
    main()
