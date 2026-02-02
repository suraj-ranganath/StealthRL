#!/usr/bin/env python3
"""
Generate config files for reward weight ablation study.

This creates a grid of (detector_weight, semantic_weight) combinations
to explore the Pareto frontier of detector evasion vs semantic fidelity.
"""

import yaml
from pathlib import Path
from typing import List, Tuple

# Base config to modify
BASE_CONFIG = "configs/tinker_mage_10k.yaml"
OUTPUT_DIR = Path("configs/ablations")

# Weight combinations to test (detector_weight, semantic_weight)
# Simplified ablation: fix detector=1.0, sweep semantic weight
WEIGHT_GRID: List[Tuple[float, float]] = [
    (1.0, 0.0),  # Pure evasion
    (1.0, 0.1),  # Slight fidelity constraint
    (1.0, 0.2),  # Current best run baseline
    (1.0, 0.3),  # Balanced
]

# Fast ablation settings (for quick iteration)
ABLATION_OVERRIDES = {
    "dataset": {
        "max_train_examples": 1000,  # 1K samples for speed
        "max_test_examples": 100,     # More test samples for reliable eval
    },
    "training": {
        "learning_rate": 2e-5,  # Match successful run (scaled to 2e-4 after 10x)
        "num_epochs": 3,  # Fewer epochs
        "batch_size": 16,
        "group_size": 8,
    },
    "logging": {
        "path": "outputs/ablations",
        "eval_interval": 25,  # More frequent eval
        "save_interval": 100,
    },
    "parallel": {
        "mode": "sync",  # Match successful run (sync mode)
    },
}


def load_base_config(config_path: str) -> dict:
    """Load base configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_ablation_config(
    base_config: dict,
    detector_weight: float,
    semantic_weight: float,
    output_path: Path,
) -> None:
    """Create ablation config with specified weights."""
    config = base_config.copy()
    
    # Update reward weights
    config["reward"]["detector_weight"] = detector_weight
    config["reward"]["semantic_weight"] = semantic_weight
    
    # Apply ablation overrides
    for key, value in ABLATION_OVERRIDES.items():
        if isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    
    # Update output path
    config["logging"]["path"] = f"outputs/ablations/det{detector_weight:.1f}_sem{semantic_weight:.1f}"
    
    # Add ablation metadata
    config["ablation"] = {
        "detector_weight": detector_weight,
        "semantic_weight": semantic_weight,
        "study": "reward_weights",
    }
    
    # Save config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created: {output_path}")


def main():
    """Generate all ablation configs."""
    print("Generating reward weight ablation configs...")
    print(f"Base config: {BASE_CONFIG}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Grid size: {len(WEIGHT_GRID)} combinations")
    print()
    
    # Load base config
    base_config = load_base_config(BASE_CONFIG)
    
    # Generate configs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for detector_weight, semantic_weight in WEIGHT_GRID:
        config_name = f"det{detector_weight:.1f}_sem{semantic_weight:.1f}.yaml"
        output_path = OUTPUT_DIR / config_name
        
        create_ablation_config(
            base_config=base_config,
            detector_weight=detector_weight,
            semantic_weight=semantic_weight,
            output_path=output_path,
        )
    
    print()
    print(f"✓ Generated {len(WEIGHT_GRID)} configs in {OUTPUT_DIR}")
    print()
    print("Next steps:")
    print("  1. Run: python scripts/run_ablation_study.py")
    print("  2. Analyze: python scripts/analyze_ablation_results.py")


if __name__ == "__main__":
    main()
