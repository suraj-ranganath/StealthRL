#!/bin/bash
# Run all ablation experiments

set -e  # Exit on error

ABLATION_DIR="configs/ablations"
OUTPUT_BASE="outputs/ablations"

echo "=========================================="
echo "Running StealthRL Ablation Studies"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Function to run a single ablation
run_ablation() {
    local config=$1
    local name=$(basename "$config" .yaml)
    
    echo ""
    echo "=== Running ablation: $name ==="
    echo "Config: $config"
    
    python scripts/train_stealthrl.py \
        --config "$config" \
        2>&1 | tee "$OUTPUT_BASE/${name}_train.log"
    
    echo "✓ Completed: $name"
}

# Check if ablation configs exist
if [ ! -d "$ABLATION_DIR" ]; then
    echo "ERROR: Ablation config directory not found: $ABLATION_DIR"
    exit 1
fi

# Run each ablation
echo ""
echo "Found ablation configs:"
ls -1 "$ABLATION_DIR"/*.yaml

# 1. Single detector (Fast-DetectGPT only)
if [ -f "$ABLATION_DIR/single_detector_fast_detectgpt.yaml" ]; then
    run_ablation "$ABLATION_DIR/single_detector_fast_detectgpt.yaml"
fi

# 2. No fairness penalty
if [ -f "$ABLATION_DIR/no_fairness.yaml" ]; then
    run_ablation "$ABLATION_DIR/no_fairness.yaml"
fi

# 3. No semantic fidelity
if [ -f "$ABLATION_DIR/no_semantic.yaml" ]; then
    run_ablation "$ABLATION_DIR/no_semantic.yaml"
fi

# 4. No quality constraints
if [ -f "$ABLATION_DIR/no_quality.yaml" ]; then
    run_ablation "$ABLATION_DIR/no_quality.yaml"
fi

# 5. Detector reward only
if [ -f "$ABLATION_DIR/detector_only.yaml" ]; then
    run_ablation "$ABLATION_DIR/detector_only.yaml"
fi

echo ""
echo "=========================================="
echo "All ablations complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "To evaluate ablations, run:"
echo "  python scripts/evaluate_ablations.py --ablation_dir $OUTPUT_BASE"
