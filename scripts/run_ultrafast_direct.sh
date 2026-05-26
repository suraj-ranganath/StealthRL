#!/bin/bash
# Direct ultrafast training without tmux
# Usage: ./scripts/run_ultrafast_direct.sh

cd "$(dirname "$0")/.."

echo "==============================================================="
echo "ULTRA-FAST STEALTHRL TRAINING (Direct Execution)"
echo "==============================================================="
echo ""
echo "This will run training directly in this terminal."
echo "Press Ctrl+C to stop."
echo ""
echo "Expected duration: ~2.5 hours"
echo "==============================================================="
echo ""

# Activate venv if not already activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run training with caffeinate to prevent sleep
caffeinate -i python scripts/train_ultrafast.py

echo ""
echo "==============================================================="
echo "Training completed or interrupted"
echo "==============================================================="
