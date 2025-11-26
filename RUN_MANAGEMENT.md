# StealthRL Run Management

## Overview
Training runs are now automatically organized in timestamped directories with comprehensive tracking and monitoring capabilities.

## Quick Start

### Start a Training Run
```bash
# Activate virtual environment first
source venv/bin/activate

# Basic run (auto-generates timestamp)
python -m stealthrl.tinker.train --data-path data/tinker --num-epochs 1 --batch-size 4

# Named run
python -m stealthrl.tinker.train --data-path data/tinker --run-name my_experiment --num-epochs 3 --batch-size 8

# Custom output location
python -m stealthrl.tinker.train --data-path data/tinker --output-dir outputs/experiments --num-epochs 2

# All output (stdout/stderr) is automatically captured to training.log in the run directory
```

### Track All Runs
```bash
# List all runs with metrics
python scripts/track_runs.py --list

# Show detailed info for a specific run
python scripts/track_runs.py --details run_20240115_143022

# Compare multiple runs
python scripts/track_runs.py --compare run_20240115_143022 run_20240115_150130

# Export full report to JSON
python scripts/track_runs.py --export my_report.json

# Show cost information (placeholder for Tinker API integration)
python scripts/track_runs.py --cost
```

### Monitor a Run in Real-Time
```bash
# Monitor with default 5-second refresh
python scripts/monitor_runs.py run_20240115_143022

# Custom refresh interval
python scripts/monitor_runs.py run_20240115_143022 --interval 10
```

## Directory Structure

Each run creates a timestamped directory:
```
outputs/runs/
├── run_20240115_143022/
│   ├── run_metadata.json     # Run configuration and status
│   ├── metrics.jsonl         # Training metrics (reward, KL, etc.)
│   ├── training.log          # Complete training log (all console output)
│   └── [checkpoints]         # Model checkpoints (if enabled)
├── run_20240115_150130/
│   ├── run_metadata.json
│   ├── metrics.jsonl
│   ├── training.log
│   └── ...
```

### Run Metadata Format
```json
{
  "run_name": "run_20240115_143022",
  "start_time": "2024-01-15T14:30:22",
  "end_time": "2024-01-15T14:45:10",
  "status": "completed",
  "data_path": "data/tinker",
  "num_epochs": 3,
  "batch_size": 8,
  "config_file": null
}
```

## Command-Line Arguments

### Training Script (`stealthrl.tinker.train`)
- `--config`: Path to config YAML file (optional)
- `--data-path`: Path to training data (default: `data/tinker`)
- `--output-dir`: Base output directory (default: `outputs/runs`)
- `--num-epochs`: Number of training epochs (default: 3)
- `--batch-size`: Batch size (default: 8)
- `--run-name`: Custom run name (default: auto-generated `run_YYYYMMDD_HHMMSS`)

### Tracking Script (`scripts/track_runs.py`)
- `--runs-dir`: Directory containing runs (default: `outputs/runs`)
- `--list`: List all runs in a table
- `--details RUN_NAME`: Show detailed information for a run
- `--compare RUN1 RUN2 ...`: Compare multiple runs
- `--export FILE`: Export report to JSON
- `--cost`: Show cost/credit information

### Monitoring Script (`scripts/monitor_runs.py`)
- `run_name`: Name of the run to monitor (required)
- `--runs-dir`: Directory containing runs (default: `outputs/runs`)
- `--interval`: Refresh interval in seconds (default: 5)

## Run Tracking Features

### List Runs Table
Shows all runs with:
- Run name
- Status (running/completed/failed)
- Start time
- Duration
- Configuration (epochs, batch size)
- Average metrics (reward, KL divergence)

### Real-Time Monitoring
The monitor displays:
- Run status and runtime
- Configuration details
- Recent training iterations (last 5)
- Average metrics across all iterations
- Live updates with configurable refresh interval

### Run Comparison
Compare metrics across multiple runs to:
- Identify best hyperparameters
- Track experiment progression
- Analyze training stability

## Cost Tracking (TODO)

The cost tracking feature is a placeholder. To implement:

1. **Use Tinker RestClient API**:
   ```python
   from tinker_cookbook.rest import RestClient
   client = RestClient()
   # Query billing/usage endpoints
   ```

2. **Endpoints to implement**:
   - Get credit balance
   - Get usage history
   - Calculate cost per run
   - Track API call counts

3. **Integration points**:
   - `RunTracker.get_cost_info()` in `scripts/track_runs.py`
   - Add cost tracking to run metadata
   - Real-time cost display in monitor

## Example Workflow

```bash
# 1. Start a training run
python -m stealthrl.tinker.train \
  --data-path data/tinker \
  --run-name experiment_001 \
  --num-epochs 3 \
  --batch-size 4

# 2. Monitor in another terminal
python scripts/monitor_runs.py experiment_001

# 3. After completion, view all runs
python scripts/track_runs.py --list

# 4. Compare with other runs
python scripts/track_runs.py --compare experiment_001 experiment_002

# 5. Export detailed report
python scripts/track_runs.py --export results.json
```

## Troubleshooting

### RuntimeWarning about sys.modules
**Fixed**: Removed train imports from `__init__.py` to prevent circular imports when running as `python -m stealthrl.tinker.train`.

### Runs not showing up
- Check that `outputs/runs/` directory exists
- Verify `run_metadata.json` exists in each run directory
- Make sure training completed successfully

### Monitor not updating
- Verify the run is actually running
- Check that `metrics.jsonl` is being written
- Try increasing `--interval` if file system is slow

### Cost tracking shows N/A
This feature requires Tinker API integration. See "Cost Tracking (TODO)" section above.

## Files Modified

1. `stealthrl/tinker/__init__.py` - Removed train imports
2. `stealthrl/tinker/train.py` - Added run management and timestamping
3. `scripts/track_runs.py` - New run tracking script
4. `scripts/monitor_runs.py` - New real-time monitoring script

## Next Steps

1. Implement actual Tinker API cost tracking
2. Add visualization for training curves
3. Create automated hyperparameter search
4. Implement run archiving and cleanup tools
5. Add email/Slack notifications for run completion
