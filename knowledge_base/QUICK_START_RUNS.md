# Quick Start: Running and Tracking StealthRL Training

## Setup (One-Time)
```bash
cd /Users/suraj/Desktop/StealthRL
source venv/bin/activate
```

## Start a Training Run

### Basic Command
```bash
python -m stealthrl.tinker.train \
  --data-path data/tinker \
  --num-epochs 1 \
  --batch-size 4
```

This creates a timestamped run directory like: `outputs/runs/run_20241125_143022/`

### Named Run
```bash
python -m stealthrl.tinker.train \
  --data-path data/tinker \
  --run-name my_experiment \
  --num-epochs 3 \
  --batch-size 8
```

This creates: `outputs/runs/my_experiment/`

## What Gets Saved

Each run directory contains:
- **`run_metadata.json`** - Configuration, timestamps, status
- **`metrics.jsonl`** - Training metrics (reward, KL divergence, etc.) - one file per run
- **`training.log`** - Complete training log (all console output)

All files for a run are kept together in the same directory.

## View All Runs

```bash
python scripts/track_runs.py --list
```

Example output:
```
Run Name                       Status       Start Time           Duration     Epochs   Batch    Avg Reward  
====================================================================================================
run_20241125_143022            completed    2024-11-25 14:30:22  12m 45s      3        8        0.7234      
my_experiment                  running      2024-11-25 15:10:00  N/A          1        4        N/A         
====================================================================================================
```

## Monitor Live Training

In a separate terminal:
```bash
source venv/bin/activate
python scripts/monitor_runs.py run_20241125_143022
```

This shows real-time updates every 5 seconds.

## View Training Logs

```bash
# View the complete log
less outputs/runs/run_20241125_143022/training.log

# Follow log in real-time (while training)
tail -f outputs/runs/run_20241125_143022/training.log

# Search for errors
grep -i error outputs/runs/run_20241125_143022/training.log
```

## Compare Multiple Runs

```bash
python scripts/track_runs.py --compare run_20241125_143022 my_experiment
```

## Stop/Terminate a Training Run

### Option 1: Keyboard Interrupt (Local)
Press `Ctrl+C` in the terminal where training is running. This stops the local script but **does not unload the model from Tinker servers**.

### Option 2: Unload Model on Tinker (Recommended)
To properly terminate the training and free up Tinker resources:

```python
import tinker
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def stop_training():
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    
    # List all your training runs
    runs = await rest_client.list_training_runs_async()
    
    for run in runs:
        print(f"Training Run ID: {run.training_run_id}")
        print(f"Base Model: {run.base_model}")
        print(f"Last Active: {run.last_request_time}")
        print("---")
        
        # Unload the model to stop training (uncomment when ready)
        # await rest_client.unload_model_async(model_id=run.training_run_id)

asyncio.run(stop_training())
```

Or use the command line:
```bash
python -c "
import tinker
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def main():
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    runs = await rest_client.list_training_runs_async()
    
    if runs:
        run_id = runs[0].training_run_id
        print(f'Unloading model: {run_id}')
        await rest_client.unload_model_async(model_id=run_id)
        print('Training terminated on Tinker')
    else:
        print('No active training runs found')

asyncio.run(main())
"
```

## Tips

1. **Always activate venv**: `source venv/bin/activate` before running
2. **Use named runs**: Use `--run-name` for experiments you want to track
3. **Monitor in real-time**: Open a second terminal to watch progress
4. **Check logs**: If something fails, look at `training.log` in the run directory
5. **All runs tracked**: Nothing overwrites previous runs
6. **Properly terminate**: Use `unload_model_async()` to free Tinker resources when stopping training
