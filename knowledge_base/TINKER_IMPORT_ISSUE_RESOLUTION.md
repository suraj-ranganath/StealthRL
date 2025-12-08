# Tinker Import Issue Resolution

## Issue Summary

Training pipeline failed with:
```
ModuleNotFoundError: No module named 'tinker'
  File: stealthrl/tinker/env.py, line 13
```

Despite tinker SDK being correctly installed (verified via `pip list | grep tinker`).

## Root Cause

**Python package name collision**: The directory `stealthrl/tinker/` creates a Python package named `tinker`, which shadows the actual Tinker SDK package.

When Python executes:
```python
# Inside stealthrl/tinker/env.py
import tinker  # Line 13
```

Python's import resolution:
1. Checks current package: `stealthrl.tinker` (FOUND - local package)
2. Tries to import from local package instead of installed SDK
3. Fails because local package doesn't have the SDK modules

## Solution

**Rename the directory** to avoid conflict with Tinker SDK:

```bash
# Option 1: Rename to 'training'
mv stealthrl/tinker stealthrl/training

# Option 2: Rename to 'rl_training'  
mv stealthrl/tinker stealthrl/rl_training

# Option 3: Rename to 'tinker_env' (keep tinker reference but avoid exact collision)
mv stealthrl/tinker stealthrl/tinker_env
```

**Recommended**: Use `stealthrl/training/` for clarity and to avoid any Tinker SDK naming conflicts.

## Files to Update After Rename

After renaming `stealthrl/tinker/` → `stealthrl/training/`, update imports in:

1. **scripts/run_research_pipeline.py**:
   ```python
   # Change:
   from stealthrl.tinker.train import train_stealthrl
   # To:
   from stealthrl.training.train import train_stealthrl
   ```

2. **Any other scripts** that import from stealthrl.tinker:
   ```bash
   # Find all references
   grep -r "from stealthrl.tinker" .
   grep -r "import stealthrl.tinker" .
   ```

3. **Update __init__.py** if it exists:
   ```bash
   # Check for references
   cat stealthrl/__init__.py
   ```

## Verification Steps

After renaming and updating imports:

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Test import
python -c "import tinker; print(tinker.__version__)"
# Expected output: 0.5.1

# 3. Test training module import
python -c "from stealthrl.training.env import StealthEnv; print('Success')"
# Expected output: Success

# 4. Run pipeline
python scripts/run_research_pipeline.py --stage all
```

## Related Findings from Tinker Documentation

### Training Run Management
- **NO SDK API exists** to list or cancel training runs
- Must use **Tinker web console/dashboard** for run management
- ServiceClient does NOT have `list_runs()` or `cancel_run()` methods
- Previous attempt to create `cancel_tinker_runs.py` script was non-functional (deleted)

### LoRA Best Practices for StealthRL Training
Based on Tinker documentation review:

1. **Learning Rate**:
   - LoRA requires **10x higher LR** than full fine-tuning
   - For Llama-3.1-8B with LoRA: **recommended LR ≈ 2.8e-4**
   - Formula: `LR = 5e-5 * 10 * (2000/hidden_size)^0.781` for Llama models

2. **LoRA Configuration**:
   - Apply LoRA to **ALL layers** (MLP + attention) for best performance
   - Attention-only LoRA significantly underperforms
   - Default rank=32 is optimal for RL post-training
   - Rank does NOT affect optimal learning rate

3. **Training Dynamics**:
   - LoRA matches full fine-tuning performance in RL scenarios
   - Even rank=1 performs well for RL (information-theoretic argument)
   - Use `importance_sampling` loss for policy gradient RL
   - Monitor KL divergence (should stay < 0.01 for stability)

4. **Hyperparameters**:
   - Batch size scaling: `LR ∝ sqrt(batch_size)`
   - Use small batch sizes (128-256) for better generalization
   - Default Adam params: β1=0.9, β2=0.95, ε=1e-8
   - Initialize: A ~ Uniform(-1/d, 1/d), B = 0, α=32

## Action Checklist

- [ ] **MANUAL**: Check Tinker web console to cancel any active training runs
- [ ] Rename `stealthrl/tinker/` directory (recommended: `stealthrl/training/`)
- [ ] Update all import statements in scripts and modules
- [ ] Verify imports work: `python -c "import tinker; from stealthrl.training.env import StealthEnv"`
- [ ] Review learning rate in training config (ensure ~2.8e-4 for Llama-3.1-8B with LoRA)
- [ ] Run pipeline: `python scripts/run_research_pipeline.py --stage all`
- [ ] Monitor training (estimated 6-8 hours for full pipeline)

## Prevention

To avoid similar issues in future:
1. Never name local packages after well-known PyPI packages
2. Use more specific names: `myproject_training` instead of generic names
3. Test imports in isolated environment before full pipeline execution

## Managing Tinker Training Runs (RestClient API)

### Listing and Monitoring Runs

The Tinker SDK provides a **RestClient** API (accessed via `ServiceClient.create_rest_client()`) for managing training runs programmatically:

```python
import tinker

service_client = tinker.ServiceClient()
rest_client = service_client.create_rest_client()

# List all training runs
response = await rest_client.list_training_runs_async()
runs = response.training_runs  # List of TrainingRun objects

# Get details for a specific run
run = await rest_client.get_training_run_async(run_id)

# List active sessions
sessions = await rest_client.list_sessions_async()
```

### Available RestClient Methods

- `list_training_runs()` / `list_training_runs_async()` - List all training runs
- `get_training_run(run_id)` / `get_training_run_async()` - Get specific run details
- `list_sessions()` / `list_sessions_async()` - List active training sessions
- `list_checkpoints(session_id)` / `list_checkpoints_async()` - List checkpoints for a session
- `delete_checkpoint(checkpoint_id)` / `delete_checkpoint_async()` - Delete a checkpoint

### Using the cancel_tinker_runs.py Script

We've created a script to list and monitor Tinker training runs:

```bash
# List all training runs
python scripts/cancel_tinker_runs.py --list

# List active sessions
python scripts/cancel_tinker_runs.py --list-sessions

# Show all runs and sessions
python scripts/cancel_tinker_runs.py --cancel-all
```

The script will:
- Show all training runs with status (🟢 ACTIVE / 🟡 RECENT / 🔵 IDLE)
- Highlight runs that have been active in the last 30 minutes
- Display checkpoint information and last request time

### Stopping Training Runs

**Important**: The Tinker RestClient API does not have a direct method to cancel/stop active training runs. To stop a training run:

1. **Stop the training script**: Press `Ctrl+C` in the terminal running the training
2. **Tinker handles cleanup**: The API automatically cleans up resources when client disconnects
3. **For stuck runs**: Contact Tinker support if a run doesn't stop properly

### TrainingRun Object Structure

```python
class TrainingRun:
    training_run_id: str              # Unique identifier (e.g., "3180e577-...:train:0")
    base_model: str                    # Base model name (e.g., "Qwen/Qwen3-4B-Instruct-2507")
    is_lora: bool                      # Whether using LoRA fine-tuning
    lora_rank: int                     # LoRA rank (if applicable)
    last_request_time: datetime        # Last API request time (UTC)
    corrupted: bool                    # Whether run is corrupted
    last_checkpoint: Optional[Checkpoint]        # Latest training checkpoint
    last_sampler_checkpoint: Optional[Checkpoint]  # Latest sampler checkpoint
```

## References

- Tinker Documentation: https://tinker-docs.thinkingmachines.ai/llms-full.txt
- Tinker Cookbook: https://github.com/thinking-machines-lab/tinker-cookbook
- LoRA Without Regret: https://thinkingmachines.ai/blog/lora/
- Python Import System: https://docs.python.org/3/reference/import.html
