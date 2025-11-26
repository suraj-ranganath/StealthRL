# Checkpoint Management Guide

This guide explains how checkpoints work in StealthRL and how to access your trained models.

## Overview

StealthRL uses Tinker's checkpoint system, which stores model checkpoints **remotely on Tinker's servers**, not as local files. Checkpoints are accessed via `tinker://` URIs rather than file paths.

## Types of Checkpoints

Tinker provides two types of checkpoints:

### 1. Sampler Weights (for inference)
- **Purpose**: Load the model for inference/sampling
- **Method**: `training_client.save_weights_for_sampler(name="checkpoint_name")`
- **Contents**: Model weights only (no optimizer state)
- **Size**: Smaller, faster to save/load
- **Use Case**: When you want to test or deploy your trained model

### 2. Full State (for resuming training)
- **Purpose**: Resume training from a checkpoint
- **Method**: `training_client.save_state(name="checkpoint_name")`
- **Contents**: Model weights + optimizer state + training metadata
- **Size**: Larger (includes momentum, learning rate schedules, etc.)
- **Use Case**: When you want to continue training later or try different hyperparameters

## How Checkpoints are Saved

### Automatic Saving During Training

The `StealthRLConfig` has two parameters that control checkpoint frequency:

```python
save_every: int = 10  # Save checkpoint every N iterations
eval_every: int = 5   # Run evaluation every N iterations
```

**Recommendations for `save_every`:**
- **Short runs (< 20 iterations)**: Use `save_every=5` to checkpoint frequently
- **Medium runs (20-100 iterations)**: Use `save_every=10` (default)
- **Long runs (> 100 iterations)**: Use `save_every=20` to reduce overhead

**Recommendations for `eval_every`:**
- **Testing/debugging**: Use `eval_every=1` to monitor every iteration
- **Normal training**: Use `eval_every=5` (default)
- **Production runs**: Use `eval_every=10` to reduce evaluation overhead

### Final Checkpoint

At the end of training, StealthRL automatically saves:
1. A final **full state** checkpoint (for resuming): `tinker://<model_id>/final`
2. A final **sampler weights** checkpoint (for inference): `tinker://<model_id>/final_sampler`
3. A local **checkpoint info file**: `outputs/runs/<run_name>/checkpoints/final_checkpoint_info.json`

## Accessing Trained Models

### Method 1: Using the Export Script

```bash
python scripts/export_model.py outputs/runs/my_training_run
```

This will display:
- Model ID and base model info
- Checkpoint paths (tinker:// URIs)
- Code examples for loading the model

You can also generate a ready-to-use inference script:

```bash
python scripts/export_model.py outputs/runs/my_training_run --generate-script
```

This creates `outputs/runs/my_training_run/inference.py` that you can run directly.

### Method 2: Manual Loading for Inference

```python
import tinker

# Initialize service client
service_client = tinker.ServiceClient()

# Load your trained model using the sampler weights path
sampling_client = service_client.create_sampling_client(
    model_path="tinker://<model_id>/final_sampler"  # Get this from checkpoint_info.json
)

# Now you can sample from the model
# ... your inference code here ...
```

### Method 3: Resuming Training

```python
import tinker

# Initialize service client  
service_client = tinker.ServiceClient()

# Create a training client with the same base model and LoRA rank
training_client = await service_client.create_lora_training_client_async(
    base_model="meta-llama/Llama-3.1-8B",  # Must match original
    rank=32  # Must match original
)

# Load the full state checkpoint
training_client.load_state("tinker://<model_id>/final")  # Get this from checkpoint_info.json

# Continue training...
# ... your training code here ...
```

## Checkpoint Storage Location

### Remote Storage (Tinker Servers)
- **Location**: Tinker's cloud infrastructure
- **Format**: `tinker://<model_id>/<checkpoint_name>`
- **Access**: Via Tinker API using `ServiceClient`
- **Persistence**: Checkpoints remain available as long as you have access to Tinker

### Local Metadata
- **Location**: `outputs/runs/<run_name>/checkpoints/`
- **Files**:
  - `final_checkpoint_info.json` - Contains model_id and checkpoint paths
  - `checkpoints.jsonl` - Log of all checkpoints saved during training
- **Purpose**: Provides easy access to checkpoint URIs and model metadata

## Checkpoint Info File Format

The `final_checkpoint_info.json` file contains:

```json
{
  "model_id": "tinker_generated_model_id",
  "base_model": "meta-llama/Llama-3.1-8B",
  "lora_rank": 32,
  "checkpoints": {
    "final_state": "tinker://<model_id>/final",
    "sampler_weights": "tinker://<model_id>/final_sampler"
  },
  "usage": {
    "load_for_training": "training_client.load_state('tinker://...')",
    "load_for_inference": "service_client.create_sampling_client(model_path='tinker://...')"
  }
}
```

## Best Practices

### 1. Always Save Final Checkpoints
The training script automatically saves final checkpoints. Don't interrupt training before it completes unless you've set up intermediate checkpoints.

### 2. Use Appropriate `save_every` Values
- **Too frequent** (e.g., save_every=1): Slows down training
- **Too infrequent** (e.g., save_every=100): Risk losing progress if training fails
- **Recommended**: save_every=10 for most runs

### 3. Keep Local Checkpoint Info Files
The `checkpoints/` directory in your run folder is small but crucial for accessing your models later. Back it up or commit it to git.

### 4. Document Your Model IDs
When running multiple experiments, keep track of which `model_id` corresponds to which experiment. The checkpoint info file helps with this.

### 5. Test Model Loading Early
After your first training run completes, test that you can load the model using the export script or manual loading code. This ensures your checkpoint access pipeline works.

## Troubleshooting

### "Checkpoint info not found"
- **Cause**: Training didn't complete or crashed before saving final checkpoints
- **Solution**: Re-run training or check intermediate checkpoints in `checkpoints.jsonl`

### "Cannot access tinker:// URI"
- **Cause**: Invalid model_id or insufficient Tinker permissions
- **Solution**: Check your TINKER_API_KEY and verify the model_id is correct

### "Model ID mismatch"
- **Cause**: Trying to load checkpoint with wrong base model or LoRA rank
- **Solution**: Use the same `base_model` and `rank` as shown in `checkpoint_info.json`

### Checkpoints not saving during training
- **Cause**: `save_every` value too high, or training ended before first checkpoint
- **Solution**: Reduce `save_every` value or run for more iterations

## Additional Resources

- **Tinker Documentation**: https://tinker-docs.thinkingmachines.ai/
- **Saving and Loading Guide**: https://tinker-docs.thinkingmachines.ai/save-load
- **StealthRL Training Guide**: See [`SETUP_AND_RUN.md`](SETUP_AND_RUN.md)
- **Tinker Platform Guide**: See [`TINKER_README.md`](TINKER_README.md)
- **Quick Start Guide**: See [`QUICKSTART.md`](QUICKSTART.md)
