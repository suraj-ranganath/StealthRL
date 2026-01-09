#!/usr/bin/env python3
"""
Ultra-fast training script for StealthRL using Tinker.
Same as train.py but with speed optimizations for 4-hour completion.

Speed optimizations applied:
- 1 epoch (vs 3)
- 1000 train samples (vs 4625)
- 200 test samples (vs 1157)
- Batch size 32 (vs 8)
- LoRA rank 16 (maintained)
- Max tokens 256 (vs 512)
- Eval every 50 steps (vs 5)
- Group size 4 (maintained for variance)
"""

import asyncio
import sys
import logging
import os
import json
import yaml
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tinker
from stealthrl.tinker.train import StealthRLTrainer, StealthRLConfig
from stealthrl.tinker.dataset import StealthRLDatasetBuilder

logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


async def main():
    """
    Main entry point for ultra-fast StealthRL training.
    
    Uses the same structure as train.py but with speed optimizations.
    """
    
    print("=" * 70)
    print("ULTRA-FAST STEALTHRL TRAINING (4-Hour Target)")
    print("=" * 70)
    print()
    print("Speed Optimizations Applied:")
    print("  ✓ 1 epoch (3x faster)")
    print("  ✓ 1000 train samples (4.6x faster)")
    print("  ✓ 200 test samples")
    print("  ✓ Batch size 32 (4x faster)")
    print("  ✓ Max tokens 256 (2x faster)")
    print("  ✓ Eval every 50 steps (10x less overhead)")
    print("  ✓ Group size 4 (for reward variance)")
    print()
    print("Expected Duration: ~2.5-3 hours")
    print("=" * 70)
    print()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Setup paths
    base_path = Path(__file__).parent.parent
    config_file = base_path / "configs" / "tinker_stealthrl_ultrafast.yaml"
    
    # Load YAML config
    print(f"Loading config from: {config_file}")
    yaml_config = load_config(config_file)
    print("✓ Config loaded")
    print()
    
    data_path = base_path / "data" / "tinker_large"
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    
    base_output_dir = base_path / "outputs" / "tinker_ultrafast"
    output_dir = base_output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save run metadata
    run_metadata = {
        "run_name": run_name,
        "start_time": datetime.now().isoformat(),
        "data_path": str(data_path),
        "num_epochs": 1,  # ULTRAFAST: 1 epoch
        "batch_size": 32,  # ULTRAFAST: Large batch
        "max_train_examples": 1000,  # ULTRAFAST: Limited dataset
        "max_test_examples": 200,  # ULTRAFAST: Limited test set
        "config": "ultrafast",
        "status": "running",
        "optimizations": [
            "1_epoch",
            "1000_train_samples",
            "200_test_samples",
            "batch_size_32",
            "max_tokens_256",
            "eval_every_50",
            "lora_rank_16"
        ]
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)
    
    # Set up logging to file
    log_file = output_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Add file handler to root logger to capture all logs
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    logger.info(f"Starting run: {run_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Training log: {log_file}")
    
    # Get Tinker API key
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        raise ValueError("TINKER_API_KEY environment variable not set. Please add it to .env file.")
    
    # Initialize Tinker clients
    print("Initializing Tinker clients...")
    logger.info("Initializing Tinker clients...")
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        rank=16  # Standard rank
    )
    print("✓ Tinker clients initialized")
    print()
    
    # Extract config values with defaults (ensure proper types)
    train_config = yaml_config.get('training', {})
    dataset_config = yaml_config.get('dataset', {})
    reward_config = yaml_config.get('reward', {})
    model_config = yaml_config.get('model', {})
    
    batch_size = int(train_config.get('batch_size', 32))
    group_size = int(train_config.get('group_size', 4))
    max_train = int(dataset_config.get('max_train_examples', 1000))
    max_test = int(dataset_config.get('max_test_examples', 200))
    
    # Create dataset builder with config from YAML
    print(f"Setting up dataset ({max_train} train, {max_test} test)...")
    logger.info(f"Setting up dataset builder: batch_size={batch_size}, group_size={group_size}")
    
    # Build reward config from YAML (matching TinkerCompositeReward __init__ params)
    reward_cfg = {
        "detector_weight": float(reward_config.get('detector_weight', 1.0)),
        "semantic_weight": float(reward_config.get('semantic_weight', 1.0)),
        "perplexity_weight": float(reward_config.get('perplexity_weight', 0.5)),
        "fairness_weight": float(reward_config.get('fairness_weight', 0.2)),
    }
    
    # Add detector config from YAML (flat structure for TinkerCompositeReward)
    if 'detectors' in reward_config:
        reward_cfg['detector_names'] = reward_config['detectors'].get('names', ['fast_detectgpt'])
        reward_cfg['detector_weights'] = reward_config['detectors'].get('weights', {'fast_detectgpt': 1.0})
        if 'cache_path' in reward_config['detectors']:
            reward_cfg['detector_cache_path'] = reward_config['detectors']['cache_path']
    
    # Add semantic config from YAML (flat structure)
    if 'semantic' in reward_config:
        reward_cfg['semantic_model'] = reward_config['semantic'].get('model_name', 'intfloat/e5-small-v2')
        reward_cfg['semantic_threshold'] = float(reward_config['semantic'].get('threshold', 0.85))
    
    # Add perplexity config from YAML (flat structure)
    if 'perplexity' in reward_config:
        reward_cfg['ppl_model'] = reward_config['perplexity'].get('model_name', 'gpt2')
        reward_cfg['ppl_min'] = float(reward_config['perplexity'].get('ppl_min', 5.0))
        reward_cfg['ppl_max'] = float(reward_config['perplexity'].get('ppl_max', 80.0))
        reward_cfg['ppl_target'] = float(reward_config['perplexity'].get('ppl_target', 30.0))
    
    # Add fairness config from YAML
    if 'fairness' in reward_config and 'penalty_strength' in reward_config['fairness']:
        reward_cfg['fairness_mode'] = 'esl_penalty'
    
    dataset_builder = StealthRLDatasetBuilder(
        data_path=str(data_path),
        batch_size=batch_size,
        group_size=group_size,
        model_name_for_tokenizer=model_config.get('name', 'Qwen/Qwen3-4B-Instruct-2507'),
        renderer_name=model_config.get('renderer', 'qwen3'),
        reward_config=reward_cfg,
        seed=dataset_config.get('seed', 42),
        max_train_examples=max_train,
        max_test_examples=max_test,
    )
    
    # Extract additional config values
    lora_config = yaml_config.get('lora', {})
    sampling_config = yaml_config.get('sampling', {})
    grpo_config = yaml_config.get('grpo', {})
    kl_config = yaml_config.get('kl', {})
    logging_config = yaml_config.get('logging', {})
    parallel_config = yaml_config.get('parallel', {})

    parallel_mode = parallel_config.get('mode', 'sync')
    async_config = parallel_config.get('async', {})
    stream_config = parallel_config.get('stream_minibatch', {})

    async_groups_per_batch = async_config.get('groups_per_batch')
    if async_groups_per_batch is not None:
        async_groups_per_batch = int(async_groups_per_batch)

    stream_groups_per_batch = stream_config.get('groups_per_batch')
    if stream_groups_per_batch is not None:
        stream_groups_per_batch = int(stream_groups_per_batch)
    
    # Create config with settings from YAML (ensure proper types)
    config = StealthRLConfig(
        model_name=model_config.get('name', 'Qwen/Qwen3-4B-Instruct-2507'),
        lora_rank=int(lora_config.get('rank', 16)),
        lora_alpha=int(lora_config.get('alpha', 16)),
        lora_dropout=float(lora_config.get('dropout', 0.05)),
        learning_rate=float(train_config.get('learning_rate', 5e-5)),
        batch_size=batch_size,
        group_size=group_size,
        num_epochs=int(train_config.get('num_epochs', 1)),
        num_substeps=int(train_config.get('num_substeps', 1)),
        max_tokens=int(train_config.get('max_tokens', 400)),
        temperature=float(sampling_config.get('temperature', 1.0)),
        temperature_schedule=sampling_config.get('temperature_schedule', 'constant'),
        kl_penalty_coef=float(kl_config.get('penalty_coef', 0.01)),
        normalize_advantages=bool(grpo_config.get('normalize_advantages', True)),
        advantage_clip=float(grpo_config.get('advantage_clip', 10.0)),
        remove_constant_reward_groups=bool(grpo_config.get('remove_constant_reward_groups', True)),
        dataset_builder=dataset_builder,
        log_path=str(output_dir),
        save_every=int(logging_config.get('save_every', 100)),
        eval_every=int(logging_config.get('eval_every', 50)),
        training_mode=parallel_mode,
        async_max_steps_off_policy=int(async_config.get('max_steps_off_policy', 1)),
        async_groups_per_batch=async_groups_per_batch,
        stream_groups_per_batch=stream_groups_per_batch,
        stream_num_minibatches=int(stream_config.get('num_minibatches', 2)),
    )

    run_metadata["training_mode"] = parallel_mode
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Metrics will be saved to: {output_dir}/metrics.jsonl")
    logger.info(f"HTML logs will be saved to: {output_dir}/")
    
    print("✓ Configuration ready")
    print()
    print(f"Output directory: {output_dir}")
    print(f"Training log: {log_file}")
    print()
    print("Starting training...")
    print("-" * 70)
    print()
    
    # Initialize trainer
    trainer = StealthRLTrainer(
        config=config,
        training_client=training_client,
        service_client=service_client,
    )
    
    # Run training
    await trainer.train()
    
    # Update run metadata with completion info
    run_metadata["status"] = "completed"
    run_metadata["end_time"] = datetime.now().isoformat()
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)
    
    logger.info("Training complete!")
    logger.info(f"Run completed: {run_name}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Metrics saved to: {output_dir}/metrics.jsonl")
    logger.info(f"Training log saved to: {output_dir}/training.log")
    logger.info(f"HTML logs saved to: {output_dir}/*.html")
    
    print()
    print("=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print()
    print(f"Results saved to: {output_dir}")
    print(f"Training log: {output_dir}/training.log")
    print(f"Metrics: {output_dir}/metrics.jsonl")
    print(f"TensorBoard: {output_dir}/tensorboard")
    print()
    print("View TensorBoard:")
    print(f"  tensorboard --logdir {output_dir}/tensorboard --port 6006")
    print()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"\n❌ Training failed: {e}")
        raise
