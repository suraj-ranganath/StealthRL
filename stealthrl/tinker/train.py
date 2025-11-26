"""
GRPO Training Loop for StealthRL on Tinker.

This module implements Group-based Reinforcement Policy Optimization (GRPO)
for training StealthRL policies using Tinker's RL infrastructure.

Based on Tinker Cookbook patterns and GRPO-RL-Training skill:
- Group-based reward centering (variance reduction)
- KL divergence penalty (preserve fluency)
- Adaptive advantage normalization
- All-negative group handling
- Curriculum learning
- Temperature scheduling
"""

import logging
import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

import torch
import chz
import tinker
from tinker_cookbook.rl import train as rl_train
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)


@chz.chz
class StealthRLConfig:
    """
    Configuration for StealthRL training on Tinker.
    
    Follows Tinker Cookbook Config patterns with GRPO-specific settings.
    """
    
    # Model settings
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name: str = "qwen3"  # Tinker renderer name
    
    # LoRA settings
    lora_rank: int = 16  # 8-16 recommended for RL
    lora_alpha: int | None = None  # If None, use lora_rank
    lora_dropout: float = 0.05
    lora_target_modules: List[str] | None = None  # None = all linear layers
    
    # Learning rate (will be scaled for LoRA)
    learning_rate: float = 1e-5  # Base LR, will be multiplied by LoRA factor
    
    # GRPO settings
    batch_size: int = 8  # Number of different prompts per batch
    group_size: int = 4  # Number of rollouts per prompt (for centering)
    num_epochs: int = 3
    num_substeps: int = 1  # Gradient accumulation steps
    
    # Sampling settings
    max_tokens: int = 512  # Max tokens per generation
    temperature: float = 1.0  # Higher = more exploration
    temperature_schedule: str = "constant"  # "constant" | "decay"
    temperature_decay: float = 0.95  # If schedule="decay"
    
    # KL divergence penalty (AuthorMist-inspired)
    kl_penalty_coef: float = 0.001  # β in L = -E[R] + β*KL(π || π_ref)
    kl_target: float | None = None  # Target KL for adaptive penalty
    kl_adapt_rate: float = 0.1  # Adaptation rate for KL penalty
    
    # Reward normalization (GRPO-specific)
    normalize_advantages: bool = True  # Group-normalize advantages
    advantage_clip: float = 5.0  # Clip advantages to [-clip, clip]
    reward_clip: float | None = None  # Optional reward clipping
    
    # All-negative group handling
    all_negative_min_reward: float = 0.01  # Minimum reward for all-negative groups
    all_negative_downweight: float = 0.5  # Downweight all-negative groups
    
    # Curriculum learning
    curriculum_enabled: bool = False
    curriculum_start_quantile: float = 0.7  # Start with easy examples (top 70%)
    curriculum_end_quantile: float = 0.0  # End with all examples
    curriculum_steps: int = 1000  # Steps to transition
    
    # Dataset
    dataset_builder: RLDatasetBuilder = None  # Will be set from config
    
    # Logging
    log_path: str = "/tmp/stealthrl"
    log_interval: int = 10  # Log every N batches
    eval_interval: int = 100  # Eval every N batches
    save_interval: int = 500  # Save checkpoint every N batches
    
    # Debug
    num_groups_to_log: int = 4  # Number of groups to log in detail
    debug_mode: bool = False
    
    # Verifiable rewards
    remove_constant_reward_groups: bool = True  # Remove groups with identical rewards


class StealthRLTrainer:
    """
    GRPO trainer for StealthRL on Tinker.
    
    Implements group-based RL with:
    - Reward centering within groups (variance reduction)
    - KL penalty to reference model (preserve fluency)
    - Adaptive advantage normalization
    - All-negative group handling
    - Curriculum learning
    - Temperature scheduling
    """
    
    def __init__(
        self,
        config: StealthRLConfig,
        training_client: tinker.TrainingClient,
        service_client: tinker.ServiceClient,
    ):
        """
        Initialize StealthRL trainer.
        
        Args:
            config: Training configuration
            training_client: Tinker training client
            service_client: Tinker service client
        """
        self.config = config
        self.training_client = training_client
        self.service_client = service_client
        
        # Initialize tokenizer
        self.tokenizer = get_tokenizer(config.model_name)
        
        # Statistics tracking
        self.step = 0
        self.all_negative_count = 0
        self.total_groups = 0
        
        # KL penalty adaptation
        self.kl_penalty_coef = config.kl_penalty_coef
        
        # Temperature scheduling
        self.current_temperature = config.temperature
        
        # Curriculum learning
        self.current_quantile = config.curriculum_start_quantile if config.curriculum_enabled else 0.0
        
        # Create log directory (config.log_path should be the specific run directory)
        Path(config.log_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize ML logger (use JsonLogger which writes to a file)
        # metrics.jsonl should be directly in the run directory, not in a subdirectory
        log_file = str(Path(config.log_path) / "metrics.jsonl")
        self.ml_logger = ml_log.JsonLogger(log_file)
        
        logger.info("Initialized StealthRLTrainer")
        logger.info(f"Model: {config.model_name}")
        logger.info(f"LoRA rank: {config.lora_rank}")
        logger.info(f"Batch size: {config.batch_size}, Group size: {config.group_size}")
        logger.info(f"KL penalty: {self.kl_penalty_coef}")
        logger.info(f"Logging metrics to: {log_file}")
    
    async def train(self):
        """
        Main training loop.
        
        Delegates to Tinker's train.do_sync_training or do_async_training.
        """
        # Build dataset
        train_dataset, test_dataset = await self.config.dataset_builder()
        
        logger.info(f"Train dataset: {len(train_dataset)} batches")
        if test_dataset:
            logger.info(f"Test dataset: {len(test_dataset)} batches")
        
        logger.info(f"Training for {self.config.num_epochs} epoch(s)")
        logger.info(f"Total training iterations: {len(train_dataset) * self.config.num_epochs}")
        
        # Convert config to Tinker train.Config
        tinker_config = self._build_tinker_config(train_dataset)
        
        # Create evaluators
        evaluators = []
        if test_dataset:
            from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
            evaluators.append(
                RLTestSetEvaluator(
                    dataset=test_dataset,
                    max_tokens=self.config.max_tokens,
                    name="test",
                    num_groups_to_log=self.config.num_groups_to_log,
                )
            )
        
        # Run training (using Tinker's do_sync_training)
        # Based on Tinker source: do_sync_training iterates for i_batch in range(start_batch, end_batch)
        # It calls dataset.get_batch(i_batch) directly (NOT i_batch % num_batches)
        # Therefore: num_batches must equal end_batch to avoid IndexError
        num_train_batches = len(train_dataset)
        total_iterations = num_train_batches * self.config.num_epochs
        
        logger.info(f"Starting training: {total_iterations} iterations across {self.config.num_epochs} epoch(s)")
        logger.info(f"Dataset size: {num_train_batches} batches per epoch")
        
        await rl_train.do_sync_training(
            start_batch=0,
            end_batch=total_iterations,  # Exclusive upper bound (like Python range)
            num_batches=total_iterations,  # Must equal end_batch to avoid IndexError
            cfg=tinker_config,
            training_client=self.training_client,
            service_client=self.service_client,
            evaluators=evaluators,
            dataset=train_dataset,
            ml_logger=self.ml_logger,
            tokenizer=self.tokenizer,
        )
        
        logger.info("Training complete")
    
    def _build_tinker_config(self, dataset) -> rl_train.Config:
        """
        Build Tinker rl_train.Config from StealthRLConfig.
        
        Args:
            dataset: Training dataset
        
        Returns:
            Tinker rl_train.Config instance
        """
        # Calculate LoRA LR scaling factor
        from tinker_cookbook.hyperparam_utils import get_lora_lr_over_full_finetune_lr
        lora_lr_factor = get_lora_lr_over_full_finetune_lr(self.config.model_name)
        scaled_lr = self.config.learning_rate * lora_lr_factor
        
        logger.info(f"LoRA LR scaling factor: {lora_lr_factor:.1f}x")
        logger.info(f"Scaled LR: {scaled_lr:.2e}")
        
        # Build Tinker config
        tinker_config = rl_train.Config(
            model_name=self.config.model_name,
            learning_rate=scaled_lr,
            lora_rank=self.config.lora_rank,
            kl_penalty_coef=self.kl_penalty_coef,
            max_tokens=self.config.max_tokens,
            temperature=self.current_temperature,
            num_substeps=self.config.num_substeps,
            dataset_builder=self.config.dataset_builder,
            remove_constant_reward_groups=self.config.remove_constant_reward_groups,
            log_path=self.config.log_path,
        )
        
        return tinker_config
    
    def _process_trajectory_group(self, trajectory_group, rewards):
        """
        Process trajectory group with GRPO-specific enhancements.
        
        Implements:
        - Group-based reward centering
        - Advantage normalization and clipping
        - All-negative group handling
        - Logging
        
        Args:
            trajectory_group: List of trajectories in group
            rewards: List of rewards for each trajectory
        
        Returns:
            Processed advantages and metrics
        """
        self.total_groups += 1
        
        # Compute group statistics
        rewards_tensor = torch.tensor(rewards)
        mean_reward = rewards_tensor.mean().item()
        std_reward = rewards_tensor.std().item()
        
        # Check for all-negative groups
        all_negative = all(r <= 0.0 for r in rewards)
        if all_negative:
            self.all_negative_count += 1
            
            # Handle all-negative groups
            if self.config.all_negative_min_reward > 0:
                # Add small shaped signal based on relative scores
                relative_scores = rewards_tensor - rewards_tensor.min()
                if relative_scores.max() > 0:
                    rewards_tensor = relative_scores / relative_scores.max() * self.config.all_negative_min_reward
                else:
                    rewards_tensor = torch.ones_like(rewards_tensor) * self.config.all_negative_min_reward
                
                # Downweight contributions
                weight = self.config.all_negative_downweight
            else:
                weight = 1.0
        else:
            weight = 1.0
        
        # Group-based advantage computation (GRPO)
        if self.config.normalize_advantages:
            # Center by group mean
            advantages = rewards_tensor - mean_reward
            
            # Normalize by group std
            if std_reward > 1e-6:
                advantages = advantages / (std_reward + 1e-6)
            
            # Clip advantages
            if self.config.advantage_clip:
                advantages = torch.clamp(advantages, -self.config.advantage_clip, self.config.advantage_clip)
        else:
            advantages = rewards_tensor
        
        # Metrics
        metrics = {
            "reward/group_mean": mean_reward,
            "reward/group_std": std_reward,
            "reward/group_min": rewards_tensor.min().item(),
            "reward/group_max": rewards_tensor.max().item(),
            "all_negative_frac": float(all_negative),
            "group_weight": weight,
        }
        
        return advantages.tolist(), metrics, weight
    
    def _update_curriculum(self):
        """Update curriculum quantile based on training progress."""
        if not self.config.curriculum_enabled:
            return
        
        progress = self.step / self.config.curriculum_steps
        progress = min(1.0, progress)
        
        # Linear interpolation from start to end quantile
        self.current_quantile = (
            self.config.curriculum_start_quantile * (1 - progress) +
            self.config.curriculum_end_quantile * progress
        )
    
    def _update_temperature(self):
        """Update sampling temperature based on schedule."""
        if self.config.temperature_schedule == "decay":
            self.current_temperature *= self.config.temperature_decay
            self.current_temperature = max(0.1, self.current_temperature)  # Minimum temp
    
    def _adapt_kl_penalty(self, kl_div: float):
        """
        Adapt KL penalty coefficient based on observed KL divergence.
        
        Args:
            kl_div: Current KL divergence
        """
        if self.config.kl_target is None:
            return
        
        # Increase penalty if KL too high, decrease if too low
        kl_error = kl_div - self.config.kl_target
        self.kl_penalty_coef *= (1.0 + self.config.kl_adapt_rate * kl_error)
        self.kl_penalty_coef = max(1e-6, min(1.0, self.kl_penalty_coef))
    
    def log_step(self, metrics: Dict[str, Any]):
        """
        Log step metrics.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        # Add step-level stats
        metrics["step"] = self.step
        metrics["all_negative_frac_total"] = self.all_negative_count / max(1, self.total_groups)
        metrics["curriculum_quantile"] = self.current_quantile
        metrics["temperature"] = self.current_temperature
        metrics["kl_penalty_coef"] = self.kl_penalty_coef
        
        # Write to ML logger
        self.ml_logger.log(metrics)
        
        # Print summary
        if self.step % self.config.log_interval == 0:
            logger.info(f"Step {self.step}: reward={metrics.get('reward/total', 0):.3f}, "
                       f"kl={metrics.get('kl', 0):.3f}, "
                       f"all_neg={metrics.get('all_negative_frac_total', 0):.2%}")
        
        self.step += 1
    
    def save_debug_samples(self, samples: List[Dict[str, Any]]):
        """
        Save debug samples to JSONL file.
        
        Args:
            samples: List of sample dictionaries with prompts, responses, rewards
        """
        debug_path = Path(self.config.log_path) / "debug_samples.jsonl"
        
        with open(debug_path, 'a') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\\n')


async def main():
    """
    Main entry point for StealthRL training.
    
    Loads config, initializes Tinker clients, and runs training.
    """
    import os
    import argparse
    import json
    from datetime import datetime
    from dotenv import load_dotenv
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train StealthRL with Tinker")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--data-path", type=str, default="data/tinker", help="Path to training data")
    parser.add_argument("--output-dir", type=str, default="outputs/runs", help="Base output directory for runs")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--run-name", type=str, help="Custom run name (default: auto-generated with timestamp)")
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Create timestamped run directory
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    
    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save run metadata
    run_metadata = {
        "run_name": run_name,
        "start_time": datetime.now().isoformat(),
        "data_path": args.data_path,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "config_file": args.config,
        "status": "running",
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
    
    # Get Tinker API key (it's automatically loaded from environment by tinker.ServiceClient)
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        raise ValueError("TINKER_API_KEY environment variable not set. Please add it to .env file.")
    
    # Initialize Tinker clients (ServiceClient loads API key from environment automatically)
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        rank=16
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config (placeholder - would load from YAML)
    from stealthrl.tinker.dataset import StealthRLDatasetBuilder
    
    dataset_builder = StealthRLDatasetBuilder(
        data_path=args.data_path,
        batch_size=args.batch_size,
        group_size=4,
        model_name_for_tokenizer="Qwen/Qwen3-4B-Instruct-2507",
        renderer_name="qwen3",
        reward_config={
            "detector_weight": 1.0,
            "semantic_weight": 1.0,
            "perplexity_weight": 0.5,
            "fairness_weight": 0.2,
        },
        seed=42,
    )
    
    config = StealthRLConfig(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        lora_rank=16,
        learning_rate=1e-5,
        batch_size=args.batch_size,
        group_size=4,
        num_epochs=args.num_epochs,
        kl_penalty_coef=0.001,
        dataset_builder=dataset_builder,
        log_path=str(output_dir),  # Save metrics to output directory
    )
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Metrics will be saved to: {output_dir}/metrics.jsonl")
    logger.info(f"HTML logs will be saved to: {output_dir}/")
    
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
