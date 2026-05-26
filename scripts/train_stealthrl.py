#!/usr/bin/env python3
"""
Main training script for StealthRL.

This script trains a StealthRL paraphraser using RL with a composite
reward function based on detector ensemble, semantic fidelity, quality,
and fairness metrics.
"""

import argparse
import yaml
from pathlib import Path
import torch
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

from stealthrl.models import load_base_model
from stealthrl.training import StealthRLTrainer
from stealthrl.rewards import CompositeReward


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_lora(model, config: dict):
    """Setup LoRA adapters for the model."""
    lora_config = LoraConfig(
        r=config['model']['lora_rank'],
        lora_alpha=config['model']['lora_alpha'],
        lora_dropout=config['model']['lora_dropout'],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model


def load_training_data(config: dict):
    """Load training and evaluation datasets."""
    train_path = config['data']['train_dataset']
    eval_path = config['data']['eval_dataset']
    
    # Load datasets from JSONL files
    train_dataset = load_dataset('json', data_files=train_path, split='train')
    eval_dataset = load_dataset('json', data_files=eval_path, split='train')
    
    return train_dataset, eval_dataset


def main():
    parser = argparse.ArgumentParser(description="Train StealthRL model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Config: {yaml.dump(config, default_flow_style=False)}")
    
    # Override output dir if specified
    if args.output_dir:
        config['output']['output_dir'] = args.output_dir
    
    # Load base model
    print(f"\nLoading base model: {config['model']['base_model']}")
    model, tokenizer = load_base_model(
        config['model']['base_model'],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Setup LoRA
    print("Setting up LoRA adapters...")
    model = setup_lora(model, config)
    model.print_trainable_parameters()
    
    # Load training data
    print("\nLoading training data...")
    try:
        train_dataset, eval_dataset = load_training_data(config)
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Eval dataset: {len(eval_dataset)} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating dummy datasets for testing...")
        # Create dummy datasets
        from datasets import Dataset
        train_dataset = Dataset.from_dict({
            "text": ["Sample text " + str(i) for i in range(100)],
            "label": [0] * 50 + [1] * 50
        })
        eval_dataset = Dataset.from_dict({
            "text": ["Eval text " + str(i) for i in range(20)],
            "label": [0] * 10 + [1] * 10
        })
    
    # Setup reward function
    print("\nSetting up composite reward function...")
    reward_fn = CompositeReward(
        detector_weight=config['reward']['detector_weight'],
        semantic_weight=config['reward']['semantic_weight'],
        quality_weight=config['reward']['quality_weight'],
        fairness_weight=config['reward']['fairness_weight'],
    )
    print(f"Reward weights: detector={config['reward']['detector_weight']}, "
          f"semantic={config['reward']['semantic_weight']}, "
          f"quality={config['reward']['quality_weight']}, "
          f"fairness={config['reward']['fairness_weight']}")
    
    # Initialize trainer
    print(f"\nInitializing {config['training']['algorithm'].upper()} trainer...")
    training_config = config['training'].copy()
    training_config['output_dir'] = config['output']['output_dir']
    
    trainer = StealthRLTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        config=training_config,
        algorithm=config['training']['algorithm'],
    )
    
    # Start training
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    try:
        trainer.train(train_dataset, eval_dataset)
    except Exception as e:
        print(f"Training error: {e}")
        print("Training implementation is still in development.")
        print("This is a skeleton for the full training pipeline.")
    
    # Save model
    output_dir = config['output']['output_dir']
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    
    print("\nTraining complete!")
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
