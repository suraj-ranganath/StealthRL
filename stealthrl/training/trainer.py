"""
StealthRL trainer using HuggingFace TRL with KL regularization.

We follow AuthorMist and RLHF best practices by adding a KL penalty to keep 
the StealthRL policy close to the base LM, which helps preserve fluency and 
semantics while still optimizing detector-oriented rewards.

See https://arxiv.org/abs/2503.08716 for AuthorMist's approach.
"""

from typing import Dict, Optional, List
import torch
import torch.nn.functional as F
from trl import GRPOTrainer, GRPOConfig, PPOTrainer, PPOConfig
from transformers import AutoTokenizer, TrainingArguments
from datasets import Dataset


class StealthRLTrainer:
    """
    Trainer for StealthRL using RL algorithms from HuggingFace TRL.
    Supports GRPO and PPO training with composite reward functions and KL regularization.
    """
    
    def __init__(
        self,
        model,
        ref_model,
        tokenizer: AutoTokenizer,
        reward_fn,
        config: Dict,
        algorithm: str = "grpo",
        kl_beta: float = 0.001,
    ):
        """
        Initialize StealthRL trainer with KL regularization.
        
        Args:
            model: Base model with LoRA adapters (policy model)
            ref_model: Reference model for KL divergence (frozen base model)
            tokenizer: Tokenizer for the model
            reward_fn: Composite reward function
            config: Training configuration dictionary
            algorithm: RL algorithm to use ("grpo" or "ppo")
            kl_beta: KL penalty coefficient (default 0.001, following AuthorMist)
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config
        self.algorithm = algorithm.lower()
        self.kl_beta = kl_beta
        self.trainer = None
        
        # Freeze reference model
        if self.ref_model is not None:
            for param in self.ref_model.parameters():
                param.requires_grad = False
        
        # Initialize trainer based on algorithm
        self._initialize_trainer()
        
    def _initialize_trainer(self):
        """Initialize TRL trainer based on algorithm."""
        if self.algorithm == "grpo":
            training_args = GRPOConfig(
                output_dir=self.config.get("output_dir", "checkpoints"),
                learning_rate=self.config.get("learning_rate", 1e-5),
                per_device_train_batch_size=self.config.get("batch_size", 8),
                gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
                max_steps=self.config.get("max_steps", 10000),
                logging_steps=self.config.get("logging_steps", 10),
                save_steps=self.config.get("save_steps", 1000),
                eval_steps=self.config.get("eval_steps", 500),
                warmup_steps=self.config.get("warmup_steps", 500),
                max_grad_norm=self.config.get("max_grad_norm", 1.0),
                fp16=torch.cuda.is_available(),
            )
            
            self.trainer = GRPOTrainer(
                model=self.model,
                ref_model=self.ref_model,  # Pass reference model for KL
                args=training_args,
                tokenizer=self.tokenizer,
            )
            
        elif self.algorithm == "ppo":
            training_args = PPOConfig(
                output_dir=self.config.get("output_dir", "checkpoints"),
                learning_rate=self.config.get("learning_rate", 1e-5),
                batch_size=self.config.get("batch_size", 8),
                gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
                max_grad_norm=self.config.get("max_grad_norm", 1.0),
            )
            
            self.trainer = PPOTrainer(
                model=self.model,
                ref_model=self.ref_model,  # Pass reference model for KL
                config=training_args,
                tokenizer=self.tokenizer,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
    def train(self, dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """
        Run RL training loop.
        
        Args:
            dataset: Training dataset (should contain 'text' or 'prompt' field)
            eval_dataset: Optional evaluation dataset
        """
        print(f"Starting {self.algorithm.upper()} training...")
        print(f"Training samples: {len(dataset)}")
        
        if self.algorithm == "grpo":
            # GRPO training with reward function
            def reward_function(prompts: List[str], completions: List[str]) -> List[float]:
                """Compute rewards for generated completions."""
                # Compute composite reward
                rewards = self._compute_batch_rewards(prompts, completions)
                return rewards.tolist()
            
            # Train using GRPO
            self.trainer.train(
                dataset=dataset,
                eval_dataset=eval_dataset,
                reward_fn=reward_function
            )
            
        elif self.algorithm == "ppo":
            # PPO training loop
            for epoch in range(self.config.get("num_epochs", 1)):
                for batch in dataset:
                    # Generate completions
                    prompts = batch["prompt"] if "prompt" in batch else batch["text"]
                    
                    # Get model outputs
                    query_tensors = self.tokenizer(
                        prompts, 
                        return_tensors="pt", 
                        padding=True,
                        truncation=True
                    ).input_ids
                    
                    response_tensors = self.model.generate(
                        query_tensors,
                        max_new_tokens=256,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.7
                    )
                    
                    # Decode responses
                    responses = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
                    
                    # Compute rewards
                    rewards = self._compute_batch_rewards(prompts, responses)
                    
                    # PPO step
                    self.trainer.step(query_tensors, response_tensors, rewards)
        
        print("Training complete!")
    
    def _compute_kl_divergence(
        self, 
        policy_logits: torch.Tensor, 
        ref_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between policy and reference model distributions.
        
        KL(π || π_ref) encourages policy to stay close to the reference model,
        preserving fluency and preventing mode collapse.
        
        Args:
            policy_logits: Logits from policy model [batch, seq_len, vocab]
            ref_logits: Logits from reference model [batch, seq_len, vocab]
            
        Returns:
            KL divergence [scalar]
        """
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)
        
        kl = torch.sum(ref_probs * (torch.log(ref_probs + 1e-10) - policy_log_probs), dim=-1)
        return kl.mean()
        
    def _compute_batch_rewards(self, prompts: List[str], completions: List[str]) -> torch.Tensor:
        """
        Compute composite rewards for a batch of completions.
        
        Args:
            prompts: Original prompts/texts
            completions: Generated completions
            
        Returns:
            Tensor of reward values
        """
        # Use the composite reward function
        # This would call detector ensemble, semantic similarity, quality, and fairness
        try:
            rewards = self.reward_fn.compute(
                detector_scores=torch.zeros(len(completions)),  # Placeholder
                semantic_scores=torch.ones(len(completions)) * 0.8,  # Placeholder
                quality_scores=torch.ones(len(completions)) * 0.7,  # Placeholder
                fairness_scores=torch.zeros(len(completions)),  # Placeholder
            )
            return rewards
        except Exception as e:
            print(f"Error computing rewards: {e}")
            return torch.zeros(len(completions))
        
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("Running evaluation...")
        
        # Generate completions for eval set
        all_prompts = []
        all_completions = []
        
        for batch in eval_dataset:
            prompts = batch["prompt"] if "prompt" in batch else batch["text"]
            
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True
            )
            
            completions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            all_prompts.extend(prompts)
            all_completions.extend(completions)
        
        # Compute metrics
        rewards = self._compute_batch_rewards(all_prompts, all_completions)
        
        metrics = {
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()),
            "min_reward": float(rewards.min()),
            "max_reward": float(rewards.max()),
        }
        
        print(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self, output_dir: str):
        """Save the trained model."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
