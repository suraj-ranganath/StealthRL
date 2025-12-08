"""
StealthRL Dataset for Tinker RL Training.

This module provides dataset classes for loading and batching
StealthRL training examples with AI-generated texts, human references,
domain labels, and ESL flags.
"""

import logging
import math
from dataclasses import dataclass
from typing import Sequence, List, Dict, Any, Literal
from pathlib import Path
import json

import chz
from tinker_cookbook import renderers
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .env import StealthEnvGroupBuilder

logger = logging.getLogger(__name__)


@dataclass
class StealthRLExample:
    """
    Single StealthRL training example.
    
    Attributes:
        ai_text: AI-generated text to paraphrase
        human_reference: Human-written reference for similarity baseline
        domain: Text domain (academic, informal, news, etc.)
        is_esl: Whether text exhibits ESL characteristics
        metadata: Additional metadata (model family, detector scores, etc.)
    """
    ai_text: str
    human_reference: str
    domain: str
    is_esl: bool
    metadata: Dict[str, Any]


class StealthRLDataset(RLDataset):
    """
    RLDataset for StealthRL training.
    
    Manages batching of StealthRL examples and creation of environment groups.
    """
    
    def __init__(
        self,
        examples: List[StealthRLExample],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        reward_fn: Any,  # CompositeReward instance
        convo_prefix: List[renderers.Message] | None = None,
        seed: int = 0,
    ):
        """
        Initialize StealthRL dataset.
        
        Args:
            examples: List of StealthRL training examples
            batch_size: Number of different prompts per batch
            group_size: Number of rollouts per prompt (for GRPO)
            renderer: Tinker renderer for prompt formatting
            reward_fn: Composite reward function instance
            convo_prefix: Optional few-shot examples
            seed: Random seed for shuffling
        """
        self.examples = examples
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.reward_fn = reward_fn
        self.convo_prefix = convo_prefix
        self.seed = seed
        
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """
        Get batch of environment group builders.
        
        For multi-epoch training, use modulo to wrap around the dataset.
        
        Args:
            index: Batch index (can be >= len(self))
        
        Returns:
            List of StealthEnvGroupBuilder instances
        """
        # Use modulo to support multi-epoch training
        num_batches = len(self)
        batch_index = index % num_batches
        
        batch_start = batch_index * self.batch_size
        batch_end = min((batch_index + 1) * self.batch_size, len(self.examples))
        
        builders = []
        for i in range(batch_start, batch_end):
            example = self.examples[i]
            builder = StealthEnvGroupBuilder(
                ai_text=example.ai_text,
                human_reference=example.human_reference,
                domain=example.domain,
                is_esl=example.is_esl,
                renderer=self.renderer,
                reward_fn=self.reward_fn,
                num_envs=self.group_size,
                convo_prefix=self.convo_prefix,
            )
            builders.append(builder)
        
        return builders
    
    def __len__(self) -> int:
        """Return number of batches."""
        return math.ceil(len(self.examples) / self.batch_size)


@chz.chz
class StealthRLDatasetBuilder(RLDatasetBuilder):
    """
    Builder for creating StealthRL train/test datasets.
    
    Loads data from JSONL files with AI/human texts, domains, and ESL flags.
    """
    
    data_path: str
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    reward_config: Dict[str, Any]  # Config for initializing CompositeReward
    convo_prefix: List[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    max_examples: int | None = None  # Optional limit for debugging (applies to both)
    max_train_examples: int | None = None  # Optional limit for train set only
    max_test_examples: int | None = None  # Optional limit for test set only
    
    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        """
        Build train and test datasets.
        
        Returns:
            (train_dataset, test_dataset) tuple
        """
        # Get tokenizer and renderer
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        
        # Initialize reward function (import here to avoid circular deps)
        from stealthrl.tinker.reward import TinkerCompositeReward
        reward_fn = TinkerCompositeReward(**self.reward_config)
        
        # Setup conversation prefix
        if self.convo_prefix == "standard":
            convo_prefix = self._standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        
        # Load train and test examples
        train_examples = self._load_examples(split="train")
        test_examples = self._load_examples(split="test")
        
        # Create datasets
        train_dataset = StealthRLDataset(
            examples=train_examples,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            reward_fn=reward_fn,
            convo_prefix=convo_prefix,
            seed=self.seed,
        )
        
        # Test dataset uses group_size=1 for deterministic evaluation
        test_dataset = StealthRLDataset(
            examples=test_examples,
            batch_size=self.batch_size,
            group_size=1,  # Single rollout for eval
            renderer=renderer,
            reward_fn=reward_fn,
            convo_prefix=convo_prefix,
            seed=self.seed,
        ) if test_examples else None
        
        logger.info(f"Loaded {len(train_examples)} train examples, "
                   f"{len(test_examples) if test_examples else 0} test examples")
        
        return train_dataset, test_dataset
    
    def _load_examples(self, split: Literal["train", "test"]) -> List[StealthRLExample]:
        """
        Load examples from JSONL file.
        
        Expected format:
        {
            "ai_text": "...",
            "human_reference": "...",
            "domain": "academic"|"informal"|"news",
            "is_esl": true|false,
            "metadata": {...}
        }
        """
        data_path = Path(self.data_path)
        file_path = data_path / f"{split}.jsonl"
        
        if not file_path.exists():
            logger.warning(f"Data file not found: {file_path}")
            return []
        
        examples = []
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Check limits: use specific limit if set, otherwise fall back to max_examples
                limit = None
                if split == "train" and self.max_train_examples:
                    limit = self.max_train_examples
                elif split == "test" and self.max_test_examples:
                    limit = self.max_test_examples
                elif self.max_examples:
                    limit = self.max_examples
                
                if limit and len(examples) >= limit:
                    break
                
                try:
                    data = json.loads(line)
                    example = StealthRLExample(
                        ai_text=data["ai_text"],
                        human_reference=data.get("human_reference", ""),
                        domain=data.get("domain", "unknown"),
                        is_esl=data.get("is_esl", False),
                        metadata=data.get("metadata", {}),
                    )
                    examples.append(example)
                except Exception as e:
                    logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
        
        return examples
    
    @staticmethod
    def _standard_fewshot_prefix() -> List[renderers.Message]:
        """
        Return standard few-shot examples for paraphrasing.
        
        These demonstrate the desired paraphrasing style.
        """
        return [
            {
                "role": "user",
                "content": (
                    "Please paraphrase the following text while maintaining its meaning "
                    "and ensuring it reads naturally:\n\n"
                    "The implementation of neural networks requires careful consideration "
                    "of hyperparameters and architectural choices.\n\n"
                    "Paraphrased text:"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Developing neural networks demands thoughtful attention to both "
                    "hyperparameter selection and structural design decisions."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Please paraphrase the following text while maintaining its meaning "
                    "and ensuring it reads naturally:\n\n"
                    "Recent advances in machine learning have enabled significant progress "
                    "in natural language processing tasks.\n\n"
                    "Paraphrased text:"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "New developments in machine learning have led to substantial improvements "
                    "across various language processing applications."
                ),
            },
        ]
