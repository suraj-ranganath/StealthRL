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
    Single StealthRL training example (DEFENSIVE MODE).
    
    Attributes:
        ai_text: AI-generated text (kept for compatibility)
        human_reference: Human-written text to paraphrase (INPUT)
        domain: Text domain (inferred from source)
        metadata: Additional metadata (source, etc.)
    
    DEFENSIVE USE CASE: Model learns to paraphrase human text to avoid false positives.
    """
    ai_text: str
    human_reference: str
    domain: str
    metadata: Dict[str, Any]


class StealthRLDataset(RLDataset):
    """
    RLDataset for StealthRL training (DEFENSIVE MODE).
    
    Manages batching of StealthRL examples and creation of environment groups.
    Model learns to paraphrase human text to avoid false positive detection.
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
        Load examples from data source (supports both JSONL files and HuggingFace datasets).
        
        For JSONL format (Tinker):
        {
            "ai_text": "...",
            "human_reference": "...",
            "domain": "academic"|"informal"|"news",
            "metadata": {...}
        }
        
        For MAGE dataset (HuggingFace):
        - Loaded from data/mage/test split
        - Has columns: [text, label, src]
        - label: 1=human, 0=AI
        - src: source identifier (e.g., "eli5_human", "gpt3_machine")
        """
        data_path = Path(self.data_path)
        
        # Try JSONL first (Tinker format)
        file_path = data_path / f"{split}.jsonl"
        if file_path.exists():
            return self._load_jsonl_examples(file_path, split)
        
        # Try HuggingFace dataset (MAGE format)
        if data_path.name == "mage":
            return self._load_mage_examples(split)
        
        logger.warning(f"No data found for path: {data_path}")
        return []
    
    def _load_jsonl_examples(self, file_path: Path, split: str) -> List[StealthRLExample]:
        """Load examples from JSONL file (Tinker format)."""
        examples = []
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Check limits
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
                        metadata=data.get("metadata", {}),
                    )
                    examples.append(example)
                except Exception as e:
                    logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
        
        return examples
    
    def _load_mage_examples(self, split: str) -> List[StealthRLExample]:
        """Load examples from MAGE HuggingFace dataset."""
        try:
            from datasets import load_from_disk
        except ImportError:
            logger.error("datasets package required for MAGE loading. Install with: pip install datasets")
            return []
        
        try:
            # MAGE is typically in data/mage/test (single test split)
            ds = load_from_disk('data/mage/test')
            examples = []
            
            # Determine limit
            limit = None
            if split == "train" and self.max_train_examples:
                limit = self.max_train_examples
            elif split == "test" and self.max_test_examples:
                limit = self.max_test_examples
            elif self.max_examples:
                limit = self.max_examples
            
            # Extract domain from src field
            # Format: "eli5_human", "gpt3_davinci_002", etc.
            def extract_domain(src: str) -> str:
                # Take everything before first underscore or up to first number
                import re
                match = re.match(r'([a-z_]+)', src)
                if match:
                    base = match.group(1).rstrip('_')
                    # Map known domains
                    domain_map = {
                        'eli5': 'informal',
                        'hswag': 'reasoning',
                        'xsum': 'news',
                        'roct': 'reading',
                        'wp': 'creative',
                        'yelp': 'review',
                        'sci_gen': 'academic',
                        'tldr': 'news',
                        'squad': 'reading',
                        'cmv': 'social',
                        'cnn': 'news',
                        'imdb': 'review',
                        'pubmed': 'academic',
                        'dialogsum': 'dialogue',
                    }
                    return domain_map.get(base, base)
                return 'unknown'
            
            for i, item in enumerate(ds):
                if limit and len(examples) >= limit:
                    break
                
                # MAGE format: {text, label, src}
                # label: 1=human, 0=AI
                text = item.get('text', '')
                label = item.get('label', 0)
                src = item.get('src', 'unknown')
                
                # Only use human-written text for DEFENSIVE training
                if label == 1:
                    # Create synthetic AI text as placeholder (not used in DEFENSIVE mode)
                    example = StealthRLExample(
                        ai_text="[PLACEHOLDER - NOT USED IN DEFENSIVE MODE]",
                        human_reference=text,
                        domain=extract_domain(src),
                        metadata={'source': src, 'original_label': int(label)},
                    )
                    examples.append(example)
            
            logger.info(f"Loaded {len(examples)} human examples from MAGE dataset")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load MAGE dataset: {e}")
            return []
    
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
