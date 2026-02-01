"""
Dataset loading and preparation for StealthRL evaluation.

Supports MAGE, RAID, and PadBen benchmarks with consistent schema.
"""

import json
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal

import datasets
from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """Single evaluation sample with consistent schema."""
    id: str
    label: Literal["human", "ai"]
    domain: str
    generator: Optional[str]
    text: str
    metadata: Dict[str, Any]
    
    @property
    def is_ai(self) -> bool:
        """Check if sample is AI-generated."""
        return self.label == "ai"
    
    @property
    def is_human(self) -> bool:
        """Check if sample is human-written."""
        return self.label == "human"
    
    def to_dict(self) -> dict:
        return asdict(self)


class BaseEvalDataset:
    """Base class for evaluation datasets."""
    
    def __init__(
        self,
        samples: List[EvalSample],
        name: str,
    ):
        self.samples = samples
        self.name = name
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> EvalSample:
        return self.samples[idx]
    
    def __iter__(self):
        return iter(self.samples)
    
    @property
    def human_samples(self) -> List[EvalSample]:
        return [s for s in self.samples if s.label == "human"]
    
    @property
    def ai_samples(self) -> List[EvalSample]:
        return [s for s in self.samples if s.label == "ai"]
    
    def filter_by_length(
        self,
        min_tokens: int = 100,
        max_tokens: int = 500,
    ) -> "BaseEvalDataset":
        """Filter samples by token length (approximate using words)."""
        filtered = []
        for sample in self.samples:
            word_count = len(sample.text.split())
            # Approximate: 1 word ≈ 1.3 tokens
            token_estimate = int(word_count * 1.3)
            if min_tokens <= token_estimate <= max_tokens:
                filtered.append(sample)
        return self.__class__(filtered, self.name)
    
    def balance(
        self,
        n_human: int = 1000,
        n_ai: int = 1000,
        seed: int = 42,
    ) -> "BaseEvalDataset":
        """Create balanced dataset with specified counts."""
        random.seed(seed)
        
        human = self.human_samples
        ai = self.ai_samples
        
        if len(human) < n_human:
            logger.warning(f"Only {len(human)} human samples available (requested {n_human})")
            n_human = len(human)
        if len(ai) < n_ai:
            logger.warning(f"Only {len(ai)} AI samples available (requested {n_ai})")
            n_ai = len(ai)
        
        sampled_human = random.sample(human, n_human)
        sampled_ai = random.sample(ai, n_ai)
        
        return self.__class__(sampled_human + sampled_ai, self.name)
    
    def save(self, path: str):
        """Save dataset to JSONL."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            for sample in self.samples:
                f.write(json.dumps(sample.to_dict()) + "\n")
        
        logger.info(f"Saved {len(self.samples)} samples to {path}")
    
    @classmethod
    def load(cls, path: str, name: str = "custom") -> "BaseEvalDataset":
        """Load dataset from JSONL."""
        samples = []
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                samples.append(EvalSample(**data))
        return cls(samples, name)


class MAGEDataset(BaseEvalDataset):
    """
    MAGE benchmark dataset loader.
    
    Paper: https://arxiv.org/abs/2305.13242
    HF: https://huggingface.co/datasets/yaful/MAGE
    """
    
    HF_DATASET = "yaful/MAGE"
    
    # Domain mapping from MAGE source fields
    DOMAIN_MAP = {
        "peerread": "academic",
        "outfox": "academic",
        "writing_prompts": "creative",
        "yelp": "reviews",
        "xsum": "news",
        "squad": "qa",
    }
    
    def __init__(self, samples: List[EvalSample], name: str = "mage"):
        super().__init__(samples, name)
    
    @classmethod
    def download(
        cls,
        split: str = "test",
        cache_dir: Optional[str] = None,
        domains: Optional[List[str]] = None,
    ) -> "MAGEDataset":
        """
        Download MAGE dataset from HuggingFace.
        
        Args:
            split: Dataset split ("train", "test", or "validation")
            cache_dir: Cache directory for downloads
            domains: Optional list of domains to include
        
        Returns:
            MAGEDataset instance
        """
        logger.info(f"Downloading MAGE dataset (split={split})...")
        
        try:
            # Load from HuggingFace
            ds = load_dataset(cls.HF_DATASET, split=split, cache_dir=cache_dir)
        except Exception as e:
            logger.error(f"Failed to download MAGE: {e}")
            raise
        
        samples = []
        for idx, item in enumerate(ds):
            # MAGE format: 'text', 'label' (1=human, 0=machine/AI)
            # Verified from 'src' field: *_human sources have label=1, *_machine_* have label=0
            label = "human" if item.get("label", 0) == 1 else "ai"
            
            # Infer domain from source if available
            source = item.get("source", "unknown")
            domain = cls.DOMAIN_MAP.get(source, source)
            
            # Skip if filtering by domain
            if domains and domain not in domains:
                continue
            
            sample = EvalSample(
                id=f"mage_{split}_{idx}",
                label=label,
                domain=domain,
                generator=item.get("model", None),
                text=item["text"],
                metadata={
                    "source": source,
                    "original_label": item.get("label"),
                },
            )
            samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} samples from MAGE {split}")
        return cls(samples)


class RAIDDataset(BaseEvalDataset):
    """
    RAID benchmark dataset loader.
    
    Paper: https://aclanthology.org/2024.acl-long.674/
    GitHub: https://github.com/liamdugan/raid
    HuggingFace: https://huggingface.co/datasets/liamdugan/raid
    
    RAID Dataset Fields:
        - id: uuid4 uniquely identifying the generation
        - model: Generator model name (chatgpt, gpt4, gpt3, gpt2, llama-chat, 
                 mistral, mistral-chat, mpt, mpt-chat, cohere, cohere-chat)
                 Empty/None for human-written text
        - domain: Genre (abstracts, books, code, czech, german, news, poetry, 
                  recipes, reddit, reviews, wiki)
        - generation: The text content
        - attack: Adversarial attack type if any (homoglyph, paraphrase, etc.)
        - decoding: Decoding strategy (greedy, sampling)
    """
    
    HF_DATASET = "liamdugan/raid"
    
    # Map RAID models to normalized generator names
    MODEL_MAP = {
        "chatgpt": "ChatGPT",
        "gpt4": "GPT-4",
        "gpt3": "GPT-3",
        "gpt2": "GPT-2",
        "llama-chat": "Llama-2-Chat",
        "mistral": "Mistral-7B",
        "mistral-chat": "Mistral-7B-Chat",
        "mpt": "MPT-30B",
        "mpt-chat": "MPT-30B-Chat",
        "cohere": "Cohere",
        "cohere-chat": "Cohere-Chat",
    }
    
    def __init__(self, samples: List[EvalSample], name: str = "raid"):
        super().__init__(samples, name)
    
    @classmethod
    def download(
        cls,
        split: str = "train",  # Use train by default (test labels are hidden)
        cache_dir: Optional[str] = None,
        n_samples: int = 400,
        seed: int = 42,
        include_adversarial: bool = False,
        domains: Optional[List[str]] = None,
    ) -> "RAIDDataset":
        """
        Download RAID dataset from HuggingFace.
        
        Args:
            split: Dataset split ('train', 'test', or 'extra')
                   Note: 'test' labels are hidden; use 'train' for evaluation
            cache_dir: Cache directory for HuggingFace datasets
            n_samples: Number of samples to include (balanced 50/50 human/AI)
            seed: Random seed for sampling
            include_adversarial: Whether to include adversarial attack samples
            domains: Optional list of domains to filter by
        
        Returns:
            RAIDDataset instance with balanced human/AI samples
        """
        logger.info(f"Downloading RAID dataset (split={split}, n_samples={n_samples})...")
        
        try:
            # RAID dataset is large; we'll stream it to avoid full download
            ds = load_dataset(
                cls.HF_DATASET,
                name="raid",  # Use the 'raid' subset
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.error(f"Failed to download RAID: {e}")
            logger.info("Tip: RAID is ~11GB. Try using streaming or a smaller subset.")
            return cls([])
        
        human_samples = []
        ai_samples = []
        
        # Process samples
        for idx, item in enumerate(ds):
            # Get the text content
            text = item.get("generation", "")
            if not text or not text.strip():
                continue
            
            # Determine label: model=None/empty means human-written
            model = item.get("model", None)
            if model is None or model == "" or (isinstance(model, str) and model.strip() == ""):
                label = "human"
                generator = None
            else:
                label = "ai"
                generator = cls.MODEL_MAP.get(model, model)
            
            # Get domain
            domain = item.get("domain", "general")
            
            # Filter by domain if specified
            if domains and domain not in domains:
                continue
            
            # Filter adversarial attacks if not wanted
            attack = item.get("attack", None)
            if not include_adversarial and attack is not None and attack != "":
                continue
            
            sample = EvalSample(
                id=f"raid_{split}_{idx}",
                label=label,
                domain=domain,
                generator=generator,
                text=text,
                metadata={
                    "attack": attack,
                    "decoding": item.get("decoding", None),
                    "repetition_penalty": item.get("repetition_penalty", None),
                    "source_id": item.get("source_id", None),
                },
            )
            
            # Collect by class
            if label == "human":
                human_samples.append(sample)
            else:
                ai_samples.append(sample)
            
            # Early stop if we have enough samples for efficiency
            n_per_class = n_samples // 2
            if len(human_samples) >= n_per_class * 3 and len(ai_samples) >= n_per_class * 3:
                break
        
        # Balance the dataset
        n_per_class = n_samples // 2
        
        random.seed(seed)
        if len(human_samples) > n_per_class:
            human_samples = random.sample(human_samples, n_per_class)
        if len(ai_samples) > n_per_class:
            ai_samples = random.sample(ai_samples, n_per_class)
        
        all_samples = human_samples + ai_samples
        random.shuffle(all_samples)
        
        logger.info(
            f"Loaded {len(all_samples)} samples from RAID {split} "
            f"(human={len(human_samples)}, ai={len(ai_samples)})"
        )
        
        return cls(all_samples)


class PadBenDataset(BaseEvalDataset):
    """
    PadBen benchmark dataset loader.
    
    Paper: https://arxiv.org/abs/2511.00416
    HF: https://huggingface.co/datasets/JonathanZha/PADBen
    
    PadBen has multiple configurations:
    - exhaustive-task{1-5}: Binary classification (text, label)
    - sampling-{ratio}-task{1-5}: Different label ratios
    - sentence-pair-task{1-5}: Comparative tasks
    
    For our evals, we use exhaustive-task2 (human original vs LLM generated).
    """
    
    HF_DATASET = "JonathanZha/PADBen"
    DEFAULT_CONFIG = "exhaustive-task2"  # General Text Authorship Detection
    
    def __init__(self, samples: List[EvalSample], name: str = "padben"):
        super().__init__(samples, name)
    
    @classmethod
    def download(
        cls,
        split: str = "train",  # PadBen uses train split
        cache_dir: Optional[str] = None,
        config: str = None,
    ) -> "PadBenDataset":
        """
        Download PadBen dataset.
        
        Args:
            split: Dataset split (PadBen has "train" split)
            cache_dir: Cache directory
            config: PadBen configuration (default: exhaustive-task2)
        """
        config = config or cls.DEFAULT_CONFIG
        logger.info(f"Downloading PadBen dataset (config={config}, split={split})...")
        
        try:
            ds = load_dataset(cls.HF_DATASET, config, split=split, cache_dir=cache_dir)
        except Exception as e:
            logger.error(f"Failed to download PadBen: {e}")
            return cls([])
        
        samples = []
        for idx, item in enumerate(ds):
            # PadBen single-sentence tasks have: idx, sentence, label
            # label: 0 = human, 1 = machine
            label = "ai" if item.get("label", 1) == 1 else "human"
            
            # PadBen uses "sentence" field (not "text")
            text = item.get("sentence", item.get("text", ""))
            
            sample = EvalSample(
                id=f"padben_{config}_{idx}",
                label=label,
                domain="general",
                generator=config,  # Use config as "generator" for tracking
                text=text,
                metadata={"padben_config": config, **dict(item)},
            )
            samples.append(sample)
            samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} samples from PadBen {config}")
        return cls(samples)


def load_eval_dataset(
    name: str,
    split: str = "test",
    cache_dir: Optional[str] = None,
    n_human: int = 1000,
    n_ai: int = 1000,
    min_tokens: int = 100,
    max_tokens: int = 500,
    seed: int = 42,
    **kwargs,
) -> BaseEvalDataset:
    """
    Convenience function to load and prepare evaluation datasets.
    
    Args:
        name: Dataset name ("mage", "raid", "padben")
        split: Dataset split
        cache_dir: Cache directory
        n_human: Number of human samples
        n_ai: Number of AI samples
        min_tokens: Minimum token count
        max_tokens: Maximum token count
        seed: Random seed
        **kwargs: Additional dataset-specific arguments
    
    Returns:
        Prepared dataset
    """
    dataset_classes = {
        "mage": MAGEDataset,
        "raid": RAIDDataset,
        "padben": PadBenDataset,
    }
    
    if name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(dataset_classes.keys())}")
    
    dataset_cls = dataset_classes[name]
    
    # Handle special case for PadBen (uses "train" split)
    if name == "padben":
        split = "train"
    
    dataset = dataset_cls.download(split=split, cache_dir=cache_dir, **kwargs)
    
    # Apply filters
    dataset = dataset.filter_by_length(min_tokens=min_tokens, max_tokens=max_tokens)
    
    # Balance datasets
    # PadBen now has both human and AI samples
    dataset = dataset.balance(n_human=n_human, n_ai=n_ai, seed=seed)
    
    return dataset


def prepare_mage_eval(
    output_path: str = "data/mage_eval.jsonl",
    n_human: int = 1000,
    n_ai: int = 1000,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> MAGEDataset:
    """
    Prepare MAGE evaluation dataset as specified in SPEC.md.
    
    Builder task: Create balanced evaluation set with length filtering.
    """
    logger.info("Preparing MAGE evaluation dataset...")
    
    dataset = load_eval_dataset(
        name="mage",
        split="test",
        cache_dir=cache_dir,
        n_human=n_human,
        n_ai=n_ai,
        seed=seed,
    )
    
    dataset.save(output_path)
    logger.info(f"MAGE eval dataset saved to {output_path}")
    
    return dataset


def prepare_raid_slice(
    output_path: str = "data/raid_slice.jsonl",
    n_human: int = 200,
    n_ai: int = 200,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> RAIDDataset:
    """
    Prepare RAID slice as specified in SPEC.md.
    
    Builder task: Sample lightweight subset for OOD evaluation.
    """
    logger.info("Preparing RAID slice dataset...")
    
    dataset = load_eval_dataset(
        name="raid",
        split="test",
        cache_dir=cache_dir,
        n_human=n_human,
        n_ai=n_ai,
        n_samples=(n_human + n_ai),
        seed=seed,
    )
    
    dataset.save(output_path)
    logger.info(f"RAID slice saved to {output_path}")
    
    return dataset
