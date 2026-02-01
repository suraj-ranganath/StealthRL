"""
Base class for attack methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging
import time

logger = logging.getLogger(__name__)


# Validity check thresholds (from SPEC.md §13)
MIN_WORDS = 10
MAX_LENGTH_RATIO = 3.0


@dataclass
class AttackOutput:
    """Output from an attack method."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional: multiple candidates with scores
    all_candidates: Optional[List[str]] = None
    candidate_scores: Optional[List[float]] = None
    
    # Validity info
    valid: bool = True
    fail_reason: Optional[str] = None
    
    # Original text for reference
    original_text: Optional[str] = None


def validate_attack_output(
    original: str,
    output: str,
    min_words: int = MIN_WORDS,
    max_length_ratio: float = MAX_LENGTH_RATIO,
) -> tuple[bool, Optional[str]]:
    """
    Validate attack output meets quality thresholds.
    
    From SPEC.md §13:
    - non-empty
    - >= 10 words
    - <= 3× original length
    
    Args:
        original: Original input text
        output: Attack output text
        min_words: Minimum word count
        max_length_ratio: Maximum length ratio vs original
    
    Returns:
        (is_valid, fail_reason)
    """
    if not output or not output.strip():
        return False, "empty_output"
    
    output_words = len(output.split())
    
    if output_words < min_words:
        return False, f"too_short_{output_words}_words"
    
    orig_len = len(original)
    output_len = len(output)
    
    if orig_len > 0:
        length_ratio = output_len / orig_len
        if length_ratio > max_length_ratio:
            return False, f"too_long_ratio_{length_ratio:.2f}"
    
    return True, None


class BaseAttackMethod(ABC):
    """Base class for attack methods."""
    
    def __init__(
        self,
        name: str,
        validate_outputs: bool = True,
        min_words: int = MIN_WORDS,
        max_length_ratio: float = MAX_LENGTH_RATIO,
    ):
        self.name = name
        self._loaded = False
        
        # Validity check settings
        self.validate_outputs = validate_outputs
        self.min_words = min_words
        self.max_length_ratio = max_length_ratio
    
    @abstractmethod
    def load(self):
        """Load any required models."""
        pass
    
    @abstractmethod
    def _attack_impl(
        self,
        text: str,
        n_candidates: int = 1,
        **kwargs,
    ) -> AttackOutput:
        """
        Internal attack implementation.
        
        Subclasses should implement this method.
        """
        pass
    
    def attack(
        self,
        text: str,
        n_candidates: int = 1,
        **kwargs,
    ) -> AttackOutput:
        """
        Generate attacked version of text with validation.
        
        Args:
            text: Original AI-generated text
            n_candidates: Number of candidates to generate
            **kwargs: Method-specific arguments
        
        Returns:
            AttackOutput with best candidate and metadata
        """
        if not self._loaded:
            self.load()
        
        # Run attack implementation
        result = self._attack_impl(text, n_candidates, **kwargs)
        
        # Store original text
        result.original_text = text
        
        # Validate output if enabled
        if self.validate_outputs:
            valid, fail_reason = validate_attack_output(
                text,
                result.text,
                min_words=self.min_words,
                max_length_ratio=self.max_length_ratio,
            )
            result.valid = valid
            result.fail_reason = fail_reason
            
            if not valid:
                logger.warning(f"Invalid output from {self.name}: {fail_reason}")
        
        return result
    
    def attack_batch(
        self,
        texts: List[str],
        n_candidates: int = 1,
        **kwargs,
    ) -> List[AttackOutput]:
        """
        Batch attack (default: sequential).
        
        Override for parallel processing.
        """
        if not self._loaded:
            self.load()
        
        results = []
        batch_start = time.time()
        log_interval = max(1, len(texts) // 10)  # Log ~10 updates
        
        for i, text in enumerate(texts):
            try:
                sample_start = time.time()
                result = self.attack(text, n_candidates, **kwargs)
                results.append(result)
                sample_elapsed = time.time() - sample_start
                
                # Log progress every log_interval samples
                if (i + 1) % log_interval == 0 or i == len(texts) - 1:
                    batch_elapsed = time.time() - batch_start
                    rate = (i + 1) / batch_elapsed if batch_elapsed > 0 else 0
                    eta = (len(texts) - i - 1) / rate if rate > 0 else 0
                    logger.info(f"[{self.name}] Progress: {i+1}/{len(texts)} ({sample_elapsed:.2f}s/sample, {rate:.2f} samples/s, ETA: {eta:.0f}s)")
                    
            except Exception as e:
                logger.error(f"Attack failed for sample {i}: {e}")
                # Return original text as fallback
                results.append(AttackOutput(
                    text=text,
                    original_text=text,
                    valid=False,
                    fail_reason=f"attack_exception: {str(e)}",
                    metadata={"error": str(e)},
                ))
        
        return results
