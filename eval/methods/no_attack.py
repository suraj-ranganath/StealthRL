"""
No-attack baseline (M0).

Simply returns the original AI-generated text unchanged.
"""

from typing import List
from .base import BaseAttackMethod, AttackOutput


class NoAttack(BaseAttackMethod):
    """
    M0: No attack baseline.
    
    Returns original AI text unchanged for comparison.
    """
    
    def __init__(self):
        super().__init__(name="no_attack", validate_outputs=False)  # No validation for passthrough
    
    def load(self):
        """No models to load."""
        self._loaded = True
    
    def _attack_impl(
        self,
        text: str,
        n_candidates: int = 1,
        **kwargs,
    ) -> AttackOutput:
        """Return original text."""
        return AttackOutput(
            text=text,
            metadata={"method": self.name},
            all_candidates=[text],
            candidate_scores=[1.0],
        )
    
    def attack_batch(
        self,
        texts: List[str],
        n_candidates: int = 1,
        **kwargs,
    ) -> List[AttackOutput]:
        """Efficient batch processing."""
        return [
            AttackOutput(
                text=text,
                original_text=text,
                metadata={"method": self.name},
                all_candidates=[text],
                candidate_scores=[1.0],
                valid=True,
            )
            for text in texts
        ]
