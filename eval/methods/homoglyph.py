"""
Homoglyph attack baseline (M5) - SilverSpeak Implementation.

Based on: "SilverSpeak: Evading AI-Generated Text Detectors using Homoglyphs"
Paper: https://arxiv.org/abs/2406.11239
Repo: https://github.com/ACMCMC/silverspeak

Uses the official SilverSpeak library for research-grade homoglyph attacks.
"""

import logging
import random
import unicodedata
from typing import Dict, List, Optional

from .base import BaseAttackMethod, AttackOutput

logger = logging.getLogger(__name__)

# Check if silverspeak is installed (v0.2+ has top-level API)
try:
    from silverspeak import random_attack as silverspeak_random_attack
    from silverspeak import greedy_attack as silverspeak_greedy_attack
    SILVERSPEAK_AVAILABLE = True
except ImportError:
    try:
        # Legacy import path
        from silverspeak.homoglyphs.random_attack import random_attack as silverspeak_random_attack
        from silverspeak.homoglyphs.targeted_attack import targeted_attack as silverspeak_greedy_attack
        SILVERSPEAK_AVAILABLE = True
    except ImportError:
        SILVERSPEAK_AVAILABLE = False
        logger.warning(
            "SilverSpeak not installed. Install with: pip install silverspeak\n"
            "Falling back to basic homoglyph implementation."
        )


# Fallback homoglyph mappings (used if SilverSpeak not available)
# These are visually similar characters that may bypass tokenization
HOMOGLYPH_MAP: Dict[str, List[str]] = {
    # Latin letters -> Cyrillic/Greek/other lookalikes
    'a': ['а', 'ɑ', 'α'],  # Cyrillic a, Latin alpha, Greek alpha
    'c': ['с', 'ϲ', 'ᴄ'],  # Cyrillic es, Greek lunate sigma
    'e': ['е', 'ε', 'ɛ'],  # Cyrillic ie, Greek epsilon
    'i': ['і', 'ι', 'ɪ'],  # Cyrillic i, Greek iota
    'o': ['о', 'ο', 'ⲟ'],  # Cyrillic o, Greek omicron, Coptic o
    'p': ['р', 'ρ', 'ⲣ'],  # Cyrillic er, Greek rho
    's': ['ѕ', 'ꜱ'],       # Cyrillic dze, small capital S
    'u': ['υ', 'ս', 'ᴜ'],  # Greek upsilon, Armenian vo
    'x': ['х', 'χ', 'ⅹ'],  # Cyrillic ha, Greek chi, Roman numeral
    'y': ['у', 'γ', 'ү'],  # Cyrillic u, Greek gamma
    'A': ['А', 'Α', 'Ꭺ'],  # Cyrillic A, Greek Alpha
    'B': ['В', 'Β', 'Ᏼ'],  # Cyrillic Ve, Greek Beta
    'C': ['С', 'Ϲ', 'Ꮯ'],  # Cyrillic Es, Greek lunate Sigma
    'E': ['Е', 'Ε', 'Ꭼ'],  # Cyrillic Ie, Greek Epsilon
    'H': ['Н', 'Η', 'Ꮋ'],  # Cyrillic En, Greek Eta
    'I': ['І', 'Ι', 'Ꮖ'],  # Cyrillic I, Greek Iota
    'K': ['К', 'Κ', 'Ꮶ'],  # Cyrillic Ka, Greek Kappa
    'M': ['М', 'Μ', 'Ꮇ'],  # Cyrillic Em, Greek Mu
    'N': ['Ν', 'Ꮑ'],       # Greek Nu
    'O': ['О', 'Ο', 'Ꮎ'],  # Cyrillic O, Greek Omicron
    'P': ['Р', 'Ρ', 'Ꮲ'],  # Cyrillic Er, Greek Rho
    'S': ['Ѕ', 'Ꮪ'],       # Cyrillic Dze
    'T': ['Т', 'Τ', 'Ꭲ'],  # Cyrillic Te, Greek Tau
    'X': ['Х', 'Χ', 'Ⅹ'],  # Cyrillic Ha, Greek Chi
    'Y': ['Υ', 'У', 'Ү'],  # Greek Upsilon, Cyrillic U
    'Z': ['Ζ', 'Ꮓ'],       # Greek Zeta
}

# Zero-width characters for insertion
ZERO_WIDTH_CHARS = [
    '\u200b',  # Zero-width space
    '\u200c',  # Zero-width non-joiner
    '\u200d',  # Zero-width joiner
    '\ufeff',  # Zero-width no-break space (BOM)
]


class HomoglyphAttack(BaseAttackMethod):
    """
    M5: Homoglyph substitution attack (SilverSpeak).
    
    Uses the official SilverSpeak library from the paper:
    "SilverSpeak: Evading AI-Generated Text Detectors using Homoglyphs"
    
    Falls back to basic implementation if SilverSpeak not installed.
    """
    
    def __init__(
        self,
        substitution_rate: float = 0.1,  # SilverSpeak default is 10%
        use_silverspeak: bool = True,
        use_targeted: bool = False,  # Use targeted attack (slower but more effective)
        seed: int = 42,
    ):
        """
        Initialize Homoglyph attack.
        
        Args:
            substitution_rate: Fraction of characters to substitute (default: 10% per SilverSpeak)
            use_silverspeak: Whether to use SilverSpeak library (recommended)
            use_targeted: Whether to use targeted attack (slower but better evasion)
            seed: Random seed
        """
        super().__init__(name=f"homoglyph_p{substitution_rate}")
        
        self.substitution_rate = substitution_rate
        self.use_silverspeak = use_silverspeak and SILVERSPEAK_AVAILABLE
        self.use_targeted = use_targeted
        self.seed = seed
        
        if use_silverspeak and not SILVERSPEAK_AVAILABLE:
            logger.warning(
                "SilverSpeak requested but not installed. "
                "Install with: pip install silverspeak"
            )
    
    def load(self):
        """No models to load for basic homoglyph attack."""
        self._loaded = True
        
        if self.use_silverspeak:
            logger.info(f"✓ SilverSpeak homoglyph attack loaded (rate={self.substitution_rate})")
        else:
            logger.info(f"✓ Basic homoglyph attack loaded (rate={self.substitution_rate})")
    
    def _silverspeak_attack(self, text: str) -> str:
        """Apply SilverSpeak homoglyph attack."""
        try:
            if self.use_targeted:
                # Greedy attack - more aggressive substitution
                return silverspeak_greedy_attack(text)
            else:
                # Random attack - faster, uses substitution_rate
                return silverspeak_random_attack(text, self.substitution_rate)
        except Exception as e:
            logger.warning(f"SilverSpeak attack failed: {e}, falling back to basic")
            return self._basic_substitute(text, self.substitution_rate)
    
    def _basic_substitute(self, text: str, rate: float) -> str:
        """Fallback basic homoglyph substitution."""
        random.seed(self.seed)
        
        chars = list(text)
        n_substitute = max(1, int(len(chars) * rate))
        
        # Find substitutable positions
        substitutable = [
            i for i, c in enumerate(chars) 
            if c in HOMOGLYPH_MAP
        ]
        
        if not substitutable:
            return text
        
        # Random sample of positions to substitute
        n_substitute = min(n_substitute, len(substitutable))
        positions = random.sample(substitutable, n_substitute)
        
        # Perform substitutions
        for pos in positions:
            original_char = chars[pos]
            replacement = random.choice(HOMOGLYPH_MAP[original_char])
            chars[pos] = replacement
        
        return ''.join(chars)
    
    def _attack_impl(
        self,
        text: str,
        n_candidates: int = 1,
        **kwargs,
    ) -> AttackOutput:
        """
        Apply homoglyph attack to text.
        
        Note: Homoglyph attacks are deterministic for a given rate.
        n_candidates > 1 not applicable here.
        
        Args:
            text: Original AI-generated text
            n_candidates: Not used (deterministic transform)
        
        Returns:
            AttackOutput with transformed text
        """
        # Apply attack
        if self.use_silverspeak:
            transformed = self._silverspeak_attack(text)
        else:
            transformed = self._basic_substitute(text, self.substitution_rate)
        
        # Count changes
        n_changes = sum(1 for a, b in zip(text, transformed) if a != b)
        
        return AttackOutput(
            text=transformed,
            metadata={
                "method": self.name,
                "substitution_rate": self.substitution_rate,
                "use_silverspeak": self.use_silverspeak,
                "use_targeted": self.use_targeted,
                "n_substitutions": n_changes,
                "original_len": len(text),
                "transformed_len": len(transformed),
            },
            all_candidates=[transformed],
            candidate_scores=[0.0],
        )


def sanitize_text(text: str) -> str:
    """
    Sanitize text by removing homoglyphs and zero-width characters.
    
    This is the "defense-aware" sanitization mentioned in SPEC.md:
    1. Unicode normalize (NFKC)
    2. Remove zero-width characters
    3. Map known homoglyphs back to ASCII
    """
    # Unicode normalize (NFKC decomposition + compatibility)
    text = unicodedata.normalize('NFKC', text)
    
    # Remove zero-width characters
    for zwc in ZERO_WIDTH_CHARS:
        text = text.replace(zwc, '')
    
    # Map homoglyphs back to ASCII
    # Build reverse mapping
    reverse_map = {}
    for ascii_char, homoglyphs in HOMOGLYPH_MAP.items():
        for h in homoglyphs:
            reverse_map[h] = ascii_char
    
    # Apply reverse mapping
    result = []
    for c in text:
        if c in reverse_map:
            result.append(reverse_map[c])
        else:
            result.append(c)
    
    return ''.join(result)


class HomoglyphSweep:
    """
    Generate multiple homoglyph attack variants with different rates.
    
    Used for budget sweep analysis as specified in SPEC.md.
    """
    
    DEFAULT_RATES = [0.001, 0.005, 0.01, 0.02]  # 0.1%, 0.5%, 1%, 2%
    
    def __init__(
        self,
        rates: List[float] = None,
        use_zero_width: bool = False,
        seed: int = 42,
    ):
        """
        Initialize sweep.
        
        Args:
            rates: List of substitution rates to test
            use_zero_width: Whether to include zero-width chars
            seed: Random seed
        """
        self.rates = rates or self.DEFAULT_RATES
        self.use_zero_width = use_zero_width
        self.seed = seed
        
        # Create attack instances
        self.attacks = {
            f"homoglyph_p{rate}": HomoglyphAttack(
                substitution_rate=rate,
                use_zero_width=use_zero_width,
                seed=seed,
            )
            for rate in self.rates
        }
    
    def attack_all(self, text: str) -> Dict[str, AttackOutput]:
        """
        Apply all homoglyph variants to text.
        
        Returns:
            Dict mapping rate name to AttackOutput
        """
        return {
            name: attack.attack(text)
            for name, attack in self.attacks.items()
        }
