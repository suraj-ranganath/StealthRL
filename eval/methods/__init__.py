"""
Attack methods for StealthRL evaluation.

Implements baseline methods as specified in SPEC.md:
- M0: No attack (original AI text)
- M1: Simple paraphrase (base LM without RL)
- M2: StealthRL (RL-trained policy) - supports Tinker and local PEFT
- M3: Adversarial Paraphrasing (guided selection fallback)
- M4: AuthorMist (HF model)
- M5: Homoglyph (text transform)
"""

from .base import BaseAttackMethod, AttackOutput
from .no_attack import NoAttack
from .simple_paraphrase import SimpleParaphrase, SimpleParaphraseWithReranking
from .stealthrl import (
    StealthRLAttack, 
    StealthRLAttackWithReranking, 
    StealthRLTinker,
    load_stealthrl_checkpoint,
)
from .adversarial_paraphrasing import AdversarialParaphrasing, AdversarialParaphrasingEnsemble
from .authormist import AuthorMist, AuthorMistOllama, AuthorMistFallback
from .homoglyph import HomoglyphAttack, HomoglyphSweep, sanitize_text

__all__ = [
    # Base
    "BaseAttackMethod",
    "AttackOutput",
    # Methods
    "NoAttack",
    "SimpleParaphrase",
    "SimpleParaphraseWithReranking",
    "StealthRLAttack",
    "StealthRLAttackWithReranking",
    "StealthRLTinker",
    "load_stealthrl_checkpoint",
    "AdversarialParaphrasing",
    "AdversarialParaphrasingEnsemble",
    "AuthorMist",
    "AuthorMistOllama",
    "AuthorMistFallback",
    "HomoglyphAttack",
    "HomoglyphSweep",
    "sanitize_text",
]


# Method registry for CLI
METHOD_REGISTRY = {
    "m0": NoAttack,
    "no_attack": NoAttack,
    "m1": SimpleParaphrase,
    "simple_paraphrase": SimpleParaphrase,
    "m2": StealthRLTinker,  # Default to Tinker for cloud inference
    "stealthrl": StealthRLTinker,
    "m2_local": StealthRLAttack,  # Local PEFT version
    "stealthrl_local": StealthRLAttack,
    "m3": AdversarialParaphrasing,
    "adversarial_paraphrasing": AdversarialParaphrasing,
    "m3_ensemble": AdversarialParaphrasingEnsemble,
    "adversarial_paraphrasing_ensemble": AdversarialParaphrasingEnsemble,
    "m4": AuthorMistOllama,  # Default to Ollama for M4 Mac
    "authormist": AuthorMistOllama,
    "m4_hf": AuthorMist,  # HuggingFace version (requires GPU)
    "authormist_hf": AuthorMist,
    "m5": HomoglyphAttack,
    "homoglyph": HomoglyphAttack,
}


# Guidance-specific variants for transfer ablation study (§7.2)
GUIDANCE_VARIANTS = {
    "m3_roberta": ("roberta", "Adversarial paraphrasing guided by RoBERTa"),
    "m3_fastdetect": ("fast_detectgpt", "Adversarial paraphrasing guided by Fast-DetectGPT"),
    "m3_ensemble": ("ensemble", "Adversarial paraphrasing guided by ensemble"),
}


def get_method(name: str, **kwargs) -> BaseAttackMethod:
    """
    Get attack method by name.
    
    Supports standard methods (m0-m5) and guidance-specific variants
    for transfer ablation study.
    
    For M2 (StealthRL), pass checkpoint_json for Tinker or checkpoint_path for local PEFT.
    """
    # Handle guidance-specific variants
    if name in GUIDANCE_VARIANTS:
        guidance, desc = GUIDANCE_VARIANTS[name]
        if guidance == "ensemble":
            return AdversarialParaphrasingEnsemble(**kwargs)
        else:
            return AdversarialParaphrasing(guidance_detector=guidance, **kwargs)
    
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHOD_REGISTRY.keys())} + {list(GUIDANCE_VARIANTS.keys())}")
    return METHOD_REGISTRY[name](**kwargs)
