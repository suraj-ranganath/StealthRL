"""
Text sanitization module for defense-aware evaluation.

Implements sanitization as discussed in homoglyph attack paper:
https://arxiv.org/abs/2406.11239

Sanitization steps:
1. Unicode normalization (NFKC)
2. Remove zero-width characters
3. Map known homoglyphs back to ASCII
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SanitizationResult:
    """Result of text sanitization."""
    original: str
    sanitized: str
    zero_width_removed: int = 0
    homoglyphs_replaced: int = 0
    unicode_normalized: bool = False
    changes: List[Tuple[int, str, str]] = field(default_factory=list)  # (position, before, after)
    
    @property
    def was_modified(self) -> bool:
        """Check if sanitization made any changes."""
        return self.original != self.sanitized
    
    @property
    def total_changes(self) -> int:
        """Total number of character changes."""
        return self.zero_width_removed + self.homoglyphs_replaced


# Zero-width and invisible characters to remove
ZERO_WIDTH_CHARS = [
    '\u200b',  # Zero Width Space
    '\u200c',  # Zero Width Non-Joiner
    '\u200d',  # Zero Width Joiner
    '\u200e',  # Left-to-Right Mark
    '\u200f',  # Right-to-Left Mark
    '\u2060',  # Word Joiner
    '\u2061',  # Function Application
    '\u2062',  # Invisible Times
    '\u2063',  # Invisible Separator
    '\u2064',  # Invisible Plus
    '\ufeff',  # Zero Width No-Break Space (BOM)
    '\u00ad',  # Soft Hyphen
    '\u034f',  # Combining Grapheme Joiner
]

# Common homoglyph mappings (visual similar -> ASCII)
# Based on common attack patterns
HOMOGLYPH_MAP = {
    # Cyrillic -> Latin
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x',
    'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O',
    'Р': 'P', 'С': 'C', 'Т': 'T', 'У': 'Y', 'Х': 'X',
    
    # Greek -> Latin
    'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Ζ': 'Z', 'Η': 'H', 'Ι': 'I', 'Κ': 'K',
    'Μ': 'M', 'Ν': 'N', 'Ο': 'O', 'Ρ': 'P', 'Τ': 'T', 'Υ': 'Y', 'Χ': 'X',
    'ο': 'o', 'ν': 'v',
    
    # Math/Special symbols -> Latin/Numbers
    'ℓ': 'l', 'ⅰ': 'i', 'ⅱ': 'ii', 'ⅲ': 'iii', 'ⅳ': 'iv', 'ⅴ': 'v',
    '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
    '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
    
    # Common confusables
    'ⅼ': 'l', 'ⅽ': 'c', 'ⅾ': 'd', 'ⅿ': 'm',
    '‐': '-', '‑': '-', '‒': '-', '–': '-', '—': '-',  # Various dashes
    ''': "'", ''': "'", '"': '"', '"': '"',  # Quotes
    '…': '...', '․': '.', '‥': '..',  # Dots
    
    # Fullwidth -> ASCII
    'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E', 'Ｆ': 'F',
    'Ｇ': 'G', 'Ｈ': 'H', 'Ｉ': 'I', 'Ｊ': 'J', 'Ｋ': 'K', 'Ｌ': 'L',
    'Ｍ': 'M', 'Ｎ': 'N', 'Ｏ': 'O', 'Ｐ': 'P', 'Ｑ': 'Q', 'Ｒ': 'R',
    'Ｓ': 'S', 'Ｔ': 'T', 'Ｕ': 'U', 'Ｖ': 'V', 'Ｗ': 'W', 'Ｘ': 'X',
    'Ｙ': 'Y', 'Ｚ': 'Z',
    'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e', 'ｆ': 'f',
    'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i', 'ｊ': 'j', 'ｋ': 'k', 'ｌ': 'l',
    'ｍ': 'm', 'ｎ': 'n', 'ｏ': 'o', 'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r',
    'ｓ': 's', 'ｔ': 't', 'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x',
    'ｙ': 'y', 'ｚ': 'z',
}


def remove_zero_width(text: str) -> str:
    """Remove zero-width and invisible characters."""
    for char in ZERO_WIDTH_CHARS:
        text = text.replace(char, '')
    return text


def normalize_unicode(text: str, form: str = 'NFKC') -> str:
    """
    Apply Unicode normalization.
    
    NFKC is recommended as it normalizes compatibility characters
    to their canonical equivalents.
    """
    return unicodedata.normalize(form, text)


def replace_homoglyphs(text: str, mapping: Dict[str, str] = None) -> str:
    """Replace known homoglyphs with ASCII equivalents."""
    mapping = mapping or HOMOGLYPH_MAP
    result = []
    for char in text:
        result.append(mapping.get(char, char))
    return ''.join(result)


def sanitize(
    text: str,
    normalize: bool = True,
    remove_zero: bool = True,
    map_homoglyphs: bool = True,
    normalization_form: str = 'NFKC',
) -> str:
    """
    Apply full text sanitization.
    
    Sanitization pipeline:
    1. Unicode normalization (NFKC by default)
    2. Remove zero-width characters
    3. Map known homoglyphs to ASCII
    
    Args:
        text: Input text
        normalize: Apply Unicode normalization
        remove_zero: Remove zero-width characters
        map_homoglyphs: Replace homoglyphs with ASCII
        normalization_form: Unicode normalization form
    
    Returns:
        Sanitized text
    """
    if normalize:
        text = normalize_unicode(text, normalization_form)
    
    if remove_zero:
        text = remove_zero_width(text)
    
    if map_homoglyphs:
        text = replace_homoglyphs(text)
    
    return text


def compute_sanitization_diff(original: str, sanitized: str) -> Dict:
    """
    Compute statistics about what changed during sanitization.
    
    Returns:
        Dict with sanitization statistics
    """
    orig_len = len(original)
    san_len = len(sanitized)
    
    # Count changed characters
    changed_chars = sum(1 for a, b in zip(original, sanitized) if a != b)
    removed_chars = orig_len - san_len
    
    # Identify specific changes
    changes = []
    for i, (a, b) in enumerate(zip(original, sanitized)):
        if a != b:
            changes.append({
                'position': i,
                'original': a,
                'original_code': ord(a),
                'replaced': b,
            })
    
    return {
        'original_length': orig_len,
        'sanitized_length': san_len,
        'chars_changed': changed_chars,
        'chars_removed': removed_chars,
        'change_rate': (changed_chars + removed_chars) / orig_len if orig_len > 0 else 0,
        'changes': changes[:20],  # Limit to first 20 changes
    }


def sanitize_batch(texts: List[str], **kwargs) -> List[str]:
    """Sanitize a batch of texts."""
    return [sanitize(text, **kwargs) for text in texts]


def run_sanitize_evaluation(
    texts: List[str],
    detectors: Dict,
    labels: List[str] = None,
    methods: List[str] = None,
) -> Dict:
    """
    Run sanitization evaluation.
    
    Scores texts before and after sanitization with all detectors.
    
    Args:
        texts: List of texts to evaluate
        detectors: Dict of detector name -> detector instance
        labels: Optional labels for each text
        methods: Optional method names for each text
    
    Returns:
        Dict with before/after scores and statistics
    """
    logger.info(f"Running sanitize evaluation on {len(texts)} texts...")
    
    results = {
        'before': {},
        'after': {},
        'diff': {},
        'sanitization_stats': [],
    }
    
    # Sanitize all texts
    sanitized_texts = sanitize_batch(texts)
    
    # Compute sanitization statistics
    for orig, san in zip(texts, sanitized_texts):
        stats = compute_sanitization_diff(orig, san)
        results['sanitization_stats'].append(stats)
    
    # Score with each detector before and after
    for det_name, detector in detectors.items():
        logger.info(f"  Scoring with {det_name}...")
        
        # Before sanitization
        before_scores = detector.get_scores(texts)
        if isinstance(before_scores, float):
            before_scores = [before_scores]
        results['before'][det_name] = before_scores
        
        # After sanitization
        after_scores = detector.get_scores(sanitized_texts)
        if isinstance(after_scores, float):
            after_scores = [after_scores]
        results['after'][det_name] = after_scores
        
        # Compute diff
        results['diff'][det_name] = [
            a - b for a, b in zip(after_scores, before_scores)
        ]
    
    logger.info("Sanitize evaluation complete")
    return results


def create_sanitize_report(
    eval_results: Dict,
    detector_names: List[str],
    method_name: str = "homoglyph",
) -> str:
    """
    Create markdown report of sanitization effects.
    
    Returns:
        Markdown-formatted report
    """
    lines = [
        f"# Sanitization Defense Evaluation",
        f"",
        f"Method: **{method_name}**",
        f"",
        f"## Detection Scores Before/After Sanitization",
        f"",
        f"| Detector | Before (mean) | After (mean) | Δ Score | Δ TPR |",
        f"|----------|---------------|--------------|---------|-------|",
    ]
    
    for det_name in detector_names:
        before = eval_results['before'].get(det_name, [])
        after = eval_results['after'].get(det_name, [])
        
        if before and after:
            mean_before = sum(before) / len(before)
            mean_after = sum(after) / len(after)
            delta = mean_after - mean_before
            
            lines.append(
                f"| {det_name} | {mean_before:.4f} | {mean_after:.4f} | "
                f"{delta:+.4f} | - |"
            )
    
    # Add sanitization statistics
    stats = eval_results.get('sanitization_stats', [])
    if stats:
        mean_change_rate = sum(s['change_rate'] for s in stats) / len(stats)
        lines.extend([
            f"",
            f"## Sanitization Statistics",
            f"",
            f"- Total samples: {len(stats)}",
            f"- Mean character change rate: {mean_change_rate:.4%}",
        ])
    
    return '\n'.join(lines)
