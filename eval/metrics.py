"""
Evaluation metrics for StealthRL.

Implements standard detector evaluation metrics as specified in SPEC.md:
- AUROC
- TPR@1%FPR (a.k.a. T@1%F)
- ASR (Attack Success Rate)
- Bootstrap confidence intervals

Also includes text quality metrics:
- Semantic similarity (E5)
- Perplexity
- Edit metrics
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


@dataclass
class DetectorMetrics:
    """Metrics for a single detector on a single method."""
    detector: str
    method: str
    dataset: str
    
    # Core metrics
    auroc: float
    auroc_ci_low: float
    auroc_ci_high: float
    
    tpr_at_1fpr: float  # TPR @ 1% FPR
    tpr_at_1fpr_ci_low: float
    tpr_at_1fpr_ci_high: float
    
    asr: float  # Attack Success Rate
    asr_ci_low: float
    asr_ci_high: float
    
    # Threshold used
    threshold_1fpr: float
    
    # Sample counts
    n_human: int
    n_ai: int
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass  
class QualityMetrics:
    """Text quality metrics for a single sample."""
    sample_id: str
    method: str
    setting: str
    
    # Similarity
    sim_e5: float
    
    # Fluency
    ppl_score: float
    
    # Edit distance metrics
    edit_rate: float  # Character-level edit rate
    token_change_rate: Optional[float] = None  # Token-level change rate
    chrf: Optional[float] = None  # Character n-gram F-score
    
    # Length metrics
    len_ratio: Optional[float] = None  # len(output) / len(input)
    len_tokens_in: Optional[int] = None
    len_tokens_out: Optional[int] = None
    
    # Validity
    valid: bool = True
    fail_reason: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


def compute_auroc(
    y_true: Union[List[int], np.ndarray],
    y_scores: Union[List[float], np.ndarray],
) -> float:
    """
    Compute Area Under ROC Curve.
    
    Args:
        y_true: True labels (0 = human, 1 = AI)
        y_scores: Predicted AI probability scores
    
    Returns:
        AUROC score
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    # Check for degenerate cases
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true, returning 0.5")
        return 0.5
    
    try:
        return float(roc_auc_score(y_true, y_scores))
    except Exception as e:
        logger.error(f"AUROC computation failed: {e}")
        return 0.5


def compute_threshold_at_fpr(
    human_scores: Union[List[float], np.ndarray],
    target_fpr: float = 0.01,
    higher_is_ai: bool = True,
) -> float:
    """
    Compute detection threshold that achieves target FPR on human samples.
    
    Args:
        human_scores: Detector scores for human-written text
        target_fpr: Target false positive rate (default: 1%)
        higher_is_ai: If True, scores above threshold are classified as AI
    
    Returns:
        Threshold value
    """
    human_scores = np.asarray(human_scores)
    
    if higher_is_ai:
        # Higher score = AI, so threshold at (1-target_fpr) quantile
        # to get target_fpr of humans classified as AI
        threshold = np.quantile(human_scores, 1 - target_fpr)
    else:
        # Lower score = AI, so threshold at target_fpr quantile
        threshold = np.quantile(human_scores, target_fpr)
    
    return float(threshold)


def compute_tpr_at_fpr(
    ai_scores: Union[List[float], np.ndarray],
    threshold: float,
    higher_is_ai: bool = True,
) -> float:
    """
    Compute TPR (True Positive Rate) at a given threshold.
    
    This is the fraction of AI samples correctly identified as AI.
    
    Args:
        ai_scores: Detector scores for AI-generated text
        threshold: Detection threshold (from calibration)
        higher_is_ai: If True, scores above threshold are classified as AI
    
    Returns:
        TPR value
    """
    ai_scores = np.asarray(ai_scores)
    
    if higher_is_ai:
        tpr = np.mean(ai_scores >= threshold)
    else:
        tpr = np.mean(ai_scores <= threshold)
    
    return float(tpr)


def compute_asr(
    ai_scores: Union[List[float], np.ndarray],
    threshold: float,
    higher_is_ai: bool = True,
) -> float:
    """
    Compute ASR (Attack Success Rate) - fraction of AI samples that evade detection.
    
    ASR = 1 - TPR at the given threshold.
    
    Args:
        ai_scores: Detector scores for AI-generated (attacked) text
        threshold: Detection threshold (from calibration)
        higher_is_ai: If True, scores above threshold are classified as AI
    
    Returns:
        ASR value (higher = more successful attack)
    """
    tpr = compute_tpr_at_fpr(ai_scores, threshold, higher_is_ai)
    return 1.0 - tpr


def compute_bootstrap_ci(
    y_true: Union[List[int], np.ndarray],
    y_scores: Union[List[float], np.ndarray],
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    **metric_kwargs,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        y_true: True labels
        y_scores: Predicted scores
        metric_fn: Function to compute metric (takes y_true, y_scores)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        seed: Random seed
        **metric_kwargs: Additional arguments to metric_fn
    
    Returns:
        (point_estimate, ci_low, ci_high)
    """
    np.random.seed(seed)
    
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n_samples = len(y_true)
    
    # Point estimate
    point = metric_fn(y_true, y_scores, **metric_kwargs)
    
    # Bootstrap
    bootstrap_values = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_true = y_true[indices]
        boot_scores = y_scores[indices]
        
        # Skip if only one class in bootstrap sample
        if len(np.unique(boot_true)) < 2:
            continue
        
        try:
            bootstrap_values.append(metric_fn(boot_true, boot_scores, **metric_kwargs))
        except Exception:
            continue
    
    if len(bootstrap_values) < 10:
        logger.warning(f"Only {len(bootstrap_values)} valid bootstrap samples")
        return point, point, point
    
    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_values, alpha / 2 * 100)
    ci_high = np.percentile(bootstrap_values, (1 - alpha / 2) * 100)
    
    return float(point), float(ci_low), float(ci_high)


def calibrate_thresholds(
    human_scores: Dict[str, List[float]],
    target_fpr: float = 0.01,
    higher_is_ai: bool = True,
) -> Dict[str, float]:
    """
    Calibrate detection thresholds for multiple detectors.
    
    IMPORTANT: Thresholds must be computed on HUMAN text only.
    
    Args:
        human_scores: Dict mapping detector names to human sample scores
        target_fpr: Target false positive rate (default: 1%)
        higher_is_ai: Score direction convention
    
    Returns:
        Dict mapping detector names to thresholds
    """
    thresholds = {}
    
    for detector, scores in human_scores.items():
        threshold = compute_threshold_at_fpr(
            scores, 
            target_fpr=target_fpr, 
            higher_is_ai=higher_is_ai,
        )
        thresholds[detector] = threshold
        logger.info(f"Calibrated {detector}: threshold={threshold:.4f} (FPR={target_fpr})")
    
    return thresholds


def compute_detector_metrics(
    human_scores: List[float],
    ai_scores: List[float],
    detector: str,
    method: str,
    dataset: str,
    target_fpr: float = 0.01,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> DetectorMetrics:
    """
    Compute all metrics for a detector-method-dataset combination.
    
    Args:
        human_scores: Detector scores for human samples
        ai_scores: Detector scores for AI/attacked samples
        detector: Detector name
        method: Method name (e.g., "original", "stealthrl", "simple_paraphrase")
        dataset: Dataset name
        target_fpr: Target FPR for threshold calibration
        n_bootstrap: Bootstrap samples for CI
        seed: Random seed
    
    Returns:
        DetectorMetrics with all computed values
    """
    human_scores = np.asarray(human_scores)
    ai_scores = np.asarray(ai_scores)
    
    # Combine for AUROC
    y_true = np.concatenate([np.zeros(len(human_scores)), np.ones(len(ai_scores))])
    y_scores = np.concatenate([human_scores, ai_scores])
    
    # AUROC with bootstrap CI
    auroc, auroc_ci_low, auroc_ci_high = compute_bootstrap_ci(
        y_true, y_scores, compute_auroc, n_bootstrap=n_bootstrap, seed=seed,
    )
    
    # Calibrate threshold on human scores
    threshold = compute_threshold_at_fpr(human_scores, target_fpr=target_fpr)
    
    # TPR@1%FPR
    tpr = compute_tpr_at_fpr(ai_scores, threshold)
    
    # Bootstrap CI for TPR
    def tpr_metric(y_true, y_scores):
        ai_mask = y_true == 1
        return compute_tpr_at_fpr(y_scores[ai_mask], threshold)
    
    _, tpr_ci_low, tpr_ci_high = compute_bootstrap_ci(
        y_true, y_scores, tpr_metric, n_bootstrap=n_bootstrap, seed=seed,
    )
    
    # ASR (attack success rate)
    asr = compute_asr(ai_scores, threshold)
    asr_ci_low = 1 - tpr_ci_high
    asr_ci_high = 1 - tpr_ci_low
    
    return DetectorMetrics(
        detector=detector,
        method=method,
        dataset=dataset,
        auroc=auroc,
        auroc_ci_low=auroc_ci_low,
        auroc_ci_high=auroc_ci_high,
        tpr_at_1fpr=tpr,
        tpr_at_1fpr_ci_low=tpr_ci_low,
        tpr_at_1fpr_ci_high=tpr_ci_high,
        asr=asr,
        asr_ci_low=asr_ci_low,
        asr_ci_high=asr_ci_high,
        threshold_1fpr=threshold,
        n_human=len(human_scores),
        n_ai=len(ai_scores),
    )


# ============================================================================
# Text Quality Metrics
# ============================================================================

class E5SimilarityScorer:
    """
    Semantic similarity scorer using BGE-M3 via Ollama (default) or E5 via HuggingFace.
    
    BGE-M3 is a state-of-the-art multilingual embedding model with better accuracy
    than E5-base-v2, especially for paraphrase detection.
    """
    
    DEFAULT_MODEL = "bge-m3:latest"  # Ollama model
    FALLBACK_MODEL = "intfloat/e5-base-v2"  # HuggingFace fallback
    OLLAMA_URL = "http://localhost:11434/api/embed"
    
    def __init__(self, model_name: str = None, device: str = None, use_ollama: bool = True):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.use_ollama = use_ollama
        self.model = None
        self.tokenizer = None
        self._ollama_available = None
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is running and has BGE-M3."""
        if self._ollama_available is not None:
            return self._ollama_available
        
        try:
            import requests
            resp = requests.post(
                self.OLLAMA_URL,
                json={"model": self.model_name, "input": "test"},
                timeout=5
            )
            self._ollama_available = resp.status_code == 200
        except Exception:
            self._ollama_available = False
        
        return self._ollama_available
    
    def load(self):
        """Load model (Ollama or HuggingFace fallback)."""
        if self.use_ollama and self._check_ollama():
            logger.info(f"Using Ollama BGE-M3 for semantic similarity")
            return  # Ollama doesn't need explicit loading
        
        # Fallback to HuggingFace E5
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        logger.info(f"Loading HuggingFace E5 model: {self.FALLBACK_MODEL}")
        
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.FALLBACK_MODEL)
        self.model = AutoModel.from_pretrained(self.FALLBACK_MODEL).to(self.device)
        self.model.eval()
        self.use_ollama = False
    
    def _encode_ollama(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Ollama BGE-M3."""
        import requests
        
        embeddings = []
        for text in texts:
            resp = requests.post(
                self.OLLAMA_URL,
                json={"model": self.model_name, "input": text},
                timeout=30
            )
            resp.raise_for_status()
            emb = resp.json()["embeddings"][0]
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-10)
    
    def _encode_hf(self, texts: List[str]) -> np.ndarray:
        """Encode texts using HuggingFace E5."""
        import torch
        
        # E5 prefix for similarity tasks
        texts = [f"query: {t}" for t in texts]
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = embeddings.cpu().numpy()
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-10)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        if self.use_ollama and self._check_ollama():
            return self._encode_ollama(texts)
        
        if self.model is None:
            self.load()
        
        return self._encode_hf(texts)
    
    def compute_similarity(
        self,
        original_texts: List[str],
        paraphrased_texts: List[str],
    ) -> List[float]:
        """Compute cosine similarity between original and paraphrased texts."""
        orig_emb = self.encode(original_texts)
        para_emb = self.encode(paraphrased_texts)
        
        # Cosine similarity (embeddings are normalized)
        similarities = np.sum(orig_emb * para_emb, axis=1)
        
        return similarities.tolist()


class PerplexityScorer:
    """Perplexity scorer using a reference language model."""
    
    DEFAULT_MODEL = "gpt2"
    
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.model = None
        self.tokenizer = None
    
    def load(self):
        """Load language model."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        logger.info(f"Loading PPL model: {self.model_name}")
        
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def compute_perplexity(self, texts: List[str]) -> List[float]:
        """Compute perplexity for each text."""
        import torch
        
        if self.model is None:
            self.load()
        
        perplexities = []
        
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs.input_ids)
                ppl = torch.exp(outputs.loss).item()
            
            perplexities.append(ppl)
        
        return perplexities


def compute_edit_rate(original: str, paraphrased: str) -> float:
    """
    Compute edit rate (character-level edit distance / original length).
    """
    from difflib import SequenceMatcher
    
    matcher = SequenceMatcher(None, original, paraphrased)
    ratio = matcher.ratio()
    
    # Edit rate = 1 - similarity ratio
    return 1.0 - ratio


def compute_token_change_rate(original: str, paraphrased: str) -> float:
    """
    Compute token-level change rate.
    """
    orig_tokens = set(original.lower().split())
    para_tokens = set(paraphrased.lower().split())
    
    if len(orig_tokens) == 0:
        return 1.0
    
    intersection = orig_tokens & para_tokens
    change_rate = 1.0 - len(intersection) / len(orig_tokens)
    
    return change_rate


def compute_chrf(reference: str, hypothesis: str, beta: float = 2.0) -> float:
    """
    Compute chrF score (character n-gram F-score).
    
    chrF is a character-level metric that correlates well with human judgments,
    especially for morphologically rich languages and paraphrase evaluation.
    
    Args:
        reference: Reference text (original)
        hypothesis: Hypothesis text (paraphrase)
        beta: Weight of recall vs precision (default: 2.0 = recall weighted 2x)
    
    Returns:
        chrF score in [0, 1] (higher = more similar)
    """
    def get_char_ngrams(text: str, n: int) -> Dict[str, int]:
        """Extract character n-grams with counts."""
        ngrams = {}
        text = text.replace(" ", "")  # Remove spaces for character n-grams
        for i in range(len(text) - n + 1):
            ngram = text[i:i+n]
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams
    
    def compute_ngram_fscore(ref_ngrams: Dict[str, int], hyp_ngrams: Dict[str, int], beta: float) -> float:
        """Compute F-score for a single n-gram order."""
        # Count matches
        matches = 0
        for ngram, count in hyp_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
        
        # Precision and recall
        hyp_total = sum(hyp_ngrams.values())
        ref_total = sum(ref_ngrams.values())
        
        precision = matches / hyp_total if hyp_total > 0 else 0
        recall = matches / ref_total if ref_total > 0 else 0
        
        # F-score
        if precision + recall == 0:
            return 0
        
        beta_sq = beta ** 2
        return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    
    # Compute chrF over n-gram orders 1-6 (standard)
    max_n = 6
    fscores = []
    
    for n in range(1, max_n + 1):
        ref_ngrams = get_char_ngrams(reference, n)
        hyp_ngrams = get_char_ngrams(hypothesis, n)
        
        if len(ref_ngrams) == 0 and len(hyp_ngrams) == 0:
            fscores.append(1.0)  # Both empty at this n-gram order
        else:
            fscores.append(compute_ngram_fscore(ref_ngrams, hyp_ngrams, beta))
    
    # Average over all n-gram orders
    return sum(fscores) / len(fscores) if fscores else 0.0


def compute_word_chrf(reference: str, hypothesis: str, beta: float = 2.0) -> float:
    """
    Compute word-level chrF (sometimes called chrF++).
    
    Combines character n-gram F-score with word n-gram F-score.
    """
    # Character-level chrF
    char_chrf = compute_chrf(reference, hypothesis, beta)
    
    # Word-level component
    def get_word_ngrams(text: str, n: int) -> Dict[str, int]:
        words = text.lower().split()
        ngrams = {}
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams
    
    def compute_word_fscore(ref_words: List[str], hyp_words: List[str], beta: float) -> float:
        ref_set = set(ref_words)
        hyp_set = set(hyp_words)
        
        matches = len(ref_set & hyp_set)
        precision = matches / len(hyp_set) if hyp_set else 0
        recall = matches / len(ref_set) if ref_set else 0
        
        if precision + recall == 0:
            return 0
        
        beta_sq = beta ** 2
        return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    
    word_fscore = compute_word_fscore(
        reference.lower().split(),
        hypothesis.lower().split(),
        beta
    )
    
    # Average char and word components
    return (char_chrf + word_fscore) / 2


def validate_output(
    original: str,
    paraphrased: str,
    min_words: int = 10,
    max_length_ratio: float = 3.0,
) -> Tuple[bool, Optional[str]]:
    """
    Validate paraphrase output.
    
    Returns:
        (is_valid, fail_reason)
    """
    if not paraphrased or not paraphrased.strip():
        return False, "empty_output"
    
    para_words = len(paraphrased.split())
    
    if para_words < min_words:
        return False, f"too_short_{para_words}_words"
    
    orig_len = len(original)
    para_len = len(paraphrased)
    
    if orig_len > 0 and para_len / orig_len > max_length_ratio:
        return False, f"too_long_ratio_{para_len/orig_len:.2f}"
    
    return True, None


def compute_quality_metrics(
    original_texts: List[str],
    paraphrased_texts: List[str],
    sample_ids: List[str],
    method: str,
    setting: str,
    similarity_scorer: Optional[E5SimilarityScorer] = None,
    perplexity_scorer: Optional[PerplexityScorer] = None,
    compute_chrf_scores: bool = True,
) -> List[QualityMetrics]:
    """
    Compute quality metrics for paraphrased outputs.
    
    Args:
        original_texts: Original texts
        paraphrased_texts: Paraphrased outputs
        sample_ids: Sample identifiers
        method: Method name
        setting: Setting string (e.g., "N=4_rerank=roberta")
        similarity_scorer: E5 similarity scorer (will be created if None)
        perplexity_scorer: Perplexity scorer (will be created if None)
        compute_chrf_scores: Whether to compute chrF (slower but recommended)
    
    Returns:
        List of QualityMetrics
    """
    # Initialize scorers if needed
    if similarity_scorer is None:
        similarity_scorer = E5SimilarityScorer()
    if perplexity_scorer is None:
        perplexity_scorer = PerplexityScorer()
    
    # Compute similarities
    similarities = similarity_scorer.compute_similarity(original_texts, paraphrased_texts)
    
    # Compute perplexities
    perplexities = perplexity_scorer.compute_perplexity(paraphrased_texts)
    
    # Compute per-sample metrics
    results = []
    for i, (orig, para, sid) in enumerate(zip(original_texts, paraphrased_texts, sample_ids)):
        valid, fail_reason = validate_output(orig, para)
        edit_rate = compute_edit_rate(orig, para)
        token_change = compute_token_change_rate(orig, para)
        
        # Compute chrF if requested
        chrf_score = None
        if compute_chrf_scores:
            chrf_score = compute_chrf(orig, para)
        
        results.append(QualityMetrics(
            sample_id=sid,
            method=method,
            setting=setting,
            sim_e5=similarities[i],
            ppl_score=perplexities[i],
            edit_rate=edit_rate,
            token_change_rate=token_change,
            chrf=chrf_score,
            valid=valid,
            fail_reason=fail_reason,
        ))
    
    return results


def save_thresholds(thresholds: Dict[str, float], path: str):
    """Save calibrated thresholds to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(thresholds, f, indent=2)
    
    logger.info(f"Saved thresholds to {path}")


def load_thresholds(path: str) -> Dict[str, float]:
    """Load calibrated thresholds from JSON."""
    with open(path, "r") as f:
        return json.load(f)
