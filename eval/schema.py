"""
Output schema definitions for StealthRL evaluation.

Implements the data schemas from SPEC.md §10:
- scores.parquet: Per-sample detector scores
- quality.parquet: Text quality metrics
- metrics.json: Aggregated metrics with CIs

Ensures consistent, structured output across all evaluation runs.
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Score Record Schema (§10.1)
# ============================================================================

@dataclass
class ScoreRecord:
    """
    Schema for scores.parquet - one row per (sample_id, dataset, method, detector, setting).
    
    Matches SPEC.md §10.1 exactly.
    """
    # Identifiers
    sample_id: str
    dataset: str  # mage / raid_slice / padben
    label: str  # human / ai
    method: str  # M0..M5 or method name
    setting: str  # e.g., "N=4_rerank=roberta", "homoglyph_p=0.005"
    candidate_rank: int = 0  # 0 = selected output, 1+ = other candidates
    
    # Text content
    text_in: Optional[str] = None  # Original input text
    text_out: Optional[str] = None  # Attacked/paraphrased output
    
    # Length metrics
    len_tokens_in: Optional[int] = None
    len_tokens_out: Optional[int] = None
    len_ratio: Optional[float] = None
    
    # Detector scores
    detector_name: Optional[str] = None
    detector_score_raw: Optional[float] = None  # Original detector score
    detector_score_ai: Optional[float] = None  # Normalized: higher = more AI
    
    # Metadata
    generator: Optional[str] = None  # Original AI model (if known)
    domain: Optional[str] = None  # Text domain
    timestamp: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityRecord:
    """
    Schema for quality.parquet - one row per (sample_id, dataset, method, setting).
    
    Matches SPEC.md §10.2 exactly.
    """
    # Identifiers
    sample_id: str
    dataset: str
    method: str
    setting: str
    
    # Similarity metrics
    sim_e5: Optional[float] = None  # E5 cosine similarity
    
    # Fluency metrics
    ppl_score: Optional[float] = None  # Perplexity
    
    # Edit metrics
    edit_rate: Optional[float] = None  # Character edit rate
    token_change_rate: Optional[float] = None  # Token change rate
    chrf: Optional[float] = None  # chrF score
    
    # Length metrics
    len_ratio: Optional[float] = None
    len_tokens_in: Optional[int] = None
    len_tokens_out: Optional[int] = None
    
    # Validity
    valid: bool = True
    fail_reason: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MetricsRecord:
    """
    Schema for individual metric in metrics.json.
    
    Matches SPEC.md §10.3.
    """
    # Metric values with confidence intervals
    value: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Data Storage Classes
# ============================================================================

class ScoresTable:
    """
    Manages the scores table with proper schema.
    
    Provides efficient append, query, and export operations.
    """
    
    SCHEMA = {
        'sample_id': 'string',
        'dataset': 'string',
        'label': 'string',
        'method': 'string',
        'setting': 'string',
        'candidate_rank': 'int32',
        'text_in': 'string',
        'text_out': 'string',
        'len_tokens_in': 'Int32',  # Nullable int
        'len_tokens_out': 'Int32',
        'len_ratio': 'float64',
        'detector_name': 'string',
        'detector_score_raw': 'float64',
        'detector_score_ai': 'float64',
        'generator': 'string',
        'domain': 'string',
        'timestamp': 'string',
    }
    
    def __init__(self):
        self.records: List[ScoreRecord] = []
    
    def add(self, record: Union[ScoreRecord, dict]):
        """Add a score record."""
        if isinstance(record, dict):
            record = ScoreRecord(**record)
        self.records.append(record)
    
    def add_batch(self, records: List[Union[ScoreRecord, dict]]):
        """Add multiple records."""
        for r in records:
            self.add(r)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with proper dtypes."""
        if not self.records:
            return pd.DataFrame(columns=list(self.SCHEMA.keys()))
        
        df = pd.DataFrame([r.to_dict() for r in self.records])
        
        # Apply schema dtypes
        for col, dtype in self.SCHEMA.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError):
                    pass  # Keep original dtype if conversion fails
        
        return df
    
    def save_parquet(self, path: str):
        """Save to parquet file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df = self.to_dataframe()
        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} score records to {path}")
    
    def save_csv(self, path: str):
        """Save to CSV file (fallback)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} score records to {path}")
    
    def query(
        self,
        dataset: Optional[str] = None,
        method: Optional[str] = None,
        detector: Optional[str] = None,
        label: Optional[str] = None,
    ) -> List[ScoreRecord]:
        """Query records by criteria."""
        results = self.records
        
        if dataset:
            results = [r for r in results if r.dataset == dataset]
        if method:
            results = [r for r in results if r.method == method]
        if detector:
            results = [r for r in results if r.detector_name == detector]
        if label:
            results = [r for r in results if r.label == label]
        
        return results
    
    def get_scores(
        self,
        dataset: str,
        method: str,
        detector: str,
        label: str,
    ) -> List[float]:
        """Get detector scores for a specific combination."""
        records = self.query(dataset=dataset, method=method, detector=detector, label=label)
        return [r.detector_score_ai for r in records if r.detector_score_ai is not None]


class QualityTable:
    """Manages the quality metrics table."""
    
    SCHEMA = {
        'sample_id': 'string',
        'dataset': 'string',
        'method': 'string',
        'setting': 'string',
        'sim_e5': 'float64',
        'ppl_score': 'float64',
        'edit_rate': 'float64',
        'token_change_rate': 'float64',
        'chrf': 'float64',
        'len_ratio': 'float64',
        'len_tokens_in': 'Int32',
        'len_tokens_out': 'Int32',
        'valid': 'bool',
        'fail_reason': 'string',
    }
    
    def __init__(self):
        self.records: List[QualityRecord] = []
    
    def add(self, record: Union[QualityRecord, dict]):
        """Add a quality record."""
        if isinstance(record, dict):
            record = QualityRecord(**record)
        self.records.append(record)
    
    def add_batch(self, records: List[Union[QualityRecord, dict]]):
        """Add multiple records."""
        for r in records:
            self.add(r)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        if not self.records:
            return pd.DataFrame(columns=list(self.SCHEMA.keys()))
        
        df = pd.DataFrame([r.to_dict() for r in self.records])
        return df
    
    def save_parquet(self, path: str):
        """Save to parquet file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df = self.to_dataframe()
        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} quality records to {path}")
    
    def save_csv(self, path: str):
        """Save to CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} quality records to {path}")


class MetricsStore:
    """
    Stores aggregated metrics in hierarchical structure.
    
    Structure: dataset -> detector -> method -> metric_name -> MetricsRecord
    """
    
    def __init__(self):
        self.data: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
        self.thresholds: Dict[str, float] = {}
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
        }
    
    def add_threshold(self, detector: str, threshold: float, fpr: float = 0.01):
        """Add calibrated threshold for a detector."""
        self.thresholds[detector] = {
            "value": threshold,
            "target_fpr": fpr,
        }
    
    def add_metric(
        self,
        dataset: str,
        detector: str,
        method: str,
        metric_name: str,
        value: float,
        ci_low: Optional[float] = None,
        ci_high: Optional[float] = None,
    ):
        """Add a metric value."""
        if dataset not in self.data:
            self.data[dataset] = {}
        if detector not in self.data[dataset]:
            self.data[dataset][detector] = {}
        if method not in self.data[dataset][detector]:
            self.data[dataset][detector][method] = {}
        
        self.data[dataset][detector][method][metric_name] = {
            "value": value,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    
    def get_metric(
        self,
        dataset: str,
        detector: str,
        method: str,
        metric_name: str,
    ) -> Optional[Dict]:
        """Get a metric value."""
        try:
            return self.data[dataset][detector][method][metric_name]
        except KeyError:
            return None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": self.metadata,
            "thresholds": self.thresholds,
            "metrics": self.data,
        }
    
    def save_json(self, path: str):
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved metrics to {path}")
    
    @classmethod
    def load_json(cls, path: str) -> "MetricsStore":
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        store = cls()
        store.metadata = data.get("metadata", {})
        store.thresholds = data.get("thresholds", {})
        store.data = data.get("metrics", {})
        
        return store


# ============================================================================
# Utility Functions
# ============================================================================

def count_tokens(text: str, tokenizer=None) -> int:
    """
    Count tokens in text.
    
    Uses simple whitespace tokenization if no tokenizer provided.
    """
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    return len(text.split())


def compute_length_metrics(text_in: str, text_out: str, tokenizer=None) -> Dict[str, Any]:
    """Compute length-related metrics."""
    len_in = count_tokens(text_in, tokenizer)
    len_out = count_tokens(text_out, tokenizer)
    
    return {
        "len_tokens_in": len_in,
        "len_tokens_out": len_out,
        "len_ratio": len_out / len_in if len_in > 0 else 0,
    }


def create_score_record(
    sample_id: str,
    dataset: str,
    label: str,
    method: str,
    detector_name: str,
    detector_score: float,
    text_in: str = None,
    text_out: str = None,
    setting: str = "default",
    candidate_rank: int = 0,
    generator: str = None,
    domain: str = None,
    higher_is_ai: bool = True,
) -> ScoreRecord:
    """
    Create a ScoreRecord with computed fields.
    """
    # Compute length metrics if texts provided
    len_metrics = {}
    if text_in and text_out:
        len_metrics = compute_length_metrics(text_in, text_out)
    
    # Normalize score direction
    score_ai = detector_score if higher_is_ai else (1 - detector_score)
    
    return ScoreRecord(
        sample_id=sample_id,
        dataset=dataset,
        label=label,
        method=method,
        setting=setting,
        candidate_rank=candidate_rank,
        text_in=text_in,
        text_out=text_out,
        len_tokens_in=len_metrics.get("len_tokens_in"),
        len_tokens_out=len_metrics.get("len_tokens_out"),
        len_ratio=len_metrics.get("len_ratio"),
        detector_name=detector_name,
        detector_score_raw=detector_score,
        detector_score_ai=score_ai,
        generator=generator,
        domain=domain,
        timestamp=datetime.now().isoformat(),
    )


def create_quality_record(
    sample_id: str,
    dataset: str,
    method: str,
    setting: str,
    text_in: str,
    text_out: str,
    sim_e5: float = None,
    ppl_score: float = None,
    chrf: float = None,
) -> QualityRecord:
    """
    Create a QualityRecord with computed fields.
    """
    from .metrics import (
        compute_edit_rate,
        compute_token_change_rate,
        compute_chrf,
        validate_output,
    )
    
    # Compute metrics
    edit_rate = compute_edit_rate(text_in, text_out)
    token_change = compute_token_change_rate(text_in, text_out)
    
    if chrf is None:
        chrf = compute_chrf(text_in, text_out)
    
    valid, fail_reason = validate_output(text_in, text_out)
    len_metrics = compute_length_metrics(text_in, text_out)
    
    return QualityRecord(
        sample_id=sample_id,
        dataset=dataset,
        method=method,
        setting=setting,
        sim_e5=sim_e5,
        ppl_score=ppl_score,
        edit_rate=edit_rate,
        token_change_rate=token_change,
        chrf=chrf,
        len_ratio=len_metrics.get("len_ratio"),
        len_tokens_in=len_metrics.get("len_tokens_in"),
        len_tokens_out=len_metrics.get("len_tokens_out"),
        valid=valid,
        fail_reason=fail_reason,
    )
