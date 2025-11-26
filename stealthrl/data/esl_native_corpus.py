"""
ESL vs Native academic writing corpus loader for fairness evaluation.

Expected directory structure:
    data/esl/toefl11.jsonl
    data/esl/icnale_written.jsonl
    data/esl/ellipse.jsonl
    data/native/native_academic.jsonl
    
Each JSONL line must contain:
    - id: str (unique identifier)
    - text: str (essay/writing sample)
    - source: str (corpus name, e.g., "TOEFL11", "ICNALE", "NATIVE_ACAD")
    - is_esl: bool (True for ESL, False for native)
    - proficiency_level: Optional[str] (e.g., "low", "medium", "high", "C1", "C2")
    - prompt_id: Optional[str] (writing prompt/topic identifier)
    
Output splits:
    data/processed/esl_native_dev.jsonl
    data/processed/esl_native_test.jsonl
"""

import json
import logging
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ESLNativeRecord:
    """Schema for ESL/native academic writing samples."""
    
    id: str
    text: str
    source: str
    is_esl: bool
    proficiency_level: Optional[str] = None
    prompt_id: Optional[str] = None
    split: Optional[str] = None  # "dev" or "test"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ESLNativeRecord":
        """Create record from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            source=data["source"],
            is_esl=data["is_esl"],
            proficiency_level=data.get("proficiency_level"),
            prompt_id=data.get("prompt_id"),
            split=data.get("split"),
        )


def load_esl_native_jsonl(
    file_path: Path,
    validate: bool = True,
) -> List[ESLNativeRecord]:
    """
    Load ESL/native records from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        validate: Whether to validate required fields
        
    Returns:
        List of ESLNativeRecord objects
        
    Raises:
        ValueError: If validation fails
    """
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return []
    
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Validate required fields
                if validate:
                    required_fields = ["id", "text", "source", "is_esl"]
                    missing = [f for f in required_fields if f not in data]
                    if missing:
                        raise ValueError(
                            f"Line {line_num}: Missing required fields: {missing}"
                        )
                
                record = ESLNativeRecord.from_dict(data)
                records.append(record)
                
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: Invalid JSON - {e}")
                continue
            except Exception as e:
                logger.error(f"Line {line_num}: Error loading record - {e}")
                continue
    
    logger.info(f"Loaded {len(records)} records from {file_path}")
    return records


def load_all_esl_native_data(
    data_dir: Path,
) -> tuple[List[ESLNativeRecord], List[ESLNativeRecord]]:
    """
    Load all ESL and native data from standard locations.
    
    Args:
        data_dir: Root data directory (should contain esl/ and native/ subdirs)
        
    Returns:
        (esl_records, native_records) tuple
    """
    esl_dir = data_dir / "esl"
    native_dir = data_dir / "native"
    
    # Load ESL corpora
    esl_files = [
        esl_dir / "toefl11.jsonl",
        esl_dir / "icnale_written.jsonl",
        esl_dir / "ellipse.jsonl",
    ]
    
    esl_records = []
    for file_path in esl_files:
        records = load_esl_native_jsonl(file_path, validate=True)
        esl_records.extend(records)
        if records:
            logger.info(f"  Loaded {len(records)} ESL records from {file_path.name}")
    
    # Load native corpora
    native_files = [
        native_dir / "native_academic.jsonl",
    ]
    
    native_records = []
    for file_path in native_files:
        records = load_esl_native_jsonl(file_path, validate=True)
        native_records.extend(records)
        if records:
            logger.info(f"  Loaded {len(records)} native records from {file_path.name}")
    
    logger.info(f"Total: {len(esl_records)} ESL, {len(native_records)} native records")
    return esl_records, native_records


def build_esl_native_eval_split(
    data_dir: Path,
    output_dir: Path,
    dev_size: int = 200,
    test_size: int = 500,
    esl_ratio: float = 0.4,
    seed: int = 42,
    stratify_by_source: bool = True,
    stratify_by_proficiency: bool = True,
) -> tuple[List[ESLNativeRecord], List[ESLNativeRecord]]:
    """
    Build dev and test splits with ESL/native balance.
    
    Strategy:
        1. Load all ESL and native data
        2. Stratify by source to ensure diversity
        3. Optionally stratify by proficiency level
        4. Sample to achieve target ESL ratio (~40% ESL, ~60% native)
        5. Save splits to processed/ directory
    
    Args:
        data_dir: Root data directory containing esl/ and native/ subdirs
        output_dir: Output directory for processed splits
        dev_size: Total size of dev split
        test_size: Total size of test split
        esl_ratio: Target ratio of ESL samples (0.4 = 40% ESL)
        seed: Random seed for reproducibility
        stratify_by_source: Ensure samples from multiple sources
        stratify_by_proficiency: Balance proficiency levels if available
        
    Returns:
        (dev_records, test_records) tuple
    """
    random.seed(seed)
    
    # Load all data
    esl_records, native_records = load_all_esl_native_data(data_dir)
    
    if not esl_records or not native_records:
        raise ValueError("Need both ESL and native data to build splits")
    
    # Group by source for stratification
    esl_by_source = defaultdict(list)
    native_by_source = defaultdict(list)
    
    for rec in esl_records:
        esl_by_source[rec.source].append(rec)
    
    for rec in native_records:
        native_by_source[rec.source].append(rec)
    
    logger.info("ESL sources: " + ", ".join(
        f"{src}={len(recs)}" for src, recs in esl_by_source.items()
    ))
    logger.info("Native sources: " + ", ".join(
        f"{src}={len(recs)}" for src, recs in native_by_source.items()
    ))
    
    def sample_stratified(
        records_by_source: Dict[str, List[ESLNativeRecord]],
        target_size: int,
    ) -> List[ESLNativeRecord]:
        """Sample records ensuring diversity across sources."""
        all_sources = list(records_by_source.keys())
        
        if not all_sources:
            return []
        
        # Calculate samples per source
        samples_per_source = target_size // len(all_sources)
        remainder = target_size % len(all_sources)
        
        sampled = []
        for i, source in enumerate(all_sources):
            source_records = records_by_source[source]
            
            # Add one extra to first sources for remainder
            n_samples = samples_per_source + (1 if i < remainder else 0)
            n_samples = min(n_samples, len(source_records))
            
            # Shuffle and sample
            random.shuffle(source_records)
            sampled.extend(source_records[:n_samples])
        
        return sampled
    
    # Sample for dev split
    dev_esl_size = int(dev_size * esl_ratio)
    dev_native_size = dev_size - dev_esl_size
    
    dev_esl = sample_stratified(esl_by_source, dev_esl_size)
    dev_native = sample_stratified(native_by_source, dev_native_size)
    
    # Mark remaining for test pool
    used_esl_ids = {rec.id for rec in dev_esl}
    used_native_ids = {rec.id for rec in dev_native}
    
    test_esl_pool = [rec for rec in esl_records if rec.id not in used_esl_ids]
    test_native_pool = [rec for rec in native_records if rec.id not in used_native_ids]
    
    # Rebuild source groups for test
    test_esl_by_source = defaultdict(list)
    test_native_by_source = defaultdict(list)
    
    for rec in test_esl_pool:
        test_esl_by_source[rec.source].append(rec)
    
    for rec in test_native_pool:
        test_native_by_source[rec.source].append(rec)
    
    # Sample for test split
    test_esl_size = int(test_size * esl_ratio)
    test_native_size = test_size - test_esl_size
    
    test_esl = sample_stratified(test_esl_by_source, test_esl_size)
    test_native = sample_stratified(test_native_by_source, test_native_size)
    
    # Mark splits
    for rec in dev_esl + dev_native:
        rec.split = "dev"
    
    for rec in test_esl + test_native:
        rec.split = "test"
    
    dev_records = dev_esl + dev_native
    test_records = test_esl + test_native
    
    # Shuffle
    random.shuffle(dev_records)
    random.shuffle(test_records)
    
    # Log statistics
    logger.info(f"\nDev split: {len(dev_records)} total")
    logger.info(f"  ESL: {len(dev_esl)} ({len(dev_esl)/len(dev_records)*100:.1f}%)")
    logger.info(f"  Native: {len(dev_native)} ({len(dev_native)/len(dev_records)*100:.1f}%)")
    
    logger.info(f"\nTest split: {len(test_records)} total")
    logger.info(f"  ESL: {len(test_esl)} ({len(test_esl)/len(test_records)*100:.1f}%)")
    logger.info(f"  Native: {len(test_native)} ({len(test_native)/len(test_records)*100:.1f}%)")
    
    # Save splits
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dev_path = output_dir / "esl_native_dev.jsonl"
    test_path = output_dir / "esl_native_test.jsonl"
    
    with open(dev_path, "w", encoding="utf-8") as f:
        for rec in dev_records:
            f.write(json.dumps(rec.to_dict()) + "\n")
    
    with open(test_path, "w", encoding="utf-8") as f:
        for rec in test_records:
            f.write(json.dumps(rec.to_dict()) + "\n")
    
    logger.info(f"\nSaved splits:")
    logger.info(f"  Dev: {dev_path}")
    logger.info(f"  Test: {test_path}")
    
    return dev_records, test_records


def get_split_statistics(records: List[ESLNativeRecord]) -> dict:
    """Get statistics about a data split."""
    total = len(records)
    esl_count = sum(1 for r in records if r.is_esl)
    native_count = total - esl_count
    
    # Count by source
    source_counts = defaultdict(int)
    for rec in records:
        source_counts[rec.source] += 1
    
    # Count by proficiency
    proficiency_counts = defaultdict(int)
    for rec in records:
        if rec.proficiency_level:
            proficiency_counts[rec.proficiency_level] += 1
    
    return {
        "total": total,
        "esl": esl_count,
        "native": native_count,
        "esl_ratio": esl_count / total if total > 0 else 0,
        "sources": dict(source_counts),
        "proficiency_levels": dict(proficiency_counts),
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Build splits
    data_dir = Path("data")
    output_dir = Path("data/processed")
    
    try:
        dev_records, test_records = build_esl_native_eval_split(
            data_dir=data_dir,
            output_dir=output_dir,
            dev_size=200,
            test_size=500,
            esl_ratio=0.4,
            seed=42,
        )
        
        # Print statistics
        print("\n=== Dev Split Statistics ===")
        dev_stats = get_split_statistics(dev_records)
        print(json.dumps(dev_stats, indent=2))
        
        print("\n=== Test Split Statistics ===")
        test_stats = get_split_statistics(test_records)
        print(json.dumps(test_stats, indent=2))
        
    except ValueError as e:
        logger.error(f"Error: {e}")
        logger.info("\nTo use this module, prepare JSONL files at:")
        logger.info("  data/esl/toefl11.jsonl")
        logger.info("  data/esl/icnale_written.jsonl")
        logger.info("  data/esl/ellipse.jsonl")
        logger.info("  data/native/native_academic.jsonl")
