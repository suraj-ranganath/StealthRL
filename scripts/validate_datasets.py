#!/usr/bin/env python3
"""
Validate datasets for StealthRL training.

Checks:
- JSONL format validity
- Required fields present
- ESL/native ratio
- Text length statistics
- Duplicate detection

Usage:
    python scripts/validate_datasets.py \
        --esl-data data/esl/toefl11.jsonl \
        --native-data data/native/native_academic.jsonl \
        --output logs/validation.log
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import defaultdict
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validator for StealthRL datasets."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)

    def validate_jsonl_format(self, file_path: Path) -> List[Dict[str, Any]]:
        """Validate JSONL file format and load records."""
        records = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        self.errors.append(
                            f"{file_path.name} line {line_num}: Invalid JSON - {e}"
                        )

        except FileNotFoundError:
            self.errors.append(f"File not found: {file_path}")
            return []

        logger.info(f"✓ Loaded {len(records)} records from {file_path.name}")
        return records

    def validate_required_fields(
        self,
        records: List[Dict[str, Any]],
        file_name: str
    ) -> bool:
        """Validate that all required fields are present."""
        required_fields = ['id', 'text', 'source', 'is_esl']

        for idx, record in enumerate(records):
            missing = [f for f in required_fields if f not in record]

            if missing:
                self.errors.append(
                    f"{file_name} record {idx}: Missing fields: {missing}"
                )

            # Check field types
            if 'id' in record and not isinstance(record['id'], str):
                self.warnings.append(
                    f"{file_name} record {idx}: 'id' should be string"
                )

            if 'text' in record and not isinstance(record['text'], str):
                self.errors.append(
                    f"{file_name} record {idx}: 'text' must be string"
                )

            if 'is_esl' in record and not isinstance(record['is_esl'], bool):
                self.errors.append(
                    f"{file_name} record {idx}: 'is_esl' must be boolean"
                )

        if not self.errors:
            logger.info(f"✓ All required fields present in {file_name}")
            return True
        return False

    def validate_text_lengths(
        self,
        records: List[Dict[str, Any]],
        file_name: str
    ):
        """Validate text length statistics."""
        lengths = [len(r['text'].split()) for r in records if 'text' in r]

        if not lengths:
            self.errors.append(f"{file_name}: No valid text fields found")
            return

        min_len = min(lengths)
        max_len = max(lengths)
        mean_len = sum(lengths) / len(lengths)

        logger.info(f"Text lengths in {file_name}:")
        logger.info(f"  Min: {min_len} words")
        logger.info(f"  Max: {max_len} words")
        logger.info(f"  Mean: {mean_len:.1f} words")

        # Warn about very short or long texts
        if min_len < 20:
            self.warnings.append(
                f"{file_name}: Some texts are very short (< 20 words)"
            )

        if max_len > 1000:
            self.warnings.append(
                f"{file_name}: Some texts are very long (> 1000 words)"
            )

        self.stats[f'{file_name}_min_length'] = min_len
        self.stats[f'{file_name}_max_length'] = max_len
        self.stats[f'{file_name}_mean_length'] = mean_len

    def check_duplicates(
        self,
        records: List[Dict[str, Any]],
        file_name: str
    ):
        """Check for duplicate IDs and texts."""
        ids: Set[str] = set()
        texts: Set[str] = set()
        duplicate_ids = []
        duplicate_texts = []

        for record in records:
            record_id = record.get('id')
            text = record.get('text', '')

            if record_id in ids:
                duplicate_ids.append(record_id)
            ids.add(record_id)

            # Check for exact text duplicates
            text_hash = hash(text)
            if text_hash in texts:
                duplicate_texts.append(record_id)
            texts.add(text_hash)

        if duplicate_ids:
            self.warnings.append(
                f"{file_name}: {len(duplicate_ids)} duplicate IDs found"
            )

        if duplicate_texts:
            self.warnings.append(
                f"{file_name}: {len(duplicate_texts)} duplicate texts found"
            )

        if not duplicate_ids and not duplicate_texts:
            logger.info(f"✓ No duplicates found in {file_name}")

    def validate_esl_ratio(
        self,
        esl_records: List[Dict[str, Any]],
        native_records: List[Dict[str, Any]]
    ):
        """Validate ESL/native ratio."""
        total = len(esl_records) + len(native_records)

        if total == 0:
            self.errors.append("No records found in either dataset")
            return

        esl_ratio = len(esl_records) / total
        native_ratio = len(native_records) / total

        logger.info("\nDataset Composition:")
        logger.info(f"  ESL records:    {len(esl_records)} ({esl_ratio*100:.1f}%)")
        logger.info(f"  Native records: {len(native_records)} ({native_ratio*100:.1f}%)")
        logger.info(f"  Total records:  {total}")

        # Target: 40% ESL, 60% native
        target_esl = 0.40
        if abs(esl_ratio - target_esl) > 0.10:  # Allow 10% deviation
            self.warnings.append(
                f"ESL ratio {esl_ratio*100:.1f}% is far from target {target_esl*100:.0f}%"
            )
        else:
            logger.info(f"✓ ESL ratio is within target range (40% ± 10%)")

        self.stats['total_records'] = total
        self.stats['esl_records'] = len(esl_records)
        self.stats['native_records'] = len(native_records)
        self.stats['esl_ratio'] = esl_ratio

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors[:10]:  # Show first 10
                print(f"  - {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings[:10]:  # Show first 10
                print(f"  - {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")

        if not self.errors and not self.warnings:
            print("\n✅ All validation checks passed!")

        print("\nSTATISTICS:")
        for key, value in self.stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

        print("="*60 + "\n")

        return len(self.errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Validate StealthRL datasets")
    parser.add_argument(
        '--esl-data',
        type=str,
        required=True,
        help='Path to ESL JSONL file'
    )
    parser.add_argument(
        '--native-data',
        type=str,
        required=True,
        help='Path to native JSONL file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output log file (optional)'
    )

    args = parser.parse_args()

    # Setup logging to file if specified
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(args.output)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)

    validator = DatasetValidator()

    logger.info("Starting dataset validation...")

    # Validate ESL data
    logger.info(f"\nValidating ESL data: {args.esl_data}")
    esl_records = validator.validate_jsonl_format(Path(args.esl_data))

    if esl_records:
        validator.validate_required_fields(esl_records, "ESL")
        validator.validate_text_lengths(esl_records, "ESL")
        validator.check_duplicates(esl_records, "ESL")

    # Validate native data
    logger.info(f"\nValidating native data: {args.native_data}")
    native_records = validator.validate_jsonl_format(Path(args.native_data))

    if native_records:
        validator.validate_required_fields(native_records, "Native")
        validator.validate_text_lengths(native_records, "Native")
        validator.check_duplicates(native_records, "Native")

    # Validate ratio
    if esl_records and native_records:
        validator.validate_esl_ratio(esl_records, native_records)

    # Print summary
    success = validator.print_summary()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
