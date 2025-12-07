#!/usr/bin/env python3
"""
Convert ChatGPT-Detector-Bias data to StealthRL format.

This script extracts ESL (TOEFL) and native writing samples from the
ChatGPT-Detector-Bias dataset and converts them to the required JSONL format.

Usage:
    python scripts/convert_chatgpt_bias_data.py \
        --input data/raw/ChatGPT-Detector-Bias \
        --output-esl data/esl/toefl11.jsonl \
        --output-native data/native/native_academic.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_data_files(input_dir: Path) -> Dict[str, List[Path]]:
    """
    Find ESL and native data files in ChatGPT-Detector-Bias directory.

    Args:
        input_dir: Root directory of ChatGPT-Detector-Bias repo

    Returns:
        Dictionary with 'esl' and 'native' file lists
    """
    logger.info(f"Searching for data files in {input_dir}")

    esl_files = []
    native_files = []

    # Common patterns for ESL/TOEFL data
    esl_patterns = ['*toefl*', '*esl*', '*non-native*', '*nonnative*']
    native_patterns = ['*native*', '*human*']

    for pattern in esl_patterns:
        esl_files.extend(input_dir.rglob(pattern))

    for pattern in native_patterns:
        native_files.extend(input_dir.rglob(pattern))

    # Filter to only json/jsonl/csv files
    esl_files = [f for f in esl_files if f.suffix in ['.json', '.jsonl', '.csv', '.txt']]
    native_files = [f for f in native_files if f.suffix in ['.json', '.jsonl', '.csv', '.txt']
                   and 'toefl' not in f.name.lower() and 'esl' not in f.name.lower()]

    logger.info(f"Found {len(esl_files)} potential ESL files")
    logger.info(f"Found {len(native_files)} potential native files")

    return {'esl': esl_files, 'native': native_files}


def load_json_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load data from JSON/JSONL file."""
    data = []

    if file_path.suffix == '.jsonl':
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    elif file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                data = loaded
            else:
                data = [loaded]

    return data


def convert_to_esl_format(
    text: str,
    record_id: str,
    source: str = "TOEFL11",
    proficiency: str = None,
    prompt_id: str = None,
    metadata: Dict = None
) -> Dict[str, Any]:
    """
    Convert text to ESL/native record format.

    Required format:
    {
        "id": str,
        "text": str,
        "source": str,
        "is_esl": bool,
        "proficiency_level": Optional[str],
        "prompt_id": Optional[str]
    }
    """
    record = {
        "id": record_id,
        "text": text,
        "source": source,
        "is_esl": source.upper().startswith("TOEFL"),  # TOEFL = ESL
    }

    if proficiency:
        record["proficiency_level"] = proficiency

    if prompt_id:
        record["prompt_id"] = prompt_id

    if metadata:
        record.update(metadata)

    return record


def extract_esl_records(data_files: List[Path]) -> List[Dict[str, Any]]:
    """Extract and convert ESL records."""
    records = []

    for file_path in data_files:
        logger.info(f"Processing ESL file: {file_path.name}")

        try:
            data = load_json_file(file_path)

            for idx, item in enumerate(data):
                # Try to extract text from common field names
                text = (item.get('text') or
                       item.get('essay') or
                       item.get('content') or
                       item.get('writing') or
                       str(item) if isinstance(item, str) else None)

                if not text or len(text.strip()) < 50:  # Skip very short texts
                    continue

                record_id = f"toefl11_{len(records):04d}"

                # Extract metadata if available
                proficiency = item.get('proficiency_level') or item.get('level')
                prompt_id = item.get('prompt_id') or item.get('prompt')

                record = convert_to_esl_format(
                    text=text.strip(),
                    record_id=record_id,
                    source="TOEFL11",
                    proficiency=proficiency,
                    prompt_id=prompt_id
                )

                records.append(record)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue

    logger.info(f"Extracted {len(records)} ESL records")
    return records


def extract_native_records(data_files: List[Path]) -> List[Dict[str, Any]]:
    """Extract and convert native records."""
    records = []

    for file_path in data_files:
        logger.info(f"Processing native file: {file_path.name}")

        try:
            data = load_json_file(file_path)

            for idx, item in enumerate(data):
                # Try to extract text from common field names
                text = (item.get('text') or
                       item.get('essay') or
                       item.get('content') or
                       item.get('writing') or
                       str(item) if isinstance(item, str) else None)

                if not text or len(text.strip()) < 50:  # Skip very short texts
                    continue

                record_id = f"native_academic_{len(records):04d}"

                record = convert_to_esl_format(
                    text=text.strip(),
                    record_id=record_id,
                    source="NATIVE_ACADEMIC",
                    proficiency=None,
                    prompt_id=None
                )

                records.append(record)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue

    logger.info(f"Extracted {len(records)} native records")
    return records


def save_jsonl(records: List[Dict[str, Any]], output_path: Path):
    """Save records to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')

    logger.info(f"Saved {len(records)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ChatGPT-Detector-Bias data to StealthRL format"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory (ChatGPT-Detector-Bias repo)'
    )
    parser.add_argument(
        '--output-esl',
        type=str,
        required=True,
        help='Output path for ESL data (JSONL)'
    )
    parser.add_argument(
        '--output-native',
        type=str,
        required=True,
        help='Output path for native data (JSONL)'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=100,
        help='Minimum number of samples to extract (default: 100)'
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Find data files
    data_files = find_data_files(input_dir)

    if not data_files['esl'] and not data_files['native']:
        logger.error("No data files found! Please check the directory structure.")
        logger.info("Expected structure:")
        logger.info("  data/raw/ChatGPT-Detector-Bias/")
        logger.info("    ├── *toefl*.json(l)")
        logger.info("    └── *native*.json(l)")
        sys.exit(1)

    # Extract ESL records
    esl_records = []
    if data_files['esl']:
        esl_records = extract_esl_records(data_files['esl'])
    else:
        logger.warning("No ESL files found!")

    # Extract native records
    native_records = []
    if data_files['native']:
        native_records = extract_native_records(data_files['native'])
    else:
        logger.warning("No native files found!")

    # Check minimum samples
    if len(esl_records) < args.min_samples:
        logger.warning(f"Only extracted {len(esl_records)} ESL records (< {args.min_samples})")

    if len(native_records) < args.min_samples:
        logger.warning(f"Only extracted {len(native_records)} native records (< {args.min_samples})")

    # Save outputs
    if esl_records:
        save_jsonl(esl_records, Path(args.output_esl))

    if native_records:
        save_jsonl(native_records, Path(args.output_native))

    # Print summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"ESL records:    {len(esl_records)}")
    print(f"Native records: {len(native_records)}")
    print(f"Total records:  {len(esl_records) + len(native_records)}")
    print()
    print(f"ESL ratio:      {len(esl_records)/(len(esl_records)+len(native_records))*100:.1f}%")
    print(f"Native ratio:   {len(native_records)/(len(esl_records)+len(native_records))*100:.1f}%")
    print("="*60)

    if len(esl_records) == 0 or len(native_records) == 0:
        logger.error("Failed to extract sufficient data!")
        sys.exit(1)


if __name__ == "__main__":
    main()
