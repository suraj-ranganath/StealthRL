#!/usr/bin/env python3
"""
Direct conversion script for ChatGPT-Detector-Bias data.
Works with the actual directory structure.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_data():
    # Input paths
    base_dir = Path("data/raw/ChatGPT-Detector-Bias/Data_and_Results/Human_Data")

    # ESL data (TOEFL)
    toefl_file = base_dir / "TOEFL_real_91" / "data.json"

    # Native data
    cs224n_file = base_dir / "CS224N_real_145" / "data.json"
    college_file = base_dir / "CollegeEssay_real_70" / "data.json"

    # Output paths
    output_esl = Path("data/esl/toefl11.jsonl")
    output_native = Path("data/native/native_academic.jsonl")

    # Create output directories
    output_esl.parent.mkdir(parents=True, exist_ok=True)
    output_native.parent.mkdir(parents=True, exist_ok=True)

    # Process ESL data
    logger.info(f"Processing ESL data from {toefl_file}")
    with open(toefl_file, 'r') as f:
        toefl_data = json.load(f)

    esl_records = []
    for idx, item in enumerate(toefl_data):
        text = item.get('document', '').strip()
        if text and len(text) > 50:  # Filter very short texts
            record = {
                "id": f"toefl_{idx:04d}",
                "text": text,
                "source": "TOEFL11",
                "is_esl": True
            }
            esl_records.append(record)

    logger.info(f"Extracted {len(esl_records)} ESL records")

    # Write ESL data
    with open(output_esl, 'w') as f:
        for record in esl_records:
            f.write(json.dumps(record) + '\n')

    logger.info(f"Saved ESL data to {output_esl}")

    # Process native data
    logger.info(f"Processing native data from {cs224n_file} and {college_file}")

    native_records = []

    # CS224N data
    with open(cs224n_file, 'r') as f:
        cs_data = json.load(f)

    for idx, item in enumerate(cs_data):
        text = item.get('document', '').strip()
        if text and len(text) > 50:
            record = {
                "id": f"cs224n_{idx:04d}",
                "text": text,
                "source": "CS224N",
                "is_esl": False
            }
            native_records.append(record)

    # College essay data
    with open(college_file, 'r') as f:
        college_data = json.load(f)

    for idx, item in enumerate(college_data):
        text = item.get('document', '').strip()
        if text and len(text) > 50:
            record = {
                "id": f"college_{idx:04d}",
                "text": text,
                "source": "CollegeEssay",
                "is_esl": False
            }
            native_records.append(record)

    logger.info(f"Extracted {len(native_records)} native records")

    # Write native data
    with open(output_native, 'w') as f:
        for record in native_records:
            f.write(json.dumps(record) + '\n')

    logger.info(f"Saved native data to {output_native}")

    # Summary
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"ESL records:    {len(esl_records)} (saved to {output_esl})")
    print(f"Native records: {len(native_records)} (saved to {output_native})")
    print(f"Total records:  {len(esl_records) + len(native_records)}")
    print(f"ESL ratio:      {len(esl_records)/(len(esl_records)+len(native_records))*100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    convert_data()
