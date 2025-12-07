#!/usr/bin/env python3
"""
Direct conversion of ChatGPT-Detector-Bias data to StealthRL format.
Works with the actual directory structure of the dataset.
"""

import json
import logging
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_toefl_data(base_dir: Path, output_file: Path):
    """Convert TOEFL (ESL) data to required format."""
    # TOEFL data paths
    toefl_real = base_dir / "Data_and_Results/Human_Data/TOEFL_real_91/data.json"
    toefl_polished = base_dir / "Data_and_Results/Human_Data/TOEFL_gpt4polished_91/data.json"
    
    # GPT-generated versions for pairing
    # (We'll use CS224N GPT data since TOEFL doesn't have GPT pairs)
    gpt_data_path = base_dir / "Data_and_Results/GPT_Data/CS224N_gpt3_145/data.json"
    
    records = []
    
    # Load TOEFL real data
    if toefl_real.exists():
        logger.info(f"Loading TOEFL real data from {toefl_real}")
        with open(toefl_real) as f:
            toefl_data = json.load(f)
        
        # Load GPT data for pairing
        with open(gpt_data_path) as f:
            gpt_data = json.load(f)
        
        for i, item in enumerate(toefl_data):
            text = item.get('document', '').strip()
            if len(text) < 50:  # Skip very short texts
                continue
            
            # Use GPT text as AI-generated version (cycling through if needed)
            ai_text = gpt_data[i % len(gpt_data)].get('document', '').strip()
            
            record = {
                "ai_text": ai_text,
                "human_reference": text,
                "domain": "academic",
                "is_esl": True,
                "metadata": {
                    "source": "TOEFL11",
                    "original_file": str(toefl_real.name)
                }
            }
            records.append(record)
        
        logger.info(f"Extracted {len(records)} TOEFL (ESL) records")
    
    # Load TOEFL polished data (also ESL)
    if toefl_polished.exists():
        logger.info(f"Loading TOEFL polished data from {toefl_polished}")
        with open(toefl_polished) as f:
            toefl_pol_data = json.load(f)
        
        # Reload GPT data for pairing
        with open(gpt_data_path) as f:
            gpt_data = json.load(f)
        
        for i, item in enumerate(toefl_pol_data):
            text = item.get('document', '').strip()
            if len(text) < 50:
                continue
            
            ai_text = gpt_data[i % len(gpt_data)].get('document', '').strip()
            
            record = {
                "ai_text": ai_text,
                "human_reference": text,
                "domain": "academic",
                "is_esl": True,
                "metadata": {
                    "source": "TOEFL11_polished",
                    "original_file": str(toefl_polished.name)
                }
            }
            records.append(record)
        
        logger.info(f"Total TOEFL (ESL) records: {len(records)}")
    
    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    
    logger.info(f"Saved {len(records)} ESL records to {output_file}")
    return len(records)


def convert_native_data(base_dir: Path, output_file: Path):
    """Convert native (CS224N, College Essays) data to required format."""
    records = []
    
    # CS224N data (native academic writing)
    cs224n_human = base_dir / "Data_and_Results/Human_Data/CS224N_real_145/data.json"
    cs224n_gpt = base_dir / "Data_and_Results/GPT_Data/CS224N_gpt3_145/data.json"
    
    if cs224n_human.exists() and cs224n_gpt.exists():
        logger.info(f"Loading CS224N native data")
        with open(cs224n_human) as f:
            human_data = json.load(f)
        with open(cs224n_gpt) as f:
            gpt_data = json.load(f)
        
        for human_item, gpt_item in zip(human_data, gpt_data):
            human_text = human_item.get('document', '').strip()
            ai_text = gpt_item.get('document', '').strip()
            
            if len(human_text) < 50 or len(ai_text) < 50:
                continue
            
            record = {
                "ai_text": ai_text,
                "human_reference": human_text,
                "domain": "academic",
                "is_esl": False,
                "metadata": {
                    "source": "CS224N",
                    "original_file": "CS224N_real_145"
                }
            }
            records.append(record)
        
        logger.info(f"Extracted {len(records)} CS224N records")
    
    # College Essays (native)
    college_human = base_dir / "Data_and_Results/Human_Data/CollegeEssay_real_70/data.json"
    college_gpt = base_dir / "Data_and_Results/GPT_Data/CollegeEssay_gpt3_31/data.json"
    
    if college_human.exists() and college_gpt.exists():
        logger.info(f"Loading College Essay native data")
        with open(college_human) as f:
            human_data = json.load(f)
        with open(college_gpt) as f:
            gpt_data = json.load(f)
        
        for i, human_item in enumerate(human_data):
            human_text = human_item.get('document', '').strip()
            ai_text = gpt_data[i % len(gpt_data)].get('document', '').strip()
            
            if len(human_text) < 50 or len(ai_text) < 50:
                continue
            
            record = {
                "ai_text": ai_text,
                "human_reference": human_text,
                "domain": "academic",
                "is_esl": False,
                "metadata": {
                    "source": "CollegeEssay",
                    "original_file": "CollegeEssay_real_70"
                }
            }
            records.append(record)
        
        logger.info(f"Total native records: {len(records)}")
    
    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    
    logger.info(f"Saved {len(records)} native records to {output_file}")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description='Convert ChatGPT-Detector-Bias data')
    parser.add_argument('--input', required=True, help='Path to ChatGPT-Detector-Bias directory')
    parser.add_argument('--output-esl', required=True, help='Output path for ESL data')
    parser.add_argument('--output-native', required=True, help='Output path for native data')
    
    args = parser.parse_args()
    
    base_dir = Path(args.input)
    if not base_dir.exists():
        logger.error(f"Input directory does not exist: {base_dir}")
        return 1
    
    logger.info("="*60)
    logger.info("Converting ChatGPT-Detector-Bias Data")
    logger.info("="*60)
    
    # Convert ESL data
    esl_count = convert_toefl_data(base_dir, Path(args.output_esl))
    
    # Convert native data  
    native_count = convert_native_data(base_dir, Path(args.output_native))
    
    # Summary
    total = esl_count + native_count
    esl_ratio = esl_count / total if total > 0 else 0
    
    logger.info("="*60)
    logger.info("CONVERSION SUMMARY")
    logger.info("="*60)
    logger.info(f"ESL records:    {esl_count}")
    logger.info(f"Native records: {native_count}")
    logger.info(f"Total records:  {total}")
    logger.info(f"")
    logger.info(f"ESL ratio:      {esl_ratio:.1%}")
    logger.info(f"Native ratio:   {1-esl_ratio:.1%}")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    exit(main())
