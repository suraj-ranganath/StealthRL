#!/usr/bin/env python3
"""
Comprehensive data extraction from all downloaded datasets.
Extracts ESL and native academic writing from multiple sources.
"""

import json
import logging
from pathlib import Path
import argparse
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_from_chatgpt_bias(base_dir: Path) -> tuple[List[Dict], List[Dict]]:
    """Extract ESL and native data from ChatGPT-Detector-Bias."""
    esl_records = []
    native_records = []
    
    data_dir = base_dir / "Data_and_Results"
    
    # ESL data from TOEFL
    toefl_dirs = [
        data_dir / "Human_Data/TOEFL_real_91",
        data_dir / "Human_Data/TOEFL_gpt4polished_91"
    ]
    
    # Native data
    native_dirs = [
        (data_dir / "Human_Data/CS224N_real_145", data_dir / "GPT_Data/CS224N_gpt3_145"),
        (data_dir / "Human_Data/CollegeEssay_real_70", data_dir / "GPT_Data/CollegeEssay_gpt3_31"),
        (data_dir / "Human_Data/HewlettStudentEssay_real_88", data_dir / "GPT_Data/CollegeEssay_gpt3_31"),
    ]
    
    # GPT data for pairing with TOEFL
    gpt_ref = data_dir / "GPT_Data/CS224N_gpt3_145/data.json"
    with open(gpt_ref) as f:
        gpt_pool = json.load(f)
    
    # Extract TOEFL (ESL)
    for toefl_dir in toefl_dirs:
        data_file = toefl_dir / "data.json"
        if data_file.exists():
            with open(data_file) as f:
                data = json.load(f)
            
            for i, item in enumerate(data):
                text = item.get('document', '').strip()
                if len(text) < 50:
                    continue
                
                ai_text = gpt_pool[i % len(gpt_pool)].get('document', '').strip()
                
                esl_records.append({
                    "ai_text": ai_text,
                    "human_reference": text,
                    "domain": "academic",
                    "is_esl": True,
                    "metadata": {
                        "source": "TOEFL11",
                        "dataset": "ChatGPT-Detector-Bias",
                        "original_file": str(toefl_dir.name)
                    }
                })
            
            logger.info(f"Extracted {len(data)} records from {toefl_dir.name}")
    
    # Extract native data
    for human_dir, gpt_dir in native_dirs:
        human_file = human_dir / "data.json"
        gpt_file = gpt_dir / "data.json"
        
        if not (human_file.exists() and gpt_file.exists()):
            continue
        
        with open(human_file) as f:
            human_data = json.load(f)
        with open(gpt_file) as f:
            gpt_data = json.load(f)
        
        for i, human_item in enumerate(human_data):
            human_text = human_item.get('document', '').strip()
            ai_text = gpt_data[i % len(gpt_data)].get('document', '').strip()
            
            if len(human_text) < 50 or len(ai_text) < 50:
                continue
            
            native_records.append({
                "ai_text": ai_text,
                "human_reference": human_text,
                "domain": "academic",
                "is_esl": False,
                "metadata": {
                    "source": human_dir.name.replace("_real", "").replace("_", ""),
                    "dataset": "ChatGPT-Detector-Bias",
                    "original_file": str(human_dir.name)
                }
            })
        
        logger.info(f"Extracted {len(human_data)} records from {human_dir.name}")
    
    return esl_records, native_records


def extract_from_detectrl(base_dir: Path) -> List[Dict]:
    """Extract academic writing samples from DetectRL benchmark."""
    records = []
    
    tasks_dir = base_dir / "Benchmark/Tasks"
    if not tasks_dir.exists():
        return records
    
    # Look for academic domains: arxiv, writing prompts
    academic_files = []
    for task_dir in tasks_dir.iterdir():
        if task_dir.is_dir():
            academic_files.extend(task_dir.glob("*arxiv*.json"))
            academic_files.extend(task_dir.glob("*writing_prompt*.json"))
    
    for data_file in academic_files:
        try:
            with open(data_file) as f:
                data = json.load(f)
            
            # DetectRL format: list of dicts with 'text' and 'label'
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                text = item.get('text', '').strip()
                label = item.get('label', 0)  # 0=human, 1=AI
                
                if len(text) < 50:
                    continue
                
                # For human texts, we need to pair with AI text
                # Since we don't have direct pairs, skip for now or use as reference only
                if label == 0:  # Human text
                    records.append({
                        "text": text,
                        "is_human": True,
                        "domain": "arxiv" if "arxiv" in data_file.name else "creative_writing",
                        "source": "DetectRL",
                        "file": data_file.name
                    })
            
            logger.info(f"Extracted {len([r for r in records if r.get('file') == data_file.name])} from {data_file.name}")
        
        except Exception as e:
            logger.warning(f"Error processing {data_file}: {e}")
    
    return records


def extract_from_human_detectors(base_dir: Path) -> List[Dict]:
    """Extract data from human_detectors dataset."""
    records = []
    
    data_file = base_dir / "human_detectors.json"
    if not data_file.exists():
        return records
    
    try:
        with open(data_file) as f:
            data = json.load(f)
        
        logger.info(f"Loaded human_detectors data: {type(data)}")
        
        # Parse based on actual structure
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    text = item.get('text', item.get('content', '')).strip()
                    if len(text) > 50:
                        records.append({
                            "text": text,
                            "source": "human_detectors",
                            "metadata": item
                        })
        elif isinstance(data, dict):
            # Handle dict structure
            for key, value in data.items():
                if isinstance(value, dict) and 'text' in value:
                    text = value['text'].strip()
                    if len(text) > 50:
                        records.append({
                            "text": text,
                            "source": "human_detectors",
                            "id": key
                        })
        
        logger.info(f"Extracted {len(records)} records from human_detectors")
    
    except Exception as e:
        logger.error(f"Error processing human_detectors: {e}")
    
    return records


def main():
    parser = argparse.ArgumentParser(description='Extract data from all downloaded datasets')
    parser.add_argument('--raw-dir', default='data/raw', help='Raw data directory')
    parser.add_argument('--output-esl', default='data/esl/toefl11_extended.jsonl', help='ESL output')
    parser.add_argument('--output-native', default='data/native/native_extended.jsonl', help='Native output')
    
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    
    logger.info("="*60)
    logger.info("Comprehensive Data Extraction")
    logger.info("="*60)
    
    # Extract from ChatGPT-Detector-Bias
    chatgpt_dir = raw_dir / "ChatGPT-Detector-Bias"
    if chatgpt_dir.exists():
        logger.info("\n=== ChatGPT-Detector-Bias ===")
        esl_records, native_records = extract_from_chatgpt_bias(chatgpt_dir)
        logger.info(f"ESL: {len(esl_records)}, Native: {len(native_records)}")
    else:
        esl_records, native_records = [], []
    
    # Extract from DetectRL
    detectrl_dir = raw_dir / "DetectRL"
    if detectrl_dir.exists():
        logger.info("\n=== DetectRL ===")
        detectrl_records = extract_from_detectrl(detectrl_dir)
        logger.info(f"Academic texts: {len(detectrl_records)}")
    else:
        detectrl_records = []
    
    # Extract from human_detectors
    hd_dir = raw_dir / "human_detectors"
    if hd_dir.exists():
        logger.info("\n=== Human Detectors ===")
        hd_records = extract_from_human_detectors(hd_dir)
        logger.info(f"Records: {len(hd_records)}")
    else:
        hd_records = []
    
    # Save ESL data
    if esl_records:
        output_esl = Path(args.output_esl)
        output_esl.parent.mkdir(parents=True, exist_ok=True)
        with open(output_esl, 'w') as f:
            for record in esl_records:
                f.write(json.dumps(record) + '\n')
        logger.info(f"\n✓ Saved {len(esl_records)} ESL records to {output_esl}")
    
    # Save native data
    if native_records:
        output_native = Path(args.output_native)
        output_native.parent.mkdir(parents=True, exist_ok=True)
        with open(output_native, 'w') as f:
            for record in native_records:
                f.write(json.dumps(record) + '\n')
        logger.info(f"✓ Saved {len(native_records)} native records to {output_native}")
    
    # Summary
    total = len(esl_records) + len(native_records)
    esl_ratio = len(esl_records) / total if total > 0 else 0
    
    logger.info("\n" + "="*60)
    logger.info("EXTRACTION SUMMARY")
    logger.info("="*60)
    logger.info(f"ESL records:    {len(esl_records)}")
    logger.info(f"Native records: {len(native_records)}")
    logger.info(f"Total paired:   {total}")
    logger.info(f"ESL ratio:      {esl_ratio:.1%}")
    logger.info(f"\nAdditional datasets available:")
    logger.info(f"  DetectRL:        {len(detectrl_records)} academic texts")
    logger.info(f"  Human Detectors: {len(hd_records)} samples")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    exit(main())
