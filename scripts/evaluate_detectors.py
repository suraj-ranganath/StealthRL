#!/usr/bin/env python3
"""
Evaluate detector ensemble on text samples.

This script runs multiple detectors on input texts and produces
CSV outputs with detection scores.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import List

from stealthrl.detectors import FastDetectGPTDetector, GhostbusterDetector, BinocularsDetector


def load_texts(input_file: str) -> List[str]:
    """Load texts from JSON or JSONL file."""
    texts = []
    with open(input_file, 'r') as f:
        if input_file.endswith('.jsonl'):
            for line in f:
                data = json.loads(line)
                texts.append(data['text'])
        else:
            data = json.load(f)
            texts = [item['text'] for item in data]
    return texts


def main():
    parser = argparse.ArgumentParser(description="Evaluate detector ensemble")
    parser.add_argument("--input", type=str, required=True, help="Input text file")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    parser.add_argument("--detectors", nargs='+', default=["fast-detectgpt", "ghostbuster"], 
                        help="Detectors to run")
    
    args = parser.parse_args()
    
    # Load texts
    print(f"Loading texts from {args.input}")
    texts = load_texts(args.input)
    print(f"Loaded {len(texts)} texts")
    
    # Initialize detectors
    detector_map = {
        "fast-detectgpt": FastDetectGPTDetector(),
        "ghostbuster": GhostbusterDetector(),
        "binoculars": BinocularsDetector(),
    }
    
    results = {"text": texts}
    
    # Run each detector
    for detector_name in args.detectors:
        print(f"Running {detector_name}...")
        detector = detector_map[detector_name]
        scores = detector.detect(texts)
        results[f"{detector_name}_score"] = scores.cpu().numpy()
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
