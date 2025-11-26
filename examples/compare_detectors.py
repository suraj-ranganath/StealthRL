#!/usr/bin/env python3
"""
Example: Compare detector scores before and after StealthRL paraphrasing.
"""

import argparse
from tabulate import tabulate

from stealthrl.detectors import FastDetectGPTDetector, GhostbusterDetector, BinocularsDetector


def main():
    parser = argparse.ArgumentParser(description="Compare detector scores")
    parser.add_argument("--original_text", type=str, required=True, help="Original text")
    parser.add_argument("--paraphrased_text", type=str, required=True, help="Paraphrased text")
    parser.add_argument("--detectors", nargs='+', 
                        default=["fast-detectgpt", "ghostbuster", "binoculars"],
                        help="Detectors to compare")
    
    args = parser.parse_args()
    
    # Initialize detectors
    detector_map = {
        "fast-detectgpt": FastDetectGPTDetector(),
        "ghostbuster": GhostbusterDetector(),
        "binoculars": BinocularsDetector(),
    }
    
    # Compute scores
    results = []
    for detector_name in args.detectors:
        print(f"Running {detector_name}...")
        detector = detector_map[detector_name]
        
        original_score = detector.detect([args.original_text])[0].item()
        paraphrased_score = detector.detect([args.paraphrased_text])[0].item()
        reduction = original_score - paraphrased_score
        reduction_pct = (reduction / original_score) * 100 if original_score > 0 else 0
        
        results.append([
            detector_name,
            f"{original_score:.4f}",
            f"{paraphrased_score:.4f}",
            f"{reduction:.4f}",
            f"{reduction_pct:.1f}%"
        ])
    
    # Print results
    headers = ["Detector", "Original Score", "Paraphrased Score", "Reduction", "Reduction %"]
    print("\n" + tabulate(results, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
