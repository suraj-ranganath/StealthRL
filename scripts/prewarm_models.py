#!/usr/bin/env python3
"""
Prewarm detectors and Ollama models to reduce cold-start latency.

Example:
  python scripts/prewarm_models.py \
    --detectors roberta fast_detectgpt binoculars mage \
    --device mps \
    --roberta-batch-size 128 \
    --fast-detectgpt-batch-size 8 \
    --mage-batch-size 2 \
    --binoculars-batch-size 2
"""

import argparse
import logging
from pathlib import Path
import sys
import time

import requests

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.detectors import load_detectors


def _warm_ollama(ollama_url: str, model: str, prompt: str = "ping") -> None:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 1},
    }
    requests.post(f"{ollama_url}/api/generate", json=payload, timeout=60).raise_for_status()


def _warm_ollama_embed(ollama_url: str, model: str, text: str = "ping") -> None:
    payload = {"model": model, "input": text}
    requests.post(f"{ollama_url}/api/embed", json=payload, timeout=60).raise_for_status()


def main():
    parser = argparse.ArgumentParser(description="Prewarm detectors and Ollama models")
    parser.add_argument("--detectors", nargs="+", default=["roberta", "fast_detectgpt"], help="Detectors to prewarm")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--binoculars-full", action="store_true", help="Use Falcon-7B pair for Binoculars")
    parser.add_argument("--roberta-batch-size", type=int, default=None, help="Batch size for RoBERTa detector")
    parser.add_argument("--fast-detectgpt-batch-size", type=int, default=None, help="Batch size for Fast-DetectGPT detector")
    parser.add_argument("--mage-batch-size", type=int, default=None, help="Batch size for MAGE detector")
    parser.add_argument("--binoculars-batch-size", type=int, default=None, help="Batch size for Binoculars detector")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama prewarm")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama URL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("prewarm")

    # Prewarm detectors
    logger.info("Prewarming detectors...")
    dets = load_detectors(
        args.detectors,
        device=args.device,
        binoculars_full=args.binoculars_full,
        roberta_batch_size=args.roberta_batch_size,
        fast_detectgpt_batch_size=args.fast_detectgpt_batch_size,
        mage_batch_size=args.mage_batch_size,
        binoculars_batch_size=args.binoculars_batch_size,
    )
    warm_text = "This is a short warm-up sentence."
    for name, det in dets.items():
        start = time.time()
        det.load()
        _ = det.get_scores([warm_text])
        logger.info(f"✓ {name} warmed in {time.time() - start:.2f}s")

    if args.skip_ollama:
        logger.info("Skipping Ollama prewarm")
        return

    # Prewarm Ollama LLMs used by methods and similarity scorer
    logger.info("Prewarming Ollama models...")
    try:
        _warm_ollama(args.ollama_url, "qwen3:4b-instruct", "ping")
        _warm_ollama(args.ollama_url, "authormist", "ping")
        _warm_ollama_embed(args.ollama_url, "bge-m3", "ping")
        logger.info("✓ Ollama models warmed")
    except Exception as e:
        logger.warning(f"Ollama prewarm failed: {e}")


if __name__ == "__main__":
    main()

