#!/usr/bin/env python3
"""
Run the StealthBench evaluation harness on configured text files.

This script loads dataset files from a YAML config, normalizes detector names to
the legacy ``stealthrl.evaluation.StealthBench`` interface, runs evaluation, and
optionally writes CSV results plus comparison plots.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stealthrl.evaluation import StealthBench


TEXT_KEYS = ("text", "content", "generation", "output", "response")
LABEL_KEYS = ("label", "y", "is_ai", "target")
DETECTOR_ALIASES = {
    "fast_detectgpt": "fast-detectgpt",
    "fast-detectgpt": "fast-detectgpt",
    "ghostbuster": "ghostbuster",
    "roberta-base-openai-detector": "ghostbuster",
    "binoculars": "binoculars",
}


def load_config(config_path: str) -> dict[str, Any]:
    """Load StealthBench configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _extract_text(record: Any) -> str:
    if isinstance(record, str):
        return record.strip()
    if isinstance(record, dict):
        for key in TEXT_KEYS:
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise ValueError(f"Could not find text field in record: {record!r}")


def _extract_label(record: Any, default: int = 0) -> int:
    if isinstance(record, dict):
        for key in LABEL_KEYS:
            value = record.get(key)
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return int(value)
    return default


def _load_records(path: Path) -> list[Any]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "samples", "items", "records", "texts"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError(f"Unsupported JSON structure in {path}")
    if suffix == ".csv":
        with path.open("r", newline="") as f:
            return list(csv.DictReader(f))
    if suffix == ".txt":
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    raise ValueError(f"Unsupported dataset format: {path}")


def load_texts(path_str: str | None, required: bool = False) -> list[str] | None:
    if not path_str:
        if required:
            raise FileNotFoundError("Missing required dataset path")
        return None
    path = Path(path_str)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required dataset file: {path}")
        print(f"Skipping missing optional dataset: {path}")
        return None
    return [_extract_text(record) for record in _load_records(path)]


def load_labeled_texts(path_str: str | None) -> tuple[list[str] | None, list[int] | None]:
    if not path_str:
        return None, None
    path = Path(path_str)
    if not path.exists():
        print(f"Skipping missing optional labeled dataset: {path}")
        return None, None
    records = _load_records(path)
    texts = [_extract_text(record) for record in records]
    labels = [_extract_label(record, default=0) for record in records]
    return texts, labels


def normalize_detectors(detectors: list[str]) -> list[str]:
    normalized: list[str] = []
    for detector in detectors:
        mapped = DETECTOR_ALIASES.get(detector, detector)
        if mapped not in normalized:
            normalized.append(mapped)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Run StealthBench evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional override for output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Detector device (default: cuda)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    dataset_cfg = config.get("datasets", {})
    human_texts = load_texts(dataset_cfg.get("human_texts"), required=True)
    ai_texts = load_texts(dataset_cfg.get("ai_texts"), required=True)
    paraphrased_texts = load_texts(dataset_cfg.get("paraphrased_texts"))
    esl_texts, esl_labels = load_labeled_texts(dataset_cfg.get("esl_texts"))
    native_texts, native_labels = load_labeled_texts(dataset_cfg.get("native_texts"))

    detectors = normalize_detectors(config.get("detectors", []))
    if not detectors:
        raise ValueError("No detectors configured")

    output_cfg = config.get("output", {})
    output_dir = args.output_dir or output_cfg.get("output_dir", "outputs/stealthbench_results")

    bench = StealthBench(
        detectors=detectors,
        output_dir=output_dir,
        device=args.device,
    )
    results = bench.run(
        human_texts=human_texts,
        ai_texts=ai_texts,
        paraphrased_texts=paraphrased_texts,
        esl_texts=esl_texts,
        native_texts=native_texts,
        esl_labels=esl_labels,
        native_labels=native_labels,
    )

    if output_cfg.get("save_csv", True):
        bench.save_results(results)
    if output_cfg.get("save_plots", True):
        bench.generate_plots(results)

    if results.empty:
        print("No detector results were produced.")
    else:
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
