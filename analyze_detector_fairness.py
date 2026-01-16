#!/usr/bin/env python3
"""
Analyze detector score bias between ESL and native human text.

This script compares AI-detector scores for ESL vs native samples
using the human_reference field by default.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Tuple

from stealthrl.tinker.detectors import DetectorEnsemble, _default_device


logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> List[Dict]:
    records = []
    if not path.exists():
        logger.warning("Missing file: %s", path)
        return records
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON on line %d in %s", line_num, path)
    return records


def _select_records(data_path: Path, split: str) -> List[Dict]:
    if split == "train":
        return _load_jsonl(data_path / "train.jsonl")
    if split == "test":
        return _load_jsonl(data_path / "test.jsonl")
    if split == "both":
        return _load_jsonl(data_path / "train.jsonl") + _load_jsonl(data_path / "test.jsonl")
    raise ValueError(f"Unknown split: {split}")


def _collect_texts(records: List[Dict], text_field: str) -> Tuple[List[str], List[str]]:
    esl_texts = []
    native_texts = []
    for rec in records:
        text = (rec.get(text_field) or "").strip()
        if not text:
            continue
        if rec.get("is_esl"):
            esl_texts.append(text)
        else:
            native_texts.append(text)
    return esl_texts, native_texts


def _sample(texts: List[str], max_samples: int | None, seed: int) -> List[str]:
    if max_samples is None or max_samples >= len(texts):
        return texts
    rng = random.Random(seed)
    return rng.sample(texts, max_samples)


def _percentile(sorted_vals: List[float], pct: float) -> float | None:
    if not sorted_vals:
        return None
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _summarize(values: List[float], thresholds: List[float]) -> Dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            **{f"flag_rate@{t}": None for t in thresholds},
        }
    sorted_vals = sorted(values)
    summary: Dict[str, float | int] = {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "p25": _percentile(sorted_vals, 25),
        "p75": _percentile(sorted_vals, 75),
        "p90": _percentile(sorted_vals, 90),
        "p95": _percentile(sorted_vals, 95),
    }
    for t in thresholds:
        hits = sum(1 for v in values if v >= t)
        summary[f"flag_rate@{t}"] = hits / len(values)
    return summary


async def _score_texts(
    texts: List[str],
    ensemble: DetectorEnsemble,
    batch_size: int,
    log_every: int,
) -> Dict[str, List[float]]:
    scores: Dict[str, List[float]] = {"ensemble": []}
    for name in ensemble.detectors:
        scores[name] = []

    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results = await ensemble.compute_batch(batch)
        for res in results:
            scores["ensemble"].append(res["ensemble_prob"])
            for name, val in res["detector_scores"].items():
                scores[name].append(val)
        if log_every > 0 and (i // batch_size + 1) % log_every == 0:
            logger.info("Scored %d/%d batches", i // batch_size + 1, total_batches)

    return scores


def _parse_detector_weights(items: List[str] | None) -> Dict[str, float] | None:
    if not items:
        return None
    weights: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid detector weight: {item}")
        name, value = item.split("=", 1)
        weights[name.strip()] = float(value)
    return weights


def _parse_thresholds(value: str) -> List[float]:
    if not value:
        return [0.5]
    return [float(v.strip()) for v in value.split(",") if v.strip()]


async def _run(args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, float | int | None]]]:
    data_path = Path(args.data_path)
    records = _select_records(data_path, args.split)
    if not records:
        raise SystemExit("No records found. Check --data-path and --split.")

    text_fields = [args.text_field]
    if args.text_field == "both":
        text_fields = ["human_reference", "ai_text"]

    device = args.device or _default_device()
    weights = _parse_detector_weights(args.detector_weights)
    thresholds = _parse_thresholds(args.thresholds)

    ensemble = DetectorEnsemble(
        detector_names=args.detectors,
        detector_weights=weights,
        cache_path=args.cache_path,
        device=device,
        max_concurrent=args.max_concurrent,
    )
    if args.prewarm:
        ensemble.prewarm_models()

    results: Dict[str, Dict[str, Dict[str, float | int | None]]] = {}

    for text_field in text_fields:
        esl_texts, native_texts = _collect_texts(records, text_field)
        esl_texts = _sample(esl_texts, args.max_samples, args.seed)
        native_texts = _sample(native_texts, args.max_samples, args.seed + 1)

        logger.info(
            "Scoring %d ESL and %d native texts for %s",
            len(esl_texts),
            len(native_texts),
            text_field,
        )

        esl_scores = await _score_texts(esl_texts, ensemble, args.batch_size, args.log_every)
        native_scores = await _score_texts(native_texts, ensemble, args.batch_size, args.log_every)

        field_results: Dict[str, Dict[str, float | int | None]] = {}
        for key in esl_scores:
            field_results[f"esl/{key}"] = _summarize(esl_scores[key], thresholds)
            field_results[f"native/{key}"] = _summarize(native_scores[key], thresholds)

            if esl_scores[key] and native_scores[key]:
                field_results[f"diff/{key}"] = {
                    "mean_delta": mean(esl_scores[key]) - mean(native_scores[key]),
                    "median_delta": median(esl_scores[key]) - median(native_scores[key]),
                    **{
                        f"flag_rate_delta@{t}": (
                            field_results[f"esl/{key}"][f"flag_rate@{t}"]
                            - field_results[f"native/{key}"][f"flag_rate@{t}"]
                        )
                        for t in thresholds
                    },
                }

        results[text_field] = field_results

    ensemble.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze detector bias for ESL vs native text")
    parser.add_argument("--data-path", type=str, required=True, help="Dataset directory with train/test JSONL")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "both"])
    parser.add_argument(
        "--text-field",
        type=str,
        default="human_reference",
        choices=["human_reference", "ai_text", "both"],
        help="Which field to score",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per group")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detectors", nargs="+", default=["fast_detectgpt", "ghostbuster"])
    parser.add_argument("--detector-weights", nargs="+", default=None, help="e.g., fast_detectgpt=0.5")
    parser.add_argument("--thresholds", type=str, default="0.5", help="Comma-separated thresholds")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for scoring")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent detector evals")
    parser.add_argument("--device", type=str, default=None, help="cuda|mps|cpu")
    parser.add_argument("--cache-path", type=str, default=None, help="SQLite cache path")
    parser.add_argument("--prewarm", action="store_true", help="Preload detector models")
    parser.add_argument("--log-every", type=int, default=0, help="Log every N batches")
    parser.add_argument("--output-json", type=str, default=None, help="Write summary JSON to file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    results = asyncio.run(_run(args))
    print(json.dumps(results, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        logger.info("Wrote summary to %s", out_path)


if __name__ == "__main__":
    main()
