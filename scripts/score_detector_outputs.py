#!/usr/bin/env python3
"""
Score human text and generated outputs for a single detector.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.data import load_eval_dataset_with_ids
from eval.detectors import get_detector


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score raw outputs with one detector")
    parser.add_argument("--detector", required=True)
    parser.add_argument("--samples-dir", required=True, help="Directory containing dataset_samples.json")
    parser.add_argument("--raw-outputs", required=True, help="Combined raw_outputs.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--binoculars-full", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _detector_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {"device": args.device}
    if args.detector == "roberta" and args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    elif args.detector == "fast_detectgpt" and args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    elif args.detector == "mage" and args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    elif args.detector == "binoculars":
        kwargs["use_lightweight"] = not args.binoculars_full
        if args.batch_size is not None:
            kwargs["batch_size"] = args.batch_size
    elif args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    return kwargs


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    samples_dir = Path(args.samples_dir)
    raw_outputs = json.loads(Path(args.raw_outputs).read_text())
    ids = json.loads((samples_dir / "dataset_samples.json").read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = get_detector(args.detector, **_detector_kwargs(args))
    detector.load()

    records: list[dict] = []

    for dataset_name, dataset_ids in ids.items():
        dataset = load_eval_dataset_with_ids(
            name=dataset_name,
            human_ids=dataset_ids["human_ids"],
            ai_ids=dataset_ids["ai_ids"],
            cache_dir=args.cache_dir,
        )
        human_ids = [sample.id for sample in dataset.human_samples]
        human_texts = [sample.text for sample in dataset.human_samples]
        ai_ids = [sample.id for sample in dataset.ai_samples]

        logger.info(
            "Scoring detector %s on %s (%d human, %d AI per method)",
            args.detector,
            dataset_name,
            len(human_texts),
            len(ai_ids),
        )

        human_scores = detector.get_scores(human_texts)
        for sample_id, score in zip(human_ids, human_scores):
            records.append(
                {
                    "sample_id": sample_id,
                    "dataset": dataset_name,
                    "method": "human",
                    "label": "human",
                    "detector_name": args.detector,
                    "detector_score": score,
                }
            )

        dataset_outputs = raw_outputs.get(dataset_name, {})
        for method_name, attacked_texts in dataset_outputs.items():
            if len(attacked_texts) != len(ai_ids):
                raise RuntimeError(
                    f"Length mismatch for {dataset_name}/{method_name}: "
                    f"{len(attacked_texts)} outputs vs {len(ai_ids)} samples"
                )
            scores = detector.get_scores(attacked_texts)
            for sample_id, score in zip(ai_ids, scores):
                records.append(
                    {
                        "sample_id": sample_id,
                        "dataset": dataset_name,
                        "method": method_name,
                        "label": "ai",
                        "detector_name": args.detector,
                        "detector_score": score,
                    }
                )

    df = pd.DataFrame(records)
    df.to_parquet(out_dir / f"{args.detector}.parquet")
    df.to_csv(out_dir / f"{args.detector}.csv", index=False)
    logger.info("Saved detector scores to %s", out_dir / f"{args.detector}.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
