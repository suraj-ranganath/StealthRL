#!/usr/bin/env python3
"""
Compute quality metrics for one or more methods and save parquet/csv shards.
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
from eval.metrics import E5SimilarityScorer, PerplexityScorer, compute_quality_metrics


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute quality metrics for selected methods")
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--raw-outputs", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    samples_dir = Path(args.samples_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = json.loads((samples_dir / "dataset_samples.json").read_text())
    raw_outputs = json.loads(Path(args.raw_outputs).read_text())

    sim_scorer = E5SimilarityScorer(device=args.device)
    ppl_scorer = PerplexityScorer(device=args.device)

    rows = []
    for dataset_name, dataset_ids in ids.items():
        dataset = load_eval_dataset_with_ids(
            name=dataset_name,
            human_ids=dataset_ids["human_ids"],
            ai_ids=dataset_ids["ai_ids"],
            cache_dir=args.cache_dir,
        )
        original_texts = [sample.text for sample in dataset.ai_samples]
        sample_ids = [sample.id for sample in dataset.ai_samples]

        for method_name in args.methods:
            attacked_texts = raw_outputs.get(dataset_name, {}).get(method_name)
            if not attacked_texts:
                logger.warning("Skipping missing outputs for %s/%s", dataset_name, method_name)
                continue

            logger.info(
                "Quality metrics: dataset=%s method=%s n=%d",
                dataset_name,
                method_name,
                len(attacked_texts),
            )
            quality = compute_quality_metrics(
                original_texts=original_texts,
                paraphrased_texts=attacked_texts,
                sample_ids=sample_ids,
                method=method_name,
                setting="default",
                similarity_scorer=sim_scorer,
                perplexity_scorer=ppl_scorer,
            )
            for item in quality:
                record = item.to_dict()
                record["dataset"] = dataset_name
                rows.append(record)
            logger.info("Completed quality metrics: dataset=%s method=%s", dataset_name, method_name)

    df = pd.DataFrame(rows)
    methods_slug = "_".join(args.methods)
    df.to_parquet(out_dir / f"{methods_slug}.parquet")
    df.to_csv(out_dir / f"{methods_slug}.csv", index=False)
    logger.info("Saved %d quality rows for %s", len(df), methods_slug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
