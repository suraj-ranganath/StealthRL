#!/usr/bin/env python3
"""
Assemble a full evaluation run from combined raw outputs and per-detector score files.
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
from eval.metrics import calibrate_thresholds, save_thresholds
from eval.plots import generate_all_tables
from eval.runner import EvalRunner, setup_logging


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize assembled StealthRL evaluation run")
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--raw-outputs", required=True)
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--device", default=None)
    parser.add_argument("--precomputed-quality", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _read_scores(scores_dir: Path) -> pd.DataFrame:
    frames = []
    for parquet_path in sorted(scores_dir.glob("*.parquet")):
        frames.append(pd.read_parquet(parquet_path))
    if not frames:
        raise FileNotFoundError(f"No detector score parquet files found in {scores_dir}")
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    args = parse_args()
    setup_logging(args.out_dir, args.log_level)

    samples_dir = Path(args.samples_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = json.loads((samples_dir / "dataset_samples.json").read_text())
    raw_outputs = json.loads(Path(args.raw_outputs).read_text())
    scores_df = _read_scores(Path(args.scores_dir))

    runner = EvalRunner(
        output_dir=str(out_dir),
        device=args.device,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
    )

    for dataset_name, dataset_ids in ids.items():
        runner.datasets[dataset_name] = load_eval_dataset_with_ids(
            name=dataset_name,
            human_ids=dataset_ids["human_ids"],
            ai_ids=dataset_ids["ai_ids"],
            cache_dir=args.cache_dir,
        )

    method_names = sorted(
        {
            method_name
            for dataset_outputs in raw_outputs.values()
            for method_name in dataset_outputs.keys()
        }
    )
    detector_names = sorted(scores_df["detector_name"].dropna().unique().tolist())
    runner.methods = {name: None for name in method_names}
    runner.detectors = {name: None for name in detector_names}
    runner.all_scores = scores_df.to_dict("records")

    human_scores = {
        detector_name: scores_df[
            (scores_df["label"] == "human") & (scores_df["detector_name"] == detector_name)
        ]["detector_score"].tolist()
        for detector_name in detector_names
    }
    runner.thresholds = calibrate_thresholds(human_scores, target_fpr=0.01)
    save_thresholds(runner.thresholds, str(out_dir / "thresholds.json"))

    runner.compute_all_metrics()
    if args.precomputed_quality:
        quality_df = pd.read_parquet(args.precomputed_quality)
        runner.all_quality = quality_df.to_dict("records")
        logger.info("Loaded precomputed quality metrics from %s", args.precomputed_quality)
    else:
        runner.compute_quality_metrics(raw_outputs, setting="default")

    with open(out_dir / "raw_outputs.json", "w") as f:
        json.dump(raw_outputs, f, indent=2, ensure_ascii=False)
    with open(out_dir / "dataset_samples.json", "w") as f:
        json.dump(ids, f, indent=2)

    runner.save_all_artifacts()
    generate_all_tables(
        detector_metrics=runner.all_metrics,
        quality_metrics=runner.all_quality,
        output_dir=str(out_dir / "tables"),
        format="latex",
    )
    logger.info("Finalized assembled eval run at %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
