#!/usr/bin/env python3
"""
Compute BERTScore for an assembled StealthRL evaluation run.

The script reuses an existing run directory containing dataset_samples.json,
raw_outputs.json, and quality.parquet. It updates quality.parquet/csv in place
with bertscore_precision, bertscore_recall, and bertscore_f1 columns, saving
after each chunk so long runs can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.data import load_eval_dataset_with_ids
from eval.plots import create_quality_table


LOGGER = logging.getLogger("compute_bertscore_for_run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute BERTScore for an assembled eval run")
    parser.add_argument("--run-dir", required=True, help="Assembled run directory")
    parser.add_argument("--methods", nargs="+", default=["m0", "m1", "m2", "m3", "m4", "m5"])
    parser.add_argument("--model-type", default="roberta-large")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--device", default=None, help="Example: cuda:2 or cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--limit-per-method", type=int, default=None, help="Smoke-test cap")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--rescale-with-baseline", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute rows with existing BERTScore")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _chunks(values: List[int], size: int) -> Iterable[List[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _load_original_texts(run_dir: Path, cache_dir: str) -> Dict[str, Dict[str, str]]:
    ids = json.loads((run_dir / "dataset_samples.json").read_text())
    originals: Dict[str, Dict[str, str]] = {}

    for dataset_name, split_ids in ids.items():
        dataset = load_eval_dataset_with_ids(
            name=dataset_name,
            human_ids=split_ids["human_ids"],
            ai_ids=split_ids["ai_ids"],
            cache_dir=cache_dir,
        )
        originals[dataset_name] = {sample.id: sample.text for sample in dataset.ai_samples}

    return originals


def _load_paraphrases(run_dir: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
    ids = json.loads((run_dir / "dataset_samples.json").read_text())
    raw_outputs = json.loads((run_dir / "raw_outputs.json").read_text())
    paraphrases: Dict[str, Dict[str, Dict[str, str]]] = {}

    for dataset_name, split_ids in ids.items():
        ai_ids = split_ids["ai_ids"]
        paraphrases[dataset_name] = {}
        for method, outputs in raw_outputs[dataset_name].items():
            if len(outputs) != len(ai_ids):
                raise ValueError(
                    f"Length mismatch for {dataset_name}/{method}: {len(outputs)} vs {len(ai_ids)}"
                )
            paraphrases[dataset_name][method] = dict(zip(ai_ids, outputs))

    return paraphrases


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    run_dir = Path(args.run_dir)
    quality_path = run_dir / "quality.parquet"
    if not quality_path.exists():
        raise FileNotFoundError(f"Missing {quality_path}")

    quality_df = pd.read_parquet(quality_path)
    for col in ("bertscore_precision", "bertscore_recall", "bertscore_f1"):
        if col not in quality_df.columns:
            quality_df[col] = pd.NA

    originals = _load_original_texts(run_dir, args.cache_dir)
    paraphrases = _load_paraphrases(run_dir)

    from bert_score import BERTScorer

    LOGGER.info(
        "Loading BERTScorer(model_type=%s, lang=%s, device=%s)",
        args.model_type,
        args.lang,
        args.device or "auto",
    )
    scorer = BERTScorer(
        model_type=args.model_type,
        lang=args.lang,
        device=args.device,
        rescale_with_baseline=args.rescale_with_baseline,
    )

    target_mask = quality_df["method"].isin(args.methods)
    if not args.force:
        target_mask &= quality_df["bertscore_f1"].isna()

    pending_indices = quality_df[target_mask].index.tolist()
    if args.limit_per_method is not None:
        capped = []
        for method in args.methods:
            method_indices = quality_df[target_mask & (quality_df["method"] == method)].index.tolist()
            capped.extend(method_indices[: args.limit_per_method])
        pending_indices = capped

    LOGGER.info("Rows pending BERTScore: %d", len(pending_indices))
    if not pending_indices:
        return 0

    for chunk_no, chunk_indices in enumerate(_chunks(pending_indices, args.chunk_size), start=1):
        refs: List[str] = []
        cands: List[str] = []
        for idx in chunk_indices:
            row = quality_df.loc[idx]
            dataset_name = row["dataset"]
            method = row["method"]
            sample_id = row["sample_id"]
            refs.append(originals[dataset_name][sample_id])
            cands.append(paraphrases[dataset_name][method][sample_id])

        precision, recall, f1 = scorer.score(cands, refs, batch_size=args.batch_size)
        quality_df.loc[chunk_indices, "bertscore_precision"] = precision.cpu().numpy().tolist()
        quality_df.loc[chunk_indices, "bertscore_recall"] = recall.cpu().numpy().tolist()
        quality_df.loc[chunk_indices, "bertscore_f1"] = f1.cpu().numpy().tolist()

        quality_df.to_parquet(quality_path)
        quality_df.to_csv(run_dir / "quality.csv", index=False)
        LOGGER.info("Saved chunk %d (%d rows)", chunk_no, len(chunk_indices))

    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    create_quality_table(
        quality_df.to_dict("records"),
        output_path=str(tables_dir / "table_quality.md"),
        format="markdown",
    )
    create_quality_table(
        quality_df.to_dict("records"),
        output_path=str(tables_dir / "table_quality.tex"),
        format="latex",
    )

    summary = quality_df.groupby("method")["bertscore_f1"].mean().round(4)
    LOGGER.info("BERTScore F1 means:\n%s", summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
