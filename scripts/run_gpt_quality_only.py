#!/usr/bin/env python3
"""
Run GPT-based Likert quality evaluation only (no detectors/attacks).

This script reuses an existing eval run directory (dataset_samples.json + raw_outputs.json),
selects a shared subset of AI sample ids per dataset, and computes GPT quality + similarity
ratings for specified methods. It then updates quality.parquet/csv, the quality table,
and the Likert plot.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.data import load_eval_dataset_with_ids
from eval.plots import create_quality_table, create_quality_likert_chart
from eval.quality_judge import (
    GPTQualityConfig,
    _hash_key,
    _load_cache,
    run_gpt_quality_judge,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT quality Likert evaluation only")
    parser.add_argument("--run-dir", type=str, default="outputs/eval_runs/PLATINUM_run")
    parser.add_argument("--methods", nargs="+", default=["m1", "m2", "m3", "m5"])
    parser.add_argument("--model", type=str, default="gpt-5-nano")
    parser.add_argument("--max-per-method", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--request-timeout-s", type=int, default=60)
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--openai-api-key", type=str, default=None)
    parser.add_argument("--no-clear-existing", action="store_true")
    return parser.parse_args()


def _select_subset(ai_ids: List[str], max_per_method: int, seed: int, dataset_name: str) -> List[str]:
    rng = random.Random(f"{seed}:{dataset_name}")
    if len(ai_ids) <= max_per_method:
        return list(ai_ids)
    selected = set(rng.sample(ai_ids, max_per_method))
    return [sid for sid in ai_ids if sid in selected]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    methods = args.methods
    setting = "default"

    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set (or pass --openai-api-key)")

    quality_path = run_dir / "quality.parquet"
    if not quality_path.exists():
        raise SystemExit(f"Missing {quality_path}")
    quality_df = pd.read_parquet(quality_path)
    cache_path = run_dir / "quality_gpt.jsonl"
    cache = _load_cache(cache_path)

    ids = json.loads((run_dir / "dataset_samples.json").read_text())
    raw_outputs = json.loads((run_dir / "raw_outputs.json").read_text())

    def get_method_outputs(dataset_name: str, method: str) -> List[str]:
        ds_outputs = raw_outputs.get(dataset_name, {})
        if method in ds_outputs:
            return ds_outputs[method]

        # Fallback: look for a method-specific run directory (e.g., PLATINUM_run_m5)
        fallback_dir = run_dir.parent / f"{run_dir.name}_{method}"
        fallback_path = fallback_dir / "raw_outputs.json"
        if fallback_path.exists():
            fallback_outputs = json.loads(fallback_path.read_text())
            if dataset_name in fallback_outputs and method in fallback_outputs[dataset_name]:
                return fallback_outputs[dataset_name][method]

        raise KeyError(f"Missing outputs for {dataset_name}/{method} in {run_dir} (and fallback {fallback_dir})")

    selected_ids_by_ds: Dict[str, List[str]] = {}
    for dataset_name, split_ids in ids.items():
        selected_ids_by_ds[dataset_name] = _select_subset(
            split_ids["ai_ids"],
            args.max_per_method,
            args.seed,
            dataset_name,
        )

    if not args.no_clear_existing:
        cols_to_clear = [
            "quality_rating",
            "similarity_rating",
            "quality_justification",
            "similarity_justification",
            "quality_model",
        ]
        for dataset_name, selected_ids in selected_ids_by_ds.items():
            mask = (
                quality_df["method"].isin(methods)
                & (quality_df["setting"] == setting)
                & (quality_df["dataset"] == dataset_name)
            )
            quality_df.loc[mask, cols_to_clear] = None

    existing_ok = set()
    if args.no_clear_existing:
        existing = quality_df[
            (quality_df["method"].isin(methods))
            & (quality_df["setting"] == setting)
            & (quality_df["quality_rating"].notna())
            & (quality_df["similarity_rating"].notna())
        ][["sample_id", "method", "setting", "dataset"]]
        existing_ok = set(tuple(row) for row in existing.itertuples(index=False, name=None))

    items: List[Dict] = []
    cache_hits = 0
    for dataset_name, split_ids in ids.items():
        ai_ids = split_ids["ai_ids"]
        selected_ids = selected_ids_by_ds[dataset_name]

        dataset = load_eval_dataset_with_ids(
            name=dataset_name,
            human_ids=split_ids["human_ids"],
            ai_ids=ai_ids,
            cache_dir=args.cache_dir,
        )
        orig_map = {s.id: s.text for s in dataset.ai_samples}

        for method in methods:
            attacked_texts = get_method_outputs(dataset_name, method)
            if len(attacked_texts) != len(ai_ids):
                raise ValueError(
                    f"Length mismatch for {dataset_name}/{method}: "
                    f"{len(attacked_texts)} vs {len(ai_ids)}"
                )
            para_map = dict(zip(ai_ids, attacked_texts))
            for sid in selected_ids:
                key = (sid, method, setting, dataset_name)
                if key in existing_ok:
                    continue
                original = orig_map[sid]
                paraphrased = para_map[sid]
                quality_cache = cache.get(_hash_key(args.model, "quality", original, paraphrased))
                similarity_cache = cache.get(_hash_key(args.model, "similarity", original, paraphrased))
                if quality_cache and quality_cache.get("quality_rating") is not None:
                    mask = (
                        (quality_df["sample_id"] == sid)
                        & (quality_df["method"] == method)
                        & (quality_df["setting"] == setting)
                        & (quality_df["dataset"] == dataset_name)
                    )
                    quality_df.loc[mask, "quality_rating"] = quality_cache.get("quality_rating")
                    quality_df.loc[mask, "quality_justification"] = quality_cache.get("quality_justification")
                    quality_df.loc[mask, "quality_model"] = args.model
                if similarity_cache and similarity_cache.get("similarity_rating") is not None:
                    mask = (
                        (quality_df["sample_id"] == sid)
                        & (quality_df["method"] == method)
                        & (quality_df["setting"] == setting)
                        & (quality_df["dataset"] == dataset_name)
                    )
                    quality_df.loc[mask, "similarity_rating"] = similarity_cache.get("similarity_rating")
                    quality_df.loc[mask, "similarity_justification"] = similarity_cache.get("similarity_justification")
                    quality_df.loc[mask, "quality_model"] = args.model
                if (
                    quality_cache
                    and quality_cache.get("quality_rating") is not None
                    and similarity_cache
                    and similarity_cache.get("similarity_rating") is not None
                ):
                    cache_hits += 1
                    continue
                items.append({
                    "sample_id": sid,
                    "dataset": dataset_name,
                    "method": method,
                    "setting": setting,
                    "original": original,
                    "paraphrased": paraphrased,
                })

    if not items:
        print(f"No items to evaluate. Loaded {cache_hits} fully cached items.")
    else:
        print(f"Loaded {cache_hits} fully cached items; evaluating {len(items)} remaining items.")

    by_method_counts = {}
    for item in items:
        by_method_counts[item["method"]] = by_method_counts.get(item["method"], 0) + 1
    max_per_method_total = max(by_method_counts.values())

    config = GPTQualityConfig(
        model=args.model,
        max_per_method=max_per_method_total,
        seed=args.seed,
        cache_path=cache_path,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        request_timeout_s=args.request_timeout_s,
    )

    results = run_gpt_quality_judge(api_key=api_key, items=items, config=config) if items else []

    if results:
        gpt_df = pd.DataFrame(results).set_index(["sample_id", "method", "setting"])
        quality_df = quality_df.set_index(["sample_id", "method", "setting"])
        quality_df.update(gpt_df)
        quality_df = quality_df.reset_index()

    quality_df.to_parquet(quality_path)
    quality_df.to_csv(run_dir / "quality.csv", index=False)

    gpt_only = quality_df[
        quality_df["method"].isin(methods)
        & quality_df["quality_rating"].notna()
        & quality_df["similarity_rating"].notna()
    ].copy()
    gpt_only.to_parquet(run_dir / "quality_gpt.parquet")
    gpt_only.to_csv(run_dir / "quality_gpt.csv", index=False)

    create_quality_table(
        quality_df.to_dict("records"),
        output_path=str(run_dir / "tables" / "table_quality.md"),
        format="markdown",
    )
    create_quality_table(
        quality_df.to_dict("records"),
        output_path=str(run_dir / "tables" / "table_quality.tex"),
        format="latex",
    )
    create_quality_likert_chart(
        quality_df,
        title="Text Quality Evaluation",
        output_path=str(run_dir / "figures" / "fig_quality_likert.png"),
    )

    counts = quality_df[quality_df["quality_rating"].notna() & quality_df["method"].isin(methods)]
    print(counts["method"].value_counts().sort_index())


if __name__ == "__main__":
    main()
