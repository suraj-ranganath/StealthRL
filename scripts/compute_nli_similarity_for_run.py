#!/usr/bin/env python3
"""
Compute bidirectional NLI entailment for full-pool paraphrase quality analysis.

For each original/paraphrase pair, the script scores original=>paraphrase and
paraphrase=>original with an MNLI model, then stores the minimum entailment
probability as a conservative semantic preservation metric.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.data import load_eval_dataset_with_ids


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute bidirectional NLI similarity for an eval run")
    parser.add_argument("--run-dir", required=True, help="Assembled run directory containing raw_outputs.json")
    parser.add_argument("--samples-dir", default=None, help="Optional samples dir; defaults to run-dir")
    parser.add_argument("--raw-outputs", default=None, help="Optional raw outputs path; defaults to run-dir/raw_outputs.json")
    parser.add_argument("--out-path", default=None, help="Output parquet path")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--methods", nargs="+", default=["m0", "m1", "m2", "m3", "m4", "m5"])
    parser.add_argument("--model", default="microsoft/deberta-large-mnli")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume-path", default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _entailment_index(model) -> int:
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    for index, label in labels.items():
        if "entail" in label:
            return index
    raise RuntimeError(f"Could not find entailment label in {model.config.id2label}")


def _load_completed(path: Path | None) -> dict[tuple[str, str], dict]:
    completed: dict[tuple[str, str], dict] = {}
    if path is None or not path.exists():
        return completed
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            completed[(row["sample_id"], row["method"])] = row
    return completed


def _score_pairs(
    tokenizer,
    model,
    pairs: list[tuple[str, str]],
    entail_idx: int,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        premises = [p for p, _ in batch]
        hypotheses = [h for _, h in batch]
        encoded = tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, entail_idx]
        scores.extend(probs.detach().cpu().tolist())
    return scores


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, --num-shards)")

    run_dir = Path(args.run_dir)
    samples_dir = Path(args.samples_dir) if args.samples_dir else run_dir
    raw_outputs_path = Path(args.raw_outputs) if args.raw_outputs else run_dir / "raw_outputs.json"
    out_path = Path(args.out_path) if args.out_path else run_dir / "quality_nli.parquet"
    resume_path = Path(args.resume_path) if args.resume_path else out_path.with_suffix(".jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ids = json.loads((samples_dir / "dataset_samples.json").read_text())
    raw_outputs = json.loads(raw_outputs_path.read_text())
    completed = _load_completed(resume_path)
    logger.info("Loaded %d completed NLI rows from %s", len(completed), resume_path)

    rows_to_score: list[dict] = []
    for dataset_name, dataset_ids in ids.items():
        dataset = load_eval_dataset_with_ids(
            name=dataset_name,
            human_ids=dataset_ids["human_ids"],
            ai_ids=dataset_ids["ai_ids"],
            cache_dir=args.cache_dir,
        )
        original_by_id = {sample.id: sample.text for sample in dataset.ai_samples}
        sample_ids = [sample.id for sample in dataset.ai_samples]
        for method in args.methods:
            attacked_texts = raw_outputs.get(dataset_name, {}).get(method)
            if attacked_texts is None:
                logger.warning("Missing outputs for %s/%s", dataset_name, method)
                continue
            if len(attacked_texts) != len(sample_ids):
                raise RuntimeError(f"Length mismatch for {dataset_name}/{method}")
            for index, (sample_id, text_out) in enumerate(zip(sample_ids, attacked_texts)):
                if index % args.num_shards != args.shard_id:
                    continue
                if (sample_id, method) in completed:
                    continue
                rows_to_score.append(
                    {
                        "dataset": dataset_name,
                        "sample_id": sample_id,
                        "method": method,
                        "original_text": original_by_id[sample_id],
                        "paraphrased_text": text_out,
                    }
                )

    logger.info(
        "NLI scoring %d rows for methods=%s shard=%d/%d",
        len(rows_to_score),
        ",".join(args.methods),
        args.shard_id,
        args.num_shards,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()
    entail_idx = _entailment_index(model)
    logger.info("Loaded %s on %s; entailment index=%d", args.model, device, entail_idx)

    with resume_path.open("a", buffering=1) as cache_file:
        for start in range(0, len(rows_to_score), args.batch_size):
            batch_rows = rows_to_score[start : start + args.batch_size]
            forward_pairs = [(row["original_text"], row["paraphrased_text"]) for row in batch_rows]
            reverse_pairs = [(row["paraphrased_text"], row["original_text"]) for row in batch_rows]
            fwd = _score_pairs(tokenizer, model, forward_pairs, entail_idx, args.batch_size, device)
            rev = _score_pairs(tokenizer, model, reverse_pairs, entail_idx, args.batch_size, device)
            for row, fwd_score, rev_score in zip(batch_rows, fwd, rev):
                out = {
                    "dataset": row["dataset"],
                    "sample_id": row["sample_id"],
                    "method": row["method"],
                    "nli_entailment_fwd": float(fwd_score),
                    "nli_entailment_rev": float(rev_score),
                    "nli_bidirectional": float(min(fwd_score, rev_score)),
                }
                cache_file.write(json.dumps(out) + "\n")
            done = min(start + len(batch_rows), len(rows_to_score))
            if done % (args.batch_size * 25) == 0 or done == len(rows_to_score):
                logger.info("Progress: %d/%d rows scored", done, len(rows_to_score))

    all_rows = list(_load_completed(resume_path).values())
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["method", "sample_id"]).reset_index(drop=True)
    df.to_parquet(out_path)
    df.to_csv(out_path.with_suffix(".csv"), index=False)
    logger.info("Saved NLI metrics to %s (%d rows)", out_path, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
