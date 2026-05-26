#!/usr/bin/env python3
"""
Generate attack outputs for a single method without loading detector models.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.data import load_eval_dataset_with_ids
from eval.methods import get_method


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw outputs for a single eval method")
    parser.add_argument("--method", required=True, help="Method name (m0-m5)")
    parser.add_argument("--samples-dir", required=True, help="Directory containing dataset_samples.json")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--cache-dir", default="cache", help="Dataset/model cache directory")
    parser.add_argument("--checkpoint-json", default=None, help="Tinker checkpoint JSON for m2")
    parser.add_argument("--n-candidates", type=int, default=1, help="Candidates per sample")
    parser.add_argument("--m1-backend", default="vllm", choices=["vllm", "tinker"])
    parser.add_argument("--method-device", default=None, help="Optional detector/similarity device override")
    parser.add_argument("--vllm-max-model-len", type=int, default=None, help="Override vLLM max model length")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=None, help="Override vLLM GPU memory utilization")
    parser.add_argument("--max-invalid-retries", type=int, default=3, help="Retry invalid samples up to N additional times")
    parser.add_argument("--tinker-concurrency", type=int, default=64)
    parser.add_argument("--tinker-chunk-size", type=int, default=256)
    parser.add_argument("--tinker-max-retries", type=int, default=2)
    parser.add_argument("--tinker-backoff-s", type=float, default=0.5)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    samples_dir = Path(args.samples_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = json.loads((samples_dir / "dataset_samples.json").read_text())
    outputs: dict[str, dict[str, list[str]]] = {}
    summary: dict[str, dict[str, int]] = {}

    for dataset_name, dataset_ids in ids.items():
        dataset = load_eval_dataset_with_ids(
            name=dataset_name,
            human_ids=dataset_ids["human_ids"],
            ai_ids=dataset_ids["ai_ids"],
            cache_dir=args.cache_dir,
        )
        ai_texts = [sample.text for sample in dataset.ai_samples]
        ai_ids = [sample.id for sample in dataset.ai_samples]

        if args.method in ("m1_tinker", "simple_paraphrase_tinker") or (
            args.method in ("m1", "simple_paraphrase") and args.m1_backend == "tinker"
        ):
            method = get_method(
                "m1_tinker",
                tinker_concurrency=args.tinker_concurrency,
                tinker_chunk_size=args.tinker_chunk_size,
                tinker_max_retries=args.tinker_max_retries,
                tinker_backoff_s=args.tinker_backoff_s,
            )
        elif args.method in ("m2", "stealthrl"):
            if not args.checkpoint_json:
                raise ValueError("--checkpoint-json is required for m2/stealthrl")
            method = get_method(
                args.method,
                checkpoint_json=args.checkpoint_json,
                tinker_concurrency=args.tinker_concurrency,
                tinker_chunk_size=args.tinker_chunk_size,
                tinker_max_retries=args.tinker_max_retries,
                tinker_backoff_s=args.tinker_backoff_s,
            )
        else:
            method_kwargs = {}
            if args.method in {"m1", "simple_paraphrase", "m3", "adversarial_paraphrasing", "m4", "authormist"}:
                if args.method_device is not None:
                    method_kwargs["device"] = args.method_device
                if args.vllm_max_model_len is not None:
                    method_kwargs["max_model_len"] = args.vllm_max_model_len
                if args.vllm_gpu_memory_utilization is not None:
                    method_kwargs["gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
            method = get_method(args.method, **method_kwargs)

        logger.info(
            "Generating outputs for %s on %s (%d AI samples, n_candidates=%d)",
            args.method,
            dataset_name,
            len(ai_texts),
            args.n_candidates,
        )
        results = method.attack_batch(ai_texts, n_candidates=args.n_candidates)
        if len(results) != len(ai_texts):
            raise RuntimeError(
                f"{args.method} returned {len(results)} outputs for {len(ai_texts)} inputs"
            )

        invalid_indices = [index for index, result in enumerate(results) if not result.valid]
        for retry_round in range(args.max_invalid_retries):
            if not invalid_indices:
                break
            logger.warning(
                "%s produced %d invalid outputs on %s; retry round %d/%d",
                args.method,
                len(invalid_indices),
                dataset_name,
                retry_round + 1,
                args.max_invalid_retries,
            )
            retry_texts = [ai_texts[index] for index in invalid_indices]
            retry_results = [method.attack(text, n_candidates=args.n_candidates) for text in retry_texts]
            for index, retry_result in zip(invalid_indices, retry_results):
                results[index] = retry_result
            invalid_indices = [index for index, result in enumerate(results) if not result.valid]

        invalid = []
        attacked_texts = []
        for sample_id, result in zip(ai_ids, results):
            if not result.valid:
                invalid.append({"sample_id": sample_id, "fail_reason": result.fail_reason})
            attacked_texts.append(result.text)

        if invalid:
            preview = ", ".join(
                f"{item['sample_id']}:{item['fail_reason']}" for item in invalid[:10]
            )
            raise RuntimeError(
                f"{args.method} produced {len(invalid)} invalid outputs on {dataset_name}. "
                f"Examples: {preview}"
            )

        outputs[dataset_name] = {args.method: attacked_texts}
        summary[dataset_name] = {
            "n_human": len(dataset.human_samples),
            "n_ai": len(dataset.ai_samples),
        }

    (out_dir / "dataset_samples.json").write_text((samples_dir / "dataset_samples.json").read_text())
    with open(out_dir / "raw_outputs.json", "w") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    with open(out_dir / "generation_summary.json", "w") as f:
        json.dump(
            {
                "method": args.method,
                "n_candidates": args.n_candidates,
                "datasets": summary,
            },
            f,
            indent=2,
        )

    logger.info("Saved raw outputs to %s", out_dir / "raw_outputs.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
