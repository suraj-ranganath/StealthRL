#!/usr/bin/env python3
"""
Generate StealthRL (M2) outputs from a PEFT adapter or merged checkpoint.

This is used for stochastic-repeat analysis when the original Tinker sampler is
not available. It writes the same raw_outputs.json schema as
scripts/generate_method_outputs.py and keeps a resumable JSONL cache so full
MAGE runs can be restarted without losing completed generations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.data import load_eval_dataset_with_ids
from eval.methods.stealthrl import PARAPHRASE_PROMPT


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full StealthRL M2 outputs with HF PEFT")
    parser.add_argument("--samples-dir", required=True, help="Directory containing dataset_samples.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--method-name", default="m2", help="Method key to store in raw_outputs.json")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--adapter-model",
        default=None,
        help="Optional PEFT adapter path. Omit when --base-model is an already merged StealthRL checkpoint.",
    )
    parser.add_argument("--no-lora", action="store_true", help="Treat --base-model as an already-merged StealthRL checkpoint")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-model-len", type=int, default=1536)
    parser.add_argument("--resume-jsonl", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _load_completed(path: Path) -> dict[str, str]:
    completed: dict[str, str] = {}
    if not path.exists():
        return completed
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            completed[row["sample_id"]] = row["text_out"]
    return completed


def _format_prompt(tokenizer, text: str) -> str:
    prompt = PARAPHRASE_PROMPT.format(text=text)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _clean_generation(text: str) -> str:
    text = text.strip()
    if text.startswith('"') and text.endswith('"') and len(text) > 1:
        text = text[1:-1].strip()
    return text


def _load_eval_payload(args: argparse.Namespace, completed: dict[str, str]) -> tuple[dict, dict, dict]:
    ids = json.loads((Path(args.samples_dir) / "dataset_samples.json").read_text())
    datasets: dict[str, dict] = {}
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
        attacked_texts: list[str | None] = [completed.get(sample_id) for sample_id in ai_ids]
        datasets[dataset_name] = {
            "ai_texts": ai_texts,
            "ai_ids": ai_ids,
            "attacked_texts": attacked_texts,
            "n_human": len(dataset.human_samples),
            "n_ai": len(dataset.ai_samples),
        }
        summary[dataset_name] = {
            "n_human": len(dataset.human_samples),
            "n_ai": len(dataset.ai_samples),
        }
        outputs[dataset_name] = {args.method_name: attacked_texts}
    return datasets, outputs, summary


def _generate_with_transformers(args: argparse.Namespace, completed: dict[str, str], resume_path: Path) -> tuple[dict, dict]:
    logger.info("Loading tokenizer: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info("Loading base model: %s", args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=_dtype(args.dtype),
        device_map=args.device_map,
    )
    logger.info("Loading adapter: %s", args.adapter_model)
    model = PeftModel.from_pretrained(base_model, args.adapter_model, is_trainable=False)
    model.eval()

    datasets, outputs, summary = _load_eval_payload(args, completed)

    cache_file = resume_path.open("a", buffering=1)
    try:
        for dataset_name, payload in datasets.items():
            ai_texts = payload["ai_texts"]
            ai_ids = payload["ai_ids"]
            attacked_texts = payload["attacked_texts"]

            pending = [(i, sample_id, text) for i, (sample_id, text) in enumerate(zip(ai_ids, ai_texts)) if attacked_texts[i] is None]
            logger.info(
                "Generating %s/%s: %d total, %d pending, batch_size=%d",
                dataset_name,
                args.method_name,
                len(ai_texts),
                len(pending),
                args.batch_size,
            )

            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                indices, sample_ids, texts = zip(*batch)
                prompts = [_format_prompt(tokenizer, text) for text in texts]
                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_input_tokens,
                ).to(model.device)

                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                input_len = encoded["input_ids"].shape[1]
                decoded = tokenizer.batch_decode(generated[:, input_len:], skip_special_tokens=True)
                for index, sample_id, text_out in zip(indices, sample_ids, decoded):
                    cleaned = _clean_generation(text_out)
                    if not cleaned:
                        raise RuntimeError(f"Empty generation for sample {sample_id}")
                    attacked_texts[index] = cleaned
                    cache_file.write(json.dumps({"sample_id": sample_id, "text_out": cleaned}, ensure_ascii=False) + "\n")

                done = min(start + len(batch), len(pending))
                if done % (args.batch_size * 25) == 0 or done == len(pending):
                    logger.info("Progress %s/%s: %d/%d pending completed", dataset_name, args.method_name, done, len(pending))

            if any(text is None for text in attacked_texts):
                missing = sum(text is None for text in attacked_texts)
                raise RuntimeError(f"Missing {missing} generations for {dataset_name}/{args.method_name}")
            outputs[dataset_name][args.method_name] = [str(text) for text in attacked_texts]
    finally:
        cache_file.close()
    return outputs, summary


def _generate_with_vllm(args: argparse.Namespace, completed: dict[str, str], resume_path: Path) -> tuple[dict, dict]:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    logger.info("Preparing tokenizer: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    datasets, outputs, summary = _load_eval_payload(args, completed)

    adapter_path = None
    if not args.no_lora:
        logger.info("Resolving adapter: %s", args.adapter_model)
        adapter_path = snapshot_download(args.adapter_model) if "/" in args.adapter_model and not Path(args.adapter_model).exists() else args.adapter_model
        logger.info("Loading vLLM base=%s adapter=%s", args.base_model, adapter_path)
    else:
        logger.info("Loading vLLM merged model=%s", args.base_model)

    llm_kwargs = {
        "model": args.base_model,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "trust_remote_code": True,
    }
    if not args.no_lora:
        llm_kwargs.update({"enable_lora": True, "max_lora_rank": 32})
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    lora_request = None if args.no_lora else LoRARequest("stealthrl", 1, str(adapter_path))

    with resume_path.open("a", buffering=1) as cache_file:
        for dataset_name, payload in datasets.items():
            ai_texts = payload["ai_texts"]
            ai_ids = payload["ai_ids"]
            attacked_texts = payload["attacked_texts"]
            pending = [(i, sample_id, text) for i, (sample_id, text) in enumerate(zip(ai_ids, ai_texts)) if attacked_texts[i] is None]
            logger.info(
                "Generating %s/%s with vLLM: %d total, %d pending",
                dataset_name,
                args.method_name,
                len(ai_texts),
                len(pending),
            )
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                indices, sample_ids, texts = zip(*batch)
                prompts = [_format_prompt(tokenizer, text) for text in texts]
                if lora_request is None:
                    results = llm.generate(prompts, sampling, use_tqdm=False)
                else:
                    results = llm.generate(prompts, sampling, lora_request=lora_request, use_tqdm=False)
                if len(results) != len(batch):
                    raise RuntimeError(f"vLLM returned {len(results)} outputs for batch of {len(batch)}")
                for index, sample_id, result in zip(indices, sample_ids, results):
                    cleaned = _clean_generation(result.outputs[0].text)
                    if not cleaned:
                        raise RuntimeError(f"Empty generation for sample {sample_id}")
                    attacked_texts[index] = cleaned
                    cache_file.write(json.dumps({"sample_id": sample_id, "text_out": cleaned}, ensure_ascii=False) + "\n")
                done = min(start + len(batch), len(pending))
                if done % (args.batch_size * 10) == 0 or done == len(pending):
                    logger.info("Progress %s/%s: %d/%d pending completed", dataset_name, args.method_name, done, len(pending))

            if any(text is None for text in attacked_texts):
                missing = sum(text is None for text in attacked_texts)
                raise RuntimeError(f"Missing {missing} generations for {dataset_name}/{args.method_name}")
            outputs[dataset_name][args.method_name] = [str(text) for text in attacked_texts]
    return outputs, summary


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.backend != "vllm" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    samples_dir = Path(args.samples_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = Path(args.resume_jsonl) if args.resume_jsonl else out_dir / f"{args.method_name}_hf_cache.jsonl"

    completed = _load_completed(resume_path)
    logger.info("Loaded %d completed generations from %s", len(completed), resume_path)

    if args.backend == "vllm":
        outputs, summary = _generate_with_vllm(args, completed, resume_path)
    else:
        outputs, summary = _generate_with_transformers(args, completed, resume_path)

    (out_dir / "dataset_samples.json").write_text((samples_dir / "dataset_samples.json").read_text())
    with (out_dir / "raw_outputs.json").open("w") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    with (out_dir / "generation_summary.json").open("w") as f:
        json.dump(
            {
                "method": args.method_name,
                "backend": "hf_peft",
                "inference_backend": args.backend,
                "base_model": args.base_model,
                "adapter_model": None if args.no_lora else args.adapter_model,
                "no_lora": args.no_lora,
                "batch_size": args.batch_size,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
                "datasets": summary,
            },
            f,
            indent=2,
        )
    logger.info("Saved outputs to %s", out_dir / "raw_outputs.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
