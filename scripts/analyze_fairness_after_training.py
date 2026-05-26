#!/usr/bin/env python3
"""
Evaluate fairness after training by scoring model outputs.

This script loads a trained sampler checkpoint, generates paraphrases for
ESL vs native inputs, and compares detector scores across groups.
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

import tinker
import yaml
from tinker.types import SamplingParams
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from stealthrl.tinker.dataset import StealthRLDatasetBuilder
from stealthrl.tinker.detectors import DetectorEnsemble, _default_device

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

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


def _split_groups(records: List[Dict], max_samples: int | None, seed: int) -> Tuple[List[Dict], List[Dict]]:
    esl = [rec for rec in records if rec.get("is_esl")]
    native = [rec for rec in records if not rec.get("is_esl")]

    rng = random.Random(seed)
    if max_samples is not None:
        if len(esl) > max_samples:
            esl = rng.sample(esl, max_samples)
        if len(native) > max_samples:
            native = rng.sample(native, max_samples)
    return esl, native


def _read_sampler_path(run_dir: Path) -> str:
    info_path = run_dir / "checkpoints" / "final_checkpoint_info.json"
    if info_path.exists():
        with info_path.open() as f:
            info = json.load(f)
        return info["checkpoints"]["sampler_weights"]

    ckpt_path = run_dir / "checkpoints.jsonl"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint info found in {run_dir}")

    last = None
    with ckpt_path.open() as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    if not last:
        raise ValueError(f"No checkpoints found in {ckpt_path}")
    return last["sampler_path"]


def _read_model_name(run_dir: Path, config_file: str | None) -> str | None:
    if config_file:
        config_path = Path(config_file)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            model_cfg = cfg.get("model", {})
            name = model_cfg.get("name")
            if name:
                return name

    run_meta = run_dir / "run_metadata.json"
    if run_meta.exists():
        with run_meta.open() as f:
            meta = json.load(f)
        config_hint = meta.get("config_file")
        if config_hint:
            return _read_model_name(run_dir, config_hint)
    return None


def _build_prompt(ai_text: str) -> str:
    return (
        "Please paraphrase the following text while maintaining its meaning "
        "and ensuring it reads naturally:\n\n"
        f"{ai_text}\n\n"
        "Paraphrased text:"
    )


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


def _parse_thresholds(value: str) -> List[float]:
    if not value:
        return [0.5]
    return [float(v.strip()) for v in value.split(",") if v.strip()]


async def _sample_one(
    sampling_client,
    renderer,
    prompt_messages,
    params: SamplingParams,
    semaphore: asyncio.Semaphore,
    tokenizer,
):
    async with semaphore:
        prompt = renderer.build_generation_prompt(prompt_messages)
        response = await sampling_client.sample_async(
            prompt=prompt,
            sampling_params=params,
            num_samples=1,
        )
        if hasattr(response, "result_async"):
            result = await response.result_async()
        else:
            result = response
        seq = result.sequences[0]
        text = getattr(seq, "text", None)
        if not text:
            tokens = getattr(seq, "tokens", None)
            if tokens is None:
                raise ValueError("Sample response missing text and tokens")
            text = tokenizer.decode(tokens)
        return text


async def _generate_paraphrases(
    sampling_client,
    renderer,
    records: List[Dict],
    params: SamplingParams,
    max_concurrent: int,
    convo_prefix: List[renderers.Message],
    tokenizer,
    label: str,
    show_progress: bool,
):
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []
    for rec in records:
        prompt_messages = convo_prefix + [
            {
                "role": "user",
                "content": _build_prompt(rec.get("ai_text", "")),
            }
        ]
        task = asyncio.create_task(
            _sample_one(sampling_client, renderer, prompt_messages, params, semaphore, tokenizer)
        )
        tasks.append(task)

    if show_progress and tqdm is not None:
        results: List[str] = []
        with tqdm(total=len(tasks), desc=f"Sampling {label}", unit="sample") as pbar:
            for task in asyncio.as_completed(tasks):
                res = await task
                results.append(res)
                pbar.update(1)
        return results

    return await asyncio.gather(*tasks)


async def _run(args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, float | int | None]]]:
    data_path = Path(args.data_path)
    records = _select_records(data_path, args.split)
    if not records:
        raise SystemExit("No records found. Check --data-path and --split.")

    esl_records, native_records = _split_groups(records, args.max_samples, args.seed)
    logger.info("Sampling %d ESL and %d native examples", len(esl_records), len(native_records))

    sampler_path = args.sampler_path
    if not sampler_path:
        sampler_path = _read_sampler_path(Path(args.run_dir))
    logger.info("Using sampler path: %s", sampler_path)

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=sampler_path)
    model_name = args.model_name
    if not model_name:
        model_name = _read_model_name(Path(args.run_dir), args.config_file)
    if not model_name:
        raise SystemExit("Could not infer base model name; pass --model-name")
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(args.renderer, tokenizer=tokenizer)

    convo_prefix = []
    if args.few_shot == "standard":
        convo_prefix = StealthRLDatasetBuilder._standard_fewshot_prefix()

    params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    esl_paraphrases = await _generate_paraphrases(
        sampling_client,
        renderer,
        esl_records,
        params,
        args.max_concurrent,
        convo_prefix,
        tokenizer,
        label="ESL",
        show_progress=args.progress,
    )
    native_paraphrases = await _generate_paraphrases(
        sampling_client,
        renderer,
        native_records,
        params,
        args.max_concurrent,
        convo_prefix,
        tokenizer,
        label="native",
        show_progress=args.progress,
    )

    device = args.device or _default_device()
    detectors = DetectorEnsemble(
        detector_names=args.detectors,
        detector_weights=None,
        cache_path=args.cache_path,
        device=device,
        max_concurrent=args.detector_max_concurrent,
    )
    if args.prewarm:
        detectors.prewarm_models()

    thresholds = _parse_thresholds(args.thresholds)
    esl_scores = await detectors.compute_batch(esl_paraphrases)
    native_scores = await detectors.compute_batch(native_paraphrases)

    def extract_scores(results: List[Dict]) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {"ensemble": []}
        for name in detectors.detectors:
            out[name] = []
        for res in results:
            out["ensemble"].append(res["ensemble_prob"])
            for name, val in res["detector_scores"].items():
                out[name].append(val)
        return out

    esl_vals = extract_scores(esl_scores)
    native_vals = extract_scores(native_scores)

    summary: Dict[str, Dict[str, Dict[str, float | int | None]]] = {}
    for key in esl_vals:
        summary[f"esl/{key}"] = _summarize(esl_vals[key], thresholds)
        summary[f"native/{key}"] = _summarize(native_vals[key], thresholds)
        if esl_vals[key] and native_vals[key]:
            summary[f"diff/{key}"] = {
                "mean_delta": mean(esl_vals[key]) - mean(native_vals[key]),
                "median_delta": median(esl_vals[key]) - median(native_vals[key]),
                **{
                    f"flag_rate_delta@{t}": (
                        summary[f"esl/{key}"][f"flag_rate@{t}"]
                        - summary[f"native/{key}"][f"flag_rate@{t}"]
                    )
                    for t in thresholds
                },
            }

    detectors.close()

    if args.save_samples:
        samples_path = Path(args.save_samples)
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        with samples_path.open("w", encoding="utf-8") as f:
            for rec, para in zip(esl_records, esl_paraphrases):
                f.write(json.dumps({"group": "esl", "ai_text": rec.get("ai_text"), "paraphrase": para}) + "\n")
            for rec, para in zip(native_records, native_paraphrases):
                f.write(json.dumps({"group": "native", "ai_text": rec.get("ai_text"), "paraphrase": para}) + "\n")
        logger.info("Saved samples to %s", samples_path)

    return {"paraphrase_detector_scores": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fairness after training using model outputs")
    parser.add_argument("--run-dir", type=str, default=None, help="Training run dir with checkpoints.jsonl")
    parser.add_argument("--sampler-path", type=str, default=None, help="tinker:// sampler checkpoint path")
    parser.add_argument("--config-file", type=str, default=None, help="Config file path to infer model name")
    parser.add_argument("--model-name", type=str, default=None, help="Base model name for tokenizer")
    parser.add_argument("--data-path", type=str, required=True, help="Dataset directory with train/test JSONL")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "both"])
    parser.add_argument("--max-samples", type=int, default=200, help="Max samples per group")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--renderer", type=str, default="qwen3")
    parser.add_argument("--few-shot", type=str, default="standard", choices=["standard", "none"])
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--detectors", nargs="+", default=["fast_detectgpt", "ghostbuster"])
    parser.add_argument("--thresholds", type=str, default="0.5")
    parser.add_argument("--cache-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--prewarm", action="store_true")
    parser.add_argument("--detector-max-concurrent", type=int, default=4)
    parser.add_argument("--save-samples", type=str, default=None)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress bars")
    args = parser.parse_args()

    if not args.run_dir and not args.sampler_path:
        raise SystemExit("Provide --run-dir or --sampler-path")

    if args.few_shot == "none":
        args.few_shot = ""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    results = asyncio.run(_run(args))
    print(json.dumps(results, indent=2))

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote summary to %s", out_path)


if __name__ == "__main__":
    main()
