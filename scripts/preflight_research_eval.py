#!/usr/bin/env python3
"""
Research-grade preflight for the staged full-MAGE evaluation pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.data import MAGEDataset, load_eval_dataset


logger = logging.getLogger(__name__)


TEST_TEXT = (
    "Artificial intelligence systems are increasingly used in education, healthcare, and scientific "
    "research, where reliability and interpretability remain essential for real-world deployment."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight the staged StealthRL eval pipeline")
    parser.add_argument("--env-file", default="/data/suraj/.config/stealthrl/eval.env")
    parser.add_argument("--checkpoint-json", default="/data/suraj/.config/stealthrl/m2_checkpoint.json")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _load_env_file(path: str) -> dict[str, str]:
    env = os.environ.copy()
    env_path = Path(path)
    if not env_path.exists():
        raise FileNotFoundError(f"Missing env file: {env_path}")
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        env[key.strip()] = value
    return env


def _run(label: str, cmd: list[str], env: dict[str, str], gpu: int | list[int]) -> None:
    proc_env = env.copy()
    if isinstance(gpu, list):
        proc_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu)
    else:
        proc_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logger.info("[%s] GPU %s | %s", label, gpu, shlex.join(cmd))
    subprocess.run(cmd, env=proc_env, check=True, cwd=project_root)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    env = _load_env_file(args.env_file)

    if "OPENAI_API_KEY" not in env:
        raise RuntimeError("OPENAI_API_KEY missing from env file")
    if "TINKER_API_KEY" not in env:
        raise RuntimeError("TINKER_API_KEY missing from env file")
    if not Path(args.checkpoint_json).exists():
        raise FileNotFoundError(f"Missing checkpoint JSON: {args.checkpoint_json}")

    raw_dataset = MAGEDataset.download(split="test", cache_dir=args.cache_dir)
    raw_counts = {
        "total": len(raw_dataset),
        "human": len(raw_dataset.human_samples),
        "ai": len(raw_dataset.ai_samples),
    }
    filtered_dataset = load_eval_dataset(
        "mage",
        n_human=999999,
        n_ai=999999,
        cache_dir=args.cache_dir,
        seed=42,
    )
    filtered_counts = {
        "total": len(filtered_dataset),
        "human": len(filtered_dataset.human_samples),
        "ai": len(filtered_dataset.ai_samples),
    }
    logger.info("MAGE raw test counts: %s", json.dumps(raw_counts, sort_keys=True))
    logger.info("MAGE filtered eval counts: %s", json.dumps(filtered_counts, sort_keys=True))

    if raw_counts != {"total": 60743, "human": 30265, "ai": 30478}:
        raise RuntimeError(f"Unexpected raw MAGE counts: {raw_counts}")
    if filtered_counts != {"total": 29966, "human": 15310, "ai": 14656}:
        raise RuntimeError(f"Unexpected filtered MAGE counts: {filtered_counts}")

    detector_gpu = args.gpus[0]
    method_gpus = args.gpus

    for detector in ("roberta", "fast_detectgpt", "ghostbuster", "binoculars", "mage"):
        _run(
            f"detector:{detector}",
            [
                str(project_root / "venv" / "bin" / "python"),
                "-c",
                (
                    "from eval.detectors import get_detector; "
                    f"d=get_detector('{detector}'); "
                    "d.load(); "
                    f"score=d.get_scores({TEST_TEXT!r}); "
                    "print({'score': score})"
                ),
            ],
            env=env,
            gpu=detector_gpu,
        )

    method_cmds = [
        ("m1", {"max_model_len": 2048, "gpu_memory_utilization": 0.82}),
        ("m2", {"checkpoint_json": args.checkpoint_json}),
        (
            "m3",
            {
                "device": "cuda:1" if len(method_gpus) >= 2 else "cpu",
                "max_model_len": 2048,
                "gpu_memory_utilization": 0.82,
            },
        ),
        ("m4", {"max_model_len": 2048, "gpu_memory_utilization": 0.82}),
        ("m5", {}),
    ]
    for index, (method, extra) in enumerate(method_cmds):
        kwargs_expr = ", ".join(f"{key}={value!r}" for key, value in extra.items())
        constructor = f"m=get_method('{method}'" + (f", {kwargs_expr}" if kwargs_expr else "") + ")"
        if method == "m3" and len(method_gpus) >= 2:
            method_gpu: int | list[int] = [method_gpus[index % len(method_gpus)], method_gpus[(index + 1) % len(method_gpus)]]
            n_candidates = 4
        else:
            method_gpu = method_gpus[index % len(method_gpus)]
            n_candidates = 1
        _run(
            f"method:{method}",
            [
                str(project_root / "venv" / "bin" / "python"),
                "-c",
                (
                    "from eval.methods import get_method; "
                    f"{constructor}; "
                    "m.load(); "
                    f"out=m.attack({TEST_TEXT!r}, n_candidates={n_candidates}); "
                    "assert out.valid, out.fail_reason; "
                    "print({'words': len(out.text.split())})"
                ),
            ],
            env=env,
            gpu=method_gpu,
        )

    logger.info("Preflight completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
