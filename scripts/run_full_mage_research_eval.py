#!/usr/bin/env python3
"""
Orchestrate the staged full-MAGE research evaluation run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.data import MAGEDataset, load_eval_dataset


FULL_MAGE_COUNTS = {"human": 15310, "ai": 14656}
DEFAULT_METHODS = ["m0", "m1", "m2", "m3", "m4", "m5"]
DEFAULT_DETECTORS = ["roberta", "fast_detectgpt", "binoculars", "ghostbuster", "mage"]

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full staged MAGE evaluation")
    parser.add_argument(
        "--run-root",
        default=f"outputs/eval_runs/full_mage_research_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--env-file", default="/data/suraj/.config/stealthrl/eval.env")
    parser.add_argument("--checkpoint-json", default="/data/suraj/.config/stealthrl/m2_checkpoint.json")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--detectors", nargs="+", default=DEFAULT_DETECTORS)
    parser.add_argument("--n-candidates", type=int, default=1)
    parser.add_argument("--gpt-model", default="gpt-5-nano")
    parser.add_argument("--gpt-max-per-method", type=int, default=500)
    parser.add_argument("--gpt-concurrency", type=int, default=32)
    parser.add_argument("--gpus", nargs="+", type=int, default=None)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--vllm-max-model-len", type=int, default=2048)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--poll-seconds", type=int, default=30)
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


def _discover_gpus(min_free_gb: float) -> list[int]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True, cwd=project_root)
    available = []
    for line in output.strip().splitlines():
        index_s, total_s, used_s, util_s = [part.strip() for part in line.split(",")]
        total_gb = float(total_s) / 1024.0
        used_gb = float(used_s) / 1024.0
        util = float(util_s)
        free_gb = total_gb - used_gb
        if free_gb >= min_free_gb and util <= 10:
            available.append(int(index_s))
    return available


def _run_checked(cmd: list[str], env: dict[str, str], cwd: Path, log_path: Path | None = None) -> None:
    logger.info("%s", shlex.join(cmd))
    if log_path is None:
        subprocess.run(cmd, env=env, cwd=cwd, check=True)
        return
    with log_path.open("w") as log_file:
        subprocess.run(
            cmd,
            env=env,
            cwd=cwd,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )


def _start_job(
    name: str,
    cmd: list[str],
    env: dict[str, str],
    cwd: Path,
    gpu: int | list[int] | None,
    log_dir: Path,
) -> tuple[subprocess.Popen, Path]:
    proc_env = env.copy()
    if gpu is None:
        proc_env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        if isinstance(gpu, list):
            proc_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu)
        else:
            proc_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path = log_dir / f"{name}.log"
    cmd_path = log_dir / f"{name}.cmd.sh"
    cmd_path.write_text(shlex.join(cmd) + "\n")
    log_file = log_path.open("w")
    process = subprocess.Popen(
        cmd,
        env=proc_env,
        cwd=cwd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    if gpu is None:
        logger.info("Started %s on CPU (pid=%s)", name, process.pid)
    elif isinstance(gpu, list):
        logger.info("Started %s on GPUs %s (pid=%s)", name, gpu, process.pid)
    else:
        logger.info("Started %s on GPU %s (pid=%s)", name, gpu, process.pid)
    return process, log_path


def _create_anchor(samples_dir: Path, cache_dir: str) -> None:
    samples_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_eval_dataset(
        "mage",
        n_human=999999,
        n_ai=999999,
        cache_dir=cache_dir,
        seed=42,
    )
    counts = {"human": len(dataset.human_samples), "ai": len(dataset.ai_samples)}
    if counts != FULL_MAGE_COUNTS:
        raise RuntimeError(f"Unexpected MAGE counts: {counts}")
    payload = {
        "mage": {
            "human_ids": [sample.id for sample in dataset.human_samples],
            "ai_ids": [sample.id for sample in dataset.ai_samples],
        }
    }
    with open(samples_dir / "dataset_samples.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(samples_dir / "dataset_counts.json", "w") as f:
        json.dump({"mage": counts}, f, indent=2)


def _combine_raw_outputs(method_dirs: dict[str, Path], combined_path: Path) -> None:
    payload: dict[str, dict[str, list[str]]] = {}
    for method, method_dir in method_dirs.items():
        data = json.loads((method_dir / "raw_outputs.json").read_text())
        for dataset_name, dataset_outputs in data.items():
            payload.setdefault(dataset_name, {})
            payload[dataset_name][method] = dataset_outputs[method]
    with open(combined_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _method_n_candidates(method: str, default_n_candidates: int) -> int:
    if method in {"m3", "adversarial_paraphrasing", "m3_roberta", "m3_fastdetect", "m3_ensemble"}:
        return 4
    return default_n_candidates


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

    gpus = args.gpus or _discover_gpus(args.min_free_gb)
    if not gpus:
        raise RuntimeError("No sufficiently free GPUs found")

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir = run_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    samples_dir = run_root / "samples"
    method_root = run_root / "method_runs"
    scores_dir = run_root / "detector_scores"
    method_root.mkdir(exist_ok=True)
    scores_dir.mkdir(exist_ok=True)

    _create_anchor(samples_dir, args.cache_dir)
    with open(run_root / "manifest.json", "w") as f:
        json.dump(
            {
                "run_root": str(run_root),
                "methods": args.methods,
                "detectors": args.detectors,
                "n_candidates": args.n_candidates,
                "gpus": gpus,
                "counts": FULL_MAGE_COUNTS,
                "env_file": args.env_file,
                "checkpoint_json": args.checkpoint_json,
            },
            f,
            indent=2,
        )

    preflight_cmd = [
        str(project_root / "venv" / "bin" / "python"),
        str(project_root / "scripts" / "preflight_research_eval.py"),
        "--env-file",
        args.env_file,
        "--checkpoint-json",
        args.checkpoint_json,
        "--cache-dir",
        args.cache_dir,
        "--gpus",
        *[str(gpu) for gpu in gpus[: max(1, min(len(gpus), 4))]],
    ]
    _run_checked(preflight_cmd, env=env, cwd=project_root, log_path=logs_dir / "preflight.log")

    method_dirs = {method: method_root / method for method in args.methods}
    pending = deque()
    cpu_pending = deque()
    for method in args.methods:
        raw_outputs_path = method_dirs[method] / "raw_outputs.json"
        if raw_outputs_path.exists():
            logger.info("Skipping %s; existing outputs found at %s", method, raw_outputs_path)
            continue

        method_n_candidates = _method_n_candidates(method, args.n_candidates)
        method_gpu_count = 1
        cmd = [
            str(project_root / "venv" / "bin" / "python"),
            str(project_root / "scripts" / "generate_method_outputs.py"),
            "--method",
            method,
            "--samples-dir",
            str(samples_dir),
            "--out-dir",
            str(method_dirs[method]),
            "--cache-dir",
            args.cache_dir,
            "--n-candidates",
            str(method_n_candidates),
            "--m1-backend",
            "vllm",
            "--vllm-max-model-len",
            str(args.vllm_max_model_len),
            "--vllm-gpu-memory-utilization",
            str(args.vllm_gpu_memory_utilization),
            "--max-invalid-retries",
            "3",
        ]
        if method == "m2":
            cmd.extend(["--checkpoint-json", args.checkpoint_json])
        if method == "m3":
            if len(gpus) >= 2:
                method_gpu_count = 2
                cmd.extend(["--method-device", "cuda:1"])
            else:
                cmd.extend(["--method-device", "cpu"])
        if method in {"m0", "m5"}:
            cpu_pending.append((method, cmd))
        else:
            pending.append((method, cmd, method_gpu_count))

    running: dict[str, tuple[subprocess.Popen, int | list[int] | None, Path]] = {}
    while cpu_pending:
        method, cmd = cpu_pending.popleft()
        process, log_path = _start_job(method, cmd, env, project_root, None, logs_dir)
        running[method] = (process, None, log_path)

    free_gpus = deque(gpus)
    while pending or running:
        while pending:
            method, cmd, gpu_count = pending[0]
            if len(free_gpus) < gpu_count:
                break
            pending.popleft()
            if gpu_count == 1:
                gpu: int | list[int] | None = free_gpus.popleft()
            else:
                gpu = [free_gpus.popleft() for _ in range(gpu_count)]
            process, log_path = _start_job(method, cmd, env, project_root, gpu, logs_dir)
            running[method] = (process, gpu, log_path)

        time.sleep(args.poll_seconds)
        completed = []
        for method, (process, gpu, log_path) in running.items():
            code = process.poll()
            if code is None:
                continue
            if code != 0:
                location = f"GPU {gpu}" if gpu is not None else "CPU"
                raise RuntimeError(f"Method job {method} failed on {location}. See {log_path}")
            if gpu is None:
                logger.info("Completed method %s on CPU", method)
            elif isinstance(gpu, list):
                logger.info("Completed method %s on GPUs %s", method, gpu)
                for gpu_id in gpu:
                    free_gpus.append(gpu_id)
            else:
                logger.info("Completed method %s on GPU %s", method, gpu)
                free_gpus.append(gpu)
            completed.append(method)
        for method in completed:
            del running[method]

    combined_raw_outputs = run_root / "raw_outputs.json"
    _combine_raw_outputs(method_dirs, combined_raw_outputs)

    pending = deque()
    detector_batch_sizes = {
        "roberta": 32,
        "fast_detectgpt": 4,
        "binoculars": 8,
        "ghostbuster": 32,
        "mage": 8,
    }
    for detector in args.detectors:
        cmd = [
            str(project_root / "venv" / "bin" / "python"),
            str(project_root / "scripts" / "score_detector_outputs.py"),
            "--detector",
            detector,
            "--samples-dir",
            str(samples_dir),
            "--raw-outputs",
            str(combined_raw_outputs),
            "--out-dir",
            str(scores_dir),
            "--cache-dir",
            args.cache_dir,
            "--device",
            "cuda",
        ]
        batch_size = detector_batch_sizes.get(detector)
        if batch_size is not None:
            cmd.extend(["--batch-size", str(batch_size)])
        pending.append((f"score_{detector}", cmd))

    running = {}
    free_gpus = deque(gpus)
    while pending or running:
        while pending and free_gpus:
            name, cmd = pending.popleft()
            gpu = free_gpus.popleft()
            process, log_path = _start_job(name, cmd, env, project_root, gpu, logs_dir)
            running[name] = (process, gpu, log_path)

        time.sleep(args.poll_seconds)
        completed = []
        for name, (process, gpu, log_path) in running.items():
            code = process.poll()
            if code is None:
                continue
            if code != 0:
                raise RuntimeError(f"Scoring job {name} failed on GPU {gpu}. See {log_path}")
            logger.info("Completed scoring job %s on GPU %s", name, gpu)
            free_gpus.append(gpu)
            completed.append(name)
        for name in completed:
            del running[name]

    assembled_dir = run_root / "assembled"
    finalize_cmd = [
        str(project_root / "venv" / "bin" / "python"),
        str(project_root / "scripts" / "finalize_eval_run.py"),
        "--samples-dir",
        str(samples_dir),
        "--raw-outputs",
        str(combined_raw_outputs),
        "--scores-dir",
        str(scores_dir),
        "--out-dir",
        str(assembled_dir),
        "--cache-dir",
        args.cache_dir,
        "--device",
        "cuda",
        "--n-bootstrap",
        "500",
    ]
    finalize_env = env.copy()
    finalize_env["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
    _run_checked(finalize_cmd, env=finalize_env, cwd=project_root, log_path=logs_dir / "finalize.log")

    gpt_cmd = [
        str(project_root / "venv" / "bin" / "python"),
        str(project_root / "scripts" / "run_gpt_quality_only.py"),
        "--run-dir",
        str(assembled_dir),
        "--methods",
        "m1",
        "m2",
        "m3",
        "m4",
        "m5",
        "--model",
        args.gpt_model,
        "--max-per-method",
        str(args.gpt_max_per_method),
        "--concurrency",
        str(args.gpt_concurrency),
    ]
    _run_checked(gpt_cmd, env=env, cwd=project_root, log_path=logs_dir / "gpt_quality.log")

    logger.info("Full staged MAGE evaluation completed: %s", assembled_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
