"""
Concurrent Tinker sampling utilities for fast StealthRL evaluation.

This module batches many SamplingClient requests concurrently using asyncio
and supports chunking + resumable caches for large evaluations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SamplingResult:
    index: int
    candidates: List[str]
    error: Optional[str]
    attempts: int
    latency_s: float
    text_hash: str
    sampling_params_hash: str


def _hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _hash_params(params: Dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return sha256(payload.encode("utf-8")).hexdigest()[:12]


def _load_cache(path: Path) -> Dict[int, SamplingResult]:
    if not path.exists():
        return {}
    cached: Dict[int, SamplingResult] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    idx = int(obj["index"])
                    cached[idx] = SamplingResult(
                        index=idx,
                        candidates=obj.get("candidates", []) or [],
                        error=obj.get("error"),
                        attempts=int(obj.get("attempts", 0)),
                        latency_s=float(obj.get("latency_s", 0.0)),
                        text_hash=obj.get("text_hash", ""),
                        sampling_params_hash=obj.get("sampling_params_hash", ""),
                    )
                except Exception:
                    continue
    except Exception as e:
        logger.warning(f"Failed to load resume cache {path}: {e}")
    return cached


def _append_cache(path: Path, results: Iterable[SamplingResult]) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "index": r.index,
                "candidates": r.candidates,
                "error": r.error,
                "attempts": r.attempts,
                "latency_s": r.latency_s,
                "text_hash": r.text_hash,
                "sampling_params_hash": r.sampling_params_hash,
            }, ensure_ascii=False) + "\n")


async def _sample_one(
    index: int,
    text: str,
    build_model_input: Callable[[str], Any],
    sampling_client: Any,
    sampling_params: Any,
    num_samples: int,
    tokenizer: Any,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    backoff_base_s: float,
    sampling_params_hash: str,
) -> SamplingResult:
    text_hash = _hash_text(text)
    attempts = 0
    last_error: Optional[str] = None
    async with semaphore:
        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            try:
                model_input = build_model_input(text)
                start = time.time()
                result = await sampling_client.sample_async(
                    prompt=model_input,
                    sampling_params=sampling_params,
                    num_samples=num_samples,
                )
                latency = time.time() - start
                candidates: List[str] = []
                for sample in result.sequences:
                    output_text = tokenizer.decode(sample.tokens, skip_special_tokens=True).strip()
                    if output_text:
                        candidates.append(output_text)
                return SamplingResult(
                    index=index,
                    candidates=candidates,
                    error=None,
                    attempts=attempts,
                    latency_s=latency,
                    text_hash=text_hash,
                    sampling_params_hash=sampling_params_hash,
                )
            except Exception as e:
                last_error = str(e)
                sleep_s = backoff_base_s * (2 ** attempt) + random.uniform(0, backoff_base_s)
                await asyncio.sleep(sleep_s)
        return SamplingResult(
            index=index,
            candidates=[],
            error=last_error or "unknown_error",
            attempts=attempts,
            latency_s=0.0,
            text_hash=text_hash,
            sampling_params_hash=sampling_params_hash,
        )


async def _run_concurrent(
    texts: Sequence[str],
    build_model_input: Callable[[str], Any],
    sampling_client: Any,
    sampling_params: Any,
    num_samples: int,
    tokenizer: Any,
    concurrency: int,
    chunk_size: int,
    max_retries: int,
    backoff_base_s: float,
    resume_cache_path: Optional[Path],
    sampling_params_hash: str,
) -> List[SamplingResult]:
    semaphore = asyncio.Semaphore(concurrency)
    results: List[SamplingResult] = [None] * len(texts)  # type: ignore[assignment]

    cached = _load_cache(resume_cache_path) if resume_cache_path else {}
    cached_hits = 0
    for idx, text in enumerate(texts):
        if idx in cached:
            cached_result = cached[idx]
            if cached_result.text_hash == _hash_text(text) and cached_result.sampling_params_hash == sampling_params_hash:
                results[idx] = cached_result
                cached_hits += 1
    if cached_hits:
        logger.info(f"[TINKER] Resuming from cache: {cached_hits}/{len(texts)} samples")

    pending_indices = [i for i, r in enumerate(results) if r is None]
    if not pending_indices:
        return results  # type: ignore[return-value]

    for start in range(0, len(pending_indices), chunk_size):
        chunk_indices = pending_indices[start:start + chunk_size]
        tasks = [
            asyncio.create_task(
                _sample_one(
                    index=idx,
                    text=texts[idx],
                    build_model_input=build_model_input,
                    sampling_client=sampling_client,
                    sampling_params=sampling_params,
                    num_samples=num_samples,
                    tokenizer=tokenizer,
                    semaphore=semaphore,
                    max_retries=max_retries,
                    backoff_base_s=backoff_base_s,
                    sampling_params_hash=sampling_params_hash,
                )
            )
            for idx in chunk_indices
        ]
        chunk_results = await asyncio.gather(*tasks)
        for r in chunk_results:
            results[r.index] = r
        if resume_cache_path:
            _append_cache(resume_cache_path, chunk_results)
        logger.info(f"[TINKER] Completed chunk {start // chunk_size + 1} ({len(chunk_indices)} samples)")

    return results  # type: ignore[return-value]


def run_sampling_concurrent(
    texts: Sequence[str],
    build_model_input: Callable[[str], Any],
    sampling_client: Any,
    sampling_params: Any,
    num_samples: int,
    tokenizer: Any,
    concurrency: int = 64,
    chunk_size: int = 256,
    max_retries: int = 2,
    backoff_base_s: float = 0.5,
    resume_cache_path: Optional[str] = None,
    sampling_params_dict: Optional[Dict[str, Any]] = None,
) -> List[SamplingResult]:
    """
    Run concurrent sampling for a list of texts.

    Returns a list of SamplingResult in the original input order.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    sampling_params_hash = _hash_params(sampling_params_dict or {})
    cache_path = Path(resume_cache_path) if resume_cache_path else None

    coro = _run_concurrent(
        texts=texts,
        build_model_input=build_model_input,
        sampling_client=sampling_client,
        sampling_params=sampling_params,
        num_samples=num_samples,
        tokenizer=tokenizer,
        concurrency=concurrency,
        chunk_size=chunk_size,
        max_retries=max_retries,
        backoff_base_s=backoff_base_s,
        resume_cache_path=cache_path,
        sampling_params_hash=sampling_params_hash,
    )

    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "asyncio.run()" in str(e):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        raise
