"""
GPT-based text quality evaluation (Adversarial Paraphrasing-style).

Supports optional gpt-4o-mini judging with caching and per-method caps.
"""

import asyncio
import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from .plots import (
    QUALITY_RATING_PROMPT,
    get_quality_rating_messages,
    parse_quality_rating_response,
)

logger = logging.getLogger(__name__)


SIMILARITY_RATING_PROMPT = """
You are an expert linguist and paraphrase evaluator. Your task is to assess how well the paraphrased text preserves the meaning of the original source text.

Be generous when meaning is substantially preserved. Use the full 1-5 scale and avoid defaulting to 3 unless it truly fits. If the paraphrase keeps the core meaning with only minor changes, prefer 4 or 5. Use 2 or 1 only for clear meaning loss or topic drift.

Scoring criteria:
5 - Approximately equivalent: Meaning preserved; only wording/structure changes.
4 - Nearly equivalent: Meaning mostly preserved; small factual or emphasis shifts.
3 - Somewhat equivalent: Partial meaning preserved; important details differ.
2 - Topically related: Same topic but most meaning lost.
1 - Not related: Different topic or meaning.

Provide your final output as a JSON object in this format:
{
"score": <score from 1 to 5>,
"justification": "<brief explanation>"
}
"""


def get_similarity_rating_messages(original_text: str, paraphrased_text: str) -> List[Dict]:
    """
    Generate GPT messages for semantic similarity rating.
    """
    user_prompt = (
        "Evaluate the following paraphrase using the criteria above:\\n\\n"
        f"Original Text:\\n\"\"\"{original_text}\"\"\"\\n\\n"
        f"Paraphrased Text:\\n\"\"\"{paraphrased_text}\"\"\"\\n\\n"
        "What score (1 to 5) would you assign for semantic similarity, and why?"
    )

    return [
        {"role": "system", "content": SIMILARITY_RATING_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


@dataclass
class GPTQualityConfig:
    model: str = "gpt-5-nano"
    max_per_method: int = 200
    temperature: Optional[float] = 0.0
    max_output_tokens: int = 256
    seed: int = 42
    cache_path: Optional[Path] = None
    concurrency: int = 32
    request_timeout_s: int = 60
    max_retries: int = 6
    reasoning_effort: Optional[str] = None
    text_verbosity: Optional[str] = None


def _hash_key(model: str, prompt_type: str, original: str, paraphrased: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(prompt_type.encode("utf-8"))
    h.update(original.encode("utf-8"))
    h.update(paraphrased.encode("utf-8"))
    return h.hexdigest()


def _load_cache(path: Optional[Path]) -> Dict[str, Dict]:
    if not path or not path.exists():
        return {}

    cache: Dict[str, Dict] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = rec.get("cache_key")
                if key:
                    cache[key] = rec
            except Exception:
                continue
    return cache


def _append_cache(path: Optional[Path], record: Dict) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _extract_response_text(data: Dict) -> str:
    if isinstance(data, dict):
        if "output_text" in data and isinstance(data["output_text"], str):
            return data["output_text"]
        if "output" in data:
            for item in data.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        return content.get("text", "")
        if "choices" in data:
            try:
                return data["choices"][0]["message"]["content"]
            except Exception:
                pass
    return ""


def _response_to_dict(response) -> Dict:
    if isinstance(response, dict):
        return response
    for attr in ("model_dump", "dict"):
        if hasattr(response, attr):
            try:
                return getattr(response, attr)()
            except Exception:
                continue
    return {}


async def _call_openai_async(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict],
    temperature: Optional[float],
    max_output_tokens: int,
    reasoning_effort: Optional[str],
    text_verbosity: Optional[str],
) -> str:
    payload = {
        "model": model,
        "input": messages,
        "max_output_tokens": max_output_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    if text_verbosity:
        payload["text"] = {"verbosity": text_verbosity}

    response = await client.responses.create(**payload)
    if hasattr(response, "output_text") and isinstance(response.output_text, str):
        return response.output_text
    return _extract_response_text(_response_to_dict(response))


async def _judge_single_async(
    client: AsyncOpenAI,
    config: GPTQualityConfig,
    original_text: str,
    paraphrased_text: str,
    prompt_type: str,
) -> Tuple[int, str]:
    if prompt_type == "quality":
        messages = get_quality_rating_messages(original_text, paraphrased_text)
    else:
        messages = get_similarity_rating_messages(original_text, paraphrased_text)

    last_err: Optional[Exception] = None
    temperature = config.temperature
    reasoning_effort = config.reasoning_effort
    text_verbosity = config.text_verbosity
    if "gpt-5" in config.model:
        if temperature is not None:
            temperature = None
        if not reasoning_effort:
            reasoning_effort = "minimal"
        if not text_verbosity:
            text_verbosity = "low"
    for attempt in range(config.max_retries + 1):
        try:
            response_text = await _call_openai_async(
                client=client,
                model=config.model,
                messages=messages,
                temperature=temperature,
                max_output_tokens=config.max_output_tokens,
                reasoning_effort=reasoning_effort,
                text_verbosity=text_verbosity,
            )
            score, justification = parse_quality_rating_response(response_text)
            return score, justification
        except Exception as e:
            msg = str(e).lower()
            if "temperature" in msg and "unsupported" in msg and temperature is not None:
                temperature = None
                continue
            if "rate limit" in msg or "rate_limit" in msg or "429" in msg:
                wait_s = 1.0 * (2 ** attempt) + random.uniform(0.0, 0.5)
                if "try again in" in msg:
                    try:
                        tail = msg.split("try again in", 1)[1]
                        num = "".join(ch for ch in tail if ch.isdigit() or ch == ".")
                        if num:
                            wait_s = max(wait_s, float(num) / 1000.0)
                    except Exception:
                        pass
                await asyncio.sleep(wait_s)
                last_err = e
                continue
            last_err = e
            if attempt < config.max_retries:
                await asyncio.sleep(0.5 * (2 ** attempt))
            else:
                raise last_err
    raise last_err


async def run_gpt_quality_judge_async(
    api_key: str,
    items: List[Dict],
    config: GPTQualityConfig,
) -> List[Dict]:
    """
    Run GPT quality evaluation on a list of items.

    Each item must have: sample_id, dataset, method, setting, original, paraphrased.
    """
    if not api_key:
        raise ValueError("OpenAI API key is required for GPT quality evaluation")

    cache = _load_cache(config.cache_path)
    rng = random.Random(config.seed)

    # Sample up to max_per_method per method
    by_method: Dict[str, List[Dict]] = {}
    for item in items:
        by_method.setdefault(item["method"], []).append(item)

    selected: List[Dict] = []
    for method, group in by_method.items():
        if len(group) <= config.max_per_method:
            selected.extend(group)
        else:
            selected.extend(rng.sample(group, config.max_per_method))

    results: List[Dict] = []
    logger.info(f"[GPT-QUALITY] Evaluating {len(selected)} samples (cap={config.max_per_method} per method)")

    sem = asyncio.Semaphore(max(1, config.concurrency))
    cache_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    progress = {"done": 0, "total": 0}

    async with AsyncOpenAI(api_key=api_key, timeout=config.request_timeout_s) as client:
        tasks = []

        async def run_task(record: Dict, cache_key: str, prompt_type: str, original: str, paraphrased: str):
            async with sem:
                try:
                    score, justification = await _judge_single_async(
                        client=client,
                        config=config,
                        original_text=original,
                        paraphrased_text=paraphrased,
                        prompt_type=prompt_type,
                    )
                except Exception as e:
                    logger.warning(f"[GPT-QUALITY] Failed {prompt_type} for sample {record.get('sample_id')}: {e}")
                    score, justification = None, None

            record[f"{prompt_type}_rating"] = score
            record[f"{prompt_type}_justification"] = justification

            if config.cache_path:
                cache_record = {
                    "cache_key": cache_key,
                    "model": config.model,
                    "prompt_type": prompt_type,
                    "original": original,
                    "paraphrased": paraphrased,
                    f"{prompt_type}_rating": score,
                    f"{prompt_type}_justification": justification,
                }
                async with cache_lock:
                    _append_cache(config.cache_path, cache_record)
                    cache[cache_key] = cache_record

            async with progress_lock:
                progress["done"] += 1
                if progress["done"] % 25 == 0:
                    logger.info(f"[GPT-QUALITY] {progress['done']}/{progress['total']} completed")

        for item in selected:
            original = item["original"]
            paraphrased = item["paraphrased"]

            record = {
                "sample_id": item["sample_id"],
                "dataset": item["dataset"],
                "method": item["method"],
                "setting": item["setting"],
                "quality_model": config.model,
            }

            for prompt_type in ("quality", "similarity"):
                cache_key = _hash_key(config.model, prompt_type, original, paraphrased)
                cached = cache.get(cache_key)
                cached_score = cached.get(f"{prompt_type}_rating") if cached else None
                if cached and cached_score is not None:
                    record[f"{prompt_type}_rating"] = cached.get(f"{prompt_type}_rating")
                    record[f"{prompt_type}_justification"] = cached.get(f"{prompt_type}_justification")
                else:
                    progress["total"] += 1
                    tasks.append(asyncio.create_task(
                        run_task(record, cache_key, prompt_type, original, paraphrased)
                    ))

            results.append(record)

        if tasks:
            await asyncio.gather(*tasks)

    return results


def run_gpt_quality_judge(
    api_key: str,
    items: List[Dict],
    config: GPTQualityConfig,
) -> List[Dict]:
    """
    Synchronous wrapper for async GPT quality evaluation.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_gpt_quality_judge_async(api_key=api_key, items=items, config=config))
    raise RuntimeError("run_gpt_quality_judge cannot be called from a running event loop. Use run_gpt_quality_judge_async.")
