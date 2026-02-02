"""
GPT-based text quality evaluation (Adversarial Paraphrasing-style).

Supports optional GPT-5-mini judging with caching and per-method caps.
"""

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

from .plots import (
    QUALITY_RATING_PROMPT,
    get_quality_rating_messages,
    parse_quality_rating_response,
)

logger = logging.getLogger(__name__)


SIMILARITY_RATING_PROMPT = """
You are an expert linguist and paraphrase evaluator. Your task is to assess how well the paraphrased text preserves the meaning of the original source text. Use the following scoring criteria:

5 - Approximately equivalent: Meaning is preserved; differences are only in wording or structure.
4 - Nearly equivalent: Meaning is mostly preserved; minor factual details differ.
3 - Somewhat equivalent: Some meaning is preserved; important details or meanings differ.
2 - Topically related: The texts are on the same topic but most meaning is lost.
1 - Not topically related: The texts are not related in topic or meaning.

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
    model: str = "gpt-5-mini"
    max_per_method: int = 200
    temperature: float = 0.0
    max_output_tokens: int = 256
    seed: int = 42
    cache_path: Optional[Path] = None


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


def _call_openai(
    api_key: str,
    model: str,
    messages: List[Dict],
    temperature: float,
    max_output_tokens: int,
    timeout: int = 60,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": messages,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }

    with httpx.Client(timeout=timeout) as client:
        resp = client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return _extract_response_text(data)


def _judge_single(
    api_key: str,
    config: GPTQualityConfig,
    original_text: str,
    paraphrased_text: str,
    prompt_type: str,
) -> Tuple[int, str]:
    if prompt_type == "quality":
        messages = get_quality_rating_messages(original_text, paraphrased_text)
    else:
        messages = get_similarity_rating_messages(original_text, paraphrased_text)

    response_text = _call_openai(
        api_key=api_key,
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
    )

    score, justification = parse_quality_rating_response(response_text)
    return score, justification


def run_gpt_quality_judge(
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

    for idx, item in enumerate(selected, 1):
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
            if cached:
                score = cached.get(f"{prompt_type}_rating")
                justification = cached.get(f"{prompt_type}_justification")
            else:
                try:
                    score, justification = _judge_single(
                        api_key=api_key,
                        config=config,
                        original_text=original,
                        paraphrased_text=paraphrased,
                        prompt_type=prompt_type,
                    )
                except Exception as e:
                    logger.warning(f"[GPT-QUALITY] Failed {prompt_type} for sample {item['sample_id']}: {e}")
                    score, justification = None, None

                cache_record = {
                    "cache_key": cache_key,
                    "model": config.model,
                    "prompt_type": prompt_type,
                    "original": original,
                    "paraphrased": paraphrased,
                    f"{prompt_type}_rating": score,
                    f"{prompt_type}_justification": justification,
                }
                _append_cache(config.cache_path, cache_record)
                cache[cache_key] = cache_record

            record[f"{prompt_type}_rating"] = score
            record[f"{prompt_type}_justification"] = justification

        results.append(record)

        if idx % 25 == 0:
            logger.info(f"[GPT-QUALITY] {idx}/{len(selected)} completed")
        time.sleep(0.05)

    return results
