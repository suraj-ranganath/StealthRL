"""Inference backends for the StealthRL demo."""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
import uuid
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .config import DemoSettings


@dataclass(frozen=True)
class ParaphraseResult:
    request_id: str
    input_text: str
    output_text: str
    backend: str
    latency_ms: int
    metrics: dict[str, Any]
    metadata: dict[str, Any]


class BaseDemoBackend:
    name = "base"

    async def paraphrase(self, text: str, temperature: float, top_p: float) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError


class MockStealthBackend(BaseDemoBackend):
    """Deterministic, no-cost paraphraser for local and public UI testing."""

    name = "mock"

    _phrase_swaps = (
        (r"\bAI-generated text\b", "machine-written prose"),
        (r"\bAI text\b", "machine-written text"),
        (r"\bdetectors\b", "detection systems"),
        (r"\brobustness\b", "resilience"),
        (r"\bevaluate\b", "assess"),
        (r"\bevaluation\b", "assessment"),
        (r"\bhowever\b", "still"),
        (r"\btherefore\b", "as a result"),
        (r"\bimportant\b", "central"),
        (r"\bsignificant\b", "substantial"),
    )

    async def paraphrase(self, text: str, temperature: float, top_p: float) -> tuple[str, dict[str, Any]]:
        await asyncio.sleep(0)
        sentences = _split_sentences(text)
        rewritten: list[str] = []
        for idx, sentence in enumerate(sentences):
            clean = sentence.strip()
            if not clean:
                continue
            clean = self._rewrite_sentence(clean, idx)
            rewritten.append(clean)
        output = " ".join(rewritten).strip()
        if not output:
            output = text.strip()
        return output, {
            "mode": "deterministic_preview",
            "note": "Set STEALTHRL_DEMO_INFERENCE_BACKEND=tinker for real StealthRL sampler calls.",
        }

    def _rewrite_sentence(self, sentence: str, idx: int) -> str:
        out = sentence
        for pattern, repl in self._phrase_swaps:
            out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
        if idx % 3 == 1 and "," in out:
            head, tail = out.split(",", 1)
            if 4 <= len(head.split()) <= 18:
                out = f"{tail.strip().rstrip('.')}, {head.strip().lower()}."
        elif idx % 3 == 2 and len(out.split()) > 14:
            out = re.sub(r"\bThis\b", "The result", out, count=1)
            out = re.sub(r"\bIt\b", "This pattern", out, count=1)
        return _normalize_spacing(out)


class TinkerStealthBackend(BaseDemoBackend):
    name = "tinker"

    def __init__(self, settings: DemoSettings) -> None:
        if not settings.checkpoint_json:
            raise RuntimeError("STEALTHRL_DEMO_CHECKPOINT_JSON is required for tinker backend")
        self.settings = settings
        self._runtime: dict[str, Any] | None = None
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        async with self._lock:
            if self._runtime is not None:
                return self._runtime

            def _load() -> dict[str, Any]:
                from tinker import ServiceClient

                checkpoint = json.loads(Path(self.settings.checkpoint_json).read_text())
                sampler_path = checkpoint["checkpoints"]["sampler_weights"]
                service_client = ServiceClient()
                sampling_client = service_client.create_sampling_client(model_path=sampler_path)
                tokenizer = sampling_client.get_tokenizer()
                return {
                    "checkpoint": checkpoint,
                    "sampler_path": sampler_path,
                    "sampling_client": sampling_client,
                    "tokenizer": tokenizer,
                }

            self._runtime = await asyncio.to_thread(_load)
            return self._runtime

    async def paraphrase(self, text: str, temperature: float, top_p: float) -> tuple[str, dict[str, Any]]:
        runtime = await self._ensure_loaded()

        def _run() -> str:
            from tinker import types

            tokenizer = runtime["tokenizer"]
            prompt_text = PARAPHRASE_PROMPT.format(text=text)
            messages = [{"role": "user", "content": prompt_text}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer.encode(formatted)
            model_input = types.ModelInput.from_ints(input_ids)
            params = types.SamplingParams(
                max_tokens=_estimate_generation_max_tokens(text, default_max_tokens=512),
                temperature=temperature,
                top_p=top_p,
            )
            future = runtime["sampling_client"].sample(
                prompt=model_input,
                sampling_params=params,
                num_samples=1,
            )
            result = future.result()
            if not result.sequences:
                raise RuntimeError("Tinker sampler returned no sequences")
            return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True).strip()

        output = await asyncio.to_thread(_run)
        return output, {
            "method": "stealthrl",
            "backend": "tinker",
            "sampler_path": runtime["sampler_path"],
        }


def build_backend(settings: DemoSettings) -> BaseDemoBackend:
    if settings.inference_backend == "tinker":
        return TinkerStealthBackend(settings)
    if settings.inference_backend != "mock":
        raise ValueError("STEALTHRL_DEMO_INFERENCE_BACKEND must be 'mock' or 'tinker'")
    return MockStealthBackend()


async def run_paraphrase(
    backend: BaseDemoBackend,
    text: str,
    temperature: float,
    top_p: float,
    timeout_s: int,
) -> ParaphraseResult:
    start = time.perf_counter()
    output, metadata = await asyncio.wait_for(
        backend.paraphrase(text=text, temperature=temperature, top_p=top_p),
        timeout=timeout_s,
    )
    latency_ms = int((time.perf_counter() - start) * 1000)
    return ParaphraseResult(
        request_id=str(uuid.uuid4()),
        input_text=text,
        output_text=output,
        backend=backend.name,
        latency_ms=latency_ms,
        metrics=compute_metrics(text, output),
        metadata=metadata,
    )


def compute_metrics(input_text: str, output_text: str) -> dict[str, Any]:
    input_words = len(input_text.split())
    output_words = len(output_text.split())
    ratio = SequenceMatcher(None, input_text, output_text).ratio()
    word_delta_pct = 0.0
    if input_words:
        word_delta_pct = ((output_words - input_words) / input_words) * 100.0
    return {
        "input_words": input_words,
        "output_words": output_words,
        "word_delta_pct": round(word_delta_pct, 1),
        "char_edit_rate": round(1.0 - ratio, 3),
        "length_ratio": round((len(output_text) / max(1, len(input_text))), 3),
    }


def _split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [piece for piece in pieces if piece]


def _estimate_generation_max_tokens(original: str, default_max_tokens: int, min_tokens: int = 64) -> int:
    word_count = len(original.split())
    char_based = max(int(len(original) / 6), min_tokens)
    word_based = max(int(word_count * 1.6), min_tokens)
    return min(default_max_tokens, max(char_based, word_based))


def _normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    if text and text[-1] not in ".!?":
        text += "."
    # Tiny deterministic variation to avoid every preview feeling templated.
    if len(text.split()) > 22 and random.Random(text).random() < 0.18:
        text = text.replace(" and ", "; and ", 1)
    return text
