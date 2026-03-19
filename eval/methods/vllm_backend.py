"""
Shared vLLM generation backend for local evaluation baselines.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VLLMEngineConfig:
    model_name: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    dtype: str = "auto"
    trust_remote_code: bool = True
    enforce_eager: bool = False
    disable_log_stats: bool = True


_ENGINE_CACHE: Dict[VLLMEngineConfig, "VLLMTextGenerator"] = {}


class VLLMTextGenerator:
    """Thin wrapper around a cached vLLM engine."""

    def __init__(self, config: VLLMEngineConfig):
        self.config = config
        self._llm = None

    def load(self) -> None:
        if self._llm is not None:
            return

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError("vLLM is not installed. Install with: pip install vllm") from exc

        logger.info(
            "Loading vLLM engine for %s (tp=%s, max_model_len=%s)",
            self.config.model_name,
            self.config.tensor_parallel_size,
            self.config.max_model_len,
        )
        self._llm = LLM(
            model=self.config.model_name,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            dtype=self.config.dtype,
            trust_remote_code=self.config.trust_remote_code,
            enforce_eager=self.config.enforce_eager,
            disable_log_stats=self.config.disable_log_stats,
        )

    def generate_batch(
        self,
        prompts: List[str],
        *,
        n: int = 1,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> List[List[str]]:
        if not prompts:
            return []

        self.load()

        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise RuntimeError("vLLM is not installed. Install with: pip install vllm") from exc

        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )
        outputs = self._llm.generate(prompts, sampling_params, use_tqdm=False)

        grouped: List[List[str]] = []
        for request_output in outputs:
            texts = [candidate.text.strip() for candidate in request_output.outputs if candidate.text.strip()]
            grouped.append(texts)
        return grouped


def get_vllm_generator(**kwargs) -> VLLMTextGenerator:
    if "tensor_parallel_size" not in kwargs:
        kwargs["tensor_parallel_size"] = int(os.getenv("STEALTHRL_VLLM_TP", "1"))
    if "gpu_memory_utilization" not in kwargs:
        kwargs["gpu_memory_utilization"] = float(os.getenv("STEALTHRL_VLLM_GPU_MEM_UTIL", "0.9"))
    if "max_model_len" not in kwargs:
        kwargs["max_model_len"] = int(os.getenv("STEALTHRL_VLLM_MAX_MODEL_LEN", "4096"))
    config = VLLMEngineConfig(**kwargs)
    generator = _ENGINE_CACHE.get(config)
    if generator is None:
        generator = VLLMTextGenerator(config)
        _ENGINE_CACHE[config] = generator
    return generator


def clear_vllm_generators() -> None:
    """Best-effort cache reset for long-lived processes."""
    _ENGINE_CACHE.clear()
