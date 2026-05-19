"""Configuration helpers for the StealthRL demo service."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ApiKeyConfig:
    key: str
    label: str
    daily_limit: int | None = None


@dataclass(frozen=True)
class DemoSettings:
    app_name: str = "StealthRL Demo"
    public_daily_limit: int = 20
    public_quota_scope: str = "ip"
    db_path: Path = Path("demo/demo_usage.sqlite3")
    api_keys: tuple[ApiKeyConfig, ...] = ()
    inference_backend: str = "mock"
    checkpoint_json: str | None = None
    max_chars: int = 5000
    request_timeout_s: int = 90


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _parse_api_keys(raw: str | None) -> tuple[ApiKeyConfig, ...]:
    if not raw or not raw.strip():
        return ()

    raw = raw.strip()
    configs: list[ApiKeyConfig] = []
    if raw.startswith("{"):
        payload: dict[str, Any] = json.loads(raw)
        for key, meta in payload.items():
            if isinstance(meta, dict):
                label = str(meta.get("label") or key[-6:])
                daily_limit = meta.get("daily_limit")
                daily_limit = int(daily_limit) if daily_limit is not None else None
            else:
                label = str(meta or key[-6:])
                daily_limit = None
            configs.append(ApiKeyConfig(key=key, label=label, daily_limit=daily_limit))
        return tuple(configs)

    for item in raw.split(","):
        key = item.strip()
        if key:
            configs.append(ApiKeyConfig(key=key, label=key[-6:]))
    return tuple(configs)


def load_settings() -> DemoSettings:
    quota_scope = os.getenv("STEALTHRL_DEMO_PUBLIC_QUOTA_SCOPE", "ip").strip().lower()
    if quota_scope not in {"ip", "global"}:
        raise ValueError("STEALTHRL_DEMO_PUBLIC_QUOTA_SCOPE must be 'ip' or 'global'")

    return DemoSettings(
        public_daily_limit=_int_env("STEALTHRL_DEMO_PUBLIC_DAILY_LIMIT", 20),
        public_quota_scope=quota_scope,
        db_path=Path(os.getenv("STEALTHRL_DEMO_DB_PATH", "demo/demo_usage.sqlite3")),
        api_keys=_parse_api_keys(os.getenv("STEALTHRL_DEMO_API_KEYS")),
        inference_backend=os.getenv("STEALTHRL_DEMO_INFERENCE_BACKEND", "mock").strip().lower(),
        checkpoint_json=os.getenv("STEALTHRL_DEMO_CHECKPOINT_JSON") or None,
        max_chars=_int_env("STEALTHRL_DEMO_MAX_CHARS", 5000),
        request_timeout_s=_int_env("STEALTHRL_DEMO_REQUEST_TIMEOUT_S", 90),
    )
