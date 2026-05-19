"""FastAPI application for the StealthRL demo website."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import DemoSettings, load_settings
from .inference import BaseDemoBackend, build_backend, run_paraphrase
from .rate_limit import QuotaDecision, QuotaStore

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


class ParaphraseRequest(BaseModel):
    text: str = Field(min_length=20)
    temperature: float = Field(default=0.9, ge=0.0, le=1.5)
    top_p: float = Field(default=0.95, ge=0.1, le=1.0)


def _extract_api_key(request: Request) -> str | None:
    auth_header = request.headers.get("authorization", "").strip()
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            return token
    key = request.headers.get("x-stealthrl-api-key", "").strip()
    return key or None


def _client_id(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").strip()
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _quota_payload(decision: QuotaDecision) -> dict[str, Any]:
    return {
        "authenticated": decision.authenticated,
        "label": decision.label,
        "limit": decision.limit,
        "remaining": decision.remaining,
        "scope": decision.subject,
    }


def create_app(
    settings: DemoSettings | None = None,
    quota_store: QuotaStore | None = None,
    backend: BaseDemoBackend | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    quota_store = quota_store or QuotaStore(
        db_path=settings.db_path,
        api_keys=settings.api_keys,
        public_daily_limit=settings.public_daily_limit,
        public_quota_scope=settings.public_quota_scope,
    )
    backend = backend or build_backend(settings)

    app = FastAPI(
        title="StealthRL Demo",
        description="Interactive StealthRL paraphrase demo with API-key and public quota controls.",
        version="0.1.0",
    )
    app.state.settings = settings
    app.state.quota_store = quota_store
    app.state.backend = backend

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/privacy", include_in_schema=False)
    async def privacy() -> FileResponse:
        return FileResponse(STATIC_DIR / "privacy.html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "backend": app.state.backend.name,
            "public_daily_limit": app.state.settings.public_daily_limit,
            "public_quota_scope": app.state.settings.public_quota_scope,
        }

    @app.get("/api/config")
    async def public_config(request: Request) -> dict[str, Any]:
        decision = app.state.quota_store.peek_public(_client_id(request))
        return {
            "backend": app.state.backend.name,
            "max_chars": app.state.settings.max_chars,
            "public_daily_limit": app.state.settings.public_daily_limit,
            "public_quota_scope": app.state.settings.public_quota_scope,
            "api_keys_enabled": bool(app.state.settings.api_keys),
            "public_quota": _quota_payload(decision),
        }

    @app.post("/api/paraphrase")
    async def paraphrase(payload: ParaphraseRequest, request: Request) -> dict[str, Any]:
        text = payload.text.strip()
        if len(text) > app.state.settings.max_chars:
            raise HTTPException(
                status_code=413,
                detail=f"Text is too long for the demo limit ({app.state.settings.max_chars} characters).",
            )

        api_key = _extract_api_key(request)
        if api_key and not app.state.quota_store.check_api_key(api_key):
            raise HTTPException(status_code=401, detail="Invalid API key.")
        decision = app.state.quota_store.consume(api_key=api_key, client_id=_client_id(request))
        if not decision.allowed:
            raise HTTPException(
                status_code=429,
                detail="Daily demo quota exceeded. Use an API key or try again tomorrow.",
            )

        try:
            result = await run_paraphrase(
                backend=app.state.backend,
                text=text,
                temperature=payload.temperature,
                top_p=payload.top_p,
                timeout_s=app.state.settings.request_timeout_s,
            )
        except TimeoutError:
            raise HTTPException(status_code=504, detail="StealthRL inference timed out.")
        except Exception as exc:
            logger.exception("StealthRL demo inference failed")
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

        return {
            "request_id": result.request_id,
            "input_text": result.input_text,
            "output_text": result.output_text,
            "backend": result.backend,
            "latency_ms": result.latency_ms,
            "metrics": result.metrics,
            "metadata": result.metadata,
            "quota": _quota_payload(decision),
        }

    return app


app = create_app()
