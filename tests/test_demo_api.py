from pathlib import Path

from fastapi.testclient import TestClient

from demo.stealthrl_demo.app import create_app
from demo.stealthrl_demo.config import ApiKeyConfig, DemoSettings
from demo.stealthrl_demo.rate_limit import QuotaStore


def _client(tmp_path: Path, public_limit: int = 2) -> TestClient:
    settings = DemoSettings(
        public_daily_limit=public_limit,
        public_quota_scope="ip",
        db_path=tmp_path / "usage.sqlite3",
        api_keys=(ApiKeyConfig(key="stealth-test-key", label="test", daily_limit=10),),
        inference_backend="mock",
        max_chars=1000,
    )
    quota_store = QuotaStore(
        db_path=settings.db_path,
        api_keys=settings.api_keys,
        public_daily_limit=settings.public_daily_limit,
        public_quota_scope=settings.public_quota_scope,
    )
    return TestClient(create_app(settings=settings, quota_store=quota_store))


def test_public_quota_is_enforced(tmp_path):
    client = _client(tmp_path, public_limit=1)
    payload = {
        "text": "AI text detectors can fail when adversarial paraphrasing changes surface form while preserving meaning."
    }
    first = client.post("/api/paraphrase", json=payload)
    assert first.status_code == 200
    assert first.json()["quota"]["remaining"] == 0

    second = client.post("/api/paraphrase", json=payload)
    assert second.status_code == 429


def test_index_serves_static_demo(tmp_path):
    client = _client(tmp_path)
    response = client.get("/")
    assert response.status_code == 200
    assert "StealthRL Demo" in response.text


def test_api_key_bypasses_public_quota(tmp_path):
    client = _client(tmp_path, public_limit=0)
    payload = {
        "text": "AI text detectors can fail when adversarial paraphrasing changes surface form while preserving meaning."
    }
    response = client.post(
        "/api/paraphrase",
        json=payload,
        headers={"Authorization": "Bearer stealth-test-key"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["quota"]["authenticated"] is True
    assert body["output_text"]


def test_invalid_api_key_is_rejected(tmp_path):
    client = _client(tmp_path)
    payload = {
        "text": "AI text detectors can fail when adversarial paraphrasing changes surface form while preserving meaning."
    }
    response = client.post(
        "/api/paraphrase",
        json=payload,
        headers={"X-StealthRL-API-Key": "not-a-real-key"},
    )
    assert response.status_code == 401
