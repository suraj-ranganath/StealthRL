"""SQLite-backed API-key and public quota accounting."""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import ApiKeyConfig


@dataclass(frozen=True)
class QuotaDecision:
    allowed: bool
    authenticated: bool
    limit: int | None
    remaining: int | None
    subject: str
    label: str
    reason: str | None = None


class QuotaStore:
    def __init__(
        self,
        db_path: Path,
        api_keys: Iterable[ApiKeyConfig],
        public_daily_limit: int,
        public_quota_scope: str = "ip",
    ) -> None:
        self.db_path = db_path
        self.api_keys = tuple(api_keys)
        self.public_daily_limit = public_daily_limit
        self.public_quota_scope = public_quota_scope
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_counts (
                    day TEXT NOT NULL,
                    subject_hash TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (day, subject_hash, scope)
                )
                """
            )
            conn.commit()

    @staticmethod
    def today_utc() -> str:
        return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _hash_subject(subject: str) -> str:
        return hashlib.sha256(subject.encode("utf-8")).hexdigest()

    def _match_api_key(self, api_key: str | None) -> ApiKeyConfig | None:
        if not api_key:
            return None
        for candidate in self.api_keys:
            if hmac.compare_digest(api_key, candidate.key):
                return candidate
        return None

    def check_api_key(self, api_key: str | None) -> ApiKeyConfig | None:
        return self._match_api_key(api_key)

    def consume(self, api_key: str | None, client_id: str) -> QuotaDecision:
        key_config = self._match_api_key(api_key)
        if key_config is not None:
            if key_config.daily_limit is None:
                return QuotaDecision(
                    allowed=True,
                    authenticated=True,
                    limit=None,
                    remaining=None,
                    subject="api-key",
                    label=key_config.label,
                )
            return self._consume_subject(
                subject=f"api:{key_config.key}",
                scope="api_key",
                limit=key_config.daily_limit,
                authenticated=True,
                label=key_config.label,
            )

        public_subject = "public:global" if self.public_quota_scope == "global" else f"public:{client_id}"
        return self._consume_subject(
            subject=public_subject,
            scope=f"public_{self.public_quota_scope}",
            limit=self.public_daily_limit,
            authenticated=False,
            label="public",
        )

    def _consume_subject(
        self,
        subject: str,
        scope: str,
        limit: int,
        authenticated: bool,
        label: str,
    ) -> QuotaDecision:
        day = self.today_utc()
        subject_hash = self._hash_subject(subject)
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                row = conn.execute(
                    "SELECT count FROM usage_counts WHERE day=? AND subject_hash=? AND scope=?",
                    (day, subject_hash, scope),
                ).fetchone()
                count = int(row[0]) if row else 0
                if count >= limit:
                    return QuotaDecision(
                        allowed=False,
                        authenticated=authenticated,
                        limit=limit,
                        remaining=0,
                        subject=scope,
                        label=label,
                        reason="daily_quota_exceeded",
                    )
                count += 1
                conn.execute(
                    """
                    INSERT INTO usage_counts(day, subject_hash, scope, count, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(day, subject_hash, scope)
                    DO UPDATE SET count=excluded.count, updated_at=excluded.updated_at
                    """,
                    (day, subject_hash, scope, count, now),
                )
                conn.commit()
        return QuotaDecision(
            allowed=True,
            authenticated=authenticated,
            limit=limit,
            remaining=max(0, limit - count),
            subject=scope,
            label=label,
        )

    def peek_public(self, client_id: str) -> QuotaDecision:
        day = self.today_utc()
        subject = "public:global" if self.public_quota_scope == "global" else f"public:{client_id}"
        subject_hash = self._hash_subject(subject)
        scope = f"public_{self.public_quota_scope}"
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT count FROM usage_counts WHERE day=? AND subject_hash=? AND scope=?",
                (day, subject_hash, scope),
            ).fetchone()
        count = int(row[0]) if row else 0
        return QuotaDecision(
            allowed=count < self.public_daily_limit,
            authenticated=False,
            limit=self.public_daily_limit,
            remaining=max(0, self.public_daily_limit - count),
            subject=scope,
            label="public",
            reason=None if count < self.public_daily_limit else "daily_quota_exceeded",
        )
