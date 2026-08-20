"""Mastodon public timeline via REST polling (anonymous SSE was removed in Mastodon 4.2)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import aiohttp

from shared.events import LogEvent, parse_timestamp
from ingester.sources.base import LogSource

logger = logging.getLogger(__name__)

HTML_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(content: str) -> str:
    text = HTML_TAG_RE.sub("", content)
    return " ".join(text.split())


def derive_mastodon_severity(raw: dict[str, Any]) -> str:
    content = strip_html(str(raw.get("content", "")))
    if raw.get("sensitive"):
        return "WARN"
    if len(content) > 500:
        return "ERROR"
    return "INFO"


def normalize_mastodon_event(raw: dict[str, Any]) -> LogEvent:
    account = raw.get("account") or {}
    username = str(account.get("username", "unknown"))
    ts = raw.get("created_at", datetime.now(timezone.utc))
    content = strip_html(str(raw.get("content", "")))
    preview = content[:160] + ("…" if len(content) > 160 else "")

    event_id = hashlib.sha256(f"mastodon:{raw.get('id', '')}{ts}".encode()).hexdigest()
    timestamp = parse_timestamp(ts)
    service = str(account.get("acct", "mastodon"))
    severity = derive_mastodon_severity(raw)
    message = f"@{username} posted on Mastodon: {preview or '(empty)'}"

    return LogEvent(
        id=event_id,
        timestamp=timestamp,
        service=service,
        severity=severity,
        message=message,
        source="mastodon",
        raw=raw,
    )


class MastodonSource(LogSource):
    """Polls GET /api/v1/timelines/public — works without auth on most instances."""

    name = "mastodon"

    def __init__(
        self,
        base_url: str,
        user_agent: str,
        poll_interval_sec: float = 3.0,
        access_token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.poll_interval_sec = poll_interval_sec
        self.access_token = access_token

    async def fetch_public_timeline(
        self, session: aiohttp.ClientSession, since_id: str | None
    ) -> list[dict[str, Any]]:
        params: dict[str, str | int] = {"limit": 40}
        if since_id:
            params["since_id"] = since_id

        headers = {"User-Agent": self.user_agent}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        url = f"{self.base_url}/api/v1/timelines/public"
        async with session.get(url, params=params, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()

        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(data["error"])
        if not isinstance(data, list):
            raise RuntimeError(f"unexpected Mastodon response: {type(data)}")
        return data

    async def stream(self) -> AsyncIterator[LogEvent]:
        backoff = 0.3
        max_backoff = 60.0
        since_id: str | None = None
        bootstrapped = False

        while True:
            try:
                timeout = aiohttp.ClientTimeout(total=30)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    logger.info(
                        "[mastodon] polling %s/api/v1/timelines/public",
                        self.base_url,
                    )
                    backoff = 0.3

                    while True:
                        posts = await self.fetch_public_timeline(session, since_id)
                        if posts:
                            since_id = posts[0]["id"]
                            to_emit = posts if bootstrapped else posts[:5]
                            bootstrapped = True
                            for post in reversed(to_emit):
                                yield normalize_mastodon_event(post)

                        await asyncio.sleep(self.poll_interval_sec)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "[mastodon] poll failed (%s), retrying in %.1fs",
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
