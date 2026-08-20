"""Wikimedia EventStreams recent-change feed."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import aiohttp

from shared.events import LogEvent, parse_timestamp
from ingester.sources.base import LogSource

logger = logging.getLogger(__name__)


def derive_severity(event: dict[str, Any]) -> str:
    if event.get("bot"):
        return "INFO"
    if event.get("type") == "new":
        return "WARN"
    old_len = event.get("length", {}).get("old") or 0
    new_len = event.get("length", {}).get("new") or 0
    if abs(new_len - old_len) > 5000:
        return "ERROR"
    return "INFO"


def normalize_wikimedia_event(raw: dict[str, Any]) -> LogEvent:
    title = str(raw.get("title", "unknown"))
    ts = raw.get("timestamp", datetime.now(timezone.utc))
    user = str(raw.get("user", "anonymous"))
    length = raw.get("length") or {}
    old_len = length.get("old") or 0
    new_len = length.get("new") or 0
    delta = abs(int(new_len) - int(old_len))

    event_id = hashlib.sha256(f"wikimedia:{title}{ts}".encode()).hexdigest()
    timestamp = parse_timestamp(ts)
    service = str(raw.get("server_name", raw.get("wiki", "unknown.wikimedia.org")))
    severity = derive_severity(raw)
    message = f"{user} edited {title}: +{delta} bytes"

    return LogEvent(
        id=event_id,
        timestamp=timestamp,
        service=service,
        severity=severity,
        message=message,
        source="wikimedia",
        raw=raw,
    )


class WikimediaSource(LogSource):
    name = "wikimedia"

    def __init__(self, sse_url: str, user_agent: str) -> None:
        self.sse_url = sse_url
        self.user_agent = user_agent

    async def stream(self) -> AsyncIterator[LogEvent]:
        backoff = 0.3
        max_backoff = 60.0

        while True:
            try:
                timeout = aiohttp.ClientTimeout(total=None, sock_read=120)
                headers = {"User-Agent": self.user_agent}
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(self.sse_url, headers=headers) as resp:
                        resp.raise_for_status()
                        logger.info("[wikimedia] connected to SSE stream")
                        backoff = 0.3

                        async for raw_line in resp.content:
                            line = raw_line.decode("utf-8", errors="ignore").strip()
                            if not line or line.startswith(":"):
                                continue
                            if line.startswith("data:"):
                                payload = line[5:].strip()
                                if not payload:
                                    continue
                                try:
                                    data = json.loads(payload)
                                    yield normalize_wikimedia_event(data)
                                except json.JSONDecodeError:
                                    continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "[wikimedia] disconnected (%s), reconnecting in %.1fs",
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
