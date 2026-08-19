"""Wikipedia SSE ingester — streams events into Redis."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import aiohttp
import redis.asyncio as aioredis
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ingester] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    wikimedia_sse_url: str = "https://stream.wikimedia.org/v2/stream/recentchange"
    redis_stream_key: str = "logs:raw"
    stream_max_len: int = 10_000
    backpressure_sleep_ms: int = 100
    health_log_interval_sec: int = 30
    # Wikimedia blocks requests without a descriptive User-Agent (403).
    user_agent: str = (
        "LogIntelligencePlatform/1.0 (portfolio project; aiohttp) Python/3.11"
    )


@dataclass
class LogEvent:
    id: str
    timestamp: datetime
    service: str
    severity: str
    message: str
    raw: dict[str, Any]


def derive_severity(event: dict[str, Any]) -> str:
    """Map Wikipedia edit metadata to log severity levels."""
    if event.get("bot"):
        return "INFO"
    if event.get("type") == "new":
        return "WARN"
    old_len = event.get("length", {}).get("old") or 0
    new_len = event.get("length", {}).get("new") or 0
    if abs(new_len - old_len) > 5000:
        return "ERROR"
    return "INFO"


def parse_timestamp(ts: Any) -> datetime:
    """Wikimedia sends timestamp as Unix int or ISO-8601 string."""
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(ts, str):
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return datetime.now(timezone.utc)


def normalize_event(raw: dict[str, Any]) -> LogEvent:
    title = str(raw.get("title", "unknown"))
    ts = raw.get("timestamp", datetime.now(timezone.utc))
    user = str(raw.get("user", "anonymous"))
    length = raw.get("length") or {}
    old_len = length.get("old") or 0
    new_len = length.get("new") or 0
    delta = abs(int(new_len) - int(old_len))

    event_id = hashlib.sha256(f"{title}{ts}".encode()).hexdigest()
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
        raw=raw,
    )


def event_to_redis_fields(event: LogEvent) -> dict[str, str]:
    return {
        "id": event.id,
        "timestamp": event.timestamp.isoformat(),
        "service": event.service,
        "severity": event.severity,
        "message": event.message,
        "raw": json.dumps(event.raw),
    }


class Ingester:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.redis: aioredis.Redis | None = None
        self.events_written = 0
        self.reconnect_count = 0
        self._window_start = time.monotonic()
        self._window_events = 0

    async def connect_redis(self) -> None:
        self.redis = aioredis.from_url(
            self.settings.redis_url, decode_responses=True
        )
        await self.redis.ping()

    async def apply_backpressure(self) -> None:
        assert self.redis is not None
        depth = await self.redis.xlen(self.settings.redis_stream_key)
        # Sleep when the stream is full enough to signal downstream lag.
        while depth >= self.settings.stream_max_len:
            await asyncio.sleep(self.settings.backpressure_sleep_ms / 1000)
            depth = await self.redis.xlen(self.settings.redis_stream_key)

    async def write_event(self, event: LogEvent) -> None:
        assert self.redis is not None
        await self.apply_backpressure()
        await self.redis.xadd(
            self.settings.redis_stream_key,
            event_to_redis_fields(event),
        )
        self.events_written += 1
        self._window_events += 1

        # Track ingestion rate for the /stats endpoint (minute buckets).
        minute_key = int(time.time()) // 60
        await self.redis.hincrby("stats:ingestion", str(minute_key), 1)

    async def health_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.health_log_interval_sec)
            elapsed = time.monotonic() - self._window_start
            rate = self._window_events / elapsed if elapsed > 0 else 0.0
            depth = await self.redis.xlen(self.settings.redis_stream_key) if self.redis else 0
            logger.info(
                "health events/sec=%.1f stream_depth=%d reconnects=%d total=%d",
                rate,
                depth,
                self.reconnect_count,
                self.events_written,
            )
            self._window_start = time.monotonic()
            self._window_events = 0

    async def consume_sse(self) -> None:
        backoff = 0.3  # 300ms initial backoff per spec
        max_backoff = 60.0

        while True:
            try:
                timeout = aiohttp.ClientTimeout(total=None, sock_read=120)
                headers = {"User-Agent": self.settings.user_agent}
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(
                        self.settings.wikimedia_sse_url, headers=headers
                    ) as resp:
                        resp.raise_for_status()
                        logger.info("Connected to Wikimedia SSE stream")
                        backoff = 0.3  # reset on successful connect

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
                                    event = normalize_event(data)
                                    await self.write_event(event)
                                except json.JSONDecodeError:
                                    continue
                                except Exception as exc:
                                    logger.debug("Skipping event: %s", exc)
                                    continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.reconnect_count += 1
                logger.warning(
                    "SSE disconnected (%s), reconnecting in %.1fs",
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    async def run(self) -> None:
        await self.connect_redis()
        await asyncio.gather(self.health_loop(), self.consume_sse())


async def main() -> None:
    settings = Settings()
    ingester = Ingester(settings)
    await ingester.run()


if __name__ == "__main__":
    asyncio.run(main())
