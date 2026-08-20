"""Multi-source ingester — streams events from configured adapters into Redis."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Iterable

import redis.asyncio as aioredis
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.events import LogEvent, event_to_redis_fields
from ingester.sources.base import LogSource
from ingester.sources.mastodon import MastodonSource
from ingester.sources.webhook import WebhookSource
from ingester.sources.wikimedia import WikimediaSource

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ingester] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    redis_stream_key: str = "logs:raw"
    stream_max_len: int = 10_000
    backpressure_sleep_ms: int = 100
    health_log_interval_sec: int = 30
    log_sources: str = "wikimedia,mastodon,webhook"
    wikimedia_sse_url: str = "https://stream.wikimedia.org/v2/stream/recentchange"
    mastodon_base_url: str = "https://fosstodon.org"
    mastodon_poll_interval_sec: float = 3.0
    mastodon_access_token: str | None = None
    webhook_host: str = "0.0.0.0"
    webhook_port: int = 8081
    user_agent: str = (
        "LogIntelligencePlatform/1.0 (portfolio project; aiohttp) Python/3.11"
    )


class Ingester:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.redis: aioredis.Redis | None = None
        self.events_written = 0
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
        while depth >= self.settings.stream_max_len:
            await asyncio.sleep(self.settings.backpressure_sleep_ms / 1000)
            depth = await self.redis.xlen(self.settings.redis_stream_key)

    async def write_event(self, event: LogEvent) -> None:
        assert self.redis is not None
        await self.redis.xadd(
            self.settings.redis_stream_key,
            event_to_redis_fields(event),
            maxlen=self.settings.stream_max_len,
            approximate=True,
        )
        self.events_written += 1
        self._window_events += 1

        minute_key = int(time.time()) // 60
        await self.redis.hincrby("stats:ingestion", str(minute_key), 1)
        await self.redis.hincrby("stats:ingestion_by_source", event.source, 1)

    async def pump_source(self, source: LogSource) -> None:
        async for event in source.stream():
            try:
                await self.write_event(event)
            except Exception:
                logger.exception("[%s] failed to write event", source.name)

    async def health_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.health_log_interval_sec)
            elapsed = time.monotonic() - self._window_start
            rate = self._window_events / elapsed if elapsed > 0 else 0.0
            depth = (
                await self.redis.xlen(self.settings.redis_stream_key)
                if self.redis
                else 0
            )
            logger.info(
                "health events/sec=%.1f stream_depth=%d total=%d",
                rate,
                depth,
                self.events_written,
            )
            self._window_start = time.monotonic()
            self._window_events = 0

    def build_sources(self) -> list[LogSource]:
        enabled = {
            name.strip().lower()
            for name in self.settings.log_sources.split(",")
            if name.strip()
        }
        sources: list[LogSource] = []

        if "wikimedia" in enabled:
            sources.append(
                WikimediaSource(self.settings.wikimedia_sse_url, self.settings.user_agent)
            )
        if "mastodon" in enabled:
            sources.append(
                MastodonSource(
                    self.settings.mastodon_base_url,
                    self.settings.user_agent,
                    self.settings.mastodon_poll_interval_sec,
                    self.settings.mastodon_access_token,
                )
            )
        if "webhook" in enabled:
            sources.append(
                WebhookSource(self.settings.webhook_host, self.settings.webhook_port)
            )

        if not sources:
            raise ValueError(
                f"No log sources enabled. Set LOG_SOURCES (got: {self.settings.log_sources})"
            )
        return sources

    async def run(self) -> None:
        await self.connect_redis()
        sources = self.build_sources()
        logger.info("Starting sources: %s", ", ".join(s.name for s in sources))

        tasks = [asyncio.create_task(self.pump_source(src)) for src in sources]
        tasks.append(asyncio.create_task(self.health_loop()))
        await asyncio.gather(*tasks)


async def main() -> None:
    settings = Settings()
    ingester = Ingester(settings)
    await ingester.run()


if __name__ == "__main__":
    asyncio.run(main())
