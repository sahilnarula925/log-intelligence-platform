"""HTTP webhook ingest — push any JSON log line into the pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from aiohttp import web

from shared.events import LogEvent, parse_timestamp
from ingester.sources.base import LogSource

logger = logging.getLogger(__name__)


def normalize_webhook_payload(raw: dict[str, Any]) -> LogEvent:
    message = str(raw.get("message", "")).strip()
    if not message:
        raise ValueError("message is required")

    ts = raw.get("timestamp", datetime.now(timezone.utc))
    service = str(raw.get("service", "webhook"))
    severity = str(raw.get("severity", "INFO")).upper()
    if severity not in {"INFO", "WARN", "ERROR", "DEBUG"}:
        severity = "INFO"

    timestamp = parse_timestamp(ts)
    event_id = hashlib.sha256(
        f"webhook:{service}:{message}{timestamp.isoformat()}".encode()
    ).hexdigest()
    formatted = f"[{severity}] {service}: {message}"

    return LogEvent(
        id=event_id,
        timestamp=timestamp,
        service=service,
        severity=severity,
        message=formatted,
        source="webhook",
        raw=raw,
    )


class WebhookSource(LogSource):
    """Accepts POST /ingest and enqueues events via an internal asyncio queue."""

    name = "webhook"

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._queue: asyncio.Queue[LogEvent] = asyncio.Queue()
        self._runner: web.AppRunner | None = None

    async def _handle_ingest(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=400)

        if not isinstance(payload, dict):
            return web.json_response({"error": "body must be a JSON object"}, status=400)

        try:
            event = normalize_webhook_payload(payload)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)

        await self._queue.put(event)
        return web.json_response({"status": "accepted", "id": event.id})

    async def _handle_health(self, _request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "source": "webhook"})

    async def start_server(self) -> None:
        app = web.Application()
        app.router.add_post("/ingest", self._handle_ingest)
        app.router.add_get("/health", self._handle_health)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info("[webhook] listening on http://%s:%d/ingest", self.host, self.port)

    async def stop_server(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def stream(self) -> AsyncIterator[LogEvent]:
        await self.start_server()
        try:
            while True:
                event = await self._queue.get()
                yield event
        finally:
            await self.stop_server()
