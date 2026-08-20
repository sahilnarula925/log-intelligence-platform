"""Common log event schema used across ingester, processor, and API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class LogEvent:
    id: str
    timestamp: datetime
    service: str
    severity: str
    message: str
    source: str
    raw: dict[str, Any]


def parse_timestamp(ts: Any) -> datetime:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(ts, str):
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return datetime.now(timezone.utc)


def event_to_redis_fields(event: LogEvent) -> dict[str, str]:
    return {
        "id": event.id,
        "timestamp": event.timestamp.isoformat(),
        "service": event.service,
        "severity": event.severity,
        "message": event.message,
        "source": event.source,
        "raw": json.dumps(event.raw),
    }


def redis_fields_to_event(fields: dict[str, str]) -> dict[str, Any]:
    return {
        "id": fields["id"],
        "timestamp": parse_timestamp(fields["timestamp"]),
        "service": fields["service"],
        "severity": fields["severity"],
        "message": fields["message"],
        "source": fields.get("source", "unknown"),
        "raw": json.loads(fields.get("raw", "{}")),
    }
