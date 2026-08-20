"""Time-partition helpers for Qdrant collection routing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def partition_key(ts: datetime, granularity: str = "hour") -> str:
    """Return a bucket key for the event timestamp."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)

    if granularity == "day":
        return ts.strftime("%Y%m%d")
    return ts.strftime("%Y%m%d%H")


def collection_name(prefix: str, ts: datetime, granularity: str = "hour") -> str:
    return f"{prefix}_{partition_key(ts, granularity)}"


def partition_keys_for_window(
    hours: float,
    granularity: str = "hour",
    *,
    now: datetime | None = None,
) -> list[str]:
    """List partition keys covering the last `hours` from now (UTC)."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    start = now - timedelta(hours=hours)
    keys: list[str] = []
    cursor = start.replace(minute=0, second=0, microsecond=0)
    if granularity == "day":
        cursor = cursor.replace(hour=0)

    step = timedelta(days=1) if granularity == "day" else timedelta(hours=1)
    while cursor <= now:
        keys.append(partition_key(cursor, granularity))
        cursor += step

    # Deduplicate while preserving order (DST-safe for hour buckets).
    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def collection_names_for_window(
    prefix: str,
    hours: float,
    granularity: str = "hour",
) -> list[str]:
    return [f"{prefix}_{key}" for key in partition_keys_for_window(hours, granularity)]
