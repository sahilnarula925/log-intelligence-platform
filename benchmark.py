#!/usr/bin/env python3
"""Benchmark the log intelligence pipeline.

Replays stored Redis stream events at 40x speed (~2K events/sec) and
measures ingestion throughput, deduplication ratio, embedding latency,
search latency percentiles, and Qdrant index size.

Usage:
    python benchmark.py            # uses defaults from .env.example
    REDIS_URL=redis://localhost:6379/0 python benchmark.py
"""

from __future__ import annotations

import asyncio
import os
import random
import statistics
import time
from typing import Any

import httpx
import redis.asyncio as aioredis
from qdrant_client import AsyncQdrantClient

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
API_URL = os.getenv("API_URL", "http://localhost:8000")
STREAM_KEY = os.getenv("REDIS_STREAM_KEY", "logs:raw")
REPLAY_SPEED = float(os.getenv("REPLAY_SPEED", "40"))
SOURCE_COUNT = int(os.getenv("BENCHMARK_SOURCE_COUNT", "5000"))


async def collect_source_events(r: aioredis.Redis, count: int) -> list[dict[str, str]]:
    """Read up to `count` events from the live stream for replay."""
    entries = await r.xrevrange(STREAM_KEY, count=count)
    entries.reverse()  # oldest first
    return [fields for _, fields in entries]


async def replay_events(
    r: aioredis.Redis, events: list[dict[str, str]], speed: float
) -> tuple[int, float]:
    """Re-inject events at `speed` times the nominal ~50 evt/s Wikimedia rate."""
    nominal_rate = 50.0
    target_rate = nominal_rate * speed
    interval = 1.0 / target_rate

    # Reset minute-level counters so replay metrics are clean.
    await r.delete("stats:ingestion")
    start = time.perf_counter()
    for fields in events:
        await r.xadd(STREAM_KEY, fields)
        await asyncio.sleep(interval)
    elapsed = time.perf_counter() - start
    return len(events), elapsed


async def wait_for_processing(
    r: aioredis.Redis, qdrant: AsyncQdrantClient, target: int, timeout: float = 120
) -> None:
    """Block until the processor has indexed at least `target` new points."""
    deadline = time.time() + timeout
    collection = os.getenv("QDRANT_COLLECTION", "logs")
    baseline = (await qdrant.get_collection(collection)).points_count or 0
    goal = baseline + target

    while time.time() < deadline:
        current = (await qdrant.get_collection(collection)).points_count or 0
        processed = int(await r.get("stats:total_processed") or 0)
        if current >= goal or processed >= target:
            return
        await asyncio.sleep(2)
    print(f"  Warning: timed out waiting for index (goal={goal})")


async def measure_search_latency(client: httpx.AsyncClient, n: int = 100) -> list[float]:
    queries = [
        "bot edited article",
        "vandalism reverted",
        "new page created",
        "large edit bytes",
        "anonymous user edit",
        "science article update",
        "talk page edit",
        "category change",
        "wikidata edit",
        "minor correction",
    ]
    latencies: list[float] = []
    for _ in range(n):
        q = random.choice(queries)
        t0 = time.perf_counter()
        resp = await client.post(
            f"{API_URL}/search",
            json={
                "query": q,
                "time_window_hours": 24,
                "severity": ["INFO", "WARN", "ERROR"],
                "limit": 20,
            },
        )
        resp.raise_for_status()
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def print_table(rows: list[tuple[str, str, str]]) -> None:
    col_w = [max(len(r[i]) for r in rows) for i in range(3)]
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    print(sep)
    for i, row in enumerate(rows):
        print("| " + " | ".join(row[j].ljust(col_w[j]) for j in range(3)) + " |")
        if i == 0:
            print(sep)
    print(sep)


async def main() -> None:
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    qdrant = AsyncQdrantClient(url=QDRANT_URL)

    print("Collecting source events from Redis stream…")
    events = await collect_source_events(r, SOURCE_COUNT)
    if not events:
        print("No events in stream — start docker-compose and wait for ingestion first.")
        await r.aclose()
        await qdrant.close()
        return

    print(f"Replaying {len(events)} events at {REPLAY_SPEED}x speed…")
    replayed, replay_elapsed = await replay_events(r, events, REPLAY_SPEED)
    throughput = replayed / replay_elapsed if replay_elapsed else 0

    print("Waiting for processor to catch up…")
    await wait_for_processing(r, qdrant, replayed)

    total_processed = int(await r.get("stats:total_processed") or 0)
    dedup_hits = int(await r.get("stats:dedup_hits_total") or 0)
    dedup_ratio = dedup_hits / total_processed * 100 if total_processed else 0

    embed_samples = await r.hgetall("stats:embed_latency")
    embed_latencies = [float(v) for v in embed_samples.values()] if embed_samples else [0]

    collection = os.getenv("QDRANT_COLLECTION", "logs")
    index_size = (await qdrant.get_collection(collection)).points_count or 0

    print("Running 100 search queries…")
    async with httpx.AsyncClient(timeout=30) as client:
        search_latencies = await measure_search_latency(client)

    rows: list[tuple[str, str, str]] = [
        ("Metric", "Value", "Target"),
        ("Ingestion throughput", f"{throughput:.0f} evt/s", ">2,000 evt/s"),
        ("Deduplication ratio", f"{dedup_ratio:.1f}%", ">35%"),
        ("Embedding latency (avg)", f"{statistics.mean(embed_latencies):.1f} ms/batch", "—"),
        ("Search p50", f"{percentile(search_latencies, 50):.1f} ms", "—"),
        ("Search p95", f"{percentile(search_latencies, 95):.1f} ms", "<100 ms"),
        ("Search p99", f"{percentile(search_latencies, 99):.1f} ms", "—"),
        ("Index size", f"{index_size:,} vectors", "—"),
    ]

    print("\n=== Benchmark Results ===\n")
    print_table(rows)

    await r.aclose()
    await qdrant.close()


if __name__ == "__main__":
    asyncio.run(main())
