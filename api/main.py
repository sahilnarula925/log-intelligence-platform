"""Query API — hybrid semantic + keyword search with RRF ranking."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from functools import partial
from datetime import datetime, timedelta, timezone
from typing import Any

import redis.asyncio as aioredis
from fastembed import SparseTextEmbedding
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    MatchAny,
    NamedSparseVector,
    NamedVector,
    SparseVector,
)
from sentence_transformers import SentenceTransformer

from shared.partitions import collection_names_for_window

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_prefix: str = "logs"
    partition_granularity: str = "hour"
    embedding_model: str = "all-MiniLM-L6-v2"
    sparse_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"
    rrf_k: int = 60
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str
    time_window_hours: float = 2
    severity: list[str] = Field(default_factory=lambda: ["INFO", "WARN", "ERROR"])
    sources: list[str] | None = None
    limit: int = 20


class SearchResult(BaseModel):
    message: str
    service: str
    severity: str
    source: str
    timestamp: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_searched: int
    query_ms: float


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------


class AppState:
    redis: aioredis.Redis
    qdrant: AsyncQdrantClient
    dense_model: SentenceTransformer
    sparse_model: SparseTextEmbedding


state = AppState()  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    state.qdrant = AsyncQdrantClient(url=settings.qdrant_url)

    loop = asyncio.get_event_loop()
    state.dense_model = await loop.run_in_executor(
        None, SentenceTransformer, settings.embedding_model
    )
    state.sparse_model = await loop.run_in_executor(
        None,
        partial(SparseTextEmbedding, model_name=settings.sparse_model),
    )
    yield
    await state.redis.aclose()
    await state.qdrant.close()


app = FastAPI(title="Log Intelligence Platform", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_filter(req: SearchRequest) -> Filter:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=req.time_window_hours)
    conditions = [
        FieldCondition(
            key="timestamp",
            range=DatetimeRange(gte=cutoff.isoformat()),
        ),
        FieldCondition(
            key="severity",
            match=MatchAny(any=req.severity),
        ),
    ]
    if req.sources:
        conditions.append(
            FieldCondition(
                key="source",
                match=MatchAny(any=req.sources),
            )
        )
    return Filter(must=conditions)


def reciprocal_rank_fusion(
    dense_results: list[Any],
    sparse_results: list[Any],
    k: int = 60,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Merge ranked lists using RRF: score = 1/(k+rank_dense) + 1/(k+rank_sparse)."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}

    for rank, hit in enumerate(dense_results, start=1):
        pid = str(hit.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        payloads[pid] = hit.payload or {}

    for rank, hit in enumerate(sparse_results, start=1):
        pid = str(hit.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        if pid not in payloads:
            payloads[pid] = hit.payload or {}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(pid, score, payloads[pid]) for pid, score in ranked]


def embed_query_dense(query: str) -> list[float]:
    vector = state.dense_model.encode(query)
    return vector.tolist()


def embed_query_sparse(query: str) -> SparseVector:
    # BM42 uses query_embed at search time (document embed at index time).
    result = list(state.sparse_model.query_embed(query))[0]
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


async def indexed_collections() -> list[str]:
    """All Qdrant collections that hold indexed logs (partitions + legacy)."""
    prefix = settings.qdrant_collection_prefix
    all_collections = await state.qdrant.get_collections()
    names: list[str] = []
    for coll in all_collections.collections:
        if coll.name.startswith(f"{prefix}_") or coll.name == prefix:
            names.append(coll.name)
    return names


async def collections_for_window(hours: float) -> list[str]:
    """Return collections to search for the requested time window."""
    candidates = collection_names_for_window(
        settings.qdrant_collection_prefix,
        hours,
        settings.partition_granularity,
    )
    all_collections = await state.qdrant.get_collections()
    existing = {c.name for c in all_collections.collections}
    names = [name for name in candidates if name in existing]
    legacy = settings.qdrant_collection_prefix
    if legacy in existing and legacy not in names:
        names.append(legacy)
    return names


async def total_index_size() -> int:
    total = 0
    for name in await indexed_collections():
        coll = await state.qdrant.get_collection(name)
        total += coll.points_count or 0
    return total


async def search_partitions(
    collections: list[str],
    dense_vec: list[float],
    sparse_vec: SparseVector,
    query_filter: Filter,
    limit: int,
) -> tuple[list[Any], list[Any]]:
    if not collections:
        return [], []

    async def dense_search(name: str) -> list[Any]:
        return await state.qdrant.search(
            collection_name=name,
            query_vector=NamedVector(name="dense", vector=dense_vec),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

    async def sparse_search(name: str) -> list[Any]:
        return await state.qdrant.search(
            collection_name=name,
            query_vector=NamedSparseVector(name="sparse", vector=sparse_vec),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

    dense_batches, sparse_batches = await asyncio.gather(
        asyncio.gather(*(dense_search(name) for name in collections)),
        asyncio.gather(*(sparse_search(name) for name in collections)),
    )
    dense_hits = [hit for batch in dense_batches for hit in batch]
    sparse_hits = [hit for batch in sparse_batches for hit in batch]
    return dense_hits, sparse_hits


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    stream_depth = await state.redis.xlen("logs:raw")
    total_processed = int(await state.redis.get("stats:total_processed") or 0)
    dedup_hits = int(await state.redis.get("stats:dedup_hits_total") or 0)
    dedup_ratio = dedup_hits / total_processed if total_processed else 0.0

    index_size = await total_index_size()

    # Ingestion rate from last minute bucket.
    minute_key = str(int(time.time()) // 60)
    last_minute = int(await state.redis.hget("stats:ingestion", minute_key) or 0)

    return {
        "status": "ok",
        "ingestion_rate_per_sec": round(last_minute / 60, 2),
        "dedup_ratio": round(dedup_ratio, 4),
        "index_size": index_size,
        "redis_stream_depth": stream_depth,
        "total_processed": total_processed,
    }


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    t0 = time.perf_counter()
    query_filter = build_filter(req)

    loop = asyncio.get_event_loop()
    dense_vec = await loop.run_in_executor(None, embed_query_dense, req.query)
    sparse_vec = await loop.run_in_executor(None, embed_query_sparse, req.query)

    fetch_limit = max(req.limit * 3, 60)
    collections = await collections_for_window(req.time_window_hours)
    dense_hits, sparse_hits = await search_partitions(
        collections,
        dense_vec,
        sparse_vec,
        query_filter,
        fetch_limit,
    )

    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=settings.rrf_k)
    top = fused[: req.limit]

    total_searched = await total_index_size()

    results = [
        SearchResult(
            message=p.get("message", ""),
            service=p.get("service", ""),
            severity=p.get("severity", ""),
            source=p.get("source", "unknown"),
            timestamp=p.get("timestamp", ""),
            score=round(score, 4),
        )
        for _, score, p in top
    ]

    query_ms = (time.perf_counter() - t0) * 1000
    return SearchResponse(
        results=results,
        total_searched=total_searched,
        query_ms=round(query_ms, 2),
    )


@app.get("/stats")
async def stats() -> dict[str, Any]:
    """Return time-series of events/sec for the last 60 minutes."""
    now_minute = int(time.time()) // 60
    minutes = list(range(now_minute - 59, now_minute + 1))

    raw = await state.redis.hmget(
        "stats:ingestion", *[str(m) for m in minutes]
    )
    series = []
    for minute, count in zip(minutes, raw):
        count_int = int(count or 0)
        series.append(
            {
                "minute": datetime.fromtimestamp(
                    minute * 60, tz=timezone.utc
                ).isoformat(),
                "events": count_int,
                "events_per_sec": round(count_int / 60, 2),
            }
        )

    total_processed = int(await state.redis.get("stats:total_processed") or 0)
    dedup_hits = int(await state.redis.get("stats:dedup_hits_total") or 0)
    dedup_ratio = dedup_hits / total_processed if total_processed else 0.0

    return {
        "series": series,
        "events_indexed": await total_index_size(),
        "dedup_ratio": round(dedup_ratio, 4),
        "by_source": await state.redis.hgetall("stats:ingestion_by_source"),
    }
