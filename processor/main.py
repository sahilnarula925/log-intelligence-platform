"""Stream processor — deduplication, embedding, and time-partitioned Qdrant indexing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from functools import partial
from datetime import datetime
from typing import Any

import redis.asyncio as aioredis
from fastembed import SparseTextEmbedding
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from shared.events import redis_fields_to_event
from shared.partitions import collection_name
from shared.templates import extract_template

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [processor] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def template_hash(template: str) -> str:
    return hashlib.md5(template.encode()).hexdigest()


def event_id_to_uuid(event_id: str) -> str:
    return str(uuid.UUID(event_id[:32]))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_prefix: str = "logs"
    partition_granularity: str = "hour"
    redis_stream_key: str = "logs:raw"
    retry_stream_key: str = "logs:retry"
    dlq_stream_key: str = "logs:dlq"
    consumer_group: str = "processors"
    consumer_name: str = "processor-1"
    retry_consumer_group: str = "processors-retry"
    batch_size: int = 50
    embed_batch_size: int = 32
    max_retries: int = 5
    dedup_set_key: str = "seen:templates"
    vector_cache_key: str = "cache:vectors"
    embedding_model: str = "all-MiniLM-L6-v2"
    sparse_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"


class Processor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.redis: aioredis.Redis | None = None
        self.qdrant: AsyncQdrantClient | None = None
        self.dense_model: SentenceTransformer | None = None
        self.sparse_model: SparseTextEmbedding | None = None
        self.total_processed = 0
        self.dedup_hits = 0
        self._known_collections: set[str] = set()

    async def connect(self) -> None:
        self.redis = aioredis.from_url(
            self.settings.redis_url, decode_responses=True
        )
        await self.redis.ping()

        self.qdrant = AsyncQdrantClient(url=self.settings.qdrant_url)

        loop = asyncio.get_event_loop()
        self.dense_model = await loop.run_in_executor(
            None, SentenceTransformer, self.settings.embedding_model
        )
        self.sparse_model = await loop.run_in_executor(
            None,
            partial(SparseTextEmbedding, model_name=self.settings.sparse_model),
        )
        logger.info("Models loaded")

        await self.ensure_consumer_groups()

    async def ensure_consumer_group(self, stream_key: str, group: str) -> None:
        assert self.redis is not None
        try:
            await self.redis.xgroup_create(stream_key, group, id="0", mkstream=True)
            logger.info("Created consumer group '%s' on '%s'", group, stream_key)
        except aioredis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def ensure_consumer_groups(self) -> None:
        await self.ensure_consumer_group(
            self.settings.redis_stream_key, self.settings.consumer_group
        )
        await self.ensure_consumer_group(
            self.settings.retry_stream_key, self.settings.retry_consumer_group
        )

    async def ensure_collection(self, name: str) -> None:
        assert self.qdrant is not None
        if name in self._known_collections:
            return

        collections = await self.qdrant.get_collections()
        existing = {c.name for c in collections.collections}
        if name not in existing:
            await self.qdrant.create_collection(
                collection_name=name,
                vectors_config={
                    "dense": VectorParams(size=384, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    ),
                },
            )
            logger.info("Created Qdrant collection '%s'", name)

        self._known_collections.add(name)

    async def get_cached_vector(self, tmpl_hash: str) -> list[float] | None:
        assert self.redis is not None
        cached = await self.redis.hget(self.settings.vector_cache_key, tmpl_hash)
        if cached:
            return json.loads(cached)
        return None

    async def cache_vector(self, tmpl_hash: str, vector: list[float]) -> None:
        assert self.redis is not None
        await self.redis.hset(
            self.settings.vector_cache_key, tmpl_hash, json.dumps(vector)
        )

    async def is_seen_template(self, tmpl_hash: str) -> bool:
        assert self.redis is not None
        return bool(await self.redis.sismember(self.settings.dedup_set_key, tmpl_hash))

    async def mark_template_seen(self, tmpl_hash: str) -> None:
        assert self.redis is not None
        await self.redis.sadd(self.settings.dedup_set_key, tmpl_hash)

    def embed_dense_batch(self, texts: list[str]) -> list[list[float]]:
        assert self.dense_model is not None
        vectors = self.dense_model.encode(
            texts, batch_size=self.settings.embed_batch_size
        )
        return [v.tolist() for v in vectors]

    def embed_sparse(self, text: str) -> SparseVector:
        assert self.sparse_model is not None
        result = list(self.sparse_model.embed([text]))[0]
        return SparseVector(
            indices=result.indices.tolist(),
            values=result.values.tolist(),
        )

    async def enqueue_retry(
        self,
        stream_key: str,
        group: str,
        messages: list[tuple[str, dict[str, str]]],
    ) -> None:
        assert self.redis is not None

        for msg_id, fields in messages:
            retry_count = int(fields.get("retry_count", "0")) + 1
            if retry_count > self.settings.max_retries:
                dlq_fields = dict(fields)
                dlq_fields["retry_count"] = str(retry_count)
                dlq_fields["failed_from"] = stream_key
                dlq_fields["original_msg_id"] = msg_id
                await self.redis.xadd(self.settings.dlq_stream_key, dlq_fields)
                await self.redis.hincrby("stats:dlq_total", "count", 1)
                logger.warning(
                    "Moved message to DLQ after %d attempts (id=%s)",
                    retry_count,
                    fields.get("id", "?"),
                )
            else:
                retry_fields = dict(fields)
                retry_fields["retry_count"] = str(retry_count)
                retry_fields["retry_after"] = str(
                    int(time.time()) + min(2 ** retry_count, 60)
                )
                await self.redis.xadd(self.settings.retry_stream_key, retry_fields)
                await self.redis.hincrby("stats:retry_total", "count", 1)

            await self.redis.xack(stream_key, group, msg_id)

    async def process_batch(self, messages: list[tuple[str, dict[str, str]]]) -> None:
        assert self.redis is not None and self.qdrant is not None

        events = [redis_fields_to_event(fields) for _, fields in messages]
        templates = [
            extract_template(e["message"], e.get("source", "unknown")) for e in events
        ]
        tmpl_hashes = [template_hash(t) for t in templates]

        needs_embed: list[tuple[int, str, str]] = []
        dense_vectors: list[list[float] | None] = [None] * len(events)

        for i, (tmpl, th) in enumerate(zip(templates, tmpl_hashes)):
            if await self.is_seen_template(th):
                cached = await self.get_cached_vector(th)
                if cached:
                    dense_vectors[i] = cached
                    self.dedup_hits += 1
                    await self.redis.hincrby("stats:dedup_hits", "count", 1)
                else:
                    needs_embed.append((i, tmpl, th))
            else:
                await self.mark_template_seen(th)
                needs_embed.append((i, tmpl, th))

        if needs_embed:
            unique: dict[str, tuple[str, list[int]]] = {}
            for idx, tmpl, th in needs_embed:
                if th not in unique:
                    unique[th] = (tmpl, [])
                unique[th][1].append(idx)

            texts = [v[0] for v in unique.values()]
            loop = asyncio.get_event_loop()
            t0 = time.perf_counter()
            embedded = await loop.run_in_executor(
                None, self.embed_dense_batch, texts
            )
            embed_ms = (time.perf_counter() - t0) * 1000
            await self.redis.hset(
                "stats:embed_latency",
                str(int(time.time())),
                f"{embed_ms:.2f}",
            )

            for (th, (_, indices)), vec in zip(unique.items(), embedded):
                await self.cache_vector(th, vec)
                for idx in indices:
                    dense_vectors[idx] = vec

        points_by_collection: dict[str, list[PointStruct]] = {}
        for event, tmpl, th, dense in zip(events, templates, tmpl_hashes, dense_vectors):
            if dense is None:
                continue

            ts: datetime = event["timestamp"]
            coll = collection_name(
                self.settings.qdrant_collection_prefix,
                ts,
                self.settings.partition_granularity,
            )
            await self.ensure_collection(coll)

            sparse = self.embed_sparse(event["message"])
            point = PointStruct(
                id=event_id_to_uuid(event["id"]),
                vector={"dense": dense, "sparse": sparse},
                payload={
                    "timestamp": ts.isoformat(),
                    "partition": collection_name(
                        self.settings.qdrant_collection_prefix,
                        ts,
                        self.settings.partition_granularity,
                    ).removeprefix(f"{self.settings.qdrant_collection_prefix}_"),
                    "service": event["service"],
                    "severity": event["severity"],
                    "message": event["message"],
                    "source": event.get("source", "unknown"),
                    "template_hash": th,
                },
            )
            points_by_collection.setdefault(coll, []).append(point)

        for coll, points in points_by_collection.items():
            await self.qdrant.upsert(collection_name=coll, points=points)

        self.total_processed += len(messages)
        await self.redis.set("stats:total_processed", self.total_processed)
        await self.redis.set("stats:dedup_hits_total", self.dedup_hits)

    async def read_batch(
        self,
        stream_key: str,
        group: str,
        *,
        retry_mode: bool = False,
    ) -> list[tuple[str, dict[str, str]]]:
        assert self.redis is not None
        result = await self.redis.xreadgroup(
            groupname=group,
            consumername=self.settings.consumer_name,
            streams={stream_key: ">"},
            count=self.settings.batch_size,
            block=2000 if retry_mode else 5000,
        )
        if not result:
            return []

        now = int(time.time())
        messages: list[tuple[str, dict[str, str]]] = []
        deferred: list[tuple[str, dict[str, str]]] = []

        for _, entries in result:
            for msg_id, fields in entries:
                if retry_mode:
                    retry_after = int(fields.get("retry_after", "0"))
                    if retry_after > now:
                        deferred.append((msg_id, fields))
                        continue
                messages.append((msg_id, fields))

        # Re-enqueue deferred retry messages so they aren't stuck in pending forever.
        for msg_id, fields in deferred:
            await self.redis.xack(stream_key, group, msg_id)
            await self.redis.xadd(stream_key, fields)

        return messages

    async def handle_batch(
        self,
        stream_key: str,
        group: str,
        messages: list[tuple[str, dict[str, str]]],
    ) -> None:
        assert self.redis is not None
        try:
            await self.process_batch(messages)
            for msg_id, _ in messages:
                await self.redis.xack(stream_key, group, msg_id)
        except Exception:
            logger.exception(
                "Batch processing failed (%d messages), enqueueing retry",
                len(messages),
            )
            await self.enqueue_retry(stream_key, group, messages)
            await asyncio.sleep(1)

    async def run(self) -> None:
        await self.connect()
        logger.info("Processor started, waiting for events...")

        while True:
            # Prefer retry queue when it has work.
            retry_batch = await self.read_batch(
                self.settings.retry_stream_key,
                self.settings.retry_consumer_group,
                retry_mode=True,
            )
            if retry_batch:
                await self.handle_batch(
                    self.settings.retry_stream_key,
                    self.settings.retry_consumer_group,
                    retry_batch,
                )
                continue

            batch = await self.read_batch(
                self.settings.redis_stream_key,
                self.settings.consumer_group,
            )
            if not batch:
                continue

            await self.handle_batch(
                self.settings.redis_stream_key,
                self.settings.consumer_group,
                batch,
            )

            if self.total_processed and self.total_processed % 500 == 0:
                ratio = (
                    self.dedup_hits / self.total_processed * 100
                    if self.total_processed
                    else 0
                )
                logger.info(
                    "processed=%d dedup=%.1f%%",
                    self.total_processed,
                    ratio,
                )


async def main() -> None:
    settings = Settings()
    processor = Processor(settings)
    await processor.run()


if __name__ == "__main__":
    asyncio.run(main())
