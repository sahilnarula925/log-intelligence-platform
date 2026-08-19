"""Stream processor — deduplication, embedding, and Qdrant indexing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [processor] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Strip variable parts from log messages to find structural templates.
# Example: "Alice edited Python: +42 bytes" -> "<user> edited <page>: <delta> bytes"
TEMPLATE_PATTERN = re.compile(
    r"^(.+?) edited (.+?): \+(\d+) bytes$"
)


def extract_template(message: str) -> str:
    match = TEMPLATE_PATTERN.match(message)
    if match:
        return "<user> edited <page>: <delta> bytes"
    return message


def template_hash(template: str) -> str:
    return hashlib.md5(template.encode()).hexdigest()


def event_id_to_uuid(event_id: str) -> str:
    """Qdrant accepts UUID point IDs; derive one from the sha256 event hash."""
    return str(uuid.UUID(event_id[:32]))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "logs"
    redis_stream_key: str = "logs:raw"
    consumer_group: str = "processors"
    consumer_name: str = "processor-1"
    batch_size: int = 50
    embed_batch_size: int = 32
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

    async def connect(self) -> None:
        self.redis = aioredis.from_url(
            self.settings.redis_url, decode_responses=True
        )
        await self.redis.ping()

        self.qdrant = AsyncQdrantClient(url=self.settings.qdrant_url)
        await self.ensure_collection()

        # Load models in a thread to avoid blocking the event loop.
        loop = asyncio.get_event_loop()
        self.dense_model = await loop.run_in_executor(
            None, SentenceTransformer, self.settings.embedding_model
        )
        self.sparse_model = await loop.run_in_executor(
            None,
            partial(SparseTextEmbedding, model_name=self.settings.sparse_model),
        )
        logger.info("Models loaded")

        await self.ensure_consumer_group()

    async def ensure_collection(self) -> None:
        assert self.qdrant is not None
        collections = await self.qdrant.get_collections()
        names = {c.name for c in collections.collections}
        if self.settings.qdrant_collection not in names:
            await self.qdrant.create_collection(
                collection_name=self.settings.qdrant_collection,
                vectors_config={
                    "dense": VectorParams(size=384, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
                },
            )
            logger.info("Created Qdrant collection '%s'", self.settings.qdrant_collection)

    async def ensure_consumer_group(self) -> None:
        assert self.redis is not None
        try:
            await self.redis.xgroup_create(
                self.settings.redis_stream_key,
                self.settings.consumer_group,
                id="0",
                mkstream=True,
            )
            logger.info("Created consumer group '%s'", self.settings.consumer_group)
        except aioredis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def parse_stream_entry(self, fields: dict[str, str]) -> dict[str, Any]:
        return {
            "id": fields["id"],
            "timestamp": datetime.fromisoformat(fields["timestamp"]),
            "service": fields["service"],
            "severity": fields["severity"],
            "message": fields["message"],
            "raw": json.loads(fields.get("raw", "{}")),
        }

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
        vectors = self.dense_model.encode(texts, batch_size=self.settings.embed_batch_size)
        return [v.tolist() for v in vectors]

    def embed_sparse(self, text: str) -> SparseVector:
        assert self.sparse_model is not None
        result = list(self.sparse_model.embed([text]))[0]
        return SparseVector(
            indices=result.indices.tolist(),
            values=result.values.tolist(),
        )

    async def process_batch(self, messages: list[tuple[str, dict[str, str]]]) -> None:
        assert self.redis is not None and self.qdrant is not None

        events = [self.parse_stream_entry(fields) for _, fields in messages]
        templates = [extract_template(e["message"]) for e in events]
        tmpl_hashes = [template_hash(t) for t in templates]

        # Step A — deduplication: identify which templates need fresh embeddings.
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

        # Step B — batch embed unique templates that aren't cached yet.
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

        # Step C — index into Qdrant (sparse regenerated per event even on dedup).
        points: list[PointStruct] = []
        for event, tmpl, th, dense in zip(events, templates, tmpl_hashes, dense_vectors):
            if dense is None:
                continue
            sparse = self.embed_sparse(event["message"])
            ts: datetime = event["timestamp"]
            points.append(
                PointStruct(
                    id=event_id_to_uuid(event["id"]),
                    vector={"dense": dense, "sparse": sparse},
                    payload={
                        "timestamp": ts.isoformat(),
                        "timestamp_hour": ts.hour,
                        "service": event["service"],
                        "severity": event["severity"],
                        "message": event["message"],
                        "template_hash": th,
                    },
                )
            )

        if points:
            await self.qdrant.upsert(
                collection_name=self.settings.qdrant_collection,
                points=points,
            )

        # ACK all messages in the batch.
        for msg_id, _ in messages:
            await self.redis.xack(
                self.settings.redis_stream_key,
                self.settings.consumer_group,
                msg_id,
            )

        self.total_processed += len(messages)
        await self.redis.set("stats:total_processed", self.total_processed)
        await self.redis.set("stats:dedup_hits_total", self.dedup_hits)

    async def read_batch(self) -> list[tuple[str, dict[str, str]]]:
        assert self.redis is not None
        result = await self.redis.xreadgroup(
            groupname=self.settings.consumer_group,
            consumername=self.settings.consumer_name,
            streams={self.settings.redis_stream_key: ">"},
            count=self.settings.batch_size,
            block=5000,
        )
        if not result:
            return []

        messages: list[tuple[str, dict[str, str]]] = []
        for _, entries in result:
            for msg_id, fields in entries:
                messages.append((msg_id, fields))
        return messages

    async def run(self) -> None:
        await self.connect()
        logger.info("Processor started, waiting for events...")

        while True:
            batch = await self.read_batch()
            if not batch:
                continue
            try:
                await self.process_batch(batch)
                if self.total_processed % 500 == 0:
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
            except Exception:
                logger.exception("Batch processing failed")
                await asyncio.sleep(1)


async def main() -> None:
    settings = Settings()
    processor = Processor(settings)
    await processor.run()


if __name__ == "__main__":
    asyncio.run(main())
