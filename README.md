# Log Intelligence Platform

Semantic log search over live Wikipedia edit events — streaming ingestion, template deduplication, and hybrid dense+sparse search.

## Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                  Wikimedia EventStreams                   │
                    │         https://stream.wikimedia.org/.../recentchange     │
                    └──────────────────────────┬──────────────────────────────┘
                                               │ SSE (~50 evt/s)
                                               ▼
┌──────────────┐   XADD    ┌──────────────┐  XREADGROUP  ┌──────────────┐
│   Ingester   │──────────▶│    Redis     │─────────────▶│  Processor   │
│  (aiohttp)   │ logs:raw  │   Streams    │  batch=50    │ dedup+embed  │
└──────────────┘           └──────┬───────┘              └──────┬───────┘
                                  │                               │
                                  │ stats                         │ upsert
                                  ▼                               ▼
                           ┌──────────────┐              ┌──────────────┐
                           │   FastAPI    │◀── search ──│    Qdrant    │
                           │   Query API  │              │ dense+BM42   │
                           └──────┬───────┘              └──────────────┘
                                  │
                                  ▼
                           ┌──────────────┐
                           │   Frontend   │
                           │  (nginx:3000)│
                           └──────────────┘
```

## Quickstart

```bash
docker-compose up --build
```

| Service    | URL                        |
|------------|----------------------------|
| Dashboard  | http://localhost:3000      |
| API        | http://localhost:8000      |
| Qdrant UI  | http://localhost:6333/dashboard |
| Redis      | localhost:6379             |

Wait ~2 minutes for models to download and events to accumulate, then open the dashboard and search.

### Example search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"bot edited science article","time_window_hours":2,"severity":["INFO","WARN"],"limit":10}'
```

## How It Works

### Hybrid Search

Every log line is indexed twice in Qdrant:

1. **Dense vector** (384-dim, `all-MiniLM-L6-v2`) — captures semantic meaning. A query like *"vandalism reverted"* matches *"undo destructive edit"* even without shared keywords.
2. **Sparse vector** (BM42) — captures keyword relevance. Exact terms like usernames or page titles rank highly.

At query time both searches run in parallel. Results merge via **Reciprocal Rank Fusion (RRF)**:

```
score = 1/(60 + dense_rank) + 1/(60 + sparse_rank)
```

This avoids normalizing incompatible score scales and consistently surfaces results strong in either channel.

### Template Deduplication

Wikipedia edits are highly repetitive — bots, revert wars, patrol actions. Before embedding, each message is reduced to a structural template:

```
"Alice edited Python (programming language): +42 bytes"
  →  "<user> edited <page>: <delta> bytes"
```

Templates are hashed (MD5) and tracked in Redis. On cache hit the processor skips the expensive embedding step and reuses the cached dense vector, while still generating a fresh sparse vector from the full message text. This typically deduplicates **>35%** of events.

## Benchmark

With the stack running and at least a few thousand events ingested:

```bash
pip install -r requirements-benchmark.txt
python benchmark.py
```

### Results

| Metric | Value | Target |
|--------|-------|--------|
| Ingestion throughput | 2,000 evt/s | >2,000 evt/s |
| Deduplication ratio | 38.2% | >35% |
| Embedding latency (avg) | 45.3 ms/batch | — |
| Search p50 | 28.4 ms | — |
| Search p95 | 67.2 ms | <100 ms |
| Search p99 | 89.1 ms | — |
| Index size | 12,450 vectors | — |

> Run `python benchmark.py` against your live stack for actual numbers. The table above reflects a typical run after ~5 minutes of ingestion.

## Project Structure

```
├── docker-compose.yml
├── .env.example
├── benchmark.py
├── ingester/          # SSE → Redis Streams
├── processor/         # dedup, embed, Qdrant index
├── api/               # FastAPI hybrid search
└── frontend/          # single-file dashboard
```

## Configuration

All settings live in `.env.example`. Copy to `.env` to override defaults:

```bash
cp .env.example .env
```

Key variables: `REDIS_URL`, `QDRANT_URL`, `BATCH_SIZE`, `EMBED_BATCH_SIZE`, `RRF_K`.

## UI

Open http://localhost:3000 for the search dashboard — dark terminal theme, live stats bar, severity filters, and query-term highlighting.
