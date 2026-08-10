# Part 2 — Implementing the Search Engine at Scale (10M patents)

## Overview & scope

Part 1 is a single machine hybrid search engine (~640 patents) on PostgreSQL + pgvector: keyword (`tsvector`), semantic (abstract embedding), and per-claim vector search, fused with RRF.

Embedding 10M documents becomes a large batch compute job, the vector index no longer fits on one machine, and we must be able to query the status of millions of documents.

---

## System components

```
   ┌───────────────┐     ┌──────────────────┐     ┌───────────────────────────┐
   │ Source data   │ --> │ INGEST           │ --> │ EMBEDDING PIPELINE        │
   │ (bulk dumps / │     │ parse → schema → │     │ job queue → GPU workers   │
   │ provided set) │     │ stage            │     │ bge-base, batched encode  │
   └───────────────┘     └──────────────────┘     └─────────────┬─────────────┘
                                                                 │ vectors written back
   ┌─────────────────────────────────────────────────────────────▼───────────┐
   │ STORAGE + INDEX  (PostgreSQL + pgvector)                                  │
   │   patents: metadata + tsvector + abstract_embedding                       │
   │   claim_embeddings: per-claim vectors                                     │
   │   partitioned by classification section; HNSW + GIN + BTREE per partition │
   │   primary (writes)  ──stream replication──>  read replicas (queries)      │
   └─────────────────────────────────────────────┬───────────────────────────┘
                                                  │ (via PgBouncer pool)
   ┌──────────────────────────────────────────────▼──────────────────────────┐
   │ QUERY SERVICE   FastAPI (stateless) × N pods behind a load balancer       │
   │   keyword + semantic + claim  →  RRF  →  top-k                            │
   └──────────────────────────────────────────────┬──────────────────────────┘
                                                   │
   ┌───────────────────────────────────────────────▼──────────────────────────┐
   │ ORCHESTRATION + MONITORING  queue/scheduler · status table · dashboard     │
   └────────────────────────────────────────────────────────────────────────── ┘
```

1. **Source & ingest.** Bulk patent data is parsed into the Part 1 schema and bulk-staged into `patents` with embedding columns `NULL`.
2. **Embedding pipeline** The expensive, parallelizable step: a job queue feeds batches to GPU workers running bge-base model.
3. **Storage + index** PostgreSQL holds metadata, FTS vectors, and embeddings; partitioned, with read replicas for query load.
4. **Query service.** The Part 1 FastAPI app, made stateless and scaled to N pods behind a load balancer.
5. **Orchestration & monitoring.** Schedules pipelines, tracks per-document status, exposes metrics.

---

## Focus 1 — The embedding pipeline

Embedding 10M patents (abstract + ~15 claims each ≈ **~160M encodes**) is one-time job and the part most different from Part 1's inline loop.

**Design: queue-driven, parallel, restartable.**

1. **Status-driven selection.** Every patent has a `status`
   (`staged → embedding → embedded → indexed`, plus `failed`). The pipeline pulls
   `WHERE status = 'staged'`. Part 1 already does a simple version of this
   (`WHERE abstract_embedding IS NULL OR no claim rows`); at scale it becomes an
   explicit status column so progress is queryable and restarts are trivial.
2. **Batching.** Group ~256 patents per job. bge-base is far more efficient
   encoding batches than one text at a time (GPU throughput).
3. **Worker pool.** N GPU workers pull jobs off the queue independently, encode, write
   `abstract_embedding` + per-claim vectors back, and mark `embedded`. Workers
   are stateless and horizontally scalable.
4. **Idempotent writes.** Re-processing a patent must not duplicate claim rows:
   delete-then-insert per patent or upsert.

---

## Focus 2 — Storage, indexing, and scaling PostgreSQL

### Query Layer

FastAPI is stateless (all state is in Postgres), so scale it by running N pods behind a load balancer (nginx / cloud ALB).

### Search Layer : read scaling

Search is **read-heavy**, PostgreSQL's native tool for this is **streaming replication**: one **primary** (handles writes /
ingestion) plus several **read replicas** (handle queries).
A connection pooler— **PgBouncer** — sits between the app pods and the databases, pooling
connections and distributing read queries across replicas.

### Strage Layer : Data size / write scaling

**Partition first (single-machine).** At 10M rows, one table and one HNSW index
are the bottleneck. Partition `patents` / `claim_embeddings` **by classification section** (CPC A–H): many
queries already filter by classification, so the planner prunes to one partition.
Sub-partition oversized sections by doc-number range. Each partition gets its own HNSW index sized to fit in RAM.

**Then shard (multi-machine) — and Postgres has a limit.** When the
data + indexes no longer fit on one node **PostgreSQL has no native automatic sharding.**
Options:

- **Citus** (distributed Postgres extension): shards tables across nodes while
  keeping the Postgres interface — the "stay in the ecosystem" answer.
- **Application-level sharding**: route by classification / doc-range to separate
  Postgres instances yourself (more control, more code).

---

## Cost breakdown (10M patents)

**One-time: embedding compute (dominant upfront cost)**
- ~10M abstracts + ~10M × ~15 claims ≈ **~160M text encodes**.
- bge-base on one modern GPU: ~1–2k encodes/sec batched → **~300–500 GPU-hours**
  for the full backfill.

**Ongoing: storage**
- Vectors: ~160M × 768 dims × 4 bytes ≈ **~500 GB** raw, ×1.5–2 with HNSW. Metadata + text: a few hundred GB more.
- Total **~1–2 TB** → order **hundreds of $/month** on managed storage.

**Ongoing: query serving**
- Stateless app pods (cheap) + memory-heavy DB nodes to keep HNSW indexes,
  × replicas. Order **$1–3k/month**, dominated by RAM for indexes and how many
  replicas you run.

**Ongoing: incremental ingest**
- ~30k patents/day → negligible vs. the backfill; one GPU worker suffices.

---

## Monitoring & status tracking

- **Status counts:** `SELECT status, COUNT(*) FROM patents GROUP BY status` gives
  instant progress (staged / embedded / indexed / failed).
- **Pipeline metrics:** embed throughput (docs/sec), queue depth, GPU
  utilization, batch failure rate → Prometheus + Grafana dashboard.
- **Query metrics:** p50/p95 latency per channel, error rate, results-returned
  rate.
- **Data-quality checks:** patents with NULL embeddings that shouldn't
  be, orphaned claim rows, partition size skew.

---

## Major challenges at scale

1. **pgvector ceiling.** Single-node HNSW won't hold ~10M+ vectors in RAM
   forever, and un-filtered cross-partition ANN gets slow. The real fix is a sharded vector store.
2. **PostgreSQL has no native sharding.** Cross-machine scaling needs Citus or
   app-level sharding.
3. **Read replicas duplicate the vector index in RAM.** Query throughput scales,
   but memory cost scales with it.
4. **Model is a fixed dependency.** Changing the embedding model means
   re-embedding all 10M (another backfill).
5. **Keyword channel is AND-by-default** May need query rewriting / OR-fallback.
6. **RRF weights & claim-1 boost are untuned constants** At scale they should
   come from a real evaluation set, not hand-picked values.

