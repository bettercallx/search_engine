# Patent Search Engine

A hybrid search engine over patent documents, built on PostgreSQL + pgvector.
It combines 3 retrieval signals: full-text keyword search, abstract vector search, and per-claim vector search, and uses them with Reciprocal Rank Fusion.
It also supports "more like this" similarity by patent number, classification browsing, and online ingestion of new patents from Google Patents.

## Project structure

```
search_engine/
├── app.py                  # FastAPI app (API endpoints)
├── config.py               # paths, model, DB connection, RRF weights
├── run.sh                  # setup + embed + index + serve
├── requirements.txt
├── src/
│   ├── db.py               # schema, load, reset, index creation
│   ├── embedder.py         # generate abstract + per-claim embeddings
│   ├── searcher.py         # keyword / semantic / claim / hybrid search
│   └── ingest/
│       ├── scraper.py      # fetch a patent from Google Patents
│       └── loader.py       # insert scraped patents, then embed
├── tools/
│   ├── demo.py             # end-to-end demo of all search modes
│   ├── benchmark.py        # timing + relevance + strategy comparison
│   └── patent_stats.py     # dataset statistics
├── patent_data_small/      # source dataset (provided JSON)
└── new_patent_data/        # online-ingested patents (git-ignored)
```

---

## Architecture

**Two-table design.**

- `patents` — one row per patent (title, abstract, claims[], classification,
  full-text `tsvector` columns, and a single `abstract_embedding VECTOR(768)`).
- `claim_embeddings` — one row *per claim* (`patent_id`, `claim_index`,
  `claim_text`, `embedding VECTOR(768)`).

Claims are embedded individually rather than concatenated, so each independent claim maps to its own vector.

This keeps every claim under the embedding model's token limit and lets the search layer reason about individual claims (e.g. boosting claim 1).

**Retrieval channels.**

| Channel   | Source                          | Notes                                        |
|-----------|---------------------------------|----------------------------------------------|
| keyword   | `patents.abstract_tsv` (GIN)    | `websearch_to_tsquery`                        |
| semantic  | `patents.abstract_embedding`    | cosine (`<=>`) over HNSW index                |
| claim     | `claim_embeddings.embedding`    | best boosted claim per patent (HNSW)          |

**Hybrid = keyword + semantic + claim, fused with RRF.**
Each channel ranks candidates independently; RRF combines them by rank (`weight / (k + rank)`). Channel weights and the RRF constant live in `config.py`.

**Embedding model:** `BAAI/bge-base-en-v1.5` (768-dim).
The query-side instruction prefix is applied only to queries(the BGE authors' guidance for retrieval).

---

## Setup

### 1. System dependencies (not installable via pip)

**PostgreSQL 16:**
```bash
brew install postgresql@16
brew services start postgresql@16
```

**pgvector extension:**
```bash
brew install pgvector
```

if brew version doesn't match

```bash
cd /tmp
git clone https://github.com/pgvector/pgvector.git
cd pgvector
export PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config
make
make install
```

### 2. Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
(Python 3.9)

### 3. Database connection

Connection parameters are read from environment variables (with sensible
defaults). Set at least the user to match your local PostgreSQL:

```bash
export PATENT_DB_USER=your_pg_username     # default: postgres
export PATENT_DB_NAME=patent_search        # default: patent_search
export PATENT_DB_HOST=localhost            # default: localhost
export PATENT_DB_PORT=5432                 # default: 5432
```

### 4. Data

Place the provided patent JSON files in `patent_data_small/`. The loader reads every `*.json` in that directory.

Patents ingested online (via `/ingest`) are written separately to `new_patent_data/`.

---

## Running

`run.sh` sets up the database, generates embeddings, builds vector indexes, and starts the API server:

```bash
./run.sh              # normal run (embeds only what's missing)
./run.sh --reset      # drop & recreate the database, then run
./run.sh --reembed    # keep the DB, wipe & regenerate all embeddings, use after changing the model or prefix logic
```

The server starts on http://127.0.0.1:8000.

Interactive API docs (Swagger UI) are at http://127.0.0.1:8000/docs to try each endpoint.

---

## API

| Method | Endpoint                          | Description                                   |
|--------|-----------------------------------|-----------------------------------------------|
| GET    | `/search`                         | keyword / semantic / claim / hybrid / rerank  |
| GET    | `/similar/{doc_number}`           | patents similar to a given one                 |
| GET    | `/patent/{doc_number}`            | full detail of one patent (abstract + claims)  |
| GET    | `/claim-search`                   | find patents by similar claim text             |
| GET    | `/browse/{classification_prefix}` | browse a classification (paginated)            |
| POST   | `/ingest/{doc_number}`            | fetch a patent from Google Patents and add it  |

**`/search` parameters:** `query` (required), `mode` (`hybrid` default / `rerank` /
`keyword` / `semantic` / `claim`), `top_k`, `classification`, `title_keyword`.

**`/browse` parameters:** `limit` (default 20), `offset` (default 0). Returns
`results`, `total`, `limit`, `offset`.

Examples:
```
GET /search?query=tire pressure monitoring&mode=hybrid&top_k=5
GET /search?query=tire pressure monitoring&mode=rerank&top_k=10
GET /similar/20240059096
GET /claim-search?claim_text=a rim body formed in a cylindrical shape&top_k=10
GET /browse/B60B?limit=20&offset=0
POST /ingest/12146706
```

---

## Hybrid search performance (timing with vs. without hybrid)

Same query (`tire pressure monitoring`), steady state (model already loaded),
~640 patents:

| mode      | what runs                                  | time    |
|-----------|--------------------------------------------|---------|
| keyword   | one SQL full-text query (GIN index)        | ~0.010s |
| semantic  | encode query + one vector search (HNSW)    | ~0.169s |
| hybrid    | keyword + semantic + claim, fused with RRF | ~0.107s |

**Observations.**

- **Keyword is the cheapest** — it's a single `tsvector` lookup on a GIN index.
- **The vector-based modes cost more** because each must encode the query with
  bge-base and do an ANN search; that per-query encode dominates at this scale.

At 10M patents, filtering `classification LIKE 'B60B%'` first can cut candidates from
millions to thousands, so the ANN search runs over a tiny fraction of the corpus
instead of all of it. Concretely, the levers are:

- **Filter-first / predicate pushdown** — apply classification & keyword
  constraints in SQL to shrink the set the vector index must consider.
- **Partitioning by classification** (see `Part2.md`) so a filtered query hits a
  single partition's smaller index.
- **ANN indexes (HNSW)** instead of exact search, trading a little recall for
  large speedups.
- **Parallel channels** — the three channels are independent and can run
  concurrently, so hybrid latency approaches the slowest single channel rather
  than their sum.

So hybrid is a bit more expensive on a tiny in-memory dataset, but the same
structure becomes a performance *advantage* once the corpus is large enough that
narrowing the search space matters.

---

## Design decisions

- **Per-claim embeddings, not concatenated.** Concatenating all claims into one
  vector would exceed the model's ~512-token limit (silently truncating later
  claims) and blur distinct claims together. Storing one vector per claim keeps
  each within limits and enables claim-level ranking.

- **Claim 1 boosted at query time, not baked into the vector.** The independent
  claim (`claim_index = 0`) defines the broadest scope and matters most, so the
  claim channel multiplies its similarity by a boost factor. This is applied at
  *query* time (in SQL) rather than stored, because embeddings are L2-normalized
  and cosine ignores magnitude — scaling a stored vector would have no effect.
  Keeping the boost in the query also means it can be tuned without re-embedding.

- **RRF instead of score blending.** Keyword (`ts_rank`), semantic (cosine), and
  claim scores live on incompatible scales. RRF fuses by rank, sidestepping
  score normalization, and all weights sit in `config.py` for easy tuning.

- **Query-only instruction prefix.** BGE v1.5 expects its instruction prefix on
  queries only; documents are embedded raw. Applying it to documents lowers
  retrieval quality.

- **Paginated browse.** Browsing a classification returns a page (`limit` /
  `offset`) plus a total count, rather than dumping every match — a single broad
  prefix can match hundreds of patents (and far more at scale).

- **Loader reuses the embedder.** New patents are only *inserted* by the loader;
  embedding is delegated to `embedder.embed_patents()`, so there is a single
  source of truth for embedding logic.

---

## Known limitations

- **Online ingest produces data.** Patents fetched from Google
  Patents lack fields the source XML has (`detailed_description`, `bibtex` come
  back empty). Classification codes use the slashed CPC form (`B60B7/0013`)
  rather than the dataset's compact form (`B60B704FI`), and the deepest code is
  taken as the single class — which may be an associated rather than the primary
  classification. Very recent publications may not be indexed by Google Patents
  yet, in which case fetch returns nothing.

- **claim-channel scores can exceed 1.0.** Because claim 1 is multiplied by a
  boost, its raw score may go above 1.0. This is harmless for `hybrid` (RRF uses
  rank, not the raw value) but makes the standalone `/claim-search` score less
  directly interpretable.

- **The loader does not upsert.** Ingestion skips a `doc_number` that already
  exists; it does not update it. To re-ingest a patent (e.g. after fixing the
  scraper), delete its rows first (from `patents` and `claim_embeddings`).

- **Keyword search is AND-by-default.** `websearch_to_tsquery` ANDs terms, so
  multi-word queries can return nothing if no single abstract contains all of
  them. Hybrid mode compensates via the semantic and claim channels.

---

## Part 2 & Part 3

### Part 2 — Scale architecture
See [`Part2.md`](Part2.md) for the design of running this engine at ~10M patents plus a containerization (`Dockerfile` + `docker-compose.yml`).

### Part 3 — Enhancement: two-phase search (cross-encoder reranking)

**Enhancement chosen:** two-phase searching (reranking).

**Why:** Part 1's semantic/hybrid retrieval uses a *bi-encoder* (bge-base) — the
query and each document are embedded independently and compared by cosine. It's
fast but never lets the query and document interact.
A *cross-encoder* scores the query and document
*jointly* in one pass, which is far more accurate but too slow to run over the
whole corpus. The standard fix is two-phase: use the cheap bi-encoder to recall a
candidate pool, then the expensive cross-encoder to rerank it.

**How it's implemented:** `mode=rerank` runs the existing hybrid pipeline to
recall the top ~50 candidates (phase 1), then scores each `(query, title +
abstract)` pair with `BAAI/bge-reranker-base` (phase 2) and returns the top-k by
cross-encoder score. It reuses the existing candidate pipeline and returns the
same result shape, so the API and UI needed no changes beyond one dropdown
option. The reranker model is loaded lazily on first use.

**Result:** on `tire pressure monitoring`, rerank surfaces markedly more precise
hits than hybrid, with scores tightly clustered near the top (0.99+) tapering off
— a much sharper relevance gradient than the RRF fusion scores. Per the brief the
goal was to build the pipeline rather than chase a metric; the quality
improvement is a visible bonus.

**To reproduce:** run Part 1 as below, then query `mode=rerank`:
```
GET /search?query=tire pressure monitoring&mode=rerank&top_k=10
```
(or pick `rerank` in the UI dropdown). First call downloads the reranker model.
