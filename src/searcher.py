# The query engine. Read the index, matches and ranks

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.db import get_conn
from sentence_transformers import SentenceTransformer, CrossEncoder
import time


_model = None
_prefix = None
_reranker = None

def get_search_model():
    global _model, _prefix
    if _model is None:
        model_name, _prefix = config.MODELS[0]
        _model = SentenceTransformer(model_name)
    return _model, _prefix

def get_reranker():
    # cross-encoder for two-phase reranking; loaded lazily on first use
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANKER_MODEL)
    return _reranker

def encode_query(query):
    # prepend the BGE instruction
    model, prefix = get_search_model()
    return model.encode(prefix + query, normalize_embeddings=True).tolist()

# ---------- search channels ----------

def keyword_search(cur, query, top_k=10, filters=None):
    where, params = build_filters(filters)
    # websearch_to_tsquery parses the raw query itself 
    # (handles quotes, OR, -, stray punctuation) 
    try:
        cur.execute(f"""
            SELECT id, doc_number, title, abstract,
                   ts_rank(abstract_tsv, websearch_to_tsquery('english', %s)) AS score
            FROM patents
            WHERE abstract_tsv @@ websearch_to_tsquery('english', %s) {where}
            ORDER BY score DESC
            LIMIT %s
        """, [query, query] + params + [top_k])
        return cur.fetchall()
    except Exception as e:
        print(f"DEBUG keyword error: {e}")
        conn = cur.connection
        conn.rollback()
        return []

def semantic_search(cur, query, top_k=10, filters=None):
    """abstract embedding vector search"""
    query_vec = encode_query(query)
    where, params = build_filters(filters)
    cur.execute(f"""
        SELECT id, doc_number, title, abstract,
               1 - (abstract_embedding <=> %s::vector) AS score
        FROM patents
        WHERE abstract_embedding IS NOT NULL {where}
        ORDER BY abstract_embedding <=> %s::vector
        LIMIT %s
    """, [query_vec] + params + [query_vec, top_k])
    return cur.fetchall()

def claim_search(cur, query, top_k=10, filters=None, claim1_boost=None):
    """per-claim vector search; claim1 (claim_index = 0) is boosted at query time.
    Scores per patent = best boosted claim similarity."""
    query_vec = encode_query(query)
    where, params = build_filters(filters)  # clauses start with "AND ", patents columns
    if claim1_boost is None:
        claim1_boost = config.CLAIM1_BOOST
    cur.execute(f"""
        SELECT p.id, p.doc_number, p.title, p.abstract,
               MAX((1 - (ce.embedding <=> %s::vector))
                   * CASE WHEN ce.claim_index = 0 THEN %s ELSE 1.0 END) AS score
        FROM claim_embeddings ce
        JOIN patents p ON p.id = ce.patent_id
        WHERE TRUE {where}
        GROUP BY p.id, p.doc_number, p.title, p.abstract
        ORDER BY score DESC
        LIMIT %s
    """, [query_vec, claim1_boost] + params + [top_k])
    return cur.fetchall()


def hybrid_search(cur, query, top_k=10, filters=None,
                  weights=None, claim1_boost=None):
    """semantic + keyword + claim search, fused with Reciprocal Rank Fusion (RRF)"""
    if weights is None:
        weights = config.RRF_WEIGHTS
    if claim1_boost is None:
        claim1_boost = config.CLAIM1_BOOST

    sem_results = semantic_search(cur, query, top_k=top_k * 2, filters=filters)
    kw_results = keyword_search(cur, query, top_k=top_k * 2, filters=filters)
    claim_results = claim_search(cur, query, top_k=top_k * 2, filters=filters,
                                 claim1_boost=claim1_boost)

    # Reciprocal Rank Fusion: each channel contributes weight / (k + rank)
    k = config.RRF_K  
    scores = {}
    meta = {}

    def fuse(results, weight):
        for rank, (pid, doc_num, title, abstract, _score) in enumerate(results):
            scores[pid] = scores.get(pid, 0) + weight / (k + rank + 1)
            meta[pid] = (doc_num, title, abstract)

    fuse(sem_results, weights["semantic"])
    fuse(kw_results, weights["keyword"])
    fuse(claim_results, weights["claim"])

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for pid, score in ranked:
        doc_num, title, abstract = meta[pid]
        results.append((pid, doc_num, title, abstract, score))
    return results


def rerank_search(cur, query, top_k=10, filters=None, candidate_n=50):
    """Two-phase: hybrid recall (candidate_n) -> cross-encoder rerank -> top_k.
    Phase 1 (bi-encoder hybrid) is cheap and casts a wide net; phase 2
    (cross-encoder) is expensive but accurate, scoring each (query, doc) pair
    jointly instead of comparing independent embeddings."""
    # phase 1: recall a wider candidate pool with the existing hybrid pipeline
    candidates = hybrid_search(cur, query, top_k=candidate_n, filters=filters)
    if not candidates:
        return []

    # phase 2: cross-encoder scores each (query, title+abstract) pair
    reranker = get_reranker()
    pairs = [(query, f"{c[2]} {c[3]}") for c in candidates]  # c[2]=title, c[3]=abstract
    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    # return same 5-tuple shape as other channels, with the rerank score
    return [(c[0], c[1], c[2], c[3], float(s)) for c, s in ranked[:top_k]]


# ---------- Filter Building ----------

def build_filters(filters):
    """ WHERE clauses and parameters for SQL query based on filters """
    if not filters:
        return "", []
    clauses = []
    params = []
    if "classification_prefix" in filters:
        clauses.append("AND classification LIKE %s")
        params.append(filters["classification_prefix"] + "%")
    if "title_keyword" in filters:
        clauses.append("AND title_tsv @@ plainto_tsquery('english', %s)")
        params.append(filters["title_keyword"])
    if "title_exact" in filters:
        clauses.append("AND LOWER(title) = LOWER(%s)")
        params.append(filters["title_exact"])
    return " ".join(clauses), params

# ---------- Entry Point ----------

def search(query, mode="hybrid", top_k=None, filters=None):
    if top_k is None:
        top_k = config.TOP_K
    conn = get_conn()
    cur = conn.cursor()

    start = time.time()
    if mode == "keyword":
        results = keyword_search(cur, query, top_k, filters)
    elif mode == "semantic":
        results = semantic_search(cur, query, top_k, filters)
    elif mode == "claim":
        results = claim_search(cur, query, top_k, filters)
    elif mode == "rerank":
        results = rerank_search(cur, query, top_k, filters)
    else:
        results = hybrid_search(cur, query, top_k, filters)
    elapsed = time.time() - start

    cur.close()
    conn.close()
    return results, elapsed


def search_by_patent_id(cur, doc_number, top_k=10):
    """input patent ID, find similar patents"""
    cur.execute("""
        SELECT id, abstract_embedding, title, abstract, claims, classification
        FROM patents WHERE doc_number = %s
    """, [doc_number])
    row = cur.fetchone()
    if not row:
        return None, []
    pid, vec, title, abstract, claims, classification = row
    patent_info = {"doc_number": doc_number, "title": title, "abstract": abstract,
                   "claims": claims, "classification": classification}

    # using its embedding find similar patents (vec comes back as text -> cast it)
    cur.execute("""
        SELECT id, doc_number, title, abstract,
               1 - (abstract_embedding <=> %s::vector) AS score
        FROM patents
        WHERE doc_number != %s AND abstract_embedding IS NOT NULL
        ORDER BY abstract_embedding <=> %s::vector
        LIMIT %s
    """, [vec, doc_number, vec, top_k])
    similar = cur.fetchall()
    return patent_info, similar

def get_patent_detail(cur, doc_number):
    """input patent ID, return its full detail (no similarity search)"""
    cur.execute("""
        SELECT doc_number, title, abstract, claims, classification
        FROM patents WHERE doc_number = %s
    """, [doc_number])
    row = cur.fetchone()
    if not row:
        return None
    doc, title, abstract, claims, classification = row
    return {
        "doc_number": doc,
        "title": title,
        "abstract": abstract,
        "claims": claims,
        "classification": classification,
    }

def browse_by_classification(cur, prefix_code, limit=20, offset=0):
    cur.execute("""
        SELECT doc_number, title, classification
        FROM patents
        WHERE classification LIKE %s
        ORDER BY doc_number
        LIMIT %s OFFSET %s
    """, [prefix_code + "%", limit, offset])
    rows = cur.fetchall()

    # return total count for pagination
    cur.execute("SELECT COUNT(*) FROM patents WHERE classification LIKE %s",
                [prefix_code + "%"])
    total = cur.fetchone()[0]
    return rows, total
