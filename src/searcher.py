# The query engine. Read the index, matches and ranks

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.db import get_conn
from sentence_transformers import SentenceTransformer
import time


_model = None
_prefix = None

def get_search_model():
    global _model, _prefix
    if _model is None:
        model_name, _prefix = config.MODELS[0]
        _model = SentenceTransformer(model_name)
    return _model, _prefix

def encode_query(query):
    model, prefix = get_search_model()
    return model.encode(prefix + query, normalize_embeddings=True).tolist()

# ---------- 3 search channels ----------

def keyword_search(cur, query, top_k=10, filters=None):
    where, params = build_filters(filters)
    or_query = " | ".join(query.split())
    #print(f"DEBUG keyword query: '{or_query}'")
    try:
        cur.execute(f"""
            SELECT id, doc_number, title, abstract,
                   ts_rank(abstract_tsv, to_tsquery('english', %s)) AS score
            FROM patents
            WHERE abstract_tsv @@ to_tsquery('english', %s) {where}
            ORDER BY score DESC
            LIMIT %s
        """, [or_query, or_query] + params + [top_k])
        results = cur.fetchall()
        #print(f"DEBUG keyword results: {len(results)}")
        return results
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

def hybrid_search(cur, query, top_k=10, filters=None, semantic_weight=0.7):
    """semantic + keyword search, using Reciprocal Rank Fusion (RRF)"""
    sem_results = semantic_search(cur, query, top_k=top_k * 2, filters=filters)
    kw_results = keyword_search(cur, query, top_k=top_k * 2, filters=filters)

    # Reciprocal Rank Fusion
    k = 60  # RRF constant
    scores = {}
    meta = {}

    for rank, (pid, doc_num, title, abstract, score) in enumerate(sem_results):
        scores[pid] = scores.get(pid, 0) + semantic_weight / (k + rank + 1)
        meta[pid] = (doc_num, title, abstract)

    for rank, (pid, doc_num, title, abstract, score) in enumerate(kw_results):
        scores[pid] = scores.get(pid, 0) + (1 - semantic_weight) / (k + rank + 1)
        meta[pid] = (doc_num, title, abstract)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for pid, score in ranked:
        doc_num, title, abstract = meta[pid]
        results.append((pid, doc_num, title, abstract, score))
    return results

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
    patent_info = {"doc_number": doc_number, "title": title, "abstract": abstract, "claims": claims, "classification": classification}

    # using its embedding find similar patents
    cur.execute("""
        SELECT id, doc_number, title, abstract,
               1 - (abstract_embedding <=> %s) AS score
        FROM patents
        WHERE doc_number != %s AND abstract_embedding IS NOT NULL
        ORDER BY abstract_embedding <=> %s
        LIMIT %s
    """, [vec, doc_number, vec, top_k])
    similar = cur.fetchall()
    return patent_info, similar



def search_by_claim(cur, claim_text, top_k=10, filters=None):
    """input claim text, find patents with similar claims"""
    query_vec = encode_query(claim_text)
    where, params = build_filters(filters)
    cur.execute(f"""
        SELECT id, doc_number, title, abstract,
               1 - (claims_embedding <=> %s::vector) AS score
        FROM patents
        WHERE claims_embedding IS NOT NULL {where}
        ORDER BY claims_embedding <=> %s::vector
        LIMIT %s
    """, [query_vec] + params + [query_vec, top_k])
    return cur.fetchall()


def browse_by_classification(cur, prefix_code):
    """input classification prefix, list patents"""
    cur.execute("""
        SELECT doc_number, title, classification
        FROM patents
        WHERE classification LIKE %s
        ORDER BY doc_number
    """, [prefix_code + "%"])
    return cur.fetchall()

