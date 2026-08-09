# search_engine/app.py

# pip install fastapi uvicorn
# uvicorn app:app --reload --port 8000

from fastapi import FastAPI, Query
from typing import Optional
from src.searcher import search, search_by_patent_id, claim_search, browse_by_classification
from src.db import get_conn

app = FastAPI(title="Patent Search Engine")

@app.get("/search")
def search_api(
    query: str,
    mode: str = "hybrid",
    top_k: int = 10,
    classification: Optional[str] = None,
    title_keyword: Optional[str] = None,
):
    filters = {}
    if classification:
        filters["classification_prefix"] = classification
    if title_keyword:
        filters["title_keyword"] = title_keyword
    results, elapsed = search(query, mode=mode, top_k=top_k, filters=filters or None)
    return {"results": results, "elapsed": elapsed}

@app.get("/similar/{doc_number}")
def similar_api(doc_number: str, top_k: int = 10):
    conn = get_conn()
    cur = conn.cursor()
    info, similar = search_by_patent_id(cur, doc_number, top_k)
    cur.close()
    conn.close()
    return {"patent": info, "similar": similar}

@app.get("/claim-search")
def claim_search_api(claim_text: str, top_k: int = 10):
    conn = get_conn()
    cur = conn.cursor()
    results = claim_search(cur, claim_text, top_k)
    cur.close()
    conn.close()
    return {"results": results}

@app.get("/browse/{classification_prefix}")
def browse_api(classification_prefix: str, limit: int = 20, offset: int = 0):
    conn = get_conn()
    cur = conn.cursor()
    results, total = browse_by_classification(cur, classification_prefix, limit, offset)
    cur.close()
    conn.close()
    return {"results": results, "total": total, "limit": limit, "offset": offset}


@app.post("/ingest/{doc_number}")
def ingest_api(doc_number: str):
    from src.ingest.scraper import fetch_patent, save_patent
    from src.ingest.loader import load_new_files

    patent = fetch_patent(doc_number)
    if not patent:
        return {"error": "Patent not found on USPTO"}
    save_patent(patent)
    load_new_files()
    return {"message": f"Ingested {doc_number}", "title": patent["title"]}

