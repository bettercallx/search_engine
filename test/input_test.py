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
    query_vec = model.encode(prefix + claim_text, normalize_embeddings=True).tolist()
    where, params = _build_filters(filters)
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


if __name__ == "__main__":
    query = "electric vehicle battery cooling system"

    # 3 search modes
    for mode in ["keyword", "semantic", "hybrid"]:
        results, elapsed = search(query, mode=mode)
        print(f"\n=== {mode.upper()} ({elapsed:.3f}s) ===")
        for pid, doc_num, title, abstract, score in results[:5]:
            print(f"  [{score:.4f}] {doc_num}: {title[:80]}")

    #  using patent ID
    conn = get_conn()
    cur = conn.cursor()
    print("\n=== SIMILAR TO PATENT 20250091384 ===")
    info, similar = search_by_patent_id(cur, "20250091384")
    if info:
        print(f"  Source: {info['title']}")
        for pid, doc_num, title, abstract, score in similar[:5]:
            print(f"  [{score:.4f}] {doc_num}: {title[:80]}")

    # using claim 
    print("\n=== CLAIM SEARCH ===")
    claim = "A wheel assembly comprising a hub motor integrated into the wheel rim"
    results = search_by_claim(cur, claim)
    for pid, doc_num, title, abstract, score in results[:5]:
        print(f"  [{score:.4f}] {doc_num}: {title[:80]}")

    # using classification to browse
    print("\n=== BROWSE B60B ===")
    results = browse_by_classification(cur, "B60B")
    print(f"  {len(results)} patents in B60B*")
    for doc_num, title, cls in results[:5]:
        print(f"  {doc_num}: {title[:60]} [{cls}]")

    cur.close()
    conn.close()