import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.searcher import search, search_by_patent_id, claim_search, browse_by_classification
from src.db import get_conn

def run_demo():
    query = "electric vehicle battery cooling system"

    for mode in ["keyword", "semantic", "hybrid"]:
        results, elapsed = search(query, mode=mode)
        print(f"\n=== {mode.upper()} ({elapsed:.3f}s) ===")
        for pid, doc_num, title, abstract, score in results[:5]:
            print(f"  [{score:.4f}] {doc_num}: {title[:80]}")

    conn = get_conn()
    cur = conn.cursor()

    print("\n=== SIMILAR TO PATENT 20250091384 ===")
    info, similar = search_by_patent_id(cur, "20250091384")
    if info:
        print(f"  Source: {info['title']}")
        for pid, doc_num, title, abstract, score in similar[:5]:
            print(f"  [{score:.4f}] {doc_num}: {title[:80]}")

    print("\n=== CLAIM SEARCH ===")
    claim = "A wheel assembly comprising a hub motor integrated into the wheel rim"
    results = claim_search(cur, claim)
    for pid, doc_num, title, abstract, score in results[:5]:
        print(f"  [{score:.4f}] {doc_num}: {title[:80]}")

    print("\n=== BROWSE B60B ===")
    results = browse_by_classification(cur, "B60B")
    print(f"  {len(results)} patents in B60B*")
    for doc_num, title, cls in results[:5]:
        print(f"  {doc_num}: {title[:60]} [{cls}]")

    print("\n=== HYBRID + FILTER (B60B) ===")
    results, elapsed = search(query, filters={"classification_prefix": "B60B"})
    for pid, doc_num, title, abstract, score in results[:5]:
        print(f"  [{score:.4f}] {doc_num}: {title[:80]}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    run_demo()