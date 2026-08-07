import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import time
from src.searcher import search, search_by_claim, get_search_model
from src.db import get_conn

QUERIES = [
    "electric vehicle battery cooling system",
    "autonomous driving lidar sensor",
    "wheel hub motor assembly",
    "brake pad wear detection method",
    "tire pressure monitoring wireless",
]

# mannual annotation:query -> top5 doc_number
GROUND_TRUTH = {
    "wheel hub motor assembly": ["20250083473", "20240198724", "20240109366"],
    "tire pressure monitoring wireless": ["20240083199"],
    "brake pad wear detection method": [],  # uncertain, leave empty
}

def benchmark_search_modes():
    """compare keyword / semantic / hybrid time and hits"""
    print("=" * 70)
    print("BENCHMARK: Search Mode Comparison")
    print("=" * 70)

    for query in QUERIES:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        for mode in ["keyword", "semantic", "hybrid"]:
            results, elapsed = search(query, mode=mode)
            hit_count = len(results)
            top_title = results[0][2][:60] if results else "NO RESULTS"
            print(f"  {mode:10s} | {elapsed:.3f}s | {hit_count:2d} hits | top: {top_title}")

def benchmark_filter_impact():
    """compare the impact of using filters on performance"""
    print("\n" + "=" * 70)
    print("BENCHMARK: Filter Impact on Performance")
    print("=" * 70)

    query = "electric vehicle battery cooling system"

    # no filter
    times_no_filter = []
    for _ in range(5):
        _, elapsed = search(query, mode="hybrid")
        times_no_filter.append(elapsed)

    # with filter
    times_with_filter = []
    for _ in range(5):
        _, elapsed = search(query, mode="hybrid", filters={"classification_prefix": "B60B"})
        times_with_filter.append(elapsed)

    avg_no = sum(times_no_filter) / len(times_no_filter)
    avg_with = sum(times_with_filter) / len(times_with_filter)

    print(f"\n  Hybrid (no filter):   avg {avg_no:.4f}s  runs: {[f'{t:.3f}' for t in times_no_filter]}")
    print(f"  Hybrid (B60B filter): avg {avg_with:.4f}s  runs: {[f'{t:.3f}' for t in times_with_filter]}")
    print(f"  Speedup: {avg_no/avg_with:.2f}x" if avg_with > 0 else "")

    print(f"""
  Commentary:
  - At 640 patents, filter overhead is negligible because PostgreSQL
    scans the full table either way (seq scan beats index at this scale).
  - At 10M patents, filtering by classification BEFORE vector search
    drastically reduces ANN comparisons. A partial index or partitioned
    table by classification prefix would make this even faster.
  - Hybrid search (keyword + semantic + RRF merge) costs roughly 2x
    a single-mode search since it runs both channels. This is acceptable
    because the two channels catch different types of relevance.
""")

def benchmark_relevance():
    """检查 top5 结果是否包含已知相关专利"""
    print("=" * 70)
    print("BENCHMARK: Relevance Check (ground truth)")
    print("=" * 70)

    for query, expected_docs in GROUND_TRUTH.items():
        if not expected_docs:
            continue
        results, elapsed = search(query, mode="hybrid")
        top_docs = [r[1] for r in results[:5]]
        hits = [d for d in expected_docs if d in top_docs]
        recall = len(hits) / len(expected_docs) if expected_docs else 0
        print(f"\n  Query: '{query}'")
        print(f"  Expected: {expected_docs}")
        print(f"  Got top5: {top_docs}")
        print(f"  Recall@5: {recall:.0%} ({len(hits)}/{len(expected_docs)})")


def benchmark_embedding_strategies():
    """compare different embedding strategies' retrieval quality and speed"""
    print("=" * 70)
    print("BENCHMARK: Embedding Strategy Comparison")
    print("=" * 70)

    conn = get_conn()
    cur = conn.cursor()
    model, prefix = get_search_model()

    query = "wheel hub motor integrated into rim"
    query_vec = model.encode(prefix + query, normalize_embeddings=True).tolist()

    # Strategy 1: abstract embedding
    print(f"\nQuery: '{query}'")
    print("-" * 50)

    start = time.time()
    cur.execute("""
        SELECT doc_number, title,
               1 - (abstract_embedding <=> %s::vector) AS score
        FROM patents
        WHERE abstract_embedding IS NOT NULL
        ORDER BY abstract_embedding <=> %s::vector
        LIMIT 5
    """, [query_vec, query_vec])
    results_abstract = cur.fetchall()
    t_abstract = time.time() - start

    print(f"\n  Strategy: Abstract embedding ({t_abstract:.3f}s)")
    for doc_num, title, score in results_abstract:
        print(f"    [{score:.4f}] {doc_num}: {title[:65]}")

    # Strategy 2: claims embedding
    start = time.time()
    cur.execute("""
        SELECT doc_number, title,
               1 - (claims_embedding <=> %s::vector) AS score
        FROM patents
        WHERE claims_embedding IS NOT NULL
        ORDER BY claims_embedding <=> %s::vector
        LIMIT 5
    """, [query_vec, query_vec])
    results_claims = cur.fetchall()
    t_claims = time.time() - start

    print(f"\n  Strategy: Claims embedding ({t_claims:.3f}s)")
    for doc_num, title, score in results_claims:
        print(f"    [{score:.4f}] {doc_num}: {title[:65]}")

    # Strategy 3: weighted combination (abstract 0.6 + claims 0.4)
    start = time.time()
    cur.execute("""
        SELECT doc_number, title,
               0.6 * (1 - (abstract_embedding <=> %s::vector)) +
               0.4 * (1 - (claims_embedding <=> %s::vector)) AS score
        FROM patents
        WHERE abstract_embedding IS NOT NULL AND claims_embedding IS NOT NULL
        ORDER BY score DESC
        LIMIT 5
    """, [query_vec, query_vec])
    results_weighted = cur.fetchall()
    t_weighted = time.time() - start

    print(f"\n  Strategy: Weighted (0.6*abstract + 0.4*claims) ({t_weighted:.3f}s)")
    for doc_num, title, score in results_weighted:
        print(f"    [{score:.4f}] {doc_num}: {title[:65]}")

    # Strategy 4: per-claim search (find which individual claim matches best)
    print(f"\n  Strategy: Per-claim matching")
    start = time.time()
    cur.execute("SELECT doc_number, title, claims FROM patents LIMIT 50")
    rows = cur.fetchall()

    claim_results = []
    for doc_num, title, claims in rows:
        if not claims:
            continue
        for i, claim in enumerate(claims):
            if len(claim.strip()) < 10:
                continue
            claim_vec = model.encode(prefix + claim, normalize_embeddings=True).tolist()
            score = sum(a * b for a, b in zip(query_vec, claim_vec))
            claim_results.append((doc_num, title, i + 1, claim[:80], score))

    claim_results.sort(key=lambda x: x[4], reverse=True)
    t_perclaim = time.time() - start

    print(f"  ({t_perclaim:.3f}s, searched {len(claim_results)} individual claims)")
    for doc_num, title, claim_num, claim_text, score in claim_results[:5]:
        print(f"    [{score:.4f}] {doc_num} claim#{claim_num}: {claim_text}")

    print(f"""
  Commentary:
  - Abstract embedding captures high-level topic similarity.
  - Claims embedding captures legal/technical scope overlap.
  - Weighted combination balances both signals.
  - Per-claim matching is most precise for finding specific overlapping
    claims but is O(n*m) and too slow for large-scale retrieval.
    It's best used as a reranking step on top-k candidates.
""")

    cur.close()
    conn.close()

if __name__ == "__main__":
    benchmark_search_modes()
    benchmark_filter_impact()
    benchmark_relevance()
    benchmark_embedding_strategies()