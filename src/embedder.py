# Embedding JSON into vector
import sys
from psycopg2.extras import execute_values
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.db import get_conn
from sentence_transformers import SentenceTransformer
import numpy as np

def get_model():
    # config prefix is the query-side instruction; not used for documents here
    model_name, _query_prefix = config.MODELS[0]
    model = SentenceTransformer(model_name)
    return model

def embed_patents():
    model = get_model()
    conn = get_conn()
    cur = conn.cursor()

    # retrieve patents that need embedding
    cur.execute("""
        SELECT id, abstract, claims
        FROM patents
        WHERE abstract_embedding IS NULL
    """)
    rows = cur.fetchall()
    print(f"Generating embeddings for {len(rows)} patents...")

    for i, (patent_id, abstract, claims) in enumerate(rows):
        # abstract embedding (document side: no prefix)
        abstract_vec = None
        if abstract:
            abstract_vec = model.encode(abstract, normalize_embeddings=True)

        # per-claim embeddings; claim1 boost is applied at query time via claim_index = 0
        claim_rows = []
        for idx, claim in enumerate(claims or []):
            if len(claim.strip()) < 10:
                continue
            vec = model.encode(claim, normalize_embeddings=True)
            claim_rows.append((patent_id, idx, claim, vec.tolist()))

        # write abstract vector on the patent row
        cur.execute("""
            UPDATE patents
            SET abstract_embedding = %s::vector
            WHERE id = %s
        """, (
            abstract_vec.tolist() if abstract_vec is not None else None,
            patent_id
        ))

        # idempotent: clear this patent's old claim rows before inserting fresh ones
        cur.execute("DELETE FROM claim_embeddings WHERE patent_id = %s", (patent_id,))
        if claim_rows:
            execute_values(cur, """
                INSERT INTO claim_embeddings (patent_id, claim_index, claim_text, embedding)
                VALUES %s
            """, claim_rows, template="(%s, %s, %s, %s::vector)")

        if (i + 1) % 50 == 0:
            conn.commit()
            print(f"  {i + 1}/{len(rows)} done")

    conn.commit()
    cur.close()
    conn.close()
    print(f"All {len(rows)} patents embedded")

if __name__ == "__main__":
    embed_patents()