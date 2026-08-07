# Embedding JSON into vector
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.db import get_conn
from sentence_transformers import SentenceTransformer
import numpy as np

def get_model():
    model_name, prefix = config.MODELS[0]
    model = SentenceTransformer(model_name)
    return model, prefix

def embed_patents():
    model, prefix = get_model()
    conn = get_conn()
    cur = conn.cursor()

    # retrive patents that need embedding
    cur.execute("""
        SELECT id, abstract, claims
        FROM patents
        WHERE abstract_embedding IS NULL
    """)
    rows = cur.fetchall()
    print(f"Generating embeddings for {len(rows)} patents...")

    for i, (patent_id, abstract, claims) in enumerate(rows):
        # abstract embedding
        abstract_text = prefix + abstract if abstract else ""
        abstract_vec = model.encode(abstract_text, normalize_embeddings=True)

        # claims embedding: join all claims into one string
        claims_text = ""
        if claims:
            claims_text = prefix + " ".join(claims)
        claims_vec = model.encode(claims_text, normalize_embeddings=True)

        cur.execute("""
            UPDATE patents
            SET abstract_embedding = %s::vector,
                claims_embedding = %s::vector
            WHERE id = %s
        """, (abstract_vec.tolist(), claims_vec.tolist(), patent_id))

        if (i + 1) % 50 == 0:
            conn.commit()
            print(f"  {i + 1}/{len(rows)} done")

    conn.commit()
    cur.close()
    conn.close()
    print(f"All {len(rows)} patents embedded")

if __name__ == "__main__":
    embed_patents()