# Build a PostgreSQL database for storing patents and their embeddings
# get_conn() -> create_tables(conn) -> load_patents(conn)

import psycopg2
from psycopg2.extras import execute_values
import json
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

# Connect to the PostgreSQL database
def get_conn():
    return psycopg2.connect(
        dbname=config.DB_NAME,
        user=config.DB_USER,
        host=config.DB_HOST,
        port=config.DB_PORT,
    )

# create tables for patents
def create_tables(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS patents (
                id SERIAL PRIMARY KEY,
                doc_number TEXT UNIQUE,
                title TEXT,
                abstract TEXT,
                claims TEXT[],
                detailed_description TEXT[],
                classification TEXT,
                bibtex TEXT,
                filename TEXT,

                -- indexes for full-text search
                abstract_tsv TSVECTOR,
                title_tsv TSVECTOR,

                -- vector column for abstract, bge-base-en-v1.5 (768 dimensions)
                -- per-claim vectors live in claim_embeddings; claim1 boost is applied
                abstract_embedding VECTOR(768)
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS claim_embeddings (
                id SERIAL PRIMARY KEY,
                patent_id INTEGER REFERENCES patents(id),
                claim_index INTEGER,
                claim_text TEXT,
                embedding VECTOR(768)
            );
        """)
        # create GIN index for full-text search
        cur.execute("CREATE INDEX IF NOT EXISTS idx_abstract_tsv ON patents USING GIN(abstract_tsv);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_title_tsv ON patents USING GIN(title_tsv);")

        # classification index for filtering
        cur.execute("CREATE INDEX IF NOT EXISTS idx_classification ON patents USING BTREE(classification);")

        # look up a patent's claims quickly by patent_id
        cur.execute("CREATE INDEX IF NOT EXISTS idx_claim_patent ON claim_embeddings USING BTREE(patent_id);")
    conn.commit()

# load patents from JSON
def load_patents(conn):
    folder = Path(config.DATA_DIR)
    patents = []
    for file_path in folder.glob("*.json"):
        with open(file_path, "r") as f:
            data = json.load(f)
            for patent in data:
                patents.append((
                    patent.get("doc_number"),
                    patent.get("title", ""),
                    patent.get("abstract", ""),
                    patent.get("claims", []),
                    patent.get("detailed_description", []),
                    patent.get("classification", ""),
                    patent.get("bibtex", ""),
                    patent.get("filename", ""),
                ))

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO patents (doc_number, title, abstract, claims, detailed_description, classification, bibtex, filename)
            VALUES %s
            ON CONFLICT (doc_number) DO NOTHING
        """, patents)

        # generate tsvector for full-text search
        cur.execute("""
            UPDATE patents
            SET abstract_tsv = to_tsvector('english', abstract),
                title_tsv = to_tsvector('english', title);
        """)
    conn.commit()
    print(f"Loaded {len(patents)} patents")

# clear all embeddings so they can be regenerated from scratch
# run this before re-embedding when you change the model or the prefix logic
def reset_embeddings(conn):
    with conn.cursor() as cur:
        cur.execute("TRUNCATE claim_embeddings RESTART IDENTITY;")
        cur.execute("UPDATE patents SET abstract_embedding = NULL;")
    conn.commit()
    print("Embeddings reset")

# embedding is done, create HNSW indexes
def create_vector_index(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_abstract_vec
            ON patents USING hnsw(abstract_embedding vector_cosine_ops);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_claims_vec
            ON claim_embeddings USING hnsw(embedding vector_cosine_ops);
        """)
    conn.commit()
    print("HNSW indexes created")

if __name__ == "__main__":
    conn = get_conn()
    create_tables(conn)
    load_patents(conn)
    conn.close()