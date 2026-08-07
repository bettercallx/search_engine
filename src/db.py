# Build a PostgreSQL database for storing patents and their embeddings

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
        dbname="patent_search",
        user="lixiang",
        host="localhost",
        port=5432
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

                -- vector columns for embeddings, bge-base-en-v1.5(768 deminsion)
                abstract_embedding VECTOR(768),
                claims_embedding VECTOR(768)
            );
        """)
        # create GIN index for full-text search
        cur.execute("CREATE INDEX IF NOT EXISTS idx_abstract_tsv ON patents USING GIN(abstract_tsv);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_title_tsv ON patents USING GIN(title_tsv);")

        # classification index for filtering
        cur.execute("CREATE INDEX IF NOT EXISTS idx_classification ON patents USING BTREE(classification);")
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

# embedding is done, create HNSW indexes
def create_vector_index(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_abstract_vec
            ON patents USING hnsw(abstract_embedding vector_cosine_ops);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_claims_vec
            ON patents USING hnsw(claims_embedding vector_cosine_ops);
        """)
    conn.commit()
    print("HNSW indexes created")

if __name__ == "__main__":
    conn = get_conn()
    create_tables(conn)
    load_patents(conn)
    conn.close()


