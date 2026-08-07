
# src/update.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import config
from src.db import get_conn
from src.embedder import get_model

def update_from_new_files():
    folder = Path(config.DATA_DIR)
    new_files = list(folder.glob("new_*.json"))
    if not new_files:
        print("No new files found")
        return

    conn = get_conn()
    cur = conn.cursor()
    model, prefix = get_model()
    count = 0

    for file_path in new_files:
        with open(file_path, "r") as f:
            patents = json.load(f)

        for patent in patents:
            doc_number = patent.get("doc_number")

            # Skip if already exists
            cur.execute("SELECT id FROM patents WHERE doc_number = %s", [doc_number])
            if cur.fetchone():
                print(f"  Skipped {doc_number} (already exists)")
                continue

            # Insert new patent
            cur.execute("""
                INSERT INTO patents (doc_number, title, abstract, claims, detailed_description, classification, bibtex, filename, abstract_tsv, title_tsv)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, to_tsvector('english', %s), to_tsvector('english', %s))
            """, (
                doc_number,
                patent.get("title", ""),
                patent.get("abstract", ""),
                patent.get("claims", []),
                patent.get("detailed_description", []),
                patent.get("classification", ""),
                patent.get("bibtex", ""),
                patent.get("filename", ""),
                patent.get("abstract", ""),
                patent.get("title", ""),
            ))

            # generate embedding
            abstract_vec = model.encode(prefix + patent.get("abstract", ""), normalize_embeddings=True).tolist()
            claims_text = " ".join(patent.get("claims", []))
            claims_vec = model.encode(prefix + claims_text, normalize_embeddings=True).tolist()

            cur.execute("""
                UPDATE patents
                SET abstract_embedding = %s::vector, claims_embedding = %s::vector
                WHERE doc_number = %s
            """, [abstract_vec, claims_vec, doc_number])

            count += 1
            print(f"  Added {doc_number}: {patent.get('title', '')[:60]}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"Updated {count} new patents from {len(new_files)} files")

if __name__ == "__main__":
    update_from_new_files()