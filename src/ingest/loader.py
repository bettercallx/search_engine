# Load newly scraped patents (new_*.json) into the DB, then let embedder.py generate their vectors. 
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import config
from src.db import get_conn
from src.embedder import embed_patents

def load_new_files():
    folder = Path(config.DATA_DIR)
    new_files = list(folder.glob("new_*.json"))
    if not new_files:
        print("No new files found")
        return

    conn = get_conn()
    cur = conn.cursor()
    count = 0

    for file_path in new_files:
        with open(file_path, "r") as f:
            patents = json.load(f)

        for patent in patents:
            doc_number = patent.get("doc_number")

            # skip if already in the DB
            cur.execute("SELECT id FROM patents WHERE doc_number = %s", [doc_number])
            if cur.fetchone():
                print(f"  Skipped {doc_number} (already exists)")
                continue

            # insert the row only; abstract_embedding stays NULL and no claim rows
            cur.execute("""
                INSERT INTO patents (
                    doc_number, title, abstract, claims, detailed_description,
                    classification, bibtex, filename, abstract_tsv, title_tsv
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                        to_tsvector('english', %s), to_tsvector('english', %s))
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

            count += 1
            print(f"  Added {doc_number}: {patent.get('title', '')[:60]}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"Inserted {count} new patents from {len(new_files)} files")

    # generate embeddings for whatever is still missing (includes the rows above)
    if count:
        print("Generating embeddings for new patents ...")
        embed_patents()

if __name__ == "__main__":
    load_new_files()