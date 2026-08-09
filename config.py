import os

# Database connection
DB_NAME = os.getenv("PATENT_DB_NAME", "patent_search")
DB_USER = os.getenv("PATENT_DB_USER", "postgres")
DB_HOST = os.getenv("PATENT_DB_HOST", "localhost")
DB_PORT = int(os.getenv("PATENT_DB_PORT", "5432"))

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent  # search_engine/
DATA_DIR = PROJECT_ROOT / "patent_data_small"
NEW_DATA_DIR = PROJECT_ROOT / "new_patent_data" 

EMBEDDING_FIELDS = ["abstract", "claims"]#"detailed_description"
INDEX_PATH = "index"

TOP_K = 10

# embedding model
MODELS = [
    ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: "),
]

RRF_WEIGHTS = {"semantic": 0.5, "keyword": 0.3, "claim": 0.2}
CLAIM1_BOOST = 1.5
RRF_K = 60