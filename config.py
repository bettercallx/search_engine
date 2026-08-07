from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent  # search_engine/
DATA_DIR = PROJECT_ROOT / "patent_data_small"

EMBEDDING_FIELDS = ["abstract", "claims"]#"detailed_description"
INDEX_PATH = "index"

TOP_K = 10

# embedding model
MODELS = [
    ("BAAI/bge-base-en-v1.5", "Represent this sentence: "),
]