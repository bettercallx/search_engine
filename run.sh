#!/bin/bash
# run.sh - set up database, generate embeddings, and start the server
#
#   ./run.sh            normal run (only embeds what's missing)
#   ./run.sh --reset    drop & recreate the whole database, then run
#   ./run.sh --reembed  keep the database, wipe embeddings and regenerate them(use after changing the model or the prefix logic)

set -e

if [ "$1" = "--reset" ]; then
    echo "Step 0: Resetting database ..."
    psql -c "DROP DATABASE IF EXISTS patent_search;"
    psql -c "CREATE DATABASE patent_search;"
fi

if [ "$1" = "--reembed" ]; then
    echo "Step 0: Clearing existing embeddings ..."
    python3 -c "from src.db import get_conn, reset_embeddings; conn = get_conn(); reset_embeddings(conn); conn.close()"
fi

echo "Step 1: Creating tables and loading data ..."
python3 src/db.py

echo "Step 2: Generating embeddings ..."
python3 src/embedder.py

echo "Step 3: Creating vector indexes ..."
python3 -c "from src.db import get_conn, create_vector_index; conn = get_conn(); create_vector_index(conn); conn.close()"

echo "Step 4: Starting server ..."
python3 -m uvicorn app:app --reload --port 8000