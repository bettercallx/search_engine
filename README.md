# search_engine

stats: run `patent/stats.py` for statistic of patent data

Embedding: model is bge-base-en-v1.5, 

## environment

python PostgreSQL pgvector sentence-transformers

`python3 -m venv .venv`
`python3 source ./venv/bin/activate`

install psycopg2-binary
`pip install psycopg2-binary`

install PostgreSQL server
`brew install postgresql@16
brew services start postgresql@16`

install pgvector extention to postgresql@16
`brew install pgvector
cd /tmp
git clone https://github.com/pgvector/pgvector.git
cd pgvector
export PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config
make
make install`

install sentence-transformers
`pip install sentence-transformers`


install uvicorn fastapi
`pip install uvicorn fastapi`

install beautifulsoup4 requests
`pip install beautifulsoup4 requests`