# Part 2 PoC — containerize the Part 1 FastAPI app.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# the model (bge-base) downloads on first run into this cache dir;
# mounted as a volume in docker-compose so it isn't re-downloaded every rebuild
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8000

# start the API (data must already be loaded into the db — see README_poc.md)
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]