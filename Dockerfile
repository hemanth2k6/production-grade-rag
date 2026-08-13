FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Run ingestion script to pre-build ChromaDB and BM25 indices within the container
RUN python scripts/ingest.py

# Expose the port
EXPOSE 8000

# Start the application, using the PORT environment variable if provided
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
