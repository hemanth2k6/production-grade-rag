# Production-Grade RAG Pipeline

A fully-functional, production-ready Retrieval-Augmented Generation (RAG) system prioritizing grounded answers, strict citations, and zero-hallucination.

## Core Features

- **Hybrid Retrieval System:** Combines semantic Vector search (via local `ChromaDB`) with exact-match Keyword search (`BM25Okapi`). 
- **Reciprocal Rank Fusion (RRF):** Seamlessly merges and weights both retrieval strategies to ensure no relevant documents are missed.
- **Cross-Encoder Reranking:** Re-scores the fused candidate list using `ms-marco-MiniLM-L-6-v2` to select only the top context windows, maximizing signal-to-noise ratio before LLM generation.
- **Strict Citation Enforcement:** Ensures the LLM (via `OpenRouter`) strictly relies on the provided document chunks, stripping any hallucinated citations. If the documents cannot answer the question, the system gracefully declines to answer.
- **Tracing & Telemetry:** Full structured JSONL logging per request. Captures retrieved chunk IDs, scores, vector similarities, token input/output usage, estimated cost per request, and processing latency. Includes an endpoint to extract P50/P95 latencies.
- **Automated Golden Evaluation:** Includes a golden QA dataset paired against `ragas`. Faithfulness and answer relevancy tests execute in CI via GitHub Actions. If the faithfulness score drops below the defined threshold (0.75), the pipeline rejects the PR.

## Setup Instructions

1. **Clone the repository.**
2. **Setup virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Set your OpenRouter API Key:**
   ```bash
   export OPENROUTER_API_KEY="your-api-key"
   ```
4. **Ingest your Corpus:** (This will create `chroma_db` and `bm25_index.pkl`)
   ```bash
   python scripts/ingest.py
   ```
5. **Start the API:**
   ```bash
   uvicorn app.main:app --reload
   ```

## Development and Testing

- **Run unit tests:**
  ```bash
  PYTHONPATH=. pytest tests/ -v
  ```
- **Run Golden Evaluation Locally:**
  ```bash
  PYTHONPATH=. python evals/run_evals.py
  ```

## Deployment

The application includes a `Dockerfile` that packages the application and runs ingestion during image build. You can deploy it instantly to Docker-compatible platforms (e.g. Render, AWS App Runner). A `render.yaml` configuration is also provided.
