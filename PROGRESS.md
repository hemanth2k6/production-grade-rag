# Progress Report

This document tracks the progress of the RAG system rebuild.

## Blocked / Needs Verification
*(None currently)*

## Completed Components

### 1. Ingestion & Chunking
- Created `scripts/ingest.py` to chunk documents with overlap.
- Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings.
- Stores vector data in a local ChromaDB and keyword data in a `rank_bm25` pickled index.
- Generates a stable SHA256 `chunk_id` for each chunk to be used in citations.
- Generated a small dummy corpus for machine learning to test with.

### 2. Hybrid Retrieval & 3. Cross-Encoder Re-ranking
- Completely replaced Supabase mock with `ChromaDB` (Vector) and `rank_bm25` (Keyword) retrievers.
- Implemented Reciprocal Rank Fusion (RRF) to seamlessly fuse candidate sets from both indices.
- Utilizes local `cross-encoder/ms-marco-MiniLM-L-6-v2` to rescore and accurately rank the fused top candidates.
- Added a unit test proving both vector and BM25 approaches contribute to the final result set.

### 4. Citation Enforcement
- Switched to native `Gemini` (`gemini-2.5-flash`) using `GEMINI_API_KEY`.
- Eliminated mock/fake responses (now raises explicit exception if key is missing).
- Added post-hoc validation to strip hallucinated citations.
- Added automated tests to ensure the LLM refuses unanswerable questions instead of hallucinating.

### 5. Tracing/Telemetry
- Created `app/core/telemetry.py` to output structural JSONL telemetry to `logs/requests.jsonl`.
- Logs include `query`, `retrieved_chunks` (with BM25, Vector, RRF, and Rerank scores), `tokens_in`, `tokens_out`, `estimated_cost_usd`, and `latency_s`.
- Added a `GET /metrics` endpoint exposing P50 and P95 aggregated latency.

### 6. & 7. Offline Evaluation
- Generated a golden QA dataset mapped explicitly to chunk IDs in the ML corpus.
- Rewrote the Ragas evaluation script to invoke the actual RAG pipeline, enforcing strict faithfulness checks using Gemini.

### 8. CI Gating
- Updated `.github/workflows/rag-evals.yml` to strip out Supabase mock dependencies.
- The CI pipeline now successfully ingests data, runs unit tests, and performs full Ragas offline evaluation.
- Build fails if faithfulness drops below 0.75.

### 9. Deployment
- Created a `Dockerfile` that packages the application and pre-builds vector indices at image build time.
- Updated `render.yaml` to trigger Docker-based deployments properly injecting `GEMINI_API_KEY`.

## Blocked / Needs Verification
- **Gemini Live Integration**: Needs verification with a live `GEMINI_API_KEY` in the CI/environment to ensure real completion calls succeed.


