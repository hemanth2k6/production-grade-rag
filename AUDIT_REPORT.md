# RAG Codebase Audit Report

## 1. Repository Inventory

| File | Intended Purpose | Actual Current State |
|---|---|---|
| `app/main.py` | FastAPI application entry point | **Working** (but trivial setup) |
| `app/api/routes.py` | API endpoints (`/query`) | **Working** (glues retrieval and LLM calls, but passes `None` for embeddings) |
| `app/core/config.py` | Configuration and secrets management | **Working** (uses Pydantic settings and `.env`) |
| `app/services/llm.py` | LLM generation and strict citation enforcement | **Stubbed/Faked** (Prompts OpenAI, but swallows errors, fakes responses if no API key, and lacks true citation validation) |
| `app/services/retrieval.py` | Hybrid search and reranking logic | **Stubbed/Partially Working** (Local cross-encoder works, but Supabase retrieval is mocked on error, and `query_embedding` is stubbed to a zero-vector) |
| `evals/golden_dataset.json` | 50-200 hand-verified QA pairs | **Stubbed/Faked** (Contains generic mock data like "What is fact 1?", not real QA pairs) |
| `evals/run_evals.py` | Offline faithfulness/relevancy evaluation script | **Faked** (Hardcodes generated answer to match ground truth, artificially forcing perfect scores) |
| `.github/workflows/rag-evals.yml` | CI/CD pipeline | **Working** (Executes the evaluation script, but the script itself is a facade) |
| `supabase/migrations/*_init_pgvector.sql` | Database schema and hybrid search RPC | **Stubbed** (RPC is named `hybrid_search` but only executes standard vector search; no BM25) |
| `requirements.txt` | Python dependencies | **Bloated** (Contains ~150 packages, including many massive unused libraries like `langchain`, `langgraph`, and `torch` ecosystems) |

## 2. Trace the Real Request Path

When a request hits `POST /api/v1/query`:
1. **`app/api/routes.py:20`**: `handle_query` receives the request.
2. **`app/api/routes.py:23`**: Calls `retrieval_service.retrieve_and_rerank` passing `query_embedding=None`.
3. **`app/services/retrieval.py:18`**: 
   - Overrides the `None` embedding with a hardcoded `[0.0] * 1536` array (line 26).
   - Attempts to call the Supabase RPC `hybrid_search`. 
   - If Supabase fails (or is unconfigured), a `try/except` block (line 41-48) intercepts the error and returns a **hardcoded list of static mock chunks**.
   - A local HuggingFace Cross-Encoder rescores and sorts these chunks.
4. **`app/api/routes.py:30`**: Calls `llm_service.generate_response`.
5. **`app/services/llm.py:16`**: 
   - If the OpenAI API key is missing, it returns a **hardcoded simulated output** with citations artificially set to match all provided chunks (line 27-30).
   - Otherwise, it queries OpenAI. 
   - If the LLM returns invalid JSON, an `except` block (line 61-62) swallows the error and returns a plausible-looking fallback answer with empty citations.
6. **`app/api/routes.py:35`**: The response is returned to the user.

## 3. Component-by-Component Gap Check

1. **Hybrid retrieval (BM25 + Semantic, fused)**: **STUBBED** 
   - *Evidence*: `supabase/migrations/20260812155819_init_pgvector.sql:28` - The RPC `hybrid_search` only executes pgvector similarity. BM25 and fusion (RRF) do not exist. `app/services/retrieval.py:26` hardcodes the query embedding.
2. **Cross-encoder re-ranker**: **WORKING**
   - *Evidence*: `app/services/retrieval.py:57` - Uses local `sentence-transformers` CrossEncoder successfully to rescore the chunks.
3. **Strict citation enforcement**: **STUBBED**
   - *Evidence*: `app/services/llm.py:38` - The system prompt requests citations, but there is zero programmatic validation to ensure the LLM's citations actually exist in the retrieved chunks or support the claim.
4. **Full tracing/telemetry**: **PARTIALLY WORKING**
   - *Evidence*: `app/services/llm.py:15` & `app/services/retrieval.py:18` - Langfuse `@observe` decorators are used. However, similarity/rerank scores, token counts, and cost estimates are not explicitly tracked in the spans.
5. **P50/P95 latency tracking**: **NOT PRESENT**
   - *Evidence*: Neither the codebase nor the telemetry setup (Langfuse integration) tracks or exports P50/P95 aggregate latency metrics (e.g. to Prometheus/Datadog).
6. **Golden evaluation dataset**: **STUBBED**
   - *Evidence*: `evals/golden_dataset.json` consists of fake placeholder data, not 50-200 hand-verified pairs. `evals/run_evals.py:22` bypasses evaluation entirely.
7. **CI/CD pipeline (GitHub Actions)**: **WORKING (Executing fake tests)**
   - *Evidence*: `.github/workflows/rag-evals.yml` runs successfully on PRs, but relies on `run_evals.py` which mocks a perfect evaluation.

## 4. Dependency and Config Check

- **Bloat / Unused Dependencies**: `requirements.txt` contains 154 packages. The codebase natively implements everything and does **not** use `langchain`, `langgraph`, `pandas`, `scikit-learn`, `SQLAlchemy`, etc., despite them being installed. 
- **Missing Dependencies**: No notable missing dependencies for the code that is actually written, though an embedding generator (e.g., OpenAI embeddings API) is conspicuously absent.

## 5. Secrets / Config Check (OpenRouter Migration)

Secrets are currently managed via `app/core/config.py` using `pydantic_settings` to read from `.env` and environment variables.

**Flagged for OpenRouter Migration:**
- `app/services/llm.py:8`: Direct instantiation of `AsyncOpenAI(api_key=settings.openai_api_key)`.
- `app/services/llm.py:45`: Direct call to `client.chat.completions.create(model="gpt-4o-mini", ...)`.

These call sites need to be modified to point to the OpenRouter base URL (`https://openrouter.ai/api/v1`) using the `OPENROUTER_API_KEY` instead of OpenAI's defaults.

## 6. Cohere Reranker Note

**Decision Point:** The codebase does **not** use the Cohere reranker. Instead, it uses a local HuggingFace `sentence-transformers` CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) instantiated in `app/services/retrieval.py:16`. 
Since OpenRouter does not support Cohere's rerank endpoint, this local reranker approach is technically compatible as-is, but it adds significant memory/compute overhead to the API service. We must decide whether to keep the local reranker or migrate to a different managed reranking service.

## 7. "Fake It" Detection

This codebase contains extensive "scaffolding" that artificially fakes success:
1. **Mocked Eval Scores**: `evals/run_evals.py:22` hardcodes `formatted_data["answer"].append(item["expected_answer"])` so the generated answer perfectly matches the ground truth.
2. **Mock CI Failure**: `evals/run_evals.py:42` prints `::error::Faithfulness score (0.0) is below the threshold of 0.85` strictly if the OpenAI key is missing, faking a quality check.
3. **Static Retrieval**: `app/services/retrieval.py:44-48` returns a hardcoded list of 3 chunks if the Supabase RPC fails. 
4. **Hardcoded Embeddings**: `app/services/retrieval.py:26` intercepts missing embeddings with `[0.0] * 1536`, completely nullifying vector search.
5. **Faked LLM Response**: `app/services/llm.py:27` returns `[Simulated Output]` and manually sets the citations to whatever chunks were passed in if the API key is missing. 
6. **Swallowed Errors**: `app/services/llm.py:61` catches JSON parsing errors and returns a valid `QAResponse` with empty citations instead of surfacing the 500 error.
7. **Fake RPC Logic**: The Supabase `hybrid_search` function is missing BM25 completely.

## 8. Git History Sanity Check

Running `git log --oneline -30` yields 7 perfectly scoped commits:
- `691b0f8` docs: add comprehensive FAANG-grade README
- `11c0711` ci: configure render and supabase deployment pipelines
- `bcb0c97` ci: add GitHub Actions workflow for RAG evaluations
- `230e468` feat: scaffold golden dataset and Ragas evaluation script
- `df9e7fe` feat: implement generation chain with strict citation enforcement
- `ae932f4` feat: implement hybrid search and cross-encoder reranking
- `5ab57fd` feat: scaffold FastAPI app and initialize Langfuse telemetry

**Summary**: The git history is an illusion. The commit messages claim to implement robust, production-grade features ("strict citation enforcement", "hybrid search"), but the actual code diffs were merely injecting mocks, stubs, and fallback `try/except` blocks. The history matches the claimed progress visually, but not functionally.

---

## Prioritized Punch List

### Fundamentally Broken (Needs Rewrite)
1. **Embeddings & Vector Search**: Implement actual embedding generation in `retrieval_service` instead of a 0-vector. Rewrite the Supabase RPC to genuinely merge pgvector and BM25 (via RRF).
2. **LLM Generation & Citations**: Migrate the `AsyncOpenAI` client to OpenRouter. Strip out the faked responses. Write actual validation logic to ensure the returned citations are within the retrieved chunk IDs.
3. **Evaluation Pipeline**: Remove the answer-hardcoding in `run_evals.py`. Generate a real dataset with actual chunks and QA pairs.

### Salvageable (Needs Patching)
1. **Reranker**: The local `CrossEncoder` works but is inefficiently placed. Consider offloading to a separate service or keeping it if memory permits.
2. **API Routes & Telemetry**: The FastAPI wrapper and Langfuse decorators are mostly fine, but need to be updated to log rerank scores, latency (P50/P95), and token/cost metrics properly.
3. **Dependencies**: Purge `requirements.txt` of all unused `langchain` and massive ecosystem dependencies to speed up builds.
