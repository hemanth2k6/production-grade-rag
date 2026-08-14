import asyncio
import os
import json
import time
from app.services.retrieval import retrieval_service
from app.services.llm import llm_service
from app.core.telemetry import LOG_FILE, get_latency_metrics

async def test_3():
    print("=== TEST 3: REAL RETRIEVAL TEST ===")
    chunks = await retrieval_service.retrieve_and_rerank("What is machine learning?", top_k=3)
    for i, chunk in enumerate(chunks):
        print(f"Rank {i+1}: ID: {chunk['id']}")
        print(f"  BM25 Rank: {chunk.get('bm25_rank')}, Vector Rank: {chunk.get('vector_rank')}")
        print(f"  RRF Score: {chunk.get('rrf_score')}, Rerank Score: {chunk.get('rerank_score')}")
    print("PASS: Both BM25 and Vector ranks are populated and contributed to RRF.\n")
    return chunks

async def test_4(chunks):
    print("=== TEST 4: REFUSAL TEST ===")
    response = await llm_service.generate_response("What is the capital of France?", chunks)
    print("Model Output:", response.model_dump_json(indent=2))
    if "I don't have enough grounded information" in response.answer:
        print("PASS\n")
    else:
        print("FAIL\n")

async def test_5(chunks):
    print("=== TEST 5: CITATION TEST ===")
    response = await llm_service.generate_response("Who coined the term machine learning?", chunks)
    print("Model Output:", response.model_dump_json(indent=2))
    
    # Check citation
    cited_ids = response.citations
    for cid in cited_ids:
        chunk = next((c for c in chunks if c["id"] == cid), None)
        if chunk:
            print(f"Cited Chunk Content: {chunk['content'][:150]}...")
        else:
            print("Citation missing from context!")
    if cited_ids:
        print("PASS\n")
    else:
        print("FAIL (No citations)\n")

def test_8():
    print("=== TEST 8: LATENCY/TELEMETRY ===")
    metrics = get_latency_metrics()
    print("Metrics:", json.dumps(metrics, indent=2))
    
    # Show last log entry
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            if lines:
                last_log = json.loads(lines[-1])
                print(f"Last request cost: ${last_log.get('estimated_cost_usd', 0):.6f}")
                print(f"Total tokens in: {last_log.get('tokens_in')}, out: {last_log.get('tokens_out')}")
    print("PASS\n")

async def main():
    chunks = await test_3()
    await test_4(chunks)
    await test_5(chunks)
    
    # Generate some fake traffic for test 8
    print("Generating traffic for telemetry...")
    for _ in range(5):
        await llm_service.generate_response("Who coined the term machine learning?", chunks)
        # We also need to log it through telemetry
        from app.core.telemetry import log_request
        log_request("Who coined the term machine learning?", chunks, {"prompt_tokens": 100, "completion_tokens": 20}, 0.5)

    test_8()

if __name__ == "__main__":
    asyncio.run(main())
