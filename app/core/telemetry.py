import os
import json
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
LOG_FILE = os.path.join(LOGS_DIR, "requests.jsonl")

def setup_telemetry():
    os.makedirs(LOGS_DIR, exist_ok=True)

def log_request(query: str, retrieved_chunks: List[Dict[str, Any]], usage: Dict[str, int], latency_s: float):
    """
    Log a request in JSONL format containing the required telemetry fields.
    """
    setup_telemetry()
    
    # Calculate estimated cost (based on gemini-3.5-flash pricing from Gemini)
    # $0.150 per 1M input, $0.600 per 1M output
    in_tokens = usage.get("prompt_tokens", 0)
    out_tokens = usage.get("completion_tokens", 0)
    cost = (in_tokens / 1_000_000 * 0.150) + (out_tokens / 1_000_000 * 0.600)
    
    # Format chunks for logging
    chunks_log = []
    for chunk in retrieved_chunks:
        chunks_log.append({
            "chunk_id": chunk.get("id"),
            "vector_rank": chunk.get("vector_rank"),
            "bm25_rank": chunk.get("bm25_rank"),
            "rrf_score": chunk.get("rrf_score"),
            "rerank_score": chunk.get("rerank_score")
        })

    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": query,
        "latency_s": latency_s,
        "tokens_in": in_tokens,
        "tokens_out": out_tokens,
        "estimated_cost_usd": cost,
        "chunks": chunks_log
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def get_latency_metrics() -> Dict[str, float]:
    """Reads the JSONL file and computes P50 and P95 latency."""
    if not os.path.exists(LOG_FILE):
        return {"p50_latency_s": 0.0, "p95_latency_s": 0.0, "total_requests": 0}
        
    latencies = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                latencies.append(data.get("latency_s", 0.0))
            except json.JSONDecodeError:
                pass
                
    if not latencies:
        return {"p50_latency_s": 0.0, "p95_latency_s": 0.0, "total_requests": 0}
        
    return {
        "p50_latency_s": float(np.percentile(latencies, 50)),
        "p95_latency_s": float(np.percentile(latencies, 95)),
        "total_requests": len(latencies)
    }
