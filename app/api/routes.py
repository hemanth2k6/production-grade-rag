from typing import List
from fastapi import APIRouter
from pydantic import BaseModel
from app.services.llm import llm_service
from app.services.retrieval import retrieval_service
from app.core.telemetry import log_request, get_latency_metrics
import time

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    citations: List[int]

@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    start_time = time.time()
    
    # Step 1: Retrieve and Rerank
    chunks = await retrieval_service.retrieve_and_rerank(
        query=request.query, 
        top_k=request.top_k
    )
    
    # Step 2: Generation with Citation Enforcement
    qa_response = await llm_service.generate_response(
        query=request.query, 
        retrieved_chunks=chunks
    )
    
    latency = time.time() - start_time
    
    # Step 3: Log Telemetry
    log_request(
        query=request.query,
        retrieved_chunks=chunks,
        usage=qa_response.usage,
        latency_s=latency
    )
    
    return QueryResponse(answer=qa_response.answer, citations=qa_response.citations)

@router.get("/metrics")
async def get_metrics():
    return get_latency_metrics()
