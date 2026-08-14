from typing import List
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from app.services.llm import llm_service
from app.services.retrieval import retrieval_service
from app.core.telemetry import log_request, get_latency_metrics
from app.core.limiter import limiter
import time

router = APIRouter()

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The query string")
    top_k: int = Field(default=5, ge=1, le=20)

class QueryResponse(BaseModel):
    answer: str
    citations: List[int]

@router.post("/query", response_model=QueryResponse)
@limiter.limit("5/minute")
async def handle_query(request: Request, payload: QueryRequest):
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
