from fastapi import APIRouter
from pydantic import BaseModel
from app.services.llm import LLMService

router = APIRouter()
llm_service = LLMService()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    answer = await llm_service.generate_response(request.query)
    return QueryResponse(answer=answer)
