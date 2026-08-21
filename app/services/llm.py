import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings

class QAResponse(BaseModel):
    answer: str = Field(description="The generated answer, or a declination if context is missing.")
    citations: List[str] = Field(description="A list of chunk IDs that strictly support the answer.")
    usage: Dict[str, int] = Field(default_factory=dict, description="Token usage details")

client = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=settings.gemini_api_key,
    temperature=0.0,
    max_retries=20
).with_structured_output(QAResponse) if settings.gemini_api_key else None

class LLMService:
    async def generate_response(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> QAResponse:
        """
        Generates an answer using Gemini, enforcing strict citation and JSON output.
        """
        if not client:
            raise ValueError("GEMINI_API_KEY is not configured.")

        context_text = "\n\n".join(
            [f"Chunk ID: {chunk['id']}\nContent: {chunk['content']}" for chunk in retrieved_chunks]
        )

        system_prompt = (
            "You are a strict, factual assistant. You must answer the user's query ONLY using the provided documents below.\n"
            "If the provided documents DO NOT contain the answer, you MUST decline to answer by stating exactly: "
            "'I don't have enough grounded information to answer this'.\n"
            "You must output a JSON object with 'answer' and an array of 'citations' containing the string Chunk IDs that support your answer.\n"
            "--- DOCUMENTS START ---\n"
            f"{context_text}\n"
            "--- DOCUMENTS END ---\n"
        )

        try:
            # Langchain handles the prompt and structured output natively
            result_obj: QAResponse = await client.ainvoke([
                ("system", system_prompt),
                ("user", query)
            ])
            
            # Post-hoc Citation Validation
            answer = result_obj.answer
            citations = result_obj.citations
            usage_dict = result_obj.usage
            
            if "I don't have enough grounded information to answer this" in answer:
                return QAResponse(answer=answer, citations=[], usage=usage_dict)
                
            valid_chunk_ids = {str(c["id"]) for c in retrieved_chunks}
            validated_citations = [c for c in citations if str(c) in valid_chunk_ids]
            
            if not validated_citations and len(citations) > 0:
                # LLM hallucinated citations
                return QAResponse(
                    answer="I don't have enough grounded information to answer this.",
                    citations=[],
                    usage=usage_dict
                )
                
            return QAResponse(answer=answer, citations=validated_citations, usage=usage_dict)
        except Exception:
            return QAResponse(
                answer="I don't have enough grounded information to answer this.", 
                citations=[],
                usage={}
            )

llm_service = LLMService()
