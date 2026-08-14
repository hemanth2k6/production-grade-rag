import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from app.core.config import settings

client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=settings.gemini_api_key,
) if settings.gemini_api_key else None

class QAResponse(BaseModel):
    answer: str = Field(description="The generated answer, or a declination if context is missing.")
    citations: List[str] = Field(description="A list of chunk IDs that strictly support the answer.")
    usage: Dict[str, int] = Field(default_factory=dict, description="Token usage details")

class LLMService:
    async def generate_response(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> QAResponse:
        """
        Generates an answer using OpenRouter, enforcing strict citation and JSON output.
        """
        if not client:
            raise ValueError("OPENROUTER_API_KEY is not configured.")

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

        response = await client.chat.completions.create(
            model="gemini-1.5-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        result_text = response.choices[0].message.content
        
        try:
            result_json = json.loads(result_text)
            
            # Post-hoc Citation Validation
            answer = result_json.get("answer", "")
            citations = result_json.get("citations", [])
            
            # Capture usage
            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
            
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
