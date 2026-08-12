import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langfuse.openai import AsyncOpenAI
from langfuse.decorators import observe, langfuse_context
from app.core.config import settings

client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

class QAResponse(BaseModel):
    answer: str = Field(description="The generated answer, or a declination if context is missing.")
    citations: List[int] = Field(description="A list of chunk IDs that strictly support the answer. Empty if no answer.")

class LLMService:
    @observe(name="generate_qa", as_type="generation")
    async def generate_response(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> QAResponse:
        """
        Generates an answer using OpenAI, enforcing strict citation and JSON output.
        Traced by Langfuse.
        """
        # Inject chunks as context for tracing
        langfuse_context.update_current_observation(
            input={"query": query, "chunks": retrieved_chunks}
        )

        if not client:
            return QAResponse(
                answer="[Simulated Output] No OpenAI API key configured.",
                citations=[c.get("id", 0) for c in retrieved_chunks]
            )

        # Format context for the prompt
        context_text = "\n\n".join(
            [f"Chunk ID: {chunk['id']}\nContent: {chunk['content']}" for chunk in retrieved_chunks]
        )

        system_prompt = (
            "You are a strict, factual assistant. You must answer the user's query ONLY using the provided documents below.\n"
            "If the provided documents DO NOT contain the answer, you MUST decline to answer by stating exactly: "
            "'I cannot answer this based on the provided documents'.\n"
            "You must output a JSON object with 'answer' and an array of 'citations' containing the integer Chunk IDs that support your answer.\n"
            f"Documents:\n{context_text}"
        )

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"},
            metadata={"query_length": len(query), "num_chunks": len(retrieved_chunks)}
        )

        result_text = response.choices[0].message.content
        langfuse_context.update_current_observation(output=result_text)
        
        try:
            result_json = json.loads(result_text)
            return QAResponse(**result_json)
        except Exception:
            return QAResponse(answer="Failed to parse structured output.", citations=[])

llm_service = LLMService()
