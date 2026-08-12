from langfuse.openai import AsyncOpenAI
from langfuse.decorators import observe
from app.core.config import settings

# Initialize Langfuse-wrapped AsyncOpenAI client
# It automatically picks up LANGFUSE_* environment variables if available.
client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

class LLMService:
    @observe(name="generate_qa")
    async def generate_response(self, query: str) -> str:
        """
        Generates a response using OpenAI, automatically traced by Langfuse wrapper.
        Tracks P50/P95 latency, tokens, and cost.
        """
        if not client:
            return f"[Simulated Output] No OpenAI API key configured. Query received: {query}"
            
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful RAG assistant."},
                {"role": "user", "content": query}
            ],
            metadata={"query_length": len(query)}
        )
        return response.choices[0].message.content
