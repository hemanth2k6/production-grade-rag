import pytest
from unittest.mock import AsyncMock, patch
from app.services.llm import llm_service, QAResponse

@pytest.mark.asyncio
async def test_generation_refusal():
    # We mock the OpenAI client to simulate the model obeying the prompt and refusing
    with patch('app.services.llm.client') as mock_client:
        mock_response = AsyncMock()
        mock_response.choices[0].message.content = '{"answer": "I don\'t have enough grounded information to answer this.", "citations": []}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Pass chunks that don't answer the query
        chunks = [{"id": "xyz", "content": "The sky is blue."}]
        
        response = await llm_service.generate_response(
            query="What is the capital of France?",
            retrieved_chunks=chunks
        )
        
        assert response.answer == "I don't have enough grounded information to answer this."
        assert len(response.citations) == 0

@pytest.mark.asyncio
async def test_generation_hallucinated_citation():
    # We mock the OpenAI client to simulate the model hallucinating a citation
    with patch('app.services.llm.client') as mock_client:
        mock_response = AsyncMock()
        mock_response.choices[0].message.content = '{"answer": "The capital is Paris.", "citations": ["fake_chunk_id"]}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        chunks = [{"id": "xyz", "content": "Paris is the capital of France."}]
        
        response = await llm_service.generate_response(
            query="What is the capital of France?",
            retrieved_chunks=chunks
        )
        
        # Because 'fake_chunk_id' is not in the chunks, post-hoc enforcement should strip it and refuse
        assert response.answer == "I don't have enough grounded information to answer this."
        assert len(response.citations) == 0
