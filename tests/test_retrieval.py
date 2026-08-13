import pytest
from app.services.retrieval import retrieval_service

@pytest.mark.asyncio
async def test_hybrid_retrieval():
    # Execute a query against our local chunks
    query = "Arthur Samuel IBM"
    chunks = await retrieval_service.retrieve_and_rerank(query, top_k=2)
    
    assert len(chunks) > 0, "Should retrieve at least one chunk"
    
    found_vector = False
    found_bm25 = False
    
    # We want to prove both retrievers contributed to the final score computation
    for chunk in chunks:
        assert 'rrf_score' in chunk
        assert 'vector_rank' in chunk
        assert 'bm25_rank' in chunk
        
        if chunk['vector_rank'] != -1:
            found_vector = True
        if chunk['bm25_rank'] != -1:
            found_bm25 = True
            
    assert found_vector, "Vector search should have contributed to the results."
    assert found_bm25, "BM25 keyword search should have contributed to the results."
    
    # Assert reranker applied
    assert 'rerank_score' in chunks[0], "Cross-encoder should apply a rerank score."
