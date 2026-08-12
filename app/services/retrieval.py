import asyncio
from typing import List, Dict, Any
from supabase import create_client, Client
from sentence_transformers import CrossEncoder
from app.core.config import settings
from langfuse.decorators import observe

class RetrievalService:
    def __init__(self):
        # Initialize Supabase client
        supabase_url = settings.supabase_url or "https://placeholder.supabase.co"
        supabase_key = settings.supabase_key or "placeholder_key"
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
        # Initialize HuggingFace Cross-Encoder for reranking
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    @observe(name="retrieve_and_rerank")
    async def retrieve_and_rerank(self, query: str, query_embedding: List[float] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes hybrid search (pgvector + BM25) and reranks chunks using a Cross-Encoder.
        Traced by Langfuse.
        """
        if query_embedding is None:
            # Fallback mock embedding if not provided by an embeddings service
            query_embedding = [0.0] * 1536 

        try:
            # Assumes a Supabase RPC 'hybrid_search' for BM25 + pgvector combined search
            # We fetch more chunks (top_k * 2) for the reranker to evaluate
            response = self.supabase.rpc(
                "hybrid_search",
                {
                    "query_text": query,
                    "query_embedding": query_embedding,
                    "match_count": top_k * 2 
                }
            ).execute()
            
            retrieved_chunks = response.data
        except Exception as e:
            # Offline/Scaffold fallback if Supabase is not connected or RPC is missing
            print(f"Supabase RPC failed or not configured: {e}")
            retrieved_chunks = [
                {"id": 1, "content": "Mock chunk containing the answer to the query.", "source": "doc1.txt"},
                {"id": 2, "content": "Irrelevant chunk about a completely different topic.", "source": "doc2.txt"},
                {"id": 3, "content": "Another mock document mentioning related keywords but lacking the answer.", "source": "doc3.txt"}
            ]

        if not retrieved_chunks:
            return []

        # Prepare input for Cross-Encoder: pairs of [query, chunk_content]
        cross_inp = [[query, chunk["content"]] for chunk in retrieved_chunks]
        
        # Rescore using HuggingFace cross-encoder
        scores = self.cross_encoder.predict(cross_inp)
        
        # Attach scores and sort in descending order
        for idx, chunk in enumerate(retrieved_chunks):
            chunk["rerank_score"] = float(scores[idx])
            
        reranked_chunks = sorted(retrieved_chunks, key=lambda x: x["rerank_score"], reverse=True)
        
        # Return the strictly filtered top_k chunks
        return reranked_chunks[:top_k]

retrieval_service = RetrievalService()
