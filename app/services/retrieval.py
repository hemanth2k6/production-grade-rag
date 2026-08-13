import asyncio
import pickle
import os
from typing import List, Dict, Any

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer
from app.core.config import settings

class RetrievalService:
    def __init__(self):
        # Initialize Embedder
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize Cross-Encoder for reranking
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        try:
            self.collection = self.chroma_client.get_collection(name="rag_collection")
        except Exception:
            self.collection = None
            
        # Initialize BM25
        self.bm25_data = None
        if os.path.exists("bm25_index.pkl"):
            with open("bm25_index.pkl", "rb") as f:
                self.bm25_data = pickle.load(f)

    async def retrieve_and_rerank(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes hybrid search (ChromaDB + BM25), fuses with RRF, and reranks using Cross-Encoder.
        Traced by Langfuse.
        """
        if not self.collection or not self.bm25_data:
            print("Indices not found. Run ingest.py first.")
            return []

        match_count = top_k * 3

        # 1. Vector Search
        query_embedding = self.embedder.encode([query])[0].tolist()
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=match_count
        )
        
        vector_ranks = {}
        if vector_results and vector_results['ids'] and len(vector_results['ids'][0]) > 0:
            for rank, chunk_id in enumerate(vector_results['ids'][0]):
                vector_ranks[chunk_id] = rank

        # 2. Keyword Search (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_data["bm25"].get_scores(tokenized_query)
        
        # Sort by score descending
        bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:match_count]
        
        bm25_ranks = {}
        for rank, idx in enumerate(bm25_ranked_indices):
            if bm25_scores[idx] > 0: # Only rank if it actually matches
                chunk_id = self.bm25_data["ids"][idx]
                bm25_ranks[chunk_id] = rank

        # 3. Reciprocal Rank Fusion (RRF)
        k_rrf = 60
        rrf_scores = {}
        all_ids = set(vector_ranks.keys()).union(set(bm25_ranks.keys()))
        
        for chunk_id in all_ids:
            score = 0.0
            if chunk_id in vector_ranks:
                score += 1.0 / (k_rrf + vector_ranks[chunk_id])
            if chunk_id in bm25_ranks:
                score += 1.0 / (k_rrf + bm25_ranks[chunk_id])
            rrf_scores[chunk_id] = score

        # Sort by RRF score
        sorted_rrf_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:match_count]

        # Retrieve actual chunk data
        # To do this efficiently, we look up from the bm25_data dictionary since it has all in memory
        id_to_idx = {chunk_id: idx for idx, chunk_id in enumerate(self.bm25_data["ids"])}
        
        retrieved_chunks = []
        for chunk_id in sorted_rrf_ids:
            idx = id_to_idx[chunk_id]
            retrieved_chunks.append({
                "id": chunk_id,
                "content": self.bm25_data["chunks"][idx],
                "source": self.bm25_data["metadatas"][idx].get("source", ""),
                "vector_rank": vector_ranks.get(chunk_id, -1),
                "bm25_rank": bm25_ranks.get(chunk_id, -1),
                "rrf_score": rrf_scores[chunk_id]
            })

        if not retrieved_chunks:
            return []

        # 4. Cross-Encoder Reranking
        cross_inp = [[query, chunk["content"]] for chunk in retrieved_chunks]
        
        # Run blocking inference in threadpool
        scores = await asyncio.to_thread(self.cross_encoder.predict, cross_inp)
        
        for idx, chunk in enumerate(retrieved_chunks):
            chunk["rerank_score"] = float(scores[idx])
            
        reranked_chunks = sorted(retrieved_chunks, key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked_chunks[:top_k]

retrieval_service = RetrievalService()
