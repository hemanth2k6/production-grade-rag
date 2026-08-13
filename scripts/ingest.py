import os
import glob
import hashlib
import pickle
import argparse
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer

def get_stable_chunk_id(content: str, metadata: dict) -> str:
    """Generate a stable chunk ID using SHA256."""
    unique_string = f"{metadata.get('source', '')}_{content}"
    return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Simple recursive character text splitter equivalent."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    return chunks

def ingest(data_dir: str = "data", chroma_path: str = "chroma_db", bm25_path: str = "bm25_index.pkl"):
    print(f"Loading documents from {data_dir}...")
    files = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)
    
    if not files:
        print("No documents found.")
        return

    # In a real app we might use Langchain/LlamaIndex chunkers, but keeping it simple here
    all_chunks = []
    all_metadatas = []
    all_ids = []

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            # using roughly word count for chunks here
            chunks = chunk_text(text, chunk_size=200, overlap=50) 
            for chunk in chunks:
                metadata = {"source": os.path.basename(file_path)}
                chunk_id = get_stable_chunk_id(chunk, metadata)
                all_chunks.append(chunk)
                all_metadatas.append(metadata)
                all_ids.append(chunk_id)

    print(f"Created {len(all_chunks)} chunks. Generating embeddings and building indices...")

    # ChromaDB (Vector)
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    # Using local sentence-transformers model
    collection = chroma_client.get_or_create_collection(
        name="rag_collection",
        metadata={"hnsw:space": "cosine"}
    )
    
    # We will compute embeddings manually to ensure we control the model 
    # instead of using Chroma's default.
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(all_chunks).tolist()

    # Batch add to chroma
    batch_size = 5000
    for i in range(0, len(all_chunks), batch_size):
        collection.upsert(
            documents=all_chunks[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=all_metadatas[i:i+batch_size],
            ids=all_ids[i:i+batch_size]
        )
    print("ChromaDB index built.")

    # BM25 (Keyword)
    tokenized_corpus = [chunk.lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # We save both the BM25 model and the raw data mapping so we can retrieve chunks by index later
    bm25_data = {
        "bm25": bm25,
        "chunks": all_chunks,
        "metadatas": all_metadatas,
        "ids": all_ids
    }
    
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_data, f)
    print("BM25 index built and saved to BM25 index path.")
    
    print("Ingestion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()
    ingest(data_dir=args.data_dir)
