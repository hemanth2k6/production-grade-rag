import json
import os
import sys
import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Add the project root to sys.path so we can import the app
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.services.retrieval import retrieval_service
from app.services.llm import llm_service
from app.core.config import settings

async def generate_rag_answers(dataset_path: str):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
        
    formatted_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    print(f"Generating answers for {len(data)} questions...")
    for item in data:
        question = item["question"]
        
        # 1. Retrieve
        chunks = await retrieval_service.retrieve_and_rerank(query=question, top_k=3)
        
        # 2. Generate
        qa_response = await llm_service.generate_response(query=question, retrieved_chunks=chunks)
        
        formatted_data["question"].append(question)
        formatted_data["answer"].append(qa_response.answer)
        formatted_data["contexts"].append([c["content"] for c in chunks])
        formatted_data["ground_truth"].append(item["expected_answer"])
        
    return Dataset.from_dict(formatted_data)

def run_eval():
    dataset_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        sys.exit(1)
        
    if not settings.openrouter_api_key:
        print("OPENROUTER_API_KEY not set. Cannot run evaluation.")
        print("Blocked / needs verification. Exiting with failure to enforce CI gating.")
        sys.exit(1)

    # 1. Generate answers via real RAG pipeline
    dataset = asyncio.run(generate_rag_answers(dataset_path))
    
    print("Running Ragas evaluation...")
    
    # Ragas uses Langchain Chat models. We point it to OpenRouter.
    # Note: Ragas metrics might still require embeddings (e.g. for relevancy). 
    # OpenRouter doesn't natively support OpenAI embeddings endpoint in the exact same way, 
    # but we can use an open router model or just skip answer_relevancy if embeddings fail.
    # For this script we will evaluate faithfulness which only needs LLM.
    llm = ChatOpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4o-mini"
    )

    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness], # answer_relevancy removed to avoid embedding API issues
            llm=llm,
            raise_exceptions=False
        )
        print("Evaluation Results:")
        print(result)
        
        faithfulness_score = result.get("faithfulness", 0.0)
        
        if faithfulness_score < 0.75:
            print(f"::error::Faithfulness score ({faithfulness_score}) is below the threshold of 0.75")
            sys.exit(1)
            
        print(f"Success! Faithfulness score: {faithfulness_score}")
        sys.exit(0)
    except Exception as e:
        print(f"Evaluation failed to run: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_eval()
