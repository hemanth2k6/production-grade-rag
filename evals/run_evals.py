import json
import os
import sys
import asyncio
from datasets import Dataset

# -- MONKEYPATCH FOR RAGAS IMPORT BUG --
# Ragas strictly tries to import ChatVertexAI which crashes if google-cloud is missing.
# We mock the module to allow the import to succeed without modifying site-packages.
import types
vertexai_mock = types.ModuleType('langchain_community.chat_models.vertexai')
vertexai_mock.ChatVertexAI = type('ChatVertexAI', (object,), {})
sys.modules['langchain_community.chat_models.vertexai'] = vertexai_mock
# --------------------------------------

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings

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
        
    if not settings.gemini_api_key:
        print("GEMINI_API_KEY not set. Cannot run evaluation.")
        exit(1)
        
    import urllib.request
    try:
        req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models?key={settings.gemini_api_key}")
        with urllib.request.urlopen(req) as response:
            print("AVAILABLE MODELS:", response.read().decode())
    except Exception as e:
        print("Failed to list models:", e)
        
    print("Generating answers for evaluation dataset...")
    
    # 1. Generate answers via real RAG pipeline
    dataset = asyncio.run(generate_rag_answers(dataset_path))
    
    print("Running Ragas evaluation...")

    # Ragas uses Langchain Chat models. We point it to Gemini via Langchain.
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=settings.gemini_api_key,
        temperature=0.0,
        max_retries=20
    )

    try:
        # Prevent concurrent rate limiting for free-tier constraints
        os.environ["RAGAS_MAX_CONCURRENCY"] = "1"
        os.environ["GOOGLE_MODEL_NAME"] = "gemini-3.6-flash"
        
        from ragas.run_config import RunConfig
        run_config = RunConfig(max_workers=1, max_retries=10, max_wait=60, timeout=120)
        
        from ragas.llms import LangchainLLMWrapper
        ragas_llm = LangchainLLMWrapper(llm)
        
        faithfulness.llm = ragas_llm
        
        result = evaluate(
            dataset,
            metrics=[faithfulness],
            llm=ragas_llm,
            run_config=run_config,
            raise_exceptions=False
        )
        print("Evaluation Results:")
        print(result)
        
        try:
            val = result["faithfulness"]
            if isinstance(val, list):
                faithfulness_score = float(sum(val)/len(val)) if val else 0.0
            else:
                faithfulness_score = float(val)
        except Exception as e:
            print(f"Parsing error: {e}")
            faithfulness_score = 0.0
                
        import math
        if faithfulness_score is None or math.isnan(faithfulness_score):
            print("Evaluation produced no valid score - failing closed")
            sys.exit(1)
            
        if faithfulness_score < 0.75:
            print(f"::error::Faithfulness score ({faithfulness_score}) is below the threshold of 0.75")
            sys.exit(1)
            
        print(f"Success! Faithfulness score: {faithfulness_score}")
        print("Generated Answers:")
        print(dataset["answer"])
        sys.exit(0)
    except Exception as e:
        print(f"Evaluation failed to run: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_eval()
