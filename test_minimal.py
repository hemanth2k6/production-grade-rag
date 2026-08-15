import os
from langchain_google_genai import ChatGoogleGenerativeAI

api_key = os.environ.get("GEMINI_API_KEY")

print("Initializing LangChain ChatGoogleGenerativeAI model...")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=api_key,
    temperature=0.0
)

print("Invoking the model with a test prompt...")
try:
    response = llm.invoke("What is 2+2? Keep it short.")
    print("SUCCESS!")
    print("Response Content:", response.content)
except Exception as e:
    print("FAILED:", type(e).__name__)
    print(str(e))
