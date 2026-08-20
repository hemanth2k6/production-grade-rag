import os
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=os.environ.get("GEMINI_API_KEY")
    )
    res = llm.invoke("Hello, who are you?")
    print("Response:", res.content)
except Exception as e:
    print("Error:", type(e).__name__, "-", e)
