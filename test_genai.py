from google import genai
import os

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

try:
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents='What is 2+2? Keep it short.'
    )
    print("SUCCESS!")
    print(response.text)
except Exception as e:
    print("FAILED:", type(e).__name__)
    print(str(e))
